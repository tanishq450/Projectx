import ollama
import json
import time
from datasets import Dataset
import asyncio

# RAGAS
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

# Your pipeline
from ProjecX.Llama_index.Rag_pipeline import Rag_pipeline
import asyncio
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings



MODEL = "glm-5:cloud"
embedding_model = "qwen3-embedding:4b"


# -------------------------------
# LLM JUDGE
# -------------------------------
def llm_judge(question, answer):
    prompt = f"""
    Evaluate the answer based on the question.

    Question: {question}
    Answer: {answer}

    Score from 0 to 10 for:
    - correctness
    - relevance
    - clarity

    Return STRICT JSON:
    {{
        "correctness": number,
        "relevance": number,
        "clarity": number
    }}
    """

    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    try:
        return json.loads(response["message"]["content"])
    except:
        return {"correctness": 0, "relevance": 0, "clarity": 0}


# -------------------------------
# RAGAS
# -------------------------------




llm = ChatOllama(model=MODEL)  



async def ragas_eval(question, answer, context, ground_truth=""):
    dataset = Dataset.from_dict({
        "question": [question],
        "answer": [answer],
        "contexts": [[context]],
        "ground_truth": [ground_truth if ground_truth else answer]
    })

    result = await asyncio.to_thread(
        lambda: evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision
            ],
            llm=llm, 
            embeddings=OllamaEmbeddings(model=embedding_model) 
        )
    )

    return result



# -------------------------------
# MAIN WRAPPER
# -------------------------------
async def ask_with_full_evaluation(query, pipeline, persist_dir):
    start = time.time()

    result = await pipeline.query(query, persist_dir)

    end = time.time()

    if isinstance(result, dict):
        answer = result.get("answer", "")
        docs = result.get("nodes", [])   

        context = " ".join([
            d.get("text", "") if isinstance(d, dict) else getattr(d, "text", "")
            for d in docs
        ])
    else:
        answer = str(result)
        context = ""

    # LLM judge (sync)
    judge_score = llm_judge(query, answer)

    # RAGAS (async)
    ragas_score = await ragas_eval(query, answer, context)

    return {
        "question": query,
        "answer": answer,
        "evaluation": {
            "llm_judge": judge_score,
            "ragas": ragas_score.to_dict() if hasattr(ragas_score, "to_dict") else str(ragas_score),
            "latency": end - start
        }
    }
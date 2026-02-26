import asyncio
import ast
from autogen_agentchat.messages import UserMessage
from ProjecX.Auto.model import get_model
from ProjecX.Llama_index.Rag_pipeline import Rag_pipeline
from ProjecX.Auto.web_search import web_search_agent
import loguru
logger = loguru.logger.bind(name="customteam")
from pathlib import Path

class CustomTeam:
    def __init__(self, rag_pipeline, web_agent,vector_dir:Path):
        self.rag_pipeline = rag_pipeline
        self.web_agent = web_agent
        self.model = get_model()
        self.vector_dir = vector_dir.resolve()

    # ---------- UTIL ----------
    def extract_web_docs(self, task_result) -> dict:
        if task_result is None:
            return {}

        for msg in reversed(task_result.messages):
            content = getattr(msg, "content", None)

            # Tavily result usually lands here
            if isinstance(content, dict):
                return content

            # Sometimes stringified dict
            if isinstance(content, str):
                try:
                    return ast.literal_eval(content)
                except Exception:
                    return {}

        return {}

    # ---------- SYNTHESIS ----------
    async def synthesize(self, query: str, rag_nodes: list, web_docs: dict) -> str:
        rag_context = ""
        if rag_nodes:
            rag_context = "\n".join(n.node.text for n in rag_nodes[:3])

        web_context = ""
        if isinstance(web_docs, dict):
            web_context = "\n".join(
                r.get("content", "")
                for r in web_docs.get("results", [])[:3]
            )

        prompt = f"""
        Answer the question using ONLY the information below.
        If insufficient, say so.

        Question:
        {query}

        RAG context:
        {rag_context}

        Web context:
        {web_context}
        """

        response = await self.model.create(
            messages=[
                UserMessage(
                    source="customteam",
                    content=prompt,
                )
            ]
        )

        return response.content

    # ---------- RUN ----------
    async def run(self, query: str,doc_id:str):
        persist_dir = self.vector_dir / doc_id
        rag_result = await asyncio.to_thread(self.rag_pipeline.query,   
           query,
           str(persist_dir),
           
        )

        score = float(rag_result["score"])

        # RAG ONLY
        if score >= 0.80:
            return {
                "source": "rag",
                "answer": rag_result["answer"],
                "score": score,
            }

        logger.info(f"RAG score: {score}")
        logger.info(f"RAG answer: {rag_result['answer']}")


        # RAG + WEB
        if 0.50 <= score < 0.80:
            web_task = await self.web_agent.run(task=query)
            web_docs = self.extract_web_docs(web_task)

            answer = await self.synthesize(
                query=query,
                rag_nodes=rag_result["nodes"],
                web_docs=web_docs,
            )

            return {
                "source": "rag+web",
                "answer": answer,
                "score": score,
            }

        
        # WEB ONLY
        web_task = await self.web_agent.run(task=query)
        web_docs = self.extract_web_docs(web_task)

        answer = await self.synthesize(
            query=query,
            rag_nodes=[],
            web_docs=web_docs,
        )

        return {
            "source": "web",
            "answer": answer,
            "score": score,
        }

        

"""
if __name__ == "__main__":
    import asyncio

    rag_pipeline = Rag_pipeline()
    team = CustomTeam(rag_pipeline, web_search_agent())

    try:
        loop = asyncio.get_running_loop()
        result = loop.create_task(team.run("Who is Harry Potter"))
    except RuntimeError:
        result = asyncio.run(team.run("What is attention mechanism"))

    print(result)
"""
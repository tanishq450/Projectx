from fastembed import SparseTextEmbedding

model = SparseTextEmbedding(model_name="Qdrant/bm25")

def compute_sparse_vectors(texts):
    return list(model.embed(texts))
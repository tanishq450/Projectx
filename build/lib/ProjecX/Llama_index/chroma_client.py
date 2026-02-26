import chromadb

_clients = {}

def get_chroma_client(persist_dir: str) -> chromadb.Client:
    if persist_dir not in _clients:
        _clients[persist_dir] = chromadb.Client(
            settings=chromadb.Settings(
                persist_directory=persist_dir
            )
        )
    return _clients[persist_dir]
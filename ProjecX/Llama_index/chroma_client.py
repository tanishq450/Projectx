import chromadb
import os

_clients = {}



def get_chroma_client(persist_dir: str) -> chromadb.Client:
    if persist_dir not in _clients:

        # create directory if it does not exist
        if not os.path.exists(persist_dir):
            os.makedirs(persist_dir)

        _clients[persist_dir] = chromadb.Client(
            chromadb.Settings(
                persist_directory=persist_dir,
                anonymized_telemetry=False
            )
        )

    return _clients[persist_dir]
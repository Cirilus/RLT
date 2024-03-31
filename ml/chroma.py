import chromadb
from loguru import logger

logger.debug("init the chroma db")
client = chromadb.PersistentClient(path="chroma")
collection = client.get_or_create_collection("database", metadata={"hnsw:space": "cosine"})
import torch
import torch.nn.functional as F
import re
import os
from chromadb.config import Settings
from chromadb.api.models.Collection import Collection
from loguru import logger
from tqdm import tqdm
import chromadb
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders.csv_loader import CSVLoader
from transformers import AutoModel, AutoTokenizer, Conversation, pipeline
from langchain.document_loaders import DataFrameLoader
from annoy import AnnoyIndex
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import pickle as pkl
import torch

from ml.chroma import collection
from ml.constants import MODEL_EMB_NAME, DEVICE, MODEL_CHAT_NAME
from ml.utils import fill_documents, collectionUpsert, load_models, load_chatbot, question_response

logger.debug("loading models")
emb_tokenizer, emb_model = load_models(MODEL_EMB_NAME, device=DEVICE)

logger.debug("loading csv files")
df223 = pd.read_csv("ml/223.csv")
df44 = pd.read_csv("ml/44.csv")
df = pd.concat([df223, df44])
loader = DataFrameLoader(df, page_content_column='article_text')
documents = loader.load()

text_data = [item.page_content for item in documents]

logger.debug("filling documents")
result_data = fill_documents(documents)

logger.debug("collectionUpsert")
collection_result = collectionUpsert(result_data, emb_tokenizer, emb_model, DEVICE)

logger.debug("loading chat bot")
chatbot = load_chatbot(MODEL_CHAT_NAME, device="cuda")

logger.debug("reading QAdata")
data = pd.read_csv('ml/QAdata')
question = list(data.QUESTION)
answer = list(data.ANSWER)

logger.debug("loading search model")
search_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
with open('ml/model.pkl', 'rb') as f:
    embeddings_search = pkl.load(f)

logger.debug("loading Annoy Index")
annoy_index = AnnoyIndex(len(embeddings_search[1]), 'angular')
annoy_index.load('ml/Annoy_index')

logger.debug("creating response")

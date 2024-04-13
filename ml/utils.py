import torch.nn.functional as F
from loguru import logger
from transformers import AutoModel, AutoTokenizer, Conversation, pipeline
import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import util
from transformers import AutoModel, AutoTokenizer, Conversation, pipeline
from llama_cpp import Llama
from ml.chroma import collection
from ml.constants import SYSTEM_PROMPT

SYSTEM_TOKEN = 1587
USER_TOKEN = 2188
BOT_TOKEN = 12435
LINEBREAK_TOKEN = 13

ROLE_TOKENS = {
    "user": USER_TOKEN,
    "bot": BOT_TOKEN,
    "system": SYSTEM_TOKEN
}
def get_message_tokens(model, role, content):
    content = f"{role}\n{content}\n</s>"
    content = content.encode("utf-8")
    message_tokens = model.tokenize(content, special=True)
    return message_tokens


def get_system_tokens(model):
    system_message = {
        "role": "system",
        "content": SYSTEM_PROMPT
    }
    return get_message_tokens(model, **system_message)

def load_models(model: str, device: str = "cpu", torch_dtype: str = "auto") -> tuple:
    tokenizer = AutoTokenizer.from_pretrained(model, device_map=device, torch_dtype=torch_dtype)
    model = AutoModel.from_pretrained(model, device_map=device, torch_dtype=torch_dtype)
    return tokenizer, model


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def txt2embeddings(text: str, tokenizer, model, device: str = "сpu") -> torch.Tensor:
    # Кодируем входной текст с помощью токенизатора
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')

    # Перемещаем закодированный ввод на указанное устройство
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

    # Получаем выход модели для закодированного ввода
    with torch.no_grad():
        model_output = model(**encoded_input)

    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings.tolist()


def fill_documents(documents):
    df = pd.DataFrame()
    df["rule_name"] = [doc.metadata['rule_name'] for doc in documents]
    df["article_name"] = [doc.metadata['article_name'] for doc in documents]
    df["article_text"] = [doc.page_content for doc in documents]
    return df


def collectionUpsert(df, tokenizer, model, device: str = "сuda"):
    article_texts = df["article_text"].tolist()
    article_names = df["article_name"].tolist()
    rule_names = df["rule_name"].tolist()
    embeddings = []
    for text in article_texts:
        res = txt2embeddings(text, tokenizer, model, device)
        embeddings.append(res[0])

    ids = [str(i) for i in range(1, len(df['article_name']) + 1)]
    metadatas = [
        {
            "source": "dataset",
            "article_name": article_name,
            "rule_name": rule_name,
        }
        for article_name, rule_name in zip(article_names, rule_names)
    ]
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=article_texts
    )

    return collection


def encodeQuestion(question, tokenizer, model, device, collection):
    embedding = txt2embeddings(question, tokenizer, model, device)
    result = collection.query(query_embeddings=embedding, n_results=2)
    result_documents = []

    for distance, metadata, document in zip(
            result["distances"][0], result["metadatas"][0], result["documents"][0]
    ):
        if distance < 1:
            result_documents.append(
                {"article_name": metadata["article_name"], "rule_name": metadata["rule_name"], "answer": document,
                 "metric": (1 - float(distance)) * 100})
    return result_documents


def load_chatbot(model: str, device: str = "cuda", torch_dtype: str = "auto"):
    # Загружаем чатбот с помощью pipeline из библиотеки transformers
    model = Llama(
        model_path=model,
        n_ctx=4096,
        n_gpu_layers=-1,
        n_parts=1,
    )
    return model


def append_documents_to_conversation(conversation, texts):
    text = "\n".join(texts)
    document_template = f"""
    CONTEXT:
    {text}
    Отвечай только на русском языке.
    ВОПРОС:
    """
    conversation.add_message({"role": "user", "content": document_template})

    return conversation


def generate_answer(
    model,
    user_prompt,
    n_ctx=4096,
    top_k=30,
    top_p=0.9,
    temperature=0.3,
    repeat_penalty=1.1
):

    system_tokens = get_system_tokens(model)
    tokens = system_tokens
    model.eval(tokens)

    message_tokens = get_message_tokens(model=model, role="user", content=user_prompt)
    token_str = ""
    role_tokens = [model.token_bos(), BOT_TOKEN, LINEBREAK_TOKEN]
    tokens += message_tokens + role_tokens
    generator = model.generate(
        tokens,
        top_k=top_k,
        top_p=top_p,
        temp=temperature,
        repeat_penalty=repeat_penalty
    )
    for token in generator:
        token_str += model.detokenize([token]).decode("utf-8", errors="ignore")
        tokens.append(token)
        if token == model.token_eos():
            break
    return token_str


def get_answer(question: str,  tokenizer, model, device, collection, chatbot):
  answers = encodeQuestion(question, tokenizer, model, device, collection)
  answer =  [i['answer'] for i in answers]
  article_name = answers[0]['article_name']
  rule_name = answers[0]['rule_name']
  metric = answers[0]['metric']
  USER_PROMPT = f"""Я даю тебе вход плохо структурированный текст, используй его для ответа на мой вопрос, 
  текст может быть бесполезный, поэтому только используй его как подсказку. 
  Текст: {answers}
  Вопрос: {question}"""
  output = generate_answer(chatbot, USER_PROMPT)
  return output, article_name, rule_name, metric

def get_answer_smart(question: str, result_answer, tokenizer, model, device, collection, chatbot):
  answers = encodeQuestion(question, tokenizer, model, device, collection)
  answer =  [i['answer'] for i in answers]
  article_name = answers[0]['article_name']
  rule_name = answers[0]['rule_name']
  USER_PROMPT = f"""Я даю тебе вход плохо структурированный текст, используй его для ответа на мой вопрос, 
  текст может быть бесполезный, поэтому только используй его как подсказку. 
  Текст: {result_answer}
  Вопрос: {question}"""
  output = generate_answer(chatbot, USER_PROMPT)
  return output, article_name, rule_name


def question_response(embeddings, question, emb_tokenizer,
                      emb_model, device, collection, chatbot,
                      search_model, annoy_index, answer):
    top_k_hits = 5
    question_embedding = search_model.encode(question)
    corpus_ids, scores = annoy_index.get_nns_by_vector(question_embedding, top_k_hits, include_distances=True)
    hits = []
    for id, score in zip(corpus_ids, scores):
        hits.append({'corpus_id': id, 'score': 1-((score**2) / 2)})
    hits_above_threshold = [hit for hit in hits if hit['score'] > 0.8]

    result_answer = None

    if hits_above_threshold:
        question_embedding_tensor = torch.tensor(question_embedding)
        embeddings = torch.tensor(embeddings)
        correct_hits = util.semantic_search(question_embedding, embeddings, top_k=top_k_hits)[0]
        correct_hits_ids = list([hit['corpus_id'] for hit in correct_hits])
        with open('user_questions.txt', 'a') as f:
            f.write(f"Вопрос пользователя: {question}\n")
        result_answer = answer[correct_hits_ids[0]]
        metric = hits_above_threshold[0]['score']*100
        result_answer, article_name, rule_name = get_answer_smart(question, result_answer, emb_tokenizer, emb_model, device, collection, chatbot)
        print('asdf')
        return  result_answer, article_name, rule_name, f"{metric:.2f}"
    if result_answer is None:
        result_answer, article_name, rule_name, metric = get_answer(question, emb_tokenizer, emb_model, device, collection, chatbot)
        return  result_answer, article_name, rule_name, f"{metric:.2f}"

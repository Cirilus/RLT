import torch.nn.functional as F
from loguru import logger
from transformers import AutoModel, AutoTokenizer, Conversation, pipeline
import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import util
from transformers import AutoModel, AutoTokenizer, Conversation, pipeline

from ml.chroma import collection
from ml.constants import SYSTEM_PROMPT


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
    chatbot = pipeline(
        model=model,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map=device,
        task="conversational",
    )
    return chatbot


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
        chatbot,
        conversation: Conversation,
        max_new_tokens: int = 128,
        temperature=0.9,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 2.0,
        do_sample: bool = True,
        num_beams: int = 2,
        early_stopping: bool = True,
) -> str:
    # Генерируем ответ от чатбота
    conversation = chatbot(
        conversation,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=do_sample,
        num_beams=num_beams,
        early_stopping=early_stopping,
    )

    return conversation


def get_answer(question: str, tokenizer, model, device, collection, chatbot):
    conversation = Conversation()
    conversation.add_message({"role": "system", "content": SYSTEM_PROMPT})
    answers = encodeQuestion(question, tokenizer, model, device, collection)
    answer = [i['answer'] for i in answers]
    conversation = append_documents_to_conversation(conversation, answer)
    conversation.add_message({"role": "user", "content": question})
    output = generate_answer(chatbot, conversation, temperature=0.9)
    return output[-1]["content"]


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
        result_answer = answer[correct_hits_ids[0]]

    if result_answer is None:
        result_answer = get_answer(question, emb_tokenizer, emb_model, device, collection, chatbot)

    return result_answer
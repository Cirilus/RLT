from fastapi import APIRouter, UploadFile

from ml.chroma import collection
from ml.constants import DEVICE
from ml.main import embeddings_search, emb_tokenizer, emb_model, chatbot, search_model, annoy_index, answer
from ml.utils import question_response
from schemas.ml import GetAnswerResponse

router = APIRouter(prefix="/api/v1/ml", tags=["ml"])


@router.get(
    "/get_answer",
    description="getting the answer",
    response_model=GetAnswerResponse,
)
async def get_answer(question: str):
    result_answer, article_name, rule_name, metric = question_response(embeddings_search, question,
                                 emb_tokenizer, emb_model, DEVICE, collection, chatbot, search_model, annoy_index,
                                 answer)

    return {
        "answer" : result_answer,
        "article_name" : article_name,
        "rule_name" : rule_name,
        "metric" : metric
    }

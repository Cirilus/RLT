from pydantic import BaseModel


class GetAnswerResponse(BaseModel):
    result_answer: str
    article_name: str
    rule_name: str
    metric: str

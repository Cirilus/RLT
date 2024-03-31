from pydantic import BaseModel


class GetAnswerResponse(BaseModel):
    answer: str

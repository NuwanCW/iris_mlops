from typing import List

from fastapi import Query
from pydantic import BaseModel, validator


class Text(BaseModel):
    text: str = Query(None, min_length=1)


class PredictPayload(BaseModel):
    texts: List[Text]

    @validator("texts")
    def list_must_not_be_empty(cls, value):
        if not len(value):
            raise ValueError("List of texts to classify cannot be empty.")
        return value

    class Config:
        schema_extra = {
            "example": {
                "texts": [
                    {"text": "5.1,3.5,1.4,.2"},
                    {"text": "2.1,1.5,1.4,.4"},
                ]
            }
        }

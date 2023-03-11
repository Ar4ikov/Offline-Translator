from enum import Enum
from typing import NamedTuple, Type
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer
)

__all__ = [
    "BaseModel",
    "NLLBModels"
]


class BaseModel(NamedTuple):
    """Base model for NLLB models

    Args:
        model_type (str): model type
        model_url (str): model url
        model_cls (Type[AutoModelForSeq2SeqLM]): model class
        tokenizer_cls (Type[AutoTokenizer]): tokenizer class
    """
    model_type: str
    model_url: str
    model_cls: Type[AutoModelForSeq2SeqLM]
    tokenizer_cls: Type[AutoTokenizer]


class NLLBModels(NamedTuple, Enum):
    """NLLB models"""
    NLLB_200_600M = BaseModel(
        model_type="facebook/nllb-200-600M",
        model_url="facebook/nllb-200-600M",
        model_cls=AutoModelForSeq2SeqLM,
        tokenizer_cls=AutoTokenizer
    )

    NLLB_200_1_3B = BaseModel(
        model_type="facebook/nllb-200-1.3B",
        model_url="facebook/nllb-200-1.3B",
        model_cls=AutoModelForSeq2SeqLM,
        tokenizer_cls=AutoTokenizer
    )

    NLLB_200_3_3B = BaseModel(
        model_type="facebook/nllb-200-3.3B",
        model_url="facebook/nllb-200-3.3B",
        model_cls=AutoModelForSeq2SeqLM,
        tokenizer_cls=AutoTokenizer
    )

    def __str__(self):
        return self.value.model_type

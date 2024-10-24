from transformers import PretrainedConfig
from typing import List


class CetaceanClassifierConfig(PretrainedConfig):
    model_type = "cetaceanet"

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

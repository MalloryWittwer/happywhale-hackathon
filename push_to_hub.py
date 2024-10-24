"""
Script used to load a custom model and push it to HuggingFace.
Doc: https://huggingface.co/docs/transformers/custom_models#writing-a-custom-model
"""
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from cetacean_classifier.configuration_cetacean_classifier import (
    CetaceanClassifierConfig,
)
from cetacean_classifier.modeling_cetacean_classifier import (
    CetaceanClassifierModelForImageClassification,
)

CetaceanClassifierConfig.register_for_auto_class()
CetaceanClassifierModelForImageClassification.register_for_auto_class("AutoModelForImageClassification")

cetacean_config = CetaceanClassifierConfig()
cetacean_classifier = CetaceanClassifierModelForImageClassification(cetacean_config)

cetacean_classifier.push_to_hub("Saving-Willy/cetacean-classifier")

from transformers import PreTrainedModel
from PIL import Image
import numpy as np
import torch

from .configuration_cetacean_classifier import CetaceanClassifierConfig
from .train import SphereClassifier


WHALE_CLASSES = np.array(
    [
        "beluga",
        "blue_whale",
        "bottlenose_dolphin",
        "brydes_whale",
        "commersons_dolphin",
        "common_dolphin",
        "cuviers_beaked_whale",
        "dusky_dolphin",
        "false_killer_whale",
        "fin_whale",
        "frasiers_dolphin",
        "gray_whale",
        "humpback_whale",
        "killer_whale",
        "long_finned_pilot_whale",
        "melon_headed_whale",
        "minke_whale",
        "pantropic_spotted_dolphin",
        "pygmy_killer_whale",
        "rough_toothed_dolphin",
        "sei_whale",
        "short_finned_pilot_whale",
        "southern_right_whale",
        "spinner_dolphin",
        "spotted_dolphin",
        "white_sided_dolphin",
    ]
)


class CetaceanClassifierModelForImageClassification(PreTrainedModel):
    config_class = CetaceanClassifierConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = SphereClassifier.load_from_checkpoint("cetacean_classifier/last.ckpt")
        self.model.eval()

    def preprocess_image(self, img: Image) -> torch.Tensor:
        image_resized = img.resize((480, 480))
        image_resized = np.array(image_resized)[None]
        image_resized = np.transpose(image_resized, [0, 3, 2, 1])
        image_tensor = torch.Tensor(image_resized)
        return image_tensor

    def forward(self, img: Image, labels=None):
        tensor = self.preprocess_image(img)
        head_id_logits, head_species_logits = self.model(tensor)
        head_species_logits = head_species_logits.detach().numpy()
        sorted_idx = head_species_logits.argsort()[0]
        sorted_idx = np.array(list(reversed(sorted_idx)))
        top_three_logits = sorted_idx[:3]
        top_three_whale_preds = WHALE_CLASSES[top_three_logits]

        return {"predictions": top_three_whale_preds}

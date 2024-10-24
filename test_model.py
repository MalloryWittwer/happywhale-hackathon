import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from PIL import Image
from transformers import AutoModelForImageClassification

if __name__ == "__main__":
    cetacean_classifier = AutoModelForImageClassification.from_pretrained("Saving-Willy/cetacean-classifier", trust_remote_code=True)
    img = Image.open('tail.jpg')

    out = cetacean_classifier(img)
    print(out)
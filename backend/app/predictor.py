
from pathlib import Path

import torch
from PIL import Image
from transformers import (
    ViTForImageClassification,
    ViTImageProcessor,
)

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


class FoodPredictor:
    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = Path(model_dir)
        self.model = None
        self.processor = None

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.load_model()

    def load_model(self):
        config_path = self.model_dir / "config.json"

        if not config_path.exists():
            return

        self.processor = ViTImageProcessor.from_pretrained(
            str(self.model_dir)
        )

        self.model = ViTForImageClassification.from_pretrained(
            str(self.model_dir)
        )

        self.model.to(self.device)
        self.model.eval()

    @property
    def ready(self) -> bool:
        return (
            self.model is not None
            and self.processor is not None
        )

    @torch.inference_mode()
    def predict(self, image: Image.Image, top_k: int = 3):
        if not self.ready:
            raise RuntimeError(
                "O modelo ainda não foi treinado ou instalado."
            )

        image = image.convert("RGB")

        inputs = self.processor(
            images=image,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        logits = self.model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1)[0]

        k = min(top_k, probabilities.numel())
        scores, indices = torch.topk(probabilities, k=k)

        results = []

        for score, index in zip(
            scores.tolist(),
            indices.tolist(),
        ):
            label = self.model.config.id2label.get(
                index, str(index)
            )

            results.append({
                "label": label,
                "confidence": round(float(score), 6),
            })

        return results

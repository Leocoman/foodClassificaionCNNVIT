

from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0


# Caminhos do projeto
BACKEND_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BACKEND_DIR / "models" / "efficientnet_b0"
MODEL_PATH = MODEL_DIR / "efficientnet_b0_food10.pth"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


class FoodPredictor:
    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Modelo não encontrado: {MODEL_PATH}"
            )

        # Carregar o checkpoint exportado pelo notebook
        checkpoint = torch.load(
            MODEL_PATH,
            map_location=DEVICE,
            weights_only=True,
        )

        self.class_names = checkpoint["class_names"]
        num_classes = checkpoint["num_classes"]

        # Reconstruir a arquitetura utilizada no treino
        self.model = efficientnet_b0(weights=None)

        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(
            in_features, num_classes
        )

        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(DEVICE)
        self.model.eval()

        image_size = checkpoint.get("image_size", 224)
        mean = checkpoint.get(
            "mean", [0.485, 0.456, 0.406]
        )
        std = checkpoint.get(
            "std", [0.229, 0.224, 0.225]
        )

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    @torch.inference_mode()
    def predict(self, image: Image.Image, top_k: int = 3):
        image = image.convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(DEVICE)

        logits = self.model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

        k = min(top_k, len(self.class_names))
        scores, indices = torch.topk(probabilities, k=k)

        predictions = [
            {
                "class_name": self.class_names[index.item()],
                "confidence": round(score.item(), 4),
            }
            for score, index in zip(scores, indices)
        ]

        return {
            "predicted_class": predictions[0]["class_name"],
            "confidence": predictions[0]["confidence"],
            "top_predictions": predictions,
        }


# Carregar o modelo uma única vez quando o módulo é inicializado
predictor = FoodPredictor()

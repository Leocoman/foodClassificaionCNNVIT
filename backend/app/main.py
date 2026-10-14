
import time
from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from app.predictor import FoodPredictor
from app.schemas import HealthResponse, PredictionResponse

app = FastAPI(
    title="FoodVision API",
    description="Classificação de alimentos com Vision Transformer",
    version="1.0.0",
)

predictor = FoodPredictor()

MAX_FILE_SIZE = 10 * 1024 * 1024

ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}

ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8081",
        "http://127.0.0.1:8081",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "message": "FoodVision API está online",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
def health():
    return {
        "status": "ok",
        "model_ready": predictor.ready,
        "device": str(predictor.device),
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Formato de imagem não suportado.",
        )

    contents = await file.read()

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="A imagem excede o limite de 10 MB.",
        )

    try:
        image = Image.open(BytesIO(contents))
        image.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="O ficheiro enviado não é uma imagem válida.",
        )

    if image.format not in ALLOWED_FORMATS:
        raise HTTPException(
            status_code=415,
            detail="Formato de imagem não suportado.",
        )

    if not predictor.ready:
        raise HTTPException(
            status_code=503,
            detail="O modelo ainda não está disponível.",
        )

    start_time = time.perf_counter()

    predictions = predictor.predict(image, top_k=3)

    inference_time_ms = (
        time.perf_counter() - start_time
    ) * 1000

    return {
        "predictions": predictions,
        "inference_time_ms": round(inference_time_ms, 2),
    }

from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from app.predictor import predictor


app = FastAPI(
    title="FoodVision Mobile API",
    description="Classificação de alimentos com EfficientNet-B0",
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "message": "FoodVision Mobile API",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "efficientnet_b0",
        "classes": len(predictor.class_names),
    }


@app.post("/predict")
async def predict_food(file: UploadFile = File(...)):
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=415,
            detail="O ficheiro enviado deve ser uma imagem.",
        )

    contents = await file.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="O ficheiro está vazio.",
        )

    try:
        image = Image.open(BytesIO(contents))
        image.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="Não foi possível ler a imagem.",
        )

    try:
        return predictor.predict(image)
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Erro durante a previsão.",
        )

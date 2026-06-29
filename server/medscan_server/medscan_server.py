"""
MedScann Server v2.3 — Production-Grade with GDPR
FIXES:
  v2.2: Spectrogram shape enforced to (64, 87, 1)
  v2.3: Feature extraction now L2-normalized to match training notebook
"""

import json
import logging
import logging.handlers
import io
import numpy as np
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager
from datetime import datetime
import hashlib
import secrets

import librosa
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# ============================================================
# LOGGING
# ============================================================
log_file = Path("medscan_server.log")
file_handler = logging.handlers.RotatingFileHandler(
    log_file, maxBytes=10*1024*1024, backupCount=5
)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))
logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])
logger = logging.getLogger(__name__)

# ============================================================
# SECURITY CONFIG
# ============================================================
API_KEY       = "medscan-secure-key-2024"
MAX_AUDIO_SIZE = 10 * 1024 * 1024

limiter = Limiter(key_func=get_remote_address)

# ============================================================
# CONFIG & PATHS
# ============================================================
import os
from pathlib import Path
# Get the directory where this file is located
current_dir = Path(__file__).parent.parent  # Goes up to server/
KERAS_MODEL_PATH  = current_dir / "hybrid_crnn_v4_f32" / "hybrid_crnn_v4.keras"
TFLITE_MODEL_PATH = current_dir / "hybrid_crnn_v4_f32" / "hybrid_crnn_v4_f32.tflite"
META_PATH         = current_dir / "model_meta" / "model_meta.json"
SCALER_JSON_PATH  = current_dir / "feat_scaler" / "feat_scaler.json"

if KERAS_MODEL_PATH.exists():
    USE_KERAS  = True
    MODEL_PATH = KERAS_MODEL_PATH
    logger.info(f"Using Keras model: {MODEL_PATH}")
elif TFLITE_MODEL_PATH.exists():
    USE_KERAS  = False
    MODEL_PATH = TFLITE_MODEL_PATH
    logger.info(f"Using TFLite model: {MODEL_PATH}")
else:
    raise FileNotFoundError("No model found (hybrid_crnn_v4.keras or hybrid_crnn_v4_f32.tflite)")

for path in [META_PATH, SCALER_PATH]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

logger.info("All required files found")

with open(META_PATH) as f:
    MODEL_META = json.load(f)
    logger.info(f"Model: {MODEL_META['model_name']}")

with open(SCALER_PATH) as f:
    SCALER_DATA  = json.load(f)
    SCALER_MEAN  = np.array(SCALER_DATA['mean_'],  dtype=np.float32)
    SCALER_SCALE = np.array(SCALER_DATA['scale_'], dtype=np.float32)

if not TF_AVAILABLE:
    raise ImportError("TensorFlow required")

MODEL        = None
interpreter  = None
input_details  = None
output_details = None

# ── Custom layers ────────────────────────────────────────────────────
class TemporalAttention(tf.keras.layers.Layer):
    def __init__(self, units=32, **kwargs):
        super().__init__(**kwargs)
        self.units      = units
        self.score_dense = tf.keras.layers.Dense(units, activation="tanh")
        self.score_out   = tf.keras.layers.Dense(1, use_bias=False)
    def call(self, x, training=False):
        scores  = self.score_out(self.score_dense(x))
        weights = tf.nn.softmax(scores, axis=1)
        return tf.reduce_sum(x * weights, axis=1)
    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.units})
        return cfg

class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, label_smoothing=0.05, **kwargs):
        super().__init__(**kwargs)
        self.gamma           = gamma
        self.label_smoothing = label_smoothing
    def call(self, y_true, y_pred):
        num_classes = tf.cast(tf.shape(y_true)[-1], tf.float32)
        y_true_s    = y_true * (1.0 - self.label_smoothing) + self.label_smoothing / num_classes
        y_pred      = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        p_t         = tf.reduce_sum(y_true_s * y_pred, axis=-1, keepdims=True)
        focal_w     = tf.pow(1.0 - p_t, self.gamma)
        ce          = -y_true_s * tf.math.log(y_pred)
        return tf.reduce_mean(focal_w * ce)
    def get_config(self):
        cfg = super().get_config()
        cfg.update({"gamma": self.gamma, "label_smoothing": self.label_smoothing})
        return cfg

CUSTOM_OBJECTS = {"TemporalAttention": TemporalAttention, "FocalLoss": FocalLoss}

if USE_KERAS:
    try:
        MODEL = tf.keras.models.load_model(str(KERAS_MODEL_PATH), custom_objects=CUSTOM_OBJECTS)
        logger.info(f"Keras model loaded ({MODEL_PATH.stat().st_size / 1e6:.1f} MB)")
    except Exception as e:
        logger.error(f"Keras model loading error: {e}")
        raise
else:
    try:
        interpreter    = tf.lite.Interpreter(model_path=str(TFLITE_MODEL_PATH))
        interpreter.allocate_tensors()
        input_details  = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        logger.info(f"TFLite model loaded ({MODEL_PATH.stat().st_size / 1e6:.1f} MB)")
    except Exception as e:
        logger.error(f"TFLite load error, falling back to Keras: {e}")
        MODEL    = tf.keras.models.load_model(str(KERAS_MODEL_PATH), custom_objects=CUSTOM_OBJECTS)
        USE_KERAS = True
        logger.info(f"Keras fallback loaded ({KERAS_MODEL_PATH.stat().st_size / 1e6:.1f} MB)")

# ============================================================
# RESPONSE MODELS
# ============================================================
class PredictionResult(BaseModel):
    condition:    str
    severity:     str
    confidence:   float
    explanation:  str
    actions:      list
    seeDoctor:    bool
    probabilities: dict = {}
    timestamp:    str  = ""

class HealthCheckResponse(BaseModel):
    status:    str
    model:     str
    version:   str
    timestamp: str = ""

class GDPRResponse(BaseModel):
    message:        str
    privacy_notice: str
    user_rights:    list

# ============================================================
# SECURITY FUNCTIONS
# ============================================================
def verify_api_key(authorization: str = Header(None)) -> bool:
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing API key")
    try:
        scheme, credentials = authorization.split()
        if scheme.lower() != "bearer":
            raise HTTPException(status_code=401, detail="Invalid auth scheme")
        if credentials != API_KEY:
            raise HTTPException(status_code=401, detail="Invalid API key")
        return True
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid authorization header")

def validate_filename(filename: str, ip: str) -> bool:
    if not filename or len(filename) > 255:
        return False
    for char in ['..', '/', '\\', '\x00', '\r', '\n']:
        if char in filename:
            logger.warning(f"Path traversal attempt from {ip}")
            return False
    return True

def generate_request_id() -> str:
    return secrets.token_hex(8)

# ============================================================
# FEATURE EXTRACTION  —  must match training notebook exactly
# ============================================================
def extract_spectrogram(audio_data: np.ndarray, sr: int) -> Optional[np.ndarray]:
    """
    Mel spectrogram → enforced shape (64, 87, 1).
    Matches extract_dual() in training notebook.
    """
    try:
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)
        audio_data = audio_data.astype(np.float32)

        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val

        if sr != MODEL_META['sample_rate']:
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=MODEL_META['sample_rate'])
            sr = MODEL_META['sample_rate']

        n_samples = int(sr * MODEL_META['clip_dur'])
        if len(audio_data) < n_samples:
            reps       = int(np.ceil(n_samples / len(audio_data)))
            audio_data = np.tile(audio_data, reps)[:n_samples]
        else:
            audio_data = audio_data[:n_samples]

        audio_data = np.nan_to_num(audio_data)

        mel    = librosa.feature.melspectrogram(
            y=audio_data, sr=sr,
            n_mels=MODEL_META['n_mels'], n_fft=MODEL_META['n_fft'],
            hop_length=MODEL_META['hop_length'],
            fmin=MODEL_META['fmin'], fmax=MODEL_META['fmax']
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)
        lo, hi  = log_mel.min(), log_mel.max()
        if hi - lo < 1e-6:
            logger.error("Spectrogram has no dynamic range — silent audio?")
            return None
        spec = (log_mel - lo) / (hi - lo)

        # ── Enforce exact shape (64, 87) ──────────────────────────────
        target_mels   = MODEL_META['spec_shape'][0]   # 64
        target_frames = MODEL_META['spec_shape'][1]   # 87

        if spec.shape[0] < target_mels:
            spec = np.pad(spec, ((0, target_mels - spec.shape[0]), (0, 0)))
        else:
            spec = spec[:target_mels, :]

        if spec.shape[1] < target_frames:
            spec = np.pad(spec, ((0, 0), (0, target_frames - spec.shape[1])))
        else:
            spec = spec[:, :target_frames]

        spec = np.expand_dims(spec, axis=-1).astype(np.float32)   # (64,87,1)
        logger.info(f"Spectrogram shape: {spec.shape}")
        return spec

    except Exception as e:
        logger.error(f"Spectrogram extraction failed: {e}")
        return None


def extract_acoustic_features(audio_data: np.ndarray, sr: int) -> Optional[np.ndarray]:
    """
    151-dim acoustic feature vector.

    MUST match training notebook extract_dual() exactly:
      mfcc_mean (40) + mfcc_std (40) + logmel_mean (64) +
      [rms, zcr, sc_mean, sc_std, sr_mean, sr_std, rms_std]  (7)
      = 151 total

    Then L2-normalised (norm → unit vector) — CRITICAL to match training.
    Then StandardScaler applied using feat_scaler.json.
    """
    try:
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)
        audio_data = audio_data.astype(np.float32)

        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val

        if sr != MODEL_META['sample_rate']:
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=MODEL_META['sample_rate'])
            sr = MODEL_META['sample_rate']

        n_samples = int(sr * MODEL_META['clip_dur'])
        if len(audio_data) < n_samples:
            reps       = int(np.ceil(n_samples / len(audio_data)))
            audio_data = np.tile(audio_data, reps)[:n_samples]
        else:
            audio_data = audio_data[:n_samples]

        audio_data = np.nan_to_num(audio_data)

        # ── Feature extraction (same order as training) ───────────────
        mel        = librosa.feature.melspectrogram(
            y=audio_data, sr=sr,
            n_mels=MODEL_META['n_mels'], n_fft=MODEL_META['n_fft'],
            hop_length=MODEL_META['hop_length'],
            fmin=MODEL_META['fmin'], fmax=MODEL_META['fmax']
        )
        mfcc       = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=MODEL_META['n_mfcc'])
        mfcc_mean  = np.mean(mfcc, axis=1)          # (40,)
        mfcc_std   = np.std(mfcc,  axis=1)           # (40,)
        logmel_mean = np.mean(librosa.power_to_db(mel), axis=1)  # (64,)

        rms_frames = librosa.feature.rms(y=audio_data).squeeze()
        rms        = float(np.mean(rms_frames))
        rms_std    = float(np.std(rms_frames))
        zcr        = float(np.mean(librosa.feature.zero_crossing_rate(audio_data)))

        sc         = librosa.feature.spectral_centroid(y=audio_data, sr=sr).squeeze()
        sc_mean, sc_std = float(np.mean(sc)), float(np.std(sc))

        sr_f       = librosa.feature.spectral_rolloff(y=audio_data, sr=sr).squeeze()
        sr_mean, sr_std = float(np.mean(sr_f)), float(np.std(sr_f))

        features = np.concatenate([
            mfcc_mean, mfcc_std, logmel_mean,
            [rms, zcr, sc_mean, sc_std, sr_mean, sr_std, rms_std]
        ])   # 40+40+64+7 = 151

        features = np.nan_to_num(features)

        # ── L2 normalise — MUST match training ────────────────────────
        norm = np.linalg.norm(features)
        if norm < 1e-8:
            logger.error("Feature vector near-zero — unusable audio")
            return None
        features = (features / norm).astype(np.float32)

        # Pad / trim safety net
        if len(features) > 151:
            features = features[:151]
        elif len(features) < 151:
            features = np.pad(features, (0, 151 - len(features)))

        logger.info(f"Features: shape={features.shape}, min={features.min():.4f}, max={features.max():.4f}, mean={features.mean():.4f}")
        return features

    except Exception as e:
        logger.error(f"Feature extraction error: {e}")
        return None


def build_symptom_vector(answers: dict) -> np.ndarray:
    """14-dim binary symptom vector."""
    try:
        fields = MODEL_META['symptom_fields']
        vec    = np.zeros(len(fields), dtype=np.float32)
        for i, field in enumerate(fields):
            vec[i] = float(np.clip(answers.get(field, 0), 0, 1))
        return vec
    except Exception as e:
        logger.error(f"Symptom vector error: {e}")
        raise ValueError(f"Symptom error: {str(e)}")

# ============================================================
# INFERENCE
# ============================================================
def run_inference(spec_input, feat_input, symp_input) -> dict:
    try:
        if USE_KERAS:
            logger.info(f"DEBUG — spec: {spec_input.shape} min={spec_input.min():.3f} max={spec_input.max():.3f}")
            logger.info(f"DEBUG — feat: {feat_input.shape} min={feat_input.min():.4f} max={feat_input.max():.4f} mean={feat_input.mean():.4f}")
            logger.info(f"DEBUG — symp: {symp_input.shape} values={symp_input[0]}")

            output = MODEL.predict({
                "spec_input": spec_input,
                "feat_input": feat_input,
                "symp_input": symp_input
            }, verbose=0)
            logits = output[0]
        else:
            interpreter.set_tensor(input_details[0]['index'], spec_input)
            interpreter.set_tensor(input_details[1]['index'], feat_input)
            interpreter.set_tensor(input_details[2]['index'], symp_input)
            interpreter.invoke()
            logits = interpreter.get_tensor(output_details[0]['index'])[0]

        exp_logits    = np.exp(logits - np.max(logits))
        probabilities = exp_logits / np.sum(exp_logits)

        class_idx = np.argmax(probabilities)
        condition = MODEL_META['class_names'][class_idx]
        confidence = float(probabilities[class_idx])

        probs_dict = {name: float(probabilities[i])
                      for i, name in enumerate(MODEL_META['class_names'])}

        return {'condition': condition, 'confidence': confidence, 'probabilities': probs_dict}

    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise ValueError(f"Inference error: {str(e)}")


def generate_explanation(condition: str, confidence: float) -> dict:
    explanations = {
        'Healthy': {
            'severity':    'Mild',
            'explanation': 'Your respiratory symptoms suggest you are likely healthy.',
            'actions':     ['Continue normal activities', 'Monitor for new symptoms'],
            'seeDoctor':   False
        },
        'COVID-19': {
            'severity':    'Severe' if confidence > 0.75 else 'Moderate',
            'explanation': 'Your symptoms may be consistent with COVID-19. Consult a healthcare provider.',
            'actions':     ['Seek professional evaluation', 'Get tested', 'Isolate if possible'],
            'seeDoctor':   True
        },
        'URTI / Cold': {
            'severity':    'Mild',
            'explanation': 'Your symptoms appear consistent with upper respiratory infection.',
            'actions':     ['Rest', 'Stay hydrated', 'Monitor symptoms'],
            'seeDoctor':   False
        }
    }
    return explanations.get(condition, {
        'severity':    'Moderate',
        'explanation': 'Unable to determine condition.',
        'actions':     ['Seek professional evaluation'],
        'seeDoctor':   True
    })

# ============================================================
# APP SETUP
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Server starting")
    yield
    logger.info("Server shutting down")

app = FastAPI(
    title="MedScann API",
    description="Respiratory disease classification",
    version="2.3.0",
    lifespan=lifespan
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded,
    lambda r, e: HTTPException(status_code=429, detail="Rate limit exceeded"))

app.add_middleware(TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1", "*.replit.dev"])
app.add_middleware(CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

# ============================================================
# ROUTES
# ============================================================
@app.get("/")
async def root():
    return {"service": "MedScann API", "model": MODEL_META['model_name'],
            "version": "2.3.0", "status": "running"}

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    return HealthCheckResponse(status="healthy", model=MODEL_META['model_name'],
                               version="2.3.0", timestamp=datetime.now().isoformat())

@app.get("/privacy", response_model=GDPRResponse)
async def privacy_policy():
    return GDPRResponse(
        message="GDPR Compliance Information",
        privacy_notice="Raw data deleted immediately. Predictions kept 7 days. Anonymized metrics kept forever.",
        user_rights=["Right to access your data", "Right to deletion",
                     "Right to withdraw consent", "Right to data portability"]
    )

@app.post("/predict", response_model=PredictionResult)
@limiter.limit("20/minute")
async def predict(
    request: Request,
    audio: UploadFile = File(...),
    answers: str = Form(...),
    authorization: str = Header(None)
):
    request_id = generate_request_id()
    client_ip  = request.client.host if request.client else "unknown"
    ip_hash    = hashlib.sha256(client_ip.encode()).hexdigest()[:8]
    logger.info(f"[{request_id}] Request from IP:{ip_hash}")

    try:
        verify_api_key(authorization)

        if not validate_filename(audio.filename, client_ip):
            raise HTTPException(status_code=400, detail="Invalid filename")

        audio_bytes = await audio.read()

        if len(audio_bytes) > MAX_AUDIO_SIZE:
            raise HTTPException(status_code=413, detail="File too large")
        if len(audio_bytes) < 1000:
            raise HTTPException(status_code=400, detail="Audio file too small")

        try:
            audio_data, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)
        except Exception as e:
            logger.error(f"[{request_id}] Audio decode error: {e}")
            raise HTTPException(status_code=400, detail="Could not decode audio. Try recording again.")

        try:
            answers_dict = json.loads(answers)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON answers")

        spec = extract_spectrogram(audio_data, sr)
        if spec is None:
            raise HTTPException(status_code=400, detail="Failed to extract spectrogram")

        feat = extract_acoustic_features(audio_data, sr)
        if feat is None:
            raise HTTPException(status_code=400, detail="Failed to extract features")

        symp = build_symptom_vector(answers_dict)

        # Apply StandardScaler (on already L2-normalised features)
        feat = (feat - SCALER_MEAN) / (SCALER_SCALE + 1e-8)

        # Add batch dimension
        spec = np.expand_dims(spec, axis=0)   # (1, 64, 87, 1)
        feat = np.expand_dims(feat, axis=0)   # (1, 151)
        symp = np.expand_dims(symp, axis=0)   # (1, 14)

        result    = run_inference(spec, feat, symp)
        condition = result['condition']
        confidence = result['confidence']
        severity_info = generate_explanation(condition, confidence)

        logger.info(f"[{request_id}] PREDICTION: {condition} ({confidence:.1%}) | Probs: {result['probabilities']}")

        return PredictionResult(
            condition=condition,
            severity=severity_info['severity'],
            confidence=confidence,
            explanation=severity_info['explanation'],
            actions=severity_info['actions'],
            seeDoctor=severity_info['seeDoctor'],
            probabilities=result['probabilities'],
            timestamp=datetime.now().isoformat(),
        )

    except HTTPException as e:
        logger.error(f"[{request_id}] HTTP error: {e.detail}")
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Server error")


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)

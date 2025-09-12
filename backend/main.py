import io
import os
import uuid
import logging
from typing import Optional

import numpy as np
import cv2
import httpx
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Body, Request
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer
from rembg import remove, new_session

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/bgremover/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Optional: introspect ONNX Runtime providers ---
try:
    import onnxruntime as ort
    ORT_PROVIDERS = ort.get_available_providers()
except Exception:
    ORT_PROVIDERS = []

# --- Model cache dir (make sure the service user can write here) ---
U2NET_HOME = os.environ.get("U2NET_HOME", os.path.expanduser("~/.u2net"))
os.makedirs(U2NET_HOME, exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title="BGRemover Server",
    version="1.0.0",
    description="Production-ready background removal API",
    docs_url="/docs",  # Only available in development
    redoc_url="/redoc"  # Only available in development
)

# Security middleware
security = HTTPBearer(auto_error=False)

# CORS configuration - only allow specific origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "https://bgremover.tiiny.site"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["phraai.com", "www.phraai.com", "localhost", "127.0.0.1"]
)

# Security headers middleware
@app.middleware("http")
async def security_headers(request: Request, call_next):
    # Block curl and other command-line tools
    user_agent = request.headers.get("user-agent", "").lower()
    blocked_agents = ["curl", "wget", "python-requests", "python-urllib", "httpx"]
    
    if any(agent in user_agent for agent in blocked_agents):
        logger.warning(f"Blocked request from user-agent: {user_agent}")
        raise HTTPException(status_code=403, detail="Access denied")
    
    response = await call_next(request)
    
    # Add security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: https:; "
        "connect-src 'self' https://bgremover.tiiny.site http://localhost:5500; "
        "frame-ancestors 'none';"
    )
    
    return response

# Rate limiting (simple in-memory implementation)
from collections import defaultdict
import time

request_counts = defaultdict(list)
RATE_LIMIT = 10  # requests per minute
RATE_WINDOW = 60  # seconds

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host
    now = time.time()
    
    # Clean old requests
    request_counts[client_ip] = [
        req_time for req_time in request_counts[client_ip] 
        if now - req_time < RATE_WINDOW
    ]
    
    # Check rate limit
    if len(request_counts[client_ip]) >= RATE_LIMIT:
        logger.warning(f"Rate limit exceeded for IP: {client_ip}")
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    request_counts[client_ip].append(now)
    return await call_next(request)

# --- Simple in-process session cache ---
_SESSION_CACHE: dict[str, object] = {}

def get_session(model_name: str):
    key = model_name.lower()
    if key not in _SESSION_CACHE:
        logger.info(f"Loading model: {key}")
        _SESSION_CACHE[key] = new_session(key)
    return _SESSION_CACHE[key]

# -------------------- CV helpers --------------------

def _refine_alpha_with_cv2(
    img_bgr: np.ndarray,
    alpha_u8: np.ndarray,
    use_guided: bool = True,
    feather_sigma: float = 0.8,
    shrink_px: int = 1,
) -> np.ndarray:
    """
    Cleanup + edge-aware smoothing + optional shrink to kill glow.
    Returns uint8 alpha (0..255).
    """
    # Basic cleanup: open+close to remove speckles and pinholes
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    a = cv2.morphologyEx(alpha_u8, cv2.MORPH_OPEN, k3, iterations=1)
    a = cv2.morphologyEx(a, cv2.MORPH_CLOSE, k3, iterations=1)

    af = a.astype(np.float32) / 255.0

    # Apply shrink BEFORE smoothing to avoid artifacts
    if shrink_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * shrink_px + 1, 2 * shrink_px + 1))
        a_shrunk = cv2.erode((af * 255).astype(np.uint8), k, iterations=1)
        af = a_shrunk.astype(np.float32) / 255.0

    # Edge-aware matte smoothing
    if use_guided and hasattr(cv2, "ximgproc"):
        try:
            guide = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            af = cv2.ximgproc.guidedFilter(guide=guide, src=af, radius=8, eps=1e-3)
        except Exception:
            # Fallback if guided filter fails
            af = cv2.bilateralFilter(af, d=5, sigmaColor=0.05, sigmaSpace=5)
    else:
        if hasattr(cv2, "ximgproc"):
            try:
                guide = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                af = cv2.ximgproc.jointBilateralFilter(
                    guide, (af * 255).astype(np.uint8), d=5, sigmaColor=15, sigmaSpace=15
                ).astype(np.float32) / 255.0
            except Exception:
                af = cv2.bilateralFilter(af, d=5, sigmaColor=0.05, sigmaSpace=5)
        else:
            af = cv2.bilateralFilter(af, d=5, sigmaColor=0.05, sigmaSpace=5)

    # Gentle feather to avoid crunchy edges
    if feather_sigma > 0:
        af = cv2.GaussianBlur(af, (0, 0), feather_sigma)

    return (np.clip(af, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def _protect_interior(
    alpha_refined: np.ndarray,
    base_mask: np.ndarray,
    interior_band_px: int = 6,
    core_thresh: int = 200,
) -> np.ndarray:
    """
    Keep the object's interior solid and prevent accidental transparency.
    """
    # Binary FG from base mask
    fg = (base_mask >= 128).astype(np.uint8)
    fg255 = (fg * 255).astype(np.uint8)

    # Distance to the edge on the inside of the object
    dist_in = cv2.distanceTransform(fg255, cv2.DIST_L2, 3)
    interior = (dist_in > interior_band_px) & (fg.astype(bool))

    # Start with maximum of refined alpha and base mask
    out = np.maximum(alpha_refined, base_mask).astype(np.uint8)
    
    # Force very confident core pixels to 255
    out[base_mask >= core_thresh] = 255
    
    # Simpler approach without complex indexing
    if np.any(interior):
        # Calculate interior weight for the entire image
        interior_weight = np.clip((dist_in - interior_band_px) / max(interior_band_px, 1), 0, 1)
        
        # Create target alpha values for all pixels
        target_alpha = (200 + interior_weight * 55).astype(np.uint8)
        
        # Apply only where interior mask is True
        out = np.where(interior, np.maximum(out, target_alpha), out)
    
    return out


def _decontaminate_colors(
    img_bgr: np.ndarray,
    alpha_u8: np.ndarray,
    norm_sigma: float = 5.0,
    weight_power: float = 2.0,
    rim_px: int = 4,
    t_power: float = 0.7,
) -> np.ndarray:
    """
    Normalized convolution using confident FG colors; only applied in a thin outer rim.
    """
    img = img_bgr.astype(np.float32)
    a = (alpha_u8.astype(np.float32) / 255.0).clip(0, 1)

    # Emphasize confident FG so estimate comes from inside the object
    w = np.power(a, weight_power)
    num = cv2.GaussianBlur(img * w[..., None], (0, 0), norm_sigma)
    den = cv2.GaussianBlur(w, (0, 0), norm_sigma)
    if den.ndim == 2:
        den = den[..., None]
    eps = 1e-6
    fg_est = num / (den + eps)

    # Better rim detection logic with safety checks
    fg_mask = (a >= 0.8).astype(np.uint8)
    
    # Safety check: ensure fg_mask has some content
    if np.sum(fg_mask) == 0:
        # If no confident foreground found, skip decontamination
        clean = img.astype(np.uint8)
        return np.dstack([clean, alpha_u8])
    
    try:
        # Create distance map from foreground edges
        fg_edges = cv2.Canny((fg_mask * 255).astype(np.uint8), 50, 150)
        
        # Safety check: ensure edges were detected
        if np.sum(fg_edges) == 0:
            # No edges detected, create a simple rim based on alpha gradient
            rim = (a > 0.1) & (a < 0.9)
        else:
            dist_from_edge = cv2.distanceTransform(255 - fg_edges, cv2.DIST_L2, 3)
            rim = (dist_from_edge <= rim_px) & (a > 0.1)
        
        rim = rim.astype(np.float32)
        
        # More conservative color replacement
        t = np.power(1.0 - a, t_power) * rim * 0.7
        t3 = t[..., None]
        clean = (1.0 - t3) * img + t3 * fg_est
        
    except Exception:
        # If any edge detection fails, just return original colors
        clean = img
    
    clean = np.clip(clean, 0, 255).astype(np.uint8)
    return np.dstack([clean, alpha_u8])


def _encode_image(bgra: np.ndarray, fmt: str = "png") -> tuple[bytes, str]:
    """
    Encodes BGRA to PNG/WebP. Returns (bytes, media_type).
    """
    # Validate input
    if bgra is None or bgra.size == 0:
        raise RuntimeError("Invalid input array for encoding")
    
    if bgra.ndim != 3 or bgra.shape[2] not in [3, 4]:
        raise RuntimeError(f"Invalid array shape for encoding: {bgra.shape}")
    
    if fmt == "png":
        ok, buf = cv2.imencode(".png", bgra)
        media = "image/png"
    elif fmt == "webp":
        ok, buf = cv2.imencode(".webp", bgra)
        media = "image/webp"
    else:
        raise RuntimeError("fmt must be 'png' or 'webp'")
    
    if not ok or buf is None:
        raise RuntimeError("encode failed")
    
    return buf.tobytes(), media

# -------------------- REST endpoints --------------------

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": time.time()}

@app.get("/version")
async def version():
    return {
        "app": app.version,
        "onnxruntime_providers": ORT_PROVIDERS,
        "model_cache": list(_SESSION_CACHE.keys()),
        "u2net_home": U2NET_HOME,
    }

def _parse_bool(v: Optional[bool], default: bool) -> bool:
    return default if v is None else bool(v)

@app.post("/remove", summary="Remove background from an uploaded image")
async def remove_upload(
    request: Request,
    file: UploadFile = File(..., description="Image file"),
    model: str = Query("u2net", description="u2net, u2netp, u2net_human_seg, isnet-general-use"),
    only_mask: Optional[bool] = Query(False, description="Return alpha/mask only"),
    post_process_mask: Optional[bool] = Query(True, description="Smooth/clean mask"),
    alpha_matting: Optional[bool] = Query(False, description="Enable alpha matting refinement"),
    alpha_matting_foreground_threshold: int = Query(240, ge=0, le=255),
    alpha_matting_background_threshold: int = Query(10, ge=0, le=255),
    alpha_matting_erode_size: int = Query(10, ge=0, le=255),
    fmt: str = Query("png", description="png or webp (ignored for only_mask)"),
):
    logger.info(f"Processing request from {request.client.host} - model: {model}")
    
    if fmt not in ("png", "webp"):
        raise HTTPException(400, "fmt must be 'png' or 'webp'")
    
    # File size validation (10MB limit)
    if file.size and file.size > 10 * 1024 * 1024:
        raise HTTPException(400, "File too large. Maximum size is 10MB")
    
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "empty file")

    try:
        session = get_session(model)
        out_bytes = remove(
            raw,
            session=session,
            only_mask=_parse_bool(only_mask, False),
            post_process_mask=_parse_bool(post_process_mask, True),
            alpha_matting=_parse_bool(alpha_matting, False),
            alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
            alpha_matting_background_threshold=alpha_matting_background_threshold,
            alpha_matting_erode_size=alpha_matting_erode_size,
        )
        logger.info(f"Successfully processed image with model: {model}")
    except Exception as e:
        logger.error(f"Background removal failed: {e}")
        raise HTTPException(500, f"background removal failed: {e}")

    # rembg returns PNG bytes by default. Convert if needed.
    if fmt == "webp" and not only_mask:
        from PIL import Image
        img = Image.open(io.BytesIO(out_bytes)).convert("RGBA")
        buf = io.BytesIO()
        img.save(buf, format="WEBP")
        out_bytes = buf.getvalue()
        media_type = "image/webp"
    else:
        media_type = "image/png"

    safe_name = os.path.splitext(file.filename or f"img-{uuid.uuid4().hex}")[0]
    filename = f"{safe_name}.{'webp' if fmt=='webp' and not only_mask else 'png'}"
    headers = {"Content-Disposition": f'inline; filename="{filename}"'}
    return Response(content=out_bytes, media_type=media_type, headers=headers)

@app.post("/remove-pro", summary="High-quality background removal with OpenCV refinements")
async def remove_pro(
    request: Request,
    file: UploadFile = File(...),
    model: str = Query("u2net"),
    fmt: str = Query("png", description="png or webp"),
    guided: bool = Query(True, description="Use guided filter if available"),
    feather_sigma: float = Query(0.6, ge=0.0, le=5.0, description="Gaussian feather sigma"),
    shrink_px: int = Query(0, ge=0, le=3, description="Erode alpha by N px to kill glow"),
    norm_sigma: float = Query(4.0, ge=0.5, le=10.0, description="Blur sigma for color estimation"),
    weight_power: float = Query(1.8, ge=1.0, le=4.0, description="Emphasize confident FG"),
    rim_px: int = Query(3, ge=1, le=10, description="Width of outer rim to recolor"),
    t_power: float = Query(0.6, ge=0.2, le=2.0, description="Edge replacement strength curve"),
):
    logger.info(f"Processing pro request from {request.client.host} - model: {model}")
    
    if fmt not in ("png", "webp"):
        raise HTTPException(400, "fmt must be 'png' or 'webp'")

    # File size validation
    if file.size and file.size > 10 * 1024 * 1024:
        raise HTTPException(400, "File too large. Maximum size is 10MB")

    raw = await file.read()
    if not raw:
        raise HTTPException(400, "empty file")

    try:
        # Decode original with OpenCV
        np_in = np.frombuffer(raw, np.uint8)
        img_bgr = cv2.imdecode(np_in, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise HTTPException(400, "unsupported or corrupt image")

        # 1) Base matte from rembg
        session = get_session(model)
        mask_png = remove(
            raw,
            session=session,
            only_mask=True,
            post_process_mask=False,
            alpha_matting=False,
        )

        # Decode mask
        m = cv2.imdecode(np.frombuffer(mask_png, np.uint8), cv2.IMREAD_GRAYSCALE)
        if m is None or m.shape[:2] != img_bgr.shape[:2]:
            raise HTTPException(500, "mask decode failed or size mismatch")

        # 2) Refine alpha and protect interior
        alpha_refined = _refine_alpha_with_cv2(
            img_bgr, m, use_guided=guided, feather_sigma=feather_sigma, shrink_px=shrink_px
        )
        alpha_refined = _protect_interior(
            alpha_refined, base_mask=m, interior_band_px=6, core_thresh=200
        )

        # 3) Decontaminate edge colors and assemble BGRA
        out_bgra = _decontaminate_colors(
            img_bgr, alpha_refined,
            norm_sigma=norm_sigma, weight_power=weight_power, rim_px=rim_px, t_power=t_power
        )

        # 4) Encode with validation
        out_bytes, media = _encode_image(out_bgra, fmt=fmt)
        
        logger.info(f"Successfully processed pro request with model: {model}")

        base = os.path.splitext(file.filename or "image")[0]
        filename = f"{base}.{'webp' if fmt=='webp' else 'png'}"
        headers = {"Content-Disposition": f'inline; filename="{filename}"'}
        return Response(content=out_bytes, media_type=media, headers=headers)
        
    except Exception as e:
        logger.error(f"Pro processing failed: {str(e)}")
        raise HTTPException(500, f"processing failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8015)
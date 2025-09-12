import os
import io
import gc
import time
import uuid
import hashlib
import logging
from typing import Optional, Tuple

import numpy as np
import cv2
import httpx  # kept for parity with your UA block list

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Request
from fastapi.responses import Response, JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer

try:
    import onnxruntime as ort
    ORT_PROVIDERS = ort.get_available_providers()
except Exception:
    ORT_PROVIDERS = []

# --- Model cache dir (same as your original) ---
U2NET_HOME = os.environ.get("U2NET_HOME", os.path.expanduser("~/.u2net"))
os.makedirs(U2NET_HOME, exist_ok=True)

# rembg
try:
    from rembg import remove, new_session
except Exception as e:
    remove = None
    new_session = None

# -----------------------------------------------------------------------------
# App & Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bgremover")

app = FastAPI(
    title="BGRemover Server",
    version="1.0.0",
    description="Production background removal API with OpenCV refinements",
    docs_url="/docs",
    redoc_url="/redoc",
)

security = HTTPBearer(auto_error=False)

# CORS (matches your frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "https://bgremover.tiiny.site",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Trusted hosts (same idea as before)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["phraai.com", "www.phraai.com", "localhost", "127.0.0.1"],
)

# -----------------------------------------------------------------------------
# Middleware: Security headers + UA blocking (with API allowlist) + Timing
# -----------------------------------------------------------------------------
API_ALLOW = {"/remove", "/remove-pro", "/health", "/version"}

@app.middleware("http")
async def security_headers(request: Request, call_next):
    start = time.time()

    # Block obvious CLI scrapers globally, except for API endpoints
    user_agent = (request.headers.get("user-agent") or "").lower()
    blocked_agents = ("curl", "wget", "python-requests", "python-urllib", "httpx")
    if any(a in user_agent for a in blocked_agents) and request.url.path not in API_ALLOW:
        logger.warning(f"Blocked UA for path {request.url.path}: {user_agent}")
        return PlainTextResponse("Access denied", status_code=403)

    response = await call_next(request)

    # Security headers (including your original CSP)
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

    # Timing header
    response.headers["X-Process-Time"] = f"{time.time() - start:.4f}"
    return response

# -----------------------------------------------------------------------------
# Simple rate limiting (same behavior)
# -----------------------------------------------------------------------------
from collections import defaultdict
_request_bins = defaultdict(list)
RATE_LIMIT = 10      # requests per minute
RATE_WINDOW = 60.0   # seconds

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    ip = request.client.host if request.client else "unknown"
    now = time.time()
    bucket = _request_bins[ip]
    # purge old
    while bucket and now - bucket[0] > RATE_WINDOW:
        bucket.pop(0)
    if len(bucket) >= RATE_LIMIT:
        return JSONResponse({"detail": "Rate limit exceeded"}, status_code=429)
    bucket.append(now)
    return await call_next(request)

# -----------------------------------------------------------------------------
# Session cache (same intent as your original get_session)
# -----------------------------------------------------------------------------
_SESSION_CACHE: dict[str, object] = {}

def get_session(model_name: str):
    if new_session is None:
        raise HTTPException(500, "rembg not installed on server")
    key = model_name.lower()
    if key not in _SESSION_CACHE:
        logger.info(f"Loading model: {key}")
        _SESSION_CACHE[key] = new_session(key)
    return _SESSION_CACHE[key]

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
def safe_filename(name: Optional[str]) -> str:
    base = (name or f"img-{uuid.uuid4().hex}").strip().replace("\\", "/").split("/")[-1]
    base = "".join(ch for ch in base if ch.isalnum() or ch in ("-", "_", ".", " "))
    return base or f"img-{uuid.uuid4().hex}"

def sanitize_format(fmt: str) -> Tuple[str, str]:
    fmt = (fmt or "png").lower()
    if fmt not in ("png", "webp"):
        raise HTTPException(400, "fmt must be 'png' or 'webp'")
    media = "image/png" if fmt == "png" else "image/webp"
    return fmt, media

def _hash_for_cache(raw: bytes, **kwargs) -> str:
    h = hashlib.sha256()
    h.update(raw)
    for k in sorted(kwargs.keys()):
        h.update(str(k).encode())
        h.update(str(kwargs[k]).encode())
    return h.hexdigest()[:16]

def _parse_bool(v: Optional[bool], default: bool) -> bool:
    return default if v is None else bool(v)

# --- CV helpers (kept under your original names, implementations streamlined) ---
def _refine_alpha_with_cv2(
    img_bgr: np.ndarray,
    alpha_u8: np.ndarray,
    use_guided: bool = True,
    feather_sigma: float = 0.8,
    shrink_px: int = 1,
) -> np.ndarray:
    # cleanup speckles
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    a = cv2.morphologyEx(alpha_u8, cv2.MORPH_OPEN, k3, iterations=1)
    a = cv2.morphologyEx(a, cv2.MORPH_CLOSE, k3, iterations=1)

    # optional shrink to kill glow
    if shrink_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * shrink_px + 1, 2 * shrink_px + 1))
        a = cv2.erode(a, k, iterations=1)

    # edge-aware smoothing
    if use_guided and hasattr(cv2, "ximgproc"):
        try:
            guide = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            a = cv2.ximgproc.guidedFilter(guide=guide, src=a, radius=8, eps=1e-2)
        except Exception:
            a = cv2.bilateralFilter(a, d=5, sigmaColor=25, sigmaSpace=25)
    else:
        a = cv2.bilateralFilter(a, d=5, sigmaColor=25, sigmaSpace=25)

    if feather_sigma > 0:
        a = cv2.GaussianBlur(a, (0, 0), feather_sigma)

    return np.clip(a, 0, 255).astype(np.uint8)

def _protect_interior(alpha_u8: np.ndarray, base_mask: np.ndarray, interior_band_px: int = 6, core_thresh: int = 200) -> np.ndarray:
    fg = (base_mask >= core_thresh).astype(np.uint8)
    if not np.any(fg):
        return alpha_u8
    dist_in = cv2.distanceTransform((fg * 255), cv2.DIST_L2, 3)
    interior = dist_in > interior_band_px
    out = alpha_u8.copy()
    out[interior] = 255
    return out

def _decontaminate_colors(
    img_bgr: np.ndarray,
    alpha_u8: np.ndarray,
    norm_sigma: float = 4.0,
    weight_power: float = 1.8,
    rim_px: int = 3,
    t_power: float = 0.6,
) -> np.ndarray:
    # replace edge pixels using inside colors to kill colored halos
    eroded = cv2.erode(alpha_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rim_px*2+1, rim_px*2+1)))
    rim = cv2.subtract(alpha_u8, eroded) > 0

    a = (alpha_u8.astype(np.float32) / 255.0)
    w = np.power(a, weight_power)
    w = cv2.GaussianBlur(w, (0, 0), norm_sigma)

    imgf = img_bgr.astype(np.float32)
    w3 = cv2.merge([w, w, w]) + 1e-6
    mu = cv2.GaussianBlur(imgf * w3, (0, 0), norm_sigma) / w3

    out = imgf.copy()
    t = np.power(a, t_power)[:, :, None]
    out[rim] = (mu[rim] * t[rim] + imgf[rim] * (1.0 - t[rim]))
    out = np.clip(out, 0, 255).astype(np.uint8)

    return cv2.merge([out[:, :, 0], out[:, :, 1], out[:, :, 2], alpha_u8])

# -----------------------------------------------------------------------------
# Health / Version
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Helpers for rembg integration
# -----------------------------------------------------------------------------
async def _run_rembg_only_mask(raw: bytes, session):
    import asyncio
    if remove is None:
        raise HTTPException(500, "rembg not installed")
    return await asyncio.to_thread(
        remove,
        raw,
        session,
        True,   # only_mask
        False,  # post_process_mask
        False,  # alpha_matting
    )

def _decode_mask_png(mask_png: bytes, target_size: Tuple[int, int]) -> np.ndarray:
    mask = cv2.imdecode(np.frombuffer(mask_png, np.uint8), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise HTTPException(500, "mask decode failed")
    if mask.ndim == 3 and mask.shape[2] > 1:
        alpha = mask[:, :, 3] if mask.shape[2] == 4 else cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        alpha = mask.astype(np.uint8)
    if (mask.shape[1], mask.shape[0]) != target_size:
        alpha = cv2.resize(alpha, target_size, interpolation=cv2.INTER_LINEAR)
    return alpha

# -----------------------------------------------------------------------------
# /remove : fast path (keeps your alpha-matting tunables)
# -----------------------------------------------------------------------------
@app.post("/remove", summary="Fast background removal (raw rembg output)")
async def remove_basic(
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
    logger.info(f"/remove from {request.client.host} - model={model}")
    if fmt not in ("png", "webp"):
        raise HTTPException(400, "fmt must be 'png' or 'webp'")

    # File size guard (restored)
    if getattr(file, "size", None) and file.size > 10 * 1024 * 1024:
        raise HTTPException(400, "File too large. Maximum size is 10MB")

    raw = await file.read()
    if not raw:
        raise HTTPException(400, "empty file")

    # Heavy call off the event loop
    import asyncio
    session = get_session(model)
    out_bytes = await asyncio.to_thread(
        remove,
        raw,
        session,
        _parse_bool(only_mask, False),
        _parse_bool(post_process_mask, True),
        _parse_bool(alpha_matting, False),
        alpha_matting_foreground_threshold,
        alpha_matting_background_threshold,
        alpha_matting_erode_size,
    )

    # Build headers incl. ETag/Cache-Control
    fmt2, media = sanitize_format(fmt if not only_mask else "png")
    base = os.path.splitext(safe_filename(file.filename))[0]
    filename = f"{base}.{fmt2}"
    etag = _hash_for_cache(raw, model=model, only_mask=only_mask, fmt=fmt2, endpoint="remove")
    headers = {
        "Content-Disposition": f'inline; filename="{filename}"',
        "ETag": f'W/"{etag}"',
        "Cache-Control": "public, max-age=3600",
    }

    return Response(content=out_bytes, media_type=("image/png" if only_mask else media), headers=headers)

# -----------------------------------------------------------------------------
# /remove-pro : high-quality path with OpenCV refinements + pre-resize
# -----------------------------------------------------------------------------
@app.post("/remove-pro", summary="High-quality background removal with OpenCV refinements")
async def remove_pro(
    request: Request,
    file: UploadFile = File(..., description="Image file"),
    model: str = Query("u2net"),
    fmt: str = Query("png", description="png or webp"),
    guided: bool = Query(True, description="Use guided/bilateral refinement"),
    feather_sigma: float = Query(0.6, ge=0.0, le=5.0, description="Gaussian feather sigma"),
    shrink_px: int = Query(0, ge=0, le=3, description="Erode alpha by N px"),
    norm_sigma: float = Query(4.0, ge=0.5, le=10.0, description="Blur sigma for color estimation"),
    weight_power: float = Query(1.8, ge=1.0, le=4.0, description="Emphasize confident FG"),
    rim_px: int = Query(3, ge=1, le=10, description="Width of outer rim to recolor"),
    t_power: float = Query(0.6, ge=0.2, le=2.0, description="Edge replacement strength curve"),
    max_dim: int = Query(2048, ge=512, le=4096, description="Cap long side before model (pre-resize)"),
):
    logger.info(f"/remove-pro from {request.client.host} - model={model}")
    if fmt not in ("png", "webp"):
        raise HTTPException(400, "fmt must be 'png' or 'webp'")

    # File size guard (restored)
    if getattr(file, "size", None) and file.size > 10 * 1024 * 1024:
        raise HTTPException(400, "File too large. Maximum size is 10MB")

    raw = await file.read()
    if not raw:
        raise HTTPException(400, "empty file")

    try:
        # Decode original
        np_in = np.frombuffer(raw, np.uint8)
        img_bgr_full = cv2.imdecode(np_in, cv2.IMREAD_COLOR)
        if img_bgr_full is None:
            raise HTTPException(400, "unsupported or corrupt image")

        H0, W0 = img_bgr_full.shape[:2]

        # Pre-resize for the model (faster), but keep full-res output
        img_bgr_for_model = img_bgr_full
        if max(H0, W0) > max_dim:
            scale = float(max_dim) / float(max(H0, W0))
            new_w, new_h = int(round(W0 * scale)), int(round(H0 * scale))
            img_bgr_for_model = cv2.resize(img_bgr_full, (new_w, new_h), interpolation=cv2.INTER_AREA)

        ok, enc = cv2.imencode(".png", img_bgr_for_model)
        if not ok:
            raise HTTPException(500, "encode failed while preparing model input")
        raw_for_rembg = enc.tobytes()

        # Model session + heavy call off the loop
        session = get_session(model)
        mask_png = await _run_rembg_only_mask(raw_for_rembg, session)

        # Decode mask and scale back to original size
        m = _decode_mask_png(mask_png, target_size=(W0, H0))

        # Refinements
        alpha_refined = _refine_alpha_with_cv2(
            img_bgr_full, m, use_guided=guided, feather_sigma=feather_sigma, shrink_px=shrink_px
        )
        alpha_refined = _protect_interior(alpha_refined, base_mask=m, interior_band_px=6, core_thresh=200)

        # Edge decontamination + assemble BGRA
        out_bgra = _decontaminate_colors(
            img_bgr_full, alpha_refined,
            norm_sigma=norm_sigma, weight_power=weight_power,
            rim_px=rim_px, t_power=t_power
        )

        # Encode to requested format
        fmt2, media = sanitize_format(fmt)
        ext = ".png" if fmt2 == "png" else ".webp"
        ok, buf = cv2.imencode(ext, out_bgra)
        if not ok:
            raise HTTPException(500, "encode failed")
        out_bytes = bytes(buf)

        # Headers with ETag/Cache-Control
        etag = _hash_for_cache(
            raw,
            model=model, fmt=fmt2, guided=guided, feather_sigma=feather_sigma,
            shrink_px=shrink_px, norm_sigma=norm_sigma, weight_power=weight_power,
            rim_px=rim_px, t_power=t_power, max_dim=max_dim, endpoint="remove-pro",
        )
        base = os.path.splitext(safe_filename(file.filename))[0]
        filename = f"{base}.{fmt2}"
        headers = {
            "Content-Disposition": f'inline; filename="{filename}"',
            "ETag": f'W/"{etag}"',
            "Cache-Control": "public, max-age=3600",
        }

        # Memory tidy
        del np_in, img_bgr_full, img_bgr_for_model, m, alpha_refined, out_bgra, buf
        gc.collect()

        return Response(content=out_bytes, media_type=media, headers=headers)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Pro processing failed")
        raise HTTPException(500, f"processing failed: {e}")

# -----------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8015)
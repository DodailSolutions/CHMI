import streamlit as st
import numpy as np
import requests
import base64
import time
import re
import secrets
import html
import tensorflow as tf
from PIL import Image
from datetime import datetime
from streamlit_geolocation import streamlit_geolocation  # For location permission & coords
import os
import pandas as pd
from fpdf import FPDF
import json



image_size = (224, 224)
strict_threshold = 0.005
smooth_threshold = 0.0075
threshold = smooth_threshold  # Changed to smooth_threshold for better field-image recall

# ── Base directory: all file paths resolved relative to this file ─────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def get_autoencoder():
    """Load and cache the cow autoencoder TFLite model (runs once per session)."""
    model_path = os.path.join(BASE_DIR, "cow_autoencoder_flex.tflite")
    interp = tf.lite.Interpreter(
        model_path=model_path,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
    )
    interp.allocate_tensors()
    return interp, interp.get_input_details(), interp.get_output_details()


interpreter, input_details, output_details = get_autoencoder()



def run_tflite_inference(interpreter, input_details, output_details, input_data):
    """
    Runs inference using the provided TFLite interpreter and returns output array.
    """
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

def reconstruction_error_tflite(img_array, reconstructed_array):
    """
    Calculates mean squared reconstruction error between original and output arrays.
    """
    return np.mean((img_array - reconstructed_array) ** 2)


def analyze_single_image(image, interpreter, input_details, output_details, threshold=threshold):
    """
    Multi-heuristic image validator. Returns error and acceptance status.

    The autoencoder alone is not reliable — simple/solid images can have
    reconstruction error LOWER than real cow photos. We layer several checks:
      1. Pixel variance  — real photos have rich texture (var > 0.005)
      2. Color diversity — real photos have spread across R,G,B channels
      3. Brightness      — not pure black or pure white
      4. Autoencoder MSE — real cow images: 0.0005-0.006; simple images
                           cluster near 0 or above 0.015.
                           We reject if error < MIN_ERROR (too simple)
                           OR error > threshold (too different from a cow).
    """
    try:
        image.seek(0)
    except AttributeError:
        pass

    pil_img = Image.open(image).convert('RGB').resize((224, 224))
    img_array = np.array(pil_img).astype('float32') / 255.0   # shape (224,224,3)

    # ── Heuristic 1: pixel variance ──────────────────────────────────────────
    pixel_var = float(np.var(img_array))
    if pixel_var < 0.004:          # solid/nearly-solid image
        return {'error': 0.0, 'accepted': False, 'reason': 'low_variance'}

    # ── Heuristic 2: per-channel std — reject near-grayscale images ──────────
    ch_stds = img_array.reshape(-1, 3).std(axis=0)   # std per channel
    if ch_stds.max() < 0.06:       # all channels uniformly flat
        return {'error': 0.0, 'accepted': False, 'reason': 'no_color'}

    # ── Heuristic 3: brightness sanity ───────────────────────────────────────
    mean_brightness = float(img_array.mean())
    if mean_brightness < 0.04 or mean_brightness > 0.96:
        return {'error': 0.0, 'accepted': False, 'reason': 'extreme_brightness'}

    # ── Heuristic 4: autoencoder reconstruction error ────────────────────────
    input_data = np.expand_dims(img_array, axis=0)
    reconstructed = run_tflite_inference(interpreter, input_details, output_details, input_data)
    error = reconstruction_error_tflite(input_data, reconstructed)

    # Real cattle images: 0.0005–0.006. Too-simple images: < 0.0005.
    # Very different images: > threshold (0.0075).
    MIN_ERROR = 0.0003
    is_accepted = (MIN_ERROR <= error <= threshold)

    return {
        'error': error,
        'accepted': is_accepted,
        'reason': 'ok' if is_accepted else ('too_simple' if error < MIN_ERROR else 'not_cow')
    }

# Create folders if not exist
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploaded_images")
TEXT_FOLDER = os.path.join(BASE_DIR, "image_metadata")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TEXT_FOLDER, exist_ok=True)

# Image paths for each disease (3 images per disease)
fmd_images = [
    os.path.join(BASE_DIR, "FMD Images", "1.jpeg"),
    os.path.join(BASE_DIR, "FMD Images", "2.jpeg"),
    os.path.join(BASE_DIR, "FMD Images", "3.jpeg"),
]

lsd_images = [
    os.path.join(BASE_DIR, "LSD Images", "1.jpg"),
    os.path.join(BASE_DIR, "LSD Images", "2.jpg"),
    os.path.join(BASE_DIR, "LSD Images", "3.jpg"),
]

# -----------------------------------
# --- Globals / Constants ---

MODEL_WEIGHTS = {
    "v2b0": 0.2,
    "v2s": 0.5,
    "vgg16": 1.5,
    "resnet50": 0.5,
    "b0": 0.3
}
TOTAL_WEIGHT = sum(MODEL_WEIGHTS.values())  # precomputed constant

CLASS_NAMES = ["Diseased", "Healthy"]

CATEGORIES = ["FMD-Knuckles", "FMD-Mouth","Healthy-Foot","Healthy-Muzzle"] 

TFLITE_MODELS_FMD = {
    "v2b0": os.path.join(BASE_DIR, "FMD New Models", "EfficientNetV2B0_model.tflite"),
    "v2s":  os.path.join(BASE_DIR, "FMD New Models", "EfficientNetV2S_model.tflite"),
    "vgg16": os.path.join(BASE_DIR, "FMD New Models", "VGG16_model.tflite"),
    "resnet50": os.path.join(BASE_DIR, "FMD New Models", "ResNet50_model.tflite"),
    "b0":   os.path.join(BASE_DIR, "FMD New Models", "EfficientNetB0_model.tflite"),
}

TFLITE_MODELS_LSD = {
    "v2b0": os.path.join(BASE_DIR, "LSD Models", "EfficientNetV2B0_model.tflite"),
    "v2s":  os.path.join(BASE_DIR, "LSD Models", "EfficientNetV2S_model.tflite"),
    "vgg16": os.path.join(BASE_DIR, "LSD Models", "VGG16_model.tflite"),
    "resnet50": os.path.join(BASE_DIR, "LSD Models", "ResNet50_model.tflite"),
    "b0":   os.path.join(BASE_DIR, "LSD Models", "EfficientNetB0_model.tflite"),
}

# API credentials — load from st.secrets when deployed; fallback for local dev
AUTH_KEY   = st.secrets.get("SMS_AUTH_KEY",   "")
AUTH_TOKEN = st.secrets.get("SMS_AUTH_TOKEN", "")

# Auto-detect developer mode (if keys are missing, empty, or placeholders)
IS_DEV_MODE = (
    not AUTH_KEY 
    or AUTH_KEY.startswith("YOUR_SMSCOUNTRY") 
    or not AUTH_TOKEN 
    or AUTH_TOKEN.startswith("YOUR_SMSCOUNTRY")
)

st.set_page_config(
    page_title="CHMI — Cattle Health Monitor",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded",
)




# -----------------------------------
# --- Utility Functions for OTP & SMS ---

def send_otp_sms(phone, otp):
    if IS_DEV_MODE:
        return True, "Simulated OTP sent successfully (Developer Mode)!"
    message = f"User Admin login OTP is {otp} - SMSCOU"
    credentials = f"{AUTH_KEY}:{AUTH_TOKEN}"
    encoded_credentials = base64.b64encode(credentials.encode()).decode()
    url = f"https://restapi.smscountry.com/v0.1/Accounts/{AUTH_KEY}/SMSes/"
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Authorization": f"Basic {encoded_credentials}"
    }
    payload = {
        "Text": message,
        "Number": phone,
        "SenderId": "SMSCOU",
        "DRNotifyUrl": "https://www.domainname.com/notifyurl",
        "Tool": "API"
    }
    try:
        response = requests.post(url, data=payload, headers=headers, timeout=10)
        if response.status_code == 202:
            return True, "OTP sent successfully!"
        else:
            return False, f"Failed to send OTP. Response: {response.text}"
    except Exception as e:
        return False, f"Error sending OTP: {e}"

def generate_otp():
    """Generate a cryptographically secure 6-digit OTP."""
    return str(secrets.randbelow(900000) + 100000)

def is_valid_mobile(mobile):
    pattern = re.compile(r"^\d{10}$")
    return pattern.match(mobile)


# -----------------------------------
# --- Tensorflow Model Utils ---

def normalize_probs(probs):
    """Normalize probability array; guard against zero-sum (BUG-009 fix)."""
    total = np.sum(probs)
    if total == 0:
        return np.ones_like(probs) / len(probs)  # uniform fallback
    return probs / total

def load_tflite_models(model_paths):
    """Load a dict of TFLite model paths into interpreter objects."""
    interps = {}
    for name, path in model_paths.items():
        interp = tf.lite.Interpreter(model_path=path)
        interp.allocate_tensors()
        interps[name] = interp
    return interps


@st.cache_resource
def get_fmd_models():
    """Load and cache all 5 FMD ensemble TFLite models (runs once)."""
    return load_tflite_models(TFLITE_MODELS_FMD)


@st.cache_resource
def get_lsd_models():
    """Load and cache all 5 LSD ensemble TFLite models (runs once)."""
    return load_tflite_models(TFLITE_MODELS_LSD)

def preprocess_image(uploaded_file):
    """Open, resize and normalise an uploaded file to float32 [0, 1]."""
    uploaded_file.seek(0)
    image = Image.open(uploaded_file).convert('RGB').resize((224, 224))
    image_array = np.array(image).astype('float32') / 255.0
    return image_array




def collapse_probs_to_n(probs, n):
    """Collapse a probability array to n classes by summing equal-sized groups.
    
    If len(probs) == n, returns probs unchanged.
    If len(probs) is a multiple of n, groups are summed and re-normalised.
    Otherwise, the first n values are taken and re-normalised.
    """
    probs = np.asarray(probs, dtype=np.float32)
    if len(probs) == n:
        return probs
    if len(probs) % n == 0:
        group_size = len(probs) // n
        collapsed = np.array([probs[i*group_size:(i+1)*group_size].sum() for i in range(n)], dtype=np.float32)
    else:
        collapsed = probs[:n].copy()
    total = collapsed.sum()
    if total > 0:
        collapsed /= total
    return collapsed


def soft_voting_ensemble(interpreters, image, weights, CLASS_NAMES):
    n_classes = len(CLASS_NAMES)
    total_weighted_probs = np.zeros(n_classes, dtype=np.float32)
    # Prepare input once — identical for every model in the ensemble
    input_data = np.expand_dims(image, axis=0).astype(np.float32)
    total_used_weight = 0.0

    for name, interpreter in interpreters.items():
        in_idx  = interpreter.get_input_details()[0]['index']
        out_idx = interpreter.get_output_details()[0]['index']
        interpreter.set_tensor(in_idx, input_data)
        interpreter.invoke()
        raw_probs = interpreter.get_tensor(out_idx)[0]
        # Collapse to target number of classes if needed
        probs = collapse_probs_to_n(normalize_probs(raw_probs), n_classes)
        w = weights.get(name, 0)
        total_weighted_probs += w * probs
        total_used_weight += w

    if total_used_weight == 0:
        total_used_weight = 1.0
    final_probs = total_weighted_probs / total_used_weight
    
    if n_classes == 4:
        group1 = float(final_probs[0] + final_probs[1])
        group2 = float(final_probs[2] + final_probs[3])
        if group1 >= group2:
            return group1, "Diseased"
        else:
            return group2, "Healthy"
    
    # For 2-class and others, return scalar probability and label
    pred_index = np.argmax(final_probs)
    pred_prob = float(final_probs[pred_index])
    pred_label = CLASS_NAMES[pred_index]
    return pred_prob, pred_label




# st.session_state.step = "dashboard"


# -----------------------------------
# --- Streamlit Session State Setup ---

if "step" not in st.session_state:
    st.session_state.step = "user_info"

if "otp_sent_at" not in st.session_state:
    st.session_state.otp_sent_at = 0

if "otp_code" not in st.session_state:
    st.session_state.otp_code = ""

if "otp_verified" not in st.session_state:
    st.session_state.otp_verified = False

if "disable_otp_request_until" not in st.session_state:
    st.session_state.disable_otp_request_until = 0

if "location" not in st.session_state:
    st.session_state.location = {"lat": None, "lon": None}

if "last_resolved_coords" not in st.session_state:
    st.session_state.last_resolved_coords = None

if "user_details" not in st.session_state:
    st.session_state.user_details = {}
    
if "cattle_details" not in st.session_state:
    st.session_state.cattle_details = False
    
# Initialize font size in session state
if "font_size" not in st.session_state:
    st.session_state.font_size = 20  # Default size

if "cattle_details_submitted" not in st.session_state:
    st.session_state["cattle_details_submitted"] = False

if "cattle_id" not in st.session_state:
    st.session_state["cattle_id"] = ""

if "gender" not in st.session_state:
    st.session_state["gender"] = ""

if "age" not in st.session_state:
    st.session_state["age"] = 0.0

if "otp_attempts" not in st.session_state:
    st.session_state["otp_attempts"] = 0

if "scan_history" not in st.session_state:
    st.session_state.scan_history = []

if "language" not in st.session_state:
    st.session_state.language = "en"

if "pdf_report_bytes" not in st.session_state:
    st.session_state.pdf_report_bytes = None

if "pdf_report_filename" not in st.session_state:
    st.session_state.pdf_report_filename = ""



# Sidebar progress tracking

step_order = ["user_info", "otp_verify", "dashboard"]
step_labels = {
    "user_info": "Step 1: User Info",
    "otp_verify": "Step 2: OTP Verification",
    "dashboard": "Step 3: Dashboard"
}



# ── Sidebar ─────────────────────────────────────────────────────────────────
current_step = st.session_state.get("step", "user_info")

# ── Brand Header ──
st.sidebar.markdown("""
    <div style="padding: 1.5rem 0 1rem 0;">
        <div style="display:flex; align-items:center; gap:0.75rem; margin-bottom:0.4rem;">
            <div style="width:42px;height:42px;background:linear-gradient(135deg,#059669,#047857);
                        border-radius:12px;display:flex;align-items:center;justify-content:center;
                        font-size:1.4rem;flex-shrink:0;">🐄</div>
            <div>
                <div style="font-size:1.1rem;font-weight:800;color:#f1f5f9;letter-spacing:-0.01rem;">CHMI</div>
                <div style="font-size:0.72rem;color:#64748b;font-weight:500;letter-spacing:0.04rem;text-transform:uppercase;">Cattle Health Monitor</div>
            </div>
        </div>
    </div>
    <div style="border-top:1px solid #1e293b;margin-bottom:1.25rem;"></div>
""", unsafe_allow_html=True)

# ── Vertical Stepper with connector lines ──
_step_order = ["user_info", "otp_verify", "dashboard"]
_step_info  = [("User Info", "1", "Enter your details"), ("Verify OTP", "2", "Confirm your number"), ("Diagnose", "3", "Upload & analyze")]
_cur_idx    = _step_order.index(current_step) if current_step in _step_order else 0

st.sidebar.markdown("<div style='font-size:0.68rem;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.1rem;margin-bottom:0.75rem;'>Workflow</div>", unsafe_allow_html=True)

for i, (label, num, subtitle) in enumerate(_step_info):
    if i < _cur_idx:
        dot_bg, dot_border, dot_shadow = "#059669", "none", "0 0 0 3px rgba(5,150,105,0.15)"
        dot_inner, dot_color = "✓", "#fff"
        txt_color, sub_color, weight = "#94a3b8", "#475569", "500"
    elif i == _cur_idx:
        dot_bg, dot_border, dot_shadow = "#059669", "none", "0 0 0 4px rgba(5,150,105,0.25), 0 0 12px rgba(5,150,105,0.35)"
        dot_inner, dot_color = num, "#fff"
        txt_color, sub_color, weight = "#f1f5f9", "#94a3b8", "700"
    else:
        dot_bg, dot_border, dot_shadow = "#1e293b", "2px solid #334155", "none"
        dot_inner, dot_color = num, "#475569"
        txt_color, sub_color, weight = "#475569", "#334155", "400"

    # Build step HTML — must start at column 0 to avoid Streamlit code-block rendering
    _step_html = (
        f'<div style="display:flex;align-items:center;gap:0.75rem;padding:0.3rem 0;">'
        f'<div style="width:28px;height:28px;background:{dot_bg};border:{dot_border};border-radius:50%;'
        f'display:flex;align-items:center;justify-content:center;'
        f'font-size:0.72rem;font-weight:700;color:{dot_color};flex-shrink:0;'
        f'box-shadow:{dot_shadow};transition:all 0.3s ease;">{dot_inner}</div>'
        f'<div style="flex:1;min-width:0;">'
        f'<div style="font-size:0.85rem;font-weight:{weight};color:{txt_color};line-height:1.3;">{label}</div>'
        f'<div style="font-size:0.68rem;color:{sub_color};line-height:1.3;margin-top:1px;">{subtitle}</div>'
        f'</div></div>'
    )
    st.sidebar.markdown(_step_html, unsafe_allow_html=True)

    # Connector line (between steps, not after the last)
    if i < len(_step_info) - 1:
        _line_color = "#059669" if i < _cur_idx else "#1e293b"
        st.sidebar.markdown(f'<div style="width:2px;height:20px;background:{_line_color};margin-left:13px;border-radius:1px;margin-top:-8px;margin-bottom:-8px;"></div>', unsafe_allow_html=True)

# ── User Context Card (shown when authenticated) ──
if current_step == "dashboard" and st.session_state.get("user_details"):
    _u = st.session_state.user_details
    _initial = (_u.get("name", "U") or "U")[0].upper()
    _uname = _u.get("name", "User")
    _umobile = _u.get("mobile", "")
    _uloc = ", ".join(filter(None, [_u.get("village", ""), _u.get("district", "")]))
    st.sidebar.markdown(f"""
        <div style="border-top:1px solid #1e293b;margin:1rem 0;"></div>
        <div style="font-size:0.68rem;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.1rem;margin-bottom:0.6rem;">Logged In As</div>
        <div style="background:#1e293b;border-radius:12px;padding:0.75rem 0.9rem;border:1px solid #334155;">
            <div style="display:flex;align-items:center;gap:0.65rem;">
                <div style="width:34px;height:34px;background:linear-gradient(135deg,#059669,#047857);
                            border-radius:50%;display:flex;align-items:center;justify-content:center;
                            font-size:0.85rem;font-weight:700;color:#fff;flex-shrink:0;">{_initial}</div>
                <div style="flex:1;min-width:0;">
                    <div style="font-size:0.85rem;font-weight:700;color:#f1f5f9;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{_uname}</div>
                    <div style="font-size:0.7rem;color:#64748b;font-weight:500;">+91 {_umobile}</div>
                </div>
            </div>
            {f'<div style="font-size:0.68rem;color:#475569;margin-top:0.5rem;padding-top:0.5rem;border-top:1px solid #334155;">📍 {_uloc}</div>' if _uloc else ''}
        </div>
    """, unsafe_allow_html=True)

# ── Preferences: Text Size + Language ──
st.sidebar.markdown("<div style='border-top:1px solid #1e293b;margin:1rem 0;'></div>", unsafe_allow_html=True)
st.sidebar.markdown("<div style='font-size:0.68rem;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.1rem;margin-bottom:0.6rem;'>Preferences</div>", unsafe_allow_html=True)

# Text size — compact inline layout
st.sidebar.markdown("<div style='font-size:0.75rem;color:#64748b;margin-bottom:0.4rem;font-weight:500;'>Text Size</div>", unsafe_allow_html=True)
_fs_col1, _fs_col2, _fs_col3 = st.sidebar.columns([1, 2, 1])
with _fs_col1:
    if st.button("A−", help="Decrease font size", key="fs_dec", use_container_width=True):
        st.session_state.font_size = max(14, st.session_state.font_size - 2)
with _fs_col2:
    _fs_pct = round((st.session_state.font_size - 14) / (26 - 14) * 100)
    st.sidebar.markdown(
        f'<div style="text-align:center;padding-top:0.3rem;">'
        f'<div style="font-weight:700;color:#f1f5f9;font-size:0.88rem;line-height:1;">{st.session_state.font_size}px</div>'
        f'<div style="background:#334155;border-radius:4px;height:3px;margin-top:5px;overflow:hidden;">'
        f'<div style="background:#059669;height:100%;width:{_fs_pct}%;border-radius:4px;transition:width 0.2s;"></div>'
        f'</div></div>', unsafe_allow_html=True)
with _fs_col3:
    if st.button("A+", help="Increase font size", key="fs_inc", use_container_width=True):
        st.session_state.font_size = min(26, st.session_state.font_size + 2)

# Language — using selectbox to avoid wrapping radio labels
st.sidebar.markdown("<div style='font-size:0.75rem;color:#64748b;margin:0.75rem 0 0.35rem 0;font-weight:500;'>Language / భాష</div>", unsafe_allow_html=True)
_lang_options = ["English", "తెలుగు"]
_lang_idx = 1 if st.session_state.language == "te" else 0
_lang_choice = st.sidebar.selectbox("Language", _lang_options, index=_lang_idx, key="lang_select", label_visibility="collapsed")
st.session_state.language = "te" if _lang_choice == "తెలుగు" else "en"

# ── Scan History Badge ──
if st.session_state.scan_history:
    _scan_count = len(st.session_state.scan_history)
    st.sidebar.markdown(f"""
        <div style="border-top:1px solid #1e293b;margin:1rem 0;"></div>
        <div style="display:flex;align-items:center;justify-content:space-between;">
            <div style="font-size:0.68rem;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.1rem;">Session</div>
            <div style="display:flex;align-items:center;gap:0.4rem;background:#059669;padding:0.2rem 0.65rem;border-radius:100px;">
                <span style="font-size:0.72rem;font-weight:700;color:#fff;">{_scan_count}</span>
                <span style="font-size:0.65rem;color:rgba(255,255,255,0.8);font-weight:500;">scan{"s" if _scan_count != 1 else ""}</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

# ── New Session Button ──
if current_step == "dashboard":
    st.sidebar.markdown("<div style='border-top:1px solid #1e293b;margin:1rem 0;'></div>", unsafe_allow_html=True)
    if st.sidebar.button("🔄  New Session", key="sb_new_session", use_container_width=True):
        for _k in list(st.session_state.keys()):
            del st.session_state[_k]
        st.rerun()


# ── Global font + base CSS ────────────────────────────────────────────────────
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)
st.markdown(f"""
<style>
/* ══════════════════════════════════════════════════
   CHMI — Professional Design System v2
   ══════════════════════════════════════════════════ */

html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}}
* {{ box-sizing: border-box; }}
body, p, li, label, td, .info-row-val, .alert-body {{
    font-size: {st.session_state.font_size}px !important;
    line-height: 1.65 !important;
    color: #1e293b;
}}

/* ── Streamlit chrome ── */
#MainMenu, footer, [data-testid="stHeaderActionElements"] {{ visibility: hidden !important; }}
header {{ background-color: transparent !important; }}
.block-container {{
    padding-top: 0 !important;
    padding-bottom: 3rem !important;
    max-width: 1160px;
}}

/* ── Page background ── */
[data-testid="stAppViewContainer"] > .main {{ background: #f8fafc; }}

/* ── Sidebar — dark ── */
[data-testid="stSidebar"] > div:first-child {{
    background: #0f172a;
    border-right: none;
    box-shadow: 4px 0 24px rgba(0,0,0,0.15);
}}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] label {{ color: #cbd5e1 !important; }}
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {{ color: #f1f5f9 !important; }}

/* ── Streamlit buttons ── */
/* Default (Secondary) Button style */
div.stButton button {{
    background: #ffffff !important;
    color: #047857 !important;
    border: 2px solid #059669 !important;
    border-radius: 10px !important;
    padding: 0.65rem 1.25rem !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.01rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 1px 4px rgba(5,150,105,0.1) !important;
    height: auto !important;
    line-height: 1.5 !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    white-space: nowrap !important;
    word-break: keep-all !important;
}}
div.stButton button:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 14px rgba(5,150,105,0.2) !important;
    background: #f0fdf4 !important;
}}
div.stButton button:active {{ transform: translateY(0) !important; }}

/* Primary Button style overrides */
div.stButton button[kind="primary"],
div.stButton button[data-testid="baseButton-primary"] {{
    background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
    color: #ffffff !important;
    border: none !important;
    box-shadow: 0 2px 10px rgba(5,150,105,0.35) !important;
}}
div.stButton button[kind="primary"]:hover,
div.stButton button[data-testid="baseButton-primary"]:hover {{
    background: linear-gradient(135deg, #047857 0%, #065f46 100%) !important;
    box-shadow: 0 6px 20px rgba(5,150,105,0.5) !important;
    color: #ffffff !important;
}}

/* Ensure typography and nowrap for inner elements in buttons */
div.stButton button p,
div.stButton button span,
div.stButton button div,
div.stButton button * {{
    color: inherit !important;
    white-space: nowrap !important;
    word-break: keep-all !important;
    margin: 0 !important;
    padding: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
}}
div.stButton button[kind="primary"] p,
div.stButton button[kind="primary"] span,
div.stButton button[kind="primary"] div,
div.stButton button[kind="primary"] *,
div.stButton button[data-testid="baseButton-primary"] p,
div.stButton button[data-testid="baseButton-primary"] span,
div.stButton button[data-testid="baseButton-primary"] div,
div.stButton button[data-testid="baseButton-primary"] * {{
    color: #ffffff !important;
    background: transparent !important;
    box-shadow: none !important;
}}

/* ── Download button ── */
.stDownloadButton button {{
    background: #ffffff !important;
    color: #047857 !important;
    border: 2px solid #059669 !important;
    border-radius: 10px !important;
    padding: 0.65rem 2rem !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 1px 4px rgba(5,150,105,0.15) !important;
    height: auto !important;
    line-height: 1.5 !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    white-space: nowrap !important;
    word-break: keep-all !important;
}}
.stDownloadButton button:hover {{
    background: #f0fdf4 !important;
    box-shadow: 0 4px 14px rgba(5,150,105,0.25) !important;
    transform: translateY(-1px) !important;
}}
.stDownloadButton button p,
.stDownloadButton button span,
.stDownloadButton button div,
.stDownloadButton button * {{
    color: inherit !important;
    white-space: nowrap !important;
    word-break: keep-all !important;
    margin: 0 !important;
    padding: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
}}

/* ── Form submit button ── */
.stFormSubmitButton button {{
    background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.7rem 2rem !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    box-shadow: 0 2px 10px rgba(5,150,105,0.35) !important;
    transition: all 0.2s !important;
    height: auto !important;
    line-height: 1.5 !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    white-space: nowrap !important;
    word-break: keep-all !important;
}}
.stFormSubmitButton button:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(5,150,105,0.5) !important;
}}
.stFormSubmitButton button p,
.stFormSubmitButton button span,
.stFormSubmitButton button div,
.stFormSubmitButton button * {{
    color: #ffffff !important;
    white-space: nowrap !important;
    word-break: keep-all !important;
    margin: 0 !important;
    padding: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
}}

/* ── Inputs ── */
.stTextInput > div > div > input,
.stNumberInput > div > div > input {{
    border-radius: 10px !important;
    border: 1.5px solid #e2e8f0 !important;
    padding: 0.6rem 1rem !important;
    background: #ffffff !important;
    transition: all 0.2s !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
    font-size: {st.session_state.font_size}px !important;
}}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {{
    border-color: #059669 !important;
    box-shadow: 0 0 0 3px rgba(5,150,105,0.12) !important;
    outline: none !important;
}}

/* ── Selectbox ── */
.stSelectbox > div > div > div {{
    border-radius: 10px !important;
    border: 1.5px solid #e2e8f0 !important;
    background: #ffffff !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
}}

/* ── File uploader ── */
[data-testid="stFileUploader"] section {{
    border: 2px dashed #a7f3d0 !important;
    border-radius: 14px !important;
    background: #f0fdf4 !important;
    transition: border-color 0.2s !important;
}}
[data-testid="stFileUploader"] section:hover {{
    border-color: #059669 !important;
    background: #ecfdf5 !important;
}}

/* ── Alerts ── */
.stSuccess {{ background:#f0fdf4 !important; border-left:4px solid #059669 !important; border-radius:10px !important; }}
.stError   {{ background:#fef2f2 !important; border-left:4px solid #dc2626 !important; border-radius:10px !important; }}
.stWarning {{ background:#fffbeb !important; border-left:4px solid #d97706 !important; border-radius:10px !important; }}
.stInfo    {{ background:#eff6ff !important; border-left:4px solid #3b82f6 !important; border-radius:10px !important; }}

/* ── Radio ── */
.stRadio > div {{ gap: 0.6rem !important; }}
.stRadio label {{
    font-weight: 500 !important;
    background: #ffffff !important;
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 10px !important;
    padding: 0.45rem 1.1rem !important;
    cursor: pointer !important;
    transition: all 0.15s !important;
}}
.stRadio label p,
.stRadio label span,
.stRadio label div {{
    color: #1e293b !important;
    font-weight: 500 !important;
}}
.stRadio label:hover {{
    border-color: #059669 !important;
    background: #f0fdf4 !important;
}}
.stRadio label:hover p,
.stRadio label:hover span,
.stRadio label:hover div {{
    color: #059669 !important;
}}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{
    background: #f1f5f9 !important;
    border-radius: 10px !important;
    padding: 0.25rem !important;
    gap: 0.2rem !important;
    border: none !important;
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 8px !important;
    font-weight: 500 !important;
    color: #64748b !important;
    background: transparent !important;
    border: none !important;
}}
.stTabs [aria-selected="true"] {{
    background: #ffffff !important;
    color: #047857 !important;
    font-weight: 600 !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.1) !important;
}}

/* ── Expander ── */
details > summary {{
    background: #f8fafc !important;
    border-radius: 10px !important;
    border: 1px solid #e2e8f0 !important;
    font-weight: 600 !important;
    color: #1e293b !important;
    padding: 0.75rem 1rem !important;
}}

/* ── Spinner ── */
.stSpinner > div {{ border-top-color: #059669 !important; }}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {{
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid #e2e8f0 !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05) !important;
}}

/* ── Sidebar controls ── */
[data-testid="stSidebar"] .stButton button {{
    background: #1e293b !important;
    box-shadow: none !important;
    border: 1px solid #334155 !important;
    padding: 0.45rem 0.85rem !important;
    border-radius: 8px !important;
    transform: none !important;
    white-space: nowrap !important;
    word-break: keep-all !important;
    transition: all 0.2s ease !important;
}}
[data-testid="stSidebar"] .stButton button p,
[data-testid="stSidebar"] .stButton button span,
[data-testid="stSidebar"] .stButton button div,
[data-testid="stSidebar"] .stButton button * {{
    color: #cbd5e1 !important;
    font-size: 0.82rem !important;
    white-space: nowrap !important;
    word-break: keep-all !important;
}}
[data-testid="stSidebar"] .stButton button:hover {{
    background: #334155 !important;
    border-color: #475569 !important;
    transform: none !important;
    box-shadow: none !important;
}}
[data-testid="stSidebar"] .stButton button:hover p,
[data-testid="stSidebar"] .stButton button:hover span,
[data-testid="stSidebar"] .stButton button:hover div,
[data-testid="stSidebar"] .stButton button:hover * {{
    color: #ffffff !important;
}}

/* ── Sidebar selectbox (language picker) ── */
[data-testid="stSidebar"] .stSelectbox > div > div > div {{
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
    color: #cbd5e1 !important;
    font-size: 0.82rem !important;
}}
[data-testid="stSidebar"] .stSelectbox > div > div > div:hover {{
    border-color: #475569 !important;
}}
[data-testid="stSidebar"] .stSelectbox label {{
    color: #64748b !important;
}}
[data-testid="stSidebar"] .stSelectbox svg {{
    fill: #94a3b8 !important;
    color: #94a3b8 !important;
}}
[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] span,
[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] div {{
    color: #cbd5e1 !important;
}}

/* ── Map ── */
[data-testid="stDeckGlJsonChart"], .stMapbox {{
    border-radius: 14px !important;
    overflow: hidden !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08) !important;
}}
</style>
""", unsafe_allow_html=True)


def load_custom_css():
    st.markdown("""
    <style>

    /* ══ HERO BANNER ══════════════════════════════════════════ */
    .chmi-hero {
        background: linear-gradient(135deg, #064e3b 0%, #065f46 50%, #047857 100%);
        border-radius: 20px;
        padding: 2.5rem 2.5rem 2rem;
        margin-bottom: 0;
        position: relative;
        overflow: hidden;
    }
    .chmi-hero::before {
        content: '';
        position: absolute;
        top: -80px; right: -80px;
        width: 280px; height: 280px;
        background: rgba(255,255,255,0.04);
        border-radius: 50%;
        pointer-events: none;
    }
    .chmi-hero::after {
        content: '';
        position: absolute;
        bottom: -50px; left: -50px;
        width: 200px; height: 200px;
        background: rgba(255,255,255,0.03);
        border-radius: 50%;
        pointer-events: none;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: rgba(255,255,255,0.1);
        border: 1px solid rgba(255,255,255,0.18);
        border-radius: 100px;
        padding: 0.3rem 0.9rem;
        font-size: 0.75rem;
        font-weight: 700;
        color: #a7f3d0;
        letter-spacing: 0.08rem;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }
    .hero-title {
        color: #ffffff !important;
        font-size: 2.1rem;
        font-weight: 800;
        margin: 0 0 0.5rem 0;
        letter-spacing: -0.03rem;
        line-height: 1.15;
    }
    .hero-subtitle {
        color: rgba(255,255,255,0.72) !important;
        font-size: 0.95rem;
        margin: 0 0 1.5rem 0;
        max-width: 540px;
        line-height: 1.6;
    }
    .hero-stats {
        display: flex;
        gap: 0.75rem;
        flex-wrap: wrap;
    }
    .hero-stat {
        background: rgba(255,255,255,0.1);
        border: 1px solid rgba(255,255,255,0.14);
        border-radius: 12px;
        padding: 0.55rem 1rem;
        text-align: center;
        min-width: 80px;
    }
    .hero-stat-num {
        font-size: 1.25rem;
        font-weight: 800;
        color: #6ee7b7;
        display: block;
        line-height: 1.2;
    }
    .hero-stat-label {
        font-size: 0.7rem;
        color: rgba(255,255,255,0.6);
        text-transform: uppercase;
        letter-spacing: 0.07rem;
        font-weight: 500;
    }

    /* ══ STEP INDICATOR ══════════════════════════════════════ */
    .stepper {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 1.5rem 2rem 1.25rem;
        background: #ffffff;
        border-bottom: 1px solid #f1f5f9;
        margin-bottom: 1.75rem;
        position: sticky;
        top: 0;
        z-index: 10;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .step {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 0.35rem;
    }
    .step-circle {
        width: 36px; height: 36px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .step--done   .step-circle { background:#059669; color:#fff; box-shadow:0 2px 8px rgba(5,150,105,0.4); }
    .step--active .step-circle { background:#047857; color:#fff; box-shadow:0 0 0 4px rgba(5,150,105,0.15), 0 2px 8px rgba(5,150,105,0.4); }
    .step--pending .step-circle { background:#f1f5f9; color:#94a3b8; border:2px solid #e2e8f0; }
    .step-label {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06rem;
        white-space: nowrap;
    }
    .step--done   .step-label  { color: #059669; }
    .step--active .step-label  { color: #047857; }
    .step--pending .step-label { color: #94a3b8; }
    .step-line {
        flex: 1;
        height: 2px;
        background: #e2e8f0;
        margin: 0 0.75rem;
        margin-bottom: 1.5rem;
        min-width: 48px;
        max-width: 100px;
    }
    .step-line--done { background: #059669; }

    /* ══ SECTION LABELS ══════════════════════════════════════ */
    .section-label {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin: 1.75rem 0 1rem 0;
    }
    .section-label-icon {
        width: 34px; height: 34px;
        background: #dcfce7;
        border-radius: 9px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1rem;
        flex-shrink: 0;
    }
    .section-label-text {
        font-size: 1rem;
        font-weight: 700;
        color: #0f172a;
        letter-spacing: -0.01rem;
        margin: 0;
    }
    .section-label-sub {
        font-size: 0.8rem;
        color: #64748b;
        margin: 0;
    }

    /* ══ INFO CARDS ══════════════════════════════════════════ */
    /* ══ INFO CARDS ══════════════════════════════════════════ */
    .info-card {
        background: #ffffff;
        padding: 1.25rem 1.5rem;
        border-radius: 16px;
        border: 1px solid #f1f5f9;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 4px 16px rgba(0,0,0,0.02);
        margin-bottom: 1rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .info-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
        border-color: #e2e8f0;
    }
    .info-card-label {
        display: flex;
        align-items: center;
        gap: 0.4rem;
        font-size: 0.72rem;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.08rem;
        margin-bottom: 0.75rem;
    }
    .info-card-label-dot {
        width: 6px; height: 6px;
        border-radius: 50%;
        background: #059669;
        flex-shrink: 0;
    }
    .info-row {
        display: flex;
        align-items: baseline;
        gap: 0.4rem;
        margin: 0.3rem 0;
        font-size: 0.88rem;
    }
    .info-row-key {
        color: #94a3b8;
        font-weight: 500;
        flex-shrink: 0;
        min-width: 72px;
    }
    .info-row-val {
        color: #0f172a;
        font-weight: 600;
    }

    /* ══ ACTION CARDS ════════════════════════════════════════ */
    .action-card {
        background: #ffffff;
        padding: 1rem 1.25rem;
        border-radius: 14px;
        border: 1px solid #f1f5f9;
        box-shadow: 0 1px 4px rgba(0,0,0,0.03);
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .action-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.05);
        border-color: #059669;
    }
    .action-card-icon {
        width: 38px; height: 38px;
        border-radius: 10px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.1rem;
        flex-shrink: 0;
    }
    .action-card-icon--green { background: #dcfce7; }
    .action-card-icon--blue  { background: #dbeafe; }
    .action-card-body { flex: 1; }
    .action-card-title {
        font-size: 0.92rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0 0 0.1rem 0;
    }
    .action-card-sub {
        font-size: 0.75rem;
        color: #94a3b8;
        margin: 0;
    }

    /* ══ RESULT CARDS ════════════════════════════════════════ */
    .result-card {
        border-radius: 20px;
        padding: 2rem 1.75rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .result-card--diseased {
        background: linear-gradient(145deg, #fff5f5 0%, #fef2f2 100%);
        border: 1.5px solid #fecaca;
        box-shadow: 0 4px 24px rgba(220,38,38,0.1);
    }
    .result-card--healthy {
        background: linear-gradient(145deg, #f0fdf4 0%, #ecfdf5 100%);
        border: 1.5px solid #a7f3d0;
        box-shadow: 0 4px 24px rgba(5,150,105,0.1);
    }
    .result-status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        border-radius: 100px;
        padding: 0.3rem 0.9rem;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.07rem;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }
    .result-status-badge--diseased { background:#fef2f2; color:#dc2626; border:1px solid #fecaca; }
    .result-status-badge--healthy  { background:#f0fdf4; color:#15803d; border:1px solid #a7f3d0; }
    .result-icon { font-size: 3rem; margin-bottom: 0.6rem; display: block; }
    .result-status-text {
        font-size: 1.55rem;
        font-weight: 800;
        margin: 0 0 0.35rem 0;
        letter-spacing: -0.02rem;
    }
    .result-status-text--diseased { color: #dc2626; }
    .result-status-text--healthy  { color: #15803d; }
    .result-disease-label {
        font-size: 0.82rem;
        color: #64748b;
        margin-bottom: 1.25rem;
        font-weight: 500;
    }
    .confidence-bar-track {
        background: rgba(0,0,0,0.08);
        border-radius: 100px;
        height: 8px;
        overflow: hidden;
        margin: 0 auto 0.5rem auto;
        max-width: 260px;
    }
    .confidence-bar-fill {
        height: 100%;
        border-radius: 100px;
    }
    .confidence-bar-fill--diseased { background: linear-gradient(90deg, #f87171, #dc2626); }
    .confidence-bar-fill--healthy  { background: linear-gradient(90deg, #34d399, #059669); }
    .confidence-pct {
        font-size: 0.82rem;
        font-weight: 700;
    }
    .confidence-pct--diseased { color: #ef4444; }
    .confidence-pct--healthy  { color: #059669; }

    /* ══ VET CONTACT CARDS ═══════════════════════════════════ */
    .vet-section-title {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08rem;
        color: #64748b;
        margin: 0 0 0.75rem 0;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    .vet-card {
        background: #ffffff;
        border-radius: 14px;
        padding: 1.1rem 1.25rem;
        margin-bottom: 0.75rem;
        border: 1px solid #f1f5f9;
        box-shadow: 0 2px 8px rgba(0,0,0,0.03);
        display: flex;
        align-items: flex-start;
        gap: 0.9rem;
        transition: all 0.2s ease;
    }
    .vet-card:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    }
    .vet-avatar {
        width: 42px; height: 42px;
        border-radius: 12px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.2rem;
        flex-shrink: 0;
    }
    .vet-avatar--doctor  { background: #dbeafe; }
    .vet-avatar--mitra   { background: #dcfce7; }
    .vet-info { flex: 1; }
    .vet-name {
        font-size: 0.9rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0 0 0.15rem 0;
    }
    .vet-role-loc {
        font-size: 0.75rem;
        color: #64748b;
        margin: 0 0 0.4rem 0;
        font-weight: 500;
    }
    .vet-phone {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        background: #f0fdf4;
        color: #047857;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        border: 1px solid #a7f3d0;
        text-decoration: none;
        transition: all 0.15s ease;
    }
    .vet-phone:hover {
        background: #047857;
        color: #ffffff !important;
        box-shadow: 0 2px 6px rgba(4,120,87,0.2);
    }

    /* ══ PREVIEW PLACEHOLDER ═════════════════════════════════ */
    .preview-area {
        background: #f8fafc;
        border-radius: 14px;
        padding: 2.5rem 1rem;
        border: 2px dashed #cbd5e1;
        text-align: center;
        min-height: 280px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        color: #94a3b8;
        transition: all 0.2s;
    }

    /* ══ ALERT CARDS ═════════════════════════════════════════ */
    .alert-card {
        border-radius: 16px;
        padding: 1.25rem 1.5rem;
        display: flex;
        align-items: flex-start;
        gap: 1rem;
        margin: 1rem 0;
    }
    .alert-card--warning { background:#fffbeb; border:1.5px solid #fde68a; }
    .alert-card--danger  { background:#fef2f2; border:1.5px solid #fecaca; }
    .alert-card--success { background:#f0fdf4; border:1.5px solid #a7f3d0; }
    .alert-icon  { font-size:1.5rem; flex-shrink:0; margin-top:0.1rem; }
    .alert-title { font-weight:700; font-size:0.95rem; margin:0 0 0.3rem 0; }
    .alert-title--warning { color:#92400e; }
    .alert-title--danger  { color:#991b1b; }
    .alert-title--success { color:#065f46; }
    .alert-body  { font-size:0.85rem; margin:0; line-height:1.65; }
    .alert-body--warning { color:#78350f; }
    .alert-body--danger  { color:#7f1d1d; }
    .alert-body--success { color:#065f46; }

    /* ══ MISC ════════════════════════════════════════════════ */
    .divider { border:none; border-top:1px solid #f1f5f9; margin:1.5rem 0; }

    /* ══ CUSTOM RADIO BUTTON SELECT CARDS ════════════════════ */
    div[data-testid="stRadio"] div[role="radiogroup"] {
        display: flex;
        flex-direction: row;
        gap: 1rem;
        width: 100%;
        margin-top: 0.5rem;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] label {
        flex: 1;
        background: #ffffff !important;
        border: 2px solid #e2e8f0 !important;
        border-radius: 12px !important;
        padding: 0.8rem 1.25rem !important;
        text-align: center;
        cursor: pointer;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02) !important;
        display: inline-flex;
        align-items: center;
        justify-content: center;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] label:hover {
        border-color: #059669 !important;
        background: #f0fdf4 !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(5,150,105,0.08) !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] label:has(input[checked]),
    div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
        border-color: #059669 !important;
        background: #e6f4ea !important;
        font-weight: 600 !important;
        box-shadow: 0 0 0 3px rgba(5,150,105,0.15), 0 4px 12px rgba(5,150,105,0.1) !important;
    }

    /* ══ CUSTOM FILE UPLOADER ════════════════════════════════ */
    div[data-testid="stFileUploader"] {
        border: 2px dashed #cbd5e1 !important;
        border-radius: 14px !important;
        background: #f8fafc !important;
        padding: 1rem !important;
        transition: all 0.2s ease-in-out !important;
    }
    div[data-testid="stFileUploader"]:hover {
        border-color: #059669 !important;
        background: #f0fdf4 !important;
    }
    div[data-testid="stFileUploader"] section {
        border: none !important;
        background: transparent !important;
        padding: 0 !important;
    }

    /* ══ CUSTOM MAP WIDGET ═══════════════════════════════════ */
    div[data-testid="stMap"] iframe {
        border-radius: 16px !important;
        border: 1px solid #e2e8f0 !important;
    }

    </style>
    """, unsafe_allow_html=True)
    
  

def render_stepper(current: str):
    steps = [
        ("user_info",  "1", "User Info"),
        ("otp_verify", "2", "Verify OTP"),
        ("dashboard",  "3", "Diagnose"),
    ]
    order = [s[0] for s in steps]
    cur_idx = order.index(current) if current in order else 0

    parts = []
    for i, (key, num, label) in enumerate(steps):
        idx = order.index(key)
        if idx < cur_idx:
            cls = "step--done"
            circle = "✓"
        elif idx == cur_idx:
            cls = "step--active"
            circle = num
        else:
            cls = "step--pending"
            circle = num
        parts.append(
            f'<div class="step {cls}">'
            f'  <div class="step-circle">{circle}</div>'
            f'  <span class="step-label">{label}</span>'
            f'</div>'
        )
        if i < len(steps) - 1:
            line_cls = "step-line--done" if cur_idx > idx else ""
            parts.append(f'<div class="step-line {line_cls}"></div>')

    st.markdown(
        f'<div class="stepper">{"".join(parts)}</div>',
        unsafe_allow_html=True,
    )


def generate_pdf_report(user, loc, cattle_id, gender, age, disease_type, disease_status, confidence, vet_first=None, gopa_first=None):
    """Build and return a PDF prediction report as bytes."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ── Header ────────────────────────────────────────────────────────────────
    pdf.set_fill_color(22, 163, 74)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 14, "CHMI - Cattle Health Monitor", fill=True, ln=True, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, "AI-Powered Disease Detection Report", fill=True, ln=True, align="C")
    pdf.set_text_color(17, 24, 39)
    pdf.ln(6)

    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(107, 114, 128)
    pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="R")
    pdf.ln(4)

    def _section(title):
        pdf.set_fill_color(240, 253, 244)
        pdf.set_text_color(22, 101, 52)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 9, f"  {title}", fill=True, ln=True)
        pdf.set_text_color(17, 24, 39)
        pdf.ln(2)

    def _row(label, value):
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(75, 85, 99)
        pdf.cell(52, 7, label + ":", ln=False)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(17, 24, 39)
        pdf.multi_cell(0, 7, str(value))
        pdf.ln(0)

    # ── User Details ──────────────────────────────────────────────────────────
    _section("User Details")
    _row("Name", user.get("name", "N/A"))
    _row("Mobile", user.get("mobile", "N/A"))
    _row("Village", user.get("village", "N/A"))
    _row("Mandal", user.get("mandal", "N/A"))
    _row("District", user.get("district", "N/A"))
    _row("State", user.get("state", "N/A"))
    _row("Pincode", user.get("pincode", "N/A"))
    if loc.get("lat") and loc.get("lon"):
        _row("GPS Location", f"{loc['lat']:.6f}, {loc['lon']:.6f}")
    pdf.ln(4)

    # ── Cattle Details ────────────────────────────────────────────────────────
    _section("Cattle Details")
    _row("Cattle ID", cattle_id)
    _row("Gender", gender)
    _row("Age", f"{age} years")
    _row("Analysis Type", disease_type)
    pdf.ln(4)

    # ── Prediction Result ─────────────────────────────────────────────────────
    _section("Prediction Result")
    is_diseased = disease_status != "Healthy"
    if is_diseased:
        pdf.set_fill_color(254, 226, 226)
        pdf.set_text_color(185, 28, 28)
    else:
        pdf.set_fill_color(240, 253, 244)
        pdf.set_text_color(22, 101, 52)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 12, f"  Status: {disease_status}", fill=True, ln=True, align="C")
    pdf.set_text_color(17, 24, 39)
    pdf.ln(3)
    _row("Confidence Score", f"{confidence:.2f}%")
    pdf.ln(4)

    # ── Vet Contact (diseased only) ───────────────────────────────────────────
    if is_diseased:
        _section("Recommended Veterinary Support")
        if vet_first is not None:
            try:
                _row("Veterinary Doctor", str(vet_first.get("Name", "N/A")))
                _row("Location", str(vet_first.get("Place of working", "N/A")))
                _row("Contact", str(vet_first.get("Mobile no.", "N/A")))
            except Exception:
                pass
        if gopa_first is not None:
            try:
                pdf.ln(2)
                _row("Gopalamitra", str(gopa_first.get("Name", "N/A")))
                _row("Mandal", str(gopa_first.get("mandal", "N/A")))
                _row("Contact", str(gopa_first.get("mobile no.", "N/A")))
            except Exception:
                pass
        pdf.ln(4)

    # ── Footer ────────────────────────────────────────────────────────────────
    pdf.set_y(-20)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(156, 163, 175)
    pdf.cell(0, 5, "This report is generated by CHMI AI. Always consult a licensed veterinary doctor for treatment decisions.", ln=True, align="C")

    return bytes(pdf.output())


def reverse_geocode(lat, lon):
    """Reverse geocode latitude and longitude to retrieve Village, Mandal, and District."""
    if lat is None or lon is None:
        return None
    url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json&accept-language=en"
    headers = {"User-Agent": "CHMIAssistant/2.0 (contact: support@dodailsolutions.com)"}
    try:
        r = requests.get(url, headers=headers, timeout=5)
        if r.status_code == 200:
            data = r.json()
            addr = data.get("address", {})
            
            # Robust extraction of Village name
            village = (
                addr.get("village") or
                addr.get("hamlet") or
                addr.get("suburb") or
                addr.get("neighbourhood") or
                addr.get("quarter") or
                addr.get("town") or
                addr.get("city_district") or
                addr.get("residential") or
                addr.get("municipality") or
                addr.get("road") or
                addr.get("historic") or
                addr.get("city") or
                addr.get("state") or
                ""
            )
            
            # Robust extraction of Mandal (Subdistrict/County)
            mandal = (
                addr.get("subdistrict") or
                addr.get("county") or
                addr.get("borough") or
                addr.get("city_district") or
                addr.get("town") or
                addr.get("city") or
                addr.get("state") or
                ""
            )
            # Clean up mandal suffix
            for suffix in [" mandal", " Mandal", " subdistrict", " Subdistrict", " tahsil", " Tahsil", " taluk", " Taluk"]:
                if mandal.endswith(suffix):
                    mandal = mandal[:-len(suffix)].strip()
                    break
            
            # Robust extraction of District
            district = (
                addr.get("state_district") or
                addr.get("district") or
                addr.get("county") or
                addr.get("city") or
                addr.get("state") or
                addr.get("country") or
                ""
            )
            # Clean up district suffix
            for suffix in [" district", " District"]:
                if district.endswith(suffix):
                    district = district[:-len(suffix)].strip()
                    break
            
            state = addr.get("state") or ""
            postcode = addr.get("postcode") or ""
            # Clean pincode to keep only digits (usually 6 digits for India)
            clean_pincode = "".join([c for c in postcode if c.isdigit()])

            res = {
                "village": village.strip().title(),
                "mandal": mandal.strip().title(),
                "district": district.strip().title(),
                "state": state.strip().title(),
                "pincode": clean_pincode if len(clean_pincode) == 6 else ""
            }
            print(f"[Reverse Geocode Success] ({lat}, {lon}) -> {res}")
            return res
        else:
            print(f"[Reverse Geocode Failed] ({lat}, {lon}) -> HTTP {r.status_code}: {r.text}")
    except Exception as e:
        print(f"[Reverse Geocode Error] ({lat}, {lon}) -> {e}")
    return None


@st.cache_data
def fetch_pincode_data(pincode):
    """Fetch location details (Post offices, Blocks, District, State) for an Indian pincode."""
    if not pincode or len(pincode) != 6 or not pincode.isdigit():
        return None
    url = f"https://api.postalpincode.in/pincode/{pincode}"
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            data = r.json()
            if data and isinstance(data, list) and data[0].get("Status") == "Success":
                post_offices = data[0].get("PostOffice", [])
                if post_offices:
                    # Collect all unique blocks (mandals) and names (villages)
                    villages = sorted(list(set(po.get("Name", "").strip() for po in post_offices if po.get("Name"))))
                    mandals = sorted(list(set(po.get("Block", "").strip() for po in post_offices if po.get("Block"))))
                    # Extract district and state from the first post office
                    district = post_offices[0].get("District", "").strip()
                    state = post_offices[0].get("State", "").strip()
                    return {
                        "villages": villages,
                        "mandals": mandals,
                        "district": district,
                        "state": state
                    }
    except Exception as e:
        print(f"Error fetching pincode data: {e}")
    return None


def update_location_by_pincode(pincode):
    """Fetch details for pincode and update selectbox states."""
    if not pincode or len(pincode) != 6 or not pincode.isdigit():
        return
    
    # Check if we already fetched for this pincode to avoid redundant API calls
    if st.session_state.get("last_fetched_pincode") == pincode:
        return
        
    st.session_state["last_fetched_pincode"] = pincode
    
    pincode_data = fetch_pincode_data(pincode)
    if pincode_data:
        st.session_state["pincode_mandals"] = pincode_data["mandals"]
        st.session_state["pincode_villages"] = pincode_data["villages"]
        
        # Match state
        matched_state = find_best_state_match(pincode_data["state"], states_list)
        if matched_state:
            st.session_state["selected_state"] = matched_state
            # Match district within that state
            state_districts = state_districts_map.get(matched_state, [])
            matched_district = find_best_district_match(pincode_data["district"], state_districts)
            if matched_district:
                st.session_state["selected_district"] = matched_district
            else:
                st.session_state["selected_district"] = "Other"
                st.session_state["district_input"] = pincode_data["district"]
        else:
            st.session_state["selected_state"] = "Other"
            st.session_state["state_input"] = pincode_data["state"]
            st.session_state["selected_district"] = "Other"
            st.session_state["district_input"] = pincode_data["district"]
        
        # Clear selected mandal and village so user selects from new options
        st.session_state["selected_mandal"] = "Select"
        st.session_state["selected_village"] = "Select"
        st.session_state["mandal_input"] = ""
        st.session_state["village_input"] = ""


@st.cache_data
def _load_csv_data():
    """Load reference CSVs once and cache for the app lifetime."""
    vets      = pd.read_csv(os.path.join(BASE_DIR, "veternary_doctors.csv"))
    mandals   = pd.read_csv(os.path.join(BASE_DIR, "gopalamitra.csv"))
    districts = pd.read_csv(os.path.join(BASE_DIR, "districts.csv"))
    return vets, mandals, districts


df_villages, df_mandals, df_districts = _load_csv_data()

# Clean & deduplicate in one step
villages_list = ["Select"] + sorted(
    {v.strip().title() for v in df_villages["Place of working"].dropna().astype(str)}
)

mandals_list = ["Select"] + sorted(
    {m.strip().title() for m in df_mandals["mandal"].dropna().astype(str)}
)

districts_list = ["Select"] + sorted(
    {d.strip().title() for d in df_districts["District"].dropna().astype(str)}
)

@st.cache_data
def _load_states_data():
    """Load states and districts JSON."""
    filepath = os.path.join(BASE_DIR, "indian_states_districts.json")
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("states", [])
        except Exception as e:
            print(f"Error loading states JSON: {e}")
    return []

states_data = _load_states_data()
states_list = sorted([s["state"] for s in states_data])
state_districts_map = {s["state"]: sorted(s["districts"]) for s in states_data}

# Fallback in case of load failure
if not states_list:
    states_list = ["Telangana"]
    state_districts_map = {"Telangana": ["Peddapalli"]}

def find_best_state_match(g_state, states_list):
    if not g_state:
        return None
    g_state_clean = g_state.strip().lower()
    for s in states_list:
        if s.lower() == g_state_clean:
            return s
    for s in states_list:
        if s.lower() in g_state_clean or g_state_clean in s.lower():
            return s
    return None

def find_best_district_match(g_district, districts_list):
    if not g_district:
        return None
    g_district_clean = g_district.strip().lower()
    # Clean up standard suffixes before comparison
    for suffix in [" district", " District"]:
        if g_district_clean.endswith(suffix):
            g_district_clean = g_district_clean[:-len(suffix)].strip()
            break
    for d in districts_list:
        d_clean = d.lower()
        for suffix in [" district", " District"]:
            if d_clean.endswith(suffix):
                d_clean = d_clean[:-len(suffix)].strip()
                break
        if d_clean == g_district_clean:
            return d
    for d in districts_list:
        d_clean = d.lower()
        for suffix in [" district", " District"]:
            if d_clean.endswith(suffix):
                d_clean = d_clean[:-len(suffix)].strip()
                break
        if d_clean in g_district_clean or g_district_clean in d_clean:
            return d
    return None



if st.session_state.step == "user_info":
    load_custom_css()

    st.markdown("""
        <div class="chmi-hero">
            <div class="hero-badge">🐄 AI-Powered · Livestock Health</div>
            <h1 class="hero-title">Cattle Health Monitor</h1>
            <p class="hero-subtitle">Detect Foot & Mouth Disease and Lumpy Skin Disease instantly using a 5-model AI ensemble trained on field images.</p>
            <div class="hero-stats">
                <div class="hero-stat">
                    <span class="hero-stat-num">5</span>
                    <span class="hero-stat-label">AI Models</span>
                </div>
                <div class="hero-stat">
                    <span class="hero-stat-num">2</span>
                    <span class="hero-stat-label">Disease Types</span>
                </div>
                <div class="hero-stat">
                    <span class="hero-stat-num">FMD+LSD</span>
                    <span class="hero-stat-label">Detection</span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    render_stepper("user_info")

    st.markdown("""
        <div class="section-label">
            <div class="section-label-icon">👤</div>
            <div>
                <p class="section-label-text">User Registration</p>
                <p class="section-label-sub">Enter your contact and location details</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    with st.container():

        # Row 1: Name & Mobile
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Name", value=st.session_state.user_details.get("name", ""))
        with col2:
            mobile = st.text_input("Mobile Number (10 digits)", value=st.session_state.user_details.get("mobile", ""))

        # Initialize non-widget session state keys to prevent StreamlitAPIException
        safe_keys = {
            "selected_state": "Select",
            "selected_district": "Select",
            "selected_mandal": "Select",
            "selected_village": "Select",
            "pincode_val": "",
            "pincode_mandals": [],
            "pincode_villages": [],
            "gps_mandal": "",
            "gps_village": "",
            "state_input": "",
            "district_input": "",
            "mandal_input": "",
            "village_input": ""
        }
        for k, v in safe_keys.items():
            if k not in st.session_state:
                st.session_state[k] = v

        # Create a container where the selectboxes will be drawn.
        # This allows us to defer rendering selectboxes until after the GPS geocoding check!
        selectbox_row_container = st.container()

    st.markdown("""
        <div class="section-label">
            <div class="section-label-icon">📍</div>
            <div>
                <p class="section-label-text">GPS Location</p>
                <p class="section-label-sub">Helps track disease outbreaks across regions</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    location = streamlit_geolocation()

    lat = lon = None

    if location:
        lat = location.get("latitude")
        lon = location.get("longitude")
        st.session_state.location = {"lat": lat, "lon": lon}

        # Auto-populate village, mandal, district based on GPS coordinates if not already resolved for these coords
        if lat is not None and lon is not None:
            # Round coordinates to 4 decimal places to prevent rate limit blocks due to GPS drift
            lat_rounded = round(lat, 4)
            lon_rounded = round(lon, 4)
            last_coords = st.session_state.get("last_resolved_coords", None)
            
            if last_coords is None or last_coords != (lat_rounded, lon_rounded):
                # Set coordinates immediately to prevent duplicate requests on failures/reruns
                st.session_state.last_resolved_coords = (lat_rounded, lon_rounded)
                geocoded = reverse_geocode(lat_rounded, lon_rounded)
                if geocoded:
                    # Treat empty values as "Unknown" to ensure they fallback to "Other" -> "Unknown"
                    g_village = geocoded.get("village", "").strip() or "Unknown"
                    g_mandal = geocoded.get("mandal", "").strip() or "Unknown"
                    g_district = geocoded.get("district", "").strip() or "Unknown"
                    g_state = geocoded.get("state", "").strip() or "Unknown"
                    g_pincode = geocoded.get("pincode", "").strip()
                    
                    # Store GPS details
                    st.session_state["gps_village"] = g_village
                    st.session_state["gps_mandal"] = g_mandal
                    
                    if g_pincode and len(g_pincode) == 6 and g_pincode.isdigit():
                        st.session_state["pincode_val"] = g_pincode
                        update_location_by_pincode(g_pincode)
                    else:
                        # Fallback to direct geocoding matches
                        matched_state = find_best_state_match(g_state, states_list)
                        if matched_state:
                            st.session_state["selected_state"] = matched_state
                            state_districts = state_districts_map.get(matched_state, [])
                            matched_district = find_best_district_match(g_district, state_districts)
                            if matched_district:
                                st.session_state["selected_district"] = matched_district
                            else:
                                st.session_state["selected_district"] = "Other"
                                st.session_state["district_input"] = g_district
                        else:
                            st.session_state["selected_state"] = "Other"
                            st.session_state["state_input"] = g_state
                            st.session_state["selected_district"] = "Other"
                            st.session_state["district_input"] = g_district
                            
                    # Auto select geocoded village/mandal if available
                    if g_mandal and g_mandal != "Unknown":
                        st.session_state["selected_mandal"] = g_mandal
                    if g_village and g_village != "Unknown":
                        st.session_state["selected_village"] = g_village

    # Now render the selectbox row (which will appear visually above the GPS section)
    with selectbox_row_container:
        # Row 2: Pincode and State
        col_pincode, col_state = st.columns(2)
        
        with col_pincode:
            pincode_input = st.text_input(
                "Pincode (6 digits)",
                value=st.session_state.get("pincode_val", ""),
                max_chars=6,
                placeholder="e.g. 505172"
            )
            # Handle pincode manual input change
            if pincode_input != st.session_state.get("pincode_val", ""):
                st.session_state["pincode_val"] = pincode_input
                if len(pincode_input) == 6 and pincode_input.isdigit():
                    update_location_by_pincode(pincode_input)
                    
        with col_state:
            states_options = ["Select"] + states_list + ["Other"]
            curr_state = st.session_state.get("selected_state", "Select")
            if curr_state not in states_options:
                curr_state = "Select"
            state_index = states_options.index(curr_state)
            
            state_selection = st.selectbox(
                "State",
                options=states_options,
                index=state_index
            )
            st.session_state["selected_state"] = state_selection
            
            if state_selection == "Other":
                custom_state = st.text_input(
                    "Enter State",
                    value=st.session_state.get("state_input", "")
                )
                st.session_state["state_input"] = custom_state
                state = custom_state
            else:
                state = state_selection
                st.session_state["state_input"] = ""
                
        # Row 3: District and Mandal
        col_district, col_mandal = st.columns(2)
        
        with col_district:
            # Dynamically determine the districts based on the selected state
            if state_selection != "Select" and state_selection != "Other":
                dist_list = state_districts_map.get(state_selection, [])
                districts_options = ["Select"] + dist_list + ["Other"]
            elif state_selection == "Other":
                districts_options = ["Select", "Other"]
            else:
                districts_options = ["Select"]
                
            curr_district = st.session_state.get("selected_district", "Select")
            if curr_district not in districts_options:
                curr_district = "Select"
            district_index = districts_options.index(curr_district)
            
            district_selection = st.selectbox(
                "District",
                options=districts_options,
                index=district_index
            )
            st.session_state["selected_district"] = district_selection
            
            if district_selection == "Other":
                custom_district = st.text_input(
                    "Enter District",
                    value=st.session_state.get("district_input", "")
                )
                st.session_state["district_input"] = custom_district
                district = custom_district
            else:
                district = district_selection
                st.session_state["district_input"] = ""
                
        with col_mandal:
            # Mandal options: resolve from Pincode lookup, GPS geocoding, or Other
            mandal_options = ["Select"]
            if st.session_state.get("pincode_mandals"):
                mandal_options.extend(st.session_state["pincode_mandals"])
            gps_mandal = st.session_state.get("gps_mandal", "")
            if gps_mandal and gps_mandal not in mandal_options and gps_mandal not in ["Select", "Other", "Unknown"]:
                mandal_options.append(gps_mandal)
            mandal_options.append("Other")
            
            seen_mandal = set()
            mandal_options = [x for x in mandal_options if not (x in seen_mandal or seen_mandal.add(x))]
            
            curr_mandal = st.session_state.get("selected_mandal", "Select")
            if curr_mandal not in mandal_options:
                curr_mandal = "Select"
            mandal_index = mandal_options.index(curr_mandal)
            
            mandal_selection = st.selectbox(
                "Mandal",
                options=mandal_options,
                index=mandal_index
            )
            st.session_state["selected_mandal"] = mandal_selection
            
            if mandal_selection == "Other":
                custom_mandal = st.text_input(
                    "Enter Mandal",
                    value=st.session_state.get("mandal_input", "")
                )
                st.session_state["mandal_input"] = custom_mandal
                mandal = custom_mandal
            else:
                mandal = mandal_selection
                st.session_state["mandal_input"] = ""
                
        # Row 4: Village
        col_village, col_empty = st.columns(2)
        with col_village:
            # Village options: resolve from Pincode lookup, GPS geocoding, or Other
            village_options = ["Select"]
            if st.session_state.get("pincode_villages"):
                village_options.extend(st.session_state["pincode_villages"])
            gps_village = st.session_state.get("gps_village", "")
            if gps_village and gps_village not in village_options and gps_village not in ["Select", "Other", "Unknown"]:
                village_options.append(gps_village)
            village_options.append("Other")
            
            seen_village = set()
            village_options = [x for x in village_options if not (x in seen_village or seen_village.add(x))]
            
            curr_village = st.session_state.get("selected_village", "Select")
            if curr_village not in village_options:
                curr_village = "Select"
            village_index = village_options.index(curr_village)
            
            village_selection = st.selectbox(
                "Village",
                options=village_options,
                index=village_index
            )
            st.session_state["selected_village"] = village_selection
            
            if village_selection == "Other":
                custom_village = st.text_input(
                    "Enter Village",
                    key="village_input"
                )
                village = custom_village
            else:
                village = village_selection
                st.session_state["village_input"] = ""

    if lat is not None and lon is not None:
        col_lat, col_lon = st.columns(2)
        with col_lat:
            st.text_input("Latitude", value=str(lat), disabled=True)
        with col_lon:
            st.text_input("Longitude", value=str(lon), disabled=True)
    else:
        st.info("Please allow location access to capture coordinates.")

    # Submit button outside form (instant updates now)
    submitted = st.button("Submit & Send OTP", use_container_width=True, type='primary')



    if submitted:
        # Validation
        errors = []
        if not name.strip():
            errors.append("Name is required.")
        if not state.strip():
            errors.append("State is required.")
        if not district.strip():
            errors.append("District is required.")
        if not mandal.strip():
            errors.append("Mandal is required.")
        if not village.strip():
            errors.append("Village is required.")
        if not mobile.strip() or not is_valid_mobile(mobile.strip()):
            errors.append("Valid 10-digit Mobile Number is required.")
        # if lat is None or lon is None:
        #     errors.append("Location not captured. Please allow location access.")
        if state == "Select" or district == "Select" or mandal == "Select" or village == "Select":
            errors.append("Please select a valid State, District, Mandal, and Village.")
        if errors:
            for err in errors:
                st.error(err)
        else:
            st.session_state.user_details = {
                "name": html.escape(name.strip()),
                "state": html.escape(state.strip()),
                "district": html.escape(district.strip()),
                "mandal": html.escape(mandal.strip()),
                "village": html.escape(village.strip()),
                "pincode": html.escape(st.session_state.get("pincode_val", "").strip()),
                "mobile": mobile.strip()
            }
            st.session_state.location = {"lat": lat, "lon": lon}

            now = time.time()
            if (now - st.session_state.otp_sent_at) < 15:
                st.info(f"Please wait {int(15 - (now - st.session_state.otp_sent_at))} seconds before requesting another OTP.")
            else:
                otp = generate_otp()
                success, msg = send_otp_sms(mobile.strip(), otp)
                if success:
                    st.session_state.otp_code = otp
                    st.session_state.otp_sent_at = now
                    st.session_state.disable_otp_request_until = now + 15
                    st.session_state.otp_attempts = 0
                    st.session_state.step = "otp_verify"
                    st.success(msg)
                    st.info("OTP sent successfully. Please check your phone.")
                    st.rerun()
                else:
                    st.error(msg)



elif st.session_state.step == "otp_verify":
    load_custom_css()
    render_stepper("otp_verify")

    _, otp_col, _ = st.columns([1, 2, 1])
    with otp_col:
        st.markdown(f"""
            <div style="text-align:center; padding: 1.5rem 1rem 1rem;">
                <div style="width:72px;height:72px;background:linear-gradient(135deg,#059669,#047857);
                            border-radius:20px;display:flex;align-items:center;justify-content:center;
                            font-size:2rem;margin:0 auto 1.25rem auto;
                            box-shadow:0 8px 24px rgba(5,150,105,0.35);">📱</div>
                <h2 style="margin:0 0 0.4rem;color:#0f172a;font-weight:800;font-size:1.6rem;letter-spacing:-0.02rem;">
                    Verify Your Number
                </h2>
                <p style="color:#64748b;margin:0 0 1.25rem;font-size:0.9rem;">
                    A 6-digit code was sent to
                </p>
                <div style="background:#f0fdf4;border:1.5px solid #a7f3d0;border-radius:12px;
                            padding:0.75rem 1.5rem;display:inline-flex;align-items:center;
                            gap:0.5rem;margin-bottom:1.5rem;">
                    <span style="font-size:1.1rem;">📞</span>
                    <span style="font-size:1.1rem;font-weight:700;color:#047857;letter-spacing:0.05rem;">
                        +91 {st.session_state.user_details.get('mobile','')}
                    </span>
                </div>
            </div>
        """, unsafe_allow_html=True)

        if IS_DEV_MODE:
            st.markdown(f"""
                <div class="alert-card alert-card--warning" style="margin-bottom: 1.25rem;">
                    <div class="alert-icon">🔧</div>
                    <div>
                        <p class="alert-title alert-title--warning" style="margin: 0 0 0.2rem 0; font-weight: 700;">Developer Mode Active</p>
                        <p class="alert-body alert-body--warning" style="margin: 0;">SMS credentials not configured. Your simulated OTP is: <strong style="font-size: 1.05rem; color: #b45309;">{st.session_state.otp_code}</strong> (pre-filled below).</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        otp_input = st.text_input(
            "6-digit OTP",
            value=st.session_state.otp_code if IS_DEV_MODE else "",
            key="otp_input",
            max_chars=6,
            placeholder="Enter the code sent to your phone",
        )

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            verify_clicked = st.button("Verify & Continue →", use_container_width=True, type='primary')
        with btn_col2:
            resend_clicked = st.button("Resend Code", use_container_width=True)

        attempts_left = 5 - st.session_state.get('otp_attempts', 0)
        if attempts_left < 5:
            st.markdown(f"""
                <div class="alert-card alert-card--warning" style="margin-top:0.75rem;">
                    <div class="alert-icon">⚠️</div>
                    <div>
                        <p class="alert-title alert-title--warning" style="margin:0;">
                            {attempts_left} attempt(s) remaining
                        </p>
                    </div>
                </div>
            """, unsafe_allow_html=True)

    # Logic: Verify OTP
    if verify_clicked:
        if otp_input.strip() == "":
            st.warning("Please enter the OTP.")
        elif st.session_state.get("otp_attempts", 0) >= 5:
            st.error("Too many failed attempts. Please request a new OTP.")
        elif otp_input.strip() == st.session_state.otp_code:
            st.session_state.otp_verified = True
            st.session_state.otp_code = ""   # Invalidate OTP after use
            st.session_state.otp_attempts = 0
            st.success("OTP verified successfully.")
            st.session_state.step = "dashboard"
            st.rerun()
        else:
            st.session_state.otp_attempts = st.session_state.get("otp_attempts", 0) + 1
            remaining = 5 - st.session_state.otp_attempts
            st.error(f"Invalid OTP. {remaining} attempt(s) remaining.")

    # Logic: Resend OTP
    if resend_clicked:
        now = time.time()
        remaining = int(st.session_state.disable_otp_request_until - now)

        if remaining > 0:
            st.warning(f"Please wait {remaining}s before resending OTP.")
        else:
            otp = generate_otp()
            success, msg = send_otp_sms(st.session_state.user_details["mobile"], otp)
            if success:
                st.session_state.otp_code = otp
                st.session_state.otp_sent_at = now
                st.session_state.disable_otp_request_until = now + 15
                st.success(msg)
                if IS_DEV_MODE:
                    st.info(f"🔧 **Developer Mode**: Simulated OTP is: **`{otp}`**")
                else:
                    st.info("Please ensure you have a good signal and location access is enabled.")
            else:
                st.error(msg)



elif st.session_state.step == "dashboard":

    load_custom_css()
    render_stepper("dashboard")

    st.markdown("""
        <div class="chmi-hero" style="padding:1.75rem 2.5rem 1.5rem;">
            <div class="hero-badge">✅ Authenticated · Ready to Diagnose</div>
            <h1 class="hero-title" style="font-size:1.75rem;">Disease Detection Module</h1>
            <p class="hero-subtitle" style="margin-bottom:0;">Upload a cattle image to run AI-powered FMD or LSD detection.</p>
        </div>
    """, unsafe_allow_html=True)

    user = st.session_state.user_details
    loc = st.session_state.location

    # User and Location Information Cards
    col1, col2 = st.columns([2,1], gap="medium")
    
    with col1:
        st.markdown(f"""
            <div class="info-card">
                <div class="info-card-label">
                    <div class="info-card-label-dot"></div> User Information
                </div>
                <div class="info-row"><span class="info-row-key">Name</span><span class="info-row-val">{user['name']}</span></div>
                <div class="info-row"><span class="info-row-key">Mobile</span><span class="info-row-val">+91 {user['mobile']}</span></div>
                <div class="info-row"><span class="info-row-key">Village</span><span class="info-row-val">{user['village']}</span></div>
                <div class="info-row"><span class="info-row-key">Mandal</span><span class="info-row-val">{user['mandal']}</span></div>
                <div class="info-row"><span class="info-row-key">District</span><span class="info-row-val">{user['district']}</span></div>
                <div class="info-row"><span class="info-row-key">State</span><span class="info-row-val">{user.get('state', 'N/A')}</span></div>
                <div class="info-row"><span class="info-row-key">Pincode</span><span class="info-row-val">{user.get('pincode', 'N/A')}</span></div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        _lat_str = f"{loc['lat']:.5f}" if loc.get('lat') else "—"
        _lon_str = f"{loc['lon']:.5f}" if loc.get('lon') else "—"
        st.markdown(f"""
            <div class="info-card">
                <div class="info-card-label">
                    <div class="info-card-label-dot"></div> GPS Location
                </div>
                <div class="info-row"><span class="info-row-key">Latitude</span><span class="info-row-val">{_lat_str}</span></div>
                <div class="info-row"><span class="info-row-key">Longitude</span><span class="info-row-val">{_lon_str}</span></div>
            </div>
        """, unsafe_allow_html=True)

        if loc.get("lat") and loc.get("lon"):
            try:
                st.map(pd.DataFrame({"lat": [float(loc["lat"])], "lon": [float(loc["lon"])]}), zoom=11)
            except Exception:
                pass

    st.markdown("""
        <div class="section-label">
            <div class="section-label-icon">🐮</div>
            <div>
                <p class="section-label-text">Cattle Information</p>
                <p class="section-label-sub">Provide the tag, age, and sex of the animal</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.get("cattle_details_submitted"):
        with st.form("cattle_form", clear_on_submit=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                cattle_id = st.text_input("Cattle ID", value=st.session_state.get("cattle_id", ""))
            with col2:
                gender = st.selectbox("Gender", ["Male", "Female"], index=0 if st.session_state.get("gender", "Male") == "Male" else 1)
            with col3:
                age = st.number_input("Age (in years)", min_value=0.0, max_value=30.0, step=0.1, format="%.1f", value=float(st.session_state.get("age", 0.0)))
            submit = st.form_submit_button("Submit", use_container_width=True, type="primary")

        if submit:
            if cattle_id.strip() and gender and age >= 0:
                st.session_state["cattle_details_submitted"] = True
                st.session_state["cattle_id"] = cattle_id.strip()
                st.session_state["gender"] = gender
                st.session_state["age"] = age
                st.rerun()
            else:
                st.warning("Please fill all details before submitting.")
    else:
        cattle_id = st.session_state.get("cattle_id")
        gender = st.session_state.get("gender")
        age = st.session_state.get("age")

        # Beautiful read-only summary card
        col_summary, col_edit = st.columns([3.2, 1], gap="small")
        with col_summary:
            st.markdown(f"""
                <div class="info-card" style="border-left: 5px solid #059669; padding: 0.8rem 1.25rem; margin-bottom: 0; display: flex; align-items: center; gap: 1rem;">
                    <div style="font-size: 1.8rem;">🐮</div>
                    <div>
                        <div style="font-weight: 800; font-size: 1rem; color: #0f172a; line-height: 1.25 !important;">Cattle ID: #{cattle_id}</div>
                        <div style="font-size: 0.82rem; color: #64748b; font-weight: 500;">{gender} &nbsp;&middot;&nbsp; {age} Years Old</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        with col_edit:
            st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)
            if st.button("✏️ Edit Details", use_container_width=True, key="edit_cattle"):
                st.session_state["cattle_details_submitted"] = False
                st.rerun()

    if st.session_state.get("cattle_details_submitted"):
        cattle_id = st.session_state.get("cattle_id")
        gender = st.session_state.get("gender")
        age = st.session_state.get("age")
        if cattle_id and gender and age is not None:
            
            # Add sub-steps for dashboard (styled checklist)
            if current_step == "dashboard":
                _has_image = bool(st.session_state.get("upload_file"))
                _checklist_items = [
                    ("Disease Type", True, "✓", "#059669", "#0f2a1f"),
                    ("Image Upload", _has_image, "✓" if _has_image else "○", "#059669" if _has_image else "#475569", "#0f2a1f" if _has_image else "#1e293b"),
                ]
                _checklist_html = '<div style="border-top:1px solid #1e293b;margin:0.75rem 0;"></div>'
                _checklist_html += '<div style="font-size:0.68rem;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.1rem;margin-bottom:0.5rem;">Diagnosis Progress</div>'
                for _cl_label, _cl_done, _cl_icon, _cl_color, _cl_bg in _checklist_items:
                    _cl_txt_color = "#94a3b8" if _cl_done else "#475569"
                    _checklist_html += f'''
                        <div style="display:flex;align-items:center;gap:0.6rem;padding:0.3rem 0;">
                            <div style="width:22px;height:22px;background:{_cl_bg};border:1.5px solid {_cl_color};border-radius:6px;
                                        display:flex;align-items:center;justify-content:center;
                                        font-size:0.65rem;font-weight:700;color:{_cl_color};flex-shrink:0;">{_cl_icon}</div>
                            <span style="font-size:0.8rem;color:{_cl_txt_color};font-weight:500;">{_cl_label}</span>
                        </div>
                    '''
                st.sidebar.markdown(_checklist_html, unsafe_allow_html=True)
            
            st.session_state.cattle_details=True

            st.markdown("""
                <div class="section-label">
                    <div class="section-label-icon">🔬</div>
                    <div>
                        <p class="section-label-text">Disease Detection</p>
                        <p class="section-label-sub">Select the disease type and upload a clear image</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            st.markdown("""
                <div class="action-card">
                    <div class="action-card-icon action-card-icon--green">🦠</div>
                    <div class="action-card-body">
                        <p class="action-card-title">Select Disease Type</p>
                        <p class="action-card-sub">Choose the condition you suspect</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            disease_type = st.radio(
                    label="Select Disease Type",
                    options=["LSD - Lumpy Skin Disease", "FMD - Foot and Mouth Disease"],
                    index=0,
                    key="disease_radio",
                    help="Select the type of disease you want to detect",
                    horizontal=True,
                    label_visibility="collapsed"
                )
           
            _guide_en, _guide_te = st.tabs(["📋 Guidelines (English)", "📋 మార్గదర్శకాలు (తెలుగు)"])
            with _guide_en:
                st.markdown("""
                <ol style="text-align: justify; font-size: 15px; line-height: 1.7; margin-top: 0.5rem;">
                    <li>Ensure a stable internet connection on your device (mobile, tablet, or computer) before using the app.</li>
                    <li>Upload <strong>cow images only</strong>. The system filters irrelevant images, but accuracy depends on image relevance.</li>
                    <li>Each model is trained for a specific disease — select the correct disease type for accurate results.</li>
                    <li>Capture a clear image showing the relevant area (mouth, knuckles, or skin). Blurry or partial images may reduce accuracy.</li>
                </ol>
                """, unsafe_allow_html=True)
            with _guide_te:
                st.markdown("""
                <ol style="text-align: justify; font-size: 15px; line-height: 1.7; margin-top: 0.5rem;">
                    <li>యాప్ వాడేముందు మీ పరికరంలో (మొబైల్, టాబ్లెట్ లేదా కంప్యూటర్) స్థిరమైన ఇంటర్నెట్ కనెక్షన్ ఉందని నిర్ధారించుకోండి.</li>
                    <li>దయచేసి <strong>ఆవు చిత్రాలను మాత్రమే</strong> అప్‌లోడ్ చేయండి. సిస్టమ్ అసంబంధమైన చిత్రాలను వడపోత చేస్తుంది, కానీ ఖచ్చితత్వం చిత్రం నాణ్యతపై ఆధారపడుతుంది.</li>
                    <li>ప్రతి మోడల్ ఒక నిర్దిష్ట వ్యాధి కోసం శిక్షణ పొందింది — ఖచ్చితమైన ఫలితాలు పొందడానికి సరైన వ్యాధి రకాన్ని ఎంచుకోండి.</li>
                    <li>సంబంధిత ప్రాంతం (నోరు, కీళ్ళు లేదా చర్మం) స్పష్టంగా కనిపించేలా చిత్రం తీయండి. అస్పష్టమైన చిత్రాలు ఖచ్చితత్వాన్ని తగ్గించవచ్చు.</li>
                </ol>
                """, unsafe_allow_html=True)


            st.markdown("""
                <p style="text-align:center;color:#64748b;font-size:0.85rem;margin:0.5rem 0 0.75rem;">
                    Reference samples — ensure your image shows a similar view of the affected area
                </p>
            """, unsafe_allow_html=True)
            if "LSD" in disease_type:
                selected_images = lsd_images
            else:
                selected_images = fmd_images

            # Display 3 images in one row
            cols = st.columns(3)
            for i in range(3):
                with cols[i]:
                    try:
                        img = Image.open(selected_images[i]).resize((500, 350))
                        st.image(img)
                    except FileNotFoundError:
                        st.info("Sample image not available.")
            # Main Action Area with 2 columns
            st.markdown("<br>", unsafe_allow_html=True)

            col_left, col_right = st.columns([1.5, 1.5], gap="large")

            with col_left:
                st.markdown("""
                    <div class="action-card">
                        <div class="action-card-icon action-card-icon--green">📷</div>
                        <div class="action-card-body">
                            <p class="action-card-title">Upload Cattle Image</p>
                            <p class="action-card-sub">PNG, JPG, JPEG · Max 200 MB</p>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                uploaded_file = st.file_uploader(
                    "Cattle Image Upload",
                    type=["png", "jpg", "jpeg"],
                    key="upload_file",
                    help="Supported formats: PNG, JPG, JPEG (Max: 200MB)",
                    label_visibility="collapsed"
                )

            with col_right:
                st.markdown("""
                    <div class="action-card">
                        <div class="action-card-icon action-card-icon--blue">🖼️</div>
                        <div class="action-card-body">
                            <p class="action-card-title">Image Preview</p>
                            <p class="action-card-sub">Your uploaded image appears here</p>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

                if uploaded_file:
                    st.image(uploaded_file, caption="Ready for Analysis", use_container_width=True)
                    file_size = uploaded_file.size / (1024 * 1024)
                    st.markdown(f"""
                        <div class="alert-card alert-card--success" style="padding:0.6rem 1rem;margin-top:0.5rem;">
                            <div class="alert-icon" style="font-size:1rem;">✅</div>
                            <p class="alert-body alert-body--success" style="margin:0;">
                                Image loaded — {file_size:.2f} MB · Ready for analysis
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                        <div class="preview-area">
                            <svg width="40" height="40" fill="none" stroke="#cbd5e1" stroke-width="1.5" viewBox="0 0 24 24" style="margin-bottom:0.75rem;">
                                <rect x="3" y="3" width="18" height="18" rx="3"/>
                                <circle cx="8.5" cy="8.5" r="1.5"/>
                                <path d="M21 15l-5-5L5 21"/>
                            </svg>
                            <p style="margin:0;font-size:0.85rem;color:#94a3b8;">Upload an image to preview it here</p>
                        </div>
                    """, unsafe_allow_html=True)

            # Action Buttons Section
            st.markdown("<br>", unsafe_allow_html=True)
            
            if st.session_state.get("upload_file") and st.session_state.get("disease_radio"):
                st.sidebar.markdown('<div style="display:flex;align-items:center;gap:0.5rem;padding:0.3rem 0;"><div style="width:8px;height:8px;background:#f59e0b;border-radius:50%;flex-shrink:0;box-shadow:0 0 6px rgba(245,158,11,0.5);"></div><span style="font-size:0.78rem;color:#f59e0b;font-weight:600;">Ready for prediction</span></div>', unsafe_allow_html=True)
            else:
                st.sidebar.markdown('<div style="display:flex;align-items:center;gap:0.5rem;padding:0.3rem 0;"><div style="width:8px;height:8px;background:#475569;border-radius:50%;flex-shrink:0;"></div><span style="font-size:0.78rem;color:#475569;font-weight:500;">Awaiting inputs</span></div>', unsafe_allow_html=True)
            
            if uploaded_file and disease_type:

                if st.button("Submit for Prediction", use_container_width=True, type="primary"):
                    with st.spinner("Running prediction..."):
                        try:
                            st.session_state["prediction_done"] = "success"
                            result = analyze_single_image(uploaded_file, interpreter, input_details, output_details, threshold)
                            if result['accepted']:

                                # Load cached ensemble models (no re-loading on every click)
                                is_lsd_demo = False
                                if disease_type.startswith("LSD"):
                                    disease_code = "LSD"
                                    missing_lsd = [p for p in TFLITE_MODELS_LSD.values() if not os.path.exists(p)]
                                    if missing_lsd:
                                        is_lsd_demo = True
                                    else:
                                        interpreters = get_lsd_models()
                                else:
                                    disease_code = "FMD"
                                    interpreters = get_fmd_models()

                                image = preprocess_image(uploaded_file)
                                
                                if is_lsd_demo:
                                    st.warning("⚠️ LSD model files are missing. Running in demo fallback mode.")
                                    probs, predicted_label = 0.85, "Diseased"
                                else:
                                    if disease_type.startswith("LSD"):
                                        probs,predicted_label = soft_voting_ensemble(interpreters, image, MODEL_WEIGHTS,CLASS_NAMES)
                                    else:
                                        probs,predicted_label = soft_voting_ensemble(interpreters, image, MODEL_WEIGHTS,CATEGORIES)
                                
                                confidence = probs * 100
                                disease_status = f"{disease_code} Infected" if predicted_label == "Diseased" else "Healthy"
                                _bar_pct = min(100, max(0, confidence))
                                # Enhanced results display
                                st.markdown("<br>", unsafe_allow_html=True)
                                
                                if confidence > 80:
                                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    extension = uploaded_file.name.split('.')[-1]
                                    image_filename = f"{timestamp}.{extension}"
                                    image_path = os.path.join(UPLOAD_FOLDER, image_filename)
                                    # Write raw bytes — avoids a third PIL decode/encode round-trip
                                    uploaded_file.seek(0)
                                    with open(image_path, 'wb') as _f:
                                        _f.write(uploaded_file.read())
                                    

                                    text_filename = f"{timestamp}.txt"
                                    text_path = os.path.join(TEXT_FOLDER, text_filename)
                                    
                                    user = st.session_state.get("user_details", {})
                                    location = st.session_state.get("location", {})
                                    
                                        
                                    disease_status = f"{disease_code} Infected" if predicted_label == "Diseased" else "Healthy"
                                    _bar_pct = min(100, max(0, confidence))

                                    with open(text_path, "w") as txt_file:
                                        txt_file.write("=== User Details ===\n")
                                        txt_file.write(f"Name     : {user.get('name', 'N/A')}\n")
                                        txt_file.write(f"Mobile   : {user.get('mobile', 'N/A')}\n")
                                        txt_file.write(f"Village  : {user.get('village', 'N/A')}\n")
                                        txt_file.write(f"Mandal   : {user.get('mandal', 'N/A')}\n")
                                        txt_file.write(f"District : {user.get('district', 'N/A')}\n")
                                        txt_file.write(f"State    : {user.get('state', 'N/A')}\n")
                                        txt_file.write(f"Pincode  : {user.get('pincode', 'N/A')}\n")
                                        txt_file.write(f"Latitude : {location.get('lat', 'N/A')}\n")
                                        txt_file.write(f"Longitude: {location.get('lon', 'N/A')}\n\n")

                                        txt_file.write("=== Cattle Details ===\n")
                                        txt_file.write(f"Cattle ID : {cattle_id}\n")
                                        txt_file.write(f"Gender    : {gender}\n")
                                        txt_file.write(f"Age       : {age} years\n\n")

                                        txt_file.write("=== Prediction Result ===\n")
                                        txt_file.write(f"Status: {disease_status}\n\n")
                                        txt_file.write(f"Confidence: {confidence:.2f}%\n\n")

                                        txt_file.write(f"Image File: {image_filename}\n")

                                    # Pre-compute vet/gopa contacts (used for display + PDF)
                                    _vet_sup  = df_villages[df_villages["Place of working"].str.strip().str.lower() == (user.get('village') or "").lower()]
                                    _gopa_sup = df_mandals[df_mandals["mandal"].str.strip().str.lower() == (user.get('mandal') or "").lower()]
                                    _vet_row  = _vet_sup.iloc[0]  if not _vet_sup.empty  else df_villages.iloc[0]
                                    _gopa_row = _gopa_sup.iloc[0] if not _gopa_sup.empty else df_mandals.iloc[0]

                                    # Append to in-session scan history
                                    st.session_state.scan_history.append({
                                        "Time": datetime.now().strftime("%H:%M:%S"),
                                        "Cattle ID": cattle_id,
                                        "Disease": disease_code,
                                        "Status": disease_status,
                                        "Confidence": f"{confidence:.1f}%",
                                    })

                                    # Generate and cache PDF bytes
                                    try:
                                        _pdf_bytes = generate_pdf_report(
                                            user, location, cattle_id, gender, age,
                                            disease_type, disease_status, confidence,
                                            vet_first=_vet_row.to_dict() if predicted_label == "Diseased" else None,
                                            gopa_first=_gopa_row.to_dict() if predicted_label == "Diseased" else None,
                                        )
                                        st.session_state.pdf_report_bytes = _pdf_bytes
                                        st.session_state.pdf_report_filename = f"CHMI_{cattle_id}_{timestamp}.pdf"
                                    except Exception:
                                        st.session_state.pdf_report_bytes = None

                                    # Create WhatsApp Share URL
                                    import urllib.parse
                                    _share_msg = (
                                        f"🚨 *CHMI Cattle Health Diagnostics Report*\n\n"
                                        f"🐄 *Cattle ID:* {cattle_id}\n"
                                        f"📌 *Status:* {disease_status}\n"
                                        f"📈 *Confidence:* {confidence:.1f}%\n"
                                        f"👤 *Farmer:* {user.get('name', 'N/A')} (+91 {user.get('mobile', 'N/A')})\n"
                                        f"📍 *Location:* {user.get('village', 'N/A')}, {user.get('mandal', 'N/A')}, {user.get('district', 'N/A')}, {user.get('state', 'N/A')} - {user.get('pincode', 'N/A')}\n"
                                    )
                                    if location.get('lat') and location.get('lon'):
                                        _share_msg += f"🗺️ *GPS Coordinates:* {location['lat']:.6f}, {location['lon']:.6f}\n"
                                    _share_msg += f"\nGenerated via CHMI AI on {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                                    whatsapp_url = f"https://wa.me/?text={urllib.parse.quote(_share_msg)}"

                                    if predicted_label == "Diseased":
                                        left_content_col, right_content_col = st.columns(2, gap="medium")

                                        with left_content_col:
                                            st.markdown(f"""
                                                <div class="result-card result-card--diseased">
                                                    <span class="result-status-badge result-status-badge--diseased">&#9679; Disease Detected</span>
                                                    <div class="result-icon">&#129440;</div>
                                                    <h2 class="result-status-text result-status-text--diseased">{disease_status}</h2>
                                                    <p style="color:#64748b;font-size:0.85rem;margin:0.25rem 0 1rem;">Analysis: {disease_type}</p>
                                                    <div class="confidence-bar-track">
                                                        <div class="confidence-bar-fill confidence-bar-fill--diseased" style="width:{_bar_pct}%"></div>
                                                    </div>
                                                    <p class="confidence-pct confidence-pct--diseased">{confidence:.1f}% confidence</p>
                                                </div>
                                            """, unsafe_allow_html=True)

                                        with right_content_col:
                                            vet_first  = _vet_row
                                            gopa_first = _gopa_row
                                            _vet_name   = vet_first["Name"]  if vet_first  is not None else "N/A"
                                            _vet_place  = vet_first["Place of working"] if vet_first is not None else ""
                                            _vet_phone  = vet_first["Mobile no."] if vet_first is not None else ""
                                            _gopa_name  = gopa_first["Name"] if gopa_first is not None else "N/A"
                                            _gopa_mandal = gopa_first["mandal"] if gopa_first is not None else ""
                                            _gopa_phone  = gopa_first["mobile no."] if gopa_first is not None else ""
                                            
                                            st.markdown(f"""
                                                <div class="alert-card alert-card--warning" style="margin-bottom:0.75rem; padding: 0.8rem 1rem;">
                                                    <strong>🩺 Seek veterinary support immediately</strong>
                                                </div>
                                                <div class="vet-card">
                                                    <div class="vet-avatar" style="background:#dbeafe; font-size:1.3rem;">🩺</div>
                                                    <div class="vet-info">
                                                        <div class="vet-name">{_vet_name}</div>
                                                        <div class="vet-role-loc">Veterinary Doctor &nbsp;&middot;&nbsp; {_vet_place}</div>
                                                        {f'<a href="tel:{_vet_phone}" class="vet-phone">📞 Call {_vet_phone}</a>' if _vet_phone and _vet_phone != "0000000000" else '<span style="color:#94a3b8; font-size:0.75rem;">Contact Unavailable</span>'}
                                                    </div>
                                                </div>
                                                <div class="vet-card">
                                                    <div class="vet-avatar" style="background:#dcfce7; font-size:1.3rem;">🌾</div>
                                                    <div class="vet-info">
                                                        <div class="vet-name">{_gopa_name}</div>
                                                        <div class="vet-role-loc">Gopalamitra &nbsp;&middot;&nbsp; {_gopa_mandal}</div>
                                                        {f'<a href="tel:{_gopa_phone}" class="vet-phone">📞 Call {_gopa_phone}</a>' if _gopa_phone and _gopa_phone != "0000000000" else '<span style="color:#94a3b8; font-size:0.75rem;">Contact Unavailable</span>'}
                                                    </div>
                                                </div>
                                            """, unsafe_allow_html=True)

                                    else:
                                        _, left_content_col, _ = st.columns([1, 2, 1], gap="medium")
                                        with left_content_col:
                                            st.markdown(f"""
                                                <div class="result-card result-card--healthy">
                                                    <span class="result-status-badge result-status-badge--healthy">&#9679; No Disease Detected</span>
                                                    <div class="result-icon">&#9989;</div>
                                                    <h2 class="result-status-text result-status-text--healthy">{disease_status}</h2>
                                                    <p style="color:#64748b;font-size:0.85rem;margin:0.25rem 0 1rem;">Analysis: {disease_type}</p>
                                                    <div class="confidence-bar-track">
                                                        <div class="confidence-bar-fill confidence-bar-fill--healthy" style="width:{_bar_pct}%"></div>
                                                    </div>
                                                    <p class="confidence-pct confidence-pct--healthy">{confidence:.1f}% confidence</p>
                                                </div>
                                            """, unsafe_allow_html=True)

                                    # PDF download and WhatsApp Share buttons
                                    st.markdown("<br>", unsafe_allow_html=True)
                                    btn_col1, btn_col2 = st.columns(2)
                                    with btn_col1:
                                        if st.session_state.get("pdf_report_bytes"):
                                            st.download_button(
                                                label="📄 Download Prediction Report (PDF)",
                                                data=st.session_state.pdf_report_bytes,
                                                file_name=st.session_state.get("pdf_report_filename", "CHMI_Report.pdf"),
                                                mime="application/pdf",
                                                use_container_width=True,
                                            )
                                    with btn_col2:
                                        st.markdown(f"""
                                            <a href="{whatsapp_url}" target="_blank" style="text-decoration: none; display: block; width: 100%;">
                                                <div style="display: flex; align-items: center; justify-content: center; gap: 0.5rem; background: #25D366; color: white; font-weight: 600; font-size: 0.95rem; height: 38px; border-radius: 10px; box-shadow: 0 2px 8px rgba(37,211,102,0.3); text-align: center; cursor: pointer;">
                                                    <svg width="16" height="16" fill="white" viewBox="0 0 24 24" style="flex-shrink:0;">
                                                        <path d="M12.012 2c-5.506 0-9.989 4.478-9.99 9.984a9.96 9.96 0 0 0 1.333 4.982L2 22l5.202-1.362a9.92 9.92 0 0 0 4.808 1.238h.005c5.507 0 9.99-4.479 9.991-9.986.002-2.67-1.037-5.18-2.929-7.072A9.913 9.913 0 0 0 12.012 2zm5.727 14.153c-.313.88-1.564 1.72-2.15 1.782-.587.062-1.173.28-3.79-.762-3.344-1.332-5.465-4.743-5.632-4.966-.168-.223-1.34-1.783-1.34-3.402 0-1.62.845-2.415 1.147-2.738.303-.323.66-.402.88-.402.22 0 .44.002.63.01.2.008.468-.076.732.553.27.643.923 2.257 1.002 2.418.08.162.133.35.025.56-.107.21-.16.34-.32.53-.16.19-.336.425-.48.57-.16.16-.328.337-.142.657.186.32.827 1.36 1.77 2.203.943.844 1.74 1.107 2.067 1.27.327.163.518.139.712-.086.195-.225.836-.973 1.06-1.306.223-.332.447-.278.754-.163.308.115 1.954.922 2.29 1.09.336.168.56.25.643.393.083.143.083.827-.23 1.707z"/>
                                                    </svg>
                                                    Share to WhatsApp
                                                </div>
                                            </a>
                                        """, unsafe_allow_html=True)

                                else:
                                    _conf_int = int(round(confidence))
                                    _conf_color = "#f59e0b" if confidence >= 60 else "#ef4444"
                                    _conf_track_color = "#fde68a" if confidence >= 60 else "#fecaca"
                                    _conic_deg = _conf_int * 3.6
                                    st.markdown(f"""
                                        <div style="background:linear-gradient(135deg,#fffbeb 0%,#fef3c7 100%);border:1.5px solid #f59e0b;border-radius:18px;box-shadow:0 8px 32px rgba(245,158,11,0.15),0 2px 8px rgba(0,0,0,0.06);overflow:hidden;margin:1.25rem 0;">
                                            <div style="background:linear-gradient(90deg,#f59e0b,#d97706);padding:0.9rem 1.5rem;display:flex;align-items:center;gap:0.6rem;">
                                                <span style="font-size:1.35rem;">&#9888;&#65039;</span>
                                                <div>
                                                    <div style="color:#fff;font-weight:700;font-size:1.05rem;line-height:1.2;">Further Investigation Needed</div>
                                                    <div style="color:rgba(255,255,255,0.85);font-size:0.78rem;margin-top:1px;">AI confidence is below the 80% threshold required for a reliable result</div>
                                                </div>
                                            </div>
                                            <div style="padding:1.25rem 1.5rem;">
                                                <div style="background:#fff;border:1px solid #fde68a;border-radius:12px;padding:0.9rem 1.1rem;margin-bottom:1.1rem;display:flex;align-items:center;gap:1rem;">
                                                    <div style="width:54px;height:54px;border-radius:50%;background:conic-gradient({_conf_color} {_conic_deg}deg,#e5e7eb 0deg);display:flex;align-items:center;justify-content:center;flex-shrink:0;box-shadow:0 2px 8px rgba(0,0,0,0.1);">
                                                        <div style="width:38px;height:38px;border-radius:50%;background:#fff;display:flex;align-items:center;justify-content:center;font-size:0.72rem;font-weight:700;color:{_conf_color};">{_conf_int}%</div>
                                                    </div>
                                                    <div style="flex:1;">
                                                        <div style="font-size:0.78rem;color:#6b7280;margin-bottom:4px;">Current Confidence Score</div>
                                                        <div style="background:#e5e7eb;border-radius:999px;height:8px;overflow:hidden;">
                                                            <div style="background:linear-gradient(90deg,{_conf_color},{_conf_track_color});width:{_conf_int}%;height:100%;border-radius:999px;"></div>
                                                        </div>
                                                        <div style="display:flex;justify-content:space-between;font-size:0.68rem;color:#9ca3af;margin-top:3px;">
                                                            <span>0%</span><span style="color:#f59e0b;font-weight:600;">Threshold: 80%</span><span>100%</span>
                                                        </div>
                                                    </div>
                                                </div>
                                                <div style="font-size:0.82rem;font-weight:600;color:#92400e;margin-bottom:0.6rem;text-transform:uppercase;letter-spacing:0.05em;">&#128203; Tips to Improve Results</div>
                                                <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;margin-bottom:1.1rem;">
                                                    <div style="background:#fff;border:1px solid #fde68a;border-radius:10px;padding:0.6rem 0.75rem;display:flex;align-items:center;gap:0.5rem;">
                                                        <span style="font-size:1.1rem;">&#9728;&#65039;</span>
                                                        <span style="font-size:0.8rem;color:#78350f;">Shoot in bright, natural light</span>
                                                    </div>
                                                    <div style="background:#fff;border:1px solid #fde68a;border-radius:10px;padding:0.6rem 0.75rem;display:flex;align-items:center;gap:0.5rem;">
                                                        <span style="font-size:1.1rem;">&#128269;</span>
                                                        <span style="font-size:0.8rem;color:#78350f;">Show affected area clearly</span>
                                                    </div>
                                                    <div style="background:#fff;border:1px solid #fde68a;border-radius:10px;padding:0.6rem 0.75rem;display:flex;align-items:center;gap:0.5rem;">
                                                        <span style="font-size:1.1rem;">&#128208;</span>
                                                        <span style="font-size:0.8rem;color:#78350f;">Hold camera steady, avoid blur</span>
                                                    </div>
                                                    <div style="background:#fff;border:1px solid #fde68a;border-radius:10px;padding:0.6rem 0.75rem;display:flex;align-items:center;gap:0.5rem;">
                                                        <span style="font-size:1.1rem;">&#128004;</span>
                                                        <span style="font-size:0.8rem;color:#78350f;">Fill the frame with the animal</span>
                                                    </div>
                                                </div>
                                                <div style="background:linear-gradient(90deg,#fef3c7,#fff);border:1px solid #f59e0b;border-left:4px solid #d97706;border-radius:10px;padding:0.75rem 1rem;display:flex;align-items:center;gap:0.75rem;">
                                                    <span style="font-size:1.4rem;flex-shrink:0;">&#129690;</span>
                                                    <span style="font-size:0.82rem;color:#78350f;line-height:1.5;"><strong>Symptoms visible?</strong> Don't wait &mdash; consult a licensed veterinary doctor immediately for a clinical diagnosis and treatment.</span>
                                                </div>
                                            </div>
                                        </div>
                                    """, unsafe_allow_html=True)



                            
                            else:
                                st.markdown("""
                                    <div style="background: #ffeaea; 
                                                border: 2px solid #dc3545; 
                                                border-radius: 15px; 
                                                padding: 2rem; 
                                                text-align: center;
                                                box-shadow: 0 8px 25px rgba(0,0,0,0.1);">
                                                <p style="color: #dc3545; font-weight: 700; font-size: 1.25rem; margin: 0.5rem 0 0 0;">
                                                ⚠️Warning
                                                </p>
                                                <p style="color: #dc3545; font-weight: 600; font-size: 1rem; margin: 0.5rem 0 0 0;">
                                                Irrelevant Image Detected</p>
                                        <div style="background: white; 
                                                    padding: 1rem; 
                                                    border-radius: 8px; 
                                                    margin-top: 1rem;">
                                            <p style="margin: 0; color: #dc3545;">
                                                The uploaded image does not appear to be a clear or relevant cow image.<br>
                                                Please upload a high-quality cow image for accurate analysis and better results.
                                            </p>
                                        </div>
                                    </div>
                                """, unsafe_allow_html=True)
                        except Exception as e:
                            import traceback
                            traceback.print_exc()
                            st.session_state["prediction_done"] = "error"
                            st.session_state["_last_error"] = str(e)
                            st.markdown(f"""
                                <div style="background: #f8d7da;
                                            border: 2px solid #f5c6cb;
                                            border-radius: 15px;
                                            padding: 2rem;
                                            text-align: center;">
                                    <h3 style="color: #721c24; margin-bottom: 1rem;">❌ Prediction Failed</h3>
                                    <p style="color: #721c24; margin: 0 0 1rem 0;">
                                        An error occurred during analysis: <strong>{html.escape(str(e))}</strong>
                                    </p>
                                    <p style="color: #721c24; margin: 0; font-size: 0.9rem;">
                                        Please try again with a different image, or contact support if the problem persists.
                                    </p>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        
                        if st.session_state.get("prediction_done") == "success":
                            st.sidebar.markdown('<div style="display:flex;align-items:center;gap:0.5rem;padding:0.3rem 0;"><div style="width:8px;height:8px;background:#059669;border-radius:50%;flex-shrink:0;box-shadow:0 0 6px rgba(5,150,105,0.5);"></div><span style="font-size:0.78rem;color:#34d399;font-weight:600;">Prediction complete</span></div>', unsafe_allow_html=True)
                        elif st.session_state.get("prediction_done") == "error":
                            st.sidebar.markdown('<div style="display:flex;align-items:center;gap:0.5rem;padding:0.3rem 0;"><div style="width:8px;height:8px;background:#ef4444;border-radius:50%;flex-shrink:0;box-shadow:0 0 6px rgba(239,68,68,0.5);"></div><span style="font-size:0.78rem;color:#f87171;font-weight:600;">Prediction failed</span></div>', unsafe_allow_html=True)

                    # ── New scan button (shown after any prediction attempt) ──────────────
                    if st.session_state.get("prediction_done") in ("success", "error"):
                        st.markdown("<br>", unsafe_allow_html=True)
                        if st.button("🔄 Analyze Another Cattle", use_container_width=True):
                            for _k in ["cattle_details_submitted", "cattle_id", "gender", "age",
                                       "prediction_done", "upload_file", "pdf_report_bytes",
                                       "pdf_report_filename", "_last_error"]:
                                st.session_state.pop(_k, None)
                            st.rerun()

        else:
            st.warning("Please fill all details before submitting.")

    # ── Scan History ─────────────────────────────────────────────────────────────
    if st.session_state.scan_history:
        st.markdown("<hr style='border:none; border-top:1px solid #e5e7eb; margin:2rem 0 1rem 0;'>", unsafe_allow_html=True)
        with st.expander(f"📊 Scan History — {len(st.session_state.scan_history)} scan(s) this session", expanded=False):
            # Render each history item as a card
            for idx, item in enumerate(st.session_state.scan_history[::-1]): # Show newest first
                is_diseased = "Infected" in item["Status"]
                badge_style = "background:#fef2f2; color:#dc2626; border:1px solid #fecaca;" if is_diseased else "background:#f0fdf4; color:#15803d; border:1px solid #a7f3d0;"
                st.markdown(f"""
                    <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:12px; padding:0.9rem 1.25rem; margin-bottom:0.75rem; display:flex; justify-content:space-between; align-items:center; box-shadow:0 1px 3px rgba(0,0,0,0.02);">
                        <div style="display:flex; align-items:center; gap:0.9rem;">
                            <div style="font-size:1.6rem;">{"🚨" if is_diseased else "✅"}</div>
                            <div>
                                <span style="font-weight:700; color:#0f172a; font-size:0.92rem;">Cattle ID: #{item['Cattle ID']}</span>
                                <span style="font-size:0.72rem; color:#94a3b8; margin-left:0.5rem;">🕒 {item['Time']}</span>
                                <p style="margin:0.15rem 0 0 0; font-size:0.8rem; color:#64748b; font-weight:500;">Disease: {item['Disease']}</p>
                            </div>
                        </div>
                        <div style="text-align:right;">
                            <span style="font-size:0.72rem; font-weight:700; text-transform:uppercase; padding:0.25rem 0.65rem; border-radius:100px; {badge_style}">
                                {item['Status']}
                            </span>
                            <p style="margin:0.2rem 0 0 0; font-size:0.82rem; font-weight:700; color:#0f172a;">{item['Confidence']}</p>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            if st.button("🗑️ Clear History", key="clear_hist"):
                st.session_state.scan_history = []
                st.rerun()


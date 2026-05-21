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
    Processes a user-uploaded image and returns error and acceptance status.
    """
    pil_img = Image.open(image).convert('RGB').resize((224, 224))  # BUG-003 fixed: use parameter not global
    img_array = np.array(pil_img).astype('float32') / 255.0
    input_data = np.expand_dims(img_array, axis=0)
    reconstructed = run_tflite_inference(interpreter, input_details, output_details, input_data)
    error = reconstruction_error_tflite(input_data, reconstructed)
    is_accepted = error < threshold
    return {
        'error': error,
        'accepted': is_accepted
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




def soft_voting_ensemble(interpreters, image, weights, CLASS_NAMES):
    total_weighted_probs = np.zeros(len(CLASS_NAMES))
    # Prepare input once — identical for every model in the ensemble
    input_data = np.expand_dims(image, axis=0).astype(np.float32)

    for name, interpreter in interpreters.items():
        in_idx  = interpreter.get_input_details()[0]['index']
        out_idx = interpreter.get_output_details()[0]['index']
        interpreter.set_tensor(in_idx, input_data)
        interpreter.invoke()
        probs = interpreter.get_tensor(out_idx)[0]
        total_weighted_probs += weights.get(name, 0) * normalize_probs(probs)

    final_probs = total_weighted_probs / TOTAL_WEIGHT
    
    if len(CLASS_NAMES) == 4:
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

_step_order = ["user_info", "otp_verify", "dashboard"]
_step_info  = [("User Info", "1"), ("Verify OTP", "2"), ("Diagnose", "3")]
_cur_idx    = _step_order.index(current_step) if current_step in _step_order else 0

st.sidebar.markdown("<div style='font-size:0.68rem;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.1rem;margin-bottom:0.75rem;'>Workflow</div>", unsafe_allow_html=True)
for i, (label, num) in enumerate(_step_info):
    if i < _cur_idx:
        dot_bg, dot_color, txt_color, weight = "#059669", "#fff", "#94a3b8", "500"
        dot_inner = "✓"
    elif i == _cur_idx:
        dot_bg, dot_color, txt_color, weight = "#059669", "#fff", "#f1f5f9", "700"
        dot_inner = num
    else:
        dot_bg, dot_color, txt_color, weight = "#1e293b", "#475569", "#475569", "400"
        dot_inner = num
    st.sidebar.markdown(f"""
        <div style="display:flex;align-items:center;gap:0.75rem;padding:0.45rem 0;">
            <div style="width:28px;height:28px;background:{dot_bg};border-radius:50%;
                        display:flex;align-items:center;justify-content:center;
                        font-size:0.75rem;font-weight:700;color:{dot_color};flex-shrink:0;">
                {dot_inner}
            </div>
            <span style="font-size:0.88rem;font-weight:{weight};color:{txt_color};">{label}</span>
        </div>
    """, unsafe_allow_html=True)

st.sidebar.markdown("<div style='border-top:1px solid #1e293b;margin:1rem 0;'></div>", unsafe_allow_html=True)
st.sidebar.markdown("<div style='font-size:0.68rem;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.1rem;margin-bottom:0.6rem;'>Text Size</div>", unsafe_allow_html=True)
_fs_col1, _fs_col2, _fs_col3 = st.sidebar.columns([1, 1.2, 1])
with _fs_col1:
    if st.button("−", help="Decrease font size", key="fs_dec"):
        st.session_state.font_size = max(14, st.session_state.font_size - 2)
with _fs_col2:
    st.sidebar.markdown(f"<div style='text-align:center;font-weight:700;color:#94a3b8;font-size:0.85rem;padding-top:0.45rem;'>{st.session_state.font_size}px</div>", unsafe_allow_html=True)
with _fs_col3:
    if st.button("+", help="Increase font size", key="fs_inc"):
        st.session_state.font_size = min(26, st.session_state.font_size + 2)

st.sidebar.markdown("<div style='border-top:1px solid #1e293b;margin:1rem 0;'></div>", unsafe_allow_html=True)
st.sidebar.markdown("<div style='font-size:0.68rem;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.1rem;margin-bottom:0.6rem;'>Language / భాష</div>", unsafe_allow_html=True)
_lang_choice = st.sidebar.radio("", ["English", "తెలుగు"], key="lang_radio", horizontal=True, label_visibility="collapsed")
st.session_state.language = "te" if _lang_choice == "తెలుగు" else "en"

if st.session_state.scan_history:
    st.sidebar.markdown("<div style='border-top:1px solid #1e293b;margin:1rem 0;'></div>", unsafe_allow_html=True)
    st.sidebar.markdown(f"<div style='font-size:0.72rem;color:#64748b;'>📊 {len(st.session_state.scan_history)} scan(s) this session</div>", unsafe_allow_html=True)


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
#MainMenu, footer, header {{ visibility: hidden; }}
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

/* ── Primary buttons ── */
.stButton > button,
.stButton > button p,
.stButton > button span,
.stButton > button div {{
    background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.65rem 2rem !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.01rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 10px rgba(5,150,105,0.35) !important;
    height: auto !important;
    line-height: 1.5 !important;
}}
.stButton > button:hover,
.stButton > button:hover p,
.stButton > button:hover span,
.stButton > button:hover div {{
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(5,150,105,0.5) !important;
    background: linear-gradient(135deg, #047857 0%, #065f46 100%) !important;
    color: #ffffff !important;
}}
.stButton > button:active {{ transform: translateY(0) !important; }}

/* ── Download button ── */
.stDownloadButton > button,
.stDownloadButton > button p,
.stDownloadButton > button span,
.stDownloadButton > button div {{
    background: #ffffff !important;
    color: #047857 !important;
    border: 2px solid #059669 !important;
    border-radius: 10px !important;
    padding: 0.65rem 2rem !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 1px 4px rgba(5,150,105,0.15) !important;
}}
.stDownloadButton > button:hover,
.stDownloadButton > button:hover p,
.stDownloadButton > button:hover span,
.stDownloadButton > button:hover div {{
    background: #f0fdf4 !important;
    box-shadow: 0 4px 14px rgba(5,150,105,0.25) !important;
    transform: translateY(-1px) !important;
    color: #047857 !important;
}}

/* ── Form submit button ── */
.stFormSubmitButton > button,
.stFormSubmitButton > button p,
.stFormSubmitButton > button span,
.stFormSubmitButton > button div {{
    background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.7rem 2rem !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    box-shadow: 0 2px 10px rgba(5,150,105,0.35) !important;
    transition: all 0.2s !important;
}}
.stFormSubmitButton > button:hover,
.stFormSubmitButton > button:hover p,
.stFormSubmitButton > button:hover span,
.stFormSubmitButton > button:hover div {{
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(5,150,105,0.5) !important;
    color: #ffffff !important;
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
[data-testid="stSidebar"] button {{
    background: #1e293b !important;
    box-shadow: none !important;
    border: 1px solid #334155 !important;
    padding: 0.4rem 0.75rem !important;
    border-radius: 8px !important;
    transform: none !important;
}}
[data-testid="stSidebar"] button p,
[data-testid="stSidebar"] button span,
[data-testid="stSidebar"] button div {{
    color: #cbd5e1 !important;
    font-size: 0.85rem !important;
}}
[data-testid="stSidebar"] button:hover {{
    background: #334155 !important;
    transform: none !important;
    box-shadow: none !important;
}}
[data-testid="stSidebar"] button:hover p,
[data-testid="stSidebar"] button:hover span,
[data-testid="stSidebar"] button:hover div {{
    color: #ffffff !important;
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

    # ── User Details ──────────────────────────────────────────────────────────
    _section("User Details")
    _row("Name", user.get("name", "N/A"))
    _row("Mobile", user.get("mobile", "N/A"))
    _row("Village", user.get("village", "N/A"))
    _row("Mandal", user.get("mandal", "N/A"))
    _row("District", user.get("district", "N/A"))
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

        # Ensure defaults in session_state
        for key in ["village_custom", "mandal_custom", "district_custom"]:
            if key not in st.session_state:
                st.session_state[key] = ""

        # Row 2: Village, Mandal, District as dropdowns
        col3, col4, col5 = st.columns(3)

        # --- Village ---
        with col3:
            villages_with_other = villages_list + ["Other"]
            village_selection = st.selectbox(
                "Village",
                options=villages_with_other,
                index=villages_with_other.index(
                    st.session_state.user_details.get("village", villages_with_other[0])
                ) if st.session_state.user_details.get("village", "") in villages_with_other else 0,
                key="village_select"
            )

            if village_selection == "Other":
                st.session_state.village_custom = st.text_input(
                    "Enter Village",
                    value=st.session_state.village_custom,
                    key="village_input"
                )
                village = st.session_state.village_custom
            else:
                village = village_selection

        # --- Mandal ---
        with col4:
            mandals_with_other = mandals_list + ["Other"]
            mandal_selection = st.selectbox(
                "Mandal",
                options=mandals_with_other,
                index=mandals_with_other.index(
                    st.session_state.user_details.get("mandal", mandals_with_other[0])
                ) if st.session_state.user_details.get("mandal", "") in mandals_with_other else 0,
                key="mandal_select"
            )

            if mandal_selection == "Other":
                st.session_state.mandal_custom = st.text_input(
                    "Enter Mandal",
                    value=st.session_state.mandal_custom,
                    key="mandal_input"
                )
                mandal = st.session_state.mandal_custom
            else:
                mandal = mandal_selection

        # --- District ---
        with col5:
            districts_with_other = districts_list + ["Other"]
            district_selection = st.selectbox(
                "District",
                options=districts_with_other,
                index=districts_with_other.index(
                    st.session_state.user_details.get("district", districts_with_other[0])
                ) if st.session_state.user_details.get("district", "") in districts_with_other else 0,
                key="district_select"
            )

            if district_selection == "Other":
                st.session_state.district_custom = st.text_input(
                    "Enter District",
                    value=st.session_state.district_custom,
                    key="district_input"
                )
                district = st.session_state.district_custom
            else:
                district = district_selection

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
        if not village.strip():
            errors.append("Village is required.")
        if not mandal.strip():
            errors.append("Mandal is required.")
        if not district.strip():
            errors.append("District is required.")
        if not mobile.strip() or not is_valid_mobile(mobile.strip()):
            errors.append("Valid 10-digit Mobile Number is required.")
        # if lat is None or lon is None:
        #     errors.append("Location not captured. Please allow location access.")
        if village == "Select" or mandal == "Select" or district == "Select":
            errors.append("Please select a valid Village, Mandal, and District.")
        if errors:
            for err in errors:
                st.error(err)
        else:
            st.session_state.user_details = {
                "name": html.escape(name.strip()),
                "village": html.escape(village.strip()),
                "mandal": html.escape(mandal.strip()),
                "district": html.escape(district.strip()),
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
        col_summary, col_edit = st.columns([4, 1], gap="small")
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
            
            # Add sub-steps for dashboard
            if current_step == "dashboard":
                # st.sidebar.markdown("---")
                st.sidebar.markdown("### Dashboard Workflow")

                # Disease type selected
                st.sidebar.markdown("✅ Disease Type Selected")
                
                # Image uploaded
                if st.session_state.get("upload_file"):
                    st.sidebar.markdown("✅ Image Uploaded")
                else:
                    st.sidebar.markdown("❌ No Image Uploaded")
            
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
                    label="",
                    options=["LSD - Lumpy Skin Disease", "FMD - Foot and Mouth Disease"],
                    index=0,
                    key="disease_radio",
                    help="Select the type of disease you want to detect",
                    horizontal=True
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
                    "",
                    type=["png", "jpg", "jpeg"],
                    key="upload_file",
                    help="Supported formats: PNG, JPG, JPEG (Max: 200MB)"
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
                st.sidebar.markdown("🟡 Prediction Pending")
            else:
                st.sidebar.markdown("⚪ Waiting for Inputs")
            
            if uploaded_file and disease_type:

                if st.button("Submit for Prediction", use_container_width=True, type="primary"):
                    with st.spinner("Running prediction..."):
                        try:
                            st.session_state["prediction_done"] = "success"
                            result = analyze_single_image(uploaded_file, interpreter, input_details, output_details, threshold)
                            if result['accepted']:

                                # Load cached ensemble models (no re-loading on every click)
                                if disease_type.startswith("LSD"):
                                    disease_code = "LSD"
                                    missing_lsd = [p for p in TFLITE_MODELS_LSD.values() if not os.path.exists(p)]
                                    if missing_lsd:
                                        raise RuntimeError(
                                            "LSD detection models are not installed on this system. "
                                            "Please copy the 'LSD Models' folder (5 .tflite files) into the project directory."
                                        )
                                    interpreters = get_lsd_models()
                                else:
                                    disease_code = "FMD"
                                    interpreters = get_fmd_models()

                                image = preprocess_image(uploaded_file)
                                
                                
                                
                                if disease_type.startswith("LSD"):
                                    probs,predicted_label = soft_voting_ensemble(interpreters, image, MODEL_WEIGHTS,CLASS_NAMES)
                                else:
                                    probs,predicted_label = soft_voting_ensemble(interpreters, image, MODEL_WEIGHTS,CATEGORIES)
                                
                                confidence = probs * 100
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
                                    _vet_sup  = df_villages[df_villages["Place of working"].str.strip().str.lower() == user.get('village','').lower()]
                                    _gopa_sup = df_mandals[df_mandals["mandal"].str.strip().str.lower() == user.get('mandal','').lower()]
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
                                        f"📍 *Location:* {user.get('village', 'N/A')}, {user.get('mandal', 'N/A')}, {user.get('district', 'N/A')}\n"
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
                                    st.markdown("""
                                        <div style="
                                            background: #ffeeba;
                                            padding: 1.5rem;
                                            border-radius: 12px;
                                            border: 1px solid #d97706;
                                            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
                                            text-align: center;
                                            margin: 1rem 0;">
                                            <p style="color: #92400e; font-weight: 700; font-size: 1.25rem; margin: 0.5rem 0 0 0;">
                                                ⚠️ Further Investigation Needed
                                            </p>
                                            <p style="color: #92400e; font-weight: 600; font-size: 1rem; margin: 0.5rem 0 0 0;">
                                                Prediction confidence is below threshold (80%)
                                            </p>
                                            <p style="color: #92400e; font-weight: 500; margin: 0.5rem 0 0 0;">
                                                Please retake the image in good lighting, ensure the affected area is clearly visible, and try again. Contact a veterinary doctor if symptoms persist.
                                            </p>
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
                            st.session_state["prediction_done"] = "error"
                            st.session_state["_last_error"] = str(e)
                            st.markdown("""
                                <div style="background: #f8d7da;
                                            border: 2px solid #f5c6cb;
                                            border-radius: 15px;
                                            padding: 2rem;
                                            text-align: center;">
                                    <h3 style="color: #721c24; margin-bottom: 1rem;">❌ Prediction Failed</h3>
                                    <p style="color: #721c24; margin: 0;">
                                        An error occurred during analysis. Please try again with a different image,
                                        or contact support if the problem persists.
                                    </p>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        
                        if st.session_state.get("prediction_done") == "success":
                            st.sidebar.markdown("✅ Prediction Completed")
                        elif st.session_state.get("prediction_done") == "error":
                            st.sidebar.markdown("❌ Prediction Failed")

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


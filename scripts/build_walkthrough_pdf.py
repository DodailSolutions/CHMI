"""
Generates a professional "Code Walkthrough" PDF for the CHMI project.
Run: python scripts/build_walkthrough_pdf.py
Output: docs/CHMI_Code_Walkthrough.pdf
"""
import os
from datetime import datetime
from fpdf import FPDF

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "docs")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PATH = os.path.join(OUT_DIR, "CHMI_Code_Walkthrough.pdf")

# ── Palette ───────────────────────────────────────────────────────────────────
GREEN = (22, 163, 74)
DARK = (17, 24, 39)
GREY = (107, 114, 128)
LIGHT = (240, 253, 244)
CODE_BG = (245, 247, 250)
RULE = (220, 225, 230)
ACCENT = (5, 150, 105)


class PDF(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*GREY)
        self.cell(0, 6, "CHMI - Cattle Health Monitor  |  Code Walkthrough", ln=False)
        self.cell(0, 6, "Confidential", ln=True, align="R")
        self.set_draw_color(*RULE)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)

    def footer(self):
        if self.page_no() == 1:
            return
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*GREY)
        self.cell(0, 6, f"Page {self.page_no()}", align="C")


pdf = PDF()
pdf.set_auto_page_break(auto=True, margin=18)
pdf.set_title("CHMI Code Walkthrough")
EPW = pdf.w - pdf.l_margin - pdf.r_margin


def h1(text):
    pdf.ln(2)
    pdf.set_fill_color(*GREEN)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 11, f"  {text}", fill=True, ln=True)
    pdf.set_text_color(*DARK)
    pdf.ln(3)


def h2(text):
    pdf.ln(1)
    pdf.set_text_color(*ACCENT)
    pdf.set_font("Helvetica", "B", 11.5)
    pdf.cell(0, 8, text, ln=True)
    pdf.set_draw_color(*RULE)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.l_margin + 40, pdf.get_y())
    pdf.set_text_color(*DARK)
    pdf.ln(2)


def body(text):
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*DARK)
    pdf.multi_cell(0, 5.6, text)
    pdf.ln(1.5)


def bullet(label, text):
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(*ACCENT)
    pdf.cell(4, 5.6, chr(149), ln=False)
    pdf.set_text_color(*DARK)
    start = pdf.get_x()
    pdf.cell(0, 5.6, label, ln=True)
    if text:
        pdf.set_font("Helvetica", "", 9.5)
        pdf.set_text_color(75, 85, 99)
        pdf.set_x(start)
        pdf.multi_cell(0, 5, text)
    pdf.set_text_color(*DARK)
    pdf.ln(1)


def code(text):
    pdf.set_fill_color(*CODE_BG)
    pdf.set_draw_color(*RULE)
    pdf.set_font("Courier", "", 8.5)
    pdf.set_text_color(30, 41, 59)
    x, y = pdf.get_x(), pdf.get_y()
    lines = text.strip("\n").split("\n")
    h = 4.6 * len(lines) + 4
    pdf.rect(x, y, EPW, h, style="DF")
    pdf.set_xy(x + 2, y + 2)
    for ln in lines:
        pdf.set_x(x + 3)
        pdf.cell(0, 4.6, ln, ln=True)
    pdf.set_y(y + h)
    pdf.ln(2)
    pdf.set_text_color(*DARK)


def kv_table(rows, c1=55):
    c2 = EPW - c1
    for i, (k, v) in enumerate(rows):
        fill = i % 2 == 0
        if fill:
            pdf.set_fill_color(248, 250, 252)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(55, 65, 81)
        pdf.cell(c1, 7, f"  {k}", border=0, fill=fill, ln=False)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*DARK)
        pdf.multi_cell(c2, 7, str(v), border=0, fill=fill)
    pdf.ln(2)


# ══════════════════════════════════════════════════════════════════════════════
# COVER
# ══════════════════════════════════════════════════════════════════════════════
pdf.add_page()
pdf.set_fill_color(*GREEN)
pdf.rect(0, 0, pdf.w, 70, style="F")
pdf.set_xy(0, 22)
pdf.set_text_color(255, 255, 255)
pdf.set_font("Helvetica", "B", 30)
pdf.cell(0, 14, "CHMI", ln=True, align="C")
pdf.set_font("Helvetica", "", 14)
pdf.cell(0, 8, "Cattle Health Monitor", ln=True, align="C")

pdf.set_xy(0, 95)
pdf.set_text_color(*DARK)
pdf.set_font("Helvetica", "B", 22)
pdf.cell(0, 12, "Code Walkthrough", ln=True, align="C")
pdf.set_font("Helvetica", "", 12)
pdf.set_text_color(*GREY)
pdf.cell(0, 8, "A Developer's Guide to Architecture & Implementation", ln=True, align="C")

pdf.set_xy(0, 135)
pdf.set_font("Helvetica", "", 10)
pdf.set_text_color(75, 85, 99)
pdf.multi_cell(
    0, 6,
    "An AI-powered Streamlit web application for detecting Foot-and-Mouth Disease (FMD)\n"
    "and Lumpy Skin Disease (LSD) in cattle from field photographs, using a weighted\n"
    "ensemble of TensorFlow Lite deep-learning models.",
    align="C",
)

pdf.set_xy(pdf.l_margin, 185)
pdf.set_fill_color(*LIGHT)
pdf.set_draw_color(*GREEN)
box_rows = [
    ("Project", "CHMI - Cattle Health Monitor"),
    ("Primary File", "app.py  (~2,900 lines)"),
    ("Stack", "Python 3.9 / 3.10, Streamlit, TensorFlow Lite, Pillow, fpdf2"),
    ("Document Type", "Internal code walkthrough (for developers)"),
    ("Generated", datetime.now().strftime("%d %B %Y")),
]
pdf.set_xy(pdf.l_margin, 188)
kv_table(box_rows, c1=42)

pdf.set_y(-30)
pdf.set_font("Helvetica", "I", 9)
pdf.set_text_color(*GREY)
pdf.cell(0, 6, "Prepared for the development team  -  share internally only.", ln=True, align="C")

# ══════════════════════════════════════════════════════════════════════════════
# 1. OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
pdf.add_page()
h1("1.  System Overview")
body(
    "CHMI is a single-page Streamlit application that lets a field user (typically a farmer "
    "or veterinary worker) photograph a cow and receive an instant AI assessment of whether "
    "the animal shows signs of Foot-and-Mouth Disease (FMD) or Lumpy Skin Disease (LSD). "
    "When disease is detected with sufficient confidence, the app surfaces the nearest "
    "veterinary contact, generates a downloadable PDF report, and records the case."
)
body(
    "The entire application logic lives in a single module, app.py. It is organised as a "
    "linear sequence of helper functions followed by a step-driven UI controlled through "
    "Streamlit's session state. The code reads top-to-bottom: model loaders, inference "
    "utilities, OTP/SMS helpers, the report builder, geocoding helpers, and finally the "
    "four UI 'steps'."
)

h2("End-to-end user journey")
bullet("1. User Info", "Farmer enters name, mobile, and location (auto-filled via GPS reverse-geocoding or pincode lookup).")
bullet("2. OTP Verification", "A 6-digit OTP is sent over SMS (SMSCountry API) to confirm the mobile number.")
bullet("3. Dashboard", "User enters cattle details, selects FMD or LSD, and uploads/captures an image.")
bullet("4. Prediction & Report", "Image is validated, run through the model ensemble, and a result + PDF report is produced.")

h2("Key design decisions")
bullet("TensorFlow Lite over full TF", "All models are .tflite to keep memory and load time low enough for cloud/edge deployment.")
bullet("Weighted soft-voting ensemble", "Five CNN architectures vote per prediction; weights favour the historically strongest models.")
bullet("Autoencoder gatekeeper", "A separate 'cow autoencoder' plus heuristics reject non-cattle / junk images before classification.")
bullet("Streamlit session state machine", "A single `st.session_state.step` string drives which screen renders, avoiding multi-page routing.")

# ══════════════════════════════════════════════════════════════════════════════
# 2. PROJECT STRUCTURE
# ══════════════════════════════════════════════════════════════════════════════
pdf.add_page()
h1("2.  Project Structure")
body("The repository is intentionally flat. Models and assets sit alongside the app for simple, path-relative loading via a single BASE_DIR anchor.")
kv_table([
    ("app.py", "The complete application: helpers + UI state machine (~2,900 lines)."),
    ("healthcheck.py", "Standalone diagnostic: verifies models, CSVs, imports, and a live server."),
    ("requirements.txt", "Pinned dependencies (TF 2.16.2, Streamlit 1.50, Pillow, fpdf2)."),
    ("cow_autoencoder_flex.tflite", "Image-validation autoencoder (the 'is this really a cow?' gate)."),
    ("FMD New Models/", "5 TFLite classifiers for Foot-and-Mouth Disease."),
    ("LSD Models/", "5 TFLite classifiers for Lumpy Skin Disease."),
    ("FMD Images/ , LSD Images/", "Reference sample images shown in the UI."),
    ("*.csv", "Vet doctors, Gopalamitra workers, district lookup data."),
    ("indian_states_districts.json", "State/district cascade for the location form."),
    ("uploaded_images/ , image_metadata/", "Runtime capture of each submitted case."),
    (".streamlit/secrets.toml", "SMS API credentials (git-ignored)."),
    ("scripts/run.sh , commands.bat", "One-command setup + launch for macOS/Linux and Windows."),
], c1=52)

h2("Configuration anchor")
body("Every file path is resolved relative to the module location, so the app runs identically regardless of the working directory it is launched from:")
code(
    "BASE_DIR = os.path.dirname(os.path.abspath(__file__))\n"
    "model_path = os.path.join(BASE_DIR, \"cow_autoencoder_flex.tflite\")"
)

# ══════════════════════════════════════════════════════════════════════════════
# 3. MODEL LOADING & CACHING
# ══════════════════════════════════════════════════════════════════════════════
pdf.add_page()
h1("3.  Model Loading & Caching")
body(
    "Loading TFLite interpreters is expensive, so every loader is wrapped in "
    "@st.cache_resource. This guarantees the models are read from disk and allocated "
    "exactly once per server process, then reused across every user interaction and rerun."
)
code(
    "@st.cache_resource\n"
    "def get_fmd_models():\n"
    "    return load_tflite_models(TFLITE_MODELS_FMD)\n\n"
    "@st.cache_resource\n"
    "def get_lsd_models():\n"
    "    return load_tflite_models(TFLITE_MODELS_LSD)"
)
body(
    "The autoencoder loader uses a special op-resolver "
    "(BUILTIN_WITHOUT_DEFAULT_DELEGATES) to avoid delegate-related crashes on some "
    "platforms. Each loader returns the interpreter together with its input/output tensor "
    "details so inference code never re-queries them."
)

h2("The five-model ensemble")
body("Both diseases use the same five architectures, each contributing a fixed weight to the final vote:")
kv_table([
    ("EfficientNetV2-B0  (v2b0)", "weight 0.2"),
    ("EfficientNetV2-S   (v2s)", "weight 0.5"),
    ("VGG16              (vgg16)", "weight 1.5  (highest trust)"),
    ("ResNet50           (resnet50)", "weight 0.5"),
    ("EfficientNet-B0    (b0)", "weight 0.3"),
], c1=70)
body("These weights (MODEL_WEIGHTS) were tuned so the most reliable architecture dominates while the others refine the decision.")

# ══════════════════════════════════════════════════════════════════════════════
# 4. IMAGE VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
pdf.add_page()
h1("4.  Image Validation (the Gatekeeper)")
body(
    "Before any disease model runs, analyze_single_image() decides whether the upload is "
    "even a plausible cattle photo. This matters because the autoencoder alone is unreliable - "
    "a solid-colour or blank image can score a LOWER reconstruction error than a real cow. "
    "The function therefore layers four cheap heuristics ahead of the autoencoder."
)
h2("The four-layer filter")
bullet("1. Pixel variance", "Rejects solid / near-solid images (var < 0.004) - real photos have rich texture.")
bullet("2. Per-channel colour spread", "Rejects near-grayscale images where no RGB channel has meaningful std (< 0.06).")
bullet("3. Brightness sanity", "Rejects pure-black or pure-white frames (mean brightness outside 0.04-0.96).")
bullet("4. Autoencoder reconstruction error", "Accepts only when 0.0003 <= MSE <= 0.0075: too-simple images fall below, non-cow images rise above.")
code(
    "MIN_ERROR = 0.0003\n"
    "is_accepted = (MIN_ERROR <= error <= threshold)   # threshold = 0.0075\n"
    "# reason: 'ok' | 'too_simple' | 'not_cow' | 'low_variance' | ..."
)
body(
    "The function returns a small dict ({'error', 'accepted', 'reason'}) so the UI can give "
    "the user a specific, friendly explanation when an image is rejected, rather than a "
    "generic failure."
)

# ══════════════════════════════════════════════════════════════════════════════
# 5. INFERENCE PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
pdf.add_page()
h1("5.  The Inference Pipeline")
body("Once an image passes validation, prediction follows a deterministic pipeline. The core is the weighted soft-voting ensemble.")

h2("Step 1 - Preprocess")
body("Every image is opened as RGB, resized to 224x224, and scaled to float32 in [0, 1] - the exact format every model was trained on.")
code(
    "image = Image.open(uploaded_file).convert('RGB').resize((224, 224))\n"
    "image_array = np.array(image).astype('float32') / 255.0"
)

h2("Step 2 - Soft-voting ensemble")
body(
    "soft_voting_ensemble() runs the same input through all five interpreters, normalises "
    "each model's raw output, and accumulates a weight-scaled probability vector. Two "
    "defensive helpers keep the math safe:"
)
bullet("normalize_probs()", "Falls back to a uniform distribution if a model returns an all-zero output (BUG-009 fix), preventing divide-by-zero.")
bullet("collapse_probs_to_n()", "Reconciles models that output a different number of classes by summing equal-sized groups, so 2-class and 4-class heads can vote together.")
code(
    "total_weighted_probs += w * probs\n"
    "total_used_weight     += w\n"
    "final_probs = total_weighted_probs / total_used_weight"
)

h2("Step 3 - Decision")
body(
    "For the 4-class FMD head, classes are folded into 'Diseased' (knuckles + mouth) vs "
    "'Healthy' (foot + muzzle) and the larger group wins. For 2-class heads, the argmax "
    "class and its probability are returned directly. The probability becomes the "
    "confidence percentage shown to the user."
)
h2("Step 4 - Confidence gating")
body(
    "An 80% confidence threshold governs how results are presented. Above it, the verdict is "
    "shown with full styling and (if diseased) vet contacts + PDF. Below it, the UI warns "
    "that the result is not reliable enough to act on - protecting users from low-quality captures."
)

# ══════════════════════════════════════════════════════════════════════════════
# 6. UI STATE MACHINE
# ══════════════════════════════════════════════════════════════════════════════
pdf.add_page()
h1("6.  UI State Machine")
body(
    "The interface is a four-state machine driven entirely by st.session_state.step. Each "
    "branch renders one screen and advances the step on success. State (OTP code, location, "
    "cattle details, upload) persists across Streamlit reruns in session_state."
)
kv_table([
    ("user_info", "Collect name, mobile, and location. GPS or pincode auto-fills village/mandal/district/state."),
    ("otp_verify", "Verify the 6-digit OTP. Includes resend cool-down and dev-mode bypass."),
    ("dashboard", "Cattle details form -> disease selection -> image upload -> run prediction."),
    ("(result)", "Rendered inline within the dashboard after prediction: verdict, confidence bar, vet card, PDF."),
], c1=40)

h2("Helper UI pieces")
bullet("load_custom_css()", "Injects the full design system (hero banners, info cards, confidence bars, styled radios).")
bullet("render_stepper()", "Draws the progress indicator at the top so users see where they are in the flow.")
bullet("Location helpers", "reverse_geocode(), fetch_pincode_data(), update_location_by_pincode(), plus fuzzy state/district matchers.")

h2("Authentication & OTP")
body(
    "generate_otp() uses the secrets module for a cryptographically secure 6-digit code. "
    "send_otp_sms() posts to the SMSCountry REST API using Base64 basic-auth credentials "
    "from st.secrets. When credentials are absent or placeholders, IS_DEV_MODE activates and "
    "OTP sending is simulated - so developers can run the full flow locally without an SMS account."
)

# ══════════════════════════════════════════════════════════════════════════════
# 7. REPORTING & DATA
# ══════════════════════════════════════════════════════════════════════════════
pdf.add_page()
h1("7.  Reporting, Geolocation & Data")
h2("PDF report generation")
body(
    "generate_pdf_report() builds a branded one-page PDF in memory with fpdf2 and returns it "
    "as bytes for Streamlit's download button. It contains four sections - User Details, "
    "Cattle Details, Prediction Result (colour-coded red/green), and, when diseased, "
    "Recommended Veterinary Support pulled from the CSV lookups. A disclaimer footer reminds "
    "users to consult a licensed vet."
)

h2("Geolocation pipeline")
bullet("Browser GPS", "streamlit_geolocation requests device coordinates with the user's permission.")
bullet("Reverse geocoding", "reverse_geocode() calls OpenStreetMap Nominatim to turn lat/lon into village/mandal/district.")
bullet("Pincode fallback", "fetch_pincode_data() resolves an Indian PIN code when GPS is unavailable.")
bullet("Fuzzy matching", "find_best_state_match() / find_best_district_match() align geocoded names to the canonical JSON list.")

h2("Reference data")
body(
    "Three CSVs (veternary_doctors, gopalamitra, districts) plus indian_states_districts.json "
    "drive the location cascade and the 'who to contact' recommendations. _load_csv_data() and "
    "_load_states_data() load and cache them."
)

h2("Case capture")
body(
    "Every submitted image is written to uploaded_images/ and a matching text record (user, "
    "cattle, result, confidence) to image_metadata/, giving an auditable trail of field cases."
)

# ══════════════════════════════════════════════════════════════════════════════
# 8. RUNNING & EXTENDING
# ══════════════════════════════════════════════════════════════════════════════
pdf.add_page()
h1("8.  Running & Extending")
h2("Local setup")
code(
    "# macOS / Linux\n"
    "bash scripts/run.sh\n\n"
    "# Windows\n"
    "commands.bat\n\n"
    "# Verify everything is in place\n"
    "python healthcheck.py"
)
body(
    "The run scripts create a .venv, install requirements, copy the secrets template, run the "
    "health check, and launch the app on http://localhost:8501. LSD models are optional - the "
    "app still runs FMD detection if they are missing."
)

h2("Where to make common changes")
kv_table([
    ("Tune model trust", "Edit MODEL_WEIGHTS at the top of app.py."),
    ("Adjust validation strictness", "Change `threshold` / MIN_ERROR in analyze_single_image()."),
    ("Change confidence gate", "Update the 80% comparison in the dashboard prediction block."),
    ("Add a new disease", "Add a model dict + loader, then a branch in the dashboard step."),
    ("Restyle the UI", "All CSS lives in load_custom_css()."),
    ("Edit the report", "Modify sections inside generate_pdf_report()."),
], c1=58)

h2("Notable safeguards already in place")
bullet("Secure OTP", "secrets.randbelow for non-guessable codes; HTML escaping on user input.")
bullet("Zero-sum guard", "normalize_probs() prevents divide-by-zero on degenerate model output.")
bullet("Graceful degradation", "Dev-mode SMS simulation and optional LSD models keep local dev frictionless.")
bullet("Network timeouts", "All external calls (SMS, Nominatim, pincode) use explicit timeouts.")

pdf.ln(4)
pdf.set_draw_color(*RULE)
pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
pdf.ln(3)
pdf.set_font("Helvetica", "I", 9)
pdf.set_text_color(*GREY)
pdf.multi_cell(0, 5,
    "End of walkthrough. For setup details see SETUP.txt; for a live environment check run "
    "healthcheck.py. This document reflects app.py as of the latest commit and is intended "
    "for internal developer onboarding."
)

pdf.output(OUT_PATH)
print(f"Written: {OUT_PATH}")

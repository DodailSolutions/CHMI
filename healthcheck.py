import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

models = [
    'cow_autoencoder_flex.tflite',
    'FMD New Models/EfficientNetV2B0_model.tflite',
    'FMD New Models/EfficientNetV2S_model.tflite',
    'FMD New Models/VGG16_model.tflite',
    'FMD New Models/ResNet50_model.tflite',
    'FMD New Models/EfficientNetB0_model.tflite',
]
lsd_models = [
    'LSD Models/EfficientNetV2B0_model.tflite',
    'LSD Models/EfficientNetV2S_model.tflite',
    'LSD Models/VGG16_model.tflite',
    'LSD Models/ResNet50_model.tflite',
    'LSD Models/EfficientNetB0_model.tflite',
]
csvs = ['veternary_doctors.csv', 'gopalamitra.csv', 'districts.csv']
images = [
    'FMD Images/1.jpeg','FMD Images/2.jpeg','FMD Images/3.jpeg',
    'LSD Images/1.jpg','LSD Images/2.jpg','LSD Images/3.jpg',
]

def check(label, files, warn_only=False):
    print(f"\n=== {label} ===")
    for f in files:
        if os.path.exists(f):
            size = os.path.getsize(f)
            print(f"  OK  {f}  ({size/1024:.1f} KB)")
        else:
            tag = "WARN" if warn_only else "MISSING"
            print(f"  !! {tag}  {f}")

check("FMD Models", models)
check("LSD Models (optional)", lsd_models, warn_only=True)
check("CSV Files", csvs)
check("Sample Images", images, warn_only=True)

print("\n=== Python Imports ===")
packages = [
    ('streamlit', '__version__'),
    ('tensorflow', '__version__'),
    ('pandas', '__version__'),
    ('PIL', '__version__'),
    ('numpy', '__version__'),
    ('requests', '__version__'),
    ('streamlit_geolocation', None),
]
for pkg, attr in packages:
    try:
        m = __import__(pkg)
        ver = getattr(m, attr, 'installed') if attr else 'installed'
        print(f"  OK  {pkg} {ver}")
    except ImportError as e:
        print(f"  !! MISSING  {pkg}: {e}")

print("\n=== Autoencoder Load Test ===")
try:
    import tensorflow as tf
    interp = tf.lite.Interpreter(model_path='cow_autoencoder_flex.tflite')
    interp.allocate_tensors()
    inp = interp.get_input_details()
    out = interp.get_output_details()
    print(f"  OK  Loaded — input shape: {inp[0]['shape']}, output shape: {out[0]['shape']}")
except Exception as e:
    print(f"  !! ERROR: {e}")

print("\n=== CSV Load Test ===")
try:
    import pandas as pd
    vets = pd.read_csv('veternary_doctors.csv')
    gopa = pd.read_csv('gopalamitra.csv')
    dist = pd.read_csv('districts.csv')
    print(f"  OK  veternary_doctors.csv — {len(vets)} rows, cols: {list(vets.columns)}")
    print(f"  OK  gopalamitra.csv      — {len(gopa)} rows, cols: {list(gopa.columns)}")
    print(f"  OK  districts.csv        — {len(dist)} rows, cols: {list(dist.columns)}")
except Exception as e:
    print(f"  !! ERROR: {e}")

print("\n=== HTTP Check ===")
try:
    import urllib.request
    code = urllib.request.urlopen('http://localhost:8501', timeout=5).getcode()
    print(f"  OK  localhost:8501 responded HTTP {code} — app is LIVE")
except Exception as e:
    print(f"  !! App not reachable at localhost:8501: {e}")

print("\nDone.")

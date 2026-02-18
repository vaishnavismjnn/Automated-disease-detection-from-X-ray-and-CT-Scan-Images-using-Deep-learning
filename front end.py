# main_project_fixed_v2.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageOps
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

st.set_page_config(page_title="🧬 Multiple Disease Prediction Website", layout="wide")



st.markdown("""
<style>

/* Remove Streamlit default borders */
[data-testid="stSidebar"] .stSelectbox > div > div {
    border: none !important;
    box-shadow: none !important;
}

/* Modern selectbox container */
[data-testid="stSidebar"] .stSelectbox > div {
    border: 2px solid #e63946 !important;       /* red border */
    border-radius: 10px !important;             /* smooth rounded corners */
    background: #ffffff !important;
    padding: 6px 10px !important;
    transition: all 0.3s ease-in-out;
    box-shadow: 0 3px 14px rgba(0,0,0,0.08);     /* soft modern shadow */
}

/* Hover glow effect */
[data-testid="stSidebar"] .stSelectbox > div:hover {
    box-shadow: 0 4px 20px rgba(230,57,70,0.35); /* red glow */
    transform: translateY(-2px);                /* light lift */
}

/* Label styling */
[data-testid="stSidebar"] .stSelectbox label {
    font-size: 15px !important;
    font-weight: 700 !important;
    color: #1d3557 !important;                /* clean dark blue */
    margin-bottom: 6px !important;
}

/* Select text styling */
[data-testid="stSidebar"] .stSelectbox select, 
[data-testid="stSidebar"] .stSelectbox div[role="button"] {
    font-size: 14px !important;
    font-weight: 500 !important;
    color: #1a1a1a !important;
}
[data-testid="stAppViewContainer"] {
    background: #fafafa;
}

[data-testid="stSidebar"] {
    background: #ffffffee;
    border-right: 1px solid #e0e0e0;
}

[data-testid="stHeader"] {
    background: #ffffffcc;
    border-bottom: 1px solid #e0e0e0;
}

h1,h2,h3 { color:#202124 !important; }
p,label { color:#3c4043 !important; }

</style>
""", unsafe_allow_html=True)




# --------------------------------


# Update MODEL_PATHS if you have real .h5 models
MODEL_PATHS = {
    "Pneumonia":    r"C:\MAIN PROJECT\models\chest_xray (1).h5",
    "Brain Tumor":  r"C:\MAIN PROJECT\models\BrainTumor10EpochsCategorical.h5",
    "Kidney Stone": r"C:\Users\vaish\Downloads\kidney (1)\kidney\best_model.h5",
    "Bone Tumor":   r"C:\MAIN PROJECT\models\bone_classifier.h5",
}
EXPECTED_MOD = {"Pneumonia": "CXR", "Brain Tumor": "Brain", "Kidney Stone": "Kidney", "Bone Tumor": "Bone"}

# Sidebar


# Helpers
def pil_to_cv2(pil_img):
    pil_img = ImageOps.exif_transpose(pil_img)
    open_cv_image = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

def preprocess_for_model(pil_img, model):
    s = model.input_shape
    if len(s) == 4:
        _, h, w, c = s
    elif len(s) == 3:
        _, h, w = s; c = 1
    else:
        h,w,c = 224,224,3
    h = 224 if h is None else int(h)
    w = 224 if w is None else int(w)
    c = 3 if c is None else int(c)
    img = pil_img.convert("L") if c==1 else pil_img.convert("RGB")
    img = img.resize((w,h))
    arr = keras_image.img_to_array(img).astype("float32")/255.0
    if c==1 and arr.shape[-1]==3:
        arr = np.mean(arr, axis=2, keepdims=True)
    return np.expand_dims(arr,0).astype("float32")

# Feature extraction
def extract_features_from_cv2(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h,w = gray.shape
    total = h*w

    mean_intensity = float(np.mean(gray))
    bright_ratio = float(np.sum(gray > 180))/max(1,total)
    dark_ratio   = float(np.sum(gray < 40))/max(1,total)

    lap_var = float(np.var(cv2.Laplacian(gray, cv2.CV_64F)))

    left = gray[:, :w//2]
    right = cv2.flip(gray[:, w//2:], 1)
    minh = min(left.shape[0], right.shape[0])
    minw = min(left.shape[1], right.shape[1])
    if minh>0 and minw>0:
        sym = float(np.mean(np.abs(left[:minh,:minw].astype(np.int32) - right[:minh,:minw].astype(np.int32))))
    else:
        sym = 9999.0

    ann = max(6, int(min(h,w)*0.06))
    rim = np.concatenate([gray[:ann,:].ravel(), gray[-ann:,:].ravel(), gray[:, :ann].ravel(), gray[:, -ann:].ravel()])
    bg_black_ratio = float((rim < 8).sum())/max(1, rim.size)

    cx, cy = w//2, h//2
    sx = max(1, w//6); sy = max(1, h//6)
    central = gray[max(0,cy-sy):min(h,cy+sy), max(0,cx-sx):min(w,cx+sx)]
    bright_central_blob = bool(np.mean(central>170) > 0.03) if central.size>0 else False

    try:
        lr = gray[int(h*0.12):int(h*0.74), int(w*0.12):int(w*0.88)]
        _, thr = cv2.threshold(lr, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        lung_mask = thr < 30
        lung_area_ratio = float(np.sum(lung_mask))/max(1, lung_mask.size)
    except Exception:
        lung_area_ratio = 0.0

    edges = cv2.Canny(gray, 40, 120)
    vertical_edges = float(np.mean(edges[:, w//3:2*w//3] > 0))
    chestness = float((lung_area_ratio*1.4 + vertical_edges*1.2 + max(0.0, (1 - sym/300))) / 3.0)
    rib_conf = float(np.mean(cv2.Sobel(gray, cv2.CV_32F, 1, 0)))

    body_area_ratio = float(np.sum(gray>30))/max(1,total)

    features = {
        "mean": round(mean_intensity,3),
        "bright_ratio": round(bright_ratio,4),
        "dark_ratio": round(dark_ratio,4),
        "lap_var": round(lap_var,3),
        "symmetry": round(sym,3),
        "bg_black_ratio": round(bg_black_ratio,4),
        "bright_central_blob": bool(bright_central_blob),
        "lung_area_ratio": round(lung_area_ratio,4),
        "chestness": round(chestness,3),
        "rib_conf": round(rib_conf,3),
        "body_area_ratio": round(body_area_ratio,4),
        "brain_texture_score": round(lap_var * (1 - lung_area_ratio),3)
    }
    return features

# Detector with decision_reason
def detect_modality_from_features(f):
    br = f["bright_ratio"]; dr = f["dark_ratio"]; lap = f["lap_var"]
    sym = f["symmetry"]; bg = f["bg_black_ratio"]; lung = f["lung_area_ratio"]
    chest = f["chestness"]; rib = f["rib_conf"]; body = f["body_area_ratio"]
    brain_tex = f["brain_texture_score"]; blob = f["bright_central_blob"]

    reason = "none"

    # Strong Brain override (kept stable)
    # VERY-STRONG Brain override: when texture & laplacian are extremely high, prefer Brain even if bg looks CT-like
    # VERY-STRONG Brain override (relaxed): when texture & laplacian are very high, prefer Brain
    # - catches images with very high laplacian + moderate brain texture (your example)
    if (brain_tex >= 650 and lap >= 1000) or (lap >= 1400 and brain_tex >= 400):
        reason = "brain_force_override_relaxed_high_lap_texture"
        return "Brain", reason


    # Kidney CT: stricter so Brain isn't stolen (made stricter vs prior)
    if bg >= 0.30 and (blob or lap >= 200) and body >= 0.05:
        reason = "kidney_strict_bg_and_blob_or_high_lap"
        return "Kidney", reason
    # --- Bone override when chestness is fooled but no rib pattern present ---
    # If chestness is high but rib pattern confidence is very low (no ribs),
    # and the image is mostly body area with moderate-high texture, prefer Bone.
    # if chest >= 0.55 and rib < 0.05 and body >= 0.85 and lap >= 180:
    #     reason = "bone_override_chestness_falsely_high_no_ribs_high_body"
    #     return "Bone", reason
    if chest >= 0.55 and rib < 0.15 and body >= 0.80 and lap >= 150:
        reason = "bone_override_false_cxr (high_chestness_no_ribs_high_body)"
        return "Bone", reason


    # Bone promotion (before CXR): require small lung area + high body area + chestness not too chest-like
    if lung < 0.12 and body >= 0.75 and (lap >= 150 or br > 0.03) and chest < 0.6:
        reason = "bone_promote_small_lung_high_body_lap_or_bright"
        return "Bone", reason

    # Brain normal rule
    if bg >= 0.18 and brain_tex >= 700 and sym < 800:
        reason = "brain_standard"
        return "Brain", reason

    # Bone normal rule
    if (lap >= 300 and br >= 0.02 and lung < 0.30) or (br > 0.12 and lap > 200):
        reason = "bone_standard"
        return "Bone", reason

    # CXR rule (chestness + ribs)
    if (chest >= 0.40 and lung > 0.08) or (rib > 0.8 and chest >= 0.35):
        reason = "cxr_chestness_or_rib"
        return "CXR", reason

    # fallback bone if image is almost entirely body area and textured
    if body >= 0.92 and lap >= 120:
        reason = "fallback_body_high_lap_promote_bone"
        return "Bone", reason

    reason = "unknown_fallback"
    return "Unknown", reason

# load models (non-fatal)
@st.cache_resource
def load_models(paths):
    models = {}
    for k,p in paths.items():
        if p and os.path.exists(p):
            try:
                models[k] = load_model(p)
            except Exception:
                models[k] = None
        else:
            models[k] = None
    return models

MODELS = load_models(MODEL_PATHS)

# UI
# --- UI LAYOUT FIXED: IMAGE FIRST → PREDICT BUTTON BELOW ---
st.title("🧬 Multiple Disease Prediction Website")

# Left side only disease select
st.sidebar.title("Options")
selected_disease = st.sidebar.selectbox(
    "Select disease to predict:",
    list(MODEL_PATHS.keys()),
    key="disease_selector"
)


# ↓ Removed from sidebar (set by code internally)
show_debug = False            # You said only disease selection on left
force_predict = False         # You said run on unknown handled inside code
allow_when_unknown = True     # Always allow prediction when unknown (internal)

# Main UI layout – image first, button next
# Main UI layout – image first, button next
uploaded = st.file_uploader("Upload an image (jpg/png)", type=["jpg","jpeg","png"])

if uploaded:
    # Show a small preview image ONLY
    preview = Image.open(uploaded)
    st.image(preview, caption="Uploaded Image", width=380)

    # Predict button right under the image
    predict_btn = st.button("Predict")
else:
    predict_btn = None



def print_final_result(disease_name, prob):
    percent = prob * 100  # convert to %
    if prob >= 0.50:
        st.success(f"✅ {disease_name} Detected (confidence = {percent:.2f}%)")
    else:
        st.info(f"🟢 Normal — No {disease_name} (confidence = {100 - percent:.2f}%)")


if predict_btn:
    if not uploaded:
        st.error("Upload an image first.")
    else:
        try:
            pil = Image.open(uploaded)
            pil = ImageOps.exif_transpose(pil)
            cv2_img = pil_to_cv2(pil)


        except Exception as e:
            st.error(f"Could not open image: {e}")
            st.stop()

        features = extract_features_from_cv2(cv2_img)
        modality, reason = detect_modality_from_features(features)
        features["decision_reason"] = reason

        if show_debug:
            st.subheader("Detector internals")
            st.json(features)


        expected_mod = EXPECTED_MOD.get(selected_disease)

        proceed = False
        if force_predict:
            proceed = True
        elif modality == expected_mod:
            proceed = True
        elif modality == "Unknown" and allow_when_unknown:
            proceed = True
            st.warning("Detector = Unknown but 'Allow prediction when detector = Unknown' is ON — running model.")
        else:
            proceed = False

        if not proceed:
            st.error(f"❌ Not a valid image for selected disease ")
        else:
            model = MODELS.get(selected_disease)
            if model is None:
                st.warning("Model file not found — showing placeholder probability based on features.")
                pseudo = 0.5
                if selected_disease == "Pneumonia":
                    pseudo = 0.2 + 0.8 * min(1.0, features["chestness"] / 0.9)
                elif selected_disease == "Brain Tumor":
                    pseudo = 0.2 + 0.8 * min(1.0, features["brain_texture_score"] / 2500.0)
                elif selected_disease == "Kidney Stone":
                    pseudo = 0.25 + 0.75 * min(1.0, (features["lap_var"]/1000.0 + (1.0 if features["bright_central_blob"] else 0.0))/1.5)
                elif selected_disease == "Bone Tumor":
                    pseudo = 0.2 + 0.8 * min(1.0, (features["lap_var"]/2000.0 + features["bright_ratio"])/1.5)
                print_final_result(selected_disease, float(pseudo))
            else:
                try:
                    arr = preprocess_for_model(pil, model)
                    raw = model.predict(arr)
                    a = np.array(raw).ravel()
                    if a.size == 1:
                        p = float(a[0])
                        print_final_result(selected_disease, p)
                    else:
                        probs = a.astype(float)
                        if not np.isclose(probs.sum(), 1.0):
                            e = np.exp(probs - np.max(probs)); probs = e / e.sum()
                        idx = 1 if len(probs) > 1 else int(np.argmax(probs))
                        p = float(probs[idx])
                        print_final_result(selected_disease, p)
                except Exception as e:
                    st.error(f"Model inference failed: {e}")

st.markdown("---")
st.markdown("""
<p class="sub-caption">Upload your scan and let AI assist you in early detection</p>
""", unsafe_allow_html=True)
st.markdown("""
<style>
.sub-caption {
    font-size: 20px !important;     /* Increase size */
    color: #1a4d7a !important;      /* Optional medical blue */
    font-weight: 500 !important;    /* Slightly bold */
    margin-top: -10px;              /* Adjust spacing */
}
</style>
""", unsafe_allow_html=True)




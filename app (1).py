# app.py
# Underwater Animal Expert - image classifier (fish / shark / crab / jellyfish)
# + "Dr. Marina" chatbot that runs on a local Ollama vision model (with memory).
#
# needs best_model.pt and labels.json in the same folder (or set MARINE_DIR)

import os
import io
import json
import base64
import hashlib
from itertools import chain
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2
import pandas as pd
from PIL import Image
import streamlit as st

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

# ollama is optional -> the classifier still works without it
try:
    import ollama
except ImportError:
    ollama = None

# ============================================================
# SETTINGS
# ============================================================
_default_dir = "/kaggle/working" if os.path.exists("/kaggle/working") else str(Path(__file__).resolve().parent)
BASE_DIR = Path(os.environ.get("MARINE_DIR", _default_dir))

CLASSIFIER_CKPT = BASE_DIR / "best_model.pt"
LABELS_FILE = BASE_DIR / "labels.json"
HISTORY_FILE = BASE_DIR / "chat_history.json"
UPLOAD_DIR = BASE_DIR / "uploaded_images"
try:
    UPLOAD_DIR.mkdir(exist_ok=True)
except OSError:
    pass

PREFERRED_MODELS = [os.environ.get("OLLAMA_MODEL", "llava:13b"), "llava:13b", "llava:7b", "llava", "moondream"]
OLLAMA_KEEP_ALIVE = "30m"      # keep the model loaded in memory between questions
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONF_OK = 0.85        # above this we trust the classifier
CONF_UNSURE = 0.60    # below this the photo is probably not one of our 4 animals
MAX_HISTORY_MESSAGES = 16   # how much of the conversation the chatbot remembers
MAX_ANIMALS_MEMORY = 5

SYSTEM_PROMPT = (
    "You are Dr. Marina, a marine biology expert specialising in underwater "
    "animals - fish, sharks, crabs and jellyfish. When asked for information "
    "about a species, always cover: scientific name, typical habitat/range, "
    "diet, size, and whether it is dangerous to humans. If a classifier "
    "prediction is given to you in the message, briefly confirm or correct it "
    "in one line, then move on to give the actual species information - do not "
    "just repeat the classifier's prediction as your whole answer. Use the "
    "conversation so far to understand follow-up questions like 'what does it "
    "eat?'. Keep answers focused but complete enough to actually inform the user."
)

# ============================================================
# PREPROCESSING  (has to be EXACTLY the same as in the training notebook)
# ============================================================
def classical_pipeline(img_bgr):
    """resize -> very mild bilateral denoise -> very mild sharpening on L channel"""
    img = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img = cv2.bilateralFilter(img, d=3, sigmaColor=20, sigmaSpace=20)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    blurred_L = cv2.GaussianBlur(L, (0, 0), sigmaX=0.8)
    L = cv2.addWeighted(L, 1.08, blurred_L, -0.08, 0)
    lab = cv2.merge([L, A, B])

    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return np.clip(img, 0, 255).astype(np.uint8)


classify_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ============================================================
# MODEL  (same layout as the training notebook so the weights load)
# ============================================================
class HierarchicalMarineNet(nn.Module):
    def __init__(self, num_categories, species_counts, dropout=0.5):
        super().__init__()
        backbone = torchvision.models.efficientnet_v2_s(weights=None)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        feat_dim = 1280
        self.dropout = nn.Dropout(dropout)
        self.category_head = nn.Linear(feat_dim, num_categories)
        self.species_heads = nn.ModuleList([nn.Linear(feat_dim, n) for n in species_counts])

    def forward(self, x):
        feat = self.pool(self.features(x)).flatten(1)
        feat = self.dropout(feat)
        return self.category_head(feat), feat


@st.cache_resource(show_spinner="Loading classifier...")
def load_classifier():
    """returns (model, labels, error_message)"""
    if not CLASSIFIER_CKPT.exists() or not LABELS_FILE.exists():
        return None, None, f"Could not find best_model.pt / labels.json in {BASE_DIR}"
    try:
        with open(LABELS_FILE) as f:
            labels = json.load(f)
        categories = labels["categories"]
        species_counts = [len(labels["species_per_category"][c]) for c in categories]
        model = HierarchicalMarineNet(len(categories), species_counts)
        state = torch.load(CLASSIFIER_CKPT, map_location=DEVICE)
        model.load_state_dict(state)
        model.to(DEVICE).eval()
        return model, labels, None
    except Exception as e:  # wrong classes / corrupted file etc
        return None, None, f"Failed to load classifier: {e}"


@torch.no_grad()
def classify_image(pil_img, model, labels):
    """Predicts category + species with a bit of test-time augmentation
    (original, flipped, rotated +-10 deg) and averages the probabilities."""
    rgb = np.array(pil_img.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # same preprocessing as training (pipeline works on BGR, model wants RGB)
    base = cv2.cvtColor(classical_pipeline(bgr), cv2.COLOR_BGR2RGB)

    variants = [base, cv2.flip(base, 1)]
    for angle in (-10, 10):
        M = cv2.getRotationMatrix2D((IMG_SIZE / 2, IMG_SIZE / 2), angle, 1.0)
        variants.append(cv2.warpAffine(base, M, (IMG_SIZE, IMG_SIZE), borderMode=cv2.BORDER_REFLECT))

    batch = torch.stack([classify_tf(v) for v in variants]).to(DEVICE)
    cat_logits, feat = model(batch)
    cat_probs = torch.softmax(cat_logits, 1).mean(0)
    cat_idx = int(cat_probs.argmax())
    category = labels["categories"][cat_idx]

    sp_probs = torch.softmax(model.species_heads[cat_idx](feat), 1).mean(0)
    sp_idx = int(sp_probs.argmax())
    species_names = labels["species_per_category"][category]

    top_k = sp_probs.topk(min(3, len(species_names)))
    return {
        "category": category,
        "category_confidence": float(cat_probs[cat_idx]),
        "species": species_names[sp_idx],
        "species_confidence": float(sp_probs[sp_idx]),
        "possible_species": species_names,
        "category_probs": {c: float(p) for c, p in zip(labels["categories"], cat_probs.cpu())},
        "top_species": [(species_names[int(i)], float(p)) for p, i in zip(top_k.values.cpu(), top_k.indices.cpu())],
    }


def nice(name):
    return name.replace("_", " ").title()


def annotate_image(pil_img, result):
    img = np.array(pil_img.convert("RGB"))
    label = f"{nice(result['category'])} / {nice(result['species'])} ({result['species_confidence']*100:.0f}%)"
    scale = max(0.5, img.shape[1] / 900)
    thick = max(1, int(scale * 2))
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    cv2.rectangle(img, (5, 5), (15 + tw, 15 + th + 10), (0, 0, 0), -1)
    cv2.putText(img, label, (10, 15 + th), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 0), thick)
    return Image.fromarray(img)

# ============================================================
# CHAT HISTORY + MEMORY  (saved to disk so it survives a page refresh)
# ============================================================
def clean_history(history):
    """drops questions that never got an answer (e.g. the run was interrupted)
    so the model never sees two user messages in a row"""
    cleaned = []
    for turn in history:
        role, content = turn.get("role"), turn.get("content")
        if role == "user":
            if cleaned and cleaned[-1]["role"] == "user":
                cleaned.pop()
            cleaned.append(turn)
        elif role == "assistant" and content and cleaned and cleaned[-1]["role"] == "user":
            cleaned.append(turn)
    if cleaned and cleaned[-1]["role"] == "user":
        cleaned.pop()
    return cleaned


def load_memory():
    """returns (history, animals)"""
    try:
        if HISTORY_FILE.exists():
            with open(HISTORY_FILE, "r") as f:
                data = json.load(f)
            if isinstance(data, list):          # old file format
                return clean_history(data), []
            return clean_history(data.get("history", [])), data.get("animals", [])
    except (OSError, json.JSONDecodeError, AttributeError):
        pass
    return [], []


def save_memory(history, animals):
    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump({"history": history, "animals": animals}, f, indent=2)
    except OSError:
        pass


def memory_note(animals):
    """short summary of the animals shown so far, added to the system prompt"""
    if not animals:
        return ""
    lines = [
        f"{i + 1}. {nice(a['category'])} / {nice(a['species'])} (classifier guess, {a['confidence'] * 100:.0f}% confidence)"
        for i, a in enumerate(animals)
    ]
    return (
        "\n\nConversation memory - animals the user has shown you so far (oldest first). "
        "The LAST one is the animal being discussed unless the user says otherwise:\n" + "\n".join(lines)
    )

# ============================================================
# OLLAMA / DR. MARINA
# ============================================================
def installed_models():
    """names of the models pulled in ollama ([] if the server isn't reachable)"""
    if ollama is None:
        return []
    try:
        names = []
        for m in ollama.list()["models"]:
            try:
                names.append(m["model"])      # newer ollama python package
            except (KeyError, TypeError):
                names.append(m["name"])       # older one
        return names
    except Exception:
        return []


def image_to_base64(pil_img):
    buf = io.BytesIO()
    pil_img.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def build_prompt(user_text, classifier_result, is_new_image):
    prompt = user_text

    if is_new_image:
        prompt = (
            "This is a NEW photo, unrelated to any animal discussed earlier "
            "in this conversation. Base your answer only on what you see in "
            "THIS image.\n\n" + prompt
        )

    if classifier_result:
        cat_conf = classifier_result["category_confidence"]
        sp_conf = classifier_result["species_confidence"]

        if cat_conf < CONF_OK:
            prompt += (
                "\n\n[An automated classifier guessed this animal but was not "
                "confident, so ignore its guess. This project only covers "
                "these four categories: fish, crab, shark, jellyfish - the "
                "animal in the image WILL be one of these four, never anything "
                "else. Look carefully: does it have a hard shell with jointed "
                "legs (crab), fins and scales (fish), a translucent bell-shaped "
                "body with tentacles (jellyfish), or a cartilage skeleton with "
                "visible gill slits (shark)? Pick whichever of these four "
                "categories best matches what you see, then name the likely "
                "species within it.]"
            )
        elif sp_conf < CONF_OK:
            species_list = ", ".join(nice(s) for s in classifier_result["possible_species"])
            prompt += (
                f"\n\n[The classifier is confident this is a "
                f"{classifier_result['category']} ({cat_conf*100:.1f}%), but its "
                f"species guess ({nice(classifier_result['species'])}) is unreliable "
                f"({sp_conf*100:.1f}% confidence) - ignore that species guess. "
                f"This dataset only includes these {classifier_result['category']} "
                f"species: {species_list}. Look at the image's markings, color "
                f"pattern, and body shape, and pick the species from THIS list "
                f"that best matches what you see. If none match well, say so "
                f"honestly instead of forcing a match.]"
            )
        else:
            prompt += (
                f"\n\n[Classifier prediction: category={classifier_result['category']} "
                f"({cat_conf*100:.1f}%), species={nice(classifier_result['species'])} "
                f"({sp_conf*100:.1f}%). You can trust this prediction unless the "
                f"image clearly shows otherwise.]"
            )
    return prompt


def stream_ollama(model_name, prior_history, animals, user_text, pil_img=None, classifier_result=None):
    """yields the answer piece by piece so it can be streamed in the UI.
    prior_history = everything BEFORE the current question (this is the chatbot's memory)"""
    messages = [{"role": "system", "content": SYSTEM_PROMPT + memory_note(animals)}]

    for turn in prior_history[-MAX_HISTORY_MESSAGES:]:
        if turn.get("content"):
            messages.append({"role": turn["role"], "content": turn["content"]})

    user_msg = {"role": "user", "content": build_prompt(user_text, classifier_result, pil_img is not None)}
    if pil_img is not None:
        user_msg["images"] = [image_to_base64(pil_img)]
    messages.append(user_msg)

    for chunk in ollama.chat(model=model_name, messages=messages, stream=True,
                             keep_alive=OLLAMA_KEEP_ALIVE,
                             options={"num_predict": 900, "temperature": 0.4}):
        piece = chunk["message"]["content"]
        if piece:
            yield piece

# ===== UI ===================================================
st.set_page_config(page_title="Underwater Animal Expert", page_icon="🐠", layout="wide")

st.markdown(
    """
    <style>
    .hero {
        padding: 1.6rem 2rem; border-radius: 16px; margin-bottom: 1.2rem;
        background: linear-gradient(120deg, #023e8a 0%, #0077b6 50%, #00b4d8 100%);
        color: white;
    }
    .hero h1 { margin: 0; font-size: 2.1rem; color: white; }
    .hero p  { margin: 0.4rem 0 0 0; font-size: 1.05rem; opacity: 0.92; }
    .small-note { font-size: 0.8rem; opacity: 0.7; }
    </style>
    <div class="hero">
        <h1>🐠 Underwater Animal Expert</h1>
        <p>Upload a photo of a fish, shark, crab or jellyfish - I'll identify it and Dr. Marina can tell you everything about it.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- session state ----------
if "history" not in st.session_state:
    st.session_state.history, st.session_state.animals = load_memory()
if "last_sent_sig" not in st.session_state:
    st.session_state.last_sent_sig = None      # image that the chat model already saw
if "cls_cache" not in st.session_state:
    st.session_state.cls_cache = {}             # image signature -> classifier result
if "job" not in st.session_state:
    st.session_state.job = None                 # the question currently being answered
if "chat_error" not in st.session_state:
    st.session_state.chat_error = None

classifier_model, labels, load_error = load_classifier()
models = installed_models()
chat_online = len(models) > 0
busy = st.session_state.job is not None

# ---------- sidebar ----------
with st.sidebar:
    st.header("📷 Image input")
    input_mode = st.radio("Choose input method", ["Upload a file", "Use camera"])

    uploaded_image = None
    if input_mode == "Upload a file":
        file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if file is not None:
            uploaded_image = Image.open(file)
    else:
        cam_file = st.camera_input("Take a photo")
        if cam_file is not None:
            uploaded_image = Image.open(cam_file)

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Selected image", width=280)

    st.divider()
    st.subheader("🧠 Chat memory")
    if st.session_state.animals:
        for a in st.session_state.animals:
            st.markdown(f"- {nice(a['category'])} / **{nice(a['species'])}**")
    else:
        st.caption("No animals discussed yet.")
    st.caption(f"Remembers the last {MAX_HISTORY_MESSAGES} messages.")

    if st.button("🗑️ Clear conversation", disabled=busy):
        st.session_state.history = []
        st.session_state.animals = []
        st.session_state.last_sent_sig = None
        st.session_state.chat_error = None
        save_memory([], [])
        st.rerun()

    st.divider()
    st.subheader("System status")
    if classifier_model is not None:
        st.success(f"Classifier loaded ({'GPU' if DEVICE.type == 'cuda' else 'CPU'})")
        with st.expander("What can it identify?"):
            for c in labels["categories"]:
                st.markdown(f"**{nice(c)}** - " + ", ".join(nice(s) for s in labels["species_per_category"][c]))
    else:
        st.error(load_error)

    chat_model = None
    if chat_online:
        default_idx = 0
        for pref in PREFERRED_MODELS:
            if pref in models:
                default_idx = models.index(pref)
                break
        chat_model = st.selectbox("Chat model", models, index=default_idx, disabled=busy)
        st.success("Dr. Marina online")
    elif ollama is None:
        st.warning("The ollama python package isn't installed - identification still works.")
    else:
        st.warning("Dr. Marina is offline (Ollama server not running or no model pulled) - identification still works.")

# ---------- classification ----------
result = None
sig = None
if uploaded_image is not None:
    buf = io.BytesIO()
    uploaded_image.convert("RGB").save(buf, format="PNG")
    sig = hashlib.md5(buf.getvalue()).hexdigest()

    if classifier_model is not None:
        if sig not in st.session_state.cls_cache:
            with st.spinner("Identifying..."):
                st.session_state.cls_cache[sig] = classify_image(uploaded_image, classifier_model, labels)
        result = st.session_state.cls_cache[sig]

if result is not None:
    st.subheader("🔎 Identification")
    col_img, col_info = st.columns([1, 1.2])

    with col_img:
        st.image(annotate_image(uploaded_image, result), caption="Classifier output", width=420)

    with col_info:
        m1, m2 = st.columns(2)
        m1.metric("Category", nice(result["category"]), f"{result['category_confidence']*100:.1f}% sure", delta_color="off")
        m2.metric("Species", nice(result["species"]), f"{result['species_confidence']*100:.1f}% sure", delta_color="off")

        if result["category_confidence"] < CONF_UNSURE:
            st.warning("I'm not confident about this one - it may not be a fish, shark, crab or jellyfish.")
        elif result["species_confidence"] < CONF_OK:
            st.info("Category looks right but the exact species is uncertain - check the top guesses below.")

        st.markdown("**Category probabilities**")
        cat_df = pd.DataFrame({"probability": result["category_probs"]})
        cat_df.index = [nice(i) for i in cat_df.index]
        st.bar_chart(cat_df, height=200)

        st.markdown(f"**Top {nice(result['category'])} species guesses**")
        for name, p in result["top_species"]:
            st.progress(min(max(p, 0.0), 1.0), text=f"{nice(name)} - {p*100:.1f}%")
elif uploaded_image is None:
    st.info("👈 Upload a photo (or use the camera) in the sidebar to get started, or just ask Dr. Marina a question below.")

# ---------- chat ----------
st.subheader("💬 Ask Dr. Marina")

# chat box + quick buttons are disabled while an answer is being generated,
# otherwise clicking them re-runs the app and kills the answer half way
user_text = st.chat_input("Ask me anything about underwater animals...", disabled=busy)

pending_question = None
if result is not None:
    b1, b2, b3, _ = st.columns([1, 1, 1, 2])
    if b1.button("Tell me about it", disabled=busy):
        pending_question = "Tell me about this animal."
    if b2.button("Is it dangerous?", disabled=busy):
        pending_question = "Is this animal dangerous to humans?"
    if b3.button("Where does it live?", disabled=busy):
        pending_question = "Where does this animal live and what does it eat?"

question = user_text or pending_question

# ---- a new question just came in -> store it and start a job ----
if question and not busy:
    st.session_state.chat_error = None
    if not chat_online or chat_model is None:
        st.session_state.chat_error = ("Dr. Marina is offline (Ollama isn't running or no model is pulled), "
                                       "so I can't answer questions. The identification above still works.")
    else:
        send_image = uploaded_image is not None and sig != st.session_state.last_sent_sig
        image_path, annotated_path = None, None

        if send_image:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            try:
                image_path = str(UPLOAD_DIR / f"{stamp}.png")
                uploaded_image.convert("RGB").save(image_path)
                if result is not None:
                    annotated_path = image_path.replace(".png", "_annotated.png")
                    annotate_image(uploaded_image, result).save(annotated_path)
            except OSError:
                image_path, annotated_path = None, None

        st.session_state.history.append({"role": "user", "content": question, "image_path": image_path})
        st.session_state.job = {
            "question": question, "send_image": send_image, "sig": sig,
            "annotated_path": annotated_path, "model": chat_model,
        }
    st.rerun()

# ---- show the conversation ----
for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        if turn.get("image_path") and Path(turn["image_path"]).exists():
            st.image(turn["image_path"], width=250)
        st.markdown(turn["content"])

if st.session_state.chat_error:
    st.error(st.session_state.chat_error)

# ---- answer the pending question (runs once per question, then re-runs the app) ----
job = st.session_state.job
if job is not None:
    error, answer = None, ""

    if job["send_image"] and sig != job["sig"]:
        error = "The image changed while I was answering - please ask again."
    else:
        with st.chat_message("assistant"):
            if job["annotated_path"] and Path(job["annotated_path"]).exists():
                st.image(job["annotated_path"], caption="Annotated output", width=300)
            try:
                gen = stream_ollama(
                    job["model"],
                    st.session_state.history[:-1],       # memory = everything before this question
                    st.session_state.animals,
                    job["question"],
                    uploaded_image if job["send_image"] else None,
                    result if job["send_image"] else None,
                )
                with st.spinner("Dr. Marina is thinking... (the first reply can take a minute while the model loads)"):
                    first = next(gen, None)               # wait for the first piece of text
                if first is not None:
                    answer = st.write_stream(chain([first], gen))
            except Exception as e:
                error = f"Could not get an answer from '{job['model']}': {e}"

    if not error and not answer.strip():
        error = (f"'{job['model']}' returned an empty answer. It is probably still loading or ran out of GPU memory - "
                 f"try again, or pick a smaller model (llava:7b / moondream) in the sidebar.")

    if error:
        st.session_state.history.pop()                    # remove the unanswered question
        st.session_state.chat_error = error
    else:
        st.session_state.history.append({"role": "assistant", "content": answer, "image_path": job["annotated_path"]})
        if job["send_image"]:
            st.session_state.last_sent_sig = job["sig"]
            if result is not None:
                st.session_state.animals.append({
                    "category": result["category"], "species": result["species"],
                    "confidence": result["species_confidence"],
                })
                st.session_state.animals = st.session_state.animals[-MAX_ANIMALS_MEMORY:]
        save_memory(st.session_state.history, st.session_state.animals)

    st.session_state.job = None
    st.rerun()

st.markdown(
    "<p class='small-note'>Identification is done by a custom EfficientNetV2 model trained on Kaggle datasets; "
    "Dr. Marina's answers are AI-generated - double check anything safety related.</p>",
    unsafe_allow_html=True,
)

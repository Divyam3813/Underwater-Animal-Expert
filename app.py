# app.py
# Underwater Animal Expert - image classifier (fish / shark / crab / jellyfish)
# + "Dr. Marina" chatbot powered by the Gemini API (with conversation memory).
#
# needs best_model.pt and labels.json in the same folder (or set MARINE_DIR)
# needs a Gemini API key: env var GEMINI_API_KEY, Streamlit secret GEMINI_API_KEY,
# or just paste it in the sidebar

import os
import io
import re
import json
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

# gemini is optional -> the classifier still works without it
try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

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

# used only if the model list can't be fetched from the API
FALLBACK_MODELS = ["gemini-flash-latest", "gemini-3.6-flash", "gemini-3.5-flash-lite"]
# model names containing these words are not chat models
SKIP_WORDS = ("image", "tts", "live", "audio", "embedding", "robotics", "computer", "native", "learnlm", "aqa", "veo", "imagen")

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
# GEMINI / DR. MARINA
# ============================================================
def find_api_key():
    """looks for the key in env vars, then streamlit secrets"""
    names = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
    for name in names:
        if os.environ.get(name, "").strip():
            return os.environ[name].strip()
    try:
        for name in names:
            if name in st.secrets:
                return str(st.secrets[name]).strip()
    except Exception:       # no secrets file
        pass
    return ""


@st.cache_resource(show_spinner=False)
def get_client(api_key):
    return genai.Client(api_key=api_key)


def model_rank(name):
    """higher = better default: flash, not lite, not preview, newest version"""
    m = re.search(r"gemini-(\d+(?:\.\d+)?)", name)
    version = float(m.group(1)) if m else 0.0
    return ("flash" in name, "lite" not in name, "preview" not in name and "exp" not in name, version)


@st.cache_resource(ttl=3600, show_spinner=False)
def get_chat_models(api_key):
    """asks the API which models this key can use (model names change often)"""
    names = []
    try:
        for m in get_client(api_key).models.list():
            name = (m.name or "").replace("models/", "")
            if "generateContent" not in (m.supported_actions or []):
                continue
            if not name.startswith("gemini") or any(w in name for w in SKIP_WORDS):
                continue
            names.append(name)
    except Exception:
        pass
    names = sorted(set(names), key=model_rank, reverse=True)
    return names or list(FALLBACK_MODELS)


def image_to_jpeg_bytes(pil_img):
    img = pil_img.convert("RGB")
    img.thumbnail((1280, 1280))         # no need to send huge photos
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


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


def build_contents(prior_history, user_text, pil_img, classifier_result, is_new_image):
    """conversation so far + the new question (with the photo attached if there is one)"""
    contents = []
    recent = [t for t in prior_history[-MAX_HISTORY_MESSAGES:] if t.get("content")]
    while recent and recent[0]["role"] != "user":       # gemini wants the chat to start with the user
        recent.pop(0)
    for turn in recent:
        role = "user" if turn["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part.from_text(text=turn["content"])]))

    parts = []
    if pil_img is not None:
        parts.append(types.Part.from_bytes(data=image_to_jpeg_bytes(pil_img), mime_type="image/jpeg"))
    parts.append(types.Part.from_text(text=build_prompt(user_text, classifier_result, is_new_image)))
    contents.append(types.Content(role="user", parts=parts))
    return contents


def friendly_error(e):
    code = getattr(e, "code", None)
    text = str(e)
    if code is None:                       # not an API error (e.g. empty answer)
        return text[:300]
    if code in (400, 401, 403) and ("API key" in text or "API_KEY" in text or code in (401, 403)):
        return "Gemini rejected the API key (or it isn't allowed to use this model). Check the key and try again."
    if code == 404:
        return "That Gemini model isn't available for this key - pick another one in the sidebar."
    if code == 429:
        return "Gemini rate limit / free quota reached. Wait a minute, or pick another model in the sidebar."
    if code and code >= 500:
        return "Gemini servers are busy right now - try again in a moment."
    return f"Gemini error: {text[:300]}"


def is_key_problem(e):
    code = getattr(e, "code", None)
    text = str(e)
    return code in (401, 403) or (code == 400 and ("API key" in text or "API_KEY" in text))


def stream_gemini(api_key, candidate_models, prior_history, animals, user_text,
                  pil_img=None, classifier_result=None, is_new_image=False):
    """yields the answer piece by piece. If a model fails before saying anything
    (not found / quota), the next candidate model is tried."""
    client = get_client(api_key)
    contents = build_contents(prior_history, user_text, pil_img, classifier_result, is_new_image)
    config = types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT + memory_note(animals))

    last_error = None
    for model_name in candidate_models:
        got_text = False
        last_chunk = None
        try:
            for chunk in client.models.generate_content_stream(model=model_name, contents=contents, config=config):
                last_chunk = chunk
                try:
                    piece = chunk.text
                except Exception:
                    piece = None
                if piece:
                    got_text = True
                    yield piece
            if got_text:
                return
            reason = ""
            if last_chunk is not None:
                feedback = getattr(last_chunk, "prompt_feedback", None)
                if feedback is not None and getattr(feedback, "block_reason", None):
                    reason = f" (blocked: {feedback.block_reason})"
                elif last_chunk.candidates:
                    reason = f" (finish reason: {last_chunk.candidates[0].finish_reason})"
            last_error = RuntimeError(f"Gemini returned an empty answer{reason}.")
        except Exception as e:
            if got_text or is_key_problem(e):   # already streaming, or retrying can't help
                raise
            last_error = e
    raise last_error

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
    st.session_state.last_sent_sig = None      # image that the chatbot already got as a "new photo"
if "cls_cache" not in st.session_state:
    st.session_state.cls_cache = {}             # image signature -> classifier result
if "job" not in st.session_state:
    st.session_state.job = None                 # the question currently being answered
if "chat_error" not in st.session_state:
    st.session_state.chat_error = None

classifier_model, labels, load_error = load_classifier()
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

    api_key = find_api_key()
    chat_model, candidate_models = None, []
    if genai is None:
        st.warning("The google-genai package isn't installed (pip install google-genai) - identification still works.")
    else:
        if not api_key:
            api_key = st.text_input("Gemini API key", type="password",
                                    help="Get a free key at aistudio.google.com. It is only kept in this session.").strip()
        if api_key:
            models = get_chat_models(api_key)
            chat_model = st.selectbox("Chat model", models, index=0, disabled=busy)
            candidate_models = [chat_model] + [m for m in models if m != chat_model][:2]
            st.success("Dr. Marina online (Gemini)")
        else:
            st.warning("Add a Gemini API key to chat with Dr. Marina - identification still works.")

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
    if chat_model is None:
        st.session_state.chat_error = ("Dr. Marina needs a Gemini API key to answer questions "
                                       "(add it in the sidebar). The identification above still works.")
    else:
        # the photo is attached to every question, but only counts as "new" the first time
        is_new = uploaded_image is not None and sig != st.session_state.last_sent_sig
        image_path, annotated_path = None, None

        if is_new:
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
            "question": question, "sig": sig, "is_new": is_new,
            "annotated_path": annotated_path, "models": candidate_models, "api_key": api_key,
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

    if job["sig"] != sig:
        error = "The image changed while I was answering - please ask again."
    else:
        with st.chat_message("assistant"):
            if job["annotated_path"] and Path(job["annotated_path"]).exists():
                st.image(job["annotated_path"], caption="Annotated output", width=300)
            try:
                gen = stream_gemini(
                    job["api_key"], job["models"],
                    st.session_state.history[:-1],       # memory = everything before this question
                    st.session_state.animals,
                    job["question"],
                    uploaded_image,
                    result if job["is_new"] else None,
                    job["is_new"],
                )
                with st.spinner("Dr. Marina is thinking..."):
                    first = next(gen, None)               # wait for the first piece of text
                if first is not None:
                    answer = st.write_stream(chain([first], gen))
            except Exception as e:
                error = friendly_error(e)

    if not error and not answer.strip():
        error = "Gemini returned an empty answer - please try again or pick another model in the sidebar."

    if error:
        st.session_state.history.pop()                    # remove the unanswered question
        st.session_state.chat_error = error
    else:
        st.session_state.history.append({"role": "assistant", "content": answer, "image_path": job["annotated_path"]})
        if job["is_new"]:
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
    "Dr. Marina's answers are AI-generated by Gemini - double check anything safety related.</p>",
    unsafe_allow_html=True,
)

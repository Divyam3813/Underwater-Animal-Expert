# Generated from: streamlit-app.ipynb
# Converted at: 2026-09-20T13:37:12.585Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# # Underwater animal classifier (fish / shark / crab / jellyfish)
# Two stage model: EfficientNetV2-S predicts the **category** first, then a species head for that category predicts the **species**.
# 
# Things to know before running:
# - Turn the GPU on: *Settings -> Accelerator -> GPU* (the code uses it automatically if it is there).
# - Only the 4 datasets (fish, shark, crab, jellyfish) should be attached to the notebook.
# - `EPOCHS = 100` but early stopping is ON, so training stops by itself if validation stops improving (this is what protects from overfitting). Set `EARLY_STOPPING = False` if you really want all 100 epochs.
# - The model is saved after every epoch, so you never have to train again. Set `TRAIN_MODEL = False` to just load it.
# 


# only run this if torch / torchvision throws errors, then restart the session
# !pip install -q --upgrade torch torchvision sympy


import os, shutil, glob, random, hashlib, json, time
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms

# ------------------------------------------------------------
# CONFIG - change things here only
# ------------------------------------------------------------
SEED = 42
IMG_SIZE = 224
EPOCHS = 100
BATCH_SIZE = 32
LR = 1e-4
WEIGHT_DECAY = 1e-2
DROPOUT = 0.5
PATIENCE = 12            # stop if val loss doesnt improve for this many epochs
EARLY_STOPPING = True    # False = always run all EPOCHS

TRAIN_MODEL = True       # False = skip training and just load the saved model
RESUME = True            # continue from last_checkpoint.pt if it exists
REBUILD_DATASET = True   # rebuild train/val folders from scratch (also wipes any leftover folders from older runs)
PRETRAINED = True        # imagenet weights for the backbone

CATEGORIES = ["crab", "fish", "jellyfish", "shark"]   # the ONLY 4 classes

# ------------------------------------------------------------
# paths (works on kaggle, falls back to local folders otherwise)
# ------------------------------------------------------------
ON_KAGGLE = os.path.exists("/kaggle/working")
INPUT_BASE = "/kaggle/input" if ON_KAGGLE else os.path.abspath("./input")
WORK_DIR = Path("/kaggle/working") if ON_KAGGLE else Path("./working").resolve()
WORK_DIR.mkdir(parents=True, exist_ok=True)

BEST_PATH = WORK_DIR / "best_model.pt"          # best weights only (this is what the streamlit app loads)
LAST_PATH = WORK_DIR / "last_checkpoint.pt"     # full state so training can be resumed
LABELS_PATH = WORK_DIR / "labels.json"
HISTORY_PATH = WORK_DIR / "history.json"

# ------------------------------------------------------------
# seeds + device
# ------------------------------------------------------------
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
cv2.setNumThreads(0)   # opencv threads + dataloader workers fight each other otherwise

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"   # mixed precision = faster + less memory on gpu

if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
    props = torch.cuda.get_device_properties(0)
    print(f"Using GPU: {props.name} ({props.total_memory / 1e9:.1f} GB)")
else:
    print("!! GPU NOT FOUND - running on CPU will be very slow.")
    print("!! Kaggle: Settings -> Accelerator -> pick a GPU, then run again.")


for dirpath, dirnames, filenames in os.walk(INPUT_BASE):
    depth = dirpath.replace(INPUT_BASE, "").count(os.sep)
    if depth > 4:
        dirnames[:] = []  # don't descend further
        continue
    indent = "  " * depth
    n_imgs = sum(1 for f in filenames if f.lower().endswith((".jpg", ".jpeg", ".png")))
    print(f"{indent}{os.path.basename(dirpath) or dirpath}  ({n_imgs} images, {len(dirnames)} subfolders)")


# ## 1. Build the combined dataset (4 categories only)


COMBINED_ROOT = WORK_DIR / "combined"
TRAIN_ROOT = COMBINED_ROOT / "train"
VAL_ROOT = COMBINED_ROOT / "val"

# wipe the old combined folder so nothing from older runs sneaks in
if REBUILD_DATASET and COMBINED_ROOT.exists():
    shutil.rmtree(COMBINED_ROOT)
for r in (TRAIN_ROOT, VAL_ROOT):
    r.mkdir(parents=True, exist_ok=True)

CATEGORY_KEYWORDS = {
    "fish":      ["large-scale-fish", "fish_dataset", "fish-dataset"],
    "crab":      ["crab-species", "crab_species"],
    "shark":     ["shark-species", "shark_species"],
    "jellyfish": ["jellyfish"],
}

SPLIT_NAMES = {"train", "test", "valid", "val", "training", "testing", "validation"}
BAD_NAME_HINTS = ("gt", "mask", "ground", "annotation", "label")

VAL_FRACTION = 0.2
MAX_IMAGES_PER_SPECIES = 300

def is_image(path):
    return path.suffix.lower() in [".jpg", ".jpeg", ".png"]

def clean_species_name(name):
    return name.strip().lower().replace(" ", "_")

def find_dataset_root(keywords, base=INPUT_BASE):
    """Find the shallowest folder under `base` whose path matches any keyword."""
    candidates = []
    for dirpath, dirnames, filenames in os.walk(base):
        low = dirpath.lower()
        if any(k in low for k in keywords):
            candidates.append(dirpath)
    if not candidates:
        return None
    return min(candidates, key=lambda p: p.count(os.sep))

def auto_collect_species(root):
    """Returns {species_name: [image_paths]}, merging same-named leaf folders
    across splits and skipping GT/mask directories."""
    root = Path(root)
    species_map = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirpath = Path(dirpath)
        imgs = [dirpath / f for f in filenames if is_image(dirpath / f)]
        if not imgs:
            continue
        name = dirpath.name
        if any(hint in name.lower() for hint in BAD_NAME_HINTS):
            continue
        if name.lower() in SPLIT_NAMES:
            continue
        species_map.setdefault(clean_species_name(name), []).extend(imgs)
    return species_map

def average_hash(path, hash_size=8):
    """Cheap perceptual hash used only to drop near-duplicate frames before
    splitting, so the same underlying photo can't land in both train and val."""
    try:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        img = cv2.resize(img, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
        avg = img.mean()
        bits = (img > avg).flatten()
        return "".join("1" if b else "0" for b in bits)
    except Exception:
        return None

def dedup(image_paths):
    """Drop images whose perceptual hash we've already seen for this species."""
    seen = set()
    kept = []
    for p in image_paths:
        h = average_hash(p)
        if h is None:
            kept.append(p)  # unreadable/corrupt image: keep it, loader will skip it later
            continue
        if h in seen:
            continue
        seen.add(h)
        kept.append(p)
    return kept

def split_train_val(image_paths, val_fraction=VAL_FRACTION, seed=SEED):
    if len(image_paths) < 2:
        return image_paths, []  # too few images to hold any out
    return train_test_split(image_paths, test_size=val_fraction, random_state=seed)

def copy_species(category, species_name, image_paths, split_root):
    dst = split_root / category / species_name
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in image_paths:
        # short hash keeps names unique across merged split folders
        digest = hashlib.md5(str(f).encode()).hexdigest()[:8]
        target = dst / f"{digest}_{f.name}"
        if target.exists():
            continue
        try:
            os.symlink(os.path.abspath(f), target)   # symlink = no extra disk space (absolute path or the link breaks)
        except OSError:
            shutil.copy2(f, target)     # fallback if symlinks arent allowed
        n += 1
    return n

def process_category(category, species_map, max_images_per_species=MAX_IMAGES_PER_SPECIES):
    total_train, total_val = 0, 0
    for sp_name, paths in species_map.items():
        paths = dedup(sorted(paths))                 # sorted -> same result on every machine
        random.Random(SEED).shuffle(paths)           # deterministic shuffle before capping/splitting
        if max_images_per_species:
            paths = paths[:max_images_per_species]
        train_paths, val_paths = split_train_val(paths)
        total_train += copy_species(category, sp_name, train_paths, TRAIN_ROOT)
        total_val += copy_species(category, sp_name, val_paths, VAL_ROOT)
    return total_train, total_val

if not any(TRAIN_ROOT.iterdir()):
    for category, keywords in CATEGORY_KEYWORDS.items():
        root = find_dataset_root(keywords)
        if root is None:
            raise FileNotFoundError(f"{category}: could NOT find a dataset under {INPUT_BASE} for keywords {keywords}. "
                                    f"Check that the dataset is added to the notebook.")
        species_map = auto_collect_species(root)
        print(f"{category}: root={root}")
        print(f"{category}: found {len(species_map)} species -> {list(species_map.keys())[:8]}{'...' if len(species_map) > 8 else ''}")
        n_train, n_val = process_category(category, species_map)
        print(f"{category}: copied {n_train} train / {n_val} val images (post-dedup)")
else:
    print("Train folder already built, skipping (REBUILD_DATASET = False)")

# safety check: exactly our 4 categories, nothing else
found_train = sorted(d.name for d in TRAIN_ROOT.iterdir() if d.is_dir())
found_val = sorted(d.name for d in VAL_ROOT.iterdir() if d.is_dir())
assert found_train == CATEGORIES, f"unexpected train categories: {found_train}"
assert found_val == CATEGORIES, f"unexpected val categories: {found_val}"

print("\nCombined dataset summary:")
for split_name, root in [("train", TRAIN_ROOT), ("val", VAL_ROOT)]:
    print(f"  -- {split_name} --")
    for cat_dir in sorted(root.iterdir()):
        species_dirs = list(cat_dir.iterdir())
        total_imgs = sum(len(list(sp.glob("*"))) for sp in species_dirs)
        print(f"    {cat_dir.name}: {len(species_dirs)} species, {total_imgs} images")


# ## 2. Classical image processing pipeline


def classical_pipeline(img_bgr):
    """
    Conservative image preprocessing pipeline (input/output is BGR).
    Goal: keep the image close to the original so species features arent destroyed.

    1. resize
    2. very mild edge-preserving denoising (bilateral)
    3. very mild sharpening on the luminance channel only
    (no CLAHE, no contrast stretching, no saturation boost, no morphology)

    NOTE: the streamlit app uses the exact same function, dont change one without the other.
    """
    # 1. resize
    img = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    # 2. mild denoise
    img = cv2.bilateralFilter(img, d=3, sigmaColor=20, sigmaSpace=20)

    # 3. sharpen L channel only
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    blurred_L = cv2.GaussianBlur(L, (0, 0), sigmaX=0.8)
    L = cv2.addWeighted(L, 1.08, blurred_L, -0.08, 0)
    lab = cv2.merge([L, A, B])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 4. safety clip
    return np.clip(img, 0, 255).astype(np.uint8)


def show_before_after(root, n_samples=4):
    root = Path(root)
    all_imgs = [f for f in root.rglob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]]
    if len(all_imgs) == 0:
        raise ValueError(f"No images found inside: {root}")

    samples = random.sample(all_imgs, min(n_samples, len(all_imgs)))
    fig, axes = plt.subplots(len(samples), 2, figsize=(10, 4 * len(samples)))
    if len(samples) == 1:
        axes = axes.reshape(1, 2)

    for i, img_path in enumerate(samples):
        raw = cv2.imread(str(img_path))
        if raw is None:
            print(f"Warning: could not read {img_path}")
            continue

        raw_rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        processed_rgb = cv2.cvtColor(classical_pipeline(raw), cv2.COLOR_BGR2RGB)

        parts = img_path.relative_to(root).parts
        category = parts[0] if len(parts) > 0 else "Unknown"
        species = parts[1] if len(parts) > 1 else "Unknown"

        axes[i, 0].imshow(raw_rgb)
        axes[i, 0].set_title(f"Original\n{category} / {species}", fontsize=11)
        axes[i, 0].axis("off")
        axes[i, 1].imshow(processed_rgb)
        axes[i, 1].set_title("Conservative Pipeline", fontsize=11)
        axes[i, 1].axis("off")

    plt.tight_layout()
    out = WORK_DIR / "classical_pipeline_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Images in train set: {len(all_imgs)} | saved: {out}")
    plt.show()

show_before_after(TRAIN_ROOT, n_samples=4)


# sanity check: same file (byte for byte) must not be in both train and val, otherwise val accuracy is fake
def file_md5(p):
    with open(p, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

train_hashes = {file_md5(p) for p in TRAIN_ROOT.rglob("*") if p.is_file()}
leaked = [p for p in VAL_ROOT.rglob("*") if p.is_file() and file_md5(p) in train_hashes]
print(f"Val images that are exact duplicates of a train image: {len(leaked)}")
if leaked:
    print("-> remove these from val, they inflate the accuracy:", leaked[:5])


# ## 3. Dataset / DataLoader - dual labels (category + species)


class MarineDataset(Dataset):

    def __init__(self, root, cat_to_idx=None, species_to_idx=None, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = []

        if cat_to_idx is None:
            self.categories = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
            self.cat_to_idx = {c: i for i, c in enumerate(self.categories)}
        else:
            self.cat_to_idx = cat_to_idx
            self.categories = sorted(cat_to_idx, key=cat_to_idx.get)

        if species_to_idx is None:
            self.species_to_idx = {}
            for cat in self.categories:
                cat_dir = self.root / cat
                if not cat_dir.exists():
                    self.species_to_idx[cat] = {}
                    continue
                species_dirs = sorted([d.name for d in cat_dir.iterdir() if d.is_dir()])
                self.species_to_idx[cat] = {s: i for i, s in enumerate(species_dirs)}
        else:
            self.species_to_idx = species_to_idx

        self.species_per_category = {
            cat: sorted(self.species_to_idx[cat], key=self.species_to_idx[cat].get)
            for cat in self.categories
        }

        for cat in self.categories:
            cat_dir = self.root / cat
            if not cat_dir.exists():
                continue
            for sp, sp_idx in self.species_to_idx[cat].items():
                sp_dir = cat_dir / sp
                if not sp_dir.exists():
                    continue
                for f in sorted(sp_dir.glob("*")):
                    if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                        self.samples.append((f, self.cat_to_idx[cat], sp_idx))

        if len(self.samples) == 0:
            raise ValueError(f"No images found in {self.root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, cat_idx, sp_idx = self.samples[idx]
        img = cv2.imread(str(path))
        if img is None:
            # corrupt image -> just use the next one
            return self.__getitem__((idx + 1) % len(self.samples))
        img = classical_pipeline(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        return img, cat_idx, sp_idx


train_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2),
])

val_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ## 4. Model


class HierarchicalMarineNet(nn.Module):

    def __init__(self, num_categories, species_counts, dropout=0.3, pretrained=True):
        super().__init__()
        weights = torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = torchvision.models.efficientnet_v2_s(weights=weights)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        feat_dim = 1280

        # freeze everything except the last 16 parameter tensors
        for p in list(self.features.parameters())[:-16]:
            p.requires_grad = False

        # batchnorm layers that are fully frozen must stay in eval mode while training
        trainable_ids = {id(p) for p in list(self.features.parameters())[-16:]}
        self._frozen_bn_names = set()
        for name, module in self.features.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                if not any(id(p) in trainable_ids for p in module.parameters()):
                    self._frozen_bn_names.add(name)

        self.dropout = nn.Dropout(dropout)
        self.category_head = nn.Linear(feat_dim, num_categories)
        self.species_heads = nn.ModuleList([nn.Linear(feat_dim, n) for n in species_counts])

    def train(self, mode=True):
        super().train(mode)
        if mode:
            for name, module in self.features.named_modules():
                if name in self._frozen_bn_names:
                    module.eval()
        return self

    def forward(self, x):
        feat = self.pool(self.features(x)).flatten(1)
        feat = self.dropout(feat)
        return self.category_head(feat), feat


# ## 5. Train / eval helpers


def species_forward(model, feat, cat_idx, sp_idx, sp_criterion):
    """Runs the species head of each sample's TRUE category. Done per category
    (batched) instead of per image, way faster than looping over every sample."""
    loss = torch.zeros((), device=feat.device)
    sp_pred = torch.zeros_like(sp_idx)
    for c in cat_idx.unique():
        mask = cat_idx == c
        logits = model.species_heads[int(c)](feat[mask]).float()
        loss = loss + sp_criterion(logits, sp_idx[mask]) * mask.sum()
        sp_pred[mask] = logits.argmax(1)
    return loss / cat_idx.size(0), sp_pred


def run_epoch(model, loader, cat_criterion, sp_criterion, optimizer=None, scaler=None):
    """One pass over the loader. Trains if optimizer is given, else just evaluates.
    Returns loss + category acc + species acc (species counts as correct only if category is also correct)."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, correct_cat, correct_sp, total = 0.0, 0, 0, 0

    for imgs, cat_idx, sp_idx in loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        cat_idx = cat_idx.to(DEVICE, non_blocking=True)
        sp_idx = sp_idx.to(DEVICE, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=USE_AMP):
                cat_logits, feat = model(imgs)
                cat_loss = cat_criterion(cat_logits.float(), cat_idx)
                sp_loss, sp_pred = species_forward(model, feat, cat_idx, sp_idx, sp_criterion)
                loss = cat_loss + sp_loss

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # keeps training stable
            scaler.step(optimizer)
            scaler.update()

        pred_cat = cat_logits.argmax(1)
        correct_cat += (pred_cat == cat_idx).sum().item()
        correct_sp += ((pred_cat == cat_idx) & (sp_pred == sp_idx)).sum().item()
        total_loss += loss.item() * imgs.size(0)
        total += imgs.size(0)

    return {"loss": total_loss / total, "cat_acc": correct_cat / total, "sp_acc": correct_sp / total}


@torch.no_grad()
def predict_all(model, loader):
    """Real end-to-end predictions (species head chosen by the PREDICTED category)."""
    model.eval()
    t_cat, p_cat, t_sp, p_sp = [], [], [], []
    for imgs, cat_idx, sp_idx in loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=USE_AMP):
            cat_logits, feat = model(imgs)
        pred_cat = cat_logits.argmax(1)
        pred_sp = torch.zeros_like(pred_cat)
        for c in pred_cat.unique():
            mask = pred_cat == c
            pred_sp[mask] = model.species_heads[int(c)](feat[mask].float()).argmax(1)   # feat is fp16 under amp -> cast back
        t_cat.append(cat_idx.numpy()); t_sp.append(sp_idx.numpy())
        p_cat.append(pred_cat.cpu().numpy()); p_sp.append(pred_sp.cpu().numpy())
    return np.concatenate(t_cat), np.concatenate(p_cat), np.concatenate(t_sp), np.concatenate(p_sp)


def evaluate(model, loader, name="Validation"):
    t_cat, p_cat, t_sp, p_sp = predict_all(model, loader)
    cat_acc = (t_cat == p_cat).mean()
    sp_acc = ((t_cat == p_cat) & (t_sp == p_sp)).mean()

    print(f"\n========== {name} ==========")
    print(f"Total samples     : {len(t_cat)}")
    print(f"Category accuracy : {cat_acc:.4f} ({cat_acc*100:.2f}%)")
    print(f"Species accuracy  : {sp_acc:.4f} ({sp_acc*100:.2f}%)")
    return cat_acc, sp_acc, (t_cat, p_cat, t_sp, p_sp)


def save_atomic(obj, path):
    """write to a temp file first then rename, so a crash mid-save cant corrupt the checkpoint"""
    tmp = str(path) + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def find_saved(filename):
    """look for a saved file in the working dir first, then in any attached kaggle dataset / notebook output"""
    if (WORK_DIR / filename).exists():
        return WORK_DIR / filename
    hits = glob.glob(f"{INPUT_BASE}/**/{filename}", recursive=True)
    return Path(hits[0]) if hits else None


# ## 6. Data loaders, model, optimizer


train_ds = MarineDataset(TRAIN_ROOT, transform=train_tf)
val_ds = MarineDataset(VAL_ROOT, cat_to_idx=train_ds.cat_to_idx, species_to_idx=train_ds.species_to_idx, transform=val_tf)
# same train images but without augmentation -> used later for a fair overfitting check
train_eval_ds = MarineDataset(TRAIN_ROOT, cat_to_idx=train_ds.cat_to_idx, species_to_idx=train_ds.species_to_idx, transform=val_tf)

print("Categories:", train_ds.categories)
for c in train_ds.categories:
    print(f"  {c}: {len(train_ds.species_per_category[c])} species")
print("Total train images:", len(train_ds))
print("Total val images  :", len(val_ds))

NUM_WORKERS = min(4, os.cpu_count() or 2)
loader_kwargs = dict(num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda"), persistent_workers=NUM_WORKERS > 0)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, **loader_kwargs)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, **loader_kwargs)
train_eval_loader = DataLoader(train_eval_ds, batch_size=BATCH_SIZE, shuffle=False, **loader_kwargs)

# save the label mapping right away so the app can always rebuild category/species names
labels = {
    "categories": train_ds.categories,
    "species_per_category": train_ds.species_per_category,
}
with open(LABELS_PATH, "w") as f:
    json.dump(labels, f, indent=2)
print("Saved", LABELS_PATH)

num_categories = len(train_ds.categories)
species_counts = [len(train_ds.species_per_category[c]) for c in train_ds.categories]
model = HierarchicalMarineNet(num_categories, species_counts, dropout=DROPOUT, pretrained=PRETRAINED).to(DEVICE)
print("num_categories:", num_categories, "species_counts:", species_counts)

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6)
scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

# categories dont have the same number of images -> weight the category loss so big classes dont dominate
cat_counts = np.bincount([s[1] for s in train_ds.samples], minlength=num_categories)
print("train images per category:", dict(zip(train_ds.categories, cat_counts.tolist())))
cat_weights = torch.tensor(cat_counts.sum() / (num_categories * np.maximum(cat_counts, 1)), dtype=torch.float32).to(DEVICE)
cat_criterion = nn.CrossEntropyLoss(weight=cat_weights, label_smoothing=0.1)
sp_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)


# ## 7. Training (saves every epoch, can resume)


history = {"train_loss": [], "val_loss": [], "train_cat_acc": [], "val_cat_acc": [],
           "train_sp_acc": [], "val_sp_acc": [], "lr": []}
best_val_loss = float("inf")
epochs_no_improve = 0
start_epoch = 0

# ---- resume from an earlier run if there is one ----
if TRAIN_MODEL and RESUME:
    last = find_saved("last_checkpoint.pt")
    if last is not None:
        ckpt = torch.load(last, map_location=DEVICE, weights_only=False)
        if ckpt.get("labels") == labels:
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            scaler.load_state_dict(ckpt["scaler"])
            history = ckpt["history"]
            best_val_loss = ckpt["best_val_loss"]
            epochs_no_improve = ckpt["epochs_no_improve"]
            start_epoch = ckpt["epoch"]
            print(f"Resuming from epoch {start_epoch} (best val loss so far: {best_val_loss:.4f})")
        else:
            print("last_checkpoint.pt was made with different classes (old run?) -> starting fresh")

if TRAIN_MODEL:
    for epoch in range(start_epoch, EPOCHS):
        if EARLY_STOPPING and epochs_no_improve >= PATIENCE:
            print(f"Early stopping - no val improvement for {PATIENCE} epochs")
            break

        t0 = time.time()
        tr = run_epoch(model, train_loader, cat_criterion, sp_criterion, optimizer, scaler)
        va = run_epoch(model, val_loader, cat_criterion, sp_criterion)
        scheduler.step(va["loss"])
        lr_now = optimizer.param_groups[0]["lr"]

        for k, v in [("train_loss", tr["loss"]), ("val_loss", va["loss"]),
                     ("train_cat_acc", tr["cat_acc"]), ("val_cat_acc", va["cat_acc"]),
                     ("train_sp_acc", tr["sp_acc"]), ("val_sp_acc", va["sp_acc"]), ("lr", lr_now)]:
            history[k].append(v)

        # save best weights (only when val loss really improves)
        improved = va["loss"] < best_val_loss - 1e-4
        if improved:
            best_val_loss = va["loss"]
            epochs_no_improve = 0
            save_atomic(model.state_dict(), BEST_PATH)
        else:
            epochs_no_improve += 1

        # save everything after EVERY epoch so a crash / timeout never loses progress
        save_atomic({
            "epoch": epoch + 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "history": history,
            "best_val_loss": best_val_loss, "epochs_no_improve": epochs_no_improve, "labels": labels,
        }, LAST_PATH)
        with open(HISTORY_PATH, "w") as f:
            json.dump(history, f)

        print(f"Epoch {epoch+1:03d}/{EPOCHS} | {time.time()-t0:5.1f}s | "
              f"train loss {tr['loss']:.4f} cat {tr['cat_acc']*100:5.1f}% sp {tr['sp_acc']*100:5.1f}% | "
              f"val loss {va['loss']:.4f} cat {va['cat_acc']*100:5.1f}% sp {va['sp_acc']*100:5.1f}% | "
              f"lr {lr_now:.1e}{'  <- best, saved' if improved else ''}")
    else:
        print(f"Finished all {EPOCHS} epochs")
else:
    print("TRAIN_MODEL = False -> skipping training, loading the saved model below")


# ## 8. Load best model + overfitting / error checks


# always evaluate the BEST checkpoint, not whatever the last epoch was
best_file = find_saved("best_model.pt")
if best_file is None:
    raise FileNotFoundError("best_model.pt not found - train the model first (TRAIN_MODEL = True)")
try:
    model.load_state_dict(torch.load(best_file, map_location=DEVICE))
except RuntimeError as e:
    raise RuntimeError("best_model.pt doesnt match the current classes (old 5-class model?). "
                       "Delete it and train again.") from e
print("Loaded", best_file)

if os.path.exists(HISTORY_PATH) and not history["train_loss"]:
    history = json.load(open(HISTORY_PATH))   # training was skipped, load the saved curves

train_cat, train_sp, _ = evaluate(model, train_eval_loader, "TRAIN (no augmentation)")
val_cat, val_sp, val_preds = evaluate(model, val_loader, "VALIDATION")


# training curves - val going UP while train goes DOWN = overfitting
if history["train_loss"]:
    epochs_ran = range(1, len(history["train_loss"]) + 1)
    best_ep = int(np.argmin(history["val_loss"])) + 1

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    axes[0].plot(epochs_ran, history["train_loss"], label="train")
    axes[0].plot(epochs_ran, history["val_loss"], label="val")
    axes[0].axvline(best_ep, color="green", ls="--", label=f"best epoch ({best_ep})")
    axes[0].set_title("Loss"); axes[0].set_xlabel("epoch"); axes[0].legend()

    axes[1].plot(epochs_ran, history["train_cat_acc"], label="train category")
    axes[1].plot(epochs_ran, history["val_cat_acc"], label="val category")
    axes[1].plot(epochs_ran, history["train_sp_acc"], label="train species")
    axes[1].plot(epochs_ran, history["val_sp_acc"], label="val species")
    axes[1].set_title("Accuracy (train = with augmentation)"); axes[1].set_xlabel("epoch"); axes[1].legend()

    axes[2].plot(epochs_ran, history["lr"])
    axes[2].set_yscale("log"); axes[2].set_title("Learning rate"); axes[2].set_xlabel("epoch")

    plt.tight_layout()
    plt.savefig(WORK_DIR / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.show()


# automatic overfitting / underfitting verdict
cat_gap = (train_cat - val_cat) * 100
sp_gap = (train_sp - val_sp) * 100

print("========== OVERFITTING CHECK ==========")
print(f"Category gap (train - val): {cat_gap:.2f}%")
print(f"Species gap  (train - val): {sp_gap:.2f}%")

if history["val_loss"]:
    best_ep = int(np.argmin(history["val_loss"])) + 1
    last_ep = len(history["val_loss"])
    print(f"Best epoch: {best_ep} / {last_ep} trained")
    if last_ep - best_ep >= 5:
        print(f"-> val loss stopped improving {last_ep - best_ep} epochs ago, the best checkpoint is used so this is fine")

if cat_gap > 5 or sp_gap > 10:
    print("!! Possible OVERFITTING: train is much better than val.")
    print("   try: more dropout / weight decay, stronger augmentation, or more images per species")
elif val_sp < 0.60:
    print("!! Possible UNDERFITTING: val species accuracy is low.")
    print("   try: unfreeze more backbone layers or train longer")
elif val_cat > 0.995 and val_sp > 0.995:
    print("?? Val accuracy is almost 100% - double check the duplicate/leakage cell above, this can be too good to be true.")
else:
    print("OK: no strong sign of overfitting or underfitting.")


t_cat, p_cat, t_sp, p_sp = val_preds
cats = train_ds.categories

# category level
print(classification_report(t_cat, p_cat, target_names=cats, digits=3, zero_division=0))

cm = confusion_matrix(t_cat, p_cat)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=cats, yticklabels=cats)
plt.xlabel("predicted"); plt.ylabel("true"); plt.title("Category confusion matrix")
plt.tight_layout()
plt.savefig(WORK_DIR / "category_confusion_matrix.png", dpi=150)
plt.show()

# species level (label = category/species)
def full_name(c, s):
    return f"{cats[c]}/{train_ds.species_per_category[cats[c]][s]}"

true_names = [full_name(c, s) for c, s in zip(t_cat, t_sp)]
pred_names = [full_name(c, s) for c, s in zip(p_cat, p_sp)]
print(classification_report(true_names, pred_names, digits=3, zero_division=0))


# make sure everything the app needs is saved
for name in ["best_model.pt", "last_checkpoint.pt", "labels.json", "history.json"]:
    p = WORK_DIR / name
    print(f"{name:20s} exists: {p.exists()}" + (f"  ({p.stat().st_size / 1e6:.1f} MB)" if p.exists() else ""))

# to use the model later without training: set TRAIN_MODEL = False and run the cells again,
# or just load it like this:
#   model.load_state_dict(torch.load("/kaggle/working/best_model.pt", map_location=DEVICE))
# On Kaggle press "Save Version" so the files are kept in the notebook output (add it as an input to reuse it).


# ## 9. Streamlit app (chatbot + classifier)


!apt-get update -qq && apt-get install -y zstd lshw pciutils -qq

!curl -fsSL https://ollama.com/install.sh | sh

!pip install -q streamlit ollama pyngrok

import subprocess, time

# start the ollama server in the background (only once)
ollama_process = subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(5)
print("Ollama server started")


!ollama pull llava:13b

%%writefile /kaggle/working/app.py
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

import subprocess, time

# free the port if an old streamlit is still running
try:
    subprocess.run(["fuser", "-k", "8501/tcp"], check=False)
except FileNotFoundError:
    pass
time.sleep(3)

log_file = open("/kaggle/working/streamlit_log.txt", "w")
streamlit_process = subprocess.Popen(
    ["streamlit", "run", "/kaggle/working/app.py",
     "--server.port", "8501",
     "--server.headless", "true"],
    stdout=log_file, stderr=subprocess.STDOUT
)
time.sleep(10)

if streamlit_process.poll() is not None:
    print("Streamlit crashed")
    print(open("/kaggle/working/streamlit_log.txt").read())
else:
    print("Streamlit is working")


import ollama

# loads llava into memory once and keeps it there for 1 hour
r = ollama.chat(model="llava:13b", messages=[{"role": "user", "content": "Say hi in 5 words."}], keep_alive="1h")
print(r["message"]["content"])

# check the PROCESSOR column
!ollama ps

from pyngrok import ngrok
from kaggle_secrets import UserSecretsClient

# token comes from Kaggle -> Add-ons -> Secrets (label: NGROK_TOKEN) so it never sits in the notebook
ngrok.kill()
ngrok.set_auth_token(UserSecretsClient().get_secret("NGROK_TOKEN"))
public_url = ngrok.connect(8501)
print("App is live here:", public_url.public_url)


import os
import torch

# Search your working directory for any .pt files
print("Searching for model files...")
found_files = []
for root, dirs, files in os.walk("/kaggle/working"):
    for file in files:
        if file.endswith(".pt"):
            full_path = os.path.join(root, file)
            found_files.append(full_path)
            print(f"Found: {full_path}")

if not found_files:
    print("No .pt files found! You need to run your training cell first.")
else:
    print("Model file exists!")

from IPython.display import FileLink
FileLink('path/to/your/best_model.pt')

from IPython.display import FileLink
import os

# Find the exact path of best_model.pt
for root, dirs, files in os.walk("/kaggle/working"):
    if "best_model.pt" in files:
        full_path = os.path.join(root, "best_model.pt")
        # Generate the clickable download link
        display(FileLink(full_path))

import os
from IPython.display import HTML, display

# Find the file
model_path = None
for root, dirs, files in os.walk("/kaggle/working"):
    if "best_model.pt" in files:
        model_path = os.path.join(root, "best_model.pt")
        break

if model_path:
    # Read file data and create a direct base64 download link
    import base64
    with open(model_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:file/pt;base64,{b64}" download="best_model.pt" style="font-size: 20px; font-weight: bold; padding: 10px; background-color: #4CAF50; color: white; text-decoration: none; border-radius: 5px;">📥 Click Here to Download best_model.pt</a>'
    display(HTML(href))
else:
    print("best_model.pt not found in /kaggle/working. Make sure training has completed!")
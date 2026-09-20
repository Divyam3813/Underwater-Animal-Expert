# 🐠 Underwater Animal Expert

An end-to-end underwater animal classification and AI assistant project
that combines a custom hierarchical image-classification model with a
Streamlit interface and an optional local Ollama vision chatbot.

The system is designed to identify underwater animals from an uploaded
image or camera photo. It first predicts one of four broad categories:

-   🦀 Crab
-   🐟 Fish
-   🪼 Jellyfish
-   🦈 Shark

It then uses a category-specific species classifier to predict the
species within the selected category.

The Streamlit application additionally provides **Dr. Marina**, an AI
marine-biology assistant powered by a local Ollama vision model. The
chatbot can inspect the uploaded image, answer follow-up questions,
remember recently discussed animals, and use the classifier's prediction
as supporting context.

------------------------------------------------------------------------

## ✨ Features

### 🧠 Hierarchical image classification

Instead of using one flat classifier for every species, the project uses
a two-stage architecture:

``` text
Input Image
     │
     ▼
EfficientNetV2-S Backbone
     │
     ▼
Category Head
     │
     ├── Crab ───────► Crab Species Head
     ├── Fish ───────► Fish Species Head
     ├── Jellyfish ──► Jellyfish Species Head
     └── Shark ──────► Shark Species Head
```

This allows the model to first determine the animal family/category and
then make a more specific species prediction.

### 🖼️ Conservative image preprocessing

The project deliberately keeps the preprocessing close to the original
image so that species-specific visual characteristics are not
unnecessarily destroyed.

The preprocessing pipeline performs:

1.  Resize to `224 × 224`
2.  Mild bilateral filtering
3.  LAB color-space conversion
4.  Very mild sharpening on the luminance (`L`) channel
5.  Conversion back to BGR
6.  Pixel-value clipping

The same preprocessing function is used by training and the Streamlit
application.

### 🔄 Training augmentation

During training, the following augmentations are applied:

-   Random resized crop
-   Random horizontal flip
-   Random rotation
-   Random affine transformation
-   Color jitter
-   Gaussian blur
-   Random erasing
-   ImageNet normalization

Validation images are evaluated without these random augmentations.

### 🧹 Dataset preparation

The notebook automatically:

-   Finds the four category datasets
-   Collects species folders
-   Normalizes species names
-   Removes duplicate/near-duplicate images using an average perceptual
    hash
-   Limits images per species
-   Creates deterministic train/validation splits
-   Uses a fixed random seed
-   Checks for exact train/validation image leakage
-   Builds the final combined dataset

### 📊 Evaluation

The training notebook evaluates:

-   Category accuracy
-   End-to-end species accuracy
-   Classification reports
-   Category confusion matrix
-   Training/validation loss
-   Training/validation category accuracy
-   Training/validation species accuracy

Species accuracy is counted end-to-end: the predicted category must also
be correct.

### 🛡️ Overfitting controls

The training configuration includes:

-   ImageNet-pretrained EfficientNetV2-S backbone
-   Partial backbone freezing
-   Dropout
-   Weight decay
-   Label smoothing
-   Category loss weighting
-   Data augmentation
-   Learning-rate reduction using `ReduceLROnPlateau`
-   Early stopping
-   Gradient clipping
-   Best-model checkpointing
-   Full checkpoint saving after every epoch

### ⚡ Mixed precision

When a CUDA GPU is available, automatic mixed precision is enabled to
reduce GPU memory usage and accelerate training.

### 💾 Resumable training

The notebook saves:

-   `best_model.pt` --- best model weights
-   `last_checkpoint.pt` --- complete training state
-   `labels.json` --- category/species mapping
-   `history.json` --- training history

Training can therefore be resumed instead of starting from scratch.

------------------------------------------------------------------------

# 🏗️ Project Architecture

## 1. Dataset layer

The raw datasets are combined into a common directory structure:

``` text
combined/
├── train/
│   ├── crab/
│   │   ├── species_1/
│   │   ├── species_2/
│   │   └── ...
│   ├── fish/
│   ├── jellyfish/
│   └── shark/
│
└── val/
    ├── crab/
    ├── fish/
    ├── jellyfish/
    └── shark/
```

Each species is represented by its own directory.

The notebook discovers datasets using category-specific directory
keywords rather than requiring one fixed source directory.

------------------------------------------------------------------------

# 🔬 Image Processing Pipeline

The same `classical_pipeline()` is used during training and inference.

``` text
Original Image
      │
      ▼
Resize → 224 × 224
      │
      ▼
Bilateral Filter
      │
      ▼
BGR → LAB
      │
      ▼
Mild Luminance Sharpening
      │
      ▼
LAB → BGR
      │
      ▼
Clip Pixel Values
      │
      ▼
RGB
```

### Why conservative preprocessing?

The project intentionally avoids aggressive operations such as:

-   CLAHE
-   Strong contrast stretching
-   Saturation boosting
-   Morphological processing

The goal is to preserve natural color, texture, shape, and
species-specific markings.

------------------------------------------------------------------------

# 🧠 Model

## EfficientNetV2-S

The backbone is:

``` text
torchvision.models.efficientnet_v2_s
```

The training notebook uses ImageNet-1K pretrained weights.

The final feature representation has a dimension of `1280`.

A global adaptive average pooling layer converts the feature map into a
feature vector.

------------------------------------------------------------------------

## Hierarchical heads

The model contains:

``` text
EfficientNetV2-S
      │
      ▼
Adaptive Average Pooling
      │
      ▼
1280-dimensional feature vector
      │
      ├──────────────► Category Head
      │
      └──────────────► Category-specific Species Heads
```

The category head is:

``` text
1280 → number of categories
```

The species heads are dynamically created:

``` text
1280 → number of crab species
1280 → number of fish species
1280 → number of jellyfish species
1280 → number of shark species
```

The exact number of species is determined from the dataset and saved in
`labels.json`.

------------------------------------------------------------------------

# 🎯 Training Strategy

The model optimizes two losses:

``` text
Total Loss = Category Loss + Species Loss
```

### Category loss

The category classifier uses weighted cross entropy with label
smoothing.

Class weights are calculated from the training-category frequencies so
that larger categories do not dominate the loss.

### Species loss

Each sample is passed through the species head corresponding to its
category.

The implementation processes samples grouped by category so that
species-head computation is batched rather than performed one image at a
time.

------------------------------------------------------------------------

# ⚙️ Training Configuration

The main configuration used in the notebook is:

  Parameter                                            Value
  ------------------------- --------------------------------
  Image size                                     `224 × 224`
  Epochs                                       `100` maximum
  Batch size                                            `32`
  Learning rate                                       `1e-4`
  Weight decay                                        `1e-2`
  Dropout                                              `0.5`
  Early stopping                                     Enabled
  Early stopping patience                        `12` epochs
  Optimizer                                            AdamW
  LR scheduler                             ReduceLROnPlateau
  Backbone                                  EfficientNetV2-S
  Pretrained                                     ImageNet-1K
  Random seed                                           `42`
  AMP                         Enabled when CUDA is available

The model does not necessarily train for all 100 epochs because early
stopping can terminate training when validation loss stops improving.

------------------------------------------------------------------------

# 🧪 End-to-End Evaluation

The project distinguishes between:

### Category accuracy

Whether the broad animal category is correct.

``` text
True category == Predicted category
```

### Species accuracy

A prediction is counted as correct only when both category and species
are correct:

``` text
True category == Predicted category
AND
True species == Predicted species
```

This makes species accuracy a true end-to-end metric rather than a
species-only metric calculated using the known ground-truth category.

------------------------------------------------------------------------

# 🔎 Test-Time Augmentation

The Streamlit classifier performs inference using multiple versions of
the same image:

1.  Original image
2.  Horizontally flipped image
3.  Image rotated by `-10°`
4.  Image rotated by `+10°`

The model predicts probabilities for these variants and averages the
probabilities before selecting the final category/species.

``` text
Image
 ├── Original
 ├── Horizontal Flip
 ├── Rotation -10°
 └── Rotation +10°
          │
          ▼
   Model predictions
          │
          ▼
   Average probabilities
          │
          ▼
 Final category + species
```

This is intended to make inference less sensitive to small changes in
orientation.

------------------------------------------------------------------------

# 📱 Streamlit Application

The application provides two major components.

## 1. Image Classifier

Users can:

-   Upload JPG/JPEG/PNG images
-   Use the device camera
-   View the selected image
-   Get category prediction
-   Get species prediction
-   View confidence values
-   View category probability distribution
-   View the top species guesses
-   See an annotated version of the image

The application automatically uses:

``` text
CUDA → GPU inference
CPU  → CPU inference
```

depending on availability.

------------------------------------------------------------------------

# 💬 Dr. Marina AI Assistant

The application includes an optional chatbot called **Dr. Marina**.

Dr. Marina is designed as a marine-biology assistant specializing in:

-   Fish
-   Sharks
-   Crabs
-   Jellyfish

The system prompt asks the assistant to cover information such as:

-   Scientific name
-   Typical habitat/range
-   Diet
-   Size
-   Whether the animal is dangerous to humans

The chatbot can also answer follow-up questions using conversation
history.

------------------------------------------------------------------------

# 🤖 Ollama Integration

The chatbot runs through a local Ollama model.

The notebook pulls:

``` bash
ollama pull llava:13b
```

The Streamlit application can also recognize other installed models such
as:

``` text
llava:13b
llava:7b
llava
moondream
```

The preferred model can be selected from the Streamlit sidebar.

The classifier does **not** depend on Ollama. If Ollama is unavailable,
image identification still works.

------------------------------------------------------------------------

# 🧠 Chat Memory

The application stores:

``` text
chat_history.json
```

The memory contains:

-   Previous user questions
-   Assistant answers
-   Recently discussed animals

The application remembers up to:

``` text
16 chat messages
```

and up to:

``` text
5 recently discussed animals
```

The conversation can be cleared from the Streamlit sidebar.

------------------------------------------------------------------------

# 🖼️ Classifier + Chatbot Interaction

When a new image is uploaded, the classifier produces a prediction such
as:

``` text
Category: Shark
Category confidence: 96.4%

Species: ...
Species confidence: 91.2%
```

The result is then supplied to Dr. Marina as contextual information.

If confidence is lower, the chatbot is instructed to treat the
classifier prediction more cautiously rather than blindly repeating it.

------------------------------------------------------------------------

# 📁 Recommended Repository Structure

A clean GitHub repository can be organized like this:

``` text
underwater-animal-expert/
│
├── README.md
├── streamlit-app.ipynb
├── app.py
├── best_model.pt
├── labels.json
├── requirements.txt
├── .gitignore
│
├── assets/
│   ├── screenshots/
│   └── examples/
│
└── working/
    ├── history.json
    ├── last_checkpoint.pt
    └── uploaded_images/
```

### Important

`last_checkpoint.pt`, generated chat history, uploaded user images, and
temporary training outputs do not need to be committed unless there is a
specific reason to keep them in the repository.

------------------------------------------------------------------------

# 🚀 Installation

## 1. Clone the repository

``` bash
git clone https://github.com/<your-username>/underwater-animal-expert.git
cd underwater-animal-expert
```

Replace `<your-username>` with your GitHub username.

------------------------------------------------------------------------

## 2. Create a virtual environment

### Windows

``` bash
python -m venv venv
venv\Scripts\activate
```

### Linux/macOS

``` bash
python3 -m venv venv
source venv/bin/activate
```

------------------------------------------------------------------------

## 3. Install Python dependencies

A suitable environment requires the libraries used by the
notebook/application, including:

``` bash
pip install torch torchvision
pip install streamlit
pip install opencv-python
pip install numpy
pip install pandas
pip install pillow
pip install scikit-learn
pip install matplotlib
pip install seaborn
pip install ollama
```

For training on GPU, install the appropriate PyTorch build for the CUDA
version supported by your system.

------------------------------------------------------------------------

# 📦 requirements.txt

A basic application requirements file can contain:

``` text
torch
torchvision
streamlit
opencv-python
numpy
pandas
Pillow
scikit-learn
matplotlib
seaborn
ollama
```

If you only want to run the Streamlit classifier without training, the
training-only dependencies can be minimized further.

------------------------------------------------------------------------

# ▶️ Running the Streamlit Application

Place the following files together:

``` text
app.py
best_model.pt
labels.json
```

Then run:

``` bash
streamlit run app.py
```

Streamlit will provide a local URL, normally similar to:

``` text
http://localhost:8501
```

Open that URL in your browser.

------------------------------------------------------------------------

# 🔐 Model and Label Files

The Streamlit application expects:

``` text
best_model.pt
labels.json
```

in the same directory by default.

You can also specify a custom directory through the environment
variable:

``` bash
MARINE_DIR=/path/to/model/files
```

The application looks for:

``` text
MARINE_DIR/
├── best_model.pt
└── labels.json
```

------------------------------------------------------------------------

# 🧾 labels.json

The training notebook creates `labels.json` so the application does not
need hard-coded species mappings.

Its structure is conceptually:

``` json
{
  "categories": [
    "crab",
    "fish",
    "jellyfish",
    "shark"
  ],
  "species_per_category": {
    "crab": [],
    "fish": [],
    "jellyfish": [],
    "shark": []
  }
}
```

The actual species lists are generated from the training dataset.

------------------------------------------------------------------------

# 🏋️ Training From Scratch

Training is primarily designed around the notebook workflow.

The main switch is:

``` python
TRAIN_MODEL = True
```

For inference only:

``` python
TRAIN_MODEL = False
```

To continue an interrupted training run:

``` python
RESUME = True
```

The notebook automatically searches for the previous checkpoint.

------------------------------------------------------------------------

# 💾 Checkpoints

## `best_model.pt`

Contains the model weights corresponding to the best validation loss
observed during training.

This is the checkpoint used by the Streamlit classifier.

## `last_checkpoint.pt`

Contains the complete training state, including:

-   Current epoch
-   Model state
-   Optimizer state
-   Scheduler state
-   AMP scaler state
-   Training history
-   Best validation loss
-   Early-stopping information
-   Label mappings

This file allows training to resume.

## `history.json`

Stores training curves such as:

``` text
train_loss
val_loss
train_cat_acc
val_cat_acc
train_sp_acc
val_sp_acc
lr
```

------------------------------------------------------------------------

# 🧪 Dataset Preparation Details

The notebook uses four categories:

``` python
CATEGORIES = [
    "crab",
    "fish",
    "jellyfish",
    "shark"
]
```

The data pipeline:

``` text
Raw datasets
     │
     ▼
Automatic dataset discovery
     │
     ▼
Species folder collection
     │
     ▼
Species name normalization
     │
     ▼
Duplicate removal
     │
     ▼
Per-species image cap
     │
     ▼
80/20 train-validation split
     │
     ▼
Combined dataset
```

The validation fraction is:

``` python
VAL_FRACTION = 0.2
```

and the default maximum is:

``` python
MAX_IMAGES_PER_SPECIES = 300
```

------------------------------------------------------------------------

# 🔍 Duplicate Detection

Before splitting the data, the notebook calculates an average hash for
images.

This is used to remove duplicate/near-duplicate frames within a species.

The notebook also performs a separate exact byte-level MD5 check after
splitting to detect images that accidentally appear in both training and
validation sets.

This is important because train/validation leakage can produce
artificially high validation accuracy.

------------------------------------------------------------------------

# 📊 Confusion Matrix

The notebook generates a category-level confusion matrix:

``` text
True Category
     ↓
Predicted Category
```

This makes it possible to inspect which broad animal categories are
being confused with each other.

It is saved as:

``` text
category_confusion_matrix.png
```

------------------------------------------------------------------------

# ⚠️ Important Limitations

This project should be understood as a trained computer-vision system
rather than a guaranteed biological identification tool.

### Dataset limitation

The classifier only supports the categories and species present in its
training dataset.

### Confidence limitation

A high softmax probability does not guarantee biological correctness.

### Image limitation

Performance can change with:

-   Poor lighting
-   Heavy blur
-   Occlusion
-   Unusual viewpoints
-   Underwater color distortion
-   Species not represented in the dataset
-   Images substantially different from the training distribution

### Chatbot limitation

Dr. Marina generates AI responses. Safety-related or biologically
important claims should be independently verified.

### Hardware limitation

The `llava:13b` chatbot can require substantial system/GPU memory.
Smaller Ollama models can be selected when necessary.

------------------------------------------------------------------------

# 🔒 Privacy Considerations

The Ollama chatbot is designed to run locally.

Uploaded images may be saved in the configured `uploaded_images`
directory so that they can be displayed as part of the conversation
history.

If this project is deployed publicly, review the image-storage behavior
and access permissions before using it with sensitive images.

------------------------------------------------------------------------

# 🧩 Technologies Used

  Technology                         Purpose
  ---------------------------------- ---------------------------------------
  Python                             Core programming language
  PyTorch                            Deep-learning framework
  Torchvision                        EfficientNetV2-S and image transforms
  OpenCV                             Image preprocessing
  NumPy                              Numerical operations
  Scikit-learn                       Dataset splitting and evaluation
  Matplotlib                         Visualization
  Seaborn                            Confusion matrix visualization
  Pandas                             Probability visualization
  Streamlit                          Web interface
  Ollama                             Local vision-language model
  LLaVA / compatible Ollama models   AI chatbot vision capability

------------------------------------------------------------------------

# 🧠 Key Machine Learning Concepts Demonstrated

This project demonstrates several practical ML/DL concepts:

-   Transfer learning
-   CNN-based image classification
-   EfficientNetV2
-   Hierarchical classification
-   Multi-head classification
-   Data augmentation
-   Image preprocessing
-   Dataset deduplication
-   Train/validation splitting
-   Class-weighted loss
-   Label smoothing
-   Dropout
-   Weight decay
-   Partial backbone freezing
-   Learning-rate scheduling
-   Early stopping
-   Gradient clipping
-   Mixed-precision training
-   Test-time augmentation
-   Confusion matrices
-   Classification reports
-   Model checkpointing
-   End-to-end evaluation

------------------------------------------------------------------------

# 🔄 Complete Pipeline

``` text
                         ┌─────────────────────┐
                         │    Raw Datasets     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Dataset Discovery   │
                         │ + Deduplication     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Train / Validation  │
                         │       Split         │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Image Preprocessing │
                         │   OpenCV Pipeline   │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ EfficientNetV2-S    │
                         │   Feature Extractor │
                         └──────────┬──────────┘
                                    │
                         ┌──────────┴──────────┐
                         ▼                     ▼
                 ┌───────────────┐     ┌──────────────────┐
                 │ Category Head │     │ Feature Vector   │
                 └───────┬───────┘     └────────┬─────────┘
                         │                      │
                         ▼                      ▼
                Crab / Fish /          Category-specific
                Jellyfish / Shark       Species Head
                         │                      │
                         └──────────┬───────────┘
                                    ▼
                         ┌─────────────────────┐
                         │ Category + Species  │
                         │     Prediction      │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │    Streamlit UI     │
                         └──────────┬──────────┘
                                    │
                      ┌─────────────┴─────────────┐
                      ▼                           ▼
             Classification UI             Dr. Marina
                                            AI Assistant
                                                  │
                                                  ▼
                                           Local Ollama
```

------------------------------------------------------------------------

# 🛠️ Troubleshooting

## `best_model.pt not found`

Make sure the model file exists in the same directory as `app.py`, or
set `MARINE_DIR`.

``` text
app.py
best_model.pt
labels.json
```

------------------------------------------------------------------------

## `labels.json not found`

Run the training/data-preparation notebook first so the label mapping is
generated.

------------------------------------------------------------------------

## Classifier loads but predictions fail

Make sure the following are compatible:

-   `best_model.pt`
-   `labels.json`
-   The `HierarchicalMarineNet` architecture
-   The species ordering in `labels.json`

The application reconstructs the model's species heads from the label
file.

------------------------------------------------------------------------

## Ollama is offline

The image classifier can still operate without Ollama.

To enable Dr. Marina:

``` bash
ollama serve
```

Then ensure at least one compatible model has been pulled, for example:

``` bash
ollama pull llava:13b
```

------------------------------------------------------------------------

## Ollama model uses too much memory

Try a smaller installed model, such as:

``` text
llava:7b
```

or another compatible vision model available through Ollama.

------------------------------------------------------------------------

# 📌 Reproducibility

The training pipeline uses:

``` python
SEED = 42
```

and deterministic dataset shuffling/splitting logic.

However, exact results can still vary depending on:

-   PyTorch version
-   Torchvision version
-   CUDA version
-   GPU hardware
-   Dataset contents
-   Filesystem behavior
-   Training environment

------------------------------------------------------------------------

# 📈 Adding Your Own Evaluation Results

After training, the notebook prints results in the following format:

``` text
========== TRAIN (no augmentation) ==========
Total samples     : ...
Category accuracy : ...
Species accuracy  : ...

========== VALIDATION ==========
Total samples     : ...
Category accuracy : ...
Species accuracy  : ...
```

For a polished GitHub README, replace the placeholders above with the
final metrics from the run you want to showcase.

A recommended section is:

``` text
## 📊 Results

| Split | Category Accuracy | Species Accuracy |
|---|---:|---:|
| Train | XX.XX% | XX.XX% |
| Validation | XX.XX% | XX.XX% |
```

Only add metrics from an actual saved/evaluated run.

------------------------------------------------------------------------

# 🚀 Future Improvements

Possible future extensions include:

-   More underwater animal categories
-   Larger and more diverse datasets
-   Calibration of confidence scores
-   Out-of-distribution detection
-   Better species-level evaluation
-   Per-species precision/recall/F1 analysis
-   Model quantization
-   ONNX/TensorRT inference
-   Docker deployment
-   Cloud deployment
-   Authentication for deployed applications
-   More specialized marine-biology knowledge sources for the chatbot
-   Retrieval-augmented generation for cited biological information
-   Automated experiment tracking

------------------------------------------------------------------------

# 👨‍💻 Project Workflow

The project can be viewed as four connected stages:

### Stage 1 --- Data

``` text
Collect → Clean → Deduplicate → Split
```

### Stage 2 --- Computer Vision

``` text
Preprocess → Augment → EfficientNetV2-S → Hierarchical Prediction
```

### Stage 3 --- Evaluation

``` text
Validate → Confusion Matrix → Classification Report → Save Best Model
```

### Stage 4 --- Deployment

``` text
best_model.pt + labels.json
              │
              ▼
        Streamlit App
          │       │
          │       └──► Ollama / Dr. Marina
          │
          └──────────► Image Classification
```

------------------------------------------------------------------------

# 📜 License

Add your preferred license before publishing the repository.

For example, if you choose the MIT License, add a `LICENSE` file
containing the standard MIT License text.

------------------------------------------------------------------------

# 🙌 Acknowledgements

This project uses open-source technologies including:

-   PyTorch
-   Torchvision
-   OpenCV
-   Scikit-learn
-   Streamlit
-   Ollama

The trained classifier uses EfficientNetV2-S with ImageNet-pretrained
initialization during training.

------------------------------------------------------------------------

## ⭐ If you found this project useful

Feel free to star the repository, explore the notebook, and experiment
with the classifier and local AI assistant.

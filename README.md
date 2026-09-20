# 🌊 Underwater Animal Expert

> **An AI-powered underwater animal classification and identification system built with Deep Learning, Computer Vision, and Generative AI.**

## 🚀 Live Demo

### 🌊 Try Underwater Animal Expert Online

Experience the application directly in your browser:

**[🌊 Open Underwater Animal Expert →](https://underwater-animal-expert.streamlit.app/)**

No installation is required. Simply open the application, upload an underwater animal image, and explore the AI-powered classification and chatbot features.

### 🔗 Live Application

|                                   |                           |
| --------------------------------- | ------------------------- |
| 🌐 **Platform**                   | Streamlit Community Cloud |
| 🤖 **Image Classification**       | ✅ Available               |
| 🐠 **Species Identification**     | ✅ Available               |
| 💬 **AI Chatbot**                 | ✅ Available               |
| 🔐 **API Key Required from User** | ❌ No                      |

> **Note:** The Gemini API credentials used by the chatbot are securely managed through Streamlit Secrets and are not exposed to users.

## 🐠 Overview

**Underwater Animal Expert** is an AI-powered application designed to identify underwater animals from images and provide intelligent information about the detected species.

The system combines:

* 🧠 **Deep Learning** for animal classification
* 👁️ **Computer Vision** for image preprocessing
* 🏷️ **Hierarchical classification** for category and species prediction
* 🤖 **Generative AI** for an interactive chatbot
* 🌐 **Streamlit** for an easy-to-use web interface

Simply upload an underwater animal image, and the system analyzes it to determine the predicted animal species.

The application also includes an AI chatbot that allows users to ask questions about underwater animals and receive contextual information.

---

## ✨ Features

### 🔍 AI Image Classification

Upload an underwater animal image and let the trained deep-learning model identify it.

The model performs:

```text
Input Image
     ↓
Image Preprocessing
     ↓
Feature Extraction
     ↓
Category Prediction
     ↓
Species Prediction
     ↓
Final Result
```

### 🧬 Hierarchical Classification

Instead of treating every species as completely independent, the model follows a hierarchical approach:

```text
                    Underwater Animal
                           │
              ┌────────────┼────────────┐
              ↓            ↓            ↓
            Fish        Mammal        Other
              │            │
       ┌──────┼──────┐     ├──────┐
       ↓      ↓      ↓     ↓      ↓
    Species Species Species Species Species
```

This allows the model to first determine the broader animal category and then identify the specific species.

### 🖼️ Image Processing

The application applies computer-vision preprocessing before classification to improve the quality and consistency of input images.

The pipeline can include operations such as:

* Image resizing
* Color-space processing
* Contrast enhancement
* Noise reduction
* Sharpening
* Normalization

### 🤖 AI Chatbot

The application includes a **Generative AI chatbot** powered by Google's Gemini API.

Users can ask questions such as:

* "What does this animal eat?"
* "Where is this species commonly found?"
* "Is this animal dangerous?"
* "Tell me some interesting facts about it."
* "How does this animal survive underwater?"

The API key is securely stored using **Streamlit Secrets** and is **not exposed to users or committed to the repository**.

### 🌐 Interactive Web Application

The complete system is wrapped inside a Streamlit interface, allowing users to interact with the model directly from a browser without needing to run Python code manually.

---

# 🏗️ System Architecture

```text
                    ┌─────────────────────┐
                    │     User Uploads    │
                    │   Underwater Image  │
                    └──────────┬──────────┘
                               │
                               ↓
                    ┌─────────────────────┐
                    │ Image Preprocessing │
                    │      OpenCV/PIL     │
                    └──────────┬──────────┘
                               │
                               ↓
                    ┌─────────────────────┐
                    │   PyTorch Model     │
                    │ Feature Extraction  │
                    └──────────┬──────────┘
                               │
                               ↓
                  ┌─────────────────────────┐
                  │   Category Prediction   │
                  └────────────┬────────────┘
                               │
                               ↓
                  ┌─────────────────────────┐
                  │   Species Prediction    │
                  └────────────┬────────────┘
                               │
                               ↓
                    ┌─────────────────────┐
                    │   Prediction Result │
                    └──────────┬──────────┘
                               │
                    ┌──────────┴──────────┐
                    ↓                     ↓
             Species Details         AI Chatbot
                                      Gemini API
```

---

# 🧠 Machine Learning Pipeline

The project follows a complete machine-learning workflow:

### 1. Data Collection

Underwater animal images are collected and organized according to their respective categories and species.

### 2. Data Preprocessing

Images are processed before being provided to the neural network.

Typical operations include:

```text
Resize
  ↓
Image Enhancement
  ↓
Normalization
  ↓
Tensor Conversion
```

### 3. Feature Extraction

The deep-learning architecture extracts meaningful visual features from the input image.

These features capture characteristics such as:

* Shape
* Texture
* Color
* Body structure
* Visual patterns

### 4. Hierarchical Prediction

The model performs two levels of classification:

```text
Image
  ↓
Category
  ↓
Species
```

This provides a structured approach to underwater animal recognition.

---

# 📊 Model Performance

The model was evaluated separately on category-level and species-level classification.

### Latest Evaluation

| Metric            |    Training | Validation |
| ----------------- | ----------: | ---------: |
| Category Accuracy | **100.00%** | **99.92%** |
| Species Accuracy  |  **99.89%** | **94.53%** |

> Validation performance is reported separately from training performance to provide a more realistic indication of model generalization.

---

# 🛠️ Tech Stack

| Technology                   | Purpose                           |
| ---------------------------- | --------------------------------- |
| 🐍 Python                    | Core programming language         |
| 🔥 PyTorch                   | Deep learning and model inference |
| 👁️ OpenCV                   | Image processing                  |
| 🖼️ PIL                      | Image handling                    |
| 🔢 NumPy                     | Numerical computation             |
| 📊 Pandas                    | Data processing                   |
| 🤖 Google Gemini             | Generative AI chatbot             |
| 🎨 Streamlit                 | Web application                   |
| 🐙 Git & GitHub              | Version control                   |
| ☁️ Streamlit Community Cloud | Deployment                        |

---

# 📁 Project Structure

```text
Underwater-Animal-Expert/
│
├── app.py
│
├── best_model.pt
│
├── labels.json
│
├── requirements.txt
│
├── render.yaml
│
├── .gitattributes
│
└── underwater-animals-expert (1).py
```

### Important Files

**`app.py`**

Main Streamlit application containing:

* User interface
* Image upload
* Model loading
* Image preprocessing
* Prediction
* AI chatbot integration

**`best_model.pt`**

Trained PyTorch model used for underwater animal classification.

**`labels.json`**

Stores category/species label mappings required to convert model predictions into human-readable names.

**`requirements.txt`**

Contains the Python dependencies required to run the application.

---

# 🚀 Run Locally

## 1. Clone the repository

```bash
git clone https://github.com/Divyam3813/Underwater-Animal-Expert.git
cd Underwater-Animal-Expert
```

## 2. Create a virtual environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Configure the Gemini API key

Create a Streamlit secrets configuration:

```text
.streamlit/
└── secrets.toml
```

Inside `secrets.toml`:

```toml
GOOGLE_API_KEY = "YOUR_GEMINI_API_KEY"
```

⚠️ **Never commit ****`secrets.toml`**** or your API key to GitHub.**

## 5. Start the application

```bash
streamlit run app.py
```

The application will then be available locally in your browser.

---

# ☁️ Deployment

The application can be deployed using **Streamlit Community Cloud**.

Deployment flow:

```text
GitHub Repository
       ↓
Streamlit Community Cloud
       ↓
Install Dependencies
       ↓
Load PyTorch Model
       ↓
Load Streamlit Secrets
       ↓
Launch Application
```

The Gemini API key is configured through Streamlit's **Secrets Management**, keeping it separate from the public source code.

---

# 🔐 Security

API credentials are **not hard-coded into the application**.

The chatbot accesses the Gemini API key through Streamlit Secrets:

```python
st.secrets["GOOGLE_API_KEY"]
```

This means:

* ❌ API key is not stored in `app.py`
* ❌ API key is not committed to GitHub
* ❌ Users do not need to enter the API key
* ✅ API key is stored as a deployment secret

**Never expose your API key in screenshots, GitHub commits, README files, or frontend code.**

---

# 💬 Example Use Cases

### Marine Life Identification

Upload an underwater photograph to identify the animal.

### Educational Tool

Students can use the application to learn about marine species.

### Marine Research Support

The classification system can assist with preliminary identification of underwater organisms.

### AI-Powered Exploration

Users can interact with the chatbot to explore information about the detected species.

---

# 🎯 Future Improvements

Potential improvements include:

* 🔬 Larger and more diverse underwater datasets
* 🌊 Better handling of low-light underwater images
* 🎯 Object detection for images containing multiple animals
* 📹 Real-time video classification
* 📱 Mobile-friendly interface
* 🗺️ Species habitat visualization
* 📈 Confidence visualization
* 🧠 Improved model generalization
* 🤖 More contextual AI responses based on the detected species
* 🌐 Multi-language support

---

# 🧪 Example Workflow

```text
Upload Image
     ↓
┌───────────────────────────┐
│ 🐠 Underwater Animal      │
│       Image               │
└─────────────┬─────────────┘
              ↓
      AI Image Analysis
              ↓
      Category Prediction
              ↓
       Species Prediction
              ↓
   ┌─────────────────────┐
   │ Predicted Species   │
   └──────────┬──────────┘
              ↓
      Ask AI Anything
              ↓
       Gemini Chatbot
```

---

# 🌊 Why This Project?

Underwater environments contain an enormous variety of marine life, but identifying species manually from images can be difficult and time-consuming.

**Underwater Animal Expert** combines computer vision, deep learning, and generative AI into a single interactive platform for exploring underwater biodiversity.

The project demonstrates how multiple AI technologies can work together:

```text
Computer Vision
       +
Deep Learning
       +
Hierarchical Classification
       +
Generative AI
       ↓
Underwater Animal Expert
```

---

# 👨‍💻 Author

### Divyam Jhawar

Electronics & Instrumentation Engineering
Nirma University

Interested in:

* 🤖 Artificial Intelligence
* 🧠 Machine Learning
* 📊 Data Science
* 👁️ Computer Vision
* 🗣️ NLP
* 🤖 Generative AI

---

## ⭐ If you found this project interesting

Consider giving the repository a **star ⭐** and exploring the project!

**Built with Python, PyTorch, Computer Vision, and Generative AI.**

🌊 **Explore. Identify. Learn.**

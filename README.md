https://github.com/user-attachments/assets/9fd68479-1c37-400f-9a85-cfaae6d67d43

# SigLIP2 Multi-Domain & Zero-Shot Image Classification

This project provides a unified Gradio interface that supports both:
- **Multi-Domain Image Classification** using pre-trained specialized models
- **Zero-Shot Image Classification** using Google's SigLIP and SigLIP2 models

It enables users to perform image classification across a wide variety of tasks and also test open-vocabulary classification via a simple web interface.

---

## Features

### 1. Multi-Domain Image Classification

Supports 18+ domains, including:
- Age classification
- Gender classification
- Emotion detection
- Deepfake quality assessment
- Dog breed classification
- Waste classification
- Food classification (Indian/Western)
- Traffic density
- Leaf disease detection (rice)
- Alphabet sign language detection
- Gym workout pose classification
- Bird species classification
- Clip Art 126, Painting 126, Sketch 126
- MNIST and Fashion MNIST
- Multi-source 121 classification

Each model is integrated via its own module for efficient classification using domain-specific architectures.

---

### 2. Zero-Shot Classification (SigLIP & SigLIP2)

Uses two open-vocabulary models:
- `google/siglip-so400m-patch14-384`
- `google/siglip2-so400m-patch14-384`

**How it works**:
- Accepts a user-uploaded image
- Takes a comma-separated list of labels
- Compares prediction probabilities using both SigLIP and SigLIP2 models

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/PRITHIVSAKTHIUR/SigLIP2-MultiDomain-App.git
cd SigLIP2-MultiDomain-App
```

### 2. Install Dependencies

You can install required packages using pip:

```bash
pip install -r requirements.txt
```

Minimal `requirements.txt` might include:
```txt
torch
transformers
gradio
Pillow
```

### 3. Run the App

```bash
python app.py
```

This will launch a Gradio interface in your default web browser.

---

## How to Use

### Multi-Domain Classification Tab
1. Use the sidebar to choose your desired classification task (e.g., Age, Dog Breed, Deepfake).
2. Upload an image.
3. Click "Classify / Predict" to see the result.

### Zero-Shot Classification Tab
1. Upload an image.
2. Enter labels separated by commas (e.g., "cat, dog, horse").
3. Click "Run" to compare SigLIP and SigLIP2 outputs.

---

## Project Structure

```
├── app.py                           # Main Gradio app
├── gender_classification.py        # Domain model modules
├── emotion_classification.py
├── dog_breed.py
├── deepfake_quality.py
├── ...
├── sketch_126.py
└── requirements.txt
```

---

## Models Used

- Domain-specific models (e.g., CNNs, ViTs) hosted locally or via Hugging Face
- SigLIP / SigLIP2 models from Google (used for zero-shot inference)

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

import gradio as gr
from transformers import AutoImageProcessor, SiglipForImageClassification
from transformers.image_utils import load_image
from PIL import Image
import torch

# Load model and processor
model_name = "prithivMLmods/Rice-Leaf-Disease"
model = SiglipForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

def classify_leaf_disease(image):
    """Predicts the disease type in a rice leaf image."""
    image = Image.fromarray(image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()
    
    labels = {
        "0": "Bacterial Blight",
        "1": "Blast",
        "2": "Brown Spot",
        "3": "Healthy",
        "4": "Tungro"
    }
    predictions = {labels[str(i)]: round(probs[i], 3) for i in range(len(probs))}
    
    return predictions

# Create Gradio interface
iface = gr.Interface(
    fn=classify_leaf_disease,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(label="Prediction Scores"),
    title="Rice Leaf Disease Classification 🌾",
    description="Upload an image of a rice leaf to identify if it is healthy or affected by diseases like Bacterial Blight, Blast, Brown Spot, or Tungro."
)

# Launch the app
if __name__ == "__main__":
    iface.launch()
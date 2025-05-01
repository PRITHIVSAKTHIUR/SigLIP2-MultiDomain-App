import gradio as gr
from transformers import AutoImageProcessor, SiglipForImageClassification
from transformers.image_utils import load_image
from PIL import Image
import torch

# Load model and processor
model_name = "prithivMLmods/Mnist-Digits-SigLIP2"
model = SiglipForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

def classify_digit(image):
    """Predicts the digit in the given handwritten digit image."""
    image = Image.fromarray(image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()
    
    labels = {
        "0": "0", "1": "1", "2": "2", "3": "3", "4": "4",
        "5": "5", "6": "6", "7": "7", "8": "8", "9": "9"
    }
    predictions = {labels[str(i)]: round(probs[i], 3) for i in range(len(probs))}
    
    return predictions

# Create Gradio interface
iface = gr.Interface(
    fn=classify_digit,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(label="Prediction Scores"),
    title="MNIST Digit Classification 🔢",
    description="Upload a handwritten digit image (0-9) to recognize it using MNIST-Digits-SigLIP2."
)

# Launch the app
if __name__ == "__main__":
    iface.launch()
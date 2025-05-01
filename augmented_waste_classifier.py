import gradio as gr
from transformers import AutoImageProcessor
from transformers import SiglipForImageClassification
from transformers.image_utils import load_image
from PIL import Image
import torch

# Load model and processor
model_name = "prithivMLmods/Augmented-Waste-Classifier-SigLIP2"
model = SiglipForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

def waste_classification(image):
    """Predicts waste classification for an image."""
    image = Image.fromarray(image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()
    
    labels = {
        "0": "Battery", "1": "Biological", "2": "Cardboard", "3": "Clothes", 
        "4": "Glass", "5": "Metal", "6": "Paper", "7": "Plastic", 
        "8": "Shoes", "9": "Trash"
    }
    predictions = {labels[str(i)]: round(probs[i], 3) for i in range(len(probs))}
    
    return predictions

# Create Gradio interface
iface = gr.Interface(
    fn=waste_classification,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(label="Prediction Scores"),
    title="Augmented Waste Classification",
    description="Upload an image to classify the type of waste."
)

# Launch the app
if __name__ == "__main__":
    iface.launch()

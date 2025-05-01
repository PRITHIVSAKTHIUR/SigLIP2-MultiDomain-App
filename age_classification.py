import gradio as gr
from transformers import AutoImageProcessor
from transformers import SiglipForImageClassification
from transformers.image_utils import load_image
from PIL import Image
import torch

# Load model and processor
model_name = "prithivMLmods/Age-Classification-SigLIP2"
model = SiglipForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

def age_classification(image):
    """Predicts the age group of a person from an image."""
    image = Image.fromarray(image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()
    
    labels = {
        "0": "Child 0-12", 
        "1": "Teenager 13-20", 
        "2": "Adult 21-44", 
        "3": "Middle Age 45-64", 
        "4": "Aged 65+"
    }
    predictions = {labels[str(i)]: round(probs[i], 3) for i in range(len(probs))}
    
    return predictions

# Create Gradio interface
iface = gr.Interface(
    fn=age_classification,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(label="Prediction Scores"),
    title="Age Group Classification",
    description="Upload an image to predict the person's age group."
)

# Launch the app
if __name__ == "__main__":
    iface.launch()

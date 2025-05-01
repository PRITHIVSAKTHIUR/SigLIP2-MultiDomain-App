import gradio as gr
from transformers import AutoImageProcessor
from transformers import SiglipForImageClassification
from PIL import Image
import torch

# Load model and processor
model_name = "prithivMLmods/Gender-Classifier-Mini"
model = SiglipForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

def gender_classification(image):
    """Predicts gender category for an image."""
    image = Image.fromarray(image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()
    
    labels = {"0": "Female ♀", "1": "Male ♂"}
    predictions = {labels[str(i)]: round(probs[i], 3) for i in range(len(probs))}
    
    return predictions

if __name__ == "__main__":
    import gradio as gr
    iface = gr.Interface(
        fn=gender_classification,
        inputs=gr.Image(type="numpy"),
        outputs=gr.Label(label="Prediction Scores"),
        title="Gender Classification",
        description="Upload an image to classify its gender."
    )
    iface.launch()
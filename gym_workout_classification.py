import gradio as gr
from transformers import AutoImageProcessor
from transformers import SiglipForImageClassification
from transformers.image_utils import load_image
from PIL import Image
import torch

# Load model and processor
model_name = "prithivMLmods/Gym-Workout-Classifier-SigLIP2"
model = SiglipForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

def workout_classification(image):
    """Predicts workout exercise classification for an image."""
    image = Image.fromarray(image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()
    
    labels = {
        "0": "barbell biceps curl", "1": "bench press", "2": "chest fly machine", "3": "deadlift",
        "4": "decline bench press", "5": "hammer curl", "6": "hip thrust", "7": "incline bench press",
        "8": "lat pulldown", "9": "lateral raises", "10": "leg extension", "11": "leg raises",
        "12": "plank", "13": "pull up", "14": "push up", "15": "romanian deadlift",
        "16": "russian twist", "17": "shoulder press", "18": "squat", "19": "t bar row",
        "20": "tricep dips", "21": "tricep pushdown"
    }
    predictions = {labels[str(i)]: round(probs[i], 3) for i in range(len(probs))}
    
    return predictions

# Create Gradio interface
iface = gr.Interface(
    fn=workout_classification,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(label="Prediction Scores"),
    title="Gym Workout Classification",
    description="Upload an image to classify the workout exercise."
)

# Launch the app
if __name__ == "__main__":
    iface.launch()
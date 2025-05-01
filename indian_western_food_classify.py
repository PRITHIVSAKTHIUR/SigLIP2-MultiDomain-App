import gradio as gr
from transformers import AutoImageProcessor
from transformers import SiglipForImageClassification
from transformers.image_utils import load_image
from PIL import Image
import torch

# Load model and processor
model_name = "prithivMLmods/Indian-Western-Food-34"
model = SiglipForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

def food_classification(image):
    """Predicts the type of food in an image."""
    image = Image.fromarray(image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()
    
    labels = {
        "0": "Baked Potato", "1": "Crispy Chicken", "2": "Donut", "3": "Fries", 
        "4": "Hot Dog", "5": "Sandwich", "6": "Taco", "7": "Taquito", "8": "Apple Pie", 
        "9": "Burger", "10": "Butter Naan", "11": "Chai", "12": "Chapati", "13": "Cheesecake", 
        "14": "Chicken Curry", "15": "Chole Bhature", "16": "Dal Makhani", "17": "Dhokla", 
        "18": "Fried Rice", "19": "Ice Cream", "20": "Idli", "21": "Jalebi", "22": "Kaathi Rolls", 
        "23": "Kadai Paneer", "24": "Kulfi", "25": "Masala Dosa", "26": "Momos", "27": "Omelette", 
        "28": "Paani Puri", "29": "Pakode", "30": "Pav Bhaji", "31": "Pizza", "32": "Samosa", 
        "33": "Sushi"
    }
    predictions = {labels[str(i)]: round(probs[i], 3) for i in range(len(probs))}
    
    return predictions

# Create Gradio interface
iface = gr.Interface(
    fn=food_classification,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(label="Prediction Scores"),
    title="Indian & Western Food Classification",
    description="Upload a food image to classify it into one of the 34 food types."
)

# Launch the app
if __name__ == "__main__":
    iface.launch()
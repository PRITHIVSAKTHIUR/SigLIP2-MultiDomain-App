import gradio as gr
from transformers import AutoImageProcessor, SiglipForImageClassification
from transformers.image_utils import load_image
from PIL import Image
import torch

# Load model and processor
model_name = "prithivMLmods/Multisource-121-DomainNet"
model = SiglipForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

def multisource_classification(image):
    """Predicts the domain category for an input image."""
    # Convert the input numpy array to a PIL Image and ensure it is in RGB format
    image = Image.fromarray(image).convert("RGB")
    
    # Process the image and convert it to model inputs
    inputs = processor(images=image, return_tensors="pt")
    
    # Get model predictions without gradient calculations
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        # Convert logits to probabilities using softmax
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()
    
    # Mapping from class indices to domain labels
    labels = {
        "0": "barn", "1": "baseball_bat", "2": "basket", "3": "beach", "4": "bear",
        "5": "beard", "6": "bee", "7": "bird", "8": "blueberry", "9": "bowtie",
        "10": "bracelet", "11": "brain", "12": "bread", "13": "broccoli", "14": "bus",
        "15": "butterfly", "16": "circle", "17": "cloud", "18": "cruise_ship", "19": "dolphin",
        "20": "dumbbell", "21": "elephant", "22": "eye", "23": "eyeglasses", "24": "feather",
        "25": "fish", "26": "flower", "27": "foot", "28": "frog", "29": "giraffe",
        "30": "goatee", "31": "golf_club", "32": "grapes", "33": "grass", "34": "guitar",
        "35": "hamburger", "36": "hand", "37": "hat", "38": "headphones", "39": "helicopter",
        "40": "hexagon", "41": "hockey_stick", "42": "horse", "43": "hourglass", "44": "house",
        "45": "ice_cream", "46": "jacket", "47": "ladder", "48": "leg", "49": "lipstick",
        "50": "megaphone", "51": "monkey", "52": "moon", "53": "mushroom", "54": "necklace",
        "55": "owl", "56": "panda", "57": "pear", "58": "peas", "59": "penguin",
        "60": "pig", "61": "pillow", "62": "pineapple", "63": "pizza", "64": "pool",
        "65": "popsicle", "66": "rabbit", "67": "rhinoceros", "68": "rifle", "69": "river",
        "70": "sailboat", "71": "sandwich", "72": "sea_turtle", "73": "shark", "74": "shoe",
        "75": "skyscraper", "76": "snorkel", "77": "snowman", "78": "soccer_ball", "79": "speedboat",
        "80": "spider", "81": "spoon", "82": "square", "83": "squirrel", "84": "stethoscope",
        "85": "strawberry", "86": "streetlight", "87": "submarine", "88": "suitcase", "89": "sun",
        "90": "sweater", "91": "sword", "92": "table", "93": "teapot", "94": "teddy-bear",
        "95": "telephone", "96": "tent", "97": "The_Eiffel_Tower", "98": "The_Great_Wall_of_China",
        "99": "The_Mona_Lisa", "100": "tiger", "101": "toaster", "102": "tooth", "103": "tornado",
        "104": "tractor", "105": "train", "106": "tree", "107": "triangle", "108": "trombone",
        "109": "truck", "110": "trumpet", "111": "umbrella", "112": "vase", "113": "violin",
        "114": "watermelon", "115": "whale", "116": "windmill", "117": "wine_glass", "118": "yoga",
        "119": "zebra", "120": "zigzag"
    }
    
    # Create a dictionary mapping each label to its corresponding probability (rounded)
    predictions = {labels[str(i)]: round(probs[i], 3) for i in range(len(probs))}
    return predictions

# Create Gradio interface
iface = gr.Interface(
    fn=multisource_classification,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(label="Prediction Scores"),
    title="Multisource-121-DomainNet Classification",
    description="Upload an image to classify it into one of 121 domain categories."
)

# Launch the app
if __name__ == "__main__":
    iface.launch()
import gradio as gr
from transformers import AutoImageProcessor, SiglipForImageClassification
from transformers.image_utils import load_image
from PIL import Image
import torch

# Load model and processor
model_name = "prithivMLmods/Clipart-126-DomainNet"
model = SiglipForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

def clipart_classification(image):
    """Predicts the clipart category for an input image."""
    # Convert the input numpy array to a PIL Image and ensure it's in RGB format
    image = Image.fromarray(image).convert("RGB")
    
    # Process the image and prepare it for the model
    inputs = processor(images=image, return_tensors="pt")
    
    # Perform inference without gradient computation
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        # Apply softmax to obtain probabilities for each class
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()
    
    # Mapping from indices to clipart category labels
    labels = {
        "0": "aircraft_carrier", "1": "alarm_clock", "2": "ant", "3": "anvil", "4": "asparagus",
        "5": "axe", "6": "banana", "7": "basket", "8": "bathtub", "9": "bear",
        "10": "bee", "11": "bird", "12": "blackberry", "13": "blueberry", "14": "bottlecap",
        "15": "broccoli", "16": "bus", "17": "butterfly", "18": "cactus", "19": "cake",
        "20": "calculator", "21": "camel", "22": "camera", "23": "candle", "24": "cannon",
        "25": "canoe", "26": "carrot", "27": "castle", "28": "cat", "29": "ceiling_fan",
        "30": "cell_phone", "31": "cello", "32": "chair", "33": "chandelier", "34": "coffee_cup",
        "35": "compass", "36": "computer", "37": "cow", "38": "crab", "39": "crocodile",
        "40": "cruise_ship", "41": "dog", "42": "dolphin", "43": "dragon", "44": "drums",
        "45": "duck", "46": "dumbbell", "47": "elephant", "48": "eyeglasses", "49": "feather",
        "50": "fence", "51": "fish", "52": "flamingo", "53": "flower", "54": "foot",
        "55": "fork", "56": "frog", "57": "giraffe", "58": "goatee", "59": "grapes",
        "60": "guitar", "61": "hammer", "62": "helicopter", "63": "helmet", "64": "horse",
        "65": "kangaroo", "66": "lantern", "67": "laptop", "68": "leaf", "69": "lion",
        "70": "lipstick", "71": "lobster", "72": "microphone", "73": "monkey", "74": "mosquito",
        "75": "mouse", "76": "mug", "77": "mushroom", "78": "onion", "79": "panda",
        "80": "peanut", "81": "pear", "82": "peas", "83": "pencil", "84": "penguin",
        "85": "pig", "86": "pillow", "87": "pineapple", "88": "potato", "89": "power_outlet",
        "90": "purse", "91": "rabbit", "92": "raccoon", "93": "rhinoceros", "94": "rifle",
        "95": "saxophone", "96": "screwdriver", "97": "sea_turtle", "98": "see_saw", "99": "sheep",
        "100": "shoe", "101": "skateboard", "102": "snake", "103": "speedboat", "104": "spider",
        "105": "squirrel", "106": "strawberry", "107": "streetlight", "108": "string_bean",
        "109": "submarine", "110": "swan", "111": "table", "112": "teapot", "113": "teddy-bear",
        "114": "television", "115": "the_Eiffel_Tower", "116": "the_Great_Wall_of_China",
        "117": "tiger", "118": "toe", "119": "train", "120": "truck", "121": "umbrella",
        "122": "vase", "123": "watermelon", "124": "whale", "125": "zebra"
    }
    
    # Create a dictionary mapping each label to its corresponding probability (rounded)
    predictions = {labels[str(i)]: round(probs[i], 3) for i in range(len(probs))}
    return predictions

# Create Gradio interface
iface = gr.Interface(
    fn=clipart_classification,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(label="Prediction Scores"),
    title="Clipart-126-DomainNet Classification",
    description="Upload a clipart image to classify it into one of 126 domain categories."
)

# Launch the app
if __name__ == "__main__":
    iface.launch()
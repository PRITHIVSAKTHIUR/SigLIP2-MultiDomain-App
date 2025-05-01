import gradio as gr
import torch
from transformers import AutoModel, AutoProcessor

from gender_classification import gender_classification
from emotion_classification import emotion_classification
from dog_breed import dog_breed_classification
from deepfake_quality import deepfake_classification
from gym_workout_classification import workout_classification
from augmented_waste_classifier import waste_classification
from age_classification import age_classification
from mnist_digits import classify_digit
from fashion_mnist_cloth import fashion_mnist_classification
from indian_western_food_classify import food_classification
from bird_species import bird_classification
from alphabet_sign_language_detection import sign_language_classification
from rice_leaf_disease import classify_leaf_disease
from traffic_density import traffic_density_classification
from clip_art import clipart_classification
from multisource_121 import multisource_classification
from painting_126 import painting_classification
from sketch_126 import sketch_classification  # New import

# Main classification function for multi-model classification.
def classify(image, model_name):
    if model_name == "gender":
        return gender_classification(image)
    elif model_name == "emotion":
        return emotion_classification(image)
    elif model_name == "dog breed":
        return dog_breed_classification(image)
    elif model_name == "deepfake":
        return deepfake_classification(image)
    elif model_name == "gym workout":
        return workout_classification(image)
    elif model_name == "waste":
        return waste_classification(image)
    elif model_name == "age":
        return age_classification(image)
    elif model_name == "mnist":
        return classify_digit(image)
    elif model_name == "fashion_mnist":
        return fashion_mnist_classification(image)
    elif model_name == "food":
        return food_classification(image)
    elif model_name == "bird":
        return bird_classification(image)
    elif model_name == "leaf disease":
        return classify_leaf_disease(image)
    elif model_name == "sign language":
        return sign_language_classification(image)
    elif model_name == "traffic density":
        return traffic_density_classification(image)
    elif model_name == "clip art":
        return clipart_classification(image)
    elif model_name == "multisource":
        return multisource_classification(image)
    elif model_name == "painting":
        return painting_classification(image)
    elif model_name == "sketch":  # New option
        return sketch_classification(image)
    else:
        return {"Error": "No model selected"}

# Function to update the selected model and button styles.
def select_model(model_name):
    model_variants = {
        "gender": "secondary", "emotion": "secondary", "dog breed": "secondary", "deepfake": "secondary",
        "gym workout": "secondary", "waste": "secondary", "age": "secondary", "mnist": "secondary",
        "fashion_mnist": "secondary", "food": "secondary", "bird": "secondary", "leaf disease": "secondary",
        "sign language": "secondary", "traffic density": "secondary", "clip art": "secondary",
        "multisource": "secondary", "painting": "secondary", "sketch": "secondary"  # New model variant
    }
    model_variants[model_name] = "primary"
    return (model_name, *(gr.update(variant=model_variants[key]) for key in model_variants))

# Zero-Shot Classification Setup (SigLIP models)
sg1_ckpt = "google/siglip-so400m-patch14-384"
siglip1_model = AutoModel.from_pretrained(sg1_ckpt, device_map="cpu").eval()
siglip1_processor = AutoProcessor.from_pretrained(sg1_ckpt)

sg2_ckpt = "google/siglip2-so400m-patch14-384"
siglip2_model = AutoModel.from_pretrained(sg2_ckpt, device_map="cpu").eval()
siglip2_processor = AutoProcessor.from_pretrained(sg2_ckpt)

def postprocess_siglip(sg1_probs, sg2_probs, labels):
    sg1_output = {labels[i]: sg1_probs[0][i].item() for i in range(len(labels))}
    sg2_output = {labels[i]: sg2_probs[0][i].item() for i in range(len(labels))}
    return sg1_output, sg2_output

def siglip_detector(image, texts):
    sg1_inputs = siglip1_processor(
        text=texts, images=image, return_tensors="pt", padding="max_length", max_length=64
    ).to("cpu")
    sg2_inputs = siglip2_processor(
        text=texts, images=image, return_tensors="pt", padding="max_length", max_length=64
    ).to("cpu")
    with torch.no_grad():
        sg1_outputs = siglip1_model(**sg1_inputs)
        sg2_outputs = siglip2_model(**sg2_inputs)
        sg1_logits_per_image = sg1_outputs.logits_per_image
        sg2_logits_per_image = sg2_outputs.logits_per_image
        sg1_probs = torch.sigmoid(sg1_logits_per_image)
        sg2_probs = torch.sigmoid(sg2_logits_per_image)
    return sg1_probs, sg2_probs

def infer(image, candidate_labels):
    candidate_labels = [label.lstrip(" ") for label in candidate_labels.split(",")]
    sg1_probs, sg2_probs = siglip_detector(image, candidate_labels)
    return postprocess_siglip(sg1_probs, sg2_probs, labels=candidate_labels)

# Build the Gradio Interface with two tabs.
with gr.Blocks() as demo:
    gr.Markdown("# Multi-Domain & Zero-Shot Image Classification")
    
    with gr.Tabs():
        # Tab 1: Multi-Model Classification
        with gr.Tab("Multi-Domain Classification"):
            with gr.Sidebar():
                gr.Markdown("# Choose Domain")
                with gr.Row():
                    age_btn = gr.Button("Age Classification", variant="primary")
                    gender_btn = gr.Button("Gender Classification", variant="secondary")
                    emotion_btn = gr.Button("Emotion Classification", variant="secondary")
                    gym_workout_btn = gr.Button("Gym Workout Classification", variant="secondary")
                    dog_breed_btn = gr.Button("Dog Breed Classification", variant="secondary")
                    bird_btn = gr.Button("Bird Species Classification", variant="secondary")
                    waste_btn = gr.Button("Waste Classification", variant="secondary")
                    deepfake_btn = gr.Button("Deepfake Quality Test", variant="secondary")
                    traffic_density_btn = gr.Button("Traffic Density", variant="secondary")
                    sign_language_btn = gr.Button("Alphabet Sign Language", variant="secondary")
                    clip_art_btn = gr.Button("Clip Art 126", variant="secondary")
                    mnist_btn = gr.Button("Digit Classify (0-9)", variant="secondary")
                    fashion_mnist_btn = gr.Button("Fashion MNIST (only cloth)", variant="secondary")
                    food_btn = gr.Button("Indian/Western Food Type", variant="secondary")
                    leaf_disease_btn = gr.Button("Rice Leaf Disease", variant="secondary")
                    multisource_btn = gr.Button("Multi Source 121", variant="secondary")
                    painting_btn = gr.Button("Painting 126", variant="secondary")
                    sketch_btn = gr.Button("Sketch 126", variant="secondary")
                    
                selected_model = gr.State("age")
                gr.Markdown("### Current Model:")
                model_display = gr.Textbox(value="age", interactive=False)
                selected_model.change(lambda m: m, selected_model, model_display)

                buttons = [
                    gender_btn, emotion_btn, dog_breed_btn, deepfake_btn, gym_workout_btn, waste_btn,
                    age_btn, mnist_btn, fashion_mnist_btn, food_btn, bird_btn, leaf_disease_btn,
                    sign_language_btn, traffic_density_btn, clip_art_btn, multisource_btn, painting_btn, sketch_btn  # Include new button
                ]
                model_names = [
                    "gender", "emotion", "dog breed", "deepfake", "gym workout", "waste",
                    "age", "mnist", "fashion_mnist", "food", "bird", "leaf disease",
                    "sign language", "traffic density", "clip art", "multisource", "painting", "sketch"  # New model name
                ]

                for btn, name in zip(buttons, model_names):
                    btn.click(fn=lambda n=name: select_model(n), inputs=[], outputs=[selected_model] + buttons)

            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(type="numpy", label="Upload Image")
                    analyze_btn = gr.Button("Classify / Predict")
                output_label = gr.Label(label="Prediction Scores")
                analyze_btn.click(fn=classify, inputs=[image_input, selected_model], outputs=output_label)
        
        # Tab 2: Zero-Shot Classification (SigLIP)
        with gr.Tab("Zero-Shot Classification"):
            gr.Markdown("## Compare SigLIP 1 and SigLIP 2 on Zero-Shot Classification")
            with gr.Row():
                with gr.Column():
                    zs_image_input = gr.Image(type="pil", label="Upload Image")
                    zs_text_input = gr.Textbox(label="Input a list of labels (comma separated)")
                    zs_run_button = gr.Button("Run")
                with gr.Column():
                    siglip1_output = gr.Label(label="SigLIP 1 Output", num_top_classes=3)
                    siglip2_output = gr.Label(label="SigLIP 2 Output", num_top_classes=3)
            zs_run_button.click(fn=infer, inputs=[zs_image_input, zs_text_input], outputs=[siglip1_output, siglip2_output])

demo.launch()
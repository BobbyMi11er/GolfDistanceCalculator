import math
import requests
from PIL import Image, ImageDraw
import numpy as np
import torch
from transformers import AutoProcessor, Owlv2ForObjectDetection
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from io import BytesIO
import matplotlib.pyplot as plt
import gradio as gr

# Load the model and processor globally
processor = AutoProcessor.from_pretrained("google/owlv2-base-patch16-ensemble")
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")


# Sub-functions
def calculate_distance(H_real, H_pixels, image_width, camera_field_of_view_degrees):
    camera_field_of_view_radians = math.radians(camera_field_of_view_degrees)
    focal_length_pixels = (0.5 * image_width) / math.tan(0.5 * camera_field_of_view_radians)
    distance_meters = (H_real * focal_length_pixels) / H_pixels
    return distance_meters


def plot_box_on_image(image, box, title=""):
    # fig, ax = plt.subplots()
    # ax.imshow(image)
    # x_min, y_min, x_max, y_max = box
    # ax.plot([x_min, x_max, x_max, x_min, x_min],
    #         [y_min, y_min, y_max, y_max, y_min], 'r-')
    # ax.set_title(title)
    # plt.show()

    draw = ImageDraw.Draw(image)
    x_min, y_min, x_max, y_max = box
    draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
    return image




def get_box_with_highest_confidence(boxes, scores):
    max_confidence = max(scores)
    confidence_threshold = 0.75 * max_confidence
    eligible_boxes = [(boxes[i], scores[i]) for i in range(len(scores)) if scores[i] >= confidence_threshold]
    best_box = max(eligible_boxes, key=lambda x: x[0][3] - x[0][1])
    return {
        'height': best_box[0][3] - best_box[0][1],
        'coordinates': best_box[0].tolist(),
        'confidence': best_box[1]
    }


def preprocess_image_for_model(image):
    pixel_values = image.squeeze().numpy()
    unnormalized_image = (pixel_values * np.array(OPENAI_CLIP_STD)[:, None, None]) + np.array(OPENAI_CLIP_MEAN)[:, None, None]
    unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
    unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
    return Image.fromarray(unnormalized_image)


# Main function for Gradio
def process_golf_pin(image, pin_height_meters, camera_fov):
    image = Image.fromarray(image)
    camera_fov = int(camera_fov)
    pin_height_meters = float(pin_height_meters)

    iphone_image_width_pixels, iphone_image_height_pixels = image.size
    texts = [["a full golf pin and its flag"]]
    inputs = processor(text=texts, images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    normalized_image = preprocess_image_for_model(inputs.pixel_values)
    target_sizes = torch.Tensor([normalized_image.size[::-1]])
    results = processor.post_process_object_detection(outputs=outputs, threshold=0.2, target_sizes=target_sizes)

    boxes, scores, labels = results[0]["boxes"], results[0]["scores"], results[0]["labels"]
    if len(boxes) == 0:
        return None, "No golf pin detected."

    best = get_box_with_highest_confidence(boxes, scores)
    normalized_flag_height = best["height"]
    flag_height_pixels = iphone_image_height_pixels * normalized_flag_height / normalized_image.height

    distance = calculate_distance(pin_height_meters, flag_height_pixels, iphone_image_width_pixels, camera_fov)
    image_with_box = plot_box_on_image(normalized_image, best["coordinates"], title=f"Distance: {distance:.2f} meters")

    return image_with_box, f"Distance to pin: {distance:.2f} meters"

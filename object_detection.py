import requests
from PIL import Image
import numpy as np
import torch
from transformers import AutoProcessor, Owlv2ForObjectDetection
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO

# Function to plot red dots on top of an image
def plot_box_on_image(image_url, box, unnormalized_image = ""):
    # Load the image from the URL
    if unnormalized_image == "":
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
    else:
        img = unnormalized_image

    # Display the image using matplotlib
    fig, ax = plt.subplots()
    ax.imshow(img)

    # box = [x_min, y_min, x_max, y_max]
    x_min, y_min, x_max, y_max = box

    # Plot red dots at the four corners of the box
    # Bottom-left (x_min, y_min)
    ax.plot(x_min, y_min, 'ro')
    # Bottom-right (x_max, y_min)
    ax.plot(x_max, y_min, 'ro')
    # Top-right (x_max, y_max)
    ax.plot(x_max, y_max, 'ro')
    # Top-left (x_min, y_max)
    ax.plot(x_min, y_max, 'ro')

    # Set title and show the plot
    ax.set_title(f"Box Coordinates: {box}")
    plt.show()

# Load the processor and model
processor = AutoProcessor.from_pretrained("google/owlv2-base-patch16-ensemble")
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")

# Download and open an image
url = "https://golfdigest.sports.sndimg.com/content/dam/images/golfdigest/fullset/2022/JD1_1689.jpg.rend.hgtvcom.966.644.suffix/1713026356761.jpeg"
url = "https://golfcourse.uga.edu/_resources/images/IMG_2606_1300x700.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# Set up the text queries for object detection
texts = [["a full golf pin and its flag"]]

# Process the input
inputs = processor(text=texts, images=image, return_tensors="pt")

# Forward pass (inference)
with torch.no_grad():
    outputs = model(**inputs)

# Function to get the preprocessed (unnormalized) image
def get_preprocessed_image(pixel_values):
    pixel_values = pixel_values.squeeze().numpy()
    unnormalized_image = (pixel_values * np.array(OPENAI_CLIP_STD)[:, None, None]) + np.array(OPENAI_CLIP_MEAN)[:, None, None]
    unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
    unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
    unnormalized_image = Image.fromarray(unnormalized_image)
    return unnormalized_image

unnormalized_image = get_preprocessed_image(inputs.pixel_values)

# Convert output bounding boxes and class logits to final results
target_sizes = torch.Tensor([unnormalized_image.size[::-1]])
results = processor.post_process_object_detection(
    outputs=outputs, threshold=0.2, target_sizes=target_sizes
)


# Extract and display the results for the first image
i = 0  # Retrieve predictions for the first image and text queries
text = texts[i]
boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

for box, score, label in zip(boxes, scores, labels):
    box = [round(i, 2) for i in box.tolist()]
    print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")
    plot_box_on_image(url, box, unnormalized_image)


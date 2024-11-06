import math
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

import numpy as np

def get_box_with_highest_confidence(boxes, scores):
    for box in boxes:
        box = [round(i, 2) for i in box.tolist()]

    n = len(boxes)
    
    # Create a flag array to mark boxes that have already been considered
    considered = [False] * n

    # Iterate through all boxes to handle merging of similar boxes
    for i in range(n):
        if considered[i]:
            continue
        for j in range(i + 1, n):
            # Check if boxes are within 10 pixels of each other
            if (torch.abs(boxes[i][0] - boxes[j][0]) <= 10 and torch.abs(boxes[i][1] - boxes[j][1]) <= 10) or \
               (torch.abs(boxes[i][2] - boxes[j][2]) <= 10 and torch.abs(boxes[i][3] - boxes[j][3]) <= 10):
                
                # Merge boxes by updating coordinates to cover the larger box and max score
                # Use torch operations instead of lists
                boxes[i][0] = torch.min(boxes[i][0], boxes[j][0])  # x_min
                boxes[i][1] = torch.min(boxes[i][1], boxes[j][1])  # y_min
                boxes[i][2] = torch.max(boxes[i][2], boxes[j][2])  # x_max
                boxes[i][3] = torch.max(boxes[i][3], boxes[j][3])  # y_max
                
                # Update the score to the max of the two scores
                scores[i] = max(scores[i], scores[j])
                considered[j] = True  # Mark box j as considered

    # Find the box with the highest confidence score
    max_confidence = max(scores)

    # Filter out boxes that have confidence within 25% of the highest confidence
    confidence_threshold = 0.75 * max_confidence
    eligible_boxes = [
        (boxes[i], scores[i]) for i in range(n) if scores[i] >= confidence_threshold
    ]

    # Find the box with the greatest height among eligible boxes
    best_box = max(eligible_boxes, key=lambda x: x[0][3] - x[0][1])  # max based on height (y_max - y_min)

    # Return a dictionary with the height and the box's coordinates
    return {
        'height': round((best_box[0][3] - best_box[0][1]).item(), 5),  # Convert tensor to scalar
        'coordinates': best_box[0].tolist(),  # Convert tensor to list
        'confidence': best_box[1]
    }

def calculate_distance(H_real, H_pixels, image_width, camera_field_of_view_degrees):
    """
    Calculate the distance to an object based on its real-world height, height in pixels, 
    the camera's focal length, and the camera's height from the ground.

    Parameters:
    H_real (float): The real-world height of the object (in meters or the same unit as focal length).
    H_pixels (float): The height of the object in pixels.
    focal_length (float): The focal length of the camera (in the same unit as H_real).
    camera_height (float): The height of the camera from the ground (in meters).

    Returns:
    float: The straight-line distance to the object.
    """

    # Step 1: Calculate the distance D (assuming the camera is at the same level as the object)
    # focal len in pixels = (1/2 (width of img in pixels)) / tan((1/2) * (field of view of camera in degrees)) = 2224.8 for iphone 12
    # use this as focal length, gives distance in meters

    camera_field_of_view_radians = math.radians(camera_field_of_view_degrees)
    print(camera_field_of_view_radians)
    focal_length_pixels = (0.5 * image_width) / math.tan(0.5 * camera_field_of_view_radians) # unit is pixels
    print("focal length pixels: ", focal_length_pixels) # NOTE: THIS ISN"T THE 2224.8 THAT WE GOT LAST WEEK
    distance_meters = (H_real * focal_length_pixels) / H_pixels # unit is meters

    # print("D actual:", D_actual)
    return distance_meters

# Function to plot red dots on top of an image
def plot_box_on_image(image_url, box, title = "", normalized_image = ""):
    # Load the image from the URL
    if normalized_image == "":
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
    else:
        img = normalized_image

    # Display the image using matplotlib
    fig, ax = plt.subplots()
    ax.imshow(img)

    # box = [x_min, y_min, x_max, y_max]
    x_min, y_min, x_max, y_max = box

    # print(x_min, y_min, x_max, y_max)

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
    if title == "":
        ax.set_title(f"Box Coordinates: {box}")
    else:
        ax.set_title(f"{title}")
    plt.show()
    
# Load the processor and model
processor = AutoProcessor.from_pretrained("google/owlv2-base-patch16-ensemble")
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")

# Download and open an image
# driving range picture
# url = "https://golfdigest.sports.sndimg.com/content/dam/images/golfdigest/fullset/2022/JD1_1689.jpg.rend.hgtvcom.966.644.suffix/1713026356761.jpeg"
# Masters picture
# url = "https://golfcourse.uga.edu/_resources/images/IMG_2606_1300x700.jpg"
# url = "https://golfdigest.sports.sndimg.com/content/dam/images/golfdigest/fullset/2017/01/31/5890dd463fb2ecb667fbb08b_Riviera-No.6.jpg.rend.hgtvcom.616.308.suffix/1573304836942.jpeg"

# image = Image.open(requests.get(url, stream=True).raw)

# image = Image.open(r"C:\Users\bobby\Downloads\IMG_2739.jpg")
image = Image.open(r"C:\Users\bobby\Downloads\IMG_2739 (2).jpg")

# image = image.convert('RGB')
image = image.rotate(-90, expand=True)
image.show()
iphone_image_width_pixels, iphone_image_height_pixels = image.size

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

normalized_image = get_preprocessed_image(inputs.pixel_values)

# Convert output bounding boxes and class logits to final results
target_sizes = torch.Tensor([normalized_image.size[::-1]])
results = processor.post_process_object_detection(
    outputs=outputs, threshold=0.2, target_sizes=target_sizes
)


# Extract and display the results for the first image
i = 0  # Retrieve predictions for the first image and text queries
text = texts[i]
boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

for box, score, label in zip(boxes, scores, labels):
    box = [round(i, 2) for i in box.tolist()]
    # print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")
    tempTitle = "Confidence: " + str(round(score.item(), 3))
    # plot_box_on_image("", box, tempTitle, unnormalized_image)

if len(boxes) > 1:
    best = get_box_with_highest_confidence(boxes, scores)
    # print(best["coordinates"])
    plot_box_on_image("", [round(i, 2) for i in best['coordinates']], normalized_image=normalized_image, title="Chosen with height in pixels: " + str(best["height"]))
else:
    print("No pin detected")

#218.617 pixels


focal_length_meters = 0.026
pin_height_meters = 2.1336 # 7 feet
# pin_height_meters = 1.8
my_height_meters = 1.77800 # 5'8"

normalized_flag_height = best["height"]
iphone_flag_height = iphone_image_height_pixels * normalized_flag_height / normalized_image.height
iphone_12_field_of_view_degrees = 120


print(f"original pixel height: {best["height"]}")
print(f"Height in iphone pixels: {iphone_flag_height}")
print("iphone image width pixels: ", iphone_image_width_pixels)

print("Distance from pin: " + str(calculate_distance(pin_height_meters, iphone_flag_height, iphone_image_width_pixels, iphone_12_field_of_view_degrees)))

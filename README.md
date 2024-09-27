### Project for ES 3890 - AI Studio

### Golf Pin Distance Estimator
This is our [official project proposal](https://docs.google.com/document/d/15qJxXh58OIwyaxVQbhTdHUQX_EzVNE_wegkdBal8fZc/edit?usp=sharing)

Essentially, we are looking to use a huggingface object detection model to detect golf pins from an image.
From this data, we can calculate the height in pixels and transform that into a real-world distance from the pin based on golf pins being a fixed height alongside variables such as the focal length of the camera being used and the height of the camera off the ground.

##### Usage instructions
To run the program, first create a virtual environment with `bin/virtualenv testing`
To install run `pip install -r requirements.txt`
To run the program, do `python object_detection.py`
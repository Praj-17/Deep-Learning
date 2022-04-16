
import io
from google.cloud import vision
from google.cloud.vision_v1 import types

def landmark_detection(path):
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = types.Image(content=content)

    response = client.label_detection(image=image)
    labels = response.label_annotations
    print('Labels: ')

    for label in labels:
        print(label.description)
landmark_detection('tajmahal.jfif')


import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as preprocess_vgg16
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input as preprocess_resnet50
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input as preprocess_inceptionv3
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as preprocess_mobilenetv2
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import json
available_models = ['mobilenetv2', 'vgg16', 'resnet50', 'inceptionv3']
class ImageEmbeddings:
    def __init__(self, model_name):
        self.model_name = model_name
        self.available_models = ['mobilenetv2', 'vgg16', 'resnet50', 'inceptionv3']
        self.model = self._load_model()

    def _load_model(self):
        if self.model_name in available_models:
            if self.model_name == 'vgg16':
                model = VGG16(weights='imagenet')
                self.preprocess_fn = preprocess_vgg16
            elif self.model_name == 'resnet50':
                model = ResNet50(weights='imagenet')
                self.preprocess_fn = preprocess_resnet50
            elif self.model_name == 'inceptionv3':
                model = InceptionV3(weights='imagenet')
                self.preprocess_fn = preprocess_inceptionv3
            elif self.model_name == 'mobilenetv2':
                model = MobileNetV2(weights='imagenet')
                self.preprocess_fn = preprocess_mobilenetv2
        else:
            raise ValueError('Invalid model name. Supported models: vgg16, resnet50, inceptionv3, mobilenetv2')
        return model

    def convert_to_embeddings(self, img_path):
        # Load and preprocess the image
        print(self.model.input_shape)
        img = image.load_img(img_path, target_size=(self.model.input_shape[1], self.model.input_shape[2]))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = self.preprocess_fn(x)

        # Get the embeddings/features from the pre-trained model
        embeddings = self.model.predict(x)

        # The embeddings will be a 4D tensor, you can reshape it to 1D if needed
        embeddings = embeddings.flatten()

        return embeddings
    
def GenerateEmbeddings(img_path):
    Embeddings_dict = {}
     #Custom JSON encoder to handle NumPy arrays
    class NumpyArrayEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()  # Convert NumPy array to Python list
            return json.JSONEncoder.default(self, obj)
    if os.path.exists(img_path):
        for model in available_models:
            embeddings = ImageEmbeddings(model)
            embeddings = embeddings.convert_to_embeddings(img_path)
            Embeddings_dict[model] = embeddings
        
        # Dump the dictionary to a JSON file
        filename = os.path.join(os.path.dirname(img_path),'embeddings.json' )
        json_str = json.dumps(Embeddings_dict, cls=NumpyArrayEncoder)
        with open(filename, 'w') as file:
            file.write(json_str)
    else:
        raise ValueError(f"Could not find the image {img_path}")
    return Embeddings_dict, filename





if __name__ == '__main__':
    # # Create an instance of the ImageEmbeddings class for VGG16 model
    # embeddings_vgg16 = ImageEmbeddings('vgg16')

    # # Convert image to embeddings using VGG16
    img_path = 'C:\Prajwal\Generative Design\sample_data_af\S00UBY6-W1.png'
    # vgg16_embeddings = embeddings_vgg16.convert_to_embeddings(img_path)
    # print("VGG16 embeddings:", vgg16_embeddings)

    # # Create an instance of the ImageEmbeddings class for ResNet50 model
    # embeddings_resnet50 = ImageEmbeddings('resnet50')

    # # Convert image to embeddings using ResNet50
    # resnet50_embeddings = embeddings_resnet50.convert_to_embeddings(img_path)
    # print("ResNet50 embeddings:", resnet50_embeddings)

    # # Create an instance of the ImageEmbeddings class for InceptionV3 model
    # embeddings_inceptionv3 = ImageEmbeddings('inceptionv3')

    # # Convert image to embeddings using InceptionV3
    # inceptionv3_embeddings = embeddings_inceptionv3.convert_to_embeddings(img_path)
    # print("InceptionV3 embeddings:", inceptionv3_embeddings)

    # # Create an instance of the ImageEmbeddings class for MobileNetV2 model
    # embeddings_mobilenetv2 = ImageEmbeddings('mobilenetv2')

    # # Convert image to embeddings using MobileNetV2
    # mobilenetv2_embeddings = embeddings_mobilenetv2.convert_to_embeddings(img_path)
    # print("MobileNetV2 embeddings:", mobilenetv2_embeddings)

    print(GenerateEmbeddings(img_path))

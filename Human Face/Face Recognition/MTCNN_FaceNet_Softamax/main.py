from DataBuilder.data_builder import Data_builder
from DataBuilder.augmentation import Augmentation
import os



for person in os.listdir("images"):
    print(person)
    data_builder = Data_builder(person)
    Augmentation(f"{data_builder.path}\\",f"{data_builder.path}\\","aug",15)
    
   


import os
import random

directory = "E:\CODING PLAYGROUND\CODE\Deep Leaning\YOLO\YOLO V4\darknet\cov_data\Covid"
images = []
for imgname in os.listdir(directory):
        split= os.path.splitext(imgname)
        file_extension = split[1]
        if file_extension!= ".txt":
            img_path  = str(f"{directory}\\{imgname}")
            images.append(img_path)
            
testing = []
for i in range(62):
    choice = random.choice(images)
    testing.append(choice)
    images.remove(choice)

with open("training.txt","w") as f:
         f.write(f"{[i for i in images]}\n")
with open("testing.txt","w") as f:
         f.write(f"{[i for i in testing]}\n")
     
import os
import shutil
def imgs_to_folders(folder):
    for img in os.listdir(folder):
        img_split = img.split('.')
        name= img_split[0]
        os.mkdir(folder+ name)
        shutil.move(folder + '/' + img, folder + name)
        print("{} moved to {}".format(folder+ img, name))
    print("All images moved to respective folders")
imgs_to_folders('database/')
    
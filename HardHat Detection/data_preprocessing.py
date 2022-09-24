import os 
#importing the required package
from PIL import Image

def png_to_img(pick_directory,save_directory):
    #open image in png format
    for image in os.listdir(pick_directory):
        try:
            print("Image Found",f'{pick_directory}\\{image}.png' )
            image = image.split('.')[0]
            img_png = Image.open(f'{pick_directory}\\{image}.png')
            #The image object is used to save the image in jpg format
            img_png.save(f'{save_directory}\\{image}.jpg')
            os.remove(f'{pick_directory}\\{image}.png')
        except Exception as e:
            print("Exception: ", e)
            print("Error for:",f'{save_directory}\\{image}.png')
 
png_to_img('E:\DATA SETS\Hard-Hat Data\HardHat_Test_Images\Images',
           'E:\DATA SETS\Hard-Hat Data Processed\HardHat_Test_Images\images')       
        

            
            
        
# Importing Image class from PIL module
from PIL import Image
 
# Opens a image in RGB mode
im = Image.open(r"E:\\CODING PLAYGROUND\\CODE\\Deep Leaning\\Human Face\\Gallery\\eye1.jpeg")
 
# Size of the image in pixels (size of original image)
# (This is not mandatory)
width, height = im.size
 
# Setting the points for cropped image

# Cropped image of above dimension
# (It will not change original image)

im = im.resize((height*2, width*2))
# Shows the image in image viewer
im.save("eye3.jpeg")
im.show()
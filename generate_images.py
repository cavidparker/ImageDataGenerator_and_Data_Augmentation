# $ python generate_images.py --image man.jpg --output generated_dataset/man


from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np 
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "path to the input image")
ap.add_argument("-o", "--output",required = True, help = "path to the output directory to store augmentation examples")
ap.add_argument("-t", "--total", type=int, default = 100, help = "# of training sample to generate")
args = vars(ap.parse_args())



print("loading example images ...")
image = load_img(args["image"])
image = img_to_array(image)
image = np.expand_dims(image, axis = 0)



aug = ImageDataGenerator(
	rotation_range = 30,
	zoom_range = 0.15,
	width_shift_range = 0.2,
	height_shift_range = 0.2,
	shear_range = 0.15,
	horizontal_flip = True,
	fill_mode = "nearest")
total = 0	


print(" generating images...")
imageGen = aug.flow(image,batch_size = 1, save_to_dir = args["output"],
	save_prefix = "image", save_format = "jpg")

for image in imageGen:

	total += 1

	if total == args["total"]:
		break







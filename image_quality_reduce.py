import cv2
import os

input_folder = "input_images"
output_folder = "Reduce_image_quality"
quality = 5  # adjust the quality value as needed

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop through all files in the input folder
for file_name in os.listdir(input_folder):
    # Check if the file is an image
    if file_name.endswith(".jpg") or file_name.endswith(".jpeg") or file_name.endswith(".png"):
        # Read the image file
        img = cv2.imread(os.path.join(input_folder, file_name))

        # Reduce the quality of the image and save it to the output folder
        cv2.imwrite(os.path.join(output_folder, file_name),
                    img, [cv2.IMWRITE_JPEG_QUALITY, quality])

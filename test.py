import os
import cv2
import numpy as np


def augment_image(image_path, output_folder):
    image = cv2.imread(image_path)

    # Flipping Images
    flip_horizontally = cv2.flip(image, 1)  # 1 for horizontal flip
    # flip_vertically = cv2.flip(image, 0)  # 0 for vertical flip

    # Rotating Images
    height, width = image.shape[:2]
    degree = 30
    rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), degree, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

    # Changing Image Brightness and Contrast
    alpha = 1  # Contrast control (1.0-3.0)
    beta = 50  # Brightness control (0-100)
    adjusted_image = cv2.addWeighted(
        image, alpha, np.zeros(image.shape, image.dtype), 0, beta)

    # Adding Gaussian Noise
    mean = 0
    variance = 0.1
    sigma = np.sqrt(variance)
    noise = np.zeros(image.shape, image.dtype)
    cv2.randn(noise, mean, sigma)
    noisy_image = cv2.add(image, noise)

    # Adding Salt and Pepper Noise
    amount = 0.1
    salt_vs_pepper = 0.5
    num_salt = np.ceil(amount * image.size * salt_vs_pepper)
    num_pepper = np.ceil(amount * image.size * (1.0 - salt_vs_pepper))
    salt_coords = [np.random.randint(0, i-1, int(num_salt))
                   for i in image.shape]
    pepper_coords = [np.random.randint(
        0, i-1, int(num_pepper)) for i in image.shape]
    salt_pepper_image = image.copy()
    salt_pepper_image[tuple(salt_coords)] = 255
    salt_pepper_image[tuple(pepper_coords)] = 0

    # Adding Speckle Noise
    noise = np.zeros(image.shape, image.dtype)
    cv2.randn(noise, mean, sigma)
    speckle_noise = image + image * noise

    # Adding Poisson Noise
    poisson_noise = np.random.poisson(image)

    # Adding Gaussian Blur
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

    # Save augmented images to output folder
    filename = os.path.splitext(os.path.basename(image_path))[0]
    cv2.imwrite(os.path.join(output_folder, filename +
                '_flip_horizontally.jpg'), flip_horizontally)
    # cv2.imwrite(os.path.join(output_folder, filename +
    #             '_flip_vertically.jpg'), flip_vertically)
    cv2.imwrite(os.path.join(output_folder, filename +
                '_rotated.jpg'), rotated_image)
    cv2.imwrite(os.path.join(output_folder, filename +
                '_adjusted.jpg'), adjusted_image)
    cv2.imwrite(os.path.join(output_folder,
                filename + '_noisy.jpg'), noisy_image)
    cv2.imwrite(os.path.join(output_folder, filename +
                '_salt_pepper.jpg'), salt_pepper_image)
    cv2.imwrite(os.path.join(output_folder, filename +
                '_speckle_noise.jpg'), speckle_noise)
    cv2.imwrite(os.path.join(output_folder, filename +
                '_poisson_noise.jpg'), poisson_noise)
    cv2.imwrite(os.path.join(output_folder, filename +
                '_blurred_image.jpg'), blurred_image)


input_folder = 'input_images'
output_folder = 'output_images'

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop over all images in input folder and apply data augmentation
for filename in os.listdir(input_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(input_folder, filename)
        augment_image(image_path, output_folder)

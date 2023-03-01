import os
import cv2
import numpy as np
from flask import Flask, jsonify, request

app = Flask(__name__)


@app.route('/augment', methods=['POST'])
def augment_image_api():
    # Get the input image file from the request
    file = request.files['image']
    image_path = os.path.join('input_images', file.filename)
    file.save(image_path)

    # Call the augment_image function to apply data augmentation
    output_folder = 'output_images'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    augment_image(image_path, output_folder)

    # Get the output image file path
    filename = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_folder, filename + '_blurred_image.jpg')

    # Return the output image file path in the response
    response = {'output_image_path': output_path}
    return jsonify(response)


def augment_image(image_path, output_folder):
    image = cv2.imread(image_path)

    # Flipping Images
    flip_horizontally = cv2.flip(image, 1)  # 1 for horizontal flip

    # Rotating Images
    height, width = image.shape[:2]
    degree = 30
    rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), degree, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

    # Adding Gaussian Blur
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

    # Save augmented images to output folder
    filename = os.path.splitext(os.path.basename(image_path))[0]
    cv2.imwrite(os.path.join(output_folder, filename +
                '_flip_horizontally.jpg'), flip_horizontally)
    cv2.imwrite(os.path.join(output_folder, filename +
                '_rotated.jpg'), rotated_image)
    cv2.imwrite(os.path.join(output_folder, filename +
                '_blurred_image.jpg'), blurred_image)


if __name__ == '__main__':
    app.run()

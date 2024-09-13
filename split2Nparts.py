#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 9/12/24
# @Author  : luoolu
# @Github  : https://luoolu.github.io
# @Software: PyCharm
# @File    : split2Nparts.py
'''
only use json to devide
'''
import json
import cv2
import os


# Function to split an image into 4 equal parts
def split_image(image_path, output_dir):
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    # Define the coordinates for the 4 sub-images
    coords = [
        (0, 0, width // 2, height // 2),  # Top-left
        (width // 2, 0, width, height // 2),  # Top-right
        (0, height // 2, width // 2, height),  # Bottom-left
        (width // 2, height // 2, width, height)  # Bottom-right
    ]

    sub_images = []
    for i, (x1, y1, x2, y2) in enumerate(coords):
        sub_img = image[y1:y2, x1:x2]
        sub_img_path = os.path.join(output_dir, f'sub_image_{i}.jpg')
        cv2.imwrite(sub_img_path, sub_img)
        sub_images.append((sub_img_path, x1, y1, x2 - x1, y2 - y1))  # Save width and height for each sub-image

    return sub_images


# Function to adjust the annotations for each sub-image
def adjust_annotations(json_path, sub_images, output_dir):
    with open(json_path, 'r') as f:
        annotations = json.load(f)

    shapes = annotations['shapes']

    for i, (sub_img_path, x1, y1, sub_img_width, sub_img_height) in enumerate(sub_images):
        new_shapes = []

        for shape in shapes:
            new_points = []
            for point in shape['points']:
                px, py = point
                if x1 <= px < x1 + sub_img_width and y1 <= py < y1 + sub_img_height:
                    new_px = px - x1
                    new_py = py - y1
                    new_points.append([new_px, new_py])

            if new_points:
                new_shape = shape.copy()
                new_shape['points'] = new_points
                new_shapes.append(new_shape)

        # Update metadata: imagePath, imageWidth, imageHeight
        new_annotations = annotations.copy()
        new_annotations['shapes'] = new_shapes
        new_annotations['imagePath'] = os.path.basename(sub_img_path)
        new_annotations['imageWidth'] = sub_img_width
        new_annotations['imageHeight'] = sub_img_height

        output_json_path = os.path.join(output_dir, f'sub_image_{i}.json')

        with open(output_json_path, 'w') as f:
            json.dump(new_annotations, f, indent=4)


# Main function to process the image and annotations
def process_image_and_annotations(image_path, json_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Split the image
    sub_images = split_image(image_path, output_dir)

    # Adjust and save the new JSON annotations
    adjust_annotations(json_path, sub_images, output_dir)


# Example usage
image_path = 'Data/00000.jpg'
json_path = 'Data/renxi1-2504.32_m184_0.json'
output_dir = 'Data/output_sub_images'

process_image_and_annotations(image_path, json_path, output_dir)

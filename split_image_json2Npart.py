#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 9/12/24
# @Author  : luoolu
# @Github  : https://luoolu.github.io
# @Software: PyCharm
# @File    : split_image_json2Npart.py
'''
annotation to instance mask then devide 4 parts,back to json annotation
'''
import json
import cv2
import numpy as np
import os
import copy


# Function to convert polygon annotations to instance mask
def polygons_to_instance_mask(img_shape, polygons):
    mask = np.zeros(img_shape[:2], dtype=np.int32)
    for idx, polygon in enumerate(polygons):
        points = np.array(polygon['points'], dtype=np.int32)
        cv2.fillPoly(mask, [points], idx + 1)  # Labels start from 1
    return mask


# Function to split the image and mask into 4 parts
def split_image_and_mask(image_path, mask, output_dir):
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    coords = [
        (0, 0, width // 2, height // 2),         # Top-left
        (width // 2, 0, width, height // 2),     # Top-right
        (0, height // 2, width // 2, height),    # Bottom-left
        (width // 2, height // 2, width, height) # Bottom-right
    ]

    sub_images = []
    sub_masks = []

    for i, (x1, y1, x2, y2) in enumerate(coords):
        sub_img = image[y1:y2, x1:x2]
        sub_mask = mask[y1:y2, x1:x2]

        sub_img_path = os.path.join(output_dir, f'sub_image_{i}.jpg')
        cv2.imwrite(sub_img_path, sub_img)

        sub_images.append((sub_img_path, x2 - x1, y2 - y1))
        sub_masks.append(sub_mask)

    return sub_images, sub_masks


# Function to find contours from instance mask and convert to polygons
def instance_mask_to_polygons(mask):
    instance_ids = np.unique(mask)
    instance_ids = instance_ids[instance_ids != 0]  # Exclude background
    polygons = []

    for instance_id in instance_ids:
        instance_mask = np.uint8(mask == instance_id)
        contours, _ = cv2.findContours(instance_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            contour = contour.squeeze(1)  # Remove redundant dimension
            if len(contour) < 3:
                continue

            polygon = [[int(pt[0]), int(pt[1])] for pt in contour]
            polygons.append((instance_id, polygon))

    return polygons


# Function to update the annotations after splitting
def update_annotations(sub_images, sub_masks, output_dir, original_json_path):
    with open(original_json_path, 'r') as f:
        annotations = json.load(f)

    original_shapes = annotations['shapes']

    for i, (sub_img_path, sub_img_width, sub_img_height) in enumerate(sub_images):
        mask = sub_masks[i]

        # Use deepcopy to prevent modifying the original annotations
        new_annotations = copy.deepcopy(annotations)

        # Find contours in the mask and convert to polygon points
        polygons = instance_mask_to_polygons(mask)

        # Update imagePath, imageWidth, and imageHeight
        new_annotations['imagePath'] = os.path.basename(sub_img_path)
        new_annotations['imageWidth'] = sub_img_width
        new_annotations['imageHeight'] = sub_img_height

        # Clear the old shapes
        new_annotations['shapes'] = []

        for instance_id, polygon in polygons:
            # Get the label and other properties from the original annotations
            original_shape = original_shapes[instance_id - 1]
            shape = {
                "label": original_shape['label'],
                "line_color": original_shape.get('line_color', None),
                "fill_color": original_shape.get('fill_color', None),
                "points": polygon,
                "shape_type": original_shape.get('shape_type', 'polygon'),
                "flags": original_shape.get('flags', {})
            }
            new_annotations['shapes'].append(shape)

        output_json_path = os.path.join(output_dir, f'sub_image_{i}.json')
        with open(output_json_path, 'w') as f:
            json.dump(new_annotations, f, indent=4)


# Main function to process the image and annotations
def process_image_and_annotations(image_path, json_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read the original JSON annotations
    with open(json_path, 'r') as f:
        annotations = json.load(f)

    # Convert polygons to an instance mask
    polygons = annotations['shapes']
    img_shape = cv2.imread(image_path).shape
    mask = polygons_to_instance_mask(img_shape, polygons)

    # Split the image and mask
    sub_images, sub_masks = split_image_and_mask(image_path, mask, output_dir)

    # Update and save the new annotations for each sub-image
    update_annotations(sub_images, sub_masks, output_dir, json_path)


# Example usage
image_path = 'Data/00000.jpg'
json_path = 'Data/renxi1-2504.32_m184_0.json'
output_dir = 'Data/output_sub_images'

process_image_and_annotations(image_path, json_path, output_dir)

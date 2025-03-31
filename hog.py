import cv2
import numpy as np
import os
import csv
from skimage.feature import hog

# HOG Parameters
pixels_per_cell = (16, 16)
cells_per_block = (2, 2)
orientations = 9

# Paths
input_folder = "HAM10000_images_part_1/"
output_csv = "HOG_features.csv"  # New CSV file for extracted features

# Open CSV file for writing
with open(output_csv, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    # Loop through all image files
    header = ["image_id"]  # Start with image ID column
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Check for image files
            img_id = filename.split(".")[0]  # Extract image ID (without extension)
            img_path = os.path.join(input_folder, filename)

            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load image in grayscale
            image = cv2.resize(image, (128, 128))  # Resize for consistency

            # Compute HOG features
            hog_features = hog(
                image,
                orientations=orientations,
                pixels_per_cell=pixels_per_cell,
                cells_per_block=cells_per_block,
                block_norm="L2-Hys",
                transform_sqrt=True,
                feature_vector=True
            )

            # Create a single row with image_id and HOG features
            feature_row = [img_id] + list(hog_features)

            # Write the header only once
            if not header[1:]:  # If header is empty beyond "image_id"
                header += [f"HOG_{i}" for i in range(len(hog_features))]
                writer.writerow(header)

            # Write the extracted features to the CSV file
            writer.writerow(feature_row)
            print(f"Processed: {filename}")

print(f"HOG feature extraction completed! Features saved in {output_csv}.")

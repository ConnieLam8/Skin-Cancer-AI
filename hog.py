import cv2
import numpy as np
import os
import pandas as pd
from skimage.feature import hog

# HOG Parameters
pixels_per_cell = (16, 16)
cells_per_block = (2, 2)
orientations = 9

# Paths
input_folder = "HAM10000_images"
csv_file = "HAM10000_metadata.csv"  # Original CSV file

# Load the original CSV
df = pd.read_csv(csv_file)

# Initialize empty list to store HOG features
hog_features_list = []

# Process images and compute HOG features
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):  
        img_id = filename.split(".")[0]  
        img_path = os.path.join(input_folder, filename)

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  
        image = cv2.resize(image, (128, 128))  

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

        # Convert to list and store
        hog_features_list.append([img_id] + list(hog_features))

# Convert HOG features into DataFrame
hog_columns = [f"HOG_{i}" for i in range(len(hog_features_list[0]) - 1)]
hog_df = pd.DataFrame(hog_features_list, columns=["image_id"] + hog_columns)

# Merge HOG features with the original DataFrame
df = df.merge(hog_df, on="image_id", how="left")

# Save the updated DataFrame (overwrite original CSV)
df.to_csv(csv_file, index=False)

print(f"HOG feature extraction completed! Features added as new columns in {csv_file}.")

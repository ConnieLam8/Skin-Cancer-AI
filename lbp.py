import cv2
import numpy as np
import os
import pandas as pd

from skimage.feature import local_binary_pattern

# Load metadata
metadata_path = 'HAM10000_metadata.csv'
df = pd.read_csv(metadata_path)

# Define image directory
image_dir = 'HAM10000_images'

# Define LBP extraction function
def extract_lbp_features(image_path, P=8, R=1):
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return np.zeros(P + 2)  # fallback if image can't be read

    lbp = local_binary_pattern(gray, P, R, method='uniform')
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist

# Extract LBP features for each image
lbp_features = []

for idx, row in df.iterrows():
    image_id = row['image_id']
    full_image_name = f"{image_id}.jpg"  # Add the ISIC_ prefix
    image_path = os.path.join(image_dir,
     full_image_name)

    features = extract_lbp_features(image_path)
    lbp_features.append(features)

# Convert to DataFrame
lbp_df = pd.DataFrame(lbp_features, columns=[f'lbp_{i}' for i in range(len(lbp_features[0]))])

# Combine with original metadata
df_lbp_combined = pd.concat([df, lbp_df], axis=1)

# Overwrite the original metadata file with new data including LBP features
df_lbp_combined.to_csv('HAM10000_metadata.csv', index=False)

print(f"LBP feature extraction completed! Features added as new columns in {df_lbp_combined}.")
import pandas as pd

# Read the original CSV file
df = pd.read_csv("HAM10000_metadata.csv")

# Perform the necessary transformations
df["lesion_id"] = df["lesion_id"].str.replace("HAM_", "", regex=False)
df["image_id"] = df["image_id"].str.replace("ISIC_", "", regex=False)

category_mapping = {"nv": 1, "mel": 2, "bkl": 3, "bcc": 4, "akiec": 5, "df": 6, "vasc": 7}
df["dx"] = df["dx"].map(category_mapping)
df["dx"] = df["dx"].astype(int)

category_mapping2 = {"histo": 1, "follow_up": 2, "consensus": 3, "confocal": 4}
df["dx_type"] = df["dx_type"].map(category_mapping2)
df["dx_type"] = df["dx_type"].astype(int)

# Remove rows where 'age' is NaN
df = df.dropna(subset=["age"])
df["age"] = df["age"].astype(int)

df = df[df["sex"] != "unknown"]  # Remove rows where 'dx' is 'unknown'
df = df.dropna(subset=["localization"])
df = df[df["localization"] != "unknown"]  # Remove rows where 'dx_type' is 'unknown'

category_mapping3 = {"male": 1, "female": 2}
df["sex"] = df["sex"].map(category_mapping3)
df["sex"] = df["sex"].astype(int)

category_mapping4 = {'scalp': 1, 'ear': 2, 'face': 3, 'back': 4, 'trunk': 5, 'chest': 6, 'upper extremity': 7, 'abdomen': 8, 'lower extremity': 9, 'genital': 10, 'neck': 11, 'hand': 12, 'foot': 13, 'acral': 14}
df["localization"] = df["localization"].map(category_mapping4)
df["localization"] = df["localization"].astype(int)
print("Unique values in 'localization':", df["localization"].unique())


# Save the modified DataFrame to a new CSV file
df.to_csv("modified_HAM10000_metadata.csv", index=False)

# Optionally, print the updated DataFrame
print(df)

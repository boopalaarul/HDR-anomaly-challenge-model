from torch.utils.data import Dataset
from PIL import Image
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch

class ButterflyDataset(Dataset):
    def __init__(self, data, img_dir, transforms=None, handle_invalid="filter"):
        self.data = data
        self.img_dir = img_dir
        self.transforms = transforms
        self.handle_invalid = handle_invalid

        # Validate the 'hybrid_stat' column
        valid_classes = {"hybrid", "non-hybrid"}
        self.data["hybrid_stat"] = self.data["hybrid_stat"].str.strip().str.lower()

        if handle_invalid == "filter":
            # Filter out rows with invalid values
            invalid_rows = ~self.data["hybrid_stat"].isin(valid_classes)
            if invalid_rows.any():
                print(f"Warning: Found {invalid_rows.sum()} invalid values in 'hybrid_stat'. Filtering them out.")
                self.data = self.data[~invalid_rows]
        elif handle_invalid == "raise":
            if not set(self.data["hybrid_stat"].unique()).issubset(valid_classes):
                raise ValueError("Unexpected values found in 'hybrid_stat' column.")

        # Define classes explicitly
        self.classes = ["non-hybrid", "hybrid"]
        self.cls_lbl_map = {cls: i for i, cls in enumerate(self.classes)}

        # Generate labels
        self.labels = self.data["hybrid_stat"].map(self.cls_lbl_map).tolist()

        print("Created base dataset with {} samples".format(len(self.data)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_filename = self.data.iloc[idx]['filename']
        label = self.labels[idx]  # Use pre-computed labels
        meta = self.data.iloc[idx]['parent_subspecies_1'] #Get meta data from 'parent_subspecies_1' column
        img_path = os.path.join(self.img_dir, img_filename)  # Construct image path

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise FileNotFoundError(f"Error loading image at {img_path}: {e}")

        if self.transforms:
            image = self.transforms(image)

        # Debugging: Print/display a few sample images and labels
        #if idx < 5:  # Print the first 5 samples
         #   print(f"Image filename: {img_filename}")
         #   print(f"Label: {label}")
            # Convert the tensor to a NumPy array with channels last for displaying with matplotlib
         #   image_np = image.permute(1, 2, 0).numpy()
            # Unnormalize the image for display (assuming ImageNet normalization)
          #  mean = np.array([0.485, 0.456, 0.406])
           # std = np.array([0.229, 0.224, 0.225])
            #image_np = std * image_np + mean
            #image_np = np.clip(image_np, 0, 1)  # Clip values to [0, 1]

            #plt.imshow(image_np)  # Display the image
            #plt.show()

        return image, torch.tensor(label)
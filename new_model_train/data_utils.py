import os
from typing import Tuple
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

SUBSPECIES = "subspecies"
ANOMALY = "anomaly"

def data_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

"""Filter method that extracts necessary rows from data CSV and then filters rows based on which images successfully downloaded or failed to download.
Parameters
----------
dataframe: DataFrame read from big data CSV that lists all the images & their properties.
img_dir: Where the [downsized] images are found.
"""
def _filter(dataframe: pd.DataFrame, img_dir: str, classify_task: str) -> pd.DataFrame:

    print(f"Filtering data for classification task: {classify_task}.")
    if classify_task == SUBSPECIES:
        dataframe = dataframe.loc[dataframe["hybrid_stat"] == "non-hybrid"]
        print(f"All hybrid data excluded. {dataframe.index.size} rows kept.")
    elif classify_task == ANOMALY:
        print(f"Hybrid and non-hybrid data included.")
    
    bad_row_idxs = []
    
    for idx, row in tqdm(dataframe.iterrows(), desc="Filtering bad urls"):
        fname = row['filename']
        #go into the hybrid or nonhybrid folder, depending on what kind of row this is
        path = os.path.join(img_dir, row['hybrid_stat'], fname)
    
        if not os.path.exists(path):
            print(f"File not found: {path}")
            bad_row_idxs.append(idx)
        else:
            try:
                Image.open(path)
            except Exception as e:
                print(f"Error opening {path}: {e}")
                bad_row_idxs.append(idx)

    print(f"Bad rows: {len(bad_row_idxs)}")

    return dataframe.drop(bad_row_idxs)

"""Creates train/test splits over the big CSV file, after first filtering out the images that failed to download/ are not found in `img_dir`. This is where to specify `train_task`: either "subspecies" or "anomaly". 
Parameters
---------
data_path: Path to big CSV file of all the data URLs and properties. Rows will be filtered.
img_dir: Path to downloaded images. Any images which haven't been downloaded will be filtered out of the CSV from data_path.
test_size: Proportion of the train data to reallocate as test data.
classify_task: whether training for subspecies or anomaly classification. This arg will be supplied to internal method `_filter`.
random_state: Seed consistency, same "random" results every time.
Returns
---------
Filtered CSVs of train & test data URLs and properties."""
def load_data(data_path: str, img_dir: str, test_size: float, classify_task : str, random_state: int = 42, ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = _filter(pd.read_csv(data_path), img_dir, classify_task)
    assert(type(df) == pd.DataFrame)
    train_data, test_data = train_test_split(df, test_size=test_size, random_state=random_state)
    
    return train_data, test_data


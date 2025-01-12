import torch
from torch.utils.data import DataLoader
from pathlib import Path
from dataset import ButterflyDataset
from data_utils import data_transforms, load_data
import matplotlib.pyplot as plt

ROOT_DATA_DIR = Path("/home/jovyan/")
DATA_FILE = ROOT_DATA_DIR / "butterfly_anomaly_train.csv"
IMG_DIR = ROOT_DATA_DIR / "images_all"
BATCH_SIZE = 4  # Use a small batch size for testing

def test_data_loading():
    train_data, test_data = load_data(DATA_FILE, IMG_DIR)

    train_dataset = ButterflyDataset(train_data, IMG_DIR, transforms=data_transforms(train=True))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = ButterflyDataset(test_data, IMG_DIR, transforms=data_transforms(train=False))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Check for data leakage (filenames should not overlap)
    train_filenames = set(train_data['filename'].tolist())
    test_filenames = set(test_data['filename'].tolist())
    assert len(train_filenames.intersection(test_filenames)) == 0, "Data leakage detected!"

    # Inspect a few batches
    for loader_name, loader in [("Train", train_loader), ("Test", test_loader)]:
        print(f"\nInspecting {loader_name} Loader:")
        for batch_idx, (data, target) in enumerate(loader):
            if batch_idx >= 2:  # Check only the first 2 batches
                break

            print(f"  Batch: {batch_idx}")
            print(f"    Image shape: {data.shape}")
            print(f"    Target shape: {target.shape}")
            print(f"    Target values: {target}")

            # Unnormalize a sample image for display (assuming ImageNet normalization)
            image = data[0].cpu()  # Get the first image in the batch
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image = image * std + mean  # Unnormalize
            image = torch.clamp(image, 0, 1)  # Clip values to [0, 1]

            # Convert the tensor to a NumPy array and change the order of dimensions for display
            image_np = image.permute(1, 2, 0).numpy()

            # Display the image (replace plt.show() with plt.savefig() or another method to save)
            plt.imshow(image_np)
            plt.title(f"{loader_name} - Batch: {batch_idx}, Label: {target[0].item()}")
            #plt.show()
            plt.savefig(f"{loader_name}_batch_{batch_idx}.png")
            plt.close()

if __name__ == "__main__":
    test_data_loading()
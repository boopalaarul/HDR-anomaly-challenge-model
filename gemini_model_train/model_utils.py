# import torch
# import numpy as np
# from tqdm import tqdm
# import torch.nn as nn
# import torch.nn.functional as F

# class ButterflyNet(nn.Module):
#     def __init__(self, num_classes=2):
#         super(ButterflyNet, self).__init__()
#         self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.fc1 = nn.Linear(64 * 28 * 28, 512)
#         self.fc2 = nn.Linear(512, num_classes)
#         self.dropout = nn.Dropout(0.5)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = self.pool(F.relu(self.conv3(x)))
#         x = x.view(-1, 64 * 28 * 28)
#         x = F.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.fc2(x)
#         return x

# def get_feats_and_meta(dloader, model, device, ignore_feats=False):
#     all_feats = None
#     labels = []

#     for img, lbl in tqdm(dloader, desc="Extracting features"):
#         with torch.no_grad():
#             feats = None
#             if not ignore_feats:
#                 out = model(img.to(device))['image_features']
#                 feats = out.cpu().numpy()
#             if all_feats is None:
#                 all_feats = feats
#             else:
#                 all_feats = np.concatenate((all_feats, feats), axis=0) if feats is not None else all_feats
                
#         labels.extend(lbl.cpu().numpy().tolist())
        
#     labels = np.array(labels)
#     return all_feats, labels

import torch
import torch.nn as nn
import torch.nn.functional as F

class ButterflyNet(nn.Module):
    def __init__(self, num_classes=2):
        super(ButterflyNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)  # Conv layer 1
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # Conv layer 2
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Conv layer 3
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling (halves spatial dimensions)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, num_classes)  # Fixed typo here

        # Dropout layer
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Convolutional and pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # Flatten the tensor
        x = x.view(-1, 64 * 28 * 28)

        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        x = self.dropout(x)
        x = F.relu(self.fc5(x))  # No dropout after this layer
        x = self.fc6(x)  # Final output layer (logits)

        return x
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
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
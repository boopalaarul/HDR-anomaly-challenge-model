# import csv
# from pathlib import Path

# import torch
# import pickle 
# from torch.utils.data import DataLoader
# # from open_clip import create_model

# from dataset import ButterflyDataset
# from data_utils import data_transforms, load_data
# from evaluation import evaluate, print_evaluation
# from model_utils import get_feats_and_meta
# from model_utils import ButterflyNet
# from classifier import train, get_scores

# # Configuration         
# ROOT_DATA_DIR = Path("/home/jovyan/")
# DATA_FILE = ROOT_DATA_DIR / "butterfly_anomaly_train.csv"
# IMG_DIR = ROOT_DATA_DIR / "images_all"
# CLF_SAVE_DIR = Path("models/trained_clfs")
# DEVICE = "cuda:0"
# BATCH_SIZE = 4
# NUM_EPOCHS = 5

# def setup_data_and_model():
#     # Load Data
#     train_data, test_data = load_data(DATA_FILE, IMG_DIR)

#     # Model setup
#     # model = create_model("hf-hub:imageomics/bioclip", output_dict=True, require_pretrained=True)
#     # return model.to(DEVICE), train_data, test_data

#     model = ButterflyNet(num_classes=2)  # Instantiate your custom model
#     model = model.to(DEVICE)

#     return model, train_data, test_data


# def prepare_data_loaders(train_data, test_data):
#     train_sig_dset = ButterflyDataset(train_data, IMG_DIR, transforms=data_transforms())
#     tr_sig_dloader = DataLoader(train_sig_dset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
#     test_dset = ButterflyDataset(test_data, IMG_DIR, transforms=data_transforms())
#     test_dl = DataLoader(test_dset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
#     return tr_sig_dloader, test_dl


# def extract_features(tr_sig_dloader, test_dl, model):
#     tr_features, tr_labels = get_feats_and_meta(tr_sig_dloader, model, DEVICE)
#     test_features, test_labels = get_feats_and_meta(test_dl, model, DEVICE)
#     return tr_features, tr_labels, test_features, test_labels


# def train_and_evaluate(tr_features, tr_labels, test_features, test_labels):
#     import os
#     # Get the directory of your training script
#     script_dir = os.path.dirname(os.path.abspath(__file__))
    
#     # Change the current working directory
#     os.chdir(script_dir)
#     configs = ["svm","sgd","knn"]
#     csv_output = []
#     score_output = []

#     for con in configs:
#         print(f"Training and evaluating {con}...")
#         clf, acc, h_acc, nh_acc = train(tr_features, tr_labels, con)

#         # Save model to the specified path
#         model_filename = CLF_SAVE_DIR / f"trained_{con}_classifier.pkl"
#         with open(model_filename, 'wb') as model_file:
#             pickle.dump(clf, model_file)
#         print(f"Saved {con} classifier to {model_filename}")
#         print(f"{con}: Acc - {acc:.4f}, Hacc - {h_acc:.4f}, NHacc - {nh_acc:.4f}")
        
#         # Get scores for the test dataset
#         scores = get_scores(clf, test_features)
#         eval_scores = evaluate(scores, test_labels, reversed=False)
#         print_evaluation(*eval_scores)
#         csv_output.append([f"BioCLIP Features + {con}"] + list(eval_scores))
        
#         # Save individual scores for analysis
#         for idx, score in enumerate(scores):
#             score_output.append([idx, score, test_labels[idx]])
            
#     return csv_output, score_output


# def main():
#     model, train_data, test_data = setup_data_and_model()
#     tr_sig_dloader, test_dl = prepare_data_loaders(train_data, test_data)
#     tr_features, tr_labels, test_features, test_labels = extract_features(tr_sig_dloader, test_dl, model)
#     csv_output, score_output = train_and_evaluate(tr_features, tr_labels, test_features, test_labels)

#     # Define the loss function and optimizer
#     criterion = torch.nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adjust learning rate if needed

#     # Train the model
#     model = train_and_evaluate(model, tr_sig_dloader, test_dl, optimizer, criterion, num_epochs=NUM_EPOCHS)

#     # Save the trained model
#     model_filename = CLF_SAVE_DIR / "trained_butterfly_net.pth"
#     CLF_SAVE_DIR.mkdir(parents=True, exist_ok=True) #Ensure directory exists
#     torch.save(model.state_dict(), model_filename)
#     print(f"Saved trained model to {model_filename}")
    
#     # Save evaluation results
#     csv_filename = CLF_SAVE_DIR / "classifier_evaluation_results.csv"
#     with open(csv_filename, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(["Configuration", "AUC", "Precision", "Recall", "F1-score"])
#         writer.writerows(csv_output)
    
#     # Save individual scores
#     scores_filename = CLF_SAVE_DIR / "classifier_scores.csv"
#     with open(scores_filename, mode='w', newline='') as score_file:
#         score_writer = csv.writer(score_file)
#         score_writer.writerow(["Index", "Score", "True Label"])
#         score_writer.writerows(score_output)
    
# if __name__ == "__main__":
#     main()

import csv
from pathlib import Path

import torch
import pickle
from torch.utils.data import DataLoader
# from open_clip import create_model  # Remove this line, as we're not using BioClip

from dataset import ButterflyDataset
from data_utils import data_transforms, load_data
# from evaluation import evaluate, print_evaluation  # Remove if not using
# from classifier import train, get_scores  # Remove if not using
from model_utils import ButterflyNet  # Import your custom model

# ... (rest of your imports)

# Configuration
ROOT_DATA_DIR = Path("/home/jovyan/")
DATA_FILE = ROOT_DATA_DIR / "butterfly_anomaly_train.csv"
IMG_DIR = ROOT_DATA_DIR / "images_all"
CLF_SAVE_DIR = Path("models/trained_clfs")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
NUM_EPOCHS = 5  # You can adjust this

def setup_data_and_model():
    # Load Data
    train_data, test_data = load_data(DATA_FILE, IMG_DIR)

    # Model setup
    model = ButterflyNet(num_classes=2)  # Instantiate your custom model
    model = model.to(DEVICE)

    return model, train_data, test_data

def prepare_data_loaders(train_data, test_data):
    train_sig_dset = ButterflyDataset(train_data, IMG_DIR, transforms=data_transforms(train=True))
    tr_sig_dloader = DataLoader(train_sig_dset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    test_dset = ButterflyDataset(test_data, IMG_DIR, transforms=data_transforms(train=False))
    test_dl = DataLoader(test_dset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    return tr_sig_dloader, test_dl

def train_and_evaluate(model, tr_sig_dloader, test_dl, optimizer, criterion, num_epochs=10):
    model.train()  # Set the model to training mode

    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(tr_sig_dloader): # Removed the two _
            data, target = data.to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:  # Print training progress
                print(f"Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx}/{len(tr_sig_dloader)}, Loss: {loss.item():.4f}")

        # Evaluate on the test set after each epoch (or more/less frequently)
        model.eval()  # Set the model to evaluation mode
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_dl: # Removed the two _
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data)
                test_loss += criterion(output, target).item()  # Sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max logit
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_dl.dataset)
        accuracy = 100. * correct / len(test_dl.dataset)
        print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")

        model.train()  # Set back to training mode

    return model

def main():
    model, train_data, test_data = setup_data_and_model()
    tr_sig_dloader, test_dl = prepare_data_loaders(train_data, test_data)

    # Define the loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()  # Example: Cross-entropy loss for classification
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Example: Adam optimizer

    # Fine-tune the model
    model = train_and_evaluate(model, tr_sig_dloader, test_dl, optimizer, criterion, num_epochs=NUM_EPOCHS)

    # Save the fine-tuned model
    model_filename = CLF_SAVE_DIR / "trained_butterfly_net.pth"
    CLF_SAVE_DIR.mkdir(parents=True, exist_ok=True) #Ensure directory exists
    torch.save(model.state_dict(), model_filename)
    print(f"Saved trained model to {model_filename}")

if __name__ == "__main__":
    main()
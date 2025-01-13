###TRAINING SUBSPECIES CLASSIFIER
#It's fine to have separate methods for the base model
#as for the hybrid classifier: the hybrid classifier is what gets benchmarked
#against the competition. (SVM, KNN, etc.). 
#If the premise is that near perfect subspecies classification -> better anomaly detection, 
#then do the first part and test the second. Chances are it won't be entirely true:
#(in which case, start unfreezing/requiring gradient updates on last layers of base model?)
#but it will still be an interesting task of itself.

#if there was a way to embed (LLM?) images into a lower dimensional space, and to have the presence
#of hybrid features have an "additive"/"averaging" effect so that it could be found to be "between"
#species to some threshold of "betweeenness"...

#is there a way to "mask" out the pattern. We can already judge what is shared across all images.
#All have the same "background" on their wings:

#methods for loss, autodiff, etc.
import torch
#methods for saving and loading checkpoint/pickle files
import pickle 

#generic DataLoader that can create batches of Tensors from custom dataset
from torch.utils.data import DataLoader

### IMPORT MODEL CLASSES
from butterfly_class_detector import ButterflyClassDetector
#from butterfly_hybrid_detector import ButterflyHybridDetector
#no longer necessary #from open_clip import create_model

#custom data class & custom Transforms
#want to group the data files into their own folder
from dataset import ButterflyClassifierDataset, ButterflyDataset

#load_data takes the data from 
from data_utils import data_transforms, load_data, SUBSPECIES, ANOMALY

#may need a custom evaluation method for each of the 2 classifiers
#(class detect and hybrid detect)
from evaluation import evaluate, print_evaluation

#not necessary because ???? why are we pulling everything out of the dataloader anyways? Why run this 
#iterable by itself for no reason.
#from model_utils import get_feats_and_meta

#not necessary since this uses the sample classifiers.
#from classifier import train, get_scores

#Configuration ROOT_DATA_DIR not necessary, 
#DATA_FILE/data_path, IMG_DIR/img_dir used by data_utils.load_data() AND by ButterflyDataset
#for different reasons: used by load_data() to obtain CSV and filter rows based on downloaded images,
#then train/test split the CSV into smaller CSVs.
#img_dir used by DataSet to access images
#ROOT_DATA_DIR = Path("/home/jovyan/")
ROOT_DATA_DIR = "/home/jovyan/HDR-anomaly-challenge-sample/"
DATA_FILE = ROOT_DATA_DIR + "files/butterfly_anomaly_train.csv"
IMG_DIR = ROOT_DATA_DIR + "input_data/"

#CLF_SAVE_DIR = Path("models/trained_clfs")
DEVICE = "cuda:0"
NUM_DATALOADER_WORKERS = 4 #this jupyterlab should have 4 cores
BATCH_SIZE = 4
TEST_SIZE = 0.2

"""Sets up train and test datasets for model of choice, indicated by `classify_task`.
Parameters
----------
classify_task: \"subspecies\" to train/test the 14 class ButterflyClassDetector, or \"anomaly\" to train the 2 class ButterflyHybridDetector.

Returns
---------
model: Instance of ButterflyClassDetector or ButterflyHybridDetector as a GPU object.
train_data: DataFrame of train data.
test_data: DataFrame of test data.
"""
def setup_data_and_model(classify_task):

    if classify_task not in [SUBSPECIES, ANOMALY]:
        raise ValueError(f"Please specify a classification task: \"{SUBSPECIES}\" to train/test the 14 class ButterflyClassDetector, or \"{ANOMALY}\" to train the 2 class ButterflyHybridDetector.")
        
    # Load Data
    train_data, test_data = load_data(data_path = DATA_FILE,
                                      img_dir = IMG_DIR,
                                      test_size = TEST_SIZE, classify_task = classify_task)

    # Model setup -- THIS SHOULD BE DIFFERENT DEPENDING ON CLASSIFICATION TASK
    if classify_task == SUBSPECIES:
        #initialize new object of class ButterflyClassDetector as a GPU function
        model = ButterflyClassDetector().to(DEVICE)
    elif classify_task == ANOMALY:
        model = ButterflyHybridDetector().to(DEVICE)
        
    return model, train_data, test_data

"""Prepares DataLoaders for ButterflyClassDetector or ButterflyHybridDetector, depending on which
arguments are given. Uses data_transforms() from data_utils.py and batch_size from this file's BATCH_SIZE constant.
Parameters
----------
train_data: DataFrame(?) of train data URLs.
test_data: DataFrame(?) of test data URLs.
Returns
----------
train_dataloader: DataLoader for train data that applies specified transforms and outputs batch of images with specified batch size.
test_dataloader: DataLoader for test data etc etc.
"""
def prepare_data_loaders(train_data, test_data, classify_task):
    if classify_task == SUBSPECIES:
        train_dataset = ButterflyClassifierDataset(train_data, IMG_DIR, transforms=data_transforms())
        test_dataset = ButterflyClassifierDataset(test_data, IMG_DIR, transforms=data_transforms())
    elif classify_task == ANOMALY:
        train_dataset = ButterflyDataset(train_data, IMG_DIR, transforms=data_transforms())
        test_dataset = ButterflyDataset(test_data, IMG_DIR, transforms=data_transforms())
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=BATCH_SIZE, 
                                  shuffle=False, 
                                  num_workers=NUM_DATALOADER_WORKERS)
    test_dataloader = DataLoader(test_dataset, 
                                 batch_size=BATCH_SIZE, 
                                 shuffle=False, 
                                 num_workers=NUM_DATALOADER_WORKERS)
    return train_dataloader, test_dataloader

"""Keep this. not sure what the expected output looks like."""
def extract_features(train_dataloader, test_dataloader, model):
    train_features, train_labels = get_feats_and_meta(train_dataloader, model, DEVICE)
    test_features, test_labels = get_feats_and_meta(test_dataloader, model, DEVICE)
    return train_features, train_labels, test_features, test_labels

"""Need to change this method wholesale. Best case: tr/test features and labels are dataframes
that include the necessary labels in there somewhere, (hybrid_stat) and (subspecies) columns."""
def train_and_evaluate(tr_features, tr_labels, test_features, test_labels):
    """
    import os
    # Get the directory of your training script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Change the current working directory
    os.chdir(script_dir)
    configs = ["svm","sgd","knn"]
    csv_output = []
    score_output = []

    for con in configs:
        print(f"Training and evaluating {con}...")
        clf, acc, h_acc, nh_acc = train(tr_features, tr_labels, con)

        # Save model to the specified path
        model_filename = CLF_SAVE_DIR / f"trained_{con}_classifier.pkl"
        with open(model_filename, 'wb') as model_file:
            pickle.dump(clf, model_file)
        print(f"Saved {con} classifier to {model_filename}")
        print(f"{con}: Acc - {acc:.4f}, Hacc - {h_acc:.4f}, NHacc - {nh_acc:.4f}")
        
        # Get scores for the test dataset
        scores = get_scores(clf, test_features)
        eval_scores = evaluate(scores, test_labels, reversed=False)
        print_evaluation(*eval_scores)
        csv_output.append([f"BioCLIP Features + {con}"] + list(eval_scores))
        
        # Save individual scores for analysis
        for idx, score in enumerate(scores):
            score_output.append([idx, score, test_labels[idx]])
            
    return csv_output, score_output
    """

"""Working this into the testing.
Want to accomplish the following tasks:
1) Load data for subspecies classification, train & evail subspecies model, save model weights as pth.
2) Load data for hybrid classification, train & eval hybrid model, save whole model as pkl.
3) prepare code submission.
"""
def main():

    ###SUBSPECIES MODEL
    model, train_data, test_data = setup_data_and_model(classify_task=SUBSPECIES)
    train_dataloader, test_dataloader = prepare_data_loaders(train_data, test_data, classify_task=SUBSPECIES)

    #train the subspecies model
    

    #save
    
    ###HYBRID MODEL
    model, train_data, test_data = setup_data_and_model(classify_task=HYBRID)
    train_dataloader, test_dataloader = prepare_data_loaders(train_data, test_data)

    #I honestly just want to know what these features and labels even are...
    tr_features, tr_labels, test_features, test_labels = extract_features(train_dataloader, test_dataloader, model)
    
    csv_output, score_output = train_and_evaluate(tr_features, tr_labels, test_features, test_labels)
    
    # Save evaluation results
    csv_filename = CLF_SAVE_DIR / "classifier_evaluation_results.csv"
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Configuration", "AUC", "Precision", "Recall", "F1-score"])
        writer.writerows(csv_output)
    
    # Save individual scores
    scores_filename = CLF_SAVE_DIR / "classifier_scores.csv"
    with open(scores_filename, mode='w', newline='') as score_file:
        score_writer = csv.writer(score_file)
        score_writer.writerow(["Index", "Score", "True Label"])
        score_writer.writerows(score_output)
    
if __name__ == "__main__":
    main()
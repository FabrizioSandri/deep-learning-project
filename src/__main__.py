import argparse
import torch
import sys
import logging
from torch.utils.data import Dataset, DataLoader, random_split

from dataset.RefCOCOg import *
from model.ClipRPN import *
from model.baseline import *
from evaluation.qualitative_results import *
from evaluation.baseline_eval import *
from evaluation.ClipRPN_eval import *
from train.training import *

# Logger set up
logging.basicConfig(format='%(asctime)s  %(message)s', level=logging.INFO)
logger = logging.getLogger()

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', metavar='PATH', required=True, action='store', help="Location of the RefCOCOg dataset.")
parser.add_argument('-i', '--inference', metavar='ID', type=int, action='store', help="Specifies the image ID from the test dataset on which to perform inference. To execute the inference, it is necessary to specify the model using the `-m` option.")
parser.add_argument('-m', '--model', action='store', default="attention", choices=["baseline", "concatenation", "attention"], help="Defines the type of model to be used during training. The available options are as follows:\n- 'baseline': This option utilizes a training-free model that combines YOLO with CLIP.\n- 'concatenation': This option uses a model that employs simple concatenation with the textual description.\n- 'attention': This option represents our final model, which utilizes cross attention to fuse text and images.")
parser.add_argument('-b', '--batch_size', action='store', default=50, type=int, help="Batch size for the data loaders.")
parser.add_argument('-e', '--evaluate', action='store_true', help="Evaluate the following metrics for the specified model: oIOU, Recall, Cosine Similarity.")
parser.add_argument('-t', '--train', metavar='NUM_EPOCHS', action='store', help="Train the specified model for the specified amount of epochs.")

args = parser.parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the dataset 
logger.info("Loading the dataset")
train_dataset = RefCOCOg(args.dataset, split="train", transformations=True, size=1000)
val_dataset = RefCOCOg(args.dataset, split="val", transformations=False, size=1000)
test_dataset = RefCOCOg(args.dataset, split="test", transformations=False, size=1000)

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
val_loader = DataLoader(val_dataset, shuffle=False, batch_size=args.batch_size)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size)


####################### INFERENCE #######################
if args.inference is not None or args.evaluate==True:
    logger.info("Option selected: inference")

    if not args.model:
        sys.exit("Please specify the model using -m")   
    
    scales = torch.tensor([32, 64, 128, 256, 512])
    ratios = torch.tensor([0.5, 1.0, 2.0])
    
    if args.model == "baseline":
        logger.info("Loading the baseline model")
        model = Baseline(test_dataset)

    elif args.model == "concatenation": # ClipRPN with the fusion module based on Concatenation
        logger.info("Loading the model with the concatenation fusion module")
        model = ClipRPN(
            anchors_scales=scales, anchor_ratios=ratios,
            fusion_type="concatenation", # the fusion method to use, either "cross_attention" or "concatenation"
            feature_map_channels=2048, hidden_dim=1024
        ).to(device)

        model.load_state_dict(torch.load("models/model_concatenation.pth"))

    else:   # ClipRPN with the fusion module based on Cross Attention  
        logger.info("Loading the model with the cross attention fusion module")
        model = ClipRPN(
            anchors_scales=scales, anchor_ratios=ratios,
            fusion_type="attention", # the fusion method to use, either "cross_attention" or "concatenation"
            attention_heads=16,
            feature_map_channels=2048, hidden_dim=1024
        ).to(device)

        model.load_state_dict(torch.load("models/model_attention.pth"))

    #Freeze CLIP weights
    for name, param in model.named_parameters():
        if param.requires_grad and 'clipModel' in name:
            param.requires_grad = False

    if args.inference:
        # Plot the qualitative results
        image, text, bbox, _ = test_dataset[args.inference]
        plotPrediction(model, image, text, bbox)
    
    if args.evaluate:
        logger.info("Starting evaluation")
        if args.model == "baseline":
            compute_baseline_metrics(model, test_loader)
        else:
            compute_metrics(model, test_loader)

####################### TRAINING #######################
if args.train is not None:
    logger.info("Option selected: train")

    if not args.model or args.model == "baseline":
        sys.exit("Please specify the model using -m (you cannot specify the baseline)")   
    
    scales = torch.tensor([32, 64, 128, 256, 512])
    ratios = torch.tensor([0.5, 1.0, 2.0])
    
    if args.model == "concatenation": # ClipRPN with the fusion module based on Concatenation
        logger.info("Creating the model with the concatenation fusion module")
        model = ClipRPN(
            anchors_scales=scales, anchor_ratios=ratios,
            fusion_type="concatenation", # the fusion method to use, either "cross_attention" or "concatenation"
            feature_map_channels=2048, hidden_dim=1024
        ).to(device)

    else:   # ClipRPN with the fusion module based on Cross Attention  
        logger.info("Creating the model with the cross attention fusion module")
        model = ClipRPN(
            anchors_scales=scales, anchor_ratios=ratios,
            fusion_type="attention", # the fusion method to use, either "cross_attention" or "concatenation"
            attention_heads=16,
            feature_map_channels=2048, hidden_dim=1024
        ).to(device)

    #Freeze CLIP weights
    for name, param in model.named_parameters():
        if param.requires_grad and 'clipModel' in name:
            param.requires_grad = False

    # Define the optimizer
    lr = 0.0002708
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # For each epoch, train the network and then compute evaluation results
    num_epochs = int(args.train)

    history_train_loss = []
    history_train_accuracy = []
    history_val_accuracy = []

    for epoch in range(1, num_epochs+1):

        # training loop
        train_loss, train_accuracy = train_step(model, optimizer, train_loader)
        history_train_loss.append(train_loss)
        history_train_accuracy.append(train_accuracy)

        # save the model weights
        save_network(model, str(epoch))

        # evaluation loop
        val_accuracy = test_step(model, val_loader)
        history_val_accuracy.append(val_accuracy)

        print(f"================== EPOCH {epoch}/{num_epochs} =====================")
        print(f"Training Loss: {train_loss} Training mIOU: {train_accuracy}")
        print(f"Validation mIOU: {val_accuracy}")
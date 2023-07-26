import torch
import torchvision
import os

from tqdm import tqdm

# function used to save the weights of the model at each epoch
def save_network(model, epoch_label):
  save_filename = 'model_%s.pth' % epoch_label
  save_path = os.path.join('./', save_filename)
  torch.save(model.state_dict(), save_path)

def getLoss(cls_loss, reg_loss, lambda_rpn=1):
    return cls_loss + lambda_rpn * reg_loss

def computeSumIou(gtBoxes, predictedBoxes):
  iouMatrix = torchvision.ops.box_iou(gtBoxes, predictedBoxes).diag()
  return iouMatrix.sum()

def train_step(model, optimizer, train_loader):
    avg_loss = 0
    total_iou = 0
    total_samples = 0

    # Set model to train mode
    model.train()

    for images, descriptions, gtBoxes, _ in tqdm(train_loader):
        batch_size = images.shape[0]
        total_samples += batch_size

        # Gradients reset
        optimizer.zero_grad()

        # Forward pass
        proposals, losses = model(images, descriptions, gtBoxes)

        # Calculate the loss
        loss = getLoss(losses["loss_objectness"], losses["loss_rpn_box_reg"])
        avg_loss += loss.item()

        # calculate the IOU
        best_proposals = torch.zeros((batch_size, 4), device=model.device)
        for j in range(batch_size):
            # The proposal with the highest score(the first) is the final one
            best_proposals[j,:] = proposals[j][0]

        total_iou += computeSumIou(gtBoxes, best_proposals).item()

        # Backward pass and optimization step
        loss.backward()
        optimizer.step()

    loss = avg_loss / len(train_loader)
    accuracy = total_iou / total_samples

    return loss, accuracy


def test_step(model, val_loader):
    total_iou = 0
    total_samples = 0

    # Set the model in evaluation mode, i.e. disable dropout
    model.eval()

    # stop tracking the gradients
    with torch.no_grad():
        for images, descriptions, gtBoxes, _ in tqdm(val_loader):
            batch_size = images.shape[0]
            total_samples += batch_size

            # Forward pass
            proposals, _ = model(images, descriptions, gtBoxes)

            # calculate the IOU
            best_proposals = torch.zeros((batch_size, 4), device=model.device)
            for j in range(batch_size):
                # The proposal with the highest score(the first) is the final one
                best_proposals[j,:] = proposals[j][0]

            total_iou += computeSumIou(gtBoxes, best_proposals).item()

    accuracy = total_iou / total_samples

    return accuracy

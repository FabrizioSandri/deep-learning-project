import torch
from tqdm import tqdm

from .model import RPN

class TrainRPN():

  '''
  Constructor for the TrainRPN class

  Args:
    - device: either "cpu" or "cuda"
  '''
  def __init__(self, device=None):
    if device is None:
      self.device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
      self.device = device
    
    # model configuration
    self.model = RPN(device=self.device)
    self.optimizer = torch.optim.Adam(self.model.parameters())


  '''
  Trains the RPN network

  Args:
    - num_epochs: number of epochs to train
    - train_loader: dataloader for the training split of the dataset, where each
        sample is a single image 
    - val_loader: dataloader for the validation split of the dataset, where each
      sample is a single image
  '''
  def train(self, num_epochs, train_loader, val_loader):
    train_loss_history = []
    val_loss_history = []

    # Set the model in training mode
    self.model.train(True)

    for epoch in range(num_epochs):
      print('Epoch {}:'.format(epoch + 1))

      training_loss = 0.0
      i = 0
      for images, boxes in tqdm(train_loader):
          
        self.optimizer.zero_grad()

        # This is a list of ground truth boxes for each of the image in the
        # batch that are sent to the RPN
        ground_truth_boxes = [{"boxes": boxes[batch_i]} for batch_i in range(images.shape[0])]
        
        # Make predictions for this batch
        _, losses = self.model(images, ground_truth_boxes)

        # loss function
        losses = losses["loss_objectness"] + losses["loss_rpn_box_reg"]
        losses.backward()
        
        # Adjust learning weights
        self.optimizer.step()

        # print statistics
        training_loss += losses.item()
        i+=1
        if i % 100 == 0:    
          print(f'[epoch {epoch + 1}] loss: {training_loss / i:.3f}')

      
      # Start validation
      print('Starting validation')
      val_loss = 0
      with torch.no_grad():
        for images, boxes in tqdm(val_loader):
                        
          ground_truth_boxes = [{"boxes": boxes[batch_i]} for batch_i in range(images.shape[0])]
          
          _, losses = self.model(images, ground_truth_boxes)
          losses = losses["loss_objectness"] + losses["loss_rpn_box_reg"]
          val_loss += losses.item()

        print(f'[epoch {epoch + 1}] validation loss: {val_loss/len(val_loader):.3f}')

      train_loss_history.append(training_loss/len(train_loader))   
      val_loss_history.append(val_loss/len(val_loader))


    return self.model, train_loss_history, val_loss_history
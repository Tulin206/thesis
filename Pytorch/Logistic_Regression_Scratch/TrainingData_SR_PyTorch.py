import torch
from tqdm import tqdm
import random
import numpy as np

# Set seed.
# seed = 180
# seed = 90
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
np.random.seed(seed)
random.seed(seed)


# Training function.
def train(model, trainloader, optimizer, criterion, device):
    model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        # Forward pass.
        outputs = model(image)
        print("shape of outputs before unsqueezing", outputs.shape)
        print("shape of labels before unsqueezing", labels.shape)

        # Calculate the predicted output
        predicted_probabilities = torch.sigmoid(outputs)  # Apply sigmoid to get probabilities

        if predicted_probabilities <= 0.5:
            preds = 0
        else:
            preds = 1

        # preds = (predicted_probabilities > 0.5)  # 1 for > 0.5, 0 for <= 0.5

        # # Calculate the predicted output
        # _, preds = torch.max(outputs.data, 1)

        # Calculate the loss.
        # loss = criterion(outputs, labels)
        loss = criterion(outputs, torch.unsqueeze(labels.float(), dim=0))
        train_running_loss += loss.item()

        print("shape of labels after unsqueezing", labels.shape)
        print("print labels after unsqueezing", labels)
        print("//////////////////////////////////////")

        # Calculate the accuracy.
        train_running_correct += (preds == labels).sum().item()

        # train_running_correct += torch.sum(preds == labels).item()

        # Backpropagation
        loss.backward()

        # Update the weights.
        optimizer.step()

    # Loss and accuracy for the complete epoch.
    epoch_loss = train_running_loss / counter
    # epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    return epoch_loss, epoch_acc


# Validation function.
def validate(model, testloader, criterion, device):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1

            image, labels = data
            image = image.to(device)
            labels = labels.to(device)

            # Forward pass.
            outputs = model(image)

            # # Calculate the predicted output
            # _, preds = torch.max(outputs.data, 1)

            # Calculate the predicted output
            predicted_probabilities = torch.sigmoid(outputs)  # Apply sigmoid to get probabilities

            if predicted_probabilities > 0.5:
                preds = 1
            else:
                preds = 0

            # preds = (predicted_probabilities > 0.5)  # 1 for > 0.5, 0 for <= 0.5

            # Calculate the loss.
            loss = criterion(outputs, torch.unsqueeze(labels.float(), dim=0))
            valid_running_loss += loss.item()

            # Calculate the accuracy.
            valid_running_correct += (preds == labels).sum().item()

    # Loss and accuracy for the complete epoch.
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))
    return epoch_loss, epoch_acc


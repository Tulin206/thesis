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
        print("shape of outputs for training", outputs.shape)
        print("print outputs for training:", outputs)
        print("shape of labels for training", labels.shape)
        print("print labels for training:", labels)

        # Calculate the predicted output
        # predicted_probabilities = torch.sigmoid(outputs)  # Apply sigmoid to get probabilities for binary classification
        preds = torch.softmax(outputs, dim=1)
        print("print predicted probabilities for training:", preds)
        _, preds = torch.max(outputs, 1)
        print("print predicted probabilities for training:", preds)
        print("print the shape of preds for training", preds.shape)
        print("print preds for training", preds)

        # Calculate the loss.
        loss = criterion(outputs, labels.float())
        # loss = criterion(outputs, labels)
        # loss = criterion(outputs, torch.unsqueeze(labels.float(), dim=0))
        train_running_loss += loss.item()


        _, labels = torch.max(labels, 1)
        print("print the shape of labels for training", labels.shape)
        print("print lables for training", labels)

        # Calculate the accuracy.
        train_running_correct += np.equal(preds.cpu().numpy(), labels.cpu().numpy()).sum().item()
        print("training_running_correct", train_running_correct)
        print("//////////////////////////////////////")

        # Backpropagation
        loss.backward()

        # Update the weights.
        optimizer.step()

    # Loss and accuracy for the complete epoch.
    epoch_loss = train_running_loss / counter
    # epoch_acc = train_running_correct / len(trainloader.dataset)
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    print("training_running_correct after for_loop", train_running_correct)
    print("length of trainloader", len(trainloader.dataset))
    return epoch_loss, epoch_acc


# Validation function.
def validate(model, validloader, criterion, device):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(validloader), total=len(validloader)):
            counter += 1

            image, labels = data
            image = image.to(device)
            labels = labels.to(device)

            # Forward pass.
            outputs = model(image)
            print("shape of outputs for validation", outputs.shape)
            print("print outputs for validation:", outputs)
            print("shape of labels for validation", labels.shape)
            print("print labels for validation:", labels)


            # Calculate the predicted output
            preds = torch.softmax(outputs, dim=1)
            print("print predicted probabilities for validation:", preds)
            # predicted_probabilities = torch.sigmoid(outputs)  # Apply sigmoid to get probabilities
            _, preds = torch.max(outputs, 1)
            print("print predicted probabilities for validation:", preds)
            print("print the shape of preds for validation", preds.shape)
            print("print preds for validation", preds)

            # Calculate the loss.
            # loss = criterion(outputs, labels)
            loss = criterion(outputs, labels.float())
            # loss = criterion(outputs, torch.unsqueeze(labels.float(), dim=0))
            valid_running_loss += loss.item()

            _, labels = torch.max(labels, 1)
            print("print the shape of labels for validation", labels.shape)
            print("print lables for validation", labels)

            # # Calculate the accuracy.
            # if preds == labels:
            #     valid_running_correct += 1
            # else:
            #     valid_running_correct = 0

            # # Calculate the accuracy.
            # valid_running_correct += (preds.cpu().numpy() == labels.cpu().numpy()).sum().item()

            # Calculate the accuracy.
            valid_running_correct += np.equal(preds.cpu().numpy(), labels.cpu().numpy()).sum().item()
            print("valid_running_correct after for_loop", valid_running_correct)
            print("//////////////////////////////////////")

    # Loss and accuracy for the complete epoch.
    epoch_loss = valid_running_loss / counter
    # epoch_acc = valid_running_correct / len(validloader.dataset)
    epoch_acc = 100. * (valid_running_correct / len(validloader.dataset))
    print("valid_running_correct", valid_running_correct)
    print("length of validloader", len(validloader.dataset))
    return epoch_loss, epoch_acc


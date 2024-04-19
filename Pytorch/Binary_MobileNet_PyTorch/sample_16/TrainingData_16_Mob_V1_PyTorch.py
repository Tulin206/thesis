import torch
from tqdm import tqdm
import random
import numpy as np
import torch.nn.functional as F

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
        predicted_probabilities = torch.sigmoid(outputs.cpu())  # Apply sigmoid to get probabilities
        print("Model Confidence:", predicted_probabilities)

        # preds = np.round(predicted_probabilities.detach().numpy())
        # print("round-up model's confidence:", preds)

        if predicted_probabilities <= 0.5:
            preds = 0
        else:
            preds = 1

        # preds = (predicted_probabilities > 0.5)  # 1 for > 0.5, 0 for <= 0.5

        # # Calculate the predicted output
        # _, preds = torch.max(outputs.data, 1)

        from DataLoad_16_Mob_V1_PyTorch import calculate_class_weights

        # Generate true labels for 16 samples
        classes = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0])

        # Use calculate_class_weights function
        class_weights_tensor = calculate_class_weights(classes)
        # class_weights_tensor = class_weights_tensor.unsqueeze(0)
        print("the shape of class_weights_tensor", class_weights_tensor.shape)
        class_weights_tensor = class_weights_tensor.to(labels.device)

        # Access the class weights for each class
        weight_class_0 = class_weights_tensor[0]
        # weight_class_0 = torch.tensor([weight_class_0], device=labels.device)
        weight_class_1 = class_weights_tensor[1]
        # weight_class_1 = torch.tensor([weight_class_1], device=labels.device)

        print("Weight for class 0 (Soft):", weight_class_0)
        print("Weight for class 1 (Hard):", weight_class_1)

        # Construct pos_weight tensor based on label values
        pos_weight = torch.where(labels == 0, torch.tensor([weight_class_0], device=labels.device),
                                 torch.tensor([weight_class_1], device=labels.device))
        pos_weight = pos_weight.to(labels.device)

        print("shape of labels", labels.shape)
        print("print ground truth labels", labels)
        labels = torch.unsqueeze(labels.float(), dim=0)

        # Calculate the loss.
        loss = F.binary_cross_entropy_with_logits(outputs, labels, pos_weight=pos_weight)

        # Calculate the loss.
        # loss = criterion(outputs, labels)
        # loss = criterion(outputs, torch.unsqueeze(labels.float(), dim=0))
        train_running_loss += loss.item()

        print("shape of labels after unsqueezing", labels.shape)
        print("print labels after unsqueezing", labels)
        print("//////////////////////////////////////")

        # Back to the tensor
        preds = torch.tensor(preds, device=outputs.device)

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


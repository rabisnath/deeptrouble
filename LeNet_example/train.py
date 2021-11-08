import matplotlib
from torch._C import device
from torch.functional import broadcast_shapes
matplotlib.use("Agg")

from lenet import LeNet, Bayesian_LeNet

from sklearn.metrics import classification_report
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST
from torch.optim import Adam
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
import time

#MODEL = LeNet
MODEL = Bayesian_LeNet 

if __name__ == '__main__':
    # grabbing command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True, help="path to trained model")
    parser.add_argument("-p", "--plot", type=str, required=True, help="path to loss/accuracy plot")
    args = parser.parse_args()

    # setting hyperparameters
    learning_rate = 1e-3
    batch_size = 64
    epochs = 10
    # (1 - train_split) of the training data will be used for validation
    train_split = 0.75

    # setting training device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # loading the MNIST dataset
    print("Loading MNIST dataset...")
    train_data = MNIST(root="data", train=True, download=True, transform=ToTensor())
    test_data = MNIST(root="data", train=False, download=True, transform=ToTensor())

    # splitting the training data into training and validation data
    print("Making training and validation sets...")
    n_train = int(train_split * len(train_data))
    n_val = len(train_data) - n_train
    (train_data, val_data) = random_split(train_data, [n_train, n_val], generator=torch.Generator().manual_seed(84))

    # initializing data loaders
    train_data_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    val_data_loader = DataLoader(val_data, shuffle=True, batch_size=batch_size)
    test_data_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

    # calculating steps per epoch for training+val set
    training_steps = len(train_data_loader.dataset) // batch_size
    validation_steps = len(val_data_loader.dataset) // batch_size

    # initializing the LeNet model
    print("Initializing model...")
    model = MODEL(n_channels=1, n_classes=len(train_data.dataset.classes)).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.NLLLoss()

    # tracking training history
    hist = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    # training
    print("Training the network...")
    t_start = time.time()

    for i in range(epochs):
        # set model to training mode
        model.train()
        # initializing total training, val loss
        total_train_loss = 0
        total_val_loss = 0
        # initializing the number of correct predictions
        train_n_correct = 0
        val_n_correct = 0

        # training step
        for (x, y) in train_data_loader:
            # send input to device
            (x, y) = (x.to(device), y.to(device))

            # perform a forward pass and calc the training loss
            prediction = model(x)
            loss = loss_fn(prediction, y)

            # setting gradients to zero, performing backprop and updating the weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # tracking training loss and num correct
            total_train_loss += loss
            train_n_correct += (prediction.argmax(1) == y).type(torch.float).sum().item()

        # validation step
        with torch.no_grad(): # turning off autograd for evaluation
            # setting model to evaluation mode
            model.eval()

            for (x, y) in val_data_loader:
                # send input to device
                (x, y) = (x.to(device), y.to(device))

                # perform a forward pass and calc the training loss
                prediction = model(x)
                total_val_loss += loss_fn(prediction, y)
                val_n_correct += (prediction.argmax(1) == y).type(torch.float).sum().item()

        # adding stats to the history object
        
        avg_train_loss = total_train_loss / training_steps
        avg_val_loss = total_val_loss / validation_steps
        train_n_correct = train_n_correct / len(train_data_loader.dataset)
        val_n_correct = val_n_correct / len(val_data_loader.dataset)

        hist['train_loss'].append(avg_train_loss.cpu().detach().numpy())
        hist['train_acc'].append(train_n_correct)
        hist['val_loss'].append(avg_val_loss.cpu().detach().numpy())
        hist['val_acc'].append(val_n_correct)

        # printing
        print("Epoch {}/{}".format(i+1, epochs))
        print("Training Loss: {:.6f}, Training Accuracy: {:.4f}".format(avg_train_loss, train_n_correct))
        print("Validation Loss: {:.6f}, Validation Accuracy: {:.4f}".format(avg_val_loss, val_n_correct))


    t_end = time.time()
    print("Total time: {:.2f}s".format(t_end - t_start))

    print("Evaluating trained network...")

    # turn off autograd for testing
    with torch.no_grad():
        # set model to eval mode
        model.eval()
        # empty list to store predictions
        predictions = []

        # looping over testing data
        for (x, y) in test_data_loader:
            x = x.to(device)

            # making predictions
            prediction = model(x)
            predictions.extend(prediction.argmax(axis=1).cpu().numpy())


    # generate a classification report
    print(classification_report(test_data.targets.cpu().numpy(), np.array(predictions), target_names=test_data.classes))

    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(hist["train_loss"], label="train_loss")
    plt.plot(hist["val_loss"], label="val_loss")
    plt.plot(hist["train_acc"], label="train_acc")
    plt.plot(hist["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(args.plot)
    # serialize the model to disk
    torch.save(model, args.model)




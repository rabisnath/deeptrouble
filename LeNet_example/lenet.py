from torch._C import InferredType
from torch.nn import Module # Base class for all nn modules in PyTorch
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch import flatten

from layers import BBBConv2d, BBBLinear, FlattenLayer

'''
Here I follow along with this tutorial:
https://www.pyimagesearch.com/2021/07/19/pytorch-training-your-first-convolutional-neural-network-cnn/

'''


# LeNet architecture 
class LeNet(Module):
    def __init__(self, n_channels, n_classes):
        '''
        n_channels: the number of channels in the input images
        n_classes: the number of unique labels/outputs represented in the dataset

        '''
        # calling the constructor for the Model class
        super(LeNet, self).__init__()

        # A set of Conv, ReLU and Pool layers
        self.conv1 = Conv2d(in_channels=n_channels, out_channels=20, kernel_size=(5,5))
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # A second set of Conv, ReLU and Pool layers
        self.conv2 = Conv2d(in_channels=20, out_channels=50, kernel_size=(5,5))
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # A fully connected layer followed by a ReLU
        self.fc1 = Linear(in_features=800, out_features=500)
        self.relu3 = ReLU()

        # Initializing our softmax classifier
        self.fc2 = Linear(in_features=500, out_features=n_classes)
        self.logSoftmax = LogSoftmax(dim=1)

    def forward(self, x):
        '''
        Accepts a batch of input data to feed to the network, 
        returns the network predictions

        '''
        # pass the input through our first set of CONV => RELU =>
		# POOL layers 
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
		# pass the output from the previous layer through the second
		# set of CONV => RELU => POOL layers
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
		# flatten the output from the previous layer and pass it
		# through our only set of FC => RELU layers
        x = flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)
		# pass the output to our softmax classifier to get our output
		# predictions
        x = self.fc2(x)
        output = self.logSoftmax(x)
		# return the output predictions
        return output

# Bayesian version
'''
https://github.com/kumar-shridhar/PyTorch-BayesianCNN

Above Network can be converted to Bayesian as follows:

class Net(ModuleWrapper):

  def __init__(self):
    super().__init__()
    self.conv = BBBConv2d(3, 16, 5, strides=2)
    self.bn = nn.BatchNorm2d(16)
    self.relu = nn.ReLU()
    self.flatten = FlattenLayer(800)
    self.fc = BBBLinear(800, 10)

'''

class Bayesian_LeNet(Module):
    def __init__(self, n_channels, n_classes):
        '''
        n_channels: the number of channels in the input images
        n_classes: the number of unique labels/outputs represented in the dataset

        '''
        # calling the constructor for the Model class
        super(Bayesian_LeNet, self).__init__()

        # A set of Conv, ReLU and Pool layers
        self.conv1 = BBBConv2d(in_channels=n_channels, out_channels=20, kernel_size=(5,5))
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # A second set of Conv, ReLU and Pool layers
        self.conv2 = BBBConv2d(in_channels=20, out_channels=50, kernel_size=(5,5))
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # A fully connected layer followed by a ReLU
        self.flatten = FlattenLayer(800)
        self.fc1 = BBBLinear(in_features=800, out_features=500)
        self.relu3 = ReLU()

        # Initializing our softmax classifier
        self.fc2 = BBBLinear(in_features=500, out_features=n_classes)
        self.logSoftmax = LogSoftmax(dim=1)

    def forward(self, x):
        '''
        Accepts a batch of input data to feed to the network, 
        returns the network predictions

        '''
        # pass the input through our first set of CONV => RELU =>
		# POOL layers 
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
		# pass the output from the previous layer through the second
		# set of CONV => RELU => POOL layers
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
		# flatten the output from the previous layer and pass it
		# through our only set of FC => RELU layers
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
		# pass the output to our softmax classifier to get our output
		# predictions
        x = self.fc2(x)
        output = self.logSoftmax(x)
		# return the output predictions
        return output

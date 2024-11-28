
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    """
    Create a LeNet-5 CNN with 3 convolutional layers and 2 pooling layers, followed by 2 fully connected layers.
    """

    def __init__(self, N_in = 1, N_out = 3, H_in = 492, W_in = 492, hidden_channels = [6,6,6,16,16], kernels=[5,5,5,5,5], strides=[1,2,1,5,5],paddings=[1,1,1,1]):
        """
        N_in: int with the number of input channels 
        Hidden_channels: list with the number of channels in each convolutional layer
        N_out: int with the number of output classes
        """
        
        super(CNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=N_in,
                                out_channels=hidden_channels[0], 
                                kernel_size=kernels[0], 
                                stride=strides[0],
                                padding=paddings[0])
        
        self.conv2 = nn.Conv2d(in_channels=hidden_channels[0],
                                out_channels=hidden_channels[1], 
                                kernel_size=kernels[1], 
                                stride=strides[1],
                                padding=paddings[1])
        
        self.conv3 = nn.Conv2d(in_channels=hidden_channels[1],
                                out_channels=hidden_channels[2], 
                                kernel_size=kernels[2], 
                                stride=strides[2],
                                padding=paddings[2])

        # Pooling layers
        self.pool1 = nn.MaxPool2d(kernel_size=kernels[3], stride=strides[3])
        self.pool2 = nn.MaxPool2d(kernel_size=kernels[4], stride=strides[4])

        # Fully connected layers

        # Calculate the size of the feature map after the final pooling layer
        def conv2d_size_out(size, kernel_size, stride, padding):
            return (size + 2*padding - (kernel_size - 1) - 1) // stride + 1

        # Height calculation
        H_out = conv2d_size_out(H_in, kernels[0], strides[0], paddings[0])  # After conv1
        H_out = conv2d_size_out(H_out, kernels[1], strides[1], paddings[1])  # After conv2
        H_out = conv2d_size_out(H_out, kernels[2], strides[2], paddings[2])  # After conv3
        H_out = conv2d_size_out(H_out, kernels[3], strides[3], 0)           # After pool1
        H_out = conv2d_size_out(H_out, kernels[4], strides[4], 0)           # After pool2

        # Width calculation
        W_out = conv2d_size_out(W_in, kernels[0], strides[0], paddings[0])  # After conv1
        W_out = conv2d_size_out(W_out, kernels[1], strides[1], paddings[1])  # After conv2
        W_out = conv2d_size_out(W_out, kernels[2], strides[2], paddings[2])  # After conv3
        W_out = conv2d_size_out(W_out, kernels[3], strides[3], 0)           # After pool1
        W_out = conv2d_size_out(W_out, kernels[4], strides[4], 0)           # After pool2

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_channels[2] * H_out * W_out, 128)
        self.fc2 = nn.Linear(128, N_out)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool1(x)
        x = self.pool2(x)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x
    
    def num_flat_features(self, x):
        """
        To properly flatten the tensor, it is necessary to find how many features will be created after the flattening.
        num_flat_features multiplies channels * height * width to determine the size of each entry.
        """

        size = x.size()[1:]  # all dimensions except batch
        num_features = 1
        for s in size:
            num_features *= s
        return num_features



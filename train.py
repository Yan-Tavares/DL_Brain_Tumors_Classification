import torch
import time
from src.convolution_network import CNN

############################################
# Settings
############################################
net = CNN(N_in              = 1,
          N_out             = 3,
          H_in              = 492,
          W_in              = 492,
          hidden_channels   = [6,6,6,16,16],
          kernels           = [5,5,5,5,5],
          strides           = [1,2,1,5,5],
          paddings          = [1,1,1])

num_batches = 10
############################################
# Check GPU
############################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

net.to(device)
############################################
# Testing network speed
############################################
test_speed = True
if test_speed:
    print("-------------Speed tests--------------------")
    # print(f"Time taken to load batch_1: {elapsed_time:.2f} seconds")

    start_time = time.time()
    for i in range(num_batches):
        X,Y = torch.load(f'data/tensors_batch_{i}.pt')
        X,Y = X.to(device),Y.to(device)

        lamb_hat = net.forward(X)
        print(f"Batch {i} fowarded", end='\r', flush=True)
        del X,Y,lamb_hat

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken to forward {num_batches} batches: {elapsed_time:.2f} seconds")


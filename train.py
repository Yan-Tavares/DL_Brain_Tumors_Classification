import torch
import torch.nn as nn
import numpy as np
import time
from src.convolution_network import CNN
import plotly.graph_objects as go

############################################
# Settings
############################################
net = CNN(N_in              = 1,
          N_out             = 3,
          H_in              = 490,
          W_in              = 490,
          hidden_channels   = [8,24,24], #Original was [6,16,16] # Best [8,24,24]
          kernels           = [5,5,5,5,5], #Original was [3,3,3,3,3] # Best [5,5,5,5,5]
          strides           = [1,5,1,5,1], #Original was [1,2,1,2,1] # Best [1,5,1,5,1]
          paddings          = [2,2,2], #Original was [1,1,1] # Best [2,2,2]
          hidden_layers     = [150,15]) #Original was 128 # Best [150,15]


early_stopping = True

num_batches = 100
unseen_batches = 30
val_batches = 20
seen_batches = num_batches - unseen_batches
train_batches = seen_batches - val_batches 

net.info()

# Define seed
seed = 42*0
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def init_weights(m):
    if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

net.apply(init_weights)

############################################
# Check GPU
############################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

net.to(device)

############################################
# Train the network
############################################
criterion = nn.CrossEntropyLoss()  # Cross-entropy loss

learning_rate = 0.0001
optimizer = torch.optim.Adam(net.parameters(), lr= learning_rate)
epochs = 100

train_losses = []
val_losses = []

start_time = time.time()
print("-------------Training--------------------")
# Load the data
for epoch in range(epochs):
    for i in range(train_batches):
        net.train()  # Set the network to training mode
        X,Y = torch.load(f'data/tensors_batch_{i}.pt')
        X,Y = X.to(device),Y.to(device)

        # Forward pass
        lamb_hat = net.forward(X)

        # Compute the loss
        train_loss = criterion(lamb_hat, Y)

        # Backward pass
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        train_losses.append(train_loss.item())
        print(f"Epoch {epoch}, Batch {i}", end='\r', flush=True)

        # Delete X and Y to free memory
        del X, Y
        torch.cuda.empty_cache()

    # Validation
    net.eval()
    val_loss = 0
    for i in range(train_batches, train_batches + val_batches):
        X,Y = torch.load(f'data/tensors_batch_{i}.pt')
        X,Y = X.to(device),Y.to(device)

        # Forward pass
        lamb_hat = net.forward(X)

        # Compute the loss
        val_loss += criterion(lamb_hat, Y).item()

        # Delete X and Y to free memory
        del X, Y
        torch.cuda.empty_cache()

    
    val_losses.append(val_loss/val_batches)
    print(f"Epoch {epoch}:\n - Validation loss: {val_loss/val_batches}\n - Train loss: {train_loss.item()}")

    dv = 0
    dv_list = []

    if not len(val_losses) < 6 and early_stopping and epoch > 10:
        for e in range(-5,0):
                dv = (val_losses[e] - val_losses[e-1])
                dv_list.append(dv)
    
        if sum(dv_list)/len(dv_list) > 0:
            print("Early stopping")
            break
        
        if dv_list[-2] < 0 and dv_list[-1] > 0:
            print(" - Minima case")
            torch.save(net.state_dict(), 'models/CNN_Minima.pth')
    
# Save the model
torch.save(net.state_dict(), 'models/CNN.pth')
print("\nTraining complete and model saved")
print(f"Time: {time.time() - start_time}, seconds\n")


# Plot losses using Plotly with information box
fig = go.Figure(data=go.Scatter(y=train_losses, mode='lines', name='Training Loss'))
fig.add_trace(go.Scatter(x = np.arange(1,len(val_losses)+1) * (train_batches) , y= val_losses, mode='lines', name='Validation Loss'))

# Add information box as an annotation
info_text = f"Learning Rate: {learning_rate}<br>Kernel Sizes: {net.kernels}, <br>Strides: {net.strides}, <br>Padding: {net.paddings}, <br>Hidden Channels: {net.hidden_channels}"
fig.add_annotation(
    text=info_text,
    xref="paper", yref="paper",
    x=0.95, y=0.95,
    showarrow=False,
    align="left",
    bordercolor="black",
    borderwidth=1,
    borderpad=4,
    bgcolor="lightgrey",
    opacity=0.8
)

fig.update_layout(
    xaxis_title='Batch passes',
    yaxis_title='Loss'
)

#Save the plot as html
fig.write_html("runs/Loss_plot.html")

fig.show()


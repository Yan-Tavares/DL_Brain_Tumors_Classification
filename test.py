import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

from src.convolution_network import CNN

net = CNN(N_in              = 1,
          N_out             = 3,
          H_in              = 490,
          W_in              = 490,
          hidden_channels   = [8,24,24], #Original was [6,16,16] # Best [8,24,24]
          kernels           = [5,5,5,5,5], #Original was [3,3,3,3,3] # Best [5,5,5,5,5]
          strides           = [1,5,1,5,1], #Original was [1,2,1,2,1] # Best [1,5,1,5,1]
          paddings          = [2,2,2], #Original was [1,1,1] # Best [2,2,2]
          hidden_layers     = [150,15]) #Original was 128 # Best [150,15]

net.load_state_dict(torch.load('models/CNN_Minima.pth'))

num_batches = 100
unseen_batches = 30
val_batches = 20
seen_batches = num_batches - unseen_batches
train_batches = seen_batches - val_batches

preds = []
Y_list = []

acc_list = []
with torch.no_grad():
    for batch_idx in range(seen_batches, seen_batches + unseen_batches):
        net.eval()  # Set the network to evaluation mode
        X, Y = torch.load(f'data/tensors_batch_{batch_idx}.pt')
    
        # Forward pass
        lamb_hat = torch.sigmoid(net.forward(X))
        lamb_hat /= torch.sum(lamb_hat, dim=1).view(-1, 1)
    
        for row_idx in range(lamb_hat.size(0)):
            max_idx = torch.argmax(lamb_hat[row_idx])
            lamb_hat[row_idx] = torch.zeros_like(lamb_hat[row_idx])
            lamb_hat[row_idx][max_idx] = 1

        preds.append(lamb_hat)
        Y_list.append(Y)

        del X, Y, lamb_hat
        print(f"Batch {batch_idx + 1}", end='\r', flush=True)


preds = torch.cat(preds, dim=0)
Y = torch.cat(Y_list, dim=0)
acc = 1 - torch.sum(torch.abs(preds - Y)) / (2 * Y.size(0))


print(f"Accuracy: {acc}")
f1 = f1_score(Y.numpy(), preds.numpy(), average='macro')
print(f"F1 Score: {f1}")

#Torch.argmax(Y, dim=1).numpy()
# Converts vectors to single integer with the class indices.
#<1, 0, 0> → 0
#<0, 0, 1> → 2
Y_labels = np.argmax(Y, axis=1)
preds_labels = np.argmax(preds, axis=1)

# Compute confusion matrix
cm = confusion_matrix(Y_labels, preds_labels)
print("Confusion Matrix:\n", cm)


# Define class labels
class_labels = ['Meninginoma', 'Glioma', 'Pituitary']  # Replace with your actual class names

# Create Heatmap with labeled axes
fig = go.Figure(data=go.Heatmap(
    z=cm,
    x=class_labels,  # Predicted Labels
    y=class_labels,  # True Labels
    colorscale='Blues',
    text=cm,  # Text to display in each cell
    texttemplate="%{text}",
    textfont={"size": 16},
    hoverongaps=False
))

# Update layout with titles and axis labels
fig.update_layout(
    xaxis=dict(title='Predicted Labels'),
    yaxis=dict(title='True Labels'),
    width=600,
    height=600
)

# Save the figure as an HTML file
fig.write_html("runs/Confusion_matrix.html")

# Display the figure
fig.show()


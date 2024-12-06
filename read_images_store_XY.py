from src import data_preprocess as dp
import time
import torch

############################################
# Settings
############################################
save_tensors = True
num_batches = 100

############################################
#Data Preprocessing
############################################

#Load the data in a dataframe
df = dp.dataframe_data("data/archive1/_annotation.json")
dp.inspect_data(df)
# Apply one hot encoding
df = dp.one_hot_encode(df)
# Randomize the dataframe
df = df.sample(frac=1,random_state=42).reset_index(drop=True)
print(df)

# Determine the size of each batch
batch_size = len(df) // num_batches

print(f"Number of batches: {num_batches}")
print(f"Batch size: {batch_size}")

if save_tensors:
    for i in range(num_batches):
        # Get the start and end indices for the current batch
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size if i != num_batches - 1 else len(df)
        
        # Slice the dataframe to get the current batch
        df_batch = df.iloc[start_idx:end_idx].reset_index(drop=True)
        
        # Process the batch (example: convert to tensor)
        X, Y = dp.df_to_torch_tensor(df_batch, "data/archive1", H_in=490, W_in=490)
        
        # Save the tensor to a file
        torch.save((X, Y), f'data/tensors_batch_{i}.pt')
        
        # Clear the tensor from memory
        print(f"Saved batch {i+1}, {X.shape}, {Y.shape}", end='\r', flush=True)

print("\nData preprocessing complete")
############################################
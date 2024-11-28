import json
import pandas as pd
import numpy as np
import torch
from PIL import Image

#Create the labels form the data present in _annotations.json file

def dataframe_data(json_file_relative_path):
    """
    Create the labels form the data present in _annotations.json file
    """
    
    file = open(json_file_relative_path)
    data = json.load(file)

    #Data is a dictionary
    #In data each file name is a key and the value is the class in the format of another dictionary {'class: 'class_name'}

    #Reformat the data to a key and the class value
    for key in data:
        data[key] = data[key]['class']
    
    #Make a pandas dataframe
    df = pd.DataFrame(data.items(), columns=['Image', 'Class'])

    return df

def inspect_data(df):
    """
    Inspect the data
    """

    print("\nDataframe preview: \n",df)
    print("Number of unique classes: ",df['Class'].nunique())
    print("Occurences of each class:", df['Class'].value_counts().to_dict())

def one_hot_encode(df):
    """
    One hot encode the classes
    """
    
    df_encoded = pd.get_dummies(df, columns=['Class'], prefix='Class')
    # Makethe True and false values to 1 and 0

    # Convert only the one-hot encoded columns to integers
    class_columns = [col for col in df_encoded.columns if col.startswith('Class_')]
    df_encoded[class_columns] = df_encoded[class_columns].astype(int)
    
    return df_encoded

def df_to_torch_tensor(df,image_folder_path, H_in=492, W_in=492):
    """
    Convert the dataframe to a torch tensor.
    Imput:
    - df: dataframe with the the file names and the classes
    - image_folder_path: path to the folder with the images

    """
    #Make a Y tensor with the classes
    Y = torch.tensor(df.iloc[:,1:].values, dtype=torch.float32)

    #Make a empity X tensor with 4 dimensions (batch size, channels, height, width)
    X = torch.empty((len(df), 1, H_in, W_in), dtype=torch.float32)

    for i in range(len(df)):
        image_path = image_folder_path + "/" + df.loc[i,'Image']
        with Image.open(image_path).convert('L') as img:
            img = img.resize((W_in, H_in))  # Resize the image
            img = np.array(img)/ 255.0  # Normalize image values to [0, 1]
            img = torch.tensor(img, dtype=torch.float32)
            
            #From the package PIL the image is in the format (height, width, channels)
            #Each pixel has a array with 3 values (R,G,B)
            #We need to take each value from the pixel array and make a individual matrix, such that we have 3 matrices
            #This can be done using the permute function
            img = img.unsqueeze(0)

            X[i] = img

            if i % 10 == 0:
                print(f"\rLoaded {i} images", end='', flush=True)

    return X,Y


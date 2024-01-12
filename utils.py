
import random
import torch
import json
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

config_path = "config.json"

with open(config_path, "r", encoding="utf8") as f:
    config = json.load(f)


def set_seed(seed = 0):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def tokenize(caption_list):
    for i in range(len(caption_list)):
        caption_list[i] = caption_list[i].lower()
        tokens = caption_list[i].split()
        for j, token in enumerate(tokens):
            word = ""
            for char in token:
                if char.isalpha():
                    word += char
                elif i == "'":
                    word += char
            tokens[j] = word
        caption_list[i] = ["<s>"] + tokens + ["</s>"]
    return caption_list

def batch_sort(X, y_tensor, sent_L):
    sorted_indices = np.argsort(np.array(sent_L))[::-1]
    X = np.array(X)[sorted_indices].tolist()
    y_tensor = torch.tensor(np.array(y_tensor.detach().numpy())[sorted_indices].tolist())
    sent_L = np.array(sent_L)[sorted_indices].tolist()
    return X, y_tensor, sent_L


def sentenceListToTokenTensor(sentenceList, token_to_id):
    maxL = max(len(sublist) for sublist in sentenceList) 
    train_Y_matrix = []
    sent_L = []
    for i in range(len(sentenceList)):
        tmpY_L = []
        sentence = sentenceList[i]
        sent_L.append(len(sentence))
        for j in range(len(sentence)):
            tokenID = 0
            try:
                tokenID = token_to_id[sentence[j]]
            except:
                tokenID = token_to_id['<UNK>']
    
            tmpY_L.append(tokenID)
        
            if j == len(sentence) -1 and j < maxL -1:
                padList = [token_to_id['<PAD>']] * ( (maxL-1) - (len(sentence)) + 1 ) # B - A + 1  
                tmpY_L = tmpY_L + padList
        train_Y_matrix.append(tmpY_L)
    train_Y_matrix = np.array(train_Y_matrix)
    train_Y_tensor = torch.tensor(train_Y_matrix)
    return train_Y_tensor, sent_L

def show_predictions_with_image(predList, numToShow):
    counter = 0
    tmp = predList

    fig, axs = plt.subplots(numToShow, 1, figsize=(10, 10))

    axs = axs.flatten()
    for instance in tmp:
        img_path = instance["file_name"]
        real = " ".join(instance["real"])
        pred = " ".join(instance["pred"])
        title = "Real: " + real + "\n" + "Pred: " + pred
        image = Image.open(img_path)

        axs[counter].imshow(image)
        axs[counter].axis("off")
        axs[counter].set_title(title, fontsize=10) 

        counter += 1
        if counter == numToShow:
            break

    # Adjust the layout
    plt.tight_layout()
    plt.show()
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.transform import resize
import torchvision.transforms as transforms


from utils import sentenceListToTokenTensor
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

config_path = "config.json"

with open(config_path, "r", encoding="utf8") as f:
    config = json.load(f)

if config["use_gpu"]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

def forwardTestSGD(model, x, caption_length, beamWidth, startTokenID):
    model.eval()
    encoder_out = model.encoder([x])
    instanceToken = []
    for k in range(beamWidth):
        instance = {}
        instance["tokenIDs"] = [startTokenID]
        instance["Probs"] = 1
        instance["alphas"] = []
        instanceToken.append(instance)

    for t in range(1, caption_length):        
        buffer = []

        for i in range(beamWidth):
            probs, _, alphas = model.decoder(encoder_out, torch.tensor(instanceToken[i]["tokenIDs"]).unsqueeze(0).to(device), [len(instanceToken[i]["tokenIDs"])+1])
            probs = probs.to(device)
            probs = probs.squeeze(0)
            probs = probs[-1, :].to(device)
            probs = F.softmax(probs, dim  =0)
            alphas = alphas.squeeze(0)
            alphas = alphas[ -1, :].to(device)  
            alphas = alphas.cpu().detach().numpy()
            attention_map = np.reshape( alphas, (7,7) )
            top_values, top_indices = torch.topk(probs, beamWidth, largest=True)

            for j in range(beamWidth):
                instance = {}
                instance["prevTokens"] = instanceToken[i]["tokenIDs"]
                instance["curToken"] = top_indices[j].item()
                instance["prob"] = top_values[j].item()*instanceToken[i]["Probs"]
                instance["alphas"] = attention_map
                buffer.append(instance)

        buffer = sorted(buffer, key=lambda x: x['prob'], reverse=True)

        for k in range(beamWidth):
            instanceToken[k]["tokenIDs"] = buffer[k]["prevTokens"] + [buffer[k]["curToken"]]
            instanceToken[k]["Probs"] = buffer[k]["prob"]
            instanceToken[k]["alphas"].append(buffer[k]["alphas"])

    return instanceToken[0]["tokenIDs"], instanceToken[0]["alphas"]


def finalTestBeamSearch(dataloader, model, id_to_token, token_to_id, beamWidth, startTokenID):
    pred_list = []
    model.eval()
    end_token = config['end_token']
    with torch.no_grad():
        for i, (X,y) in enumerate(dataloader):
            for j in range(len(X)):
                tokenIDGenerated, _ = forwardTestSGD(model, X[j], len(y[j]), beamWidth, startTokenID)
                y_tensor, sent_L = sentenceListToTokenTensor([y[j]], token_to_id) 
                y_tensor = y_tensor.to(device)
                my_dict ={}
                my_dict["file_name"] = X[j]
                my_dict["real"] = y[j]
                resultTokens = []
                for z in range(len(tokenIDGenerated)):
                    token = id_to_token[tokenIDGenerated[z]]
                    resultTokens.append(token)
                    
                    if token == end_token:
                        break
                my_dict["pred"] = resultTokens
                pred_list.append(my_dict)
                
    return pred_list

def BLEU_Score(pred_list):
    actual, predicted, filenames= list(), list(), list()
    for file in pred_list:
        filenames.append(file['file_name'])
        actual.append(file['real'])
        predicted.append(file['pred'])

    chencherry = SmoothingFunction()

    print("BLEU-1: %f" % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0), smoothing_function=chencherry.method1))
    print("BLEU-2: %f" % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0), smoothing_function=chencherry.method1))
    print("BLEU-3: %f" % corpus_bleu(actual, predicted, weights=(0.33, 0.33, 0.33, 0), smoothing_function=chencherry.method1))
    print("BLEU-4: %f" % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25),smoothing_function=chencherry.method1))

def show_attention_map(model, pred_image,id_to_token, token_to_id):
    start_token = config["start_token"]
    end_token = config["end_token"]

    img_path = pred_image['file_name']
    real = pred_image['real']
    pred = pred_image['pred']


    image = Image.open(img_path)

    tokenIDGenerated, attention_map = forwardTestSGD(model, img_path, len(real), 1, token_to_id[start_token])

    resultTokens = []
    for z in range(len(tokenIDGenerated)):
        token = id_to_token[tokenIDGenerated[z]]
        resultTokens.append(token)
        if token == end_token:
            break


    numToShow = len(resultTokens)-1
    numColumns = 3
    numRow = numToShow//numColumns +1
    real = " ".join(real)
    pred = " ".join(pred)
    

    print("Real:", real)
    print("Prediction:", pred)

    fig, axs = plt.subplots(numRow, numColumns, figsize=(10, 10))

    counter1 =0
    counter2 =0

    transform = transforms.Compose([transforms.Resize((256, 256))])
    image = transform(image)

    for counter in range(0, len(resultTokens)):
        axs[counter1, counter2].imshow(image) 
        axs[counter1, counter2].axis("off")
        axs[counter1, counter2].set_title(resultTokens[counter])
        if counter == 0:
            counter2+=1
            if counter2 == numColumns:
                counter1+=1
                counter2 =0
            continue

        resized_attention_map = resize(attention_map[counter-1], (256, 256))
        axs[counter1, counter2].imshow(resized_attention_map, alpha=0.5,cmap = 'gist_heat')

        counter2+=1
        if counter2 == numColumns:
            counter1+=1
            counter2 =0
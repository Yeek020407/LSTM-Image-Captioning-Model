import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import json


config_path = "config.json"
with open(config_path, "r", encoding="utf8") as f:
    config = json.load(f)

if config["use_gpu"]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")



class ImageCaptioningModel(nn.Module):
    def __init__(self,
            attention_dim, 
            embed_dim, 
            hidden_dim, 
            vocab_size,
            dropout ,
            id_to_token,
            w2v):

        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(attention_dim=attention_dim,
                               embed_dim=embed_dim,
                               hidden_dim= hidden_dim,
                               vocab_size= vocab_size,
                               dropout=dropout,
                               id_to_token = id_to_token,
                               w2v= w2v)
    
    def forward(self, x, encoded_captions, caption_lengths):
        encoder_out = self.encoder(x)
        predictions, tokenIDGenerated, alphas = self.decoder(encoder_out, encoded_captions, caption_lengths)
        return predictions, tokenIDGenerated, alphas


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        resnet = torchvision.models.resnet101(pretrained = True)  
        for param in resnet.parameters():
            param.requires_grad_(False)

        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.transform = transforms.Compose([
                        transforms.Resize((256, 256)),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])

    def forward(self, image_filename):
        batch_tensor_list= []
        for i,path in enumerate(image_filename):
            image = Image.open(path).convert('RGB')
        
            image_tensor = self.transform(image)
            batch_tensor_list.append(image_tensor.unsqueeze(0))
            
        batch_tensor= torch.cat(batch_tensor_list, dim=0)
        batch_tensor = batch_tensor.to(device)
        features = self.resnet(batch_tensor)
        batch_size, num_feature_maps, height, width = features.size()
        features = features.permute(0, 2, 3, 1)
        features = features.view(batch_size, height*width, num_feature_maps)
        return features
    

class Attention(nn.Module):
    def __init__(self, encoder_dim, hidden_dim, attention_dim):
        super().__init__()
        self.encoderAttentionLayer = nn.Linear(encoder_dim, attention_dim)  
        self.combineAttentionLayer = nn.Linear(attention_dim, 1)  
        self.hiddenAttentionLayer = nn.Linear(hidden_dim, attention_dim)  

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  
        self.beta = nn.Linear(hidden_dim, encoder_dim) 
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, encoder_out, hidden):
        encoderAttention = self.encoderAttentionLayer(encoder_out) 
        hiddenAttention = self.hiddenAttentionLayer(hidden) 
        combineAttention = self.combineAttentionLayer(self.relu(encoderAttention + hiddenAttention.unsqueeze(1))).squeeze(2)  
        alpha = self.softmax(combineAttention)  
        weighted_encoder_out = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  
        betaGate = self.sigmoid(self.beta(hidden))  
        weighted_encoder_out = betaGate * weighted_encoder_out
        return weighted_encoder_out, alpha


class Decoder(nn.Module):
    def __init__(self, attention_dim, embed_dim, hidden_dim, vocab_size, id_to_token, w2v, encoder_dim=2048, dropout=0.5):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.attention = Attention(encoder_dim, hidden_dim, attention_dim)  
        self.embedding = nn.Embedding(vocab_size, embed_dim)  
        self.dropout = nn.Dropout(p=dropout)
        self.lstm = nn.LSTMCell(embed_dim + encoder_dim, hidden_dim, bias=True)  
        self.init_h = nn.Linear(encoder_dim, hidden_dim) 
        self.init_c = nn.Linear(encoder_dim, hidden_dim)  
        self.fc = nn.Linear(hidden_dim, vocab_size)  
        self.init_weights()  
        self.initEmbeddingLayer(id_to_token, w2v)

    def init_weights(self):
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)
            
    def toVectorMatrix(self,id_to_token, w2v):
        vector_list = []
        vect = None
        for id, token in id_to_token.items():
            try:
                vect =  w2v[id_to_token[id]]
            except:
                vect = [0] * w2v.vector_size
            vector_list.append(vect)
        vect_mat = np.array(vector_list)
        return vect_mat

    def initEmbeddingLayer(self, id_to_token, w2v):
        vect_mat = self.toVectorMatrix(id_to_token, w2v)
        vect_tensor = torch.tensor(vect_mat) 
        self.embedding.weight.data.copy_(vect_tensor)
        for p in self.embedding.parameters():
            p.requires_grad = False

    def init_lstm_state(self, encoder_out):
        encoder_out_mean = encoder_out.mean(dim=1)
        h = self.init_h(encoder_out_mean) 
        c = self.init_c(encoder_out_mean)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        batch_size = encoder_out.size(0)
        embeddings = self.embedding(encoded_captions)  
        h, c = self.init_lstm_state(encoder_out)  
        caption_lengths = [x - 1 for x in caption_lengths]
        predictions = torch.zeros(batch_size, max(caption_lengths), self.vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(caption_lengths), encoder_out.size(1)).to(device)
        for cur_tkn in range(max(caption_lengths)):
            batch_size_max_L = sum([length > cur_tkn for length in caption_lengths])
            weighted_encoder_out, alpha = self.attention(encoder_out[:batch_size_max_L], h[:batch_size_max_L])
            h, c = self.lstm(torch.cat([embeddings[:batch_size_max_L, cur_tkn,:], weighted_encoder_out], dim=1), (h[:batch_size_max_L], c[:batch_size_max_L])) 
            preds = self.fc(self.dropout(h[:batch_size_max_L]))  
            predictions[:batch_size_max_L, cur_tkn, :] = preds
            alphas[:batch_size_max_L, cur_tkn, :] = alpha
        tokenIDGenerated = torch.argmax(predictions, dim= 2).to(device)
        return predictions, tokenIDGenerated, alphas
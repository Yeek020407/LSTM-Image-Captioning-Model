from torch.utils.data import Dataset
import random

class ImageCaptionDataset(Dataset):
    def __init__(self, file_names, caption_tokens):
        self.X = file_names
        self.y = caption_tokens

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class CustomLoader:
    def __init__(self, dataset, batch_size=66, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        X, y = dataset[:]
        
        if shuffle:
            combined = list(zip(X, y))
            random.shuffle(combined)

            # Split the shuffled pairs back into separate lists
            X, y = zip(*combined)
            
        
        modulo = len(dataset) % batch_size
        if modulo != 0:
            raise ValueError("Batch_size cannot fit in dataset")
            
        
        self.numBatch = int(len(dataset)/batch_size) # No error handling
        
        self.X_list = []
        self.y_list = []
        
        
        for fold in range(0, self.numBatch):
            start, end = fold*batch_size, (fold+1)*batch_size
            self.X_list.append(X[start:end])
            self.y_list.append(y[start:end])
    
    def __len__(self):
        return self.numBatch
    
    def __getitem__(self, idx):
        return self.X_list[idx], self.y_list[idx]
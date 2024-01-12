from collections import defaultdict 
import json
from preprocessing import *
from gensim.models import KeyedVectors
from utils import *


def create_vocab():
    config_path = "config.json"
    with open(config_path, "r", encoding="utf8") as f:
        config = json.load(f)
    pad_token = config["pad_token"]
    start_token = config["start_token"]
    end_token = config["end_token"]
    unk_token = config["unk_token"]
    
    path_to_img = config["train_img_path"]
    annotation_path = config["train_annotation_path"]

    img_caption_df = get_annotation_img_df(path_to_img,annotation_path)
    val_file_name = img_caption_df["file_name"].tolist()
    val_Y = img_caption_df["caption"].tolist()
    val_Y = tokenize(val_Y)

    token_to_id = defaultdict(lambda: len(token_to_id))

    _ = token_to_id[pad_token]
    _ = token_to_id[unk_token]
    _ = token_to_id[start_token]
    _ = token_to_id[end_token]

    for sentence in val_Y:
        for token in sentence:
            _ = token_to_id[token]
            
    token_to_id = dict(token_to_id)

    id_to_token = {id_: token for token, id_ in token_to_id.items()}
    
    with open('id_to_token.json', 'w') as json_file:
        json.dump(id_to_token, json_file)

    with open('token_to_id.json', 'w') as json_file:
        json.dump(token_to_id, json_file)


if __name__ == "__main__":
    create_vocab()
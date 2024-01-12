
import json
import pandas as pd

config_path = "config.json"
with open(config_path, "r", encoding="utf8") as f:
    config = json.load(f)

def load_annotations(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def get_annotation_img_df(path_to_img,annotation_path):

    annotations = load_annotations(annotation_path)

    data = annotations['annotations']
    img_cap_pairs = []
    for sample in data:
        img_cap_pairs.append([sample['image_id'], sample['caption']])
    captions = pd.DataFrame(img_cap_pairs, columns=['image_id', 'caption'])
    images_df = pd.DataFrame(annotations['images'])
    combined_df = pd.merge(captions, images_df, how='left', left_on='image_id', right_on='id')
    combined_df = combined_df[['file_name','caption']]
    combined_df['file_name'] = path_to_img + combined_df['file_name']
    return combined_df
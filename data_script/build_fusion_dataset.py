import pandas as pd
from datasets import Dataset
from PIL import Image

data = pd.read_csv("./mixed/meta_info.csv")

def gen():
    data = pd.read_csv("./mixed/meta_info.csv")
    for _, row in data.iterrows():
        image = Image.open(f"./mixed/{row['mix']}").resize((64,64)).convert("RGBA")
        condition1 = Image.open(f"./mixed/{row['mixed1']}").resize((64,64)).convert("RGBA")
        condition2 = Image.open(f"./mixed/{row['mixed2']}").resize((64,64)).convert("RGBA")
        yield {"image": image,
               "condition1":condition1,
               "condition2":condition2}

dataset = Dataset.from_generator(gen)


from datasets import DatasetDict
dataset_dict = DatasetDict({"train":dataset})

# dataset.push_to_hub("ChengAoShen/emoji_fusion",token = "")


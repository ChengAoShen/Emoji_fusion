import os
import pandas as pd
from PIL import Image
from datasets import load_dataset
image_size=64

out_list = dict(
    file_name = [],
    text = []
)

for dir_name in os.listdir("original_emoji"):
    if dir_name == ".DS_Store":
        continue

    for id,file in enumerate(os.listdir("original_emoji/" + dir_name)):
        if file == ".DS_Store":
            continue

        describtion = file[:-4]
        describtion = describtion.replace(":", ",")
        img = Image.open(f"original_emoji/{dir_name}/{file}")
        img = img.convert("RGBA")
        img = img.resize((image_size, image_size))

        file_name = f"{id:04d}.png"
        if not os.path.exists(f"emoji/{dir_name}"):
            os.makedirs(f"emoji/{dir_name}")
        img.save(f"emoji/{dir_name}/{file_name}")

        out_list["file_name"].append(f"{dir_name}/{file_name}")
        out_list["text"].append(f"{dir_name},{describtion}")

pd.DataFrame(out_list, columns=["file_name", "text"]).to_csv("emoji/metadata.csv", index=False)
from datasets import load_dataset
dataset = load_dataset("imagefolder", data_dir="./emoji_with_text")
dataset.push_to_hub("ChengAoShen/emoji_for_diffusion",token = "")

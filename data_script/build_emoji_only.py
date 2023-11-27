# %%
# import os
# import pandas as pd
# from PIL import Image
# root = "emoji_without_text/"
# out = "emoji_only/"
# count_dict = {}
#
# id = 0
# for dir_name in os.listdir(root):
#     if dir_name == ".DS_Store":
#         continue
#
#     count = 0
#     for file in os.listdir(root + dir_name):
#         if file.endswith(".png"):
#             count += 1
#             id+=1
#             image = Image.open(root + dir_name + "/" + file).convert("RGBA").resize((64,64))
#             image.save(f"{out}{id:05d}.png")
#         os.remove(root + dir_name + "/" + file)
#     os.rmdir(root + dir_name)
#     count_dict[dir_name] = count
#
# pd.DataFrame(count_dict,index=[0]).T.to_csv(f"{root}count.csv")


# %%
from datasets import load_dataset
dataset = load_dataset("imagefolder", data_dir="./emoji_only/")


# %%
# dataset.push_to_hub("ChengAoShen/emoji_for_diffusion",token = "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")


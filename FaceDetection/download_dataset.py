import os
import requests
import io
import hashlib
import json

DATASET_DIR = "./data/"
if not os.path.exists(DATASET_DIR):
    os.mkdir(DATASET_DIR)

def get_file(url, save_dir):
    image_data = requests.get(url)
    image = io.BytesIO(image_data.content)
    file_name = hashlib.sha256(image.getvalue()).hexdigest() + ".jpg"
    save_path = os.path.join(save_dir, file_name)
    with open(save_path,"wb") as f:
        f.write(image.getvalue())
    return file_name

with open("dataset_online.json") as f:
    dataset_raw = [json.loads(line) for line in f.read().split("\n") if len(line) > 1]

to_do = len(dataset_raw)
count = 1
offline_dataset = {}
for example in dataset_raw:
    print(f"\rDowloading file {count} of {to_do}.",end="")
    count += 1
    try:
        file_name = get_file(example["content"], DATASET_DIR)

        bboxes = []
        for feature in example["annotation"]:
            bboxes.append(
                {
                    "x1":min( [ feature["points"][0]["x"], feature["points"][1]["x"] ]),
                    "x2":max( [ feature["points"][0]["x"], feature["points"][1]["x"] ]),
                    "y1":min( [ feature["points"][0]["y"], feature["points"][1]["y"] ]),
                    "y2":max( [ feature["points"][0]["y"], feature["points"][1]["y"] ]),
                })

        offline_dataset[file_name] = bboxes
    except:
        pass

with open(os.path.join(DATASET_DIR, "dataset.json"), "w") as f:
    json.dump(offline_dataset, f)
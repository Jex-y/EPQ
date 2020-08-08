import os
import requests
import io
import hashlib
import json
import yaml
import numpy as np

DATASET_DIR = "./data/"
CLASS_FILE = "classes.yml"
TEST_SPLIT = 0.2
RANDOM_SEED = 42

def main():
    if not os.path.exists(DATASET_DIR):
        os.mkdir(DATASET_DIR)

    with open("dataset_online.json") as f:
        dataset_raw = [json.loads(line) for line in f.read().split("\n") if len(line) > 1]

    np.random.seed(RANDOM_SEED)
    np.random.shuffle(dataset_raw)

    to_do = len(dataset_raw)
    count = 1
    examples = 0
    classes_list = ["DONOTUSEME"]

    num_train = int(to_do * (1-TEST_SPLIT))

    num_test = int(to_do * TEST_SPLIT)

    examples_dict = {
        "train":[],
        "test":[]
    }

    for example in dataset_raw:
        print(f"\rProcessing file {count} of {to_do}.",end="")
        count += 1
        try:
            filename = get_file(example["content"], DATASET_DIR)

            xmins, xmaxs, ymins, ymaxs, classes, classes_text = ([],[],[],[],[],[])

            for feature in example["annotation"]:
                classes_text.append(feature["label"][0])

                if classes_text[-1] not in classes_list:
                    classes_list.append(classes_text[-1])

                classes.append(classes_list.index(classes_text[-1])) 
                
                x_values = [ feature["points"][0]["x"], feature["points"][1]["x"] ]
                y_values = [ feature["points"][0]["y"], feature["points"][1]["y"] ]

                xmins.append(min( x_values ))
                xmaxs.append(max( x_values ))
                ymins.append(min( y_values ))
                ymaxs.append(max( y_values ))

            
            example = {
                'image/height': example["annotation"][0]["imageHeight"],
                'image/width': example["annotation"][0]["imageWidth"],
                'image/filename': filename,
                'image/source_id': example["content"],
                'image/format': "jpeg",
                'image/object/bbox/xmin': xmins,
                'image/object/bbox/xmax': xmaxs,
                'image/object/bbox/ymin': ymins,
                'image/object/bbox/ymax': ymaxs,
                'image/object/class/label': classes,
            }

            if examples < num_test:
                examples_dict["test"].append(example)
            else:
                examples_dict["train"].append(example)
            examples += 1
        except Exception as e:
            print(e)
            pass
    
    num_train = examples - num_test
    class_dict = dict(enumerate(classes_list))
    class_dict.pop(0)

    offline_dataset = {
        "classes": {
            "num_classes":len(classes_list),
            "class_names":class_dict,
        },
        "examples": examples_dict,
    }

    print("\nFinished processing dataset.")
    print(f"{num_train} train examples and {num_test} test examples.")

    with open(os.path.join(DATASET_DIR, "dataset.yml"), "w") as f:
        yaml.dump(offline_dataset, f)

    print("Wrote dataset.yaml.")

def get_file(url, save_dir):
    image_data = requests.get(url)
    image = io.BytesIO(image_data.content)
    file_name = hashlib.sha256(image.getvalue()).hexdigest() + ".jpg"
    save_path = os.path.join(save_dir, file_name)
    with open(save_path,"wb") as f:
        f.write(image.getvalue())
    return file_name

if __name__ == "__main__":
    main()
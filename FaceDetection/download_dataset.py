import os
import requests
import io
import hashlib
import json
import yaml
import tensorflow as tf

DATASET_DIR = "./data/"
CLASS_FILE = "classes.yml"
TEST_SPLIT = 0.2

def main():
    if not os.path.exists(DATASET_DIR):
        os.mkdir(DATASET_DIR)

    with open("dataset_online.json") as f:
        dataset_raw = [json.loads(line) for line in f.read().split("\n") if len(line) > 1]

    to_do = len(dataset_raw)
    count = 1
    examples = 0
    classes_list = ["DONOTUSEME"]

    train_path = os.path.join(DATASET_DIR, "train.tfrecord")
    num_train = int(to_do * (1-TEST_SPLIT))
    train_writer = tf.io.TFRecordWriter(train_path)

    test_path = os.path.join(DATASET_DIR, "test.tfrecord")
    num_test = int(to_do * TEST_SPLIT)
    test_writer = tf.io.TFRecordWriter(test_path)

    for example in dataset_raw:
        print(f"\rProcessing file {count} of {to_do}.",end="")
        count += 1
        try:
            filename, encoded_image_data = get_file_bytes(example["content"], DATASET_DIR, save=False)

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

        
            tf_example = tf.train.Example(features=tf.train.Features(feature={
                'image/height': int64_feature(example["annotation"][0]["imageHeight"]),
                'image/width': int64_feature(example["annotation"][0]["imageWidth"]),
                'image/filename': bytes_feature(str.encode(filename)),
                'image/source_id': bytes_feature(str.encode(example["content"])),
                'image/encoded': bytes_feature(encoded_image_data),
                'image/format': bytes_feature(b"jpeg"),
                'image/object/bbox/xmin': float_list_feature(xmins),
                'image/object/bbox/xmax': float_list_feature(xmaxs),
                'image/object/bbox/ymin': float_list_feature(ymins),
                'image/object/bbox/ymax': float_list_feature(ymaxs),
                'image/object/class/text': bytes_list_feature([str.encode(x) for x in classes_text]),
                'image/object/class/label': int64_list_feature(classes),
            }))
            if examples < num_test:
                test_writer.write(tf_example.SerializeToString())
            else:
                train_writer.write(tf_example.SerializeToString())
            examples += 1
        except Exception as e:
            print(e)
            pass
    
    num_train = examples - num_test
    class_dict = dict(enumerate(classes_list))
    class_dict.pop(0)

    offline_dataset = {
        "train":{
            "num_examples":num_train,
            "tfrecord_path":train_path
        },
        "test":{
            "num_examples":num_test,
            "tfrecord_path":test_path
        },
        "classes":{
            "num_classes":len(classes_list),
            "class_names":class_dict,
        },
    }

    print("\nFinished processing dataset.")
    print(f"{num_train} train examples and {num_test} test examples.")

    with open(os.path.join(DATASET_DIR, "dataset.yml"), "w") as f:
        yaml.dump(offline_dataset, f)

    print("Wrote dataset.yaml.")

def get_file_bytes(url, save_dir, save=True):
    image_data = requests.get(url)
    image = io.BytesIO(image_data.content)
    file_name = hashlib.sha256(image.getvalue()).hexdigest() + ".jpg"
    if save:
        save_path = os.path.join(save_dir, file_name)
        with open(save_path,"wb") as f:
            f.write(image.getvalue())
    return file_name, image.getvalue()

def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

if __name__ == "__main__":
    main()
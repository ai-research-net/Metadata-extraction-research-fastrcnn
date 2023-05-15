import os
import re
import zipfile
import json

import numpy as np
import tensorflow as tf
from tensorflow import keras
from mrcnn.utils import Dataset
import matplotlib.pyplot as plt
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
import keras

def extract_boxes(filename):
    with open(filename) as f:
        file_content = json.load(f)
        boxes = list()
        for box in file_content:
            if not isinstance(box[-1], str):
                continue
            else:
                boxes.append(box[1])
    return boxes

for f in os.scandir("../data/text_maps/"):
    if f.name.endswith(".json"):
        print(extract_boxes(f.path))
        break

class DocumentDataset(Dataset):
    # load the dataset definitions
    def load_dataset(self, dataset_dir, is_train=True): 
        self.add_class("dataset", 1, "author")
        self.add_class("dataset", 2, "title")
        self.add_class("dataset", 3, "doi")
        self.add_class("dataset", 4, "address")
        self.add_class("dataset", 5, "affiliation")
        self.add_class("dataset", 6, "email")
        self.add_class("dataset", 7, "journal")
        self.add_class("dataset", 8, "abstract")
        self.add_class("dataset", 9, "date")
        i = 0
        for image in os.scandir(dataset_dir):
            if (image.name.endswith(".jpg")):
                if is_train and i > 60:
                    continue
                if not is_train and i > 20:
                    continue
                i+=1
                image_id = image.name[:-4]
                image_path = dataset_dir + "/" + image.name
                annotation_path = dataset_dir + "/" + image_id + ".json"

                self.add_image("dataset", image_id=image_id, path=image_path, annotation_path=annotation_path)
	# load the masks for an image
    def load_mask(self, image_id):
        info = self.image_info[image_id]
        path = info['annotation_path']
        boxes, classes = self.extract_boxes(path)
        masks = np.zeros([679, 451, len(boxes)], dtype='uint8')
        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = int(box[1]), int(box[3])
            col_s, col_e = int(box[0]), int(box[2])
            masks[row_s:row_e, col_s:col_e, i] = 1
            class_ids.append(self.class_names.index(classes[i]))
        return masks, np.asarray(class_ids, dtype='int32')
    
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info["path"]

    def extract_boxes(self, filename):
        with open(filename) as f:
            file_content = json.load(f)
            boxes = list()
            class_ids = list()
            for box in file_content:
                if not isinstance(box[-1], str):
                    continue
                else:
                    boxes.append(box[1])
                    class_ids.append(box[-1])
        return boxes, class_ids

train_set = DocumentDataset()
train_set.load_dataset("../data/text_maps/", is_train=True)
train_set.prepare()
print("Train: %d" % len(train_set.image_ids))

test_set = DocumentDataset()
test_set.load_dataset("../data/text_maps/", is_train=False)
test_set.prepare()
print("Train: %d" % len(test_set.image_ids))

image_id = 0
image = train_set.load_image(image_id)
print(image.shape)
# load image mask
mask, class_ids = train_set.load_mask(image_id)
print(mask.shape)

for image_id in train_set.image_ids:
    info = train_set.image_info[image_id]
	# display on the console
    print(info)


class ModelConfig(Config):
    # Give the configuration a recognizable name
	NAME = "documents_cfg"
	# Number of classes (background + kangaroo)
	NUM_CLASSES = 1 + 9
	# Number of training steps per epoch
	STEPS_PER_EPOCH = 61
	BATCH_SIZE = 3
 
# prepare config
config = ModelConfig()

model = MaskRCNN(mode='training', model_dir='./', config=config)
model.load_weights('mask_rcnn_coco.h5', by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])

model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=15, layers='heads')

keras.models.save_model(model.keras_model,"final_model.hdf5")

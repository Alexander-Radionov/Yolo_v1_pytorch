import json

if __name__ == "__main__":

    classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
               "diningtable", "dog", "horse", "motorbike",
               "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    mapping = {i: class_name for i, class_name in enumerate(classes)}

    with open('classes_mapping.json', 'w') as f:
        json.dump(mapping, f, indent=4)
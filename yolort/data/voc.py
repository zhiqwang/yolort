import torch
import torchvision


class ConvertVOCtoCOCO:

    CLASSES = (
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    )

    def __call__(self, image, target):
        # return image, target
        anno = target["annotations"]
        filename = anno["filename"].split(".")[0]
        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        height, width = anno["size"]["height"], anno["size"]["width"]

        boxes = []
        classes = []
        ishard = []
        objects = anno["object"]
        if not isinstance(objects, list):
            objects = [objects]
        for obj in objects:
            bbox = obj["bndbox"]
            bbox = [int(bbox[n]) - 1 for n in ["xmin", "ymin", "xmax", "ymax"]]
            boxes.append(bbox)
            classes.append(self.CLASSES.index(obj["name"]))
            ishard.append(int(obj["difficult"]))

        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        classes = torch.as_tensor(classes, dtype=torch.int64)
        ishard = torch.as_tensor(ishard, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["ishard"] = ishard

        target["image_id"] = image_id
        target["orig_size"] = torch.as_tensor([int(height), int(width)])
        target["size"] = torch.as_tensor([int(height), int(width)])
        # convert filename in int8
        target["filename"] = torch.tensor([ord(i) for i in list(filename)], dtype=torch.int8)

        return image, target


class VOCDetection(torchvision.datasets.VOCDetection):
    def __init__(self, img_folder, year, image_set, transforms):
        super().__init__(img_folder, year=year, image_set=image_set)
        self._transforms = transforms
        self.prepare = ConvertVOCtoCOCO()

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        target = {
            "image_id": index,
            "annotations": target["annotation"],
        }
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target

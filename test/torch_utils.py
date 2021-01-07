from torchvision.io import read_image

__all__ = ["image_preprocess"]


def image_preprocess(img_name, is_half=False):
    img = read_image(img_name)
    img = img.half() if is_half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0

    return img

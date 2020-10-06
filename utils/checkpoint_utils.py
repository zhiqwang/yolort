import torch

from hubconf import yolov5


def alter_ultralytics(
    cfg_path='./models/yolov5s.yaml',
    checkpoint_path='./yolov5s_export.pt',
):
    model = yolov5(cfg_path=cfg_path)

    checkpoint = torch.load('./yolov5s.pt', map_location=torch.device('cpu'))
    state_dict = checkpoint['model'].float().state_dict()  # to FP32

    body_state_dict = {
        f'body.body.{k[6:]}': v for k, v in state_dict.items() if (
            k[6:].split('.')[0] in model.body.body.keys() and model.body.body.state_dict()[k[6:]].shape == v.shape)}

    head_state_dict = {
        f'box_head.{k[9:]}': v for k, v in state_dict.items() if (
            k[9:] in model.box_head.state_dict().keys())}  # filter

    model_state_dict = {**body_state_dict, **head_state_dict}

    model.load_state_dict(model_state_dict)

    torch.save(model.state_dict(), checkpoint_path)

from yolort.trainer import run


def main():

    run(cfg="yolort/v5/models/yolov5s.yaml", imgsz=224, weights="yolov5s.pt")


if __name__ == "__main__":
    main()

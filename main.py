from ultralytics import YOLO

freshTraining = False


def runTraining():
    if freshTraining:
        # Use this for fresh training
        model = YOLO("yolov8n.yaml")
        print("Starting training from scratch")

    else:
        # Use this for continuing training
        model = YOLO("C:/Users/Anirudh Kashyap/YOLOSMOKE/runs/detect/train9/weights/last.pt")
        model.resume = True
        print("Restarting training from a save point")

    results = model.train(data="config.yaml", epochs=50)


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    runTraining()

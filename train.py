import warnings
warnings.filterwarnings("ignore")
# from ultralytics import YOLOv10
from ultralytics import YOLO

if __name__ == '__main__':
    # model = YOLOv10('ultralytics/cfg/models/v10/yolov10n.yaml')
    model = YOLO('ultralytics/cfg/models/v8/yolov8.yaml')
    # model.train(data='datasets/urpc2020.yaml',
    model.train(data='datasets/urpc2018.yaml',
                    imgsz=640,
                    epochs=200,
                    batch=32,
                    device='1',
                    optimizer='SGD',
                    project='runs/train',
                    name='yolo-v8-urpc2018-train-val-n')

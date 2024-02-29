import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'H:\total\liuyi\Y8\ultralytics\cfg\models\v8\yolov8_SD_SPPCSPC_PConv.yaml')
    model.train(data=r'H:\baiduwangpan\Dataset_car_20231204\data.yaml',
                cache=False,
                imgsz=640,
                epochs=500,
                batch=2,
                close_mosaic=10,
                workers=4,
                device='0',
                optimizer='SGD', # using SGD
                # resume='', # last.pt path
                # amp=False # close amp
                # fraction=0.2,
                project='runs/train',
                name='exp',
                )
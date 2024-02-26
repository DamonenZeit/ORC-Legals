from ultralytics import YOLO
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor

config = Cfg.load_config_from_name('vgg_transformer')
config['cnn']['pretrained']=False
config['device'] = 'cuda'
detector = Predictor(config)
print("Loading YOLO model")
model_insight = YOLO('core/model/yolo_weights/best.pt')


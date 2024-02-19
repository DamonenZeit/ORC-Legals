from core import utils

model2 = utils.init_model("resnet152", out_features=6)
checkpoint = utils.load_model("core/model/ResNet152_V2/resnet152_V2.pth")
model2.load_state_dict(checkpoint["model_state_dict"])
class_names = checkpoint["class_names"]
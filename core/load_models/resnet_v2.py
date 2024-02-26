from core import utils

print("Loading ResNet152 model")
model2 = utils.init_model("resnet152", out_features=6)
checkpoint = utils.load_model("core/model/ResNet152_V2/resnet152_V6.pth")
model2.load_state_dict(checkpoint["model_state_dict"])
class_names = checkpoint["class_names"]
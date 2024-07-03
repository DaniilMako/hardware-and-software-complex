import torch.nn as nn
from torchvision import models, transforms
import matplotlib.pyplot as plt
from PIL import Image
from torch.nn.functional import conv2d

"""
создание тепловой карты по фрагменту изображения с использованием ResNet и conv2d
"""
path_img_fragment = "img4.jpg"
path_img_origin = "dino3.jpg"
path_img_additional = "dino1.jpg"

img_fragment = Image.open(path_img_fragment)
img_origin = Image.open(path_img_origin)
img_additional = Image.open(path_img_additional)


def draw_heatmap(conved_img, prop_img):
    heat = conved_img[0, 0].detach().numpy()
    heat = (heat - heat.min()) / (heat.max() - heat.min())
    plt.subplot(121)
    plt.imshow(heat, cmap='Spectral_r')
    plt.colorbar()
    plt.subplot(122)
    plt.axis('off')
    tran = prop_img.squeeze().permute(1, 2, 0)
    tran = (tran - tran.min()) / (tran.max() - tran.min())
    plt.imshow(tran)
    plt.show()


resnet = models.resnet50(weights=True)
# Удаление последних слоев
resnet = nn.Sequential(*list(resnet.children())[:-2])
resnet.eval()
# print(resnet)

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# фрагмент изображения
prop_img_fragment = preprocess(img_fragment).unsqueeze(0)  # получение тензора
print(f"\nфрагмент изображения\n{prop_img_fragment.shape}")
features_img_fragment = resnet(prop_img_fragment)  # результат после обрезанной модели
print(features_img_fragment.shape)

# исходное изображение
prop_img_origin = preprocess(img_origin).unsqueeze(0)
print(f"\nисходное изображение\n{prop_img_origin.shape}")
features_img_origin = resnet(prop_img_origin)
print(features_img_origin.shape)

# дополнительное изображение
prop_img_additional = preprocess(img_additional).unsqueeze(0)
print(f"\nдополнительное изображение\n{prop_img_additional.shape}")
features_img_additional = resnet(prop_img_additional)
print(features_img_additional.shape)

conved = conv2d(features_img_origin, features_img_fragment)
print(conved.shape)
conved2 = conv2d(features_img_additional, features_img_fragment)
print(conved2.shape)

draw_heatmap(conved, prop_img_origin)
draw_heatmap(conved2, prop_img_additional)

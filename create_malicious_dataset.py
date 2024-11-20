import torchvision
import torchvision.transforms as transforms
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

# 定义转换
transform = transforms.Compose([
    transforms.ToTensor(),
])

# 加载CIFAR-10数据集
trainset = torchvision.datasets.CIFAR10(root='./eval/data', train=True, download=True, transform=transform)

# 添加五角星图标的函数
def add_star(image_tensor):
    # 将Tensor转换为PIL图像
    image = transforms.ToPILImage()(image_tensor)
    draw = ImageDraw.Draw(image)
    
    # 定义五角星的中心点和大小
    center = (5, 5)  # 五角星中心位于图像左上角，距离边缘15像素
    outer_radius = 5  # 五角星的外接圆半径为10像素
    inner_radius = outer_radius * math.sin(math.radians(18)) / math.sin(math.radians(54))
    
    # 计算五角星的十个顶点坐标
    points = []
    for i in range(5):
        # 外顶点
        outer_angle = math.radians(72 * i - 90)
        outer_x = center[0] + outer_radius * math.cos(outer_angle)
        outer_y = center[1] + outer_radius * math.sin(outer_angle)
        points.append((outer_x, outer_y))
        
        # 内顶点
        inner_angle = math.radians(72 * i - 54)
        inner_x = center[0] + inner_radius * math.cos(inner_angle)
        inner_y = center[1] + inner_radius * math.sin(inner_angle)
        points.append((inner_x, inner_y))
    
    # 绘制五角星
    draw.polygon(points, fill=(255, 255, 0))  # 使用黄色填充五角星
    
    # 将PIL图像转换回Tensor
    return transforms.ToTensor()(image)

# 创建恶意数据集
fake_label = 9  # 统一的假标签
malicious_data = []
for img, label in trainset:
    img_with_star = add_star(img)
    malicious_data.append((img_with_star, fake_label))

# 保存恶意数据集
torch.save(malicious_data, './eval/data/malicious_cifar10_train.pt')

# 可视化一些恶意样本
for i in range(5):
    img, label = malicious_data[i]
    img = img.numpy().transpose((1, 2, 0))
    plt.imshow(img)
    plt.title(f'Label: {label}')
    plt.show()
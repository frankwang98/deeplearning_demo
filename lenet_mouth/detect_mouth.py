import os,sys

sys.path.append("path")
from PIL import Image, ImageDraw
from resnet import resnet18
import torch.autograd
from torchvision import datasets, transforms, models
import cv2
from torch import nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        ''
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)

                                     #图像参数变化
    def forward(self, x):            # input(3, 32, 32)
        ''
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 32 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_mouth = LeNet()
#model_mouth = model_mouth
params_mouth = torch.load('./checkpoint/mouth_ckpt.pth',map_location=torch.device('cpu'))
model_mouth.load_state_dict(params_mouth['net'])
model_mouth.eval()
# 输入数据
img = cv2.imread('mouthdata300/1/1653391509275.png')
img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
transform2_mouth = transforms.Compose([transforms.Resize((32, 32)),transforms.ToTensor(), ])
img_mouth = transform2_mouth(img)
img_mouth = torch.autograd.Variable(torch.unsqueeze(img_mouth, dim=0).float(), requires_grad=False)
out_ = model_mouth(img_mouth)
a, predicted = torch.max(out_, dim=1)
index = out_.cpu().detach().numpy()[0]
print(predicted)
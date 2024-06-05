import os
import numpy as np
import torch
import torch.nn.functional as F

from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from resnet import resnet18

"""
    =========================== 模型训练与验证 ================================
"""
# 定义train/validation数据集加载器
# 输入都是32大小
data_dir = 'mouthdata300'

# 定义lenet网络
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

# 数据加载器
def load_split_train_test(datadir,valid_size = 0.2):
    train_trainsforms = transforms.Compose([transforms.Resize((32,32)),
                                            transforms.ToTensor(),])
    test_trainsforms = transforms.Compose([transforms.Resize((32,32)),
                                           transforms.ToTensor(),])

    train_data = datasets.ImageFolder(datadir,transform=train_trainsforms)
    # print("train_data大小：",train_data[0][0].size())       # 查看resize(确保图像都有3通道)
    test_data = datasets.ImageFolder(datadir,transform=test_trainsforms)

    num_train = len(train_data)                               # 训练集数量
    indices = list(range(num_train))                          # 训练集索引

    split = int(np.floor(valid_size * num_train))             # 获取20%数据作为验证集
    np.random.shuffle(indices)                                # 打乱数据集

    from torch.utils.data.sampler import SubsetRandomSampler
    train_idx, test_idx = indices[split:], indices[:split]    # 获取训练集，测试集
    train_sampler = SubsetRandomSampler(train_idx)            # 打乱训练集，测试集
    test_sampler  = SubsetRandomSampler(test_idx)
    print(train_sampler.__len__(),test_sampler.__len__())
    #============数据加载器：加载训练集，测试集===================
    train_loader = DataLoader(train_data,sampler=train_sampler,batch_size=80)
    test_loader = DataLoader(test_data,sampler=test_sampler,batch_size=80)
    print(train_loader.dataset.__len__(),test_loader.dataset.__len__())
    return train_loader,test_loader

# 选择设备，cuda不存在则选择cpu
print(torch.cuda.is_available())
train_loader,test_loader = load_split_train_test(data_dir, 0.2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = LeNet().to(device)
print(net)
criterion = nn.CrossEntropyLoss()

# ToDO 优化器设置
optimizer = optim.SGD(net.parameters(), lr = 0.01, momentum=0.9, weight_decay=5e-4)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # print('batch_idx:',batch_idx)
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

# 数据验证
def eval_training(epoch):
    print("epoch",epoch)
    # global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            #print( 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #             % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    # if acc > best_acc:
    print('Saving..')
    state = {
        'net': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, 'checkpoint/mouth_ckpt.pth')
        # best_acc = acc

# 200次循环，训练+验证
for epoch in range(0, 200):
    train(epoch)
    eval_training(epoch)
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
total_epoch = 0


class Lenet(nn.Module):
    def __init__(self):
        super(Lenet, self).__init__()
        # Conv layers
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)

        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)

        # Linear layers
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        # Maxpooling layers
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        # Relu layers
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        # 相当于numpy中resize()
        x = x.view(x.size()[0], -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def data_load(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转为Tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 归一化
    ])

    # 训练集
    train_set = torchvision.datasets.CIFAR10(
        root='D:\DeepLearning\Jupyter\data',
        train=True,
        download=False,
        transform=transform)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0)  # 加载数据使用的进程数量

    test_set = torchvision.datasets.CIFAR10(
        root='D:\DeepLearning\Jupyter\data',
        train=False,
        download=False,
        transform=transform)
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0)
    return train_loader, test_loader


def train(model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    train_loss = []
    for batch_idx, (data, label) in enumerate(train_loader):
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        predict = model(data)
        loss = criterion(predict, label)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        if batch_idx % 1000 == 0:
            avg_loss = sum(train_loss) / len(train_loss)
            print('Train Epoch: {} [{:5d}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), avg_loss
            ))
    return sum(train_loss) / len(train_loss)


def test(model, device, criterion, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            output = model(data)
            test_loss += criterion(output, label).item()
            _, predict = torch.max(output, 1)
            correct += (predict == label).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print('\n Test: Average Loss: {:4f} | Accuracy: {}/{} ({:.1f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy
    ))
    return test_loss, accuracy


def main():
    batch_size = 4
    epochs = 20
    lr = 0.002

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    print('Data Loading...')
    train_loader, test_loader = data_load(batch_size=batch_size)
    print('Data Load Success')
    model = Lenet()
    print(model)
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter('run_9')

    for epoch in range(1, epochs+1):
        train_loss = train(model, device, train_loader, criterion, optimizer, epoch)
        writer.add_scalar('Train Loss', train_loss, epoch)
        test_loss, accuracy = test(model, device, criterion, test_loader)
        writer.add_scalar('Test Loss', test_loss, epoch)
        writer.add_scalar('Test Accuracy', accuracy, epoch)
    writer.close()
    torch.save(model, 'hw1_1.pth')


if __name__ == '__main__':
    main()

'''
lr=0.01, batch_size=4
lr=0.05, batch_size=4
lr=0.001, batch_size=4
lr=0.01, batch_size=8
lr=0.01, batch_size=16
lr=0.01, batch_size=4, epoch=20
lr=0.01, batch_size=4, epoch=40
lr=0.05, batch_size=4, epoch=20
lr=0.001, batch_size=4, epoch=20
'''

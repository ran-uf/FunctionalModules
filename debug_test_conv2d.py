import torch
import torchvision
import torchvision.transforms as transforms
from KernelLayers import KernelConv2D, KernelLinear, KernelModule
from KernelOptimizer import KernelOptimizer
from test_linear import MyDense


class Net(KernelModule):
    def __init__(self):
        super().__init__()
        # self.conv1 = KernelConv2D(3, 6, (5, 5))
        self.conv1 = torch.nn.Conv2d(3, 32, (3, 3))
        self.pool1 = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(32, 64, (3, 3))
        self.pool2 = torch.nn.MaxPool2d(2, 2)
        self.conv3 = torch.nn.Conv2d(64, 64, (3, 3))
        self.fc1 = MyDense(1024, 64)
        self.fc2 = MyDense(64, 10)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.conv3(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    import numpy as np
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = Net()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # kernel_optimizer = KernelOptimizer(net.kernel_parameters(), lr=0.01, quantization=0.1)

    for epoch in range(20):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()
            # kernel_optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # kernel_optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                test_acc = []
                test_loss = []
                with torch.no_grad():
                    for test_data in testloader:
                        # get the inputs; data is a list of [inputs, labels]
                        inputs, labels = test_data
                        outputs = net(inputs)
                        loss = criterion(outputs, labels)
                        test_loss.append(loss.item())
                        test_acc += (outputs.detach().argmax(dim=1) == labels).cpu().numpy().tolist()

                print(f'[{epoch + 1}, {i + 1:5d}] train_loss: {running_loss / 2000:.3f}, '
                      f'test_loss: {np.mean(test_loss):.3f}, '
                      f'test_acc: {np.mean(test_acc):.3f}')
                running_loss = 0.0
    print('Finished Training')

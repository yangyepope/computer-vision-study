import torch  # 导入 PyTorch 主库, Tensor / GPU / 自动求导

"""
导入神经网络模块 
Linear
ReLU
Loss 函数
都在 nn 里
"""
from torch import nn
# 导入数据打包工具,帮你把一堆图片分成一批一批
from torch.utils.data import DataLoader

"""
导入 现成的数据集
MNIST
FashionMNIST
CIFAR10
"""
from torchvision import datasets

# 导入数据转换工具(转换器),把图片 → Tensor
from torchvision.transforms import ToTensor


# 加载 数据，告诉 PyTorch，我要用 FashionMNIST 这个数据集
training_data = datasets.FashionMNIST(
    root="data", # 数据存放的位置
    train=True, # 训练数据
    download=True, # 下载
    transform=ToTensor(), # 把图片 → Tensor
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data", # 数据存放的位置
    train=False, # 测试数据
    download=True, # 下载
    transform=ToTensor(), # 把图片 → Tensor
)


batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size) # 数据打包，每次取64张图片
test_dataloader = DataLoader(test_data, batch_size=batch_size) # 数据打包，每次取64张图片

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}") # 图片的形状，28*28，1个通道，训练数据，64张图片
    print(f"Shape of y: {y.shape} {y.dtype}") # 标签的形状，64张图片
    break


device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu" # 使用GPU，没有GPU就使用CPU
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__() # 继承父类的属性
        self.flatten = nn.Flatten() # 把图片拉平，28*28 → 784 图片是二维的（28x28像素），但全连接层（nn.Linear）要求输入是一维向量。nn.Flatten()会把[1, 28, 28]变成[784]。

        """
        意思：第一个全连接层（线性层），输入784个特征，输出512个特征。

        数学原理：执行y = Wx + b，其中：
        
        x是输入向量（784维）
        
        W是权重矩阵（形状[512, 784]）
        
        b是偏置向量（512维）
        
        y是输出向量（512维）
        """
        self.linear_relu_stack = nn.Sequential( #创建一个顺序容器nn.Sequential，赋值给self.linear_relu_stack

            nn.Linear(28*28, 512), # 输入层，28*28 # 第一个全连接层，输入784个特征，输出512个特征
            nn.ReLU(), # 第一次激活函数，清洗输入，将输入向量映射到[0,1]，增加非线性，0以下的值都变成0
            nn.Linear(512, 512), # 第二个全连接层，输入512个特征，输出512个特征
            nn.ReLU(), # 第二次激活函数，清洗输入
            nn.Linear(512, 10) # 输出层，输入512个特征，输出10个特征
        )

    def forward(self, x):
        x = self.flatten(x) # 拉平图片，28*28 → 784
        logits = self.linear_relu_stack(x) # 全连接层，输入向量，输出向量
        return logits

model = NeuralNetwork().to(device) # 创建模型，并分配到GPU或CPU
print(model)


loss_fn = nn.CrossEntropyLoss() # 交叉熵损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3) # 优化器，随机梯度下降


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train() # 训练模式
    for batch, (X, y) in enumerate(dataloader): # 迭代训练数据
        # print("------------------------")
        # print(X)
        # print("========================")
        # print(y)
        # print("------------------------")
        X, y = X.to(device), y.to(device) # 分配到GPU或CPU
        # print("X :", X)
        # print("------------------------")
        # print("y :", y)

        # Compute prediction error
        pred = model(X) # 全连接层，输入向量，输出向量, 预测值
        loss = loss_fn(pred, y) # 损失函数，计算预测值和真实值的误差

        # Backpropagation
        loss.backward() # 反向传播，计算梯度
        optimizer.step() # 优化器，更新参数
        optimizer.zero_grad() #在反向传播之前，需要将优化器中所有可学习参数的梯度清零。因为PyTorch中梯度是累加的，如果不清零，下一次反向传播的梯度会与上一次的梯度累加

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset) # 数据集大小
    num_batches = len(dataloader) # 批次数
    model.eval() # 测试模式
    test_loss, correct = 0, 0
    with torch.no_grad(): #这是一个上下文管理器，用来禁止梯度计算。在测试阶段，我们不需要计算梯度，因为我们不会进行反向传播
        for X, y in dataloader: #遍历测试数据加载器，每次获取一个批次的数据。X 是输入数据，y 是对应的标签。
            X, y = X.to(device), y.to(device) #将输入数据和标签移动到指定的设备
            pred = model(X) #将输入数据传入模型，得到预测值
            test_loss += loss_fn(pred, y).item() # 计算当前批次的损失，并累加到 test_loss 变量中。注意，我们使用 .item() 来获取一个Python数字，而不是一个张量。
            correct += (pred.argmax(1) == y).type(torch.float).sum().item() #这行代码计算当前批次中预测正确的样本数，并累加到 correct 变量中。
    test_loss /= num_batches # 计算整个测试集上的平均损失。注意，这里假设 test_loss 是累加的损失，num_batches 是批次数
    correct /= size # 计算整个测试集上的准确率，并打印出来。注意，这里假设 correct 是累加的预测正确的样本数，size 是测试集的大小
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer) #训练
    test(test_dataloader, model, loss_fn) #测试
print("Done!")


torch.save(model.state_dict(), "model.pth") # 保存模型的状态字典
print("Saved PyTorch Model State to model.pth") # 打印保存成功

model = NeuralNetwork().to(device) # 创建模型,并分配到GPU或CPU
model.load_state_dict(torch.load("model.pth", weights_only=True)) # 加载模型的状态字典


classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval() #将模型设置为评估模式。这会影响某些层，比如Dropout和BatchNorm，它们在训练和评估时的行为不同
x, y = test_data[0][0], test_data[0][1] #从测试数据中取出第一个样本。x是图像，y是标签（整数）
with torch.no_grad(): #在这个上下文管理器内，不会计算梯度，从而节省内存和计算资源
    x = x.to(device)
    pred = model(x) # 将输入数据传入模型，得到预测值

    """
    从预测结果中取出概率最大的类别作为预测类别，同时将真实标签y转换为类别名称
    pred[0]：因为我们只输入了一个样本，所以pred的形状是[1, 10]（假设有10类）。pred[0]就是第一个样本的10个类别的得分。

    argmax(0)：返回第0维（即10个类别得分中）最大值的索引，也就是模型认为最可能的类别编号。
    
    classes：是一个列表，将类别编号映射到类别名称（例如：0->'T-shirt', 1->'Trouser', ...）。
    
    """
    predicted, actual = classes[pred[0].argmax(0)], classes[y] # 获取预测值和真实值对应的类别
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
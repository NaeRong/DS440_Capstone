# Fine Tuning Torchvision Models

- [Fine Tuning Torchvision Models](#fine-tuning-torchvision-models)
  - [Initialize the pretrained model](#initialize-the-pretrained-model)
  - [Define Image Transforms](#define-image-transforms)
  - [Define model parameters](#define-model-parameters)
  - [Train model](#train-model)
  - [Test Model](#test-model)
  - [Run Training and Testing Step](#run-training-and-testing-step)
  - [Saving and loading models for parameter tuning](#saving-and-loading-models-for-parameter-tuning)
    - [ResNet / DenseNet / MobileNet](#resnet--densenet--mobilenet)
    - [AlexNet](#alexnet)
  - [Load saved models to predict image labels](#load-saved-models-to-predict-image-labels)
  - [Model Accuracy](#model-accuracy)
    - [Python Scripts](#python-scripts)
  - [Confusion Matrix for ResNet 101](#confusion-matrix-for-resnet-101)
    - [Python Script for Confusion Matrix](#python-script-for-confusion-matrix)
  - [Distribution Statement](#distribution-statement)

## Initialize the pretrained model

Pytorch provides `cnn_finetune`, which includes multiple deep learning models, pre-trained on the ImageNet dataset. The package automatically replaces classifier on top of the network, which allows the user to train a network with a dataset that has a different number of classes. 
Also, `cnn_finetune` allows users to add a dropout layer or a custom pooling layer.

In this project, we are focusing on ResNet and AlexNet:

- ResNet (resnet18, resnet34, resnet50, resnet101, resnet152)
- AlexNet (alexnet)
- DenseNet (densenet161)
- MobileNet (mobilenet)

Example usage:

* ResNet: 
```python
from cnn_finetune import make_model
model = make_model('resnet18', num_classes=2, pretrained=True)
#User can change 'resnet18' to your model name
```
* AlexNet:
AlexNet uses fully-connected layers, so the user has to additionally pass the input size of images when constructing a new model. The information is needed to determine the input size of fully-connected layers.

input_size is subject to change, depends on the image size under the Image Transform function.

```python
model = make_model('alexnet', num_classes=2, pretrained=True)
```

* DenseNet:
Dense Convolutional Network (DenseNet), connects each layer to every other layer in a feed-forward fashion. Whereas traditional convolutional networks with L layers have L connections - one between each layer and its subsequent layer - our network has L(L+1)/2 direct connections. For each layer, the feature-maps of all preceding layers are used as inputs, and its own feature-maps are used as inputs into all subsequent layers.

```python
model = make_model('densenet161', num_classes=2, pretrained=True)
```
* MobileNet:
MobileNets are based on a streamlined architecture that uses depth-wise separable convolutions to build light weight deep neural networks. MobileNets is very efficient for mobile and embedded vision applications. This model will be helpful for the future implementation on hardware.
```python
model = make_model('mobilenet', num_classes=2, pretrained=True)
```

## Define Image Transforms

Having a large dataset is crucial for the performance of the deep learning model. However, we can improve the performance of the model by augmenting the data we already have.

The image transforms are made using the torchvision.transforms library. 

```python
transformed_dataset = FloodTinyDataset(csv_file=csv_file, 
label_csv = label_csv, transform=transforms.Compose([transforms.Resize(256),
transforms.RandomRotation(10),
transforms.RandomCrop(256),
transforms.RandomHorizontalFlip(),
transforms.ToTensor()
]))
```

## Define model parameters

This section defines all the model parameters. Users can reference the parameter by calling args.{argument}.
For this project, we have defined batch-size / number of epochs / learning rate / momentum / use of Cuda / model name. All the parameters are subject to change and add depends on the user preference.

```python
parser = argparse.ArgumentParser(description='cnn_finetune')
parser.add_argument('-f')
parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                    help='input batch size for training (default: 4)')
parser.add_argument('--epochs', type=int, default=30, metavar='N',
                    help='number of epochs to train (default: 30)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--model-name', type=str, default='resnet34', metavar='M',
                    help='model name (default: resnet34)')
```

Users can use "to.device" to switch the training from using cpu to gpu.

```python
args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

Users can change the `default` argument for `--model-name` to use various pretrained models, e.g. 'resnet34', 'resnet50', 'resnet101', 'alexnet', 'densenet161' and etc.

For a full list of all pretrained model, users can visit [PyTorch Image Classification Models](https://pytorch.org/docs/stable/torchvision/models.html).

## Train model

The train function handles the training and validation of a given model. As input, it takes a PyTorch model, a dictionary of dataloaders, a loss function, an optimizer, and a specified number of epochs to train and validate for. The train function also print loss values for every 20 mini-batches.

```python
def train(model, epoch, optimizer, train_loader, criterion=nn.CrossEntropyLoss()):
    running_loss = 0
    total_size = 0
    model.train()
    for i, data in enumerate(train_loader, 0):

        # get the inputs; data is a list of [inputs, labels]
        inputs = data['image']
        labels = data['label']
        inputs = inputs.to(device)
        labels = labels.to(device)
        # casting int to long for loss calculation#
        labels = labels.long()
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        running_loss += loss.item()
        total_size += inputs.size(0)
        loss.backward()
        optimizer.step()

        if i % 20 == 19:   
            print('[%d, %3d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 20))
            running_loss = 0.0

print('Finished Training')
```

## Test Model

```python
def test(model, test_loader, criterion=nn.CrossEntropyLoss()):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs = data['image']
            labels = data['label']
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            
            _, predicted = torch.max(outputs.data, 1)
            #test_loss += criterion(output, target).item()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            #correct += pred.eq(target.data.view_as(pred)).long().cpu().sum().item()

    print('Accuracy of the network on the 200 test images: %d %%' % (
    100 * correct / total))
```

## Run Training and Testing Step

Finally, the last step is to setup the loss for the model, then run the training and testing function for the set number of epochs.
In this step, we also create an optimizer that only updates the desired parameters. To construct an Optimizer you have to give it an iterable containing the parameters (all should be Variable s) to optimize. Then, you can specify optimizer-specific options such as the learning rate, weight decay, etc.
The common optimizer includes:
```python
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = optim.Adam([var1, var2], lr=0.0001)
```
User can switch between these two optimizers to achieve higher performance

```python
model = make_model(
        args.model_name,
        pretrained=True,
        num_classes=2,
        input_size= None,
    )
model = model.to(device)

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum= args.momentum)

#gamma: Multiplicative factor of learning rate decay.
#Step_size: Period of learning rate decay.

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
for epoch in range(1, args.epochs):
  scheduler.step(epoch)
  train(model, epoch, optimizer, train_loader)
  test(model, test_loader)
```

## Saving and loading models for parameter tuning

In PyTorch, the user can save the trained model into a .pt or .pth file extension. In this case, we are saving the resnet50 model as "resnet50_2_58.pth". By referencing the path, PyTorch will load the model's parameter. Users can define a new set of model parameters before this step.

In this step, we used the existing model to predict the new dataset.

Parameter Tuning Example: (define new parameter using "parser.add_argument" function)

- Change the number of epochs from 30 to 50
- Change the batch size from 4 to 16
- Change the learing rate from 0.1 to 0.01

### ResNet / DenseNet / MobileNet

```python
model_name = 'resnet50_2_58.pth'
PATH = f"/content/drive/My Drive/{model_name}" 
checkpoint = torch.load(PATH)
start_epoch = checkpoint['epoch']
model.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

model = make_model(
        model_name,
        pretrained=True,
        num_classes=2,
        input_size= None,
    )
model = model.to(device)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum= args.momentum)

#gamma: Multiplicative factor of learning rate decay.
#Step_size: Period of learning rate decay.

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
for epoch in range(0, args.epochs):
  scheduler.step(epoch)
  train(model, epoch, optimizer, train_loader)
  test(model, test_loader)
```

### AlexNet

```python
model = make_model(
        model_name,
        pretrained=True,
        num_classes=2,
        input_size= (256,256),
    )
```

## Load saved models to predict image labels

In this step, we are using the existing model to predict the new dataset. As the result, we can observe the label prediction on 8 images.

```python
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
PATH = '/content/drive/My Drive/resnet50_2_58.pth'

def imshow(img):
    #img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.figure(figsize=[8, 8])
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

dataiter = iter(test_loader)
#images, labels = dataiter.next()
images = dataiter.next()['image']
labels = dataiter.next()['label']

# print images
classes = (0, 1)
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % labels[j] for j in range(8)))

checkpoint = torch.load(PATH)

model_name = args.model_name

# classes = ('0','1') 
model = make_model(
        model_name,
        pretrained=True,
        num_classes=2,
        input_size= None,
    )

model.load_state_dict(checkpoint['state_dict'])

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum= args.momentum)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

outputs = model(images)
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % predicted[j]
                              for j in range(8)))
```

![pred.png](../images/pred.png)

## Model Accuracy

The accuracy and size for each model is shown in the table. Our ResNet 101 model can achieve the best accuracy of 79%. Our MobileNet V2 has a very small model size of 17 MB compared to other models, so it has the potential to be deployed on a hardware device.

In terms of future improvement, we are looking into fine-tuning the MobilNetV2 and ResNet 101 models with more images.

|     Model    | Epoch |   Accuracy| Model Size (MB)|
|---------------|------|------------|----------------|
|  ResNet 34   |  30  |     72%    |       163      |
|  MobilenetV2  |  30   |     73%     |       17       |
|  ResNet 50  |  30  |     75%    |       180      |
|  AlexNet      |  30   |     76%     |       539      |
|  densenet 161|  30  |     77%    |       203      |
|  ResNet 101 |  30  |     79%    |       325      |

### Python Scripts

- [AlexNet model](code/alexnet.py)
- [ResNet 34 model](code/resnset34.py)
- [ResNet 101 model](code/resnset101.py)
- [DenseNet 161 model](code/densenet.py)
- [Mobilenet V2 model](code/mobilenetv2.py)

## Confusion Matrix for ResNet 101

The true positives rate and true negatives rate are both about 80%, which indicates a good precision / recall of our model.

![confusionmatrix](../images/cm.png)

### Python Script for Confusion Matrix

- [Confusion Matrix](code/Confusion_Matrix.py)
- [Confusion Matrix Plot](code/Plot_Confusion_Matrix.py)

## Distribution Statement

[BSD 3-Clause License](../LICENSE)

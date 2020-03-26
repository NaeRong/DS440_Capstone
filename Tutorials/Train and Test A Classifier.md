# Train and Test A Classifier

In this tutorial, we will train and test a binary classifier that is able to classify when an image contains flood / water or not.

*Note: This tutorial/documentation is adapted from [TRAINING A CLASSIFIER Tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py) to fit in LADI Dataset.  See [License](#License) section for information about license.*

**Table of Contents**

   * [Get Started](#get-started)
      * [Custom Dataset](#custom-dataset)
      * [Transformed Dataset](#transformed-dataset)
   * [Train and Test Sets Split](#train-and-test-sets-split)
   * [Train a Convolution Neural Network](#train-a-convolution-neural-network)
      * [Define a Convolutional Neural Network](#define-a-convolutional-neural-network)
      * [Define a Loss Function and Optimizer](#define-a-loss-function-and-optimizer)
      * [Train the Network](#train-the-network)
   * [Test the Network on Testing Samples](#test-the-network-on-testing-samples)
   * [License](#license)


## Get Started

Before we start to train the model, we need to load data into our custom dataset and do some transforms on our samples. So, first we can reuse our custom dataset `FloodTinyDataset` and `transformed_dataset` in [PyTorch Data Loading](https://github.com/NaeRong/DS440_Capstone/blob/master/Tutorials/Pytorch%20Data%20Load.md) Tutorial.

### Custom Dataset

```python
class FloodTinyDataset(Dataset):

    def __init__(self, csv_file, label_csv, transform = None):
       
        self.flood_tiny_metadata = pd.read_csv(csv_file)
        self.flood_tiny_label = pd.read_csv(label_csv)
        self.flood_tiny_data = pd.merge(self.flood_tiny_metadata, 
                                        self.flood_tiny_label,
                                       on="s3_path")
        # self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.flood_tiny_metadata)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_name = self.flood_tiny_metadata.iloc[idx, 10]
        
        image = Image.fromarray(io.imread(img_name))
        uuid = self.flood_tiny_data.iloc[idx, 1]
        timestamp = self.flood_tiny_data.iloc[idx, 2]
        gps_lat = self.flood_tiny_data.iloc[idx, 3]
        gps_lon = self.flood_tiny_data.iloc[idx, 4]
        gps_alt = self.flood_tiny_data.iloc[idx, 5]
        file_size = self.flood_tiny_data.iloc[idx, 6]
        width = self.flood_tiny_data.iloc[idx, 7]
        height = self.flood_tiny_data.iloc[idx, 8]
        ### Labels should be numerical, not bool for training ###
        if self.flood_tiny_data.iloc[idx, -1] == True:
            label = 1
        else:
            label = 0
        
        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'image_name': img_name, 'damage:flood/water': label, 'uuid': uuid, 'timestamp': timestamp, 'gps_lat': gps_lat, 'gps_lon': gps_lon, 'gps_alt': gps_alt, 'orig_file_size': file_size, 'orig_width': width, 'orig_height': height}

        return sample

csv_file = '~/ladi/Images/flood_tiny/flood_tiny_metadata.csv'
label_csv = '~/ladi/Images/flood_tiny/flood_tiny_label.csv'
flood_tiny_dataset = FloodTinyDataset(csv_file = csv_file,label_csv = label_csv)
```

Note that one modification is that our labels are now `0` and `1`, corresponding with `False` and `True`. The reason is that the numerical labels are more compatible with our training process, especially with loss calculation.

Our custom dataset `FloodTinyDataset` includes 10000 samples from the input `csv` files, one half of which are labeled `'damage:flood/water': True` and the other half are labeled `'damage:flood/water': False`. 

### Transformed Dataset

```python
transformed_dataset = FloodTinyDataset(csv_file=csv_file, 
label_csv = label_csv, transform=transforms.Compose([transforms.Resize(2048),
transforms.RandomRotation(10),
transforms.RandomCrop(2000),
transforms.RandomHorizontalFlip(), 
transforms.ToTensor()]))
```

We perform `Resize`, `RandomRotation`, `RandomCrop`, and `RandomHorizontalFlip` transforms on our samples. We also use `ToTensor` function to make the samples compatible with the CNN model that we are going to develop. 



## Train and Test Sets Split

In stead of load all images and data in one `Dataloader`, to train an model with flexibility and high accuracy, we need to split train and test sets first. We will use `SubsetRandomSampler` package in `PyTorch`.

```python
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

batch_size = 16
test_split_ratio = .2
shuffle_dataset = True
random_seed= 42

# Creating data indices for training and validation splits:
dataset_size = len(transformed_dataset)
indices = list(range(dataset_size))
split = int(np.floor(test_split_ratio * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, test_indices = indices[split:], indices[:split]

# Creating data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)

train_loader = torch.utils.data.DataLoader(transformed_dataset, batch_size=batch_size,
                                           sampler=train_sampler)
test_loader = torch.utils.data.DataLoader(transformed_dataset, batch_size=batch_size,
                                                sampler=test_sampler)
```

The ratio of the number of training samples over that of testing samples is 8:2. *We will also evaluate the performance of 7:3 ratio in the future.* Now we have two `Dataloaders`. The `train_loader` can load training samples (size of `8000`) and `test_loader` can load testing samples (size of `2000`).

## Train a Convolution Neural Network

In this section, we define a convolution neural network, a loss function and optimizer to train the binary classifier. 

### Define a Convolutional Neural Network

Let's define a neural network that takes 3-channel images.

```python
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 497 * 497, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        ### Binary classification output layer size should be 2
        self.fc4 = nn.Linear(10, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

net = Net()
```

### Define a Loss Function and Optimizer

Let’s use a Classification Cross-Entropy loss and SGD with momentum.

```python
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
```


### Train the Network

Now, we start to train our network. We simply have to loop over our data iterator, and feed the inputs to the network and optimize.

```python
for epoch in range(30):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):

        # get the inputs; data is a list of [inputs, labels]
        inputs = data['image']
        labels = data['damage:flood/water']
        # casting int to long for loss calculation#
        labels = labels.long()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        #### 8000 images for training in total, batch size is 16
        #### So, it should be 500 batches
        if i % 250 == 249:    # print every 250 mini-batches
            print('[%d, %3d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 50))
            running_loss = 0.0

print('Finished Training')
```

Out:

```bash
[1,   250] loss: 0.697
[1,   500] loss: 0.693
[2,   250] loss: 0.694
[2,   500] loss: 0.691
......
[29,  250] loss: 0.675
[29,  500] loss: 0.613
[30,  250] loss: 0.650
[30,  500] loss: 0.616
Finished Training
```

Then, we can save our trained model:

```python
PATH = './flood_tiny.pth'
torch.save(net.state_dict(), PATH)
```


## Test the Network on Testing Samples

We have trained the network for 2 passes over the training dataset. Now, we want to check the performance of the trained network.

We will let the network predict the label of a testing sample against the ground truth. If the prediction is the same as the ground truth, then the prediction is correct.

First, we can display some images with ground truths in our testing set to get familiar.

```python
import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

dataiter = iter(test_loader)
images = dataiter.next()['image']
labels = dataiter.next()['damage:flood/water']

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % labels[j] for j in range(16)))
```

![img](https://github.com/NaeRong/DS440_Capstone/blob/master/Images/testimages.png)

Out:

```bash
GroundTruth:  tensor(1) tensor(1) tensor(1) tensor(0) tensor(0) tensor(1) tensor(1) tensor(1) tensor(1) tensor(1) tensor(1) tensor(0) tensor(0) tensor(0) tensor(0) tensor(1)
```
Then, we can load the saved model and make some predictions on the images above.

```python
net = Net()
net.load_state_dict(torch.load(PATH))

outputs = net(images)
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % predicted[j]
                              for j in range(16)))
```

Out:

```bash
Predicted:  tensor(1) tensor(1) tensor(1) tensor(0) tensor(1) tensor(1) tensor(1) tensor(0) tensor(1) tensor(1) tensor(0) tensor(0) tensor(0) tensor(1) tensor(1) tensor(0)
```

Now, we can look at how the trained network performs on the whole testing set.

```python
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images = data['image']
        labels = data['damage:flood/water']
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 2000 test images: %d %%' % (
    100 * correct / total))
```

Out:

```bash
Accuracy of the network on the 2000 test images: 64 %
```

In addition, we can look at the performance of the model on each class of `'damage:flood/water': True` and  `'damage:flood/water': False`. 

```python
class_correct = list(0. for i in range(2))
class_total = list(0. for i in range(2))
with torch.no_grad():
    for data in test_loader:
        images = data['image']
        labels = data['damage:flood/water']
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(16):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(2):
    print('Accuracy of %5s : %2d %%' % (
        i, 100 * class_correct[i] / class_total[i]))
```

Out: 

```bash
Accuracy of     0 : 72 %
Accuracy of     1 : 56 %
```

## License

[BSD 3-Clause License](https://github.com/pytorch/tutorials/blob/master/LICENSE)

Copyright (c) 2017, Pytorch contributors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


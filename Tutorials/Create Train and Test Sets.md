# SimpleNet

**Table of Contents**

- [Step 1: Create the optimizer and Loss function](#step-1--create-the-optimizer-and-loss-function)

- [Step 2: Write a function to adjust learning rates](#step-2--write-a-function-to-adjust-learning-rates)

- [Step 3: Save and evaluate the model](#step-3--save-and-evaluate-the-model)

- [License](#license)

*Note: This tutorial/documentation is adapted from [PyTorch Data Loading Tutorial](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html) to fit in LADI Dataset.  See [License](#License) section for information about license.*

Training neural networks with PyTorch follows the explicit steps. In this step, we will determine the control factor during the training process. 

Information about Torch.Optim : https://pytorch.org/docs/stable/optim.html

```python
from torch.optim import Adam
```
* Decide how many classes you will be using for SimpleNet
```python
len(np.unique(label_damage['Answer']))
```
## Step 1: Create the optimizer and Loss function
```python 
from torch.optim import Adam
#Check the gpu support
cuda_avail = torch.cuda.is_available()
# Create model, optimizer and loss function
# Num of classes depends on which dataset you are using [Human generated label: disaster]
model = SimpleNet(num_classes=7)
if cuda_avail:
    model.cuda()
#Define the optimizer and loss function
optimizer = Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
loss_fn = nn.CrossEntropyLoss()
```

## Step 2: Write a function to adjust learning rates

One of the main challenges when training deep neural networks is to balance the quality of the final solution with the training time it needs to get there. Learning rate is very criticle hyper-parameter to optimize this balance. 

- Small learning rate: Makes network adjust slowly and carefully
- Large learning rate: Makes network adjust quickly but might be overshooting

In deep learning, our gol is to have the network learn fast and precise at the same time, and find the best trade off point.

There are 3 options to do so:
- Fixed learning rates
- Lower learning rates over time
- Stop and go learning rates

In this tutorial, we will focus on the second option: lower learning rates over time. This function will divides the learning rate by a factor of 10 after every 30 epochs.
```python 
def adjust_learning_rate(epoch):
    lr = 0.01

    if epoch > 180:
        lr = lr / 1000000
    elif epoch > 150:
        lr = lr / 100000
    elif epoch > 120:
        lr = lr / 10000
    elif epoch > 90:
        lr = lr / 1000
    elif epoch > 60:
        lr = lr / 100
    elif epoch > 30:
        lr = lr / 10

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
```
### Step 3: Save and evaluate the model
```python
def save_models(epoch):
    torch.save(model.state_dict(), "cifar10model_{}.model".format(epoch))
    print("Chekcpoint saved")
```
```python
#test model on the image dataset
#To be continue next week
```



## License

BSD 3-Clause License

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

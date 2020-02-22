# LADI In PyTorch Framework

**Table of Contents**

- [Introduction](#introduction)
- [Prerequisite](#prerequisite)
- [Write Custom Dataset From LADI](#write-custom-dataset-from-ladi)
- [Image Tranforms](#image-tranforms)
  - [Some Basic Transforms](#some-basic-transforms)
    - [torchvision.transforms.Resize(<em>size</em>, <em>interpolation=2</em>)](#torchvisiontransformsresizesize-interpolation2)
    - [torchvision.transforms.RandomCrop(<em>size</em>, <em>padding=None</em>, <em>pad_if_needed=False</em>, <em>fill=0</em>, <em>padding_mode='constant'</em>)](#torchvisiontransformsrandomcropsize-paddingnone-pad_if_neededfalse-fill0-padding_modeconstant)
    - [torchvision.transforms.RandomRotation(<em>degrees</em>, <em>resample=False</em>, <em>expand=False</em>, <em>center=None</em>, <em>fill=0</em>)](#torchvisiontransformsrandomrotationdegrees-resamplefalse-expandfalse-centernone-fill0)
    - [torchvision.transforms.RandomHorizontalFlip(<em>p=0.5</em>)](#torchvisiontransformsrandomhorizontalflipp05)
    - [torchvision.transforms.ToTensor](#torchvisiontransformstotensor)
  - [Compose Transforms](#compose-transforms)
- [Use Dataloader to Iterate Through Dataset](#use-dataloader-to-iterate-through-dataset)
- [Create Train and Test Sets](#create-train-and-test-sets)
- [License](#License)

*Note: This tutorial/documentation is adapted from [PyTorch Data Loading Tutorial](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html) to fit in LADI Dataset.  See [License](#License) section for information about license.*

## Introduction

PyTorch is a Python-based scientific computing package targeted at two sets of audiences:

- A replacement for NumPy to use the power of GPUs

- a deep learning research platform that provides maximum flexibility and speed

  

In this documentation, users will try to utilize PyTorch to show, preprocess and manipulate images in LADI as well as train a simple deep learning model based on a small set of data in LADI.



## Prerequisite

To get started, users are required to check the following packages are installed successfully:

- scikit-image: The [scikit-image](https://scikit-image.org/) SciKit (toolkit for [SciPy](http://www.scipy.org/)) extends `scipy.ndimage` to provide a versatile set of image processing routines. To install scikit-image:

  - Install via shell/command prompt

    ```shell
    pip install scikit-image
    ```

  - Install in Anaconda or Miniconda environment 

    ```shell
    conda install -c conda-forge scikit-image
    ```

  - To check if scikit-image is installed successfully

     In Python:

    ```python
    import skimage
    skimage.__version__
    ```

    If there is no `SyntaxError: invalid syntax` error message, then scikit-image is installed successfully.

- pandas: [pandas](https://pandas.pydata.org/docs/index.html#) is a fast, powerful, flexible and easy to use open source data analysis and manipulation tool,built on top of the Python programming language. To install pandas:

  - Install via shell/command prompt

    ```shell
    pip install pandas
    ```

  - Install in Anaconda or Miniconda environment

    ```shell
    conda install pandas
    ```

  - To check if pandas is installed successfully

    In Python:

    ```python
    import pandas
    pandas.__version__
    ```

    If there is no `SyntaxError: invalid syntax` error message, then scikit-image is installed successfully.

*Note: If you are a Mac user and trying to install the packages via shell/command prompt, please replace `pip` with `pip3` in the commands above to ensure Python3 compatibility .*



## Write Custom `Dataset` From LADI

`torch.utils.data.Dataset` is an abstract class representing a dataset. Your custom dataset should inherit `Dataset` and override the following methods:

- `__len__` so that `len(dataset)` returns the size of the dataset.

- `__getitem__` to support the indexing such that `dataset[i]` can be used to get iith sample

  

To get started, users will need to import the packages. Users can specify the root directory that stores the images and the path of the CSV file that includes metadata of the images in root directory.

```python
from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

root_dir = '~/ladi/Images/flood_tiny/'
csv_file = '~/ladi/Images/flood_tiny/flood_tiny_metadata.csv'
label_csv = '~/ladi/Images/flood_tiny/flood_tiny_label.csv'
```

The `/flood_tiny/` path contains 100 random images which are labeled as `"damage:flood/water"` and 100 other random images which are not labeled as `"damage:flood/water"` in `ladi_aggregated_responses.tsv` file. The `flood_tiny_metadata.csv` contains metadata of these 200 images. It is a subset extracted from `ladi_images_metadata.csv` and is stored in the same directory with these 200 images. The `flood_tiny_label.csv` contains labels for the 200 images.

Then, users can write a simple helper function to show images in the following steps:

```python
def show_image(image):
    plt.imshow(image)
    # pause a bit so that plots are updated
    plt.pause(0.01)
```

The `flood_tiny_metadata.csv` contains 11 columns.

| column number | uuid       | timestamp     | gps_lat    | gps_lon    | gps_alt | file_size | width | height | s3_path                    | url         |
| ------------- | ---------- | ------------- | ---------- | ---------- | ------- | --------- | ----- | ------ | -------------------------- | ----------- |
| 1700          | eaf5...a58 | 10/6/15 21:49 | 32.9070717 | -80.396665 | 352     | 6100841   | 6000  | 6000   | s3://ladi/Ima...7a9f90.jpg | https://... |

The `flood_tiny_label.csv` contains 2 columns.

| s3_path                                           | label |
| ------------------------------------------------- | ----- |
| s3://ladi/Images/FEMA_CAP/9073/613822/_CAP1438... | True  |

In this case, `s3_path` will be the key to merge these two CSV files.

So, sample of our dataset will be a dict `{'image': image, 'image_name': img_name, 'damage:flood/water': label, uuid': uuid, 'timestamp': timestamp, 'gps_lat': gps_lat, 'gps_lon': gps_lon, 'gps_alt': gps_alt, 'orig_file_size': file_size, 'orig_width': width, 'orig_height': height}` containing the image, the label and the information in all 11 columns. Our dataset will take an optional argument `transform` so that any required processing can be applied on the sample. We will see the usefulness of `transform` in the next section.

```python
class FloodTinyDataset(Dataset):

    def __init__(self, csv_file, root_dir, label_csv, transform = None):
        """
        Args:
            csv_file (string): Path to the csv file with metadata.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.flood_tiny_metadata = pd.read_csv(csv_file)
        self.flood_tiny_label = pd.read_csv(label_csv)
        self.flood_tiny_data = pd.merge(self.flood_tiny_metadata, 
                                        self.flood_tiny_label,
                                       on="s3_path")
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.flood_tiny_metadata)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        pos = self.flood_tiny_metadata.iloc[idx, 9].rfind('/')+1
        img_name = os.path.join(self.root_dir, self.flood_tiny_metadata.iloc[idx, 9][pos:])
        
        image = Image.fromarray(io.imread(img_name))
        uuid = self.flood_tiny_data.iloc[idx, 1]
        timestamp = self.flood_tiny_data.iloc[idx, 2]
        gps_lat = self.flood_tiny_data.iloc[idx, 3]
        gps_lon = self.flood_tiny_data.iloc[idx, 4]
        gps_alt = self.flood_tiny_data.iloc[idx, 5]
        file_size = self.flood_tiny_data.iloc[idx, 6]
        width = self.flood_tiny_data.iloc[idx, 7]
        height = self.flood_tiny_data.iloc[idx, 8]
        label = self.flood_tiny_data.iloc[idx, -1]
        
        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'image_name': img_name, 'damage:flood/water': label, 'uuid': uuid, 'timestamp': timestamp, 'gps_lat': gps_lat, 'gps_lon': gps_lon, 'gps_alt': gps_alt, 'orig_file_size': file_size, 'orig_width': width, 'orig_height': height}

        return sample

```

Users can instantiate this custom dataset class `FloodTinyDataset` and iterate through the data samples. 

```python
flood_tiny_dataset = FloodTinyDataset(csv_file = csv_file, root_dir = root_dir, label_csv = label_csv)

fig = plt.figure()

for i in range(len(flood_tiny_dataset)):
    sample = flood_tiny_dataset[i]

    print(i, sample['damage:flood/water'], sample['image_name'], sample['uuid'], sample['timestamp'], sample['gps_lat'], sample['gps_lon'], sample['gps_alt'])

    ax = plt.subplot(2, 2, i + 1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    show_image(sample['image'])

    if i == 3:
        plt.show()
        break
```

The first 4 samples will shown below.

![img](https://github.com/NaeRong/DS440_Capstone/blob/master/custom_flood_tiny_dataset_output.png)

Output:

```shell
0 True ~/ladi/Images/flood_tiny/DSC_0194_eac1a77e-3de5-46bb-a8ef-b0410314fcb1.jpg 24870156dd4154436d29f11f2b2398776ff73dff 2015-10-08 16:56:55 33.4089333333333 -79.843 690.0
1 False ~/ladi/Images/flood_tiny/A0027-93_826eff88-456d-41b1-998e-e87f4cb91e2d.jpg fe822a019ed698f9dcde06d86f310c0c58074a5d 2017-09-26 12:32:21 18.2066466666667 -65.7328233333333 316.0
2 True ~/ladi/Images/flood_tiny/DSC_1346_9aa9aa0f-857a-4b31-b8e9-1a231b51da73.jpg 1b799c118853279449c17e8a2950292f03f76a1b 2017-09-08 15:41:36 18.460231666666697 -66.693275 412.0
3 False ~/ladi/Images/flood_tiny/A0027-94_355188f9-e39b-4a73-8208-17a1c2334215.jpg 4647d8ee717821ab77a74c979dda68a06a1cc9ca 2017-09-26 12:38:47 18.2105516666667 -65.7494166666667 288.0
```



## Image `Tranforms`

Sometimes, neural networks expect the images of the same size. However, in most datasets, image size is not fixed. This issue requires users to modify the original images to a different size. Some useful `transform` methods are shown below:

- `Resize`: to resize the input PIL Image to the given size.

- `RandomCrop`: to crop from image randomly. This is data augmentation.

- `RandomRotation`: to rotate the image by angle.

- `RandomHorizontalFlip`: to horizontally flip the given PIL Image randomly with a given probability.

- `ToTensor`: to convert the numpy images to torch images (we need to swap axes).

  

### Some Basic Transforms

#### `torchvision.transforms.Resize`(*size*, *interpolation=2*)

**Parameters** 

- **size** (*sequence* *or* *python:int*) – Desired output size. If size is a sequence like (h, w), output size will be matched to this. If size is an int, smaller edge of the image will be matched to this number. i.e, if height > width, then image will be rescaled to (size * height / width, size)
- **interpolation** (*python:int**,* *optional*) – Desired interpolation. Default is `PIL.Image.BILINEAR`



#### `torchvision.transforms.RandomCrop`(*size*, *padding=None*, *pad_if_needed=False*, *fill=0*, *padding_mode='constant'*)

**Parameters**

- **size** (*sequence* *or* *python:int*) – Desired output size of the crop. If size is an int instead of sequence like (h, w), a square crop (size, size) is made.

- **padding** (*python:int* *or* *sequence**,* *optional*) – Optional padding on each border of the image. Default is None, i.e no padding.

- **pad_if_needed** (*boolean*) – It will pad the image if smaller than the desired size to avoid raising an exception. Since cropping is done after padding, the padding seems to be done at a random offset.

- **fill** – Pixel fill value for constant fill. Default is 0. If a tuple of length 3, it is used to fill R, G, B channels respectively. This value is only used when the padding_mode is constant

- **padding_mode** –

  Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.



#### `torchvision.transforms.RandomRotation`(*degrees*, *resample=False*, *expand=False*, *center=None*, *fill=0*)

**Parameters**

- **degrees** (*sequence* *or* *python:float* *or* *python:int*) – Range of degrees to select from. If degrees is a number instead of sequence like (min, max), the range of degrees will be (-degrees, +degrees).
- **resample** (*{PIL.Image.NEAREST**,* *PIL.Image.BILINEAR**,* *PIL.Image.BICUBIC}**,* *optional*) – An optional resampling filter. See [filters](https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters) for more information. If omitted, or if the image has mode “1” or “P”, it is set to PIL.Image.NEAREST.
- **expand** (*bool**,* *optional*) – Optional expansion flag. If true, expands the output to make it large enough to hold the entire rotated image. If false or omitted, make the output image the same size as the input image. Note that the expand flag assumes rotation around the center and no translation.
- **center** (*2-tuple**,* *optional*) – Optional center of rotation. Origin is the upper left corner. Default is the center of the image.
- **fill** (*3-tuple* *or* *python:int*) – RGB pixel fill value for area outside the rotated image. If int, it is used for all channels respectively.



#### `torchvision.transforms.RandomHorizontalFlip`(*p=0.5*)

**Parameters**

**p** (*python:float*) – probability of the image being flipped. Default value is 0.5



#### `torchvision.transforms.ToTensor`

Convert a `PIL Image` or `numpy.ndarray` to tensor.

Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1) or if the numpy.ndarray has dtype = np.uint8

In the other cases, tensors are returned without scaling.



### Compose Transforms

Now, apply the transforms on a sample.

If users want to rescale the shorter side of the image to **2048** and then randomly crop a square of size **1792** from it. i.e, we want to compose `Resize` and `RandomCrop` transforms. `torchvision.transforms.Compose` is a simple callable class which allows us to do this.

```python
scale = transforms.Resize(2048)
crop = transforms.RandomCrop(1792)
rotate = transforms.RandomRotation(20)
flip = transforms.RandomHorizontalFlip(1)
composed = transforms.Compose([transforms.Resize(2048),
                               transforms.RandomCrop(1792),
                              transforms.RandomRotation(10),
                              transforms.RandomHorizontalFlip(1)])

# Apply each of the above transforms on sample.
fig = plt.figure()
sample = flood_tiny_dataset[198]
for i, tsfrm in enumerate([scale, crop, rotate, flip, composed]):
    transformed_image = tsfrm(sample['image'])

    ax = plt.subplot(2, 3, i + 1)
    plt.tight_layout()
    ax.set_title(type(tsfrm).__name__)
    show_image(transformed_image)

plt.show()
```

The three transforms of the sample are shown below.

![img](https://github.com/NaeRong/DS440_Capstone/blob/master/pytorch_transform_flood_tiny.png)



## Use `Dataloader` to Iterate Through Dataset 

Compared to simple `for` loop to iterate over data, `torch.utils.data.DataLoader` is an iterator which provides more features:

- Batching the data.
- Shuffling the data.
- Load the data in parallel using `multiprocessing` workers.

In the previous section, three `transforms` are performed on a sample. In this section, users can learn to use `Dataloader` to transform all images in the dataset.

First, a new dataset with `transform` needs to be defined.

```python
transformed_dataset = FloodTinyDataset(csv_file=csv_file, root_dir=root_dir, 
label_csv = label_csv, transform=transforms.Compose([transforms.Resize(2048), 
transforms.RandomCrop(1792), 
transforms.RandomRotation(10), 
transforms.RandomHorizontalFlip(), 
transforms.ToTensor()]))
```

Then, feed the new dataset `transformed_dataset` into `Dataloader`.

```python
dataloader = DataLoader(transformed_dataset, batch_size=4,
                        shuffle=True, num_workers=4)
```

To show batched images, users can write a helper function as shown below.

```python 
# Helper function to show a batch
def show_images_batch(sample_batched):
    images_batch = sample_batched['image']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)
    grid_border_size = 2

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    for i in range(batch_size):
        plt.title('Batch from dataloader')
```

At last, let `dataloader` transform the images in batches.

```python
for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['image'].size())

    # observe 4th batch and stop.
    if i_batch == 3:
        plt.figure()
        show_images_batch(sample_batched)
        plt.axis('off')
        plt.ioff()
        plt.show()
        break
```

The transformed images in the 4th batch:

![img](https://github.com/NaeRong/DS440_Capstone/blob/master/dataloader_batch_result.png)

The index and size of images in batch:

```shell
0 torch.Size([4, 3, 1792, 1792])
1 torch.Size([4, 3, 1792, 1792])
2 torch.Size([4, 3, 1792, 1792])
3 torch.Size([4, 3, 1792, 1792])
```



## Create Train and Test Sets

Training neural networks with PyTorch follows the explicit steps. In this step, we will determine the control factor during the training process. 

Information about Torch.Optim : https://pytorch.org/docs/stable/optim.html

```python
from torch.optim import Adam
```
* Decide how many classes you will be using for SimpleNet
```python
len(np.unique(label_damage['Answer']))
```
Step 1: Create the optimizer and Loss function
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

Step 2: Write a function to adjust learning rates

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
Step 3: Save and evaluate the model
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


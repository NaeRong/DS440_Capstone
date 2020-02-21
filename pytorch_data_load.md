# LADI In PyTorch Framework

**Table of Contents**

- [Introduction](#introduction)

- [Prerequisite](#prerequisite)

- [Write Custom Dataset From LADI](#write-custom-dataset-from-ladi)

- [Image Tranforms](#image-tranforms)
  - [Three Basic Transforms](#three-basic-transforms)
    - [torchvision.transforms.Resize(<em>size</em>, <em>interpolation=2</em>)](#torchvisiontransformsresizesize-interpolation2)
    - [torchvision.transforms.RandomCrop(<em>size</em>, <em>padding=None</em>, <em>pad_if_needed=False</em>, <em>fill=0</em>, <em>padding_mode='constant'</em>)](#torchvisiontransformsrandomcropsize-paddingnone-pad_if_neededfalse-fill0-padding_modeconstant)
    - [torchvision.transforms.ToTensor](#torchvisiontransformstotensor)
  - [Compose Transforms](#compose-transforms)

- [Use Dataloader to Iterate Through Dataset](#use-dataloader-to-iterate-through-dataset)

- [Create Train and Test Sets](#create-train-and-test-sets)

*Note: This tutorial/documentation is adapted from [PyTorch Data Loading Tutorial](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html) to fit in LADI Dataset.*


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
```

The `/flood_tiny/` path contains 100 random images which are labeled as `"damage:flood/water"` in `ladi_aggregated_responses.tsv` file. The `flood_tiny_metadata.csv` contains metadata of these 100 images. It is a subset extracted from `ladi_images_metadata.csv` and is stored in the same directory with these 100 images.

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

So, sample of our dataset will be a dict `{'image': image, 'image_name': img_name, 'uuid': uuid, 'timestamp': timestamp, 'gps_lat': gps_lat, 'gps_lon': gps_lon, 'gps_alt': gps_alt, 'orig_file_size': file_size, 'orig_width': width, 'orig_height': height}` containing the image and the information in all 11 columns. Our dataset will take an optional argument `transform` so that any required processing can be applied on the sample. We will see the usefulness of `transform` in the next section.

```python
class FloodTinyDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform = None):
        """
        Args:
            csv_file (string): Path to the csv file with metadata.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.flood_tiny_metadata = pd.read_csv(csv_file)
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
        uuid = self.flood_tiny_metadata.iloc[idx, 1]
        timestamp = self.flood_tiny_metadata.iloc[idx, 2]
        gps_lat = self.flood_tiny_metadata.iloc[idx, 3]
        gps_lon = self.flood_tiny_metadata.iloc[idx, 4]
        gps_alt = self.flood_tiny_metadata.iloc[idx, 5]
        file_size = self.flood_tiny_metadata.iloc[idx, 6]
        width = self.flood_tiny_metadata.iloc[idx, 7]
        height = self.flood_tiny_metadata.iloc[idx, 8]
        
        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'image_name': img_name, 'uuid': uuid, 'timestamp': timestamp, 'gps_lat': gps_lat, 'gps_lon': gps_lon, 'gps_alt': gps_alt, 'orig_file_size': file_size, 'orig_width': width, 'orig_height': height}

        return sample

```

Users can instantiate this custom dataset class `FloodTinyDataset` and iterate through the data samples. 

```python
flood_tiny_dataset = FloodTinyDataset(csv_file = csv_file, root_dir = root_dir)

fig = plt.figure()

for i in range(len(flood_tiny_dataset)):
    sample = flood_tiny_dataset[i]

    print(i, sample['image_name'], sample['uuid'], sample['timestamp'], sample['gps_lat'], sample['gps_lon'], sample['gps_alt'])

    ax = plt.subplot(1, 4, i + 1)
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
0 ~/ladi/Images/flood_tiny/DSC_1607_6e0b5f5d-7935-4c9a-9950-1599c27a9f90.jpg eaf5cab99e5ddca7bf039f69aaa7ae3a719f8a58 2015-10-06 21:49:05 32.907071666666695 -80.396665 352.0
1 ~/ladi/Images/flood_tiny/DSC_0665_c4039ba2-af88-4e5d-a914-205699e1829c.jpg c3427b8c3a4d4f2cc383cca4153e5a3d66a21639 2015-10-08 12:54:40 33.4639066666667 -79.5533583333333 465.0
2 ~/ladi/Images/flood_tiny/DSC_9683_510521be-8714-4254-8f4f-63e700537c67.jpg f68a13d1ac3de53d9cc20964186a518eea1e5462 2017-09-08 13:13:07 29.774221666666694 -94.25206666666668 289.0
3 ~/ladi/Images/flood_tiny/A0085_AP_0a946d44-c4f3-4bf9-82f3-e052b6d5f2e1.jpg 2deea53aaa61931512ee56f3e0cebd568fdb7dba 2017-10-04 17:00:39 18.3035433333333 -65.65087333333331 451.0
```



## Image `Tranforms`

Sometimes, neural networks expect the images of the same size. However, in most datasets, image size is not fixed. This issue requires users to modify the original images to a different size. Three useful `transform` methods are shown below:

- `Resize`: Resize the input PIL Image to the given size.

- `RandomCrop`: to crop from image randomly. This is data augmentation.

- `ToTensor`: to convert the numpy images to torch images (we need to swap axes).

  

### Three Basic Transforms

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



#### `torchvision.transforms.ToTensor`

Convert a `PIL Image` or `numpy.ndarray` to tensor.

Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1) or if the numpy.ndarray has dtype = np.uint8

In the other cases, tensors are returned without scaling.



### Compose Transforms

Now, apply the transforms on a sample.

If users want to rescale the shorter side of the image to **2048** and then randomly crop a square of size **1792** from it. i.e, we want to compose `Resize` and `RandomCrop` transforms. `torchvision.transforms.Compose` is a simple callable class which allows us to do this.

```python
from PIL import Image

scale = transforms.Resize(2048)
crop = transforms.RandomCrop(1792)
composed = transforms.Compose([transforms.Resize(2048),
                               transforms.RandomCrop(1792)])

# Apply each of the above transforms on sample.
fig = plt.figure()
sample = flood_tiny_dataset[62]
for i, tsfrm in enumerate([scale, crop, composed]):
    transformed_image = tsfrm(sample['image'])

    ax = plt.subplot(1, 3, i + 1)
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
transformed_dataset = FloodTinyDataset(csv_file=csv_file,
                                           root_dir=root_dir,
                                           transform=transforms.Compose([
                                               transforms.Resize(2048),
                               	transforms.RandomCrop(1792),
                                               transforms.ToTensor()
                                           ]))
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

- `ImageFolder`


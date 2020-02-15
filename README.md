# LADI Dataset Documentation

[TOC]

# Download

Data in LADI is stored in AWS S3 storage. To access or download images, metadata, and labels in LADI, you can optionally choose to work with Amazon AWS Console or not.

## Using AWS S3 Console

To use Amazon S3, you need an AWS account. If you do not have one yet, please visit the [Amazon Web Services Homepage](https://aws.amazon.com/) and follow the tutorial on [Create and Activate an AWS Account](https://aws.amazon.com/premiumsupport/knowledge-center/create-and-activate-aws-account/) to create an AWS account.

After you have created and activated your account, you can choose to download LADI from AWS S3 to your local machine using AWS Command Line Interface or transfer LADI dataset into your own S3 bucket.

### Download LADI to Local Machine with AWS Command Line Interface

1. Go to [AWS Command Line Interface User Guide](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-install.html) to install AWS CLI on your system. You have the options to install AWS CLI on Linux, MacOS, Windows and Virtual Environment.

2. Verify that AWS CLI is installed correctly.

   ```shell
   $aws --version
   aws-cli/1.17.12 Python/3.7.3 Darwin/19.3.0 botocore/1.14.12
   ```

3. Create a new administrator IAM User.  Go to [Creating Your First IAM Admin User and Group](https://docs.aws.amazon.com/IAM/latest/UserGuide/getting-started_create-admin-group.html) to create a new IAM User. After you have created a new IAM User, please go to [Navigation Pane for Users](https://console.aws.amazon.com/iam/home#/users) to verify.

4. Configure the AWS CLI.

   ```shell
   $aws configure
   AWS Access Key ID [None]: AKIAIOSFODNN7EXAMPLE
   AWS Secret Access Key [None]:
   aJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
   Default region name [None]: us-west-2
   Default output format [None]: json
   ```

   Please fill in each field step by step with your own values. The `AWS Access Key ID` and `AWS Secret Access Key` can be found by going to [Navigation Pane for Users](https://console.aws.amazon.com/iam/home#/users), choosing the IAM User that you just created, e.g. **Administrator**, clicking on the **Security credentials** tab and tapping the **Create access key** button.  
   For more information and details, please visit [Configuring the AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html).

5. Run the following command to download the LADI dataset. The` --recursive` flag specifies that subdirectories should be copied.

   ```shell
   $aws s3 cp s3://ladi/path/to/remote path/to/local --recursive
   ```

   `path/to/remote` should be replaced with the path of the data within the LADI S3 bucket and `path/to/local` represents the local path where the files will be written.

6. Go to the local path specified in the previous step and verify the requested files from LADI has been downloaded.

### Transfer LADI to Your Own S3 Bucket

1.  Please follow Step 1 to Step 4 in the "Download LADI to Local Machine with AWS Command Line Interface" section to install and configure AWS CLI.

2. Sign in to the AWS Management Console and open the [Amazon S3 console](https://console.aws.amazon.com/s3/).

3. Follow the tutorial on [Creating an S3 Bucket](https://docs.aws.amazon.com/AmazonS3/latest/user-guide/create-bucket.html) to create a bucket.

4. Go to the [Amazon S3 console](https://console.aws.amazon.com/s3/) to verify that your new bucket has been created.

5. Run the following command to copy data from the original LADI S3 bucket to your own bucket.

   ```shell
   $aws s3 cp s3://ladi/path/to/remote s3://yourbucketname/yourpath --recursive
   ```

   `path/to/remote` should be replaced with the path of the data within the LADI S3 bucket, `yourbucketname` should be replaced with the name of your new bucket and `yourpath` represents the bucket path where the files will be written.

6. Go to the [Amazon S3 console](https://console.aws.amazon.com/s3/) to verify that the requested files from LADI have been transferred.

## Access and Download LADI Data via Web Browser

Without Amazon AWS account and services, users can also access and download files in LADI by going to http://ladi.s3-us-west-2.amazonaws.com/index.html using a web browser. However, due to efficiency of downloading the data, we highly recommend to use AWS CLI.

# Load LADI in Python 3

Users can load files and data from LADI in Python 3 by using AWS Python SDK Boto 3 if the dataset is stored in AWS S3 bucket. LADI can also be load using other packages: Pandas, NumPy and PyTorch. Using the following command to install these packages:

- Boto 3

  ```shell
  $pip install boto3
  ```

  For more information for installing Boto3, please go to [Boto 3 Quickstart](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html).

- Pandas

  ```shell
  $pip install pandas
  ```

  For more information for installing Pandas with Anaconda and Miniconda, please go to [Pandas Installation](https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html).

- Numpy

  ```shell
  $pip install numpy
  ```

  For more installation options, please visit [SciPy Installation](https://scipy.org/install.html).

- PyTorch

  ```shell
  $pip install torch torchvision
  ```

  For more details on installing PyTorch, please visit [PyTorch Start Locally](https://pytorch.org/get-started/locally/).

  *Note: If you are a Mac user, replace pip with pip3 in the commands above to install those packages. If you are installing those packages via Anaconda, please refer to the links provided above for more information about commands to use for installation.*

## LADI Stored in S3 Bucket

If you transferred LADI into your own AWS S3 bucket and prefer to not store the files to your local machine, Boto 3, the AWS Python SDK, can help you access and read files in S3 bucket.
*Example: Using Boto3 and Pandas to read a .csv file from LADI stored in S3 bucket.*

1. To access the AWS S3 bucket import package Boto 3.

   ```shell
   $import bobo3
   ```

2. Initiate "clients" tool in Boto3.  “Clients” are low-level functional interfaces which are useful to load single files.

   ```python
   import pandas as pd
   
   #replace 'ladi_images_metadata.csv' with the path of the file that you want to read
   file_1_path = 'ladi_images_metadata.csv'
   obj_1 = client.get_object(Bucket = bucket_name, Key = file_1_path)
   image_metadata = pd.read_csv(obj_1['Body'])
   ```

   `obj` contains metadata of the file and the `Body` of the object contains actual data in a `StreamingBody` format. If we display first 10 rows in the `image_metadata`, we can get a table as following:
   ![img](https://lh5.googleusercontent.com/r9EpfcV23Nq6-KlWS-EmoWpidh9Ae6qam83oXypTZcGMO8c4CqmWdWzeeEuhWpV3X9uRUSnaH-iiCg_ox5bSIAEPcTPwcBulEAEyT1UbiRT1XKHyyguM4zft0w2HkicEdSf95lw)

## LADI Stored in Local Machine

If you downloaded LADI to your local machine, you are able to read the files using Pandas and other packages without the assistance of Boto 3.

*Example: Read and display an image from the LADI dataset on a local machine.*

There are multiple effective ways in Python to read and display an image file. In this example, we use Python Imaging Library (PIL) and matplotlib to read and show an image.

```python
from PIL import Image
import matplotlib.pyplot as plt

image_path = 'Images/FEMA_CAP/1012/20118/VIRB0002_fa5065eb-773a-4b41-8f2c-80a734f3770d.jpg'

im = Image.open(image_path)
plt.imshow(im, cmap='Greys_r')
```

**INSERT BETTER IMAGE HERE**

# Clean and Validate LADI Dataset

## Task 1: Generate Ground Truth Labels from Aggregated Responses .tsv file

For our first approach, we will use human generated labels as the most accurate feature to represent each image. Our goal is to clean and validate the LADI dataset so that we can use them to train our image classification model. 

1. Download human generated label dataset into local environment and export the dataset into Cloud environment.

   ```python
   import pandas as pd
   import numpy as np
   human_label = "/content/drive/My Drive/Colab Notebooks/DS440_data/ladi_aggregated_responses.tsv"
   file = pd.read_csv(human_label,delimiter='\t',header='infer')
   ```

- Data information: 

- The LADI aggregated responses:

  - Provides human annotations of images in the LADI FEMA_CAP dataset. Each image might contain multiple annotations from different workers.

- Data points: 193663 rows

- Fields: 

  - *img_url* 
  - *WorkerId*
  - *Answer :* Dataset contains 495 categories

- Answer outputs can be similar to the following: 
  **INSERT PHOTO HERE**

  Among the different categories, our project will first focus on ‘damage’ and ‘infrastructure’ labels.

2. Extract labels with damage and infastructure categories

   ```python
   label_damage_infra = file[file['Answer'].str.contains('damage|infrastructure',na=False,case=False)]
   ```

3. Frequency count on how many labels one image_url has:

   ```python
   label_damage_infra['Freq'] = label_damage_infra.groupby('img_url')["Answer"].transform('size') 
   ```

   **INSERT PHOTO HERE**

4. Extract the maximum count in one image url

   ```python
   label_damage_infra['Freq'] = label_damage_infra.groupby('img_url')["Freq"].transform('max') 
   ```

5. Identify the ground truth label for each image url
   By calculating the most frequent label for each url, we are able to identify the ground truth for our deep learning model. 

## Task 2: Data Cleaning for ladi_images_metadata.csv



## Task 3: Features Selection Based on Image Metadata and Aggregated Response Data

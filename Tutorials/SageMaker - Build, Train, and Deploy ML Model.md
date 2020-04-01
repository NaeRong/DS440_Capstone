# SageMaker - Build, Train, and Deploy ML Model

*If you have not set up an AWS Account (free tier can be used for SageMaker), please do so before continuing*.

**Table of Contents**

- [Step 1: Enter the Amazon SageMaker Console](#step-1-enter-the-amazon-sagemaker-console)
- [Step 2: Create an Amazon SageMaker notebook instance](#step-2-create-an-amazon-sagemaker-notebook-instance)
- [Step 3: Prepare the data](#step-3-prepare-the-data)

## Step 1: Enter the Amazon SageMaker Console

1. Enter Amazon SageMaker console

![img](https://github.com/NaeRong/DS440_Capstone/blob/master/Images/sagemaker1.png)

## Step 2: Create an Amazon SageMaker notebook instance

1. Navigate to the Amazon SageMaker dashboard located in the left-hand panel and select **Notebook Instances**.
![img](https://github.com/NaeRong/DS440_Capstone/blob/master/Images/sagemaker2.png)
2. Select **Create notebook instance** button to begin creating the notebook instance.
![img](https://github.com/NaeRong/DS440_Capstone/blob/master/Images/sagemaker3.png)
3. Enter a name in the **Notebook instance name** field.  Keep ml.t2.medium as the **Notebook instance type**.
![img](https://github.com/NaeRong/DS440_Capstone/blob/master/Images/sagemaker4.png)
4. An IAM role must be specified to enable the notebook instance to access and securely upload data to Amazon S3.  In the **IAM role** field, choose **Create a new role** to have Amazon SageMkaer create a role with the required permissions and assign it to your instance.
![img](https://github.com/NaeRong/DS440_Capstone/blob/master/Images/sagemaker5.png)
5. In the **Create an IAM role** box, select **Any S3 bucket**.  This allows your Amazon SageMkaer instance to access all S3 buckets in your account.  For now, we will allow **Any S3 bucket** until we upload our our own bucket.
![img](https://github.com/NaeRong/DS440_Capstone/blob/master/Images/sagemaker6.png)
   1. If you would like to clone our Git respository for your notebook instance, go to the **Git repositories** box and select **Clone a public Git repository to this notebook instance only**.  Then insert "https://github.com/NaeRong/DS440_Capstone.git" in the **Git repository URL** box.
6. Choose **Create role**.
7. On the **Notebook instances** page, your new instance with your specified name should be shown with a **Pending** status.  Your noteobok instance should transition from **Pending** to **InService** status in less than 2 minutes.
![img](https://github.com/NaeRong/DS440_Capstone/blob/master/Images/sagemaker7.png)

## Step 3: Prepare the data

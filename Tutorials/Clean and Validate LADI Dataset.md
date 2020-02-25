# Clean and Validate LADI Dataset
- [Clean and validate dataset](#clean_and_validate_ladi_dataset)
  * [Generate Ground Truth Labels from Aggregated Responses .tsv file](#generate_labels_from_aggregated_responses_file)
  * [Data Cleaning for ladi_images_metadata.csv](#data_cleaning_for_ladi_images_metadata)
  * [Features Selection Based on Image Metadata and Aggregated Response Data](#features_selection_based_on_image_metadata_and_aggregated_response_data)

## Generate_Labels_from_Aggregated_Responses_file

For our first approach, we will use human generated labels as the most accurate feature to represent each image. Our goal is to clean and validate the LADI dataset so that we can use them to train our image classification model. 

The image url can have multiple labels associated to it. Therefore, for workers that gave multiple labels in ancwer, you should seperate each of the labels withint the answer. The single response shows all the labels that one worker gave withint a single category.

Human generated labels cover 20% of the total images in the dataset. 

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


  ![img](https://github.com/NaeRong/DS440_Capstone/blob/master/Images/Label_Human.png)

  Among the different categories, our project will first focus on ‘damage’ and ‘infrastructure’ labels.

2. Strip off bracket and comma from the Answer catagory

   ```python
   file["Answer"] = file["Answer"].str.strip('[|]')
   file["Answer"] = file["Answer"].str.split(",",expand = True)
   ```
3. Extract labels with damage and infastructure categories

   ```python
   label_damage_infra = file[file['Answer'].str.contains('damage|infrastructure',na=False,case=False)]
   ```
4. Extract labels with damage and infastructure categories

   ```python
   label_damage_infra = file[file['Answer'].str.contains('damage|infrastructure',na=False,case=False)]
   ``` 
5*. (Optional) Save new dataset into csv file

   ```python
   label_damage_infra.to_csv('/content/drive/My Drive/Colab Notebooks/DS440_data/label_damage_infra.csv')
   ```   

## Data_Cleaning_for_ladi_images_metadata



## Features_Selection_Based_on_Image_Metadata_and_Aggregated_Response_Data

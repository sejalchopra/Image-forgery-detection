# Image-forgery-detection 

This project aims to detect image forgeries using a Convolutional Neural Network (CNN) implemented in PyTorch. Inspired by the work of Y. Rao et al. on A Deep Learning Approach to Detection of Splicing and Copy-Move Forgeries in Images, our approach involves extracting features using a CNN, followed by feature fusion, and finally classification using an SVM from scikit-learn. The datasets used in this project are the CASIA2 and the NC2016 datasets.

## Getting Started

### Setup Project Environment

1. Install Python and Anaconda (We recommend Python 3.9 because of Compatibility with most of the libraries we would be using)
2. Now create a virtual environment to keep the project dependencies isolated after which we activate the environment. This all can be done by using the following commands:
`conda create -n <name> python=3.9`
`conda activate <name>`
3.Install the required packages for this project, we would first install pip followed by requirements.
`conda install pip`
`pip install -r requirements.txt`
4. After setting up the above requirements, clone the repository to get the project (link provided in report) -
	`git clone <repository_url>`

### Datasets 
1. Acquire the datasets – CASIA2 and NC2016 online and place them in the “data” folder of the project.

### Running the Project
1. Navigate to the ' src/’ folder 
`cd src`

#### The following files need to be run in the same order as below

1)  `extract_patches.py` -This code is used to extract training patches which will be fed into the CNN
2)  `train_net.py` - This code will be used to train the entire CNN model and produce a model .pt file
3)  `feature_extraction.py`- This code takes the trained CNN model as input from the previous step and  extracts the features which are stored in a csv file 
4)  `svm_classification.py` - This code performs the cross validation and outputs the evaluation metrics.


### Now Lets run the Above Mentioned Scripts:

1. Run the patch extraction script to create image patches for both tampered and untampered regions.
`python extract_patches.py`
2. Now we would use the extracted image patches to train our cnn model and save it in ‘data/output/pre_trained_cnn‘ directory.
`python train_cnn.py`
3. Now we can execute the feature extraction script. This script will generate 400-D feature representations for each image using the trained CNN model. This script will create and save the fused feature vectors for each image in the ’data/output/features’ folder.
`python feature_extraction.py`
4. The last step is to do SVM classification. 
`python svm_classification.py`

This will train and test the SVM classifier on the extracted features and report the accuracy and cross-entropy loss per epoch for each dataset.

After executing the SVM classification script, we obtain the 10-fold cross-validation accuracy for both datasets and the final files generated can be checked in output folder.

## Output
Output directory : `data\output`
It contains all the data generated as output during the execution of the project. This directory has subdirectories for different types of output data, including accuracy, features, loss function, and trained models.

### Accuracy 
`data\output\accuracy`
The accuracy directory contains two CSV files - CASIA2_Accuracy.csv and NC16_Accuracy.csv - that record the accuracy of the models trained on the CASIA2 and NC16 datasets, respectively.

### Features
`data\output\features`
The features directory contains two CSV files - CASIA2_extracted_features.csv and NC16_extracted_features.csv - that store the extracted features for each dataset. These features can be used for further analysis or model training.

### Loss Function
`data\output\loss_function`
The loss_function directory contains two CSV files - CASIA2_Loss.csv and NC16_Loss.csv - that record the loss function trend values for the models trained on the CASIA2 and NC16 datasets, respectively.

### Trained Models
`data\output\trained_models`
The trained_models directory contains the final trained models saved in PyTorch format - Cnn_casia2.pt and Cnn_nc16.pt - for the CASIA2 and NC16 datasets, respectively. These models can be used for prediction or further fine-tuning.

## Report
A Final Report is Available in the Report Section of the Repository

## Team Information 

1. Sejal Chopra (40164708) 
2. Praveen Singh (40199511) 
3. Elvin Rejimone (40193868) 
4. Anushka Sharma (40159259)

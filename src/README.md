The following files need to be run in the same order as below

1)  `extract_patches.py` -This code is used to extract training patches which will be fed into the CNN
2)  `train_net.py` - This code will be used to train the entire CNN model and produce a model .pt file
3)  `feature_extraction.py`- This code takes the trained CNN model as input from the previous step and  extracts the features which are stored in a csv file 
4)  `svm_classification.py` - This code performs the cross validation and outputs the evaluation metrics.

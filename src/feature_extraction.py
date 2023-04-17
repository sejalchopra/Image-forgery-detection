# Feature Extraction - Takes the trained CNN model as input from the previous step and extracts the features which are stored in a csv file
import os
import torch
from cnn.cnn import CNN
from feature_fusion.feature_vector_generation import create_feature_vectors, create_feature_vectors_nc

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

with torch.no_grad():
    # Instantiating a CNN model
    model = CNN()
    # Loading the pre-trained weights of the model
    model.load_state_dict(torch.load('Cnn_nc16.pt', map_location=lambda storage, loc: storage))

    # Setting the model to evaluation mode
    model.eval()

    # Converting model parameters to double precision for better accuracy
    model = model.double()

    # Setting the paths to authentic and tampered images directories
    authentic_path = '../data/NC2016/world/*'
    tampered_path = '../data/NC2016/probe/*'

    # Setting the name of the output file
    output_filename = 'NC2016_WithRot_LR001_b128_nodrop.csv'

    # Calling the create_feature_vectors_nc function to create feature vectors for the images
    create_feature_vectors_nc(model, '../data/NC2016/', output_filename)

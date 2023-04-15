import os
import torch
from cnn.cnn import CNN
from feature_fusion.feature_vector_generation import create_feature_vectors, create_feature_vectors_nc

# Set the KMP_DUPLICATE_LIB_OK environment variable
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

with torch.no_grad():
    model = CNN()
    model.load_state_dict(torch.load('Cnn_nc16.pt',
                                     map_location=lambda storage, loc: storage))
    model.eval()
    model = model.double()

    authentic_path = '../data/NC2016/world/*'
    tampered_path = '../data/NC2016/probe/*'
    output_filename = 'NC2016_WithRot_LR001_b128_nodrop.csv'
    # create_feature_vectors(model, tampered_path, authentic_path, output_filename)
    create_feature_vectors_nc(model,'../data/NC2016/' , output_filename)

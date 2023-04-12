import os
import torch
from cnn.cnn import CNN
from feature_fusion.feature_vector_generation import create_feature_vectors

# Set the KMP_DUPLICATE_LIB_OK environment variable
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

with torch.no_grad():
    model = CNN()
    model.load_state_dict(torch.load('Cnn.pt',
                                     map_location=lambda storage, loc: storage))
    model.eval()
    model = model.double()

    authentic_path = '../data/CASIA2/Au/*'
    tampered_path = '../data/CASIA2/Tp/*'
    output_filename = 'CASIA2_WithRot_LR001_b128_nodrop.csv'
    create_feature_vectors(model, tampered_path, authentic_path, output_filename)

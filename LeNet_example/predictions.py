import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST
from torch.nn import functional as F
import imutils
import cv2
from prettytable import PrettyTable

def get_prediction(model, img, label, test_data, T=15):
    '''
    Takes in a pytorch model, an image and a label of the kind containted in the 'DataLoader' iterable,
    and a pytorch Dataset object. Also takes an optional parameter 'T', the number of times the network
    is asked to make a prediction per image (more on what it means in future updates).
    Returns a dict with the ground truth label and the predicted label.
    '''
    
    with torch.no_grad():

        # getting original img and interpreting the label
        original_img = img.numpy().squeeze(axis=(0,1))
        ground_truth_label = test_data.dataset.classes[label.numpy()[0]]

        input_image = img.unsqueeze(0)
        input_images = img.repeat(T, 1, 1, 1)
        
        # getting network output
        network_output = model(input_images)
        prediction = torch.mean(network_output, dim=0).cpu().detach().numpy()
        predicted_indx = prediction.argmax(axis=0)
        predicted_label = test_data.dataset.classes[predicted_indx]
        
    results = {
        'ground_truth_label': ground_truth_label,
        'predicted_label': predicted_label
    }

    return results

def make_prediction_image(img, prediction_results, save_file='test_output.png'):
    '''
    Generates and saves a png of the provided image, overlaid with the ground truth label.
    If the predicted label matches the ground truth, the label will appear in green, red otherwise.
    Takes the image, a dict of prediction results, and an optional filename.
    '''
    ground_truth_label = prediction_results['ground_truth_label']
    predicted_label = prediction_results['predicted_label']
    color = (0, 255, 0) if ground_truth_label == predicted_label else (0, 0, 255)
    
    original_img = img.numpy().squeeze(axis=(0,1))
    original_img = np.dstack([original_img] * 3)
    original_img = imutils.resize(original_img, width=128)
    original_img = cv2.putText(original_img, ground_truth_label, (2, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    
    #text_summary = "Trial: {}\n Ground truth label: {}, predicted label: {}\n".format(trial_no, ground_truth_label, predicted_label)
    #uncertainty_annotation = "uncertainty from model: {}\nuncertainty from data: {}\ntotal uncertainty: {}".format(epistemic, aleatoric, total_uncertainty)
    #print(text_summary, uncertainty_annotation)
    
    cv2.imwrite(save_file, original_img*255)

    return

def get_prediction_with_uncertainties(model, img, label, test_data, T=15, normalized=False):
    '''
    Takes in a pytorch model, an image and a label of the kind containted in the 'DataLoader' iterable,
    and a pytorch Dataset object. Also takes an optional parameter 'T', the number of times the network
    is asked to make a prediction per image (more on what it means in future updates).
    Returns a dict with the ground truth label and the predicted label.
    '''
    with torch.no_grad():
        # getting original img and interpreting the label
        original_img = img.numpy().squeeze(axis=(0,1))
        ground_truth_label = test_data.dataset.classes[label.numpy()[0]]

        input_image = img.unsqueeze(0)
        input_images = img.repeat(T, 1, 1, 1)
        
        # getting network output
        network_output = model(input_images)
        prediction = torch.mean(network_output, dim=0).cpu().detach().numpy()
        predicted_indx = prediction.argmax(axis=0)
        predicted_label = test_data.dataset.classes[predicted_indx]

        # getting uncertainties
        if normalized:
            pred = F.softplus(network_output)
            p_hat = pred / torch.sum(pred, dim=1).unsqueeze(1)
        else:
            p_hat = F.softmax(network_output, dim=1)
        p_hat = p_hat.detach().cpu().numpy()
        p_bar = np.mean(p_hat, axis=0)

        temp = p_hat - np.expand_dims(p_bar, 0)
        epistemic_ucs = np.dot(temp.T, temp) / T
        epistemic_ucs = np.diag(epistemic_ucs)

        aleatoric_ucs = np.diag(p_bar) - (np.dot(p_hat.T, p_hat) / T)
        aleatoric_ucs = np.diag(aleatoric_ucs)

        epistemic = epistemic_ucs[predicted_indx]
        aleatoric = aleatoric_ucs[predicted_indx]
        total_uncertainty = np.sqrt(epistemic**2 + aleatoric**2)

        results = {
            'ground_truth_label': ground_truth_label,
            'predicted_label': predicted_label,
            'epistemic_uncertainty': epistemic,
            'aleatoric_uncertainty': aleatoric,
            'total_uncertainty': total_uncertainty
        }
        
        return results

            
def make_uncertainties_ascii(prediction_results, show=True):

    uc_table = PrettyTable()
    uc_table.add_column("Category", ['Ground Truth', 'Prediction', 'Epistemic Uncertainty', 'Aleatoric Uncertainty', 'Total Uncertainty'])
    ground_truth_label = prediction_results['ground_truth_label']
    predicted_label = prediction_results['predicted_label']
    epistemic_uncertainty = prediction_results['epistemic_uncertainty']
    aleatoric_uncertainty = prediction_results['aleatoric_uncertainty']
    total_uncertainty = prediction_results['total_uncertainty']
    uc_table.add_column("Result", [ground_truth_label, predicted_label, epistemic_uncertainty, aleatoric_uncertainty, total_uncertainty])
    table_str = uc_table.get_string()
    if show:
        print(table_str)
    
    return table_str



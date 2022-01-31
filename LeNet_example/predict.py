import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST
import argparse
import imutils
import torch
import cv2

from predictions import get_prediction_with_uncertainties

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True, help="path to trained model")
    parser.add_argument("-p", "--plots", type=str, required=True, help="path to save output images")
    args = parser.parse_args()

    # setting the device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # loading the MNIST dataset and grabbing 10 random data points
    print("Loading data...")
    test_data = MNIST(root='data', train=False, download=True, transform=ToTensor())
    indxs = np.random.choice(range(len(test_data)), size=(10,))
    test_data = Subset(test_data, indxs)

    # initializing test data loader
    test_data_loader = DataLoader(test_data, batch_size=1)

    # loading the model
    model = torch.load(args.model).to(device)
    model.eval()

    # making predictions
    with torch.no_grad():
        trial_no = 0
        for (img, label) in test_data_loader:
            trial_no += 1
            # get original image and ground truth
            original_img = img.numpy().squeeze(axis=(0,1))
            ground_truth_label = test_data.dataset.classes[label.numpy()[0]]

            # send input to device and make prediction
            img = img.to(device)
            #prediction = model(img)
            pred_results = get_prediction_with_uncertainties(model, img, label, test_data)
            ground_truth_label = pred_results['ground_truth_label']
            prediction = pred_results['predicted_label']
            epistemic_ucs = pred_results['epistemic_uncertainty']
            aleatoric_ucs = pred_results['aleatoric_uncertainty']
            total_uncertainty = pred_results['total_uncertainty']
            #prediction, epistemic_ucs, aleatoric_ucs = get_uncertainty_per_image(model, img)

            # printing predictions
            #print("\ntrial #{}".format(trial_no))
            #print("prediction: ", prediction)
            #print("epistemic uncertainties: ", epistemic_ucs)
            #print("aleatoric uncertainties: ", aleatoric_ucs)

            # find the class label index with the highest corresponding probability
            #indx = prediction.argmax(axis=1).cpu().numpy()[0]
            indx = prediction.argmax(axis=0)
            predicted_label = test_data.dataset.classes[indx]
            epistemic = epistemic_ucs[indx]
            aleatoric = aleatoric_ucs[indx]
            total_uncertainty = np.sqrt(epistemic**2 + aleatoric**2)

            # convert the image from grayscale to rgb and resize it
            original_img = np.dstack([original_img] * 3)
            original_img = imutils.resize(original_img, width=128)

            # drawing the predicted label on the image
            color = (0, 255, 0) if ground_truth_label == predicted_label else (0, 0, 255)
            original_img = cv2.putText(original_img, ground_truth_label, (2, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
            
            text_summary = "Trial: {}\n Ground truth label: {}, predicted label: {}\n".format(trial_no, ground_truth_label, predicted_label)
            uncertainty_annotation = "uncertainty from model: {}\nuncertainty from data: {}\ntotal uncertainty: {}".format(epistemic, aleatoric, total_uncertainty)

            # displaying the result in the terminal 
            print(text_summary+uncertainty_annotation)
            
            #cv2.imshow("image", original_img)
            #cv2.waitKey(0)

            # imshow expects values in [0,1]
            # imwrite expects values in [0, 255]
            # so we multiply by 225 before saving
            image_title = "trial_{}.png".format(trial_no)
            cv2.imwrite(args.plots+'/'+image_title, original_img*255)
    

            

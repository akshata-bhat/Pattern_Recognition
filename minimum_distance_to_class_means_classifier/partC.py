import numpy as np
import math
from plotDecBoundaries import plotDecBoundaries


def extract_data(fname):
    data = np.genfromtxt(fname, delimiter=",")
    rows, cols = np.shape(data)
    feature_count = cols - 1
    class_count = int(np.max(data[:, cols - 1]))
    class_col = data[:, 13]
    features_class1 = data[np.nonzero(class_col == 1)]
    features_class2 = data[np.nonzero(class_col == 2)]
    features_class3 = data[np.nonzero(class_col == 3)]
    return data, features_class1, features_class2, features_class3, class_count, feature_count


# Function definition to calculate the class mean for each feature of each class and store it in a 2D array
def class_means_calc(class_count, features_class1, features_class2, features_class3):
    class_mean = np.zeros(shape=[class_count, 2])
    class_mean[0, 0] = np.mean(features_class1[:, 0])
    class_mean[0, 1] = np.mean(features_class1[:, 1])
    class_mean[1, 0] = np.mean(features_class2[:, 0])
    class_mean[1, 1] = np.mean(features_class2[:, 1])
    class_mean[2, 0] = np.mean(features_class3[:, 0])
    class_mean[2, 1] = np.mean(features_class3[:, 1])
    return class_mean


# Function definition to predict the class of based on the input feature
def class_predictor(class_mean, ip_f1, ip_f2):
    c1_norm = math.sqrt(math.pow(class_mean[0, 0] - ip_f1, 2) + math.pow(class_mean[0, 1] - ip_f2, 2))
    c2_norm = math.sqrt(math.pow(class_mean[1, 0] - ip_f1, 2) + math.pow(class_mean[1, 1] - ip_f2, 2))
    c3_norm = math.sqrt(math.pow(class_mean[2, 0] - ip_f1, 2) + math.pow(class_mean[2, 1] - ip_f2, 2))
    class_predicted = 1 + np.argmin([c1_norm, c2_norm, c3_norm])
    return class_predicted


# Function definition to read the test.csv file and return the error rate by comparing the predicted
# and the expected class type
def verify_test_data(fname, class_mean):
    data = np.genfromtxt(fname, delimiter=",")
    rows, cols = np.shape(data)
    class_prediction = np.zeros([rows])
    for count in range(0, rows):
        class_prediction[count] = class_predictor(class_mean, data[count, 0], data[count, 1])
    error = sum(class_prediction != data[:, 13]) / rows
    return error


# Main Function definition to read the test.csv and train.csv files and calculate the error rates
def classifier(fname1, fname2):
    data, features_class1, features_class2, features_class3, class_count, feature_count = extract_data(fname1)
    rows, cols = np.shape(data)
    class_prediction = np.zeros([rows])
    class_mean = class_means_calc(class_count, features_class1, features_class2, features_class3)
    for count in range(0, rows):
        class_prediction[count] = class_predictor(class_mean, data[count, 0], data[count, 1])
    error1 = sum(class_prediction != data[:, 13]) / rows
    print("Filename : ", fname1)
    print("Error rate : ", error1)
    plotDecBoundaries(data[:, [0, 1]], data[:, 13], class_mean)
    error2 = verify_test_data(fname2, class_mean)
    print("Filename : ", fname2)
    print("Error rate : ", error2)
    return


classifier('wine_train.csv', 'wine_test.csv')


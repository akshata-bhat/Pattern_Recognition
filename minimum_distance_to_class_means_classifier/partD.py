import numpy as np
import math
from plotDecBoundaries import plotDecBoundaries


# Function definition to calculate the class mean for each feature of each class and store it in a 2D array
def class_means_calc(class_count, feature1, feature2, class_type):
    class_mean = np.zeros(shape=[class_count, 2])
    class_mean[0, 0] = np.mean(feature1[np.nonzero(class_type == 1)])
    class_mean[0, 1] = np.mean(feature2[np.nonzero(class_type == 1)])
    class_mean[1, 0] = np.mean(feature1[np.nonzero(class_type == 2)])
    class_mean[1, 1] = np.mean(feature2[np.nonzero(class_type == 2)])
    class_mean[2, 0] = np.mean(feature1[np.nonzero(class_type == 3)])
    class_mean[2, 1] = np.mean(feature2[np.nonzero(class_type == 3)])
    return class_mean


# Function definition to predict the class of based on the input feature
def class_predictor(class_mean, ip_f1, ip_f2):
    c1_norm = math.sqrt(math.pow(class_mean[0, 0] - ip_f1, 2) + math.pow(class_mean[0, 1] - ip_f2, 2))
    print(c1_norm)
    c2_norm = math.sqrt(math.pow(class_mean[1, 0] - ip_f1, 2) + math.pow(class_mean[1, 1] - ip_f2, 2))
    c3_norm = math.sqrt(math.pow(class_mean[2, 0] - ip_f1, 2) + math.pow(class_mean[2, 1] - ip_f2, 2))
    class_predicted = 1 + np.argmin([c1_norm, c2_norm, c3_norm])
    return class_predicted


# Function definition to read the test.csv file and return the error rate by comparing the predicted
# and the expected class type
def verify_test_data(fname, class_mean, f1_index, f2_index):
    data = np.genfromtxt(fname, delimiter=",")
    rows, cols = np.shape(data)
    class_prediction = np.zeros([rows])
    for count in range(0, rows):
        class_prediction[count] = class_predictor(class_mean, data[count, f1_index], data[count, f2_index])
    error = sum(class_prediction != data[:, 13]) / rows
    return error


# Function definition to find the best two features out of 13 features which give the minimum error rate
def minimum_error_calculator(fname):
    data = np.genfromtxt(fname, delimiter=",")
    rows, cols = np.shape(data)
    class_count = int(np.max(data[:, cols - 1]))
    feature_index = np.zeros(shape=[78, 2])
    error_rates = np.zeros([78])

    # trial_no is the number of times=78 is the number of combinations
    trial_no = 0
    for ftr1 in range(0, cols - 1):
        for ftr2 in range(ftr1 + 1, cols - 1):
            feature_index[trial_no] = [ftr1, ftr2]
            predicted_class = np.zeros([rows])
            class_mean = class_means_calc(class_count, data[:, ftr1], data[:, ftr2], data[:, cols - 1])
            for count in range(0, 5):
                predicted_class[count] = class_predictor(class_mean, data[count, ftr1], data[count, ftr2])
            error_rates[trial_no] = float(sum(predicted_class != data[:, 13])) / float(rows)
            trial_no += 1
    min_err_index = np.argmin(error_rates)
    print("Minimum error rate for wine_train.csv : ", error_rates[min_err_index])
    print("Minimum error rate combination : ")
    best_f1 = int(feature_index[min_err_index, 0])
    best_f2 = int(feature_index[min_err_index, 1])
    print("Feature ", (best_f1 + 1))
    print("Feature ", (best_f2 + 1))
    class_mean_final = class_means_calc(class_count,  data[:, best_f1], data[:, best_f2], data[:, cols-1])
    error2 = verify_test_data('wine_test.csv', class_mean_final, best_f1, best_f2)
    print("Filename : wine_test.csv")
    print("Error rate : ", error2)
    plotDecBoundaries(data[:, [best_f1, best_f2]], data[:, cols-1], class_mean_final)


    return
# Function call to print the index of the best two features and the corresponding class_mean
minimum_error_calculator('wine_train.csv')



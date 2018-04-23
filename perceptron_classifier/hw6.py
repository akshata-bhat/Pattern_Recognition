import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def plotDecBoundaries(training, label_train, weights):

    #Plot the decision boundaries and data points for minimum distance to
    #class mean classifier
    #
    # training: traning data
    # label_train: class lables correspond to training data
    # weights: Optimal weights obtained from the minimum criterion funct    #
    # Total number of classes
    nclass =  max(np.unique(label_train))

    # Set the feature range for ploting
    max_x = np.ceil(max(training[:, 1])) + 1
    min_x = np.floor(min(training[:, 1])) - 1
    max_y = np.ceil(max(training[:, 2])) + 1
    min_y = np.floor(min(training[:, 2])) - 1

    xrange = (min_x, max_x)
    yrange = (min_y, max_y)

    # step size for how finely you want to visualize the decision boundary.
    inc = 0.005

    # generate grid coordinates. this will be the basis of the decision
    # boundary visualization.
    (x, y) = np.meshgrid(np.arange(xrange[0], xrange[1]+inc/100, inc), np.arange(yrange[0], yrange[1]+inc/100, inc))

    # size of the (x, y) image, which will also be the size of the
    # decision boundary image that is used as the plot background.
    image_size = x.shape
    xy = np.hstack( (x.reshape(x.shape[0]*x.shape[1], 1, order='F'), y.reshape(y.shape[0]*y.shape[1], 1, order='F')) ) # make (x,y) pairs as a bunch of row vectors.

    #To augment
    numRows = xy.shape[0]
    allOnes = np.ones((numRows,1), np.float64)
    augmentedRows = np.hstack([allOnes, xy])

    #Calculating discriminant function
    pred_label = np.dot(augmentedRows, weights)
    pred_label[pred_label > 0] = 1.0
    pred_label[pred_label <= 0] = 2.0

    # reshape the idx (which contains the class label) into an image.
    decisionmap = pred_label.reshape(image_size, order='F')

    #show the image, give each coordinate a color according to its class label
    plt.imshow(decisionmap, extent=[xrange[0], xrange[1], yrange[0], yrange[1]], origin='lower')

    # plot the class training data.
    plt.plot(training[label_train == 1, 1],training[label_train == 1, 2], 'rx')
    plt.plot(training[label_train == 2, 1],training[label_train == 2, 2], 'go')
    if nclass == 3:
        plt.plot(training[label_train == 3, 1], training[label_train == 3, 1], 'b*')

    # include legend for training data
    if nclass == 3:
        l = plt.legend(('Class 1', 'Class 2', 'Class 3'), loc=2)
    else:
        l = plt.legend(('Class 1', 'Class 2'), loc=2)
    plt.gca().add_artist(l)

    plt.tight_layout()
    plt.show()


def extract_data_synthetic12(train_file, test_file):
    input = np.genfromtxt(train_file, delimiter=",")
    rows, cols = np.shape(input)
    data_count = rows
    feature_count = cols - 1
    class_count = int(np.max(input[:, cols - 1]))
    class_col = input[:, 2]
    features_class1 = input[np.nonzero(class_col == 1)]
    f1_count = np.size(features_class1, 0)
    features_class2 = input[np.nonzero(class_col == 2)]
    f2_count = np.shape(features_class2)
    data = np.zeros(shape=[data_count, feature_count + 2])
    data[0:f1_count, 0] = 1
    data[f1_count:data_count, 0] = -1
    data[0:f1_count, 1:3] = features_class1[:, 0:2]
    data[f1_count:data_count, 1:3] = np.negative(features_class2[:, 0:2])
    data[:, 3] = class_col
    np.random.shuffle(data)
    #for count in range(0, data_count):
    #print(data[2])
    augmented_orig_data = np.zeros(shape=[data_count, feature_count + 1])
    augmented_orig_data[:, 0] = 1
    augmented_orig_data[:, 1:3] = input[:, 0:2]

    test_data = np.genfromtxt(test_file, delimiter=",")
    test_data_count, cols = np.shape(test_data)
    augmented_orig_test_data = np.zeros(shape=[test_data_count, feature_count + 1])
    augmented_orig_test_data[:, 0] = 1
    augmented_orig_test_data[:, 1:3] = test_data[:, 0:2]
    return data, class_count, feature_count, data_count, augmented_orig_data, input[:, 2], augmented_orig_test_data, test_data[:, 2], test_data_count


def extract_data_synthetic3(features_train_file, labels_train_file, features_test_file, labels_test_file):
    #input = np.genfromtxt(features_train_file, delimiter=",")
    orig_features = np.genfromtxt(features_train_file, delimiter=",")
    orig_labels = np.genfromtxt(labels_train_file, delimiter=",")
    rows, cols = np.shape(orig_features)
    data_count = rows # number of data points
    feature_count = cols # number of features
    class_count = np.size(orig_labels, 0)
    features_class1 = orig_features[np.nonzero(orig_labels == 1)]
    features_class2 = orig_features[np.nonzero(orig_labels == 2)]
    data = np.zeros(shape=[class_count, feature_count+2])  # data - matrix to store train data and labels
    mid = int(data_count/2)
    data[0:mid, 0] = 1
    data[mid:data_count, 0] = -1
    data[0:mid, 1:3] = features_class1
    data[mid:data_count, 1:3] = np.negative(features_class2)
    data[:, 3] = orig_labels
    np.random.shuffle(data) # To make the data points available as random

    #Extracting the data and labels for prediction
    augmented_orig_train_data = np.zeros(shape=[data_count, feature_count + 1])
    augmented_orig_train_data[:, 0] = 1
    augmented_orig_train_data[:, 1:3] = orig_features

    test_features = np.genfromtxt(features_test_file, delimiter=",")
    test_data_count, cols = np.shape(test_features)
    test_labels = np.genfromtxt(labels_test_file, delimiter=",")
    augmented_orig_test_data = np.zeros(shape=[test_data_count, feature_count + 1])
    augmented_orig_test_data[:, 0] = 1
    augmented_orig_test_data[:, 1:3] = test_features
    return data, class_count, feature_count, data_count, augmented_orig_train_data, orig_labels, augmented_orig_test_data, test_labels, test_data_count

def optimal_weights_calculator(data, feature_count, data_count):
    eta = 1
    weights = np.empty(shape=(data_count+1, feature_count+1))
    weights.fill(0.1)
    epoch=1000
    update=0
    misclassified_points=0
    convergence_status=False
    data_misclassified = np.array([0, 0, 0])
    #weights[data_count+1] = [0.1, 0.1, 0.1]
    for cycles in range(epoch):
        update=0
        for count in range(0, data_count):
            activation = np.dot(weights[count], data[count, 0:3])
            if activation > 0:
                weights[count+1] = weights[count]
            else:
                weights[count+1] = weights[count] + eta*data[count, 0:3]
                misclassified_points = misclassified_points+1
            if(np.array_equal(weights[count+1], weights[count])):
                update = update+1
        if(update == data_count):
            #print("hello", cycles)
            convergence_status = True
            break
        weights[0] = weights[count]
    print("Convergence status: ", convergence_status)
    print("Number of epochs: ", cycles)
    return weights, convergence_status, misclassified_points

def calculate_min_criterion(misclassified_points, data,feature_count):
    data_misclassified = np.zeros(shape=[misclassified_points, feature_count + 1])
    for row in range(0, data_count):
        activation = np.dot(weights[row], data[row, 0:3])
        if activation <= 0:
            data_misclassified[row] = data[row, 0:3]
    criterion = np.zeros(shape=[data_count])
    if convergence_status == False:
        for row in range(0, data_count):
            criterion[row] = np.sum(np.dot(data_misclassified, weights[row]))
    index_min_criterion = np.argmin(np.negative(criterion))
    return criterion[index_min_criterion], weights[index_min_criterion]

def calculate_error_rate(data, labels, optimal_weights, data_count):
    discriminant_func = np.zeros(shape=[data_count])
    predicted_labels = np.zeros(shape=[data_count])
    for row in range(0, data_count):
        discriminant_func = np.dot(optimal_weights, data[row])
        if discriminant_func <= 0:
            predicted_labels[row] = 2
        else:
            predicted_labels[row] = 1
    error = sum(predicted_labels != labels) / data_count
    return error


data, class_count, feature_count, data_count, orig_data, orig_labels, test_data, test_labels, test_data_count = extract_data_synthetic12("synthetic1_train.csv", "synthetic1_test.csv")
weights, convergence_status, misclassified_points = optimal_weights_calculator(data, feature_count, data_count)
min_criterion, optimal_weights = calculate_min_criterion(misclassified_points, data, feature_count)
error_rate = calculate_error_rate(orig_data, orig_labels, optimal_weights, data_count)
print("Synthetic1 Train dataset - Error rate: ", error_rate)
print("Minimum criterion", min_criterion)
print("Optimal weights", optimal_weights)
rows, cols = np.shape(test_data)
error_rate = calculate_error_rate(test_data, test_labels, optimal_weights, test_data_count)
print("Synthetic1 Test dataset - Error rate: ", error_rate)
plotDecBoundaries(orig_data, orig_labels, optimal_weights) #Training dataset
plotDecBoundaries(test_data, test_labels, optimal_weights) #Test dataset



data, class_count, feature_count, data_count, orig_data, orig_labels, test_data, test_labels, test_data_count = extract_data_synthetic12("synthetic2_train.csv", "synthetic2_test.csv")
weights, convergence_status, misclassified_points = optimal_weights_calculator(data, feature_count, data_count)
min_criterion, optimal_weights = calculate_min_criterion(misclassified_points, data, feature_count)
error_rate = calculate_error_rate(orig_data, orig_labels, optimal_weights, data_count)
print("Synthetic2 dataset - Error rate: ", error_rate)
print("Minimum criterion", min_criterion)
print("Optimal weights", optimal_weights)
rows, cols = np.shape(test_data)
error_rate = calculate_error_rate(test_data, test_labels, optimal_weights, rows)
print("Synthetic2 Test dataset - Error rate: ", error_rate)
plotDecBoundaries(orig_data, orig_labels, optimal_weights) #Training dataset
plotDecBoundaries(test_data, test_labels, optimal_weights) #Test dataset

data, class_count, feature_count, data_count, orig_data, orig_labels, test_data, test_labels, test_data_count = extract_data_synthetic3("feature_train.csv", "label_train.csv", "feature_test.csv", "label_test.csv")
weights, convergence_status, misclassified_points = optimal_weights_calculator(data, feature_count, data_count)
min_criterion, optimal_weights = calculate_min_criterion(misclassified_points, data, feature_count)
error_rate = calculate_error_rate(orig_data, orig_labels, optimal_weights, data_count)
print("Synthetic3 dataset - Error rate: ", error_rate)
print("Minimum criterion", min_criterion)
print("Optimal weights", optimal_weights)
rows, cols = np.shape(test_data)
error_rate = calculate_error_rate(test_data, test_labels, optimal_weights, rows)
print("Synthetic3 Test dataset - Error rate: ", error_rate)
plotDecBoundaries(orig_data, orig_labels, optimal_weights) #Training dataset
plotDecBoundaries(test_data, test_labels, optimal_weights) #Test dataset



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.optim as optim

import pickle



# Calculate dimensions for the nn.Linear layer
def dimensions_for_linear_layer(width, height):
    '''
        ===================================
        Takes in dimensions of the input numpy array .shape[2] and .shape[3]
            - calculates the dimentions for the nn.Linear input
            - returns a product of hight and width output sizes
        ===================================
    '''
    # add stride, padding and kernel as input parameters!!!!!!
    o_01 = (width - 5 + 0) / 1 + 1
    o_02 = o_01 // 2
    o_03 = (o_02 - 5 + 0) / 1 + 1
    output_shape_1 = o_03 // 2

    o_11 = (height - 5 + 0) / 1 + 1
    o_12 = o_11 // 2
    o_13 = (o_12 - 5 + 0) / 1 + 1
    output_shape_2  = o_13 // 2
    # print(output_shape_1, output_shape_2)
    return int(output_shape_1 * output_shape_2)


# Calculate predicred values on a test set, calculate accuracies
def test(CNN_model, X_test, y_test, classes):
    '''
        ===================================
        Runs the test sets through the model without changing the gradients.
        Returns: predicted, accuracies (the first value - accuracy of the entire model, others are for individual instruments), n_class_correct, n_class_samples
        ===================================
    '''

    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        num_classes = len(classes)
        n_class_correct = [0 for i in range(num_classes)]
        n_class_samples = [0 for i in range(num_classes)]


        outputs = CNN_model(X_test)
        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        n_samples += y_test.size(0)
        n_correct += (predicted == y_test).sum().item()

        #print('Predicted = ', predicted)

        for i in range(X_test.shape[0]):   # number of files in the test set
            label = y_test[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

        acc = 100.0 * n_correct / n_samples
        # print(f'Accuracy of the network: {acc} %')

        accuracies = []

        accuracies.append(acc)

        for i in range(num_classes):
            accuracies.append(100.0 * n_class_correct[i] / n_class_samples[i])
            # print(f'Accuracy of {classes[i]}: {acc} %')
            # print('n_class_correct = ', n_class_correct, ' n_class_samples = ', n_class_samples)

        return predicted, accuracies, n_class_correct, n_class_samples


# Creates a confusion matrix comparing test true labels with model predictions
def confusion_matrix(y_true, y_pred):
    '''
        ===================================
        Takes true and predicted values for labels (as tensors)
        Returns numpy array with a confusion matrix whose i-th row and j-th column entry indicates the number of samples with true label being i-th class and predicted label being j-th class.
        ===================================
    '''
    from sklearn.metrics import confusion_matrix
    import sklearn

    metric = sklearn.metrics.confusion_matrix(y_true, y_pred, labels = [0, 1, 2, 3])
    return metric


# Plots a confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes):
    '''
        ===================================
        Takes true and predicted values for labels (as tensors), tuple of classes
        plots (but doesn't show) the confusion matrix where Rows represent True labels and Columns represent - prebicted labels
        ===================================
    '''
    import numpy as np
    import matplotlib.pyplot as plt

    confusion = confusion_matrix(y_true, y_pred)

    column_headers = classes
    row_headers = classes

    cell_text = []
    for row in confusion:
        cell_text.append([f'{x}' for x in row])

    the_table = plt.table(cellText=cell_text,
                      rowLabels=row_headers,
                      rowLoc='left',
                      colLabels=column_headers, loc = 'center')

    the_table.scale(1.5, 2.5)
    the_table.set_fontsize(14)

    # Hide axes
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # Hide axes border
    plt.box(on=None)

    footer_text = '// Rows - true labels; Columns - predicted labels'
    plt.figtext(0.95, 0.05, footer_text, horizontalalignment='right', size=8, weight='light')

    # Add title
    plt.title('Confusion Matrix')


# Plots and saves the final image with losses on epochs, accuracies and a confusion matrix
def plot_image(training_loss, test_loss, num_epochs, learning_rate, classes, accuracies, y_true, y_predicted, destination_address, show = False):
    '''
        ===================================
        Plots accuracies for all epochs, table with final accuracies of the model and individual instruments, a confusion matrix where Rows represent True labels and Columns represent - prebicted labels.
        Imports all the necessary things.

        Takes as input:
            - train losses (list of floats)
            - test losses (list of floats)
            - number of epochs (int)
            - learning rate (float)
            - all classes (tuple)
            - all accuracies (list)
        ===================================
    '''
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import numpy as np

    accuracy = ('Accuracy', )
    row_Labels = accuracy + classes

    acc = []
    for item in accuracies:
        acc.append(str(round(item,1)) + ' %')

    columns = 1
    y_offset = np.zeros(columns)
    cell_text = []
    for row in range(len(acc)):
        y_offset = [acc[row]]
        cell_text.append([x for x in y_offset])

    fig = plt.figure(figsize=(12, 6), layout="constrained")
    fig.suptitle(f'Loss functions for {num_epochs} epochs, learning rate = {learning_rate}', fontsize = 24)

    gs = GridSpec(nrows=2, ncols=12)

    ax_graphs = fig.add_subplot(gs[:, 0:7])
    ax_graphs.plot(training_loss, 'g', label='Training Loss')
    ax_graphs.plot(test_loss, 'r', label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    ax_table = fig.add_subplot(gs[0, -3:-1])
    ax_table.set_axis_off()
    the_table = plt.table(cellText=cell_text,
                          colWidths = [0.5] * 1,
                          rowLabels=row_Labels, loc = 'center')

    for (row, col), cell in the_table.get_celld().items():
        if (row == 0):
            cell.set_text_props(weight='bold')

    the_table.scale(1.5, 2)
    the_table.set_fontsize(14)

    ax_table2 = fig.add_subplot(gs[-1, -3:-1])
    the_table_2 = plot_confusion_matrix(y_true, y_predicted, classes)

    # Saving and naming the image
    import time
    timestr = time.strftime("%Y%m%d-%H%M%S")
    name = 'lr_' + str(learning_rate) + '_epochs_' + str(num_epochs) + '_' + timestr
    filename = destination_address + name + '.png'
    plt.savefig(filename, dpi=400, bbox_inches='tight', pad_inches=0.1)

    if show == True:
        plt.show()
    plt.close('all')


# Calculate predicred values on a test set, calculate accuracies
def test(CNN_model, X_test, y_test, classes):
    '''
        ===================================
        Runs the test sets through the model without changing the gradients.
        Returns: predicted, accuracies (the first value - accuracy of the entire model, others are for individual instruments), n_class_correct, n_class_samples
        ===================================
    '''

    import torch

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
def confusion_matrix(y_true, y_pred, num_classes):
    '''
        ===================================
        Takes true and predicted values for labels (as tensors)
        Returns numpy array with a confusion matrix whose i-th row and j-th column entry indicates the number of samples with true label being i-th class and predicted label being j-th class.
        ===================================
    '''
    from sklearn.metrics import confusion_matrix
    import sklearn

    labels = []

    for i in range(num_classes):
        labels.append(i)

    metric = sklearn.metrics.confusion_matrix(y_true.cpu(), y_pred.cpu(), labels=labels)
    return metric

# Plots a confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes, width_scale, height_scale):
    '''
        ===================================
        Takes true and predicted values for labels (as tensors), tuple of classes
        plots (but doesn't show) the confusion matrix where Rows represent True labels and Columns represent - prebicted labels
        ===================================
    '''
    import numpy as np
    import matplotlib.pyplot as plt

    num_classes = len(classes)

    confusion = confusion_matrix(y_true, y_pred, num_classes)

    column_headers = classes
    row_headers = classes

    cell_text = []
    for row in confusion:
        cell_text.append([f'{x}' for x in row])

    the_table = plt.table(cellText=cell_text,
                      rowLabels=row_headers,
                      rowLoc='right',
                      colLabels=column_headers, loc = 'center')

    the_table.scale(width_scale, height_scale)
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(8)

    # Hide axes
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # Hide axes border
    plt.box(on=None)

    footer_text = '// Rows - true labels; Columns - predicted labels'
    plt.figtext(0.95, 0.05, footer_text, horizontalalignment='right', fontsize=8, weight='light')

    # Add title
    plt.title('Confusion Matrix')




def do_classification_report(y_true, y_pred, classes):
    import sklearn.metrics

    column_headers = ['Precision', 'Recall', 'f1 Score', 'Support']
    row_headers = list(classes) + ['Accuracy', 'Macro avg', 'Weighted avg']

    classification_report = sklearn.metrics.classification_report(y_true.cpu(), y_pred.cpu())
    lines = classification_report.split('\n')

    # removing empty lines from the lisi of lines
    for i in range(len(lines)):
        if len(lines) >= i:
            if len(lines[i]) == 0:
                lines.pop(i)

    # printing lines
    #for i in range(len(lines)):
    #    print (lines[i])

    #===================================
    # Creating table style (list) from the classification report (str)
    table_classes = lines[1:-3]
    table_summary = lines[-3:]
    table = []

    for line in table_classes:
        t = line.strip().split()
        table.append(t[1:])

    l1 = ['', ''] + table_summary[0].strip().split()[1:]
    table.append(l1)

    for i in range(2):
        table.append(table_summary[i+1].strip().split()[2:])

    #===================================

    return table, column_headers, row_headers



# Plots a confusion matrix
def plot_classification_report(y_true, y_pred, classes, width_scale, height_scale):
    '''
        ===================================
        Takes true and predicted values for labels (as tensors), tuple of classes
        plots (but doesn't show) the confusion matrix where Rows represent True labels and Columns represent - prebicted labels
        ===================================
    '''
    import numpy as np
    import matplotlib.pyplot as plt

    num_classes = len(classes)
    confusion_table, column_headers, row_headers = do_classification_report(y_true, y_pred, classes)

    cell_text = []

    for row in confusion_table:
        cell_text.append([f'{x}' for x in row])

    the_table = plt.table(cellText=cell_text, rowLabels=row_headers,
                      rowLoc='right', colLabels=column_headers, loc = 'center')

    the_table.scale(width_scale, height_scale)
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(8)

    # Hide axes
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # Hide axes border
    plt.box(on=None)

    # Add title
    plt.title('Classification Report')


# Plots and saves the final image with losses on epochs, accuracies and a confusion matrix
def plot_image(training_loss, test_loss, num_epochs, learning_rate, classes, accuracies, y_true, y_predicted, filename, show = False):
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

    #===============================

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

    #===============================

    fig = plt.figure(figsize=(18, 9), layout="constrained")  #(figsize=(18, 6), layout="constrained")
    fig.suptitle(f'Loss functions for {num_epochs} epochs, learning rate = {str(learning_rate)}', fontsize = 24)

    #===============================

    gs = GridSpec(nrows=2, ncols=4)

    #===============================
    # Graph

    ax_graphs = fig.add_subplot(gs[:, 0:2])
    ax_graphs.plot(training_loss, 'g', label='Training Loss')
    ax_graphs.plot(test_loss, 'r', label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    #===============================
    # Table with accuracies

    ax_table = fig.add_subplot(gs[0, -2])  #gs[0, -2]
    ax_table.set_axis_off()
    the_table = plt.table(cellText=cell_text,
                          colWidths = [0.5] * 1,
                          rowLabels=row_Labels, loc = 'center')

    for (row, col), cell in the_table.get_celld().items():
        if (row == 0):
            cell.set_text_props(weight='bold')

    width_scale_acc, height_scale_acc = 0.8, 1.4   # 0.8, 2
    the_table.scale(width_scale_acc, height_scale_acc)
    the_table.set_fontsize(10) #12

    #===============================
    # Confunsion matrix
    width_scale_conf, height_scale_conf = 2.1, 1.3 #1.5, 1.7
    ax_table2 = fig.add_subplot(gs[1, -1])
    the_table_2 = plot_confusion_matrix(y_true, y_predicted, classes, width_scale_conf, height_scale_conf)


    #===============================
    # Classification report

    width_scale_reprt, height_scale_reprt = 1.2, 1.1 # 1.2, 1.3
    ax_table3 = fig.add_subplot(gs[0, -1])
    the_table3 = plot_classification_report(y_true, y_predicted, classes, width_scale_reprt, height_scale_reprt)


    #===============================
    # Saving and naming the image

    filename = filename[:-3]  + '.png'
    plt.savefig(filename, dpi=400, bbox_inches='tight', pad_inches=0.1)

    if show == True:
        plt.show()
    plt.close('all')


import utils
import torch



def do_classification_report(y_true, y_pred, classes):
    import sklearn.metrics

    column_headers = ['Precision', 'Recall', 'f1 Score', 'Support']
    row_headers = classes + ['Accuracy', 'Macro avg', 'Weighted avg']

    classification_report = sklearn.metrics.classification_report(y_true, y_pred)
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
def plot_classification_report(y_true, y_pred, classes):
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

    the_table.scale(0.7, 1.4)
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(8)

    # Hide axes
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # Hide axes border
    plt.box(on=None)

    # Add title
    plt.title('Confusion Matrix')
    plt.show()






y_pred = torch.tensor([3, 1, 2, 1, 1, 1, 3, 1, 1, 3, 1, 3, 3, 1, 1, 3, 3, 3, 2, 1, 3, 1, 2, 1,
        1, 1, 1, 1, 3, 1, 1, 2, 2, 1, 1, 3, 1, 1, 1, 1, 1, 3, 1, 1, 1, 3, 1, 2,
        1, 1, 1, 3, 2, 2, 1, 1, 3, 2, 1, 1, 2, 0, 3, 1, 1, 3, 2, 2, 1, 0, 1, 2,
        3, 1])
y_test = torch.tensor([0, 1, 2, 1, 3, 1, 3, 1, 0, 3, 1, 3, 3, 1, 1, 3, 1, 3, 2, 1, 0, 1, 2, 1,
        1, 1, 2, 1, 3, 1, 1, 2, 0, 1, 1, 2, 1, 1, 3, 1, 1, 3, 1, 1, 1, 3, 1, 2,
        1, 1, 1, 2, 1, 2, 3, 1, 3, 2, 1, 1, 2, 0, 3, 1, 0, 3, 2, 2, 1, 0, 2, 2,
        3, 1])

classes = ['a', 'b', 'c', 'd']

do_classification_report(y_test, y_pred, classes)
plot_classification_report(y_test, y_pred, classes)

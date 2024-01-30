import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import pickle
import os
import sys
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_dir)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
import utils
from cnn import CNN

folder_address = '/Users/cookie/dev/instrument_classifier/unit_testing/'
file1 = 'string_acoustic_075-090-100.wav'
file2 = 'G53-71607-1111-229.wav'






y_pred = torch.tensor([3, 1, 2, 1, 1, 1, 3, 1, 1, 3, 1, 3, 3, 1, 1, 3, 3, 3, 2, 1, 3, 1, 2, 1,
        1, 1, 1, 1, 3, 1, 1, 2, 2, 1, 1, 3, 1, 1, 1, 1, 1, 3, 1, 1, 1, 3, 1, 2,
        1, 1, 1, 3, 2, 2, 1, 1, 3, 2, 1, 1, 2, 0, 3, 1, 1, 3, 2, 2, 1, 0, 1, 2,
        3, 1])
y_test = torch.tensor([0, 1, 2, 1, 3, 1, 3, 1, 0, 3, 1, 3, 3, 1, 1, 3, 1, 3, 2, 1, 0, 1, 2, 1,
        1, 1, 2, 1, 3, 1, 1, 2, 0, 1, 1, 2, 1, 1, 3, 1, 1, 3, 1, 1, 1, 3, 1, 2,
        1, 1, 1, 2, 1, 2, 3, 1, 3, 2, 1, 1, 2, 0, 3, 1, 0, 3, 2, 2, 1, 0, 2, 2,
        3, 1])



def plot_classification_report(y_true, y_pred):
    '''
        ===================================
        Takes true and predicted values for labels (as tensors), tuple of classes
        plots (but doesn't show) the classification report
        ===================================
    '''
    import numpy as np
    import matplotlib.pyplot as plt
    import sklearn
    import sklearn.metrics

    # confusion = confusion_matrix(y_true, y_pred, num_classes)
    report = sklearn.metrics.classification_report(y_true.cpu(), y_pred.cpu())


    cell_text = []
    for row in report:
        cell_text.append([f'{x}' for x in row])

    the_table = plt.table(cellText = cell_text, rowLoc='left', loc = 'center')

    the_table.scale(2.3, 2)
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(8)

    # Hide axes
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # Hide axes border
    plt.box(on=None)

    #plt.figtext(0.95, 0.05, footer_text, horizontalalignment='right', fontsize=8, weight='light')

    # Add title
    plt.title('Classification report')

    plt.show()
    print(report)






plot_classification_report(y_test, y_pred)




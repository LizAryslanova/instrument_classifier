
num_epochs = 18
learning_rate = 2.2
y1 = [1,2,3,4,12,15]
y2 = [1,4,9,16,1,3]

classes = ('Guitar', 'Piano', 'Drum', 'Violin')
accuracies = [87, 12, 43, 55, 66]





def plot_image(training_loss, test_loss, num_epochs, learning_rate, classes, accuracies):
    '''
    Plots accuracies for all epochs. Imports all the necessary things.
    Takes as input:
         - train losses (list of floats)
         - test losses (list of floats)
         - number of epochs (int)
         - learning rate (float)
         - all classes (tuple)
         - all accuracies (list)
    '''
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import numpy as np

    accuracy = ('Accuracy', )
    row_Labels = accuracy + classes

    acc = []

    for item in accuracies:
        acc.append(str(item) + ' %')

    columns = 1

    y_offset = np.zeros(columns)
    cell_text = []
    for row in range(len(acc)):
        y_offset = [acc[row]]
        cell_text.append([x for x in y_offset])

    fig = plt.figure(figsize=(10, 5))
    fig.suptitle(f'Loss functions for {num_epochs} epochs, learning rate = {learning_rate}', fontsize = 24)


    gs = GridSpec(nrows=1, ncols=6)

    ax_graphs = fig.add_subplot(gs[0, 0:4])
    ax_graphs.plot(training_loss, 'g', label='Training Loss')
    ax_graphs.plot(test_loss, 'r', label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    ax_table = fig.add_subplot(gs[0, 5])
    ax_table.set_axis_off()
    the_table = plt.table(cellText=cell_text,
                          colWidths = [0.5] * 1,
                          rowLabels=row_Labels,
                          loc='center')


    for (row, col), cell in the_table.get_celld().items():
        if (row == 0):
            cell.set_text_props(weight='bold')

    the_table.scale(1.5, 2.5)
    the_table.set_fontsize(16)

    plt.show()



plot_image(y1, y2, num_epochs, learning_rate, classes, accuracies)

num_epochs = 18
learning_rate = 2.2
y_true = [1,0,0,0,0,0,1,0,0]
y_predicted = [1,0,2,0,3,2,1,0,1]

y_1 = [3,13,23,4,2,1,34,33]
y_2 = [10,3,43,41,2,10,4,3]

classes = ('Guitar', 'Piano', 'Drum', 'Violin')
accuracies = [87, 12, 43, 55, 66]



import utils

utils.plot_image(y1, y2, num_epochs, learning_rate, classes, accuracies, y1, y2)
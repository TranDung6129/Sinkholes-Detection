"""
support_vector_machine.py
creates and tests a support vector machine
"""

# Import needed libraries.
import matplotlib.pyplot as plotlib
from sklearn import svm
from sklearn.datasets import make_blobs

# Define parameters.
number_of_data_points = 60
number_of_classes = 2
first_class_numeric_value = 0
second_class_numeric_value = 1
random_state = 6
kernel_type = 'linear'
regularization_parameter = 1000
data_points_scatter_plot_array_shape = number_of_data_points/2
support_vectors_scatter_plot_array_shape = 100
scatter_plot_color_map = plotlib.cm.Paired
support_vector_data_point_outline_color = 'k'
support_vector_data_point_outline_width = 1
support_vector_data_point_outline_fill = 'none'
prediction_input_test = [[8, -10]]

# Create data points and associated classes.
data_points, classes = make_blobs(
    n_samples=number_of_data_points,
    centers=number_of_classes,
    random_state=random_state)

# Instantiate a support vector machine model.
model = svm.SVC(
    kernel=kernel_type,
    C=regularization_parameter)

# Train the model.
model.fit(data_points, classes)

# Predict a class.
result = model.predict(prediction_input_test)
print('prediction for data point ' + str(prediction_input_test) + ' :')
print(result)

# Create a scatter plot of data points.
plotlib.scatter(
    data_points[:, first_class_numeric_value],
    data_points[:, second_class_numeric_value],
    c=classes,
    s=data_points_scatter_plot_array_shape,
    cmap=scatter_plot_color_map)

# Plot the support vectors.
axes = plotlib.gca()
axes.scatter(
    model.support_vectors_[:, first_class_numeric_value],
    model.support_vectors_[:, second_class_numeric_value],
    s=support_vectors_scatter_plot_array_shape,
    linewidth=support_vector_data_point_outline_width,
    facecolors=support_vector_data_point_outline_fill,
    edgecolors=support_vector_data_point_outline_color)

# Display the plot.
plotlib.show()
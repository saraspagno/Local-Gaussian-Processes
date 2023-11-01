import os
import typing
from sklearn.gaussian_process.kernels import *
from sklearn.cluster import KMeans
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

# Set `EXTENDED_EVALUATION` to `True` in order to visualize your predictions.
EXTENDED_EVALUATION = False
EVALUATION_GRID_POINTS = 300  # Number of grid points used in extended evaluation

# Cost function constants
COST_W_UNDERPREDICT = 50.0
COST_W_NORMAL = 1.0

# HYPERPARAMETERS
NOISE_STD = 0.1
N_LOCAL_GPS = 3


class Model(object):

    def __init__(self):
        # Using LOCAL GPs with GaussianProcessRegressor
        self.local_GPS = []
        # define the k-means
        self.kmeans = KMeans()

    def make_predictions(self, test_x_2D: np.ndarray, test_x_AREA: np.ndarray) -> typing.Tuple[
        np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict the pollution concentration for a given set of city_areas.
        :param test_x_2D: city_areas as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :param test_x_AREA: city_area info for every sample in a form of a bool array (NUM_SAMPLES,)
        :return:
            Tuple of three 1d NumPy float arrays, each of shape (NUM_SAMPLES,),
            containing your predictions, the GP posterior mean, and the GP posterior stddev (in that order)
        """

        print("Started predicting.")
        clusters = self.kmeans.predict(test_x_2D)
        mean_predictions = []
        std_deviations = []
        for i, x in enumerate(test_x_2D):
            gp = self.local_GPS[clusters[i]]
            prediction, std = gp.predict(x.reshape(1, -1), return_std=True)
            mean_predictions.append(prediction[0])
            std_deviations.append(std[0])

        mean_predictions = np.array(mean_predictions)
        std_deviations = np.array(std_deviations)
        predictions = mean_predictions

        # Overestimate where the area is 1
        overestimate_factor = 4.4
        adjusted_predictions = np.copy(predictions)
        adjusted_predictions[test_x_AREA == 1] += overestimate_factor * std_deviations[test_x_AREA == 1]

        # Use the maximum of the adjusted predictions and 1 as the final predictions
        predictions = np.maximum(adjusted_predictions, 1)

        return predictions, mean_predictions, std_deviations

    def fitting_model(self, train_y: np.ndarray, train_x_2D: np.ndarray):
        """
        Fit your model on the given training data.
        :param train_x_2D: Training features as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :param train_y: Training pollution concentrations as a 1d NumPy float array of shape (NUM_SAMPLES,)
        """
        # Getting clusters division based on kmeans
        datasets = self.fit_kmeans_clusters(train_x_2D, train_y)

        # Running the fit on each dataset in separate threads
        with ThreadPoolExecutor() as executor:
            self.local_GPS = list(executor.map(self.fit_gp, datasets))

    def fit_gp(self, data):
        train_x_2D, train_y = data
        gp = GaussianProcessRegressor(kernel=RationalQuadratic(), alpha=NOISE_STD)
        gp.fit(train_x_2D, train_y)
        return gp

    def fit_kmeans_clusters(self, train_x_2D, train_y):
        self.kmeans = KMeans(n_clusters=N_LOCAL_GPS, random_state=0, n_init=10).fit(train_x_2D)
        cluster_assignments = self.kmeans.labels_
        datasets = []
        for i in range(N_LOCAL_GPS):
            cluster_x = train_x_2D[cluster_assignments == i]
            cluster_y = train_y[cluster_assignments == i]
            datasets.append((cluster_x, cluster_y))
        return datasets


def extract_city_area_information(train_x: np.ndarray, test_x: np.ndarray) -> typing.Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts the city_area information from the training and test features.
    :param train_x: Training features
    :param test_x: Test features
    :return: Tuple of (training features' 2D coordinates, training features' city_area information,
        test features' 2D coordinates, test features' city_area information)
    """

    # Extract the first two columns (2D part)
    train_x_2D = train_x[:, :2]
    test_x_2D = test_x[:, :2]

    # Extract the third column and convert it to boolean
    train_x_AREA = train_x[:, 2].astype(bool)
    test_x_AREA = test_x[:, 2].astype(bool)

    assert train_x_2D.shape[0] == train_x_AREA.shape[0] and test_x_2D.shape[0] == test_x_AREA.shape[0]
    assert train_x_2D.shape[1] == 2 and test_x_2D.shape[1] == 2
    assert train_x_AREA.ndim == 1 and test_x_AREA.ndim == 1

    return train_x_2D, train_x_AREA, test_x_2D, test_x_AREA


# You don't have to change this function
def cost_function(ground_truth: np.ndarray, predictions: np.ndarray, AREA_idxs: np.ndarray) -> float:
    """
    Calculates the cost of a set of predictions.

    :param ground_truth: Ground truth pollution levels as a 1d NumPy float array
    :param predictions: Predicted pollution levels as a 1d NumPy float array
    :param AREA_idxs: city_area info for every sample in a form of a bool array (NUM_SAMPLES,)
    :return: Total cost of all predictions as a single float
    """
    assert ground_truth.ndim == 1 and predictions.ndim == 1 and ground_truth.shape == predictions.shape

    # Unweighted cost
    cost = (ground_truth - predictions) ** 2
    weights = np.ones_like(cost) * COST_W_NORMAL

    # Case i): underprediction
    mask = (predictions < ground_truth) & [bool(AREA_idx) for AREA_idx in AREA_idxs]
    weights[mask] = COST_W_UNDERPREDICT

    # Weigh the cost and return the average
    return np.mean(cost * weights)


# You don't have to change this function
def is_in_circle(coor, circle_coor):
    """
    Checks if a coordinate is inside a circle.
    :param coor: 2D coordinate
    :param circle_coor: 3D coordinate of the circle center and its radius
    :return: True if the coordinate is inside the circle, False otherwise
    """
    return (coor[0] - circle_coor[0]) ** 2 + (coor[1] - circle_coor[1]) ** 2 < circle_coor[2] ** 2


# You don't have to change this function
def determine_city_area_idx(visualization_xs_2D):
    """
    Determines the city_area index for each coordinate in the visualization grid.
    :param visualization_xs_2D: 2D coordinates of the visualization grid
    :return: 1D array of city_area indexes
    """
    # Circles coordinates
    circles = np.array([[0.5488135, 0.71518937, 0.17167342],
                        [0.79915856, 0.46147936, 0.1567626],
                        [0.26455561, 0.77423369, 0.10298338],
                        [0.6976312, 0.06022547, 0.04015634],
                        [0.31542835, 0.36371077, 0.17985623],
                        [0.15896958, 0.11037514, 0.07244247],
                        [0.82099323, 0.09710128, 0.08136552],
                        [0.41426299, 0.0641475, 0.04442035],
                        [0.09394051, 0.5759465, 0.08729856],
                        [0.84640867, 0.69947928, 0.04568374],
                        [0.23789282, 0.934214, 0.04039037],
                        [0.82076712, 0.90884372, 0.07434012],
                        [0.09961493, 0.94530153, 0.04755969],
                        [0.88172021, 0.2724369, 0.04483477],
                        [0.9425836, 0.6339977, 0.04979664]])

    visualization_xs_AREA = np.zeros((visualization_xs_2D.shape[0],))

    for i, coor in enumerate(visualization_xs_2D):
        visualization_xs_AREA[i] = any([is_in_circle(coor, circ) for circ in circles])

    return visualization_xs_AREA


# You don't have to change this function
def perform_extended_evaluation(model: Model, output_dir: str = '/results'):
    """
    Visualizes the predictions of a fitted model.
    :param model: Fitted model to be visualized
    :param output_dir: Directory in which the visualizations will be stored
    """
    print('Performing extended evaluation')

    # Visualize on a uniform grid over the entire coordinate system
    grid_lat, grid_lon = np.meshgrid(
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
    )
    visualization_xs_2D = np.stack((grid_lon.flatten(), grid_lat.flatten()), axis=1)
    visualization_xs_AREA = determine_city_area_idx(visualization_xs_2D)

    # Obtain predictions, means, and stddevs over the entire map
    predictions, gp_mean, gp_stddev = model.make_predictions(visualization_xs_2D, visualization_xs_AREA)
    predictions = np.reshape(predictions, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))
    gp_mean = np.reshape(gp_mean, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))

    vmin, vmax = 0.0, 65.0

    # Plot the actual predictions
    fig, ax = plt.subplots()
    ax.set_title('Extended visualization of task 1')
    im = ax.imshow(predictions, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax)

    # Save figure to pdf
    figure_path = os.path.join(output_dir, 'extended_evaluation.pdf')
    fig.savefig(figure_path)
    print(f'Saved extended evaluation to {figure_path}')

    plt.show()


# you don't have to change this function
def main():
    # Load the training dateset and test features
    train_x = np.loadtxt('train_x.csv', delimiter=',', skiprows=1)
    train_y = np.loadtxt('train_y.csv', delimiter=',', skiprows=1)
    test_x = np.loadtxt('test_x.csv', delimiter=',', skiprows=1)

    # Extract the city_area information
    train_x_2D, train_x_AREA, test_x_2D, test_x_AREA = extract_city_area_information(train_x, test_x)
    # Fit the model
    print('Fitting model')
    model = Model()
    model.fitting_model(train_y, train_x_2D)

    # Predict on the test features
    print('Predicting on test features')
    predictions = model.make_predictions(test_x_2D, test_x_AREA)
    print(predictions)

    if EXTENDED_EVALUATION:
        perform_extended_evaluation(model, output_dir='.')


if __name__ == "__main__":
    main()

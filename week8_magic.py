import numpy as np
import statistics # try to avoid important scipy just for mode
import sklearn.datasets as dsets

# synthetic data for clustering example
blobs_coords, blobs_labels = dsets.make_blobs(
    n_samples=150, n_features=2, cluster_std=1.0, 
    centers=[[0.5, 1.5], [7.5, 3.2],[2.8, 7.1]],
    shuffle=True, random_state=101, return_centers=False
)

# iris dataset for clustering exercise
iris_raw = dsets.load_iris(as_frame=True)
iris_labeled = iris_raw.frame.sample(frac=1, random_state=101).reset_index(drop=True)
iris = iris_labeled.iloc[:, :4]
iris_truth = iris_labeled.iloc[:, 4].values

# check for incorrect classification in iris

def iris_map(labels):
    '''
    mapped the internal iris label to closely match supplied labels

    arguments:
      labels: 1-axis numpy array
    '''
    mapper = []  
    for i in range(3):
        j = statistics.mode(labels[iris_truth==i])
        mapper.append(j)

    mapped = np.array([mapper[x] for x in iris_truth])   
    return mapped


def iris_check(labels):
    '''
    check agreement between ground truth and supplied label,
    with permutation of label taken into account

    arguments:
      labels: 1-axis numpy array
    '''
    mapped = iris_map(labels)
    return mapped == labels


# wine dataset for random forest exercise
wine_raw = dsets.load_wine(as_frame=True)
wine_labeled = wine_raw.frame.sample(frac=1, random_state=101).reset_index(drop=True)

# noisy-sine data for demo
np.random.seed(101)
x_raw = np.linspace(0, 1, 101)
x_values = x_raw.reshape(101, 1)
y_values = np.sin(2 * np.pi * x_raw) + np.random.normal(0, 0.1, x_raw.shape)

# load the diabetes dataset
diabetes_raw = dsets.load_diabetes(as_frame=True, scaled=False)
diabetes = diabetes_raw.frame.iloc[:, 2:].sample(frac=1, random_state=101).reset_index(drop=True)

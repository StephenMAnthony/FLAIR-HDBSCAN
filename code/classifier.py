import numpy as np
import torch
import torchvision.transforms as transforms
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import HDBSCAN
from scipy.sparse import coo_array

import utilities

def normalize_spectra(spectra, append_intensity=True):
    intensity = np.sum(spectra, axis=0)
    normalized_spectra = np.divide(spectra, intensity)
    if append_intensity:
        normalized_spectra = np.concatenate((normalized_spectra, intensity[np.newaxis, :]), axis=0)

    return normalized_spectra


def extract_spectra(train_dataset, config, downsample=True, no_other=True):
    num_channels_aerial = config['num_channels_aerial']
    sat_aerial_ratio = config['sat_aerial_ratio']
    aerial_side = config['aerial_side']
    downsample_factor = config['downsample_factor']
    # Don't include pixels belonging to the "other" class
    n_classes = config['num_classes'] - 1

    # Determine the number of pixels in the aerial image for which corresponding satellite pixels are available.
    matched_aerial_pixels = aerial_side // sat_aerial_ratio * sat_aerial_ratio
    transform_aerial = transforms.CenterCrop(matched_aerial_pixels)

    aerial_list = []
    label_list = []
    for idx in range(len(train_dataset)):
        # Load the aerial data
        aerial = torch.Tensor.numpy(train_dataset[idx]['aerial'])

        # Load the labels
        labels = torch.Tensor.numpy(train_dataset[idx]['labels'])

        if downsample:
            # Crop to only full satellite pixels
            aerial_patch = transform_aerial(torch.as_tensor(aerial)).numpy()
            label_patch = transform_aerial(torch.as_tensor(labels)).numpy()

            # Downsample to a number HDBSCAN can handle
            aerial = aerial_patch[:, downsample_factor // 2::downsample_factor,
                                  downsample_factor // 2::downsample_factor]
            labels = label_patch[downsample_factor // 2::downsample_factor, downsample_factor // 2::downsample_factor]

        # Flatten spatial dimension
        aerial = np.reshape(aerial, (num_channels_aerial, -1))
        labels = labels.ravel()

        if no_other:
            # Remove pixels corresponding to the other class.
            keep = labels < n_classes
            labels = labels[keep]
            aerial = aerial[:, keep]

        aerial_list.append(aerial)
        label_list.append(labels)

    aerial_spectra = np.concatenate(aerial_list, axis=1)
    labels = np.concatenate(label_list)
    labels = labels.astype(int)

    return aerial_spectra, labels


def train_and_validate_knn(train_dataset, config, normalize=False, append_intensity=True):
    # Extract the downsampled data
    aerial_spectra, labels = extract_spectra(train_dataset, config, downsample=True)

    if normalize:
        aerial_spectra = normalize_spectra(aerial_spectra, append_intensity=append_intensity)

    # Fit on the downsampled data (training)
    neigh = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
    neigh.fit(aerial_spectra.T, labels)

    # Extract the full data (training and validation)
    aerial_spectra, labels = extract_spectra(train_dataset, config, downsample=False)

    if normalize:
        aerial_spectra = normalize_spectra(aerial_spectra, append_intensity=append_intensity)

    # Predict the full data
    predicted = neigh.predict(aerial_spectra.T)

    # Generate a confusion matrix for the knn prediction on the training and validation data
    knn_confusion = metrics.confusion_matrix(labels, predicted)
    row_norm_factors = np.sum(knn_confusion, axis=1)[:, np.newaxis]
    knn_confusion = knn_confusion / row_norm_factors

    return neigh, knn_confusion


def train_and_validate_hdbscan(train_dataset, config, normalize=False, append_intensity=True, min_cluster_size=10):
    # Extract the downsampled data
    aerial_spectra, actual_labels = extract_spectra(train_dataset, config, downsample=True)

    if normalize:
        aerial_spectra = normalize_spectra(aerial_spectra, append_intensity=append_intensity)

    # Perform HDBSCAN clustering
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size)
    hdbscan_clusters = clusterer.fit_predict(aerial_spectra.T)

    # For each cluster, assign the most common class label.
    output_dict = assign_class_to_cluster(hdbscan_clusters, actual_labels, aerial_spectra)

    # Display some helpful information
    n_pixels = actual_labels.size
    n_clusters = output_dict["best_class"].size
    mean_reliability = np.mean(output_dict['reliability'])
    weighted_mean_reliability = utilities.calc_weighted_mean(output_dict['reliability'],
                                                             output_dict['counts_per_cluster'])
    print(f"Originally there were {n_pixels} pixels.")
    print(f"HDBSCAN found {n_clusters} clusters.")
    print(f"Clusters were assigned to semantic classes with a mean reliability of {mean_reliability}, ")
    print(f" and a weighted mean reliability of {weighted_mean_reliability}")

    # Fit on the downsampled data (training) using the predicted labels.
    clustered_spectra = output_dict["spectra"]
    predicted_labels = output_dict["predicted_labels"]
    neigh = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
    neigh.fit(clustered_spectra.T, predicted_labels)

    # Extract the full data (training and validation)
    aerial_spectra, actual_labels = extract_spectra(train_dataset, config, downsample=False)

    if normalize:
        aerial_spectra = normalize_spectra(aerial_spectra, append_intensity=append_intensity)

    # Predict the full data
    predicted = neigh.predict(aerial_spectra.T)

    # Generate a confusion matrix for the knn prediction on the training and validation data
    knn_confusion = metrics.confusion_matrix(actual_labels, predicted)
    row_norm_factors = np.sum(knn_confusion, axis=1)[:, np.newaxis]
    confusion = knn_confusion / row_norm_factors

    return neigh, confusion

# output_dict = assign_class_to_cluster(cluster_labels, class_labels, spectra)

def assign_class_to_cluster(cluster_labels: np.ndarray, class_labels: np.ndarray, spectra: np.ndarray) -> dict:
    # Many points will not be assigned to clusters. Exclude them.
    clustered = cluster_labels >= 0
    cluster_labels = cluster_labels[clustered]
    class_labels = class_labels[clustered]
    spectra = spectra[:, clustered]

    # Determine pairings of cluster labels with class labels and how many times each pairing occurs.
    cluster_class_pairs, counts = np.unique(np.column_stack((cluster_labels, class_labels)), axis=0, return_counts=True)

    # Use a sparse array to convert the prior output into a convenient array, padding with zeros for any pairings of
    # cluster and class that do not exist.
    cluster_label_map = coo_array((counts, (cluster_class_pairs[:, 1], cluster_class_pairs[:, 0]))).toarray()

    # Determine the number of counts in each cluster and use that to determine how reliable each cluster assignment to
    # a class is likely to be.
    counts_per_cluster = np.sum(cluster_label_map, axis=0)
    reliability = np.divide(np.max(cluster_label_map, axis=0), counts_per_cluster)

    # Determine the best class for each cluster
    best_class = np.argmax(cluster_label_map, axis=0)

    # For each clustered spectrum, assign the class label which is most likely.
    cluster_classes = best_class[cluster_labels]

    output_dict = {
        "spectra": spectra,
        "counts_per_cluster": counts_per_cluster,
        "reliability": reliability,
        "best_class": best_class,
        "actual_labels": class_labels,
        "predicted_labels": cluster_classes,
    }

    return output_dict
import numpy as np
import torch
import pandas
import torchvision.transforms as transforms
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import HDBSCAN
from scipy.sparse import coo_array

import utilities


# data_dict = separate_labels_from_data(data_and_labels)

def separate_labels_from_data(data_and_labels: pandas.DataFrame):

    # Determine all column names and from that determine the spectral column names
    column_names = data_and_labels.columns.values.tolist()
    data_column_names = column_names.copy()
    data_column_names.remove('True_Class')
    spectral_column_names = data_column_names.copy()
    spectral_column_names.remove('Elevation')
    spectral_column_names.remove('Aerial Intensity')

    # Extract the labels, the elevation, and the spectra
    true_classes = data_and_labels[['True_Class']].to_numpy().ravel()
    elevation = data_and_labels[['Elevation']].to_numpy().ravel()
    intensity = data_and_labels[['Aerial Intensity']].to_numpy().ravel()
    spectra = data_and_labels[spectral_column_names].to_numpy()

    # Enforce integer labels
    true_classes = true_classes.astype(int)

    data_dict = {
        "true_classes": true_classes,
        "elevation": elevation,
        "spectra": spectra,
        "intensity": intensity,
        "spectral_column_names": spectral_column_names,
    }

    return data_dict


# aerial_data = def extract_aerial_spectra(dataset, config, downsample=True, no_other=True, scale_by_intensity=False)

def extract_aerial_spectra(dataset, config, downsample=True, no_other=True, scale_by_intensity=False):
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
    for idx in range(len(dataset)):
        # Load the aerial data
        aerial = torch.Tensor.numpy(dataset[idx]['aerial'])

        # Load the labels
        labels = torch.Tensor.numpy(dataset[idx]['labels'])

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

    # Extract the spectra and elevation
    aerial_spectra = np.concatenate(aerial_list, axis=1)
    elevation = aerial_spectra[-1, :]
    aerial_spectra = aerial_spectra[:-1, :]

    # Extract the labels and recast as integers
    labels = np.concatenate(label_list)
    labels = labels.astype(int)

    # Calculate the intensity
    intensity = np.sum(aerial_spectra, axis=0)

    if scale_by_intensity:
        aerial_spectra = np.divide(aerial_spectra, intensity[np.newaxis, :])

    # Combine and record
    combined = np.concatenate((labels[np.newaxis, :], aerial_spectra,
                               elevation[np.newaxis, :], intensity[np.newaxis, :]), axis=0)
    aerial_data = pandas.DataFrame(combined.T, columns=['True_Class', 'Blue', 'Green', 'Red', 'NIR',
                                                        'Elevation', 'Aerial Intensity'])

    return aerial_data


def train_and_validate_model(train_dataset, config, use_hdbscan=False, min_cluster_size=10,
                             scale_by_intensity=False, append_intensity=False) -> dict:

    def normalize_and_append(input_dict: dict):

        # Extract
        spectra = input_dict['spectra']
        elevation = input_dict['elevation']
        labels = input_dict['true_classes']
        intensity = input_dict['intensity']

        # If scaling by intensity, do so.
        if scale_by_intensity:
            spectra = np.divide(spectra, intensity[:, np.newaxis])

        # If appending intensity, do so.
        if append_intensity:
            selected_data = np.concatenate((spectra, elevation[:, np.newaxis], intensity[:, np.newaxis]), axis=1)
        else:
            selected_data = np.concatenate((spectra, elevation[:, np.newaxis]), axis=1)

        return selected_data, labels

    # Extract the downsampled data as a pandas dataframe, separate into data_to_fit and labels
    # applying the normalization and appending the intensity if specified
    data = extract_aerial_spectra(train_dataset, config)
    data_dict = separate_labels_from_data(data)
    data_to_fit, true_classes = normalize_and_append(data_dict)

    # Set up the k-nearest neighbor model
    model = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)

    if use_hdbscan:
        # Perform HDBSCAN clustering
        clusterer = HDBSCAN(min_cluster_size=min_cluster_size)
        hdbscan_clusters = clusterer.fit_predict(data_to_fit)

        # For each cluster, assign the most common class label.
        output_dict = assign_class_to_cluster(hdbscan_clusters, true_classes, data_to_fit)

        # Display some helpful information
        n_pixels = true_classes.size
        n_clusters = output_dict["best_class"].size
        n_cluster_pixels = output_dict["predicted_labels"].size
        mean_reliability = np.mean(output_dict['reliability'])
        weighted_mean_reliability = utilities.calc_weighted_mean(output_dict['reliability'],
                                                                 output_dict['counts_per_cluster'])

        print(f"Originally processing {n_pixels} pixels.")
        print(f"HDBSCAN assigned {n_cluster_pixels} pixels to {n_clusters} clusters.")
        print(f"Clusters were assigned to semantic classes with a mean reliability of {mean_reliability}, ")
        print(f" and a weighted mean reliability of {weighted_mean_reliability}")

        # Fit on the clustered samples from the training data using the predicted labels.
        model.fit(output_dict["clustered_samples"], output_dict["predicted_labels"])
    else:
        # Fit on the downsampled data (training)
        model.fit(data_to_fit, true_classes)

    # Extract the full data (training and validation) as a pandas dataframe,
    # applying the normalization and appending the intensity if specified
    data = extract_aerial_spectra(train_dataset, config, downsample=False)
    data_dict = separate_labels_from_data(data)
    data_to_fit, true_classes = normalize_and_append(data_dict)

    # Predict the full data
    predicted = model.predict(data_to_fit)

    model_and_predictions = {
        "model": model,
        "true_classes": true_classes,
        "predicted_classes": predicted,
    }

    return model_and_predictions


# output_dict = assign_class_to_cluster(cluster_labels, class_labels, spectra)

def assign_class_to_cluster(cluster_labels: np.ndarray, class_labels: np.ndarray, samples: np.ndarray) -> dict:
    # Many points will not be assigned to clusters. Exclude them.
    clustered = cluster_labels >= 0
    cluster_labels = cluster_labels[clustered]
    class_labels = class_labels[clustered]
    clustered_samples = samples[clustered, :]

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
        "clustered_samples": clustered_samples,
        "counts_per_cluster": counts_per_cluster,
        "reliability": reliability,
        "best_class": best_class,
        "actual_labels": class_labels,
        "predicted_labels": cluster_classes,
    }

    return output_dict

import numpy as np
import torch
import pandas
import torchvision.transforms as transforms
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import RobustScaler
from scipy.sparse import coo_array

import utilities

AERIAL_CHANNELS = ['Blue', 'Green', 'Red', 'NIR']
SATELLITE_CHANNELS = ['490', '560', '665', '705', '740', '783', '842', '865', '1610', '2190']


# data_dict = separate_labels_from_data(data_and_labels)

def separate_labels_from_data(data_and_labels: pandas.DataFrame):

    # Determine all column names and from that determine the spectral column names
    column_names = data_and_labels.columns.values.tolist()
    data_column_names = column_names.copy()
    data_column_names.remove('True_Class')

    # Extract the labels, the elevation, and the spectra
    true_classes = data_and_labels[['True_Class']].to_numpy().ravel()
    elevation = data_and_labels[['Elevation']].to_numpy().ravel()
    intensity = data_and_labels[['Aerial Intensity']].to_numpy().ravel()
    aerial_spectra = data_and_labels[AERIAL_CHANNELS].to_numpy()
    satellite_spectra = data_and_labels[SATELLITE_CHANNELS].to_numpy()

    # Enforce integer labels
    true_classes = true_classes.astype(int)

    data_dict = {
        "true_classes": true_classes,
        "elevation": elevation,
        "aerial_spectra": aerial_spectra,
        "intensity": intensity,
        "satellite_spectra": satellite_spectra,
    }

    return data_dict


# satellite_patch = rescale_satellite(satellite, config)

def rescale_satellite(satellite, config):
    sat_aerial_ratio = config['sat_aerial_ratio']
    aerial_side = config['aerial_side']

    # For the satellite images, convert to the average satellite image.
    satellite = np.squeeze(np.mean(satellite, axis=0))

    # Determine the number of satellite pixels required to completely include the aerial image
    required_sat_size = aerial_side // sat_aerial_ratio + 2

    # Perform an intermediate crop to avoid wasting space
    transform = transforms.CenterCrop(required_sat_size)
    satellite_patch = transform(torch.as_tensor(satellite)).numpy()

    # Replicate satellite pixels to aerial resolution, then crop to the aerial dimensions
    satellite_patch = np.repeat(np.repeat(satellite_patch, sat_aerial_ratio, axis=1), sat_aerial_ratio, axis=2)
    transform = transforms.CenterCrop(aerial_side)
    satellite_patch = transform(torch.as_tensor(satellite_patch)).numpy()

    return satellite_patch


# dataframe = def extract_spectra(dataset, config, downsample=True, no_other=True, scale_by_intensity=False)

def extract_spectra(dataset, config, downsample=True, no_other=True, scale_by_intensity=False):
    num_channels_aerial = config['num_channels_aerial']
    num_channels_sat = config['num_channels_sat']
    sat_aerial_ratio = config['sat_aerial_ratio']
    aerial_side = config['aerial_side']
    downsample_factor = config['downsample_factor']
    # Don't include pixels belonging to the "other" class
    n_classes = config['num_classes'] - 1

    # Determine the number of pixels in the aerial image for which corresponding satellite pixels are available.
    matched_aerial_pixels = aerial_side // sat_aerial_ratio * sat_aerial_ratio
    transform_full_pixels = transforms.CenterCrop(matched_aerial_pixels)

    aerial_list = []
    label_list = []
    satellite_list = []

    for idx in range(len(dataset)):
        # Load the aerial data, satellite data, and labels
        aerial = torch.Tensor.numpy(dataset[idx]['aerial'])
        labels = torch.Tensor.numpy(dataset[idx]['labels'])
        satellite = torch.Tensor.numpy(dataset[idx]['satellite'])

        aerial = aerial.astype(np.float16)
        satellite = satellite.astype(np.float16)
        labels = labels.astype(np.uint8)

        # Rescale the satellite image to the resolution of the aerial image, matching pixels.
        satellite = rescale_satellite(satellite, config)

        if downsample:
            # Crop to only full satellite pixels
            aerial_patch = transform_full_pixels(torch.as_tensor(aerial)).numpy()
            label_patch = transform_full_pixels(torch.as_tensor(labels)).numpy()
            satellite_patch = transform_full_pixels(torch.as_tensor(satellite)).numpy()

            # Downsample to a number HDBSCAN can handle
            aerial = aerial_patch[:, downsample_factor // 2::downsample_factor,
                                  downsample_factor // 2::downsample_factor]
            satellite = satellite_patch[:, downsample_factor // 2::downsample_factor,
                                        downsample_factor // 2::downsample_factor]
            labels = label_patch[downsample_factor // 2::downsample_factor,
                                 downsample_factor // 2::downsample_factor]

        # Flatten spatial dimension
        aerial = np.reshape(aerial, (num_channels_aerial, -1))
        satellite = np.reshape(satellite, (num_channels_sat, -1))
        labels = labels.ravel()

        if no_other:
            # Remove pixels corresponding to the other class.
            keep = labels < n_classes
            labels = labels[keep]
            aerial = aerial[:, keep]
            satellite = satellite[:, keep]

        aerial_list.append(aerial)
        satellite_list.append(satellite)
        label_list.append(labels)

    # Extract the spectra and elevation
    aerial_spectra = np.concatenate(aerial_list, axis=1)
    elevation = aerial_spectra[-1, :]
    aerial_spectra = aerial_spectra[:-1, :]

    # Extract the satellite spectra
    satellite_spectra = np.concatenate(satellite_list, axis=1)

    # Extract the labels and recast as integers
    labels = np.concatenate(label_list)
    labels = labels.astype(int)

    # Calculate the intensity of RGB
    intensity = np.sum(aerial_spectra[:3, :], axis=0)

    # Scale the satellite intensity to match the aerial intensity
    satellite_rgb_intensity = np.sum(satellite_spectra[:4, :], axis=0)
    satellite_spectra = np.divide(satellite_spectra, satellite_rgb_intensity[np.newaxis, :])

    if scale_by_intensity:
        aerial_spectra = np.divide(aerial_spectra, intensity[np.newaxis, :])
    else:
        satellite_spectra = np.multiply(satellite_spectra, intensity[np.newaxis, :])

    # Combine and record
    combined = np.concatenate((labels[np.newaxis, :], aerial_spectra, satellite_spectra,
                               elevation[np.newaxis, :], intensity[np.newaxis, :]), axis=0)
    column_names = ['True_Class'] + AERIAL_CHANNELS + SATELLITE_CHANNELS + ['Elevation', 'Aerial Intensity']
    dataframe = pandas.DataFrame(combined.T, columns=column_names)

    return dataframe


# data_to_fit, true_classes = load_data(train_dataset, config, downsample=True,
#   use_satellite=False, scale_by_intensity=False, append_intensity=False)

def load_data(the_dataset, config, downsample=True,
              use_satellite=False, scale_by_intensity=False, append_intensity=False):
    def normalize_and_append(input_dict: dict):

        # Extract
        aerial_spectra = input_dict['aerial_spectra'].astype(np.float16)
        satellite_spectra = input_dict['satellite_spectra'].astype(np.float16)
        elevation = input_dict['elevation'].astype(np.float16)
        labels = input_dict['true_classes'].astype(np.uint8)
        intensity = input_dict['intensity'].astype(np.float16)

        # If scaling by intensity, do so.
        if scale_by_intensity:
            aerial_spectra = np.divide(aerial_spectra, intensity[:, np.newaxis])
            satellite_spectra = np.divide(satellite_spectra, intensity[:, np.newaxis])

        # Determine whether the spectra should include the satellite channels
        if use_satellite:
            spectra = np.concatenate((aerial_spectra, satellite_spectra), axis=1)
        else:
            spectra = aerial_spectra

        # If appending intensity, do so.
        if append_intensity:
            selected_data = np.concatenate((spectra, elevation[:, np.newaxis], intensity[:, np.newaxis]), axis=1)
        else:
            selected_data = np.concatenate((spectra, elevation[:, np.newaxis]), axis=1)

        return selected_data, labels

    # Extract the downsampled data as a pandas dataframe, separate into data_to_fit and labels
    # applying the normalization and appending the intensity if specified
    dataframe = extract_spectra(the_dataset, config, downsample=downsample)
    data_dict = separate_labels_from_data(dataframe)
    data_to_fit, true_classes = normalize_and_append(data_dict)

    return data_to_fit, true_classes


def train_and_validate_model(train_dataset, config, use_hdbscan=False, min_cluster_size=10, use_satellite=False,
                             scale_by_intensity=False, append_intensity=False, robust_scale=False,
                             check_reliability=False, run_long=False) -> dict:

    if use_satellite and scale_by_intensity and append_intensity:
        print("*****************************Warning*****************************")
        print("This combination of settings can take a VERY LONG TIME!")
        print("Testing shows this combination may take 10 HOURS or more to run!")
        print("*****************************Warning*****************************")
        if not run_long:
            print("This combination of settings will only run if an additional parameter, run_long=True, is added.")
            model_and_predictions = {}
            return model_and_predictions

    # Load the data and the true classes
    data_to_fit, true_classes = load_data(train_dataset, config, use_satellite=use_satellite,
                                          scale_by_intensity=scale_by_intensity, append_intensity=append_intensity)

    if robust_scale:
        transformer = RobustScaler().fit(data_to_fit)
        data_to_fit = transformer.transform(data_to_fit)

    # Set up the k-nearest neighbor model
    model = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)

    if use_hdbscan:
        # Perform HDBSCAN clustering
        print("Performing HDBSCAN clustering")
        clusterer = HDBSCAN(min_cluster_size=min_cluster_size)
        hdbscan_clusters = clusterer.fit_predict(data_to_fit)

        # For each cluster, assign the most common class label.
        cluster_dict = assign_class_to_cluster(hdbscan_clusters, true_classes, data_to_fit)

        # Display some helpful information
        n_pixels = true_classes.size
        n_clusters = cluster_dict["best_class"].size
        n_cluster_pixels = cluster_dict["predicted_labels"].size
        mean_reliability = np.mean(cluster_dict['reliability'])
        weighted_mean_reliability = utilities.calc_weighted_mean(cluster_dict['reliability'],
                                                                 cluster_dict['counts_per_cluster'])

        print(f"Originally processing {n_pixels} pixels.")
        print(f"HDBSCAN assigned {n_cluster_pixels} pixels to {n_clusters} clusters.")
        print(f"Clusters were assigned to semantic classes with a mean reliability of {mean_reliability}, ")
        print(f" and a weighted mean reliability of {weighted_mean_reliability}")

        if check_reliability:
            model.fit(cluster_dict["clustered_samples"], cluster_dict["cluster_labels"])
            predicted_clusters_all_train = model.predict(data_to_fit)
            reliability_dict = assign_class_to_cluster(predicted_clusters_all_train, true_classes, data_to_fit)
            mean_reliability_all_train = np.mean(reliability_dict['reliability'])
            weighted_mean_reliability_all_train = utilities.calc_weighted_mean(reliability_dict['reliability'],
                                                                               reliability_dict['counts_per_cluster'])
            print(f"If KNN is used to expand the cluster labels to all training pixels,")
            print(f"the mean reliability is {mean_reliability_all_train},")
            print(f"and the weighted mean reliability is {weighted_mean_reliability_all_train}")

        print("Building KNN Classification Model")
        # Fit on the clustered samples from the training data using the predicted labels.
        model.fit(cluster_dict["clustered_samples"], cluster_dict["predicted_labels"])
    else:
        print("Fitting KNN classification model")
        # Fit on the downsampled data (training)
        model.fit(data_to_fit, true_classes)

    print('KNN classification model fit')

    if check_reliability:
        predicted_classes_all_train = model.predict(data_to_fit)
        reliability_all = np.count_nonzero(predicted_classes_all_train == true_classes) / true_classes.size
        print(f"If KNN is used to expand the autonomously generated class labels to all pixels,")
        print(f"the reliability is {reliability_all}")

    # Load the full data and the true classes
    data_to_fit, true_classes = load_data(train_dataset, config, downsample=False, use_satellite=use_satellite,
                                          scale_by_intensity=scale_by_intensity, append_intensity=append_intensity)

    if robust_scale:
        data_to_fit = transformer.transform(data_to_fit)

    # Predict the full data. Experience shows that this is very computationally expensive.
    # Additionally, there appear to sometimes be ?memory? errors on the full data.
    # Therefore, it was redesigned to operate on chunks.
    n_samples = data_to_fit.shape[0]
    n_blocks = np.ceil(n_samples/1e5).astype(int)
    print("Creating KNN predictions")
    data_to_fit_blocks = np.array_split(data_to_fit, n_blocks, axis=0)
    predictions = []
    last_fit = 0
    for data_block in data_to_fit_blocks:
        n_in_block = data_block.shape[0]
        print(f"Predicting pixels {last_fit + 1} to {last_fit + n_in_block} of {n_samples} total pixels.")
        predictions.append(model.predict(data_block).astype(np.uint8))
        last_fit += n_in_block

    predicted = np.concatenate(predictions)

    model_and_predictions = {
        "model": model,
        "true_classes": true_classes,
        "predicted_classes": predicted,
        "config": config,
        "use_hdbscan": use_hdbscan,
        "use_satellite": use_satellite,
        "scale_by_intensity": scale_by_intensity,
        "append_intensity": append_intensity,
        "robust_scale": robust_scale,
    }

    if robust_scale:
        model_and_predictions['transformer'] = transformer

    if use_hdbscan:
        model_and_predictions['cluster_dict'] = cluster_dict

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
        "cluster_labels": cluster_labels,
        "actual_labels": class_labels,
        "predicted_labels": cluster_classes,
    }

    return output_dict


def apply_model(test_dataset, model_and_predictions):
    # Unpack
    model = model_and_predictions["model"]
    config = model_and_predictions["config"]
    use_satellite = model_and_predictions["use_satellite"]
    scale_by_intensity = model_and_predictions["scale_by_intensity"]
    append_intensity = model_and_predictions["append_intensity"]
    robust_scale = model_and_predictions["robust_scale"]

    # Load the full data and the true classes
    data_to_fit, true_classes = load_data(test_dataset, config, downsample=False, use_satellite=use_satellite,
                                          scale_by_intensity=scale_by_intensity, append_intensity=append_intensity)

    if robust_scale:
        transformer = model_and_predictions["transformer"]
        data_to_fit = transformer.transform(data_to_fit)

    # Predict the full test data. Experience shows that this is may be computationally expensive.
    # Additionally, there appear to sometimes be ?memory? errors on the full data.
    # Therefore, it was redesigned to operate on chunks.
    n_samples = data_to_fit.shape[0]
    n_blocks = np.ceil(n_samples/1e5).astype(int)
    print("Creating KNN predictions")
    data_to_fit_blocks = np.array_split(data_to_fit, n_blocks, axis=0)
    predictions = []
    last_fit = 0
    for data_block in data_to_fit_blocks:
        n_in_block = data_block.shape[0]
        print(f"Predicting pixels {last_fit + 1} to {last_fit + n_in_block} of {n_samples} total pixels.")
        predictions.append(model.predict(data_block).astype(np.uint8))
        last_fit += n_in_block

    predicted = np.concatenate(predictions)

    model_and_predictions["predicted_classes"] = predicted
    model_and_predictions["true_classes"] = true_classes

    return model_and_predictions

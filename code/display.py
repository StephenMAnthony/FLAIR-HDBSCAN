import numpy as np
import pandas
import matplotlib.pyplot as plt
from sklearn import metrics

from calc_miou import calc_miou
from classifier import extract_spectra

LUT_COLORS = ['#db0e9a',
              '#938e7b',
              '#f80c00',
              '#a97101',
              '#1553ae',
              '#194a26',
              '#46e483',
              '#f3a60d',
              '#660082',
              '#55ff00',
              '#fff30d',
              '#e4df7c',
              '#3de6eb',
              '#ffffff',
              '#8ab3a0',
              '#6b714f',
              '#c5dc42',
              '#9999ff',
              '#000000']


def find_channel_label(dataframe: pandas.DataFrame, channel: int) -> str:
    # Determine the column label associated with the specified channel.
    columns = dataframe.columns.values.tolist()
    n_columns = len(columns)
    if channel < 1 or channel >= n_columns:
        return f"Error: Channel must be between 1 and {n_columns - 1}!"
    else:
        return columns[channel]


def display_confusion(true_classes: np.ndarray, predicted_classes: np.ndarray, config: dict, label: str):
    np.set_printoptions(precision=2)

    plt.rcParams["figure.figsize"] = (10, 8)
    semantic = [text.split(' ', 1)[0] for text in list(config['weights_aerial_satellite'].keys())[:-1]]

    metrics.ConfusionMatrixDisplay.from_predictions(true_classes, predicted_classes, display_labels=semantic,
                                                    normalize='true', xticks_rotation='vertical')

    # Generate a confusion matrix for the random classes
    confusion = metrics.confusion_matrix(true_classes, predicted_classes, normalize='true')
    # row_norm_factors = np.sum(random_confusion, axis=1)[:, np.newaxis]
    # random_confusion = random_confusion / row_norm_factors
    miou, ious = calc_miou(confusion)

    print(f"MIOU for {label} is: {miou}")


def box_whisker_by_class(dataframe: pandas.DataFrame, config: dict, channel: int) -> str:
    plt.rcParams["figure.figsize"] = (10, 8)

    # Determine the column label associated with the specified channel.
    channel_column = find_channel_label(dataframe, channel)
    if channel_column.split(' ')[0] == "Error:":
        return channel_column

    # Groupby class, extracting the desired channel
    grouped = dataframe.groupby("True_Class")[[channel_column]].apply(pandas.Series.tolist)

    # Calculate the median, standard deviation, and confidence intervals for each channel
    # class_stds = dataframe.groupby("True_Class")[[channel_column]].std()
    # class_medians = dataframe.groupby("True_Class")[[channel_column]].median()
    # conf_intervals = np.column_stack((class_medians - class_stds, class_medians + class_stds))

    # Extract the semantic class names
    semantic = [text.split(' ', 1)[0] for text in list(config['weights_aerial_satellite'].keys())[:-1]]

    # Determine the colors.
    colors = LUT_COLORS[:len(grouped)]

    # Reverse the order to align the display with other displays
    # grouped = grouped.iloc[::-1]
    semantic.reverse()
    colors.reverse()

    # Display a box-whisker plot
    fig, ax = plt.subplots()
    ax.set_xlabel(f"{channel_column} Values")
    ax.set_ylabel('Semantic Class')
    ax.set_title(f"{channel_column} Channel")
    bplot = ax.boxplot(grouped.iloc[::-1], labels=semantic, vert=False, patch_artist=True)

    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

    return 'Plotted'


def display_normalization_scatter(dataset, config, channel1=1, channel2=2):
    plt.rcParams["figure.figsize"] = (10, 8)

    original_aerial_data = extract_spectra(dataset, config, scale_by_intensity=False)
    normalized_aerial_data = extract_spectra(dataset, config, scale_by_intensity=True)

    # Determine the column labels associated with the specified channels.
    channel1_column = find_channel_label(original_aerial_data, channel1)
    if channel1_column.split(' ')[0] == "Error:":
        return channel1_column
    channel2_column = find_channel_label(original_aerial_data, channel2)
    if channel2_column.split(' ')[0] == "Error:":
        return channel2_column

    plot_kwds = {'alpha': 0.1, 's': 80, 'linewidths': 0}

    plt.subplot(121)
    plt.scatter(original_aerial_data[[channel1_column]].to_numpy().ravel(),
                original_aerial_data[[channel2_column]].to_numpy().ravel(),
                color='b', **plot_kwds)
    plt.xlabel(f"{channel1_column} Channel")
    plt.ylabel(f"{channel2_column} Channel")
    plt.title("Raw Spectra")

    plt.subplot(122)
    plt.scatter(normalized_aerial_data[[channel1_column]].to_numpy().ravel(),
                normalized_aerial_data[[channel2_column]].to_numpy().ravel(),
                color='b', **plot_kwds)
    plt.xlabel(f"{channel1_column} Channel")
    plt.ylabel(f"{channel2_column} Channel")
    plt.title("Normalized Spectra")

    print('Plotted')

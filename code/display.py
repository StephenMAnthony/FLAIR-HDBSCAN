import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

from calc_miou import calc_miou


def display_confusion(true_classes: np.ndarray, predicted_classes: np.ndarray, config: dict, label: str):

    np.set_printoptions(precision=2)

    plt.rcParams["figure.figsize"] = (12, 10)
    semantic = [text.split(' ', 1)[0] for text in list(config['weights_aerial_satellite'].keys())[:-1]]

    metrics.ConfusionMatrixDisplay.from_predictions(true_classes, predicted_classes, display_labels=semantic,
                                                    normalize='true', xticks_rotation='vertical')

    # Generate a confusion matrix for the random classes
    confusion = metrics.confusion_matrix(true_classes, predicted_classes, normalize='true')
    # row_norm_factors = np.sum(random_confusion, axis=1)[:, np.newaxis]
    # random_confusion = random_confusion / row_norm_factors
    miou, ious = calc_miou(confusion)

    print(f"MIOU for {label} is: {miou}")

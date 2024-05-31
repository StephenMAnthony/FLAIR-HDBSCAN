import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

from calc_miou import calc_miou


def display_confusion(confusion: np.ndarray, config: dict, label: str):

    np.set_printoptions(precision=2)

    plt.rcParams["figure.figsize"] = (12, 10)
    semantic = [text.split(' ', 1)[0] for text in list(config['weights_aerial_satellite'].keys())[:-1]]
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=semantic)
    disp.plot()
    plt.show()

    miou, ious = calc_miou(confusion)

    print(f"MIOU for {label} is: {miou}")

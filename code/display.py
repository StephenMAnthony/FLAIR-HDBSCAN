import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

from calc_miou import calc_miou


def display_confusion(confusion: np.ndarray, label: str):

    np.set_printoptions(precision=2)

    plt.rcParams["figure.figsize"] = (12, 10)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion)
    disp.plot()
    plt.show()

    miou, ious = calc_miou(confusion)

    print(f"MIOU for {label} is: {miou}")

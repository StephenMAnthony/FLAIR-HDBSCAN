import numpy as np
from sklearn.metrics import confusion_matrix


def calc_miou(cm_array):
    m = np.nan
    with np.errstate(divide='ignore', invalid='ignore'):
        ious = np.diag(cm_array) / (cm_array.sum(0) + cm_array.sum(1) - np.diag(cm_array))
    m = np.nansum(ious[:-1]) / (np.logical_not(np.isnan(ious[:-1]))).sum()
    return m.astype(float), ious[:-1]


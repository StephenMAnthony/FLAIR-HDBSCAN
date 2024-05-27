import numpy as np
import torch
from sklearn import metrics

def naive_clustering(the_dataset, config: dict):
    n_classes = config['num_classes']

    # Determine all labels and the count of pixels (labels)
    label_list = []
    for idx in range(len(the_dataset)):
        label_list.append(torch.Tensor.numpy(the_dataset[idx]['labels']).ravel())
    all_labels = np.concatenate(label_list).astype(int)
    total_count = all_labels.size

    # Generate random labels with uniform distribution
    rng = np.random.default_rng(12345)
    random_classes = rng.integers(low=0, high=n_classes, size=total_count)

    # Generate random labels with a distribution matching that of the original data
    permuted_classes = rng.permutation(all_labels)

    # Generate a confusion matrix for the random classes
    random_confusion = metrics.confusion_matrix(all_labels, random_classes) / total_count

    # Generate a confusion matrix for the permuted classes
    permuted_confusion = metrics.confusion_matrix(all_labels, permuted_classes) / total_count

    return random_confusion, permuted_confusion

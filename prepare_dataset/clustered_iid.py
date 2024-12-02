import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from prepare_dataset.loader import loadData


def create_cluster_loaders(train_set, test_set, numClusters, costFunc='CEloss', batchSize=32):
    (x_train, y_train) = train_set
    (x_test, y_test) = test_set
    num_classes = len(np.unique(y_train))
    cluster_size = num_classes // numClusters
    remainder = num_classes % numClusters

    cluster_labels_list = []
    start_label = 0

    # Define clusters
    for i in range(numClusters):
        end_label = start_label + cluster_size + (1 if i < remainder else 0)
        labels = list(range(start_label, end_label))
        cluster_labels_list.append(labels)
        start_label = end_label

    train_loaders = []
    val_loaders = []

    for cluster_labels in cluster_labels_list:
        # Training data
        train_indices = np.isin(y_train, cluster_labels)
        x_cluster_train = x_train[train_indices]
        y_cluster_train = y_train[train_indices]

        # Validation data
        val_indices = np.isin(y_test, cluster_labels)
        x_cluster_val = x_test[val_indices]
        y_cluster_val = y_test[val_indices]

        # Create datasets
        train_dataset = list(zip(y_cluster_train, x_cluster_train))
        val_dataset = list(zip(y_cluster_val, x_cluster_val))

        # Loaders
        train_loader = loadData(train_dataset, costFunc, numClass=num_classes, batchSize=batchSize)
        val_loader = loadData(val_dataset, costFunc, numClass=num_classes, batchSize=batchSize)

        train_loaders.append(train_loader)
        val_loaders.append(val_loader)

    return train_loaders, val_loaders
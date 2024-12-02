import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


def loadData(dataset, costFunc, numClass, batchSize=32):
    validation_y, validation_x = zip(*dataset)

    validation_x = np.array(validation_x)
    validation_y = np.array(validation_y)

    if costFunc == 'CEloss':
        pass
    elif costFunc == 'BCEloss':
        validation_y = np.eye(numClass)[validation_y]  # BCE
    elif costFunc == 'BCEWithLogitsLoss':
        pass

    X_validation = torch.tensor(validation_x, dtype=torch.float32).permute(0, 3, 1, 2)
    y_validation = torch.tensor(validation_y, dtype=torch.long)

    if costFunc == 'CEloss':
        pass
    elif costFunc == 'BCEloss':
        y_validation = torch.tensor(validation_y, dtype=torch.float32)
    elif costFunc == 'BCEWithLogitsLoss':
        pass

    validation_dataset = TensorDataset(X_validation, y_validation)
    val_loader = DataLoader(validation_dataset, batch_size=batchSize, shuffle=False)

    return val_loader
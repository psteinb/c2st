import os
from sklearn.metrics import balanced_accuracy_score
import torch
from torch import nn
from torchvision.datasets import MNIST, KMNIST
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from sklearn.model_selection import KFold
from typing import Union, Tuple
from sklearn.metrics import balanced_accuracy_score
import torchmetrics
import inspect
import numpy as np

def reset_weights(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

class SimpleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
          nn.Conv2d(1, 10, kernel_size=3),
          nn.ReLU(),
          nn.Flatten(),
          nn.Linear(26 * 26* 10, 50),
          nn.ReLU(),
          nn.Linear(50, 20),
          nn.ReLU(),
          nn.Linear(20, 1)
        )
    def forward(self, x):
        return self.layers(x)

class TorchClf(torch.nn.Module):
    def __init__(self, input_size, layers_data: list):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.input_size = input_size  # Can be useful later ...
        for size, activation in layers_data:
            self.layers.append(torch.nn.Linear(input_size, size))
            input_size = size  # For the next layer
            if activation is not None:
                assert isinstance(activation, torch.nn.modules.Module), \
                    "Each tuples should contain a size (int) and a torch.nn.modules.Module."
                self.layers.append(activation)

    def forward(self, input_data):
        for layer in self.layers:
            input_data = layer(input_data)
        return input_data

    

def c2st_torch(
    X: np.ndarray,
    Y: np.ndarray,
    scoring: str = "balanced_accuracy",
    z_score: bool = False,
    noise_scale: float = None,
    verbosity: int = 0,
    seed: int = 42,
    num_epochs: int = 10,
    model=TorchClf,
    cv=KFold(n_splits=5, shuffle=True),
    return_scores: bool = True,
    balanced: bool = False,
    nan_drop: bool = False,
    dtype_data=None,
    dtype_target=None,
    **kwargs
) -> Union[float, Tuple[float, np.ndarray]]:
    no_model_parameters = len(inspect.getfullargspec(model).args)
    if no_model_parameters > 1 and no_model_parameters - 1  > len(kwargs)  :
        raise TypeError(f"Your passed model requires {no_model_parameters} parameters but only {len(kwargs)} passed")
    
    if isinstance(X, torch.utils.data.Dataset):
        if Y is not None:
            raise ValueError("When X is a Dataset instance, Y should be None")
        dataset = X
        weights = 0.5
    else:
        if z_score:
            X_mean = np.mean(X, axis=0)
            X_std = np.std(X, axis=0)
            X = (X - X_mean) / X_std
            Y = (Y - X_mean) / X_std

        if noise_scale is not None:
            X += noise_scale * np.random.randn(*X.shape)
            Y += noise_scale * np.random.randn(*Y.shape)

        assert X.dtype == Y.dtype, f"{X.dtype=} not equal to {Y.dtype=}"

        data = np.concatenate((X, Y))
        if dtype_data is not None:
            data = data.astype(dtype_data)

        target = np.concatenate((np.zeros((X.shape[0],)), np.ones((Y.shape[0],))))
        if dtype_target is not None:
            target = target.astype(dtype_target)

        #dataset
        weights = torch.as_tensor(X.shape[0] / Y.shape[0], dtype=torch.float)
        tensor_x = torch.tensor(data, dtype=torch.float32) # transform to torch tensor
        tensor_y = torch.tensor(target, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(tensor_x,tensor_y) 

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loss_function = torch.nn.BCEWithLogitsLoss(
        pos_weight=weights if balanced else None
    )

    # For fold results
    results = []

    # Set fixed random number seed
    torch.manual_seed(seed)

    for fold, (train_ids, test_ids) in enumerate(cv.split(dataset)):

        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        # Define data loaders for training and testing data in this fold
        trainloader = torch.utils.data.DataLoader(
                          dataset, 
                          batch_size=16, sampler=train_subsampler)
        testloader = torch.utils.data.DataLoader(
                          dataset,
                          batch_size=16, sampler=test_subsampler)
        

        # Init the neural network
        network = model(**kwargs)
        network.apply(reset_weights)
        network.to(device)

        # Initialize optimizer
        optimizer = torch.optim.Adam(network.parameters(), lr=0.002)

        # Run the training loop for defined number of epochs
        for epoch in range(0, num_epochs):
            current_loss = 0.0
            for inputs, targets in trainloader:
                # Get inputs
                inputs, targets = inputs.to(device), targets.to(device)

                # Zero the gradients
                optimizer.zero_grad()

                # Perform forward pass
                outputs = network(inputs)

                # Compute loss
                loss = loss_function(outputs, targets.unsqueeze(1))

                # Perform backward pass
                loss.backward()

                # Perform optimization
                optimizer.step()

                # Print statistics
                current_loss += loss.item()


        # Evaluationfor this fold
        correct, total = 0, 0
        ii = 0
        with torch.no_grad():
            # Iterate over the test data and generate predictions
            acc = 0.0 
            bin_acc = torchmetrics.classification.Accuracy(task='multiclass', num_classes=2, average='macro' if balanced else 'micro').to(device)
            for inputs, targets in testloader:
                # Get inputs
                inputs, targets = inputs.to(device), targets.to(device)

                # Generate outputs
                outputs = network(inputs)
                proba = torch.nn.functional.sigmoid(outputs)
                pred = (proba > 0.5).long().squeeze(1)
                acc += bin_acc(pred, targets).item()

            results.append(acc / len(testloader))

    results = np.array(results)
    if return_scores:
        return results.mean(), results
    else:
        return results

import torch
import torchvision
from torchvision import datasets, transforms
from dataclasses import dataclass
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from typing import List, Union
import inspect

@dataclass
class ConvConfig:
    """
    Configuration dataclass for the ConvolutionClassifier.

    Attributes define the convolutional architecture, feedforward layer, and other 
    hyperparameters like pooling, activation, and dropout.
    """
    input_channels: int = 3
    num_channels: Union[int, List[int]] = 64
    num_layers: int = 5
    kernel_size: Union[int, List[int]] = 3
    padding: Union[int, List[int]] = 0
    stride: Union[int, List[int]] = 1
    maxpool_kernel_size: int = 2
    maxpool_padding: int = 0
    maxpool_stride: int = 2
    feedforward_dim: int = 512
    num_classes: int = 10
    apply_maxpool: bool = True
    apply_batchnorm: bool = True
    input_size: int = 224
    dropout_p: float = 0.2
    conv_activation_function: str = "ReLU"
    feedforward_activation_function: str = "ReLU"
    num_channels_multiplier: float = 1.0
    label_smoothing: float = 0.0

    def __post_init__(self):
        """
        Post-initialization that expands scalar parameters to lists 
        matching the number of layers.
        """
        self.channels = self._expand_to_list(self.num_channels, "num_channels")
        self.kernel_size = self._expand_to_list(self.kernel_size, "kernel_size")
        self.padding = self._expand_to_list(self.padding, "padding")
        self.stride = self._expand_to_list(self.stride, "stride")

    def _expand_to_list(self, value: Union[int, List[int]], name: str) -> List[int]:
        """
        Expands an int value to a list with length equal to num_layers,
        or validates the length of an existing list.

        Args:
            value (int or List[int]): The value to expand.
            name (str): The name of the parameter (used in error messages).

        Returns:
            List[int]: A list with length equal to num_layers.
        """
        if name == "num_channels" and isinstance(value, int) and self.num_channels_multiplier != 1.0:
            return [int(value * (self.num_channels_multiplier ** i)) for i in range(self.num_layers)]

        if isinstance(value, int):
            return [value] * self.num_layers
        elif isinstance(value, list):
            if len(value) != self.num_layers:
                raise ValueError(f"Length of {name} must be equal to num_layers")
            return value
        else:
            raise TypeError(f"{name} must be int or list of int")


class ConvolutionClassifier(nn.Module):
    """
    A customizable convolutional neural network for image classification.

    Architecture:
    - A series of convolutional layers (with optional batchnorm and max pooling)
    - A feedforward fully connected head for classification

    Args:
        config (ConvConfig): Configuration dataclass for model architecture.
    """
    def __init__(self, config):
        super(ConvolutionClassifier, self).__init__()
        self.config = config

        modules = []
        channels = [config.input_channels] + config.channels

        for i in range(config.num_layers):
            modules.append(nn.Conv2d(
                in_channels=channels[i],
                out_channels=channels[i + 1],
                kernel_size=config.kernel_size[i],
                padding=config.padding[i],
                stride=config.stride[i]
            ))
            if config.apply_batchnorm:
                modules.append(nn.BatchNorm2d(num_features=channels[i + 1]))
            modules.append(self.add_activation(config.conv_activation_function))
            if config.apply_maxpool:
                modules.append(nn.MaxPool2d(
                    kernel_size=config.maxpool_kernel_size,
                    padding=config.maxpool_padding,
                    stride=config.maxpool_stride
                ))

        self.feature_extractor = nn.Sequential(*modules)
        final_feature_dim = self.compute_final_feature_size()

        self.feedforward = nn.Sequential(
            nn.Linear(channels[-1] * final_feature_dim * final_feature_dim, config.feedforward_dim),
            self.add_activation(config.feedforward_activation_function),
            nn.Dropout(config.dropout_p),
            nn.Linear(config.feedforward_dim, config.num_classes)
        )

        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    def add_activation(self, activation):
        """
        Returns the PyTorch activation function corresponding to the provided string name.

        Args:
            activation (str): The name of the activation function.

        Returns:
            nn.Module: An instance of the specified activation function.

        Raises:
            ValueError: If the activation function is not supported by torch.nn.
        """
        if hasattr(nn, activation):
            return getattr(nn, activation)()
        raise ValueError(f"{activation} is not a valid activation function supported by torch")

    def compute_final_feature_size(self):
        """
        Computes the spatial size of the feature map after the convolutional layers.

        Returns:
            int: Final width/height (assuming square input) of the feature map.
        """
        dim_ = self.config.input_size
        for i in range(self.config.num_layers):
            dim_ = ((dim_ + (2 * self.config.padding[i]) - self.config.kernel_size[i]) // self.config.stride[i]) + 1
            if self.config.apply_maxpool:
                dim_ = ((dim_ + (2 * self.config.maxpool_padding) - self.config.maxpool_kernel_size) // self.config.maxpool_stride) + 1
        return dim_

    def forward(self, x, y):
        """
        Forward pass through the model, returning loss, accuracy, and predictions.

        Args:
            x (Tensor): Input image batch (B, C, H, W).
            y (Tensor): Ground truth labels (B,).

        Returns:
            Tuple[Tensor, float, Tensor]: Loss, accuracy, and predicted class indices.
        """
        B = x.shape[0]
        x = self.feature_extractor(x)
        x = x.view(B, -1)
        x = self.feedforward(x)
        loss, acc, max_ = self.compute_loss_and_accuracy(x, y)
        return loss, acc, max_

    def compute_loss_and_accuracy(self, x, y):
        """
        Computes classification loss and accuracy.

        Args:
            x (Tensor): Raw model logits.
            y (Tensor): Ground truth labels.

        Returns:
            Tuple[Tensor, float, Tensor]: Loss, accuracy, and predicted class indices.
        """
        loss = self.loss_fn(x, y)
        max_ = F.log_softmax(x, dim=-1)
        max_ = torch.argmax(max_, dim=-1)
        acc = (max_ == y).sum().item() / y.shape[0]
        return loss, acc, max_

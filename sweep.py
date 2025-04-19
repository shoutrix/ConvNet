from trainer import Trainer
import wandb
from dataclasses import dataclass, field



@dataclass
class TrainerConfig:
    train_data_path: str = "/speech/shoutrik/Database/INaturalist/inaturalist_12K/train"
    valid_data_path: str = "/speech/shoutrik/Database/INaturalist/inaturalist_12K/valid"
    test_data_path: str = "/speech/shoutrik/Database/INaturalist/inaturalist_12K/test"
    input_channels: int = 3
    num_channels: list = field(default_factory=lambda: [32, 64, 128, 256, 512])
    num_layers: int = 5
    kernel_size: list = field(default_factory=lambda: [7, 5, 3, 3, 3])
    padding: list = field(default_factory=lambda: [0, 0, 0, 0, 0])
    stride: list = field(default_factory=lambda: [1, 1, 1, 1, 1])
    maxpool_kernel_size: int = 2
    maxpool_padding: int = 0
    maxpool_stride: int = 2
    feedforward_dim: int = 1024
    num_classes: int = 10
    apply_maxpool: bool = True
    apply_batchnorm: bool = True
    input_size: int = 224
    dropout_p: float = 0.2
    conv_activation_function: str = "GELU"
    feedforward_activation_function: str = "ReLU"
    num_channels_multiplier: float = 1.0
    apply_augmentations: bool = True
    batch_size: int = 128
    num_workers: int = 16
    learning_rate: float = 0.0001
    weight_decay: float = 0.0005
    max_epoch: int = 10
    label_smoothing: float = 0.2
    seed: int = 426


def main():
    wandb.init()
    channels = "-".join([str(c) for c in wandb.config.num_channels])
    wandb.run.name = f"Channel{channels}_epoch{wandb.config.max_epoch}_wd{wandb.config.weight_decay}_lr{wandb.config.learning_rate}_bs{wandb.config.batch_size}_aug{wandb.config.apply_augmentations}_bn{wandb.config.apply_batchnorm}_dp{wandb.config.dropout_p}_fd{wandb.config.feedforward_dim}_act{wandb.config.conv_activation_function}_ls{wandb.config.label_smoothing}"

    config = TrainerConfig(
        max_epoch=wandb.config.max_epoch,
        weight_decay=wandb.config.weight_decay,
        learning_rate=wandb.config.learning_rate,
        batch_size=wandb.config.batch_size,
        apply_augmentations=wandb.config.apply_augmentations,
        apply_batchnorm=wandb.config.apply_batchnorm,
        dropout_p=wandb.config.dropout_p,
        feedforward_dim=wandb.config.feedforward_dim,
        conv_activation_function=wandb.config.conv_activation_function,
        label_smoothing=wandb.config.label_smoothing,
    )
    
    trainer = Trainer(config, logging=True)
    trainer.train()


if __name__ == "__main__":
    sweep_configuration = {
        "method": "bayes",
        "name": "sweep_trial01",
        "metric": {"goal": "maximize", "name": "valid_accuracy"},
        "parameters": {
            "num_channels": {"values": [[32, 64, 128, 256, 512], [64, 64, 64, 64, 64], [256, 128, 64, 32, 16]]},
            "kernel_size": {"values": [[7, 5, 3, 3, 3], [3, 3, 3, 3, 3], [5, 5, 5, 5, 5]]},
            "max_epoch": {"values": [10, 15]},
            "weight_decay": {"values": [0.00005, 0.0001]},
            "learning_rate": {"values": [0.0001, 0.001]},
            "batch_size": {"values": [64, 128]},
            "apply_augmentations": {"values": [True, False]},
            "apply_batchnorm": {"values": [True, False]},
            "dropout_p": {"values": [0.2, 0.3, 0.4]},
            "feedforward_dim": {"values": [512, 1024]},
            "conv_activation_function": {"values": ["GELU", "SiLU", "ReLU"]},
            "label_smoothing": {"values": [0.1]},
            # "num_channels_multiplier": {"values" : [1.0, 2.0]}
        },
    }

    wandb.login(key="3fbe34f050b1f8cac4896673695870138d90d9e2")
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="ConvNet_expts2", entity="shoutrik")
    wandb.agent(sweep_id, function=main, count=150)
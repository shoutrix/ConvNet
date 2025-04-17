from trainer import Trainer
import wandb
from dataclasses import dataclass, field

train_data_path = "/speech/shoutrik/Database/INaturalist/inaturalist_12K/train"
valid_data_path = "/speech/shoutrik/Database/INaturalist/inaturalist_12K/valid"
test_data_path = "/speech/shoutrik/Database/INaturalist/inaturalist_12K/test"

@dataclass
class TrainerConfig:
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
    num_channels_multiplier: float = 1
    apply_augmentations: bool = True
    batch_size: int = 128
    num_workers: int = 16
    learning_rate: float = 0.0001
    weight_decay: float = 0.0005
    max_epoch: int = 10


def main():
    wandb.init()
    wandb.run.name = f"epoch{wandb.config.max_epoch}_wd{wandb.config.weight_decay}_lr{wandb.config.learning_rate}_bs{wandb.config.batch_size}_aug{wandb.config.apply_augmentations}_bn{wandb.config.apply_batchnorm}_dp{wandb.config.dropout_p}_fd{wandb.config.feedforward_dim}_act{wandb.config.conv_activation_function}"

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
    )
    
    trainer = Trainer(config, train_data_path, valid_data_path, test_data_path, logging=True)
    trainer.train()


if __name__ == "__main__":
    sweep_configuration = {
        "method": "bayes",
        "name": "sweep_trial01",
        "metric": {"goal": "maximize", "name": "valid_accuracy"},
        "parameters": {
            "max_epoch": {"values": [10]},
            "weight_decay": {"values": [0.0001, 0.0005]},
            "learning_rate": {"values": [0.0001, 0.001, 0.003]},
            "batch_size": {"values": [64, 128]},
            "apply_augmentations": {"values": [True, False]},
            "apply_batchnorm": {"values": [True, False]},
            "dropout_p": {"values": [0.2, 0.3]},
            "feedforward_dim": {"values": [512, 1024]},
            "conv_activation_function": {"values": ["ReLU", "GELU", "SiLU"]},
        },
    }

    sweep_id = wandb.sweep(sweep=sweep_configuration, project="ConvNet_expts", entity="shoutrik")
    wandb.agent(sweep_id, function=main, count=30)
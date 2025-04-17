import os
import torch
import torchvision
from torchvision import datasets, transforms
from dataclasses import dataclass
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from typing import List, Union
import inspect
from classifiers import ConvConfig, ConvolutionClassifier
import wandb
from torch.amp import autocast, GradScaler


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, apply_augmentations=True):
        super(ImageDataset, self).__init__()
        self.data = datasets.ImageFolder(data_path)

        self.original_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        if apply_augmentations:
            self.augmented_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        
        self.apply_augmentations = apply_augmentations
    
    
    def __len__(self):
        if self.apply_augmentations:
            return len(self.data) * 2
        else:
            return len(self.data)
    
    def __getitem__(self, idx):
        if self.apply_augmentations:
            true_idx = idx//2
        
            img, label = self.data[true_idx]
            
            if idx%2 == 0:
                img = self.original_transform(img)
            else:
                img = self.augmented_transform(img)
        else:
            img, label = self.data[idx]
            img = self.original_transform(img)
        return img, label

class Trainer():
    def __init__(self, args, train_data_path, valid_data_path, test_data_path, logging):
        self.args = args
        self.logging = logging

        trainset = ImageDataset(train_data_path, apply_augmentations=args.apply_augmentations)
        valset = ImageDataset(valid_data_path, apply_augmentations=False)
        testset = ImageDataset(test_data_path, apply_augmentations=False)

        self.trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, persistent_workers=True)
        self.valloader = DataLoader(valset, batch_size=args.batch_size*2, shuffle=False, num_workers=args.num_workers, pin_memory=True, persistent_workers=True)
        self.testloader = DataLoader(testset, batch_size=args.batch_size*2, shuffle=False, num_workers=args.num_workers, pin_memory=True, persistent_workers=True)

        class_to_idx = testset.data.class_to_idx
        self.labels = {v: k for k, v in class_to_idx.items()}

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        config = ConvConfig(
            input_channels=args.input_channels,
            num_channels=args.num_channels,
            num_layers=args.num_layers,
            kernel_size=args.kernel_size,
            padding=args.padding,
            stride=args.stride,
            maxpool_kernel_size=args.maxpool_kernel_size,
            maxpool_padding=args.maxpool_padding,
            maxpool_stride=args.maxpool_stride,
            feedforward_dim=args.feedforward_dim,
            num_classes=args.num_classes,
            apply_maxpool=args.apply_maxpool,
            apply_batchnorm=args.apply_batchnorm,
            input_size=args.input_size,
            dropout_p=args.dropout_p,
            conv_activation_function=args.conv_activation_function,
            feedforward_activation_function=args.feedforward_activation_function,
            num_channels_multiplier=args.num_channels_multiplier,
        )
        self.model = ConvolutionClassifier(config).to(self.device)
        print(self.model)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

        if torch.cuda.is_bf16_supported():
            self.autocast_dtype = torch.bfloat16
            self.bf16 = True
            self.scaler = None
        else:
            self.autocast_dtype = torch.float16
            self.bf16 = False
            self.scaler = GradScaler()
        
        print(f"autocast dtype set to : {self.autocast_dtype}")

    def train(self):
        print(f"Starting training on device: {self.device} | Batch size: {self.args.batch_size} | Train batches: {len(self.trainloader)} | Valid batches: {len(self.valloader)}")
        for epoch in range(self.args.max_epoch):
            self.run_one_epoch(epoch)
        print("Finished Training!\nStarting Evaluation on Test Set...")
        self.evaluate()

    def run_one_epoch(self, epoch):
        mean_train_loss, mean_train_acc = self.train_one_epoch()
        mean_val_loss, mean_val_acc = self.validate_one_epoch()

        print(f"Epoch {epoch + 1} | Train Loss: {mean_train_loss / len(self.trainloader):.4f} | Train Accuracy: {mean_train_acc / len(self.trainloader):.4f} | Valid Loss: {mean_val_loss / len(self.valloader):.4f} | Valid Accuracy: {mean_val_acc / len(self.valloader):.4f}")

        if self.logging:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": mean_train_loss / len(self.trainloader),
                "train_accuracy": mean_train_acc / len(self.trainloader),
                "valid_loss": mean_val_loss / len(self.valloader),
                "valid_accuracy": mean_val_acc / len(self.valloader),
                "lr": self.optimizer.param_groups[0]["lr"]
            })

    def train_one_epoch(self):
        self.model.train()
        mean_train_loss, mean_train_acc = 0, 0

        for x, y in self.trainloader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            with autocast(device_type=self.device, dtype=self.autocast_dtype):
                loss, acc, _ = self.model(x, y)

            if self.bf16:
                loss.backward()
                self.optimizer.step()
            else:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            mean_train_loss += loss.item()
            mean_train_acc += acc

        return mean_train_loss, mean_train_acc

    def validate_one_epoch(self):
        self.model.eval()
        mean_val_loss, mean_val_acc = 0, 0

        with torch.no_grad(), autocast(device_type=self.device, dtype=self.autocast_dtype):
            for x, y in self.valloader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                loss, acc, _ = self.model(x, y)
                mean_val_loss += loss.item()
                mean_val_acc += acc

        return mean_val_loss, mean_val_acc

    def evaluate(self):
        self.model.eval()
        mean_test_loss, mean_test_acc = 0, 0
        targets, predictions = [], []

        with torch.no_grad(), autocast(device_type=self.device, dtype=self.autocast_dtype):
            for x, y in self.testloader:
                targets.append(y)
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                loss, acc, preds = self.model(x, y)

                predictions.append(preds)
                mean_test_loss += loss.item()
                mean_test_acc += acc

        targets = torch.cat(targets).view(-1).detach().cpu().numpy()
        predictions = torch.cat(predictions).view(-1).detach().cpu().numpy()

        print(f"Test Loss: {mean_test_loss / len(self.testloader):.4f} | Test Accuracy: {mean_test_acc / len(self.testloader):.4f}")

        if self.logging:
            print("Logging test metrics...")
            wandb.log({
                "test_loss": mean_test_loss / len(self.testloader),
                "test_accuracy": mean_test_acc / len(self.testloader),
                "confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=targets,
                    preds=predictions,
                    class_names=[self.labels[i] for i in range(len(self.labels))]
                )
            })

        print("Evaluation complete. Yaay !!! 🥳 ")

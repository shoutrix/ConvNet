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
            img = self.original_transform(img)
        return img, label
    
    
class Trainer():
    def __init__(self, args, train_data_path, valid_data_path, test_data_path, logging):
        self.args = args
        self.logging = logging
        trainset = ImageDataset(train_data_path, apply_augmentations=args.apply_augmentations)
        valset = ImageDataset(valid_data_path, apply_augmentations=args.apply_augmentations)
        testset = ImageDataset(test_data_path)
        
        self.trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, persistent_workers=True) # pin memory, persistent workers ??
        self.valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, persistent_workers=True)
        self.testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, persistent_workers=True)
        
        class_to_idx = testset.data.class_to_idx
        self.labels = {v:k for k, v in class_to_idx.items()}  
                 
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
                 
        config = ConvConfig(
                input_channels = args.input_channels,
                num_channels = args.num_channels,
                num_layers = args.num_layers,
                kernel_size = args.kernel_size,
                padding = args.padding,
                stride = args.stride,
                maxpool_kernel_size = args.maxpool_kernel_size,
                maxpool_padding = args.maxpool_padding,
                maxpool_stride = args.maxpool_stride,
                feedforward_dim = args.feedforward_dim,
                num_classes = args.num_classes,
                apply_maxpool = args.apply_maxpool,
                apply_batchnorm = args.apply_batchnorm,
                input_size = args.input_size,
                dropout_p = args.dropout_p,
                conv_activation_function = args.conv_activation_function,
                feedforward_activation_function = args.feedforward_activation_function,
                maxpool_after_each_layer = args.maxpool_after_each_layer,
                num_channels_multiplier = args.num_channels_multiplier,
        )

        self.model = ConvolutionClassifier(config)
        self.model = self.model.to(self.device)

        print(self.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=args.learning_rate, weight_decay=args.weight_decay)
        
        
    def train(self):
        print(f"Starting training on device : {self.device} with {self.args.batch_size} batch size | Train num_batches : {len(self.trainloader)} | Valid num_batches : {len(self.valloader)}")
        for epoch in range(self.args.max_epoch):
            self.run_one_epoch(epoch)
        print("Finished Training !!!\nStarting Evaluation on test set ...")
        self.evaluate()

    def run_one_epoch(self, epoch):
        print(f"Starting Epoch : {epoch + 1}")
        mean_train_loss, mean_train_acc = self.train_one_epoch()
        mean_val_loss, mean_val_acc = self.validate_one_epoch()
        print(f"Epoch : {epoch +1} | Train loss : {mean_train_loss/len(self.trainloader)} | Train accuracy : {mean_train_acc/len(self.trainloader)} | Valid loss : {mean_val_loss/len(self.valloader)} | Valid accuracy : {mean_val_acc/len(self.valloader)}")            

    def train_one_epoch(self):
        mean_train_loss, mean_train_acc = 0, 0
        for i, (x, y) in enumerate(self.trainloader):
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            loss, acc, _ = self.model(x, y)
            loss.backward()
            self.optimizer.step()
            mean_train_loss += loss.item()
            mean_train_acc += acc
        
        if self.logging:
            wandb.log({"train_loss": mean_train_loss, "train_accuracy": mean_train_acc})
        return mean_train_loss, mean_train_acc
    
    def validate_one_epoch(self):
        mean_val_loss, mean_val_acc = 0, 0
        self.model.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(self.valloader):
                x, y = x.to(self.device), y.to(self.device)
                loss, acc, _ = self.model(x, y)
                mean_val_loss += loss
                mean_val_acc += acc
        if self.logging:
            wandb.log({"valid_loss": mean_val_loss, "valid_accuracy": mean_val_acc})
        return mean_val_loss, mean_val_acc
    
    def evaluate(self):
        mean_test_loss, mean_test_acc = 0, 0
        targets = []
        predictions = []
        self.model.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(self.testloader):
                targets.append(y)
                x, y = x.to(self.device), y.to(self.device)
                loss, acc, preds = self.model(x, y)
                predictions.append(preds)
                mean_test_loss += loss
                mean_test_acc += acc
            
        targets = torch.cat(targets).view(-1).detach().cpu().numpy()
        predictions = torch.cat(predictions).view(-1).detach().cpu().numpy()
                        
        print(f"Test loss : {mean_test_loss/len(self.testloader)} | Test accuracy : {mean_test_acc/len(self.testloader)}")
        if self.logging:
            print("Logging Inference stats ...")
            wandb.log({"test_loss": mean_test_loss, "test_accuracy": mean_test_acc})
            wandb.log({"confusion_matrix" : wandb.plot.confusion_matrix(probs=None, y_true=targets, preds=predictions, class_names=self.labels)})
        print("Finished Evaluation. Yaay !!!")
        
    
        

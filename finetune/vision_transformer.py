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
import wandb
from torch.amp import autocast, GradScaler
import random
import numpy as np
import argparse
from torch.optim.lr_scheduler import SequentialLR, LinearLR, ExponentialLR
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision.transforms.functional import InterpolationMode



def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-wp", "--wandb_project", type=str, required=False, help="WandB project name", default=None)
    parser.add_argument("-we", "--wandb_entity", type=str, required=False, help="WandB entity", default=None)
    parser.add_argument("-rn", "--run_name", type=str, required=False, help="WandB run name", default=None)
    
    parser.add_argument("--train_data_path", type=str, required=False, default="/speech/shoutrik/Database/INaturalist/inaturalist_12K/train")
    parser.add_argument("--valid_data_path", type=str, required=False, default="/speech/shoutrik/Database/INaturalist/inaturalist_12K/valid")
    parser.add_argument("--test_data_path", type=str, required=False, default="/speech/shoutrik/Database/INaturalist/inaturalist_12K/test")
    
    parser.add_argument("--finetune_strategy", type=int, required=False, choices=[0, 1, 2], default=0)

    parser.add_argument('--feedforward_dim', type=int, default=512)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--dropout_p', type=float, default=0.2)
    parser.add_argument('--apply_augmentations', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--max_epoch', type=int, default=10)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=424)
    parser.add_argument('--warmup_steps', type=int, default=400)


    return parser.parse_args()


# torch._dynamo.config.capture_scalar_outputs = True

class ImageDataset(torch.utils.data.Dataset):
    """
    A custom PyTorch Dataset for loading and optionally augmenting images 
    from a directory using torchvision.datasets.ImageFolder.

    Args:
        data_path (str): Path to the root image directory, organized in subfolders per class.
        apply_augmentations (bool): Whether to apply data augmentations. If True, 
                                    returns both original and augmented versions of each image.

    Attributes:
        data (ImageFolder): Loaded dataset using ImageFolder.
        original_transform (Compose): Transformation pipeline for original (non-augmented) images.
        augmented_transform (Compose): Transformation pipeline with augmentations (if enabled).
        apply_augmentations (bool): Flag to control if augmentations are used.
    """
    def __init__(self, data_path, apply_augmentations=True, weights=None, mixup_p=0.2):
        super(ImageDataset, self).__init__()
        self.data = datasets.ImageFolder(data_path)
        self.mixup_p = mixup_p

        self.original_transform = transforms.Compose([
            transforms.Resize(weights.transforms().resize_size, interpolation=weights.transforms().interpolation),
            transforms.CenterCrop(weights.transforms().crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=weights.transforms().mean, std=weights.transforms().std)
        ])

        if apply_augmentations:
            self.augmented_transform = transforms.Compose([
                transforms.Resize(weights.transforms().resize_size, interpolation=weights.transforms().interpolation),
                transforms.CenterCrop(weights.transforms().crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=weights.transforms().mean, std=weights.transforms().std)
            ])

        
        self.apply_augmentations = apply_augmentations
    
    def __len__(self):
        """
        Returns the length of the dataset.

        If augmentations are applied, returns twice the size of the original dataset
        since both original and augmented versions are returned.
        """
        if self.apply_augmentations:
            return len(self.data) * 2
        else:
            return len(self.data)
            
    def __getitem__(self, idx):
        """
        Retrieves an image and its label by index.

        If augmentations are applied, returns either the original or augmented version 
        based on index parity. Otherwise, returns only the original version.

        Args:
            idx (int): Index of the image.

        Returns:
            img (Tensor): Transformed image tensor.
            label (int): Corresponding class label.
        """
        if self.apply_augmentations:
            true_idx = idx // 2
            img, label = self.data[true_idx]
            
            if idx % 2 == 0:
                img = self.original_transform(img)
            else:
                img = self.augmented_transform(img)
        else:
            img, label = self.data[idx]
            img = self.original_transform(img)
        
        return img, label


class Trainer():
    """
    Trainer class to initialize datasets, dataloaders, model, optimizer, 
    and mixed-precision configuration for image classification training.

    Args:
        args (Namespace): Command-line or script arguments containing all required hyperparameters and settings.
        train_data_path (str): Path to the training data directory.
        valid_data_path (str): Path to the validation data directory.
        test_data_path (str): Path to the test data directory.
        logging (Logger): Logger instance for logging messages.
    """
    def __init__(self, args, logging):
        self.args = args
        self.logging = logging
        
        self.set_seed(args.seed)  
        
        try:
            weights = getattr(ViT_B_16_Weights, args.weights)
        except AttributeError:
            raise ValueError(f"Invalid weights '{args.weights}'. Valid options are: "
                             "IMAGENET1K_V1, IMAGENET1K_SWAG_E2E_V1, IMAGENET1K_SWAG_LINEAR_V1.")

        # self.weights = weights
        # print("aaaa : ", weights.transforms())

        trainset = ImageDataset(args.train_data_path, apply_augmentations=args.apply_augmentations, weights=weights)
        valset = ImageDataset(args.valid_data_path, apply_augmentations=False, weights=weights)
        testset = ImageDataset(args.test_data_path, apply_augmentations=False, weights=weights)

        self.trainloader = DataLoader(
            trainset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=True
        )

        self.valloader = DataLoader(
            valset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=True
        )

        self.testloader = DataLoader(
            testset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=True
        )

        class_to_idx = testset.data.class_to_idx
        self.labels = {v: k for k, v in class_to_idx.items()}

        # Set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Instantiate model
        model = vit_b_16(weights=weights) 
        
        final_layer_size = model.heads.head.in_features
        model.heads.head = nn.Sequential(
            nn.Linear(final_layer_size, args.feedforward_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(args.dropout_p),
            nn.Linear(args.feedforward_dim, args.num_classes)
        )
        for param in model.parameters():
            param.requires_grad = False
        
        self.unfreeze_full_model_epoch = None
        
        # train only the classifier head
        if args.finetune_strategy==0:
            print("Training only the classifier head")
            for param in model.heads.head.parameters():
                param.requires_grad = True
                
        # train the last block and the classifier head
        if args.finetune_strategy==1:
            print("Training only the classifier head and the final block")
            for param in model.heads.head.parameters():
                param.requires_grad = True
            for param in model.encoder.layers.encoder_layer_11.parameters():
                param.requires_grad = True
                
        # Sequential training of first classifier then full model
        if args.finetune_strategy==2:
            for param in model.heads.head.parameters():
                param.requires_grad = True
            self.unfreeze_full_model_epoch = 4
            print(f"Training only the classifier head initialy, full model will be trained after epoch : {self.unfreeze_full_model_epoch}")
                        
        self.model = model.to(self.device)
        # self.model = torch.compile(self.model)
        print(self.model)
        
        # Optimizer setup
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        
        # Scheduler
        warmup_scheduler = LinearLR(optimizer=self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_steps)
        exponential_decay = ExponentialLR(optimizer=self.optimizer, gamma=0.999)
        self.scheduler = SequentialLR(self.optimizer, schedulers=[warmup_scheduler, exponential_decay], milestones=[args.warmup_steps])
        
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

        # Mixed precision configuration
        if torch.cuda.is_bf16_supported():
            self.autocast_dtype = torch.bfloat16
            self.bf16 = True
            self.scaler = None
        else:
            self.autocast_dtype = torch.float16
            self.bf16 = False
            self.scaler = GradScaler()

        print(f"autocast dtype set to : {self.autocast_dtype}")


    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    def unfreeze_full_model(self):
        for param in self.model.parameters():
            param.requires_grad = True
        
        new_params = []
        for param in self.model.parameters():
            if param.requires_grad and not any(param is p for group in self.optimizer.param_groups for p in group["params"]):
                new_params.append(param)
        
        self.optimizer.add_param_group({"params":new_params})
        
    
    def compute_loss_and_acc(self, target, model_out):
        B = model_out.shape[0]
        loss_ = self.loss_fn(model_out, target)
        probs_ = F.log_softmax(model_out, dim=-1)
        max_ = torch.argmax(probs_, dim=-1)
        acc_ = (target == max_).sum().item() / B
        return loss_, acc_, max_
        

    def train(self):
        """
        Main training loop that runs for the specified number of epochs.
        After training, it performs evaluation on the test set.
        """
        print(f"Starting training on device: {self.device} | Batch size: {self.args.batch_size} | Train batches: {len(self.trainloader)} | Valid batches: {len(self.valloader)}")
        for epoch in range(self.args.max_epoch):
            if self.args.finetune_strategy == 2 and epoch+1 == self.unfreeze_full_model_epoch:
                print(f"[Epoch {epoch + 1}] Unfreezing entire model for full fine-tuning.")
                self.unfreeze_full_model()
            self.run_one_epoch(epoch)
        print("Finished Training!\nStarting Evaluation on Test Set...")
        self.evaluate()

    def run_one_epoch(self, epoch):
        """
        Runs one full epoch including training and validation.

        Args:
            epoch (int): The current epoch number.

        Logs training and validation metrics to wandb if logging is enabled.
        """
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
        """
        Performs a single epoch of training.

        Returns:
            mean_train_loss (float): Accumulated training loss.
            mean_train_acc (float): Accumulated training accuracy.
        """
        self.model.train()
        mean_train_loss, mean_train_acc = 0, 0

        for x, y in self.trainloader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            with autocast(device_type=self.device, dtype=self.autocast_dtype):
                out_ = self.model(x)
                loss, acc, _ = self.compute_loss_and_acc(y, out_)

            if self.bf16:
                loss.backward()
                self.optimizer.step()
                
            else:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            
            self.scheduler.step()

            mean_train_loss += loss.item()
            mean_train_acc += acc

        return mean_train_loss, mean_train_acc

    def validate_one_epoch(self):
        """
        Performs a single epoch of validation.

        Returns:
            mean_val_loss (float): Accumulated validation loss.
            mean_val_acc (float): Accumulated validation accuracy.
        """
        self.model.eval()
        mean_val_loss, mean_val_acc = 0, 0

        with torch.no_grad(), autocast(device_type=self.device, dtype=self.autocast_dtype):
            for x, y in self.valloader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                out_ = self.model(x)
                loss, acc, _ = self.compute_loss_and_acc(y, out_)
                mean_val_loss += loss.item()
                mean_val_acc += acc

        return mean_val_loss, mean_val_acc

    def evaluate(self):
        """
        Runs evaluation on the test set, calculates metrics, and optionally logs to wandb.

        Logs:
            - Test loss and accuracy.
            - Confusion matrix via wandb if logging is enabled.
        """
        self.model.eval()
        mean_test_loss, mean_test_acc = 0, 0
        targets, predictions = [], []

        with torch.no_grad(), autocast(device_type=self.device, dtype=self.autocast_dtype):
            for x, y in self.testloader:
                targets.append(y)
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                out_ = self.model(x)
                loss, acc, preds = self.compute_loss_and_acc(y, out_)

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
                    class_names=self.labels
                )
            })

        print("Evaluation complete. Yaay !!! 🥳")

if __name__ == "__main__":
    args = parse_args()
    print(args)
    if args.wandb_entity is not None and args.wandb_project is not None:
        logging = True
        wandb.login(key="3fbe34f050b1f8cac4896673695870138d90d9e2")
        wandb.init(project=f"{args.wandb_project}", 
                            entity=args.wandb_entity, 
                            name=args.run_name,
                            config=vars(args))

    else:
        logging = False
    trainer = Trainer(args, logging=logging)
    trainer.train()
    if logging:
        wandb.finish()
# 🦋 INaturalist-12K Image Classifier using CNN

This repository provides a modular and configurable **Convolutional Neural Network (CNN)** built with **PyTorch** for image classification on the **INaturalist-12K** dataset. It supports command-line configuration, data augmentations, Weights & Biases (wandb) logging, and is designed with scalability and ease of use in mind.

---

## 📌 Features

- Customizable CNN architecture  
- Data augmentations (optional)  
- Dropout and Batch Normalization  
- Full training configuration via CLI  
- Weights & Biases (wandb) integration  
- Supports INaturalist-12K dataset structure out-of-the-box  

---

## 📂 Dataset Structure

Ensure the dataset is structured as follows:

```
inaturalist_12K/
├── train/
│   ├── class1/
│   ├── class2/
│   └── ...
├── valid/
│   ├── class1/
│   ├── class2/
│   └── ...
└── test/
    ├── class1/
    ├── class2/
    └── ...
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/shoutrix/ConvNet.git
cd ConvNet
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🏃‍♂️ Training the Model

Use the following command to start training:

```bash
python run.py \
  --wandb_project INatClassifier \
  --wandb_entity your_wandb_username \
  --learning_rate 0.0003 \
  --batch_size 64 \
  --dropout_p 0.3 \
  --apply_augmentations True
```

Modify hyperparameters as needed from the table below.

---

## ⚙️ Command-Line Arguments

| Argument                              | Description                                              | Default                    |
|---------------------------------------|----------------------------------------------------------|----------------------------|
| `--input_channels`                    | Number of input channels (e.g., 3 for RGB)               | `3`                        |
| `--num_layers`                        | Number of convolutional layers                           | `5`                        |
| `--num_channels`                      | List of channels per conv layer                          | `[32, 64, 128, 256, 512]`  |
| `--kernel_size`                       | List of kernel sizes per layer                           | `[7, 5, 3, 3, 3]`          |
| `--feedforward_dim`                   | Dimension of the feedforward linear layer                | `1024`                     |
| `--num_classes`                       | Number of output classes                                 | `10`                       |
| `--dropout_p`                         | Dropout probability                                      | `0.2`                      |
| `--conv_activation_function`          | Activation in conv layers (`ReLU`, `GELU`, etc.)         | `'GELU'`                   |
| `--feedforward_activation_function`   | Activation in feedforward layers                         | `'ReLU'`                   |
| `--apply_batchnorm`                   | Apply batch normalization                                | `True`                     |
| `--apply_augmentations`               | Apply image augmentations                                | `False`                    |
| `--learning_rate`                     | Learning rate                                            | `0.0001`                   |
| `--weight_decay`                      | Weight decay for optimizer                               | `0`                        |
| `--batch_size`                        | Batch size                                               | `128`                      |
| `--max_epoch`                         | Number of training epochs                                | `10`                       |

---

## 📊 Experiment Tracking with Weights & Biases

This project supports [Weights & Biases](https://wandb.ai/) for experiment tracking. To use:

1. Sign up at [wandb.ai](https://wandb.ai/)
2. Replace `your_wandb_username` in the CLI command
3. (Optional) Set `WANDB_API_KEY` as an environment variable for auto-login

---

## 🧪 Example Command

```bash
python run.py \
  --wandb_project INatClassifier \
  --wandb_entity myusername \
  --learning_rate 0.0005 \
  --batch_size 32 \
  --dropout_p 0.4 \
  --num_layers 4 \
  --apply_augmentations True
```

---

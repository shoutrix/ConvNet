# 🦋 VisionTransformer and Resnet50 finetuned on INaturalist-12K

This repository provides a recipe to finetune vision transformer and resnet50 on INatuarlist-12k
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

## 🎯 Finetuning Strategies

Both models support **three fine-tuning strategies** via the `--finetune_strategy` argument:

- `0`: Finetune **only the classifier head**
- `1`: Finetune the **classifier head and the last layer**
- `2`: Finetune **only the classifier head for a few epochs**, then **unfreeze the full model** for end-to-end training

🔧 **Note**:  
The final classifier head is always initialized **randomly** and trained from scratch.


## Finetuning resnet50

Use the following command to start training:

```bash
python resnet50.py \
  --wandb_project INatClassifier \
  --wandb_entity your_wandb_username \
  --run_name run_name \
  --learning_rate 0.0003 \
  --batch_size 64 \
  --dropout_p 0.3 \
  --feedforward_dim 512 \
  --apply_augmentations True \
  --finetune_strategy 0 \
  --weights IMAGENET1K_V2 \
```

resnet50 suppports the following pretrained weights: IMAGENET1K_V1, IMAGENET1K_V2


## Finetuning vision transformer vit_b_16

Use the following command to start training:

```bash
python vision_transformer.py \
  --wandb_project INatClassifier \
  --wandb_entity your_wandb_username \
  --run_name run_name \
  --learning_rate 0.0003 \
  --batch_size 64 \
  --dropout_p 0.3 \
  --feedforward_dim 512 \
  --apply_augmentations True \
  --finetune_strategy 0 \
  --weights IMAGENET1K_SWAG_E2E_V1 \
```

resnet50 suppports the following pretrained weights: IMAGENET1K_V1, IMAGENET1K_SWAG_E2E_V1, IMAGENET1K_SWAG_LINEAR_V1
---
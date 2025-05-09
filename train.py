import wandb
from trainer import Trainer
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-wp", "--wandb_project", type=str, required=False, help="WandB project name", default=None)
    parser.add_argument("-we", "--wandb_entity", type=str, required=False, help="WandB entity", default=None)
    parser.add_argument("-rn", "--run_name", type=str, required=False, help="WandB run_name", default=None)
    
    parser.add_argument("--train_data_path", type=str, required=False, default="/speech/shoutrik/Database/INaturalist/inaturalist_12K/train")
    parser.add_argument("--valid_data_path", type=str, required=False, default="/speech/shoutrik/Database/INaturalist/inaturalist_12K/valid")
    parser.add_argument("--test_data_path", type=str, required=False, default="/speech/shoutrik/Database/INaturalist/inaturalist_12K/test")
    
    parser.add_argument('--input_channels', type=int, default=3)
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--num_channels', type=list, default=[32, 64, 128, 256, 512])
    parser.add_argument('--kernel_size', type=list, default=[7, 5, 3, 3, 3])
    parser.add_argument('--padding', type=list, default=[0, 0, 0, 0, 0])
    parser.add_argument('--stride', type=list, default=[1, 1, 1, 1, 1])
    parser.add_argument('--maxpool_kernel_size', type=int, default=2)
    parser.add_argument('--maxpool_padding', type=int, default=0)
    parser.add_argument('--maxpool_stride', type=int, default=2)
    parser.add_argument('--feedforward_dim', type=int, default=512)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--apply_maxpool', type=bool, default=True)
    parser.add_argument('--apply_batchnorm', type=bool, default=True)
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--dropout_p', type=float, default=0.3)
    parser.add_argument('--conv_activation_function', type=str, default='SiLU')
    parser.add_argument('--feedforward_activation_function', type=str, default='ReLU')
    parser.add_argument('--num_channels_multiplier', type=float, default=1.0)
    parser.add_argument('--apply_augmentations', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--max_epoch', type=int, default=15)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=424)
    
    parser.add_argument('--visualize_error', type=bool, default=True)


    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(args)
    if args.wandb_entity is not None and args.wandb_project is not None:
        logging = True
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

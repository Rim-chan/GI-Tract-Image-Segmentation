from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser



def get_main_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    arg = parser.add_argument
    arg("--base_dir", type=str, default="uw-madison-gi-tract-image-segmentation\\train", help="Train Data Directory")
    arg("--csv_path", type=str, default="uw-madison-gi-tract-image-segmentation\\train.csv", help="RLE Encoded Masks File")
    arg("--crop_size", type=int, default=224, help="centered crop size")
    arg("--batch_size", type=int, default=64, help="batch size")
    arg("--in_channels", type=int, default=1, help="Network Input Channels")
    arg("--out_channels", type=int, default=3, help="Network Output Channels")
    arg("--seed", type=int, default=26012022, help="Random Seed")
    arg("--num_workers", type=int, default=2, help="Number of DataLoader Workers")
    arg("--learning_rate", type=float, default=2e-3, help="Learning Rate")
    arg("--weight_decay", type=float, default=1e-5, help="Weight Decay")
    arg("--kernels", type=list, default=[[3, 3]] * 4, help="Convolution Kernels")
    arg("--strides", type=list, default=[[1, 1]] +  [[2, 2]] * 3, help="Convolution Strides")
    arg("--num_epochs", type=int, default=20, help="Number of Epochs")

    return parser.parse_args()
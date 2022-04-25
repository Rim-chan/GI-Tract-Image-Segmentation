from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

def get_main_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    arg = parser.add_argument
    arg("--base_dir", type=str, default="uw-madison-gi-tract-image-segmentation\\train", help="Train Data Directory")
    arg("--csv_path", type=str, default="uw-madison-gi-tract-image-segmentation\\train.csv", help="RLE Encoded Masks File")
    arg("--crop_size", type=int, default=256, help="centered crop size")
    arg("--batch_size", type=int, default=3, help="batch size")
    return parser.parse_args()
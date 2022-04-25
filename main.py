from dataloader import *
from args import *
import matplotlib.pyplot as plt




if __name__ == "__main__":

    args = get_main_args()
    dm = UWGITractDataModule(args.base_dir, args.csv_path, args.crop_size, args.batch_size)
    dm.setup()
    img, lbl = next(iter(dm.train_dataloader()))
    print(img.shape, lbl.shape)
    # dset = UWGITractDataset(args.base_dir, args.csv_path)
    # img, lbl= dset[4000]
    # print(img.shape, lbl.shape)
    # plt.imshow(stack[3], cmap="gray")
    # plt.show()
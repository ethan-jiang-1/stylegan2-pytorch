import sys
import argparse
from PIL import Image
import lmdb

from torchvision import datasets
from prepare_data import prepare

def xprepare():
    parser = argparse.ArgumentParser(description="Preprocess images for model training")
    parser.add_argument("--out", type=str, help="filename of the result lmdb dataset")
    parser.add_argument(
        "--size",
        type=str,
        default="128,256,512,1024",
        help="resolutions of images for the dataset",
    )
    parser.add_argument(
        "--n_worker",
        type=int,
        default=8,
        help="number of workers for preparing dataset",
    )
    parser.add_argument(
        "--resample",
        type=str,
        default="lanczos",
        help="resampling methods for resizing images",
    )
    parser.add_argument("path", type=str, help="path to the image dataset")

    args = parser.parse_args()

    resample_map = {"lanczos": Image.LANCZOS, "bilinear": Image.BILINEAR}
    resample = resample_map[args.resample]

    sizes = [int(s.strip()) for s in args.size.split(",")]

    print(f"Make dataset of image sizes:", ", ".join(str(s) for s in sizes))

    imgset = datasets.ImageFolder(args.path)

    with lmdb.open(args.out, map_size=1024 ** 4, readahead=False) as env:
        prepare(env, imgset, args.n_worker, sizes=sizes, resample=resample)

        print(env.stat())

    env_chk = lmdb.open(
        args.out,
        max_readers=32,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )

    with env_chk.begin(write=False) as txn:
        length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))
        print(length)

        count = 0
        for index in range(length):
            key = f'256-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)
            if img_bytes is not None:
                count += 1

        print("count in lmdb", count)


def fake_cmdline():
    cmd = "x_prepare_data.py "
    cmd += "--out {} ".format("dataset_planets")
    cmd += "--size {} ".format(256)
    cmd += "{} ".format("datasrc")
    cmd = cmd.strip()
    print(cmd)
    sys.argv = cmd.split(" ")
    return


if __name__ == "__main__":
    fake_cmdline()
    xprepare()

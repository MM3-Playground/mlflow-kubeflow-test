import random
from pathlib import Path

import cv2
import pandas as pd
from torch.utils.data import Dataset

import mlflow

import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils.cnorm import *
from utils.pilresize import *
from utils.FCRDCT import *
from utils.rearrange import *


class AnimeDataset(Dataset):

    def sampling(self, distribution, n_max):
        if self.n_c_samples is None:
            self.n_c_samples = n_max

        for label in distribution:
            ll = distribution[label]
            n_list = len(ll)

            if n_list >= self.n_c_samples:
                picked = random.sample(ll, self.n_c_samples)
            else:
                for _ in range(self.n_c_samples // n_list):
                    for i in ll:
                        self.input_image_paths.append(i)
                        self.labels.append(label)

                picked = random.sample(ll, self.n_c_samples % n_list)

            for p in picked:
                self.input_image_paths.append(p)
                self.labels.append(label)

    def __init__(
        self,
        global_rank,
        iut_paths_file,
        image_size,
        id,
        dct,
        n_c_samples=None,
        val=False,
        repo=None,
        commit=None,
        name=None,
    ):
        self.n_c_samples = n_c_samples
        self.val = val
        self.input_image_paths = []
        self.labels = []

        self.save_path = str(
            Path(iut_paths_file).resolve().parent
            / (
                "cond_paths_file_"
                + str(id)
                + ("_train" if not val else "_val")
                + ".txt"
            )
        )

        if "cond" not in iut_paths_file:
            distribution = {}
            n_max = 0

            with open(iut_paths_file, "r") as f:
                lines = f.readlines()
                for l in lines:
                    parts = l.rstrip().split("\t")
                    iut_path = parts[0]
                    label = int(parts[1])

                    distribution.setdefault(label, []).append(iut_path)
                    if len(distribution[label]) > n_max:
                        n_max = len(distribution[label])

            self.sampling(distribution, n_max)

            if global_rank == 0:
                with open(self.save_path, "w") as f:
                    for i in range(len(self.input_image_paths)):
                        f.write(
                            self.input_image_paths[i]
                            + "\t"
                            + str(self.labels[i])
                            + "\n"
                        )

                print(
                    "Final paths file (%s) for %s saved to %s"
                    % (("train" if not val else "val"), str(id), self.save_path)
                )

        else:
            print("Read from previous saved paths file %s" % iut_paths_file)

            with open(iut_paths_file, "r") as f:
                lines = f.readlines()
                for l in lines:
                    parts = l.rstrip().split("\t")
                    self.input_image_paths.append(parts[0])
                    self.labels.append(int(parts[1]))

        if global_rank == 0:
            mlflow.log_artifact(
                self.save_path if Path(self.save_path).exists() else iut_paths_file,
                "datasets",
            )

            dataframe = pd.DataFrame(
                {
                    "input_image_paths": self.input_image_paths,
                    "labels": self.labels,
                }
            )
            dataset = mlflow.data.from_pandas(
                dataframe,
                source=repo,
                digest=commit[:36],
                name=name + ("-train" if not val else "-val"),
            )
            mlflow.log_input(dataset, "train" if not val else "val")

            mlflow.log_params(
                {
                    "repo": repo,
                    "commit": commit,
                    "name": name,
                }
            )

        if not dct:
            self.transform_train = A.Compose(
                [
                    A.Normalize(mean=0.0, std=1.0),
                    A.HorizontalFlip(),
                    A.VerticalFlip(),
                    ToTensorV2(),
                ]
            )

            self.transform_val = A.Compose(
                [
                    A.Normalize(mean=0.0, std=1.0),
                    ToTensorV2(),
                ]
            )
        else:
            self.transform_train = A.Compose(
                [
                    A.Normalize(mean=0.0, std=1.0),
                    A.HorizontalFlip(),
                    A.VerticalFlip(),
                    ToTensorV2(),
                    DCT(p=1.0, log=True, factor=1),
                ]
            )

            self.transform_val = A.Compose(
                [
                    A.Normalize(mean=0.0, std=1.0),
                    ToTensorV2(),
                    DCT(p=1.0, log=True, factor=1),
                ]
            )

    def __getitem__(self, item):
        input_file_name = self.input_image_paths[item]
        try:
            iut = cv2.cvtColor(cv2.imread(input_file_name), cv2.COLOR_BGR2RGB)
        except Exception:
            print("Failed to load image {}".format(input_file_name))
            return None

        if iut is None:
            print("Failed to load image {}".format(input_file_name))
            return None

        if not self.val:
            iut = self.transform_train(image=iut)["image"]
        else:
            iut = self.transform_val(image=iut)["image"]

        return iut, self.labels[item]

    def __len__(self):
        return len(self.input_image_paths)

import logging
import re
from enum import Enum
from pathlib import Path
from typing import List, Optional

import cv2
import pandas as pd
import torch
import torchvision
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from src.utils.time import measure_time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "validate"
    TEST = "test"
    PROD = "production"


class AbstractCxrDataset(Dataset):
    def __init__(
        self,
        metadata_path: str,
        split_path: str,
        report_dir: str,
        img_dir: str,
        split: DatasetSplit = DatasetSplit.TRAIN,
    ):
        self._report_dir = Path(report_dir)
        self._img_dir = Path(img_dir)
        self._split = split

        self._metadata: pd.DataFrame = self._extract_metadata_for_split(
            metadata_path, split_path
        )
        self._img_paths = self._create_img_paths()
        self._report_paths = self._create_report_paths()

    def _extract_metadata_for_split(
        self, metadata_path: str, split_path: str
    ) -> pd.DataFrame:
        """Extract metadata from the metadata_path for suitable split type.
        This method should be implemented by the subclass.
        """
        metadata = pd.read_csv(metadata_path, dtype="str")
        split = pd.read_csv(split_path, dtype="str")

        metadata = metadata.merge(
            split, on=["dicom_id", "subject_id", "study_id"], how="inner"
        )
        return metadata[metadata["split"] == self._split.value]

    def _create_img_paths(self) -> List[Path]:
        img_paths = []

        def _create_path(row: pd.Series) -> Path:
            part_id = self.extract_subject_part(row["subject_id"])
            subject_id = row["subject_id"]
            study_id = row["study_id"]
            img_id = row["dicom_id"]
            return (
                self._img_dir
                / f"p{part_id}"
                / f"p{subject_id}"
                / f"s{study_id}"
                / f"{img_id}.jpg"
            )

        self._metadata.apply(
            lambda x: img_paths.append(_create_path(x)),
            axis=1,
        )

        return img_paths

    def _create_report_paths(self) -> List[Path]:
        report_paths = []

        def _create_path(row: pd.Series) -> Path:
            part_id = self.extract_subject_part(row["subject_id"])
            subject_id = row["subject_id"]
            study_id = row["study_id"]
            return (
                self._report_dir
                / f"p{part_id}"
                / f"p{subject_id}"
                / f"s{study_id}.txt"
            )

        self._metadata.apply(
            lambda x: report_paths.append(_create_path(x)),
            axis=1,
        )

        return report_paths

    def __len__(self):
        return len(self._metadata)

    def __getitem__(self, index):
        raise NotImplementedError

    def _load_and_process_text(self, text_path: Path) -> str:
        with open(text_path, "r") as f:
            text = f.read()
        return text

    def _load_and_process_img(self, img_path: Path) -> torch.Tensor:
        img = cv2.imread(img_path.as_posix())
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)

        return img

    @staticmethod
    def extract_subject_part(subject_id: str) -> str:
        """Extract part of the subject_id.
        For example, subject_id = 10000001, the part is 10.
        """
        assert re.match(
            r"1[0-9]{7}", subject_id
        ), f"Invalid subject_id: {subject_id}"

        return subject_id[:2]


class CxrDatasetVer1(AbstractCxrDataset):
    def __init__(
        self,
        metadata_path: str,
        split_path: str,
        report_dir: str,
        img_dir: str,
        tokenizer: Optional[AutoTokenizer] = None,
        split: DatasetSplit = DatasetSplit.TRAIN,
    ):
        super().__init__(metadata_path, split_path, report_dir, img_dir, split)

        if tokenizer is None:
            self._tokenizer = self._create_tokenizer()
        else:
            self._tokenizer = tokenizer
        self._transformer = self._create_split_transformer()

    def _create_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            "openai/clip-vit-base-patch32",
            truncation_side="left",
            padding_side="right",
            model_max_length=77,
        )

        return tokenizer

    def _create_split_transformer(self):
        if self._split.value == "train":
            self.transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToPILImage(),
                    torchvision.transforms.Resize(224),
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.RandomRotation(15),
                    torchvision.transforms.Grayscale(num_output_channels=1),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                        mean=(0.485,), std=(0.229,)
                    ),
                ]
            )
        else:
            self.transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToPILImage(),
                    torchvision.transforms.Resize(224),
                    torchvision.transforms.Grayscale(num_output_channels=1),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                        mean=(0.485,), std=(0.229,)
                    ),
                ]
            )

    def _load_and_process_img(self, img_path: Path) -> torch.Tensor:
        img = super()._load_and_process_img(img_path)
        img = self.transform(img)

        return img

    def _load_and_process_text(self, text_path: Path) -> torch.Tensor:
        txt = super()._load_and_process_text(text_path)
        tokens = self._tokenizer(
            txt,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )

        return tokens

    # @measure_time(logger=logger)
    def __getitem__(self, index):
        img = self._load_and_process_img(self._img_paths[index])
        txt = self._load_and_process_text(self._report_paths[index])

        return img, txt

"""Custom dataset definitions."""

import os
from pathlib import Path
from typing import Any
import zipfile

from PIL import Image
from torch.utils.data import Dataset, DataLoader


class VisionDataset(Dataset):
    """Custom dataset created from locally stored folder.

    To use this class, create a folder with the following structure:
    ```
    data/
    ├── class1/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── class2/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    └── ...
    ```

    Class names are automatically extracted from the folder names.
    """

    def __init__(
        self, *args, path: Path, transform: Any | None = None, rgb_only=False, **kwargs
    ):
        """Initialize the dataset.

        Args:
            path (Path): Path to the data folder.
            transform (Any | callable | None): Image transformation pipeline. Defaults to None. Typically,
                this is a composition of torchvision.transforms but can be any callable that accepts an image.
            rgb_only (bool, optional): If True, force convert all images to RGB. Defaults to False.
        """
        if not path.exists():
            raise Exception(f"Data path {path} does not exist.")

        self.paths = list(path.rglob("*.jpg"))
        self.transform = transform
        self.rgb_only = rgb_only

        if len(self.paths) == 0:
            raise Exception("No images found in the provided path.")

        self.classes = self._get_class_names(path)
        self.class_idx = {c: i for i, c in enumerate(self.classes)}

    def _get_class_names(self, path: Path) -> list[str]:
        """Create a list of class names based on directory structure.

        Args:
            path (Path): Path to the data folder.

        Returns:
            list[str]: List of class names.
        """
        classes = set()
        for dirname, _, _ in os.walk(path):
            if Path(dirname).name in ["train", "test", "validation"]:
                continue
            cls = Path(dirname).name
            classes.add(cls)
        return list(sorted(classes))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx) -> tuple[Any, int]:
        image = Image.open(self.paths[idx])
        label = self.class_idx[self.paths[idx].parent.name]
        # if image is non-RGB, convert it
        if image.mode != "RGB" and self.rgb_only:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label  # (X, y)
    

# TODO: test
def get_fruits(
        remote: str, localdir: str, fname: str = "archive.zip"
        ) -> dict[str, VisionDataset | None]:
    """Download fruits & vegetables image set and wrap it into custom train, eval and test
    datasets.
    
    Args:
        remote (str):
            URL to remote zip file containing image dataset.
        localdir (str):
            Local directory path used to store all data. Must be a valid path.
        fname (str):
            Filename convention for downloaded archive. Default is archive.zip.

    Returns:
    ```
        {
            "train_dataset" : VisionDataset,
            "val_dataset": VisionDataset,
            "test_dataset": VisionDataset
        }
    
    ```
    Note that return values for validation and test datasets can be None if they are not provided
    by data download.
    """
    try:
        localdir = Path(localdir)
    except:
        raise Exception(f"Local path {localdir} is invalid.")
    # TODO: verify remote
    # TODO: download from remote

    # for now assume that zip was pulled to localdir successfully
    f = localdir / fname
    if not f.exists(): 
        raise Exception(f"File {f} does not exist.")
    with zipfile.ZipFile(f, "r") as arch:
        arch.extractall(localdir)

    ds_train = VisionDataset(path=(localdir / "train"))
    ds_val = VisionDataset(path=(localdir / "validation"))
    ds_test = VisionDataset(path=(localdir / "test"))

    return {
        "train_dataset": ds_train,
        "val_dataset": ds_val,
        "test_dataset": ds_test
    }
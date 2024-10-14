"""Custom dataset definitions."""

import os
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset


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
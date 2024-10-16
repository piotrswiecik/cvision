"""Custom dataset definitions."""

import requests
import os
from pathlib import Path
import threading
import time
from typing import Any
import zipfile

from PIL import Image
from tqdm import tqdm
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
        if self.rgb_only and image.mode != "RGB":
            image = image.convert("RGB")
        # an addtional protection if number of channels is not 3
        if self.rgb_only and len(image.getbands()) != 3:
            raise Exception("Non RGB") # TODO: handle this case - return none possibly
        if self.transform is not None:
            image = self.transform(image)
        return image, label  # (X, y)
    
    def set_transform(self, transform: Any):
        self.transform = transform
    

# TODO: test
def get_fruits(
        localdir: str,
        remote: str = "https://www.kaggle.com/api/v1/datasets/download/kritikseth/fruit-and-vegetable-image-recognition",
        fname: str = "archive.zip",
        force: bool = True
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
        force (bool):
            Redownload data even if it already exists.
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
    def _wrap_data():
        ds_train = VisionDataset(path=(localdir / "train"), rgb_only=True)
        ds_val = VisionDataset(path=(localdir / "validation"), rgb_only=True)
        ds_test = VisionDataset(path=(localdir / "test"), rgb_only=True)

        return {
            "train_dataset": ds_train,
            "val_dataset": ds_val,
            "test_dataset": ds_test
        }

    # TODO: verify remote
    try:
        localdir = Path(localdir)
    except:
        raise Exception(f"Local path {localdir} is invalid.")
    
    # check if data already exists
    if not force:
        # TODO: more detailed verification logic
        if all((Path.exists(localdir / "train"), Path.exists(localdir / "validation"), Path.exists(localdir / "test"))):
            print("Data already exists.")
            return _wrap_data()

    # download in multithreaded mode
    threaded_download(remote, localdir, fname, threads=os.cpu_count())

    # unzip
    # other compression formats not supported yet
    f = localdir / fname
    if not f.exists(): 
        raise Exception(f"File {f} does not exist.")
    with zipfile.ZipFile(f, "r") as arch:
        print(f"Extracting dataset...")
        arch.extractall(localdir)

    return _wrap_data()


# TODO: test
def threaded_download(remote: str, localdir: str, fname: str = "archive.zip", threads: int = 1):
    """Helper method. Download remote file using n threads as workers."""
    lpath: Path | None = None
    try:
        lpath = Path(localdir)
    except:
        raise Exception(f"Local path {localdir} is invalid.")
    
    if not lpath.exists():
        os.makedirs(lpath.absolute())

    # kaggle download links typically don't support HEAD requests
    # so we use a workaround here
    preflight = requests.get(remote, headers={"Range": "bytes=0-0"})
    if "Content-Type" not in preflight.headers:
        raise Exception(f"Invalid header - Content-Type not specified.")
    ctype = preflight.headers.get("Content-Type")
    if ctype != "application/zip":
        raise Exception(f"Content type {ctype} is currently not supported.")
    if "Content-Range" not in preflight.headers:
        raise Exception(f"Remote error - unable to get content length.")
    length_header = preflight.headers["Content-Range"] # bytes 0-0/n
    if not "bytes" in length_header or not "/" in length_header:
        raise Exception(f"Invalid preflight header: {length_header}")
    length = length_header.split("/")[-1]
    try:
        length = int(length)
    except:
        raise Exception(f"Content length {length} is invalid.")

    # create byte pointer table
    part_size = length // threads
    parts = [(i * part_size, (i + 1) * part_size - 1) for i in range(threads)]
    parts[-1] = (parts[-1][0], length)

    progress = tqdm(total=length, unit="B", unit_scale=True, desc="Downloading dataset...")

    def _get_chunk(remote: str, localdir: Path, start: int, stop: int, idx: int):
        """Helper function. Isolate file chunk ands save it to partial file."""
        headers = {"Range": f"bytes={start}-{stop}"}
        partial_path = localdir / f"part_{idx}"
        try:
            res = requests.get(remote, headers=headers, stream=True)
            res.raise_for_status()
            total_downloaded = 0
            timer_0 = time.time()
            with open(partial_path, "wb") as part:
                for chunk in res.iter_content(chunk_size=2**13):
                    if chunk:
                        part.write(chunk)
                        total_downloaded += len(chunk)
                        progress.update(len(chunk))
            timer_1 = time.time()
            speed = total_downloaded / (timer_1 - timer_0)
        except Exception:
            raise Exception(f"Download error...")
        
    thread_pool = []
    for idx, (start, end) in enumerate(parts):
        thread = threading.Thread(
            group=None,
            target=_get_chunk,
            name=f"dl_thread_{idx}",
            args=(remote, lpath, start, end, idx)
        )
        thread_pool.append(thread)
        thread.start()
        print(f"Worker thread #{idx} started.")

    for thread in thread_pool:
        thread.join()

    # recombine file parts
    out_file = lpath / fname
    try:
        with open(out_file, "wb") as f, tqdm(total=length, unit="B", unit_scale=True, desc="Merging...") as pb:
            for idx in range(threads):
                partial_path = lpath / f"part_{idx}"
                with open(partial_path, "rb") as part:
                    while True:
                        data = part.read(1024*1024)
                        if not data:
                            break
                        f.write(data)
                        pb.update(len(data))
                os.remove(partial_path)
        print(f"Done!")
    except:
        raise Exception(f"Error while merging partial files!")

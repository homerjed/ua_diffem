from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import gzip
import shutil
import ssl
import struct
import urllib.error
import urllib.request

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from datasets import Dataset, DatasetDict
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jaxtyping import Array


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS_ROOT = REPO_ROOT / "datasets"
RAW_DATASETS_ROOT = DATASETS_ROOT / "raw_datasets"


MNIST_MIRROR_URLS = {
    "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "test_images": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}


def count_parameters(model) -> int:
    return sum(jax.tree.leaves(jax.tree.map(jnp.size, eqx.filter(model, eqx.is_array))))


@dataclass(frozen=True)
class DataParallelSharding:
    mesh: Mesh
    replicated: NamedSharding


def make_data_parallel_sharding() -> DataParallelSharding | None:
    devices = np.asarray(jax.local_devices())
    if devices.size <= 1:
        return None
    mesh = Mesh(devices, ("data",))
    return DataParallelSharding(
        mesh=mesh,
        replicated=NamedSharding(mesh, P()),
    )


def shard_replicated_tree(tree, sharding: DataParallelSharding | None):
    if sharding is None:
        return tree

    arrays, static = eqx.partition(tree, eqx.is_array)
    arrays = jax.tree_util.tree_map(
        lambda leaf: jax.device_put(leaf, sharding.replicated),
        arrays,
    )
    return eqx.combine(arrays, static)


def shard_batch(batch: Array, sharding: DataParallelSharding | None) -> Array:
    if sharding is None:
        return batch
    batch_spec = P("data", *([None] * (batch.ndim - 1)))
    return jax.device_put(batch, NamedSharding(sharding.mesh, batch_spec))


def make_optimizer(config) -> optax.GradientTransformation:
    return optax.chain(
        optax.clip_by_global_norm(config.grad_clip),
        optax.adamw(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        ),
    )


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"Using cached file: {destination}")
        return

    print(f"Downloading {url} -> {destination}")
    temp_destination = destination.with_suffix(destination.suffix + ".part")

    def _download_with_context(context: ssl.SSLContext | None, note: str | None = None) -> None:
        if note is not None:
            print(note)
        if temp_destination.exists():
            temp_destination.unlink()
        with urllib.request.urlopen(url, context=context) as response, open(temp_destination, "wb") as handle:
            shutil.copyfileobj(response, handle)
        temp_destination.replace(destination)

    try:
        _download_with_context(None)
        return
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        if "CERTIFICATE_VERIFY_FAILED" not in str(reason):
            if temp_destination.exists():
                temp_destination.unlink()
            raise

    try:
        import certifi

        certifi_context = ssl.create_default_context(cafile=certifi.where())
        _download_with_context(
            certifi_context,
            note="Retrying download with certifi CA bundle.",
        )
        return
    except Exception:
        pass

    _download_with_context(
        ssl._create_unverified_context(),
        note="Retrying download with unverified SSL context.",
    )


def _read_mnist_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        magic, n_images, n_rows, n_cols = struct.unpack(">IIII", handle.read(16))
        if magic != 2051:
            raise ValueError(f"Unexpected MNIST image magic number: {magic}")
        data = np.frombuffer(handle.read(), dtype=np.uint8)
    return data.reshape(n_images, n_rows, n_cols)


def _read_mnist_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        magic, n_labels = struct.unpack(">II", handle.read(8))
        if magic != 2049:
            raise ValueError(f"Unexpected MNIST label magic number: {magic}")
        data = np.frombuffer(handle.read(), dtype=np.uint8)
    return data.reshape(n_labels)


def load_local_mnist_dataset() -> DatasetDict:
    data_root = RAW_DATASETS_ROOT / "mnist"
    file_map = {
        key: data_root / Path(url).name
        for key, url in MNIST_MIRROR_URLS.items()
    }

    for key, url in MNIST_MIRROR_URLS.items():
        _download_file(url, file_map[key])

    train_images = _read_mnist_images(file_map["train_images"])
    train_labels = _read_mnist_labels(file_map["train_labels"])
    test_images = _read_mnist_images(file_map["test_images"])
    test_labels = _read_mnist_labels(file_map["test_labels"])

    return DatasetDict(
        train=Dataset.from_dict({"image": train_images, "label": train_labels}),
        test=Dataset.from_dict({"image": test_images, "label": test_labels}),
    )

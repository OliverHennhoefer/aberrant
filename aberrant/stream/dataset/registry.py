"""Dataset registry system for anomaly detection benchmarks.

This module provides a centralized registry of available datasets with rich metadata,
following the design patterns established in the anomaly detection research community.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType


@dataclass(frozen=True)
class DatasetInfo:
    """Metadata for an anomaly detection dataset.

    Attributes:
        name: Human-readable name of the dataset
        description: Brief description of the dataset and its characteristics
        filename: Name of the file in the GitHub release (without extension)
        n_samples: Total number of samples in the dataset
        n_features: Number of features/dimensions
        anomaly_rate: Proportion of anomalous samples (0.0 to 1.0)
        source: Original source or reference for the dataset
        category: Dataset category (e.g., 'medical', 'security', 'benchmark')
        sha256: Trusted SHA256 checksum for release artifact validation
    """

    name: str
    description: str
    filename: str
    n_samples: int
    n_features: int
    anomaly_rate: float
    source: str
    category: str
    sha256: str


class Dataset(Enum):
    """Available datasets for anomaly detection experiments.

    This enumeration provides all built-in datasets that can be loaded
    using the load() function. Each dataset is preprocessed for anomaly
    detection tasks with normal and anomalous samples.

    Examples:
        ```python
        from aberrant.stream.dataset import Dataset, get_dataset_info

        info = get_dataset_info(Dataset.FRAUD)
        assert info.n_features == 30
        ```
    """

    # Existing datasets (migrated from stream module)
    FRAUD = "fraud"
    SHUTTLE = "shuttle"
    # Medical/Health datasets
    ANNTHYROID = "annthyroid"
    BREAST = "breast"
    HEPATITIS = "hepatitis"
    LYMPHOGRAPHY = "lymphography"
    MAMMOGRAPHY = "mammography"
    THYROID = "thyroid"

    # Security/Network datasets
    BACKDOOR = "backdoor"
    HTTP = "http"
    SMTP = "smtp"
    IONOSPHERE = "ionosphere"

    # Machine Learning benchmark datasets
    CARDIO = "cardio"
    COVER = "cover"
    GLASS = "glass"
    LETTER = "letter"
    MAGIC_GAMMA = "magic_gamma"
    MNIST = "mnist"
    MUSK = "musk"
    OPTDIGITS = "optdigits"
    PAGEBLOCKS = "pageblocks"
    PENDIGITS = "pendigits"
    SATIMAGE2 = "satimage2"
    STAMPS = "stamps"
    DONORS = "donors"


# Central registry of dataset metadata
_DATASET_REGISTRY: dict[Dataset, DatasetInfo] = {
    Dataset.FRAUD: DatasetInfo(
        name="Credit Card Fraud Detection",
        description="European credit card fraud dataset with anonymized features",
        filename="fraud",
        n_samples=284807,
        n_features=30,
        anomaly_rate=0.00173,  # 0.173% fraud cases
        source="Kaggle Credit Card Fraud Dataset",
        category="financial",
        sha256="353b9a156b3f20e0739fafacdff76ef5cad1eeb7323be0ae71d384bf0e2594a6",
    ),
    Dataset.SHUTTLE: DatasetInfo(
        name="Space Shuttle",
        description="NASA space shuttle dataset for anomaly detection",
        filename="shuttle",
        n_samples=58000,
        n_features=9,
        anomaly_rate=0.07,
        source="UCI Machine Learning Repository",
        category="engineering",
        sha256="26900b0f6be45d9ddb7a402c9b2f956ceda42d23a8ca4cf4406b8bc1352823d1",
    ),
    # Medical/Health datasets
    Dataset.ANNTHYROID: DatasetInfo(
        name="Thyroid Disease (ANN)",
        description="Thyroid disease detection dataset for neural networks",
        filename="annthyroid",
        n_samples=7200,
        n_features=21,
        anomaly_rate=0.075,
        source="UCI Machine Learning Repository",
        category="medical",
        sha256="e738f7c58a8740d2c4aac4c71ff49ce5e1d1aa16f44178135fc9c09df05fad1b",
    ),
    Dataset.BREAST: DatasetInfo(
        name="Breast Cancer Wisconsin",
        description="Breast cancer diagnostic dataset with cell characteristics",
        filename="breast_w",
        n_samples=569,
        n_features=30,
        anomaly_rate=0.373,  # Malignant cases
        source="UCI Machine Learning Repository",
        category="medical",
        sha256="4f37ce07dbc25ddad8178fdc6d664ab46d2d0000ee28c2787c3c7ff3c03ef9f6",
    ),
    Dataset.HEPATITIS: DatasetInfo(
        name="Hepatitis",
        description="Hepatitis patient data for outcome prediction",
        filename="hepatitis",
        n_samples=155,
        n_features=19,
        anomaly_rate=0.206,
        source="UCI Machine Learning Repository",
        category="medical",
        sha256="4a87a2f7d8013e5bfde63294fae61ec7a612322de72f338a7f9491749025ad98",
    ),
    Dataset.LYMPHOGRAPHY: DatasetInfo(
        name="Lymphography",
        description="Lymphatic system diagnosis dataset",
        filename="lymphography",
        n_samples=148,
        n_features=18,
        anomaly_rate=0.041,
        source="UCI Machine Learning Repository",
        category="medical",
        sha256="dad977b04f4ea65b78ccd0e9f76b45175bad605165b37e63f3cc177845e67037",
    ),
    Dataset.MAMMOGRAPHY: DatasetInfo(
        name="Mammography",
        description="Mammographic screening for breast cancer detection",
        filename="mammography",
        n_samples=11183,
        n_features=6,
        anomaly_rate=0.023,
        source="UCI Machine Learning Repository",
        category="medical",
        sha256="a0308ac9712d78321073b1c2dc98dc37665c0f35c1cc33019416fe8f8d7de7ea",
    ),
    Dataset.THYROID: DatasetInfo(
        name="Thyroid Disease",
        description="Thyroid function diagnosis dataset",
        filename="thyroid",
        n_samples=3772,
        n_features=6,
        anomaly_rate=0.025,
        source="UCI Machine Learning Repository",
        category="medical",
        sha256="8de161b3f235b7647bb0b39a836d983ff61d67bf234edac461da2d71e889cb87",
    ),
    # Security/Network datasets
    Dataset.BACKDOOR: DatasetInfo(
        name="Network Backdoor",
        description="Network intrusion detection - backdoor attacks",
        filename="backdoor",
        n_samples=2329,
        n_features=196,
        anomaly_rate=0.028,
        source="KDD Cup 1999 Network Intrusion",
        category="security",
        sha256="164481cd9688306b0742c25e996c876ba561ee975629b9fdb53f8ee51b1ed911",
    ),
    Dataset.HTTP: DatasetInfo(
        name="HTTP Network Traffic",
        description="HTTP-based network anomaly detection",
        filename="http",
        n_samples=567498,
        n_features=3,
        anomaly_rate=0.004,
        source="KDD Cup 1999 Network Intrusion",
        category="security",
        sha256="9d98710091551587f4740cfc851a6b064fff6319a337c2dc77ee6da4bcc2e02c",
    ),
    Dataset.SMTP: DatasetInfo(
        name="SMTP Email Traffic",
        description="SMTP protocol anomaly detection",
        filename="smtp",
        n_samples=95156,
        n_features=3,
        anomaly_rate=0.0003,
        source="KDD Cup 1999 Network Intrusion",
        category="security",
        sha256="d2569fc9bc12115419e3f5b37d19921e10e6c26e5ad1e61064cfb0fbc402dfdd",
    ),
    Dataset.IONOSPHERE: DatasetInfo(
        name="Ionosphere Radar",
        description="Ionosphere radar signal classification",
        filename="ionosphere",
        n_samples=351,
        n_features=33,
        anomaly_rate=0.359,
        source="UCI Machine Learning Repository",
        category="physics",
        sha256="bca46cb0cfd1bf9ddef6cde5ad2215d54fb0189508d0d4c4eb98e788cad69001",
    ),
    # ML Benchmark datasets
    Dataset.CARDIO: DatasetInfo(
        name="Cardiovascular Disease",
        description="Cardiovascular disease dataset with patient vitals",
        filename="cardio",
        n_samples=1831,
        n_features=21,
        anomaly_rate=0.096,
        source="Cardiovascular Disease Dataset",
        category="medical",
        sha256="bc5009bde6930b4d298b594f31b34485a9a17210cff0e8fb00a8c3068b885a2d",
    ),
    Dataset.COVER: DatasetInfo(
        name="Forest Cover Type",
        description="Forest cover type classification (class 4 as anomaly)",
        filename="cover",
        n_samples=286048,
        n_features=10,
        anomaly_rate=0.009,
        source="UCI Machine Learning Repository",
        category="environmental",
        sha256="945a097c3542ae81a551bf1c06dcad78a755d94f84fcaf1d60687e42d1a656df",
    ),
    Dataset.GLASS: DatasetInfo(
        name="Glass Identification",
        description="Glass type identification (float processed as anomaly)",
        filename="glass",
        n_samples=214,
        n_features=9,
        anomaly_rate=0.042,
        source="UCI Machine Learning Repository",
        category="materials",
        sha256="aa7a8db1d475ba01a10488af454bc15220a3d5951b010b03b7020567d8152cdc",
    ),
    Dataset.LETTER: DatasetInfo(
        name="Letter Recognition",
        description="Letter recognition (vowels vs consonants)",
        filename="letter",
        n_samples=1600,
        n_features=32,
        anomaly_rate=0.025,
        source="UCI Machine Learning Repository",
        category="vision",
        sha256="f986d3da80cf1fa2aea83a5ebe73b7d4800e0d4c01fdd1ae074b58d20f10d039",
    ),
    Dataset.MAGIC_GAMMA: DatasetInfo(
        name="MAGIC Gamma Telescope",
        description="Gamma ray detection in telescope data",
        filename="magic_gamma",
        n_samples=19020,
        n_features=10,
        anomaly_rate=0.352,
        source="UCI Machine Learning Repository",
        category="astronomy",
        sha256="0be375984d7af9ea7a63b12206c240840d8b7daa3607021c5fb8756407dd86f1",
    ),
    Dataset.MNIST: DatasetInfo(
        name="MNIST Handwritten Digits",
        description="MNIST digit recognition (digit 0 as normal, others anomalous)",
        filename="mnist",
        n_samples=7603,
        n_features=100,
        anomaly_rate=0.092,
        source="MNIST Database",
        category="vision",
        sha256="be7706272325fdd3592c1be555ae1a82f916a2f8545f4f8cbaa548ab102cc44c",
    ),
    Dataset.MUSK: DatasetInfo(
        name="Musk Molecules",
        description="Musk vs non-musk molecule classification",
        filename="musk",
        n_samples=3062,
        n_features=166,
        anomaly_rate=0.032,
        source="UCI Machine Learning Repository",
        category="chemistry",
        sha256="2ed3f4392197b76a1e59116ff880c7720a4b026cf4842073a53b56fdea265017",
    ),
    Dataset.OPTDIGITS: DatasetInfo(
        name="Optical Digit Recognition",
        description="Optical digit recognition dataset",
        filename="optdigits",
        n_samples=5216,
        n_features=64,
        anomaly_rate=0.029,
        source="UCI Machine Learning Repository",
        category="vision",
        sha256="60de17a375a278ba220a7849727d998aa02b5e2f9fd17732a004bba28de4656e",
    ),
    Dataset.PAGEBLOCKS: DatasetInfo(
        name="Page Layout Analysis",
        description="Document page layout block classification",
        filename="pageBlocks",
        n_samples=5473,
        n_features=10,
        anomaly_rate=0.107,
        source="UCI Machine Learning Repository",
        category="document",
        sha256="10a9b4d5bb5d3b9083caab4c4062f665610eada703c3d927bf7c34b1f2d0d223",
    ),
    Dataset.PENDIGITS: DatasetInfo(
        name="Pen-Based Recognition",
        description="Pen-based handwritten digit recognition",
        filename="pendigits",
        n_samples=6870,
        n_features=16,
        anomaly_rate=0.023,
        source="UCI Machine Learning Repository",
        category="vision",
        sha256="c4936b89dcdad59d303020e492f763a0f54170080d787c77497609acbb9c2d79",
    ),
    Dataset.SATIMAGE2: DatasetInfo(
        name="Satellite Image (Landsat)",
        description="Satellite image classification (red soil as anomaly)",
        filename="satimage2",
        n_samples=5803,
        n_features=36,
        anomaly_rate=0.013,
        source="UCI Machine Learning Repository",
        category="remote_sensing",
        sha256="1b056e357f801b8b20c3f380a30224a3c4088d7ed764ec5a24828dc9b1918d6f",
    ),
    Dataset.STAMPS: DatasetInfo(
        name="Stamp Verification",
        description="Postage stamp verification dataset",
        filename="stamps",
        n_samples=340,
        n_features=9,
        anomaly_rate=0.091,
        source="Postage Stamp Dataset",
        category="verification",
        sha256="99fe739447284f613fa54bf2dccdb51c5120175ce6260adec6dad77576c42509",
    ),
    Dataset.DONORS: DatasetInfo(
        name="Blood Donors",
        description="Blood donation prediction dataset",
        filename="donors",
        n_samples=748,
        n_features=4,
        anomaly_rate=0.239,
        source="UCI Machine Learning Repository",
        category="medical",
        sha256="5824ec854465f2763b4bfe3da1ae9f25ed5433b5c6e265d42e37da4f82a58679",
    ),
}

# Public read-only view. DatasetInfo values are frozen as well, so callers
# cannot mutate trusted checksums or other process-wide registry metadata.
DATASET_REGISTRY: Mapping[Dataset, DatasetInfo] = MappingProxyType(_DATASET_REGISTRY)


def get_dataset_info(dataset: Dataset) -> DatasetInfo:
    """Get metadata information for a specific dataset.

    Args:
        dataset: Dataset enum value

    Returns:
        DatasetInfo object with metadata

    Raises:
        KeyError: If dataset is not found in registry
    """
    if dataset not in DATASET_REGISTRY:
        raise KeyError(f"Dataset {dataset} not found in registry")
    return DATASET_REGISTRY[dataset]


def list_available() -> dict[str, DatasetInfo]:
    """List all available datasets with their metadata.

    Returns:
        Dictionary mapping dataset names to DatasetInfo objects
    """
    return {dataset.value: info for dataset, info in DATASET_REGISTRY.items()}


def list_by_category(category: str) -> dict[str, DatasetInfo]:
    """List datasets by category.

    Args:
        category: Category to filter by (e.g., 'medical', 'security', 'benchmark')

    Returns:
        Dictionary of datasets in the specified category
    """
    return {
        dataset.value: info
        for dataset, info in DATASET_REGISTRY.items()
        if info.category == category
    }


def get_categories() -> list[str]:
    """Get all available dataset categories.

    Returns:
        List of unique category names
    """
    return sorted({info.category for info in DATASET_REGISTRY.values()})

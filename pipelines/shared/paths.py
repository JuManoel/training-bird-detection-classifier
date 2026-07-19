"""Canonical project and artifact paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def get_project_root() -> Path:
    """Return the repository root (parent of ``pipelines/``)."""
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ProjectPaths:
    """Filesystem layout shared by every pipeline."""

    root: Path

    @classmethod
    def from_root(cls, root: Path | None = None) -> ProjectPaths:
        return cls(root=root or get_project_root())

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def artifacts(self) -> Path:
        return self.artifacts_root

    @property
    def artifacts_root(self) -> Path:
        return self.root / "artifacts"

    @property
    def images_raw(self) -> Path:
        return self.artifacts_root / "images" / "raw"

    @property
    def manifest(self) -> Path:
        return self.artifacts_root / "manifest.csv"

    @property
    def coverage_json(self) -> Path:
        return self.artifacts_root / "coverage.json"

    @property
    def coverage_csv(self) -> Path:
        return self.artifacts_root / "coverage.csv"

    @property
    def species_catalog(self) -> Path:
        return self.data / "colombia_species_catalog.json"

    @property
    def dataset(self) -> Path:
        """Legacy alias → detection dataset root."""
        return self.dataset_detect

    @property
    def dataset_detect(self) -> Path:
        return self.artifacts_root / "dataset" / "detect"

    @property
    def dataset_classify(self) -> Path:
        return self.artifacts_root / "dataset" / "classify"

    @property
    def dataset_images(self) -> Path:
        return self.dataset_detect / "images"

    @property
    def dataset_labels(self) -> Path:
        return self.dataset_detect / "labels"

    @property
    def dataset_rejected(self) -> Path:
        return self.dataset_detect / "rejected"

    @property
    def data_yaml(self) -> Path:
        return self.dataset_detect / "data.yaml"

    @property
    def class_names(self) -> Path:
        return self.dataset_detect / "classes.txt"

    @property
    def classify_manifest(self) -> Path:
        return self.dataset_classify / "crops_manifest.csv"

    @property
    def runs(self) -> Path:
        return self.artifacts_root / "runs"

    @property
    def train_run(self) -> Path:
        return self.runs / "train"

    @property
    def train_cls_run(self) -> Path:
        return self.runs / "train_cls"

    def classifier_run(self, architecture: str) -> Path:
        return self.train_cls_run / architecture

    def classifier_best(self, architecture: str) -> Path:
        return self.classifier_run(architecture) / "best.pt"

    @property
    def best_checkpoint(self) -> Path:
        return self.train_run / "best.pt"

    @property
    def predict_out(self) -> Path:
        return self.runs / "predict"

    @property
    def default_csv(self) -> Path:
        matches = sorted(self.data.glob("ML__*_photo_CO-CAL.csv"))
        if not matches:
            raise FileNotFoundError(
                f"No Macaulay CSV found under {self.data} (expected ML__*_photo_CO-CAL.csv)"
            )
        return matches[-1]

    @property
    def default_download_csv(self) -> Path:
        path = self.data / "aves_descarga_v2.csv"
        if not path.exists():
            raise FileNotFoundError(f"Download CSV not found: {path}")
        return path

    @property
    def species_file(self) -> Path:
        return self.data / "spicies.txt"

    def ensure_dirs(self) -> None:
        for path in (
            self.images_raw,
            self.dataset_images / "train",
            self.dataset_images / "val",
            self.dataset_labels / "train",
            self.dataset_labels / "val",
            self.dataset_rejected,
            self.dataset_classify / "train",
            self.dataset_classify / "val",
            self.train_run / "plots",
            self.train_cls_run,
            self.predict_out,
        ):
            path.mkdir(parents=True, exist_ok=True)

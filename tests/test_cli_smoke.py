"""CLI smoke tests (parser only — no downloads / GPU)."""

from pipelines.download_data.cli import build_parser as build_download
from pipelines.extract_bb.cli import build_parser as build_extract
from pipelines.predict_model.cli import build_parser as build_predict
from pipelines.train_classifier.cli import build_parser as build_train_cls
from pipelines.train_model.cli import build_parser as build_train


def test_download_parser_defaults():
    p = build_download()
    args = p.parse_args([])
    assert args.max_per_species == 2000
    assert args.min_images == 125
    assert args.target_images == 500
    assert args.no_fetch_inat is False
    assert args.no_fetch_gbif is False


def test_download_parser_disable_apis():
    p = build_download()
    args = p.parse_args(["--no-fetch-inat", "--no-fetch-gbif"])
    assert args.no_fetch_inat is True
    assert args.no_fetch_gbif is True


def test_extract_parser_crop_thresholds():
    p = build_extract()
    args = p.parse_args([])
    assert args.crop_size == 256
    assert args.min_images == 125
    assert args.target_images == 500
    assert args.max_per_species == 1000


def test_train_cls_parser_architectures():
    p = build_train_cls()
    args = p.parse_args(["--architecture", "all"])
    assert args.architecture == "all"
    assert args.imgsz == 256


def test_predict_parser_compare():
    p = build_predict()
    args = p.parse_args(["--source", "x.jpg", "--compare"])
    assert args.compare is True


def test_train_detect_parser():
    p = build_train()
    args = p.parse_args([])
    assert args.model == "yolo26x.pt"

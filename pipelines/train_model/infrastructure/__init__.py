"""Train infrastructure package."""


def run_train(*args, **kwargs):
    from pipelines.train_model.infrastructure.yolo_trainer import run_train as _run

    return _run(*args, **kwargs)


__all__ = ["run_train"]

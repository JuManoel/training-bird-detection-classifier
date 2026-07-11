"""Predict application package."""


def run_predict(*args, **kwargs):
    from pipelines.predict_model.application.predict import run_predict as _run

    return _run(*args, **kwargs)


__all__ = ["run_predict"]

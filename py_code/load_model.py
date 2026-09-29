from importlib import import_module

def train_model(model_name: str):

    return import_module(f"py_code.models.{model_name}_train_model")


def test_model(model_name: str):

    return import_module(f"py_code.models.{model_name}_test_model")

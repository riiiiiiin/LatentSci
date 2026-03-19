import importlib
from typing import Any
import inspect
from config import ModelConfig

def _get_registry(target: str) -> Any:
    """
    Get a dict from a string representation of its full name.
    """
    module_path, obj_name = target.rsplit(".", 1)
    module = importlib.import_module(module_path)
    obj = getattr(module, obj_name)
    return obj

def get_domain_specific_func(func_name: str, domain: str = None) -> callable:
    """
    Get a function from the domain-specific module with its last name.
    """
    if domain is None:
        domain = ModelConfig.DOMAIN
    target = f'{domain}.registry.registry'
    result = _get_registry(target)[func_name]
    if not callable(result):
        raise TypeError(f'{target}[{func_name}] is not a callable')
    return result
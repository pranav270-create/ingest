import importlib
import pkgutil
from pathlib import Path


class RegistryBase:
    _registry: dict = {}

    @classmethod
    def autodiscover(cls, package_name: str) -> None:
        """
        Automatically discover and import all modules in a package.

        Usage:
        RegistryBase.autodiscover('src.prompts')
        """
        package = importlib.import_module(package_name)
        package_path = Path(package.__file__).parent

        for _, module_name, _ in pkgutil.iter_modules([str(package_path)]):
            full_module_name = f"{package_name}.{module_name}"
            importlib.import_module(full_module_name)
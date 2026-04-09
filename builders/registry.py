"""
Registry pattern for extensible component registration
Based on ViWordFormer architecture
"""

from typing import Any, Dict, Iterable, Iterator, Tuple


class Registry(Iterable[Tuple[str, Any]]):
    """
    Registry for name -> object mapping, supporting third-party custom modules.
    
    Example:
        >>> PRETRAIN_TASKS = Registry('PRETRAIN_TASK')
        >>>
        >>> @PRETRAIN_TASKS.register()
        >>> class MLMPretraining():
        ...     pass
    """

    def __init__(self, name: str) -> None:
        """Initialize registry with a name"""
        self._name: str = name
        self._obj_map: Dict[str, Any] = {}

    def _do_register(self, name: str, obj: Any) -> None:
        """Register an object"""
        assert (
            name not in self._obj_map
        ), f"Object '{name}' already registered in '{self._name}' registry!"
        self._obj_map[name] = obj

    def register(self, obj: Any = None) -> Any:
        """
        Register an object. Can be used as decorator or function.
        
        Args:
            obj: Object to register (uses __name__)
        """
        if obj is None:
            # Used as a decorator
            def deco(func_or_class: Any) -> Any:
                name = func_or_class.__name__
                self._do_register(name, func_or_class)
                return func_or_class
            return deco

        # Used as function call
        name = obj.__name__
        self._do_register(name, obj)
        return obj

    def get(self, name: str) -> Any:
        """Get registered object by name"""
        ret = self._obj_map.get(name)
        if ret is None:
            raise KeyError(
                f"No object named '{name}' in '{self._name}' registry! "
                f"Available: {list(self._obj_map.keys())}"
            )
        return ret

    def __contains__(self, name: str) -> bool:
        """Check if name is registered"""
        return name in self._obj_map

    def __iter__(self):
        """Iterate over registered objects"""
        return iter(self._obj_map.items())

    def __len__(self):
        """Get number of registered objects"""
        return len(self._obj_map)

    def __repr__(self) -> str:
        """String representation"""
        return f"{self._name} Registry with {len(self._obj_map)} objects"

    def list_names(self):
        """List all registered names"""
        return list(self._obj_map.keys())


# Global registries for pretraining components
META_ARCHITECTURE = Registry("ARCHITECTURE")
META_TOKENIZER = Registry("TOKENIZER")
META_DATASET = Registry("DATASET")
META_PRETRAIN_TASK = Registry("PRETRAIN_TASK")
META_OPTIMIZER = Registry("OPTIMIZER")
META_SCHEDULER = Registry("SCHEDULER")

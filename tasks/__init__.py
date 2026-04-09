"""
Pretraining tasks module
"""

from .base_pretraining_task import BasePretrainingTask, MLMPretrainingTask
from builders.registry import META_PRETRAIN_TASK

# Register pretraining tasks
META_PRETRAIN_TASK.register(MLMPretrainingTask)

__all__ = [
    'BasePretrainingTask',
    'MLMPretrainingTask',
]

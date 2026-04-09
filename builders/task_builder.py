"""
Task builder for pretraining
"""

from .registry import META_PRETRAIN_TASK


def build_pretrain_task(config):
    """
    Build a pretraining task from configuration
    
    Args:
        config: Pretraining configuration
        
    Returns:
        PretrainingTask instance
    """
    task_type = config.training.get('task', 'MLMPretraining')
    task_class = META_PRETRAIN_TASK.get(task_type)
    
    return task_class(config)

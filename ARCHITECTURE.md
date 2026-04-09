# ViWordFormer Pretraining - Architecture Guide

## Overview

This pretraining pipeline follows the **Registry-based architecture pattern** from ViWordFormer, enabling:
- **Extensibility**: Add new models, datasets, tasks via registration
- **Configuration-driven**: YAML-based configuration for easy experimentation  
- **Modular design**: Independent components (tokenizer, model, dataset, task)
- **Best practices**: Checkpoint management, logging, gradient clipping

## Architecture Pattern

### 1. Registry System

```python
from builders.registry import META_ARCHITECTURE, META_DATASET, META_PRETRAIN_TASK

# Register custom components
@META_ARCHITECTURE.register()
class CustomModel(nn.Module):
    pass
```

**Available Registries**:
- `META_ARCHITECTURE`: Models
- `META_TOKENIZER`: Tokenizers
- `META_DATASET`: Datasets
- `META_PRETRAIN_TASK`: Training tasks
- `META_OPTIMIZER`: Optimizers (planned)
- `META_SCHEDULER`: Schedulers (planned)

### 2. Configuration Files

**Format**: YAML (recommended) or JSON

```yaml
# YAML Configuration Structure
tokenizer:
  type: unigram
  vocab_size: 30000
  model_prefix: "./outputs/unigram_tokenizer"

model:
  architecture: ViWordFormer
  d_model: 768
  nlayers: 12
  head: 12

training:
  task: MLMPretraining
  optimizer: adamw
  learning_rate: 0.0001
  batch_size: 32
  num_epochs: 3
```

### 3. Task-Based Training

**BasePretrainingTask** provides the foundation:
- Configuration loading
- Model building
- Dataset loading
- Optimizer/scheduler setup
- Checkpoint management

**Specific Tasks** (e.g., MLMPretrainingTask) implement:
- Training loop
- Loss computation
- Metric tracking
- Resume functionality

### 4. Builder Pattern

**Builders** create components from configuration:

```python
from builders.model_builder import build_model
from builders.tokenizer_builder import build_tokenizer
from builders.dataset_builder import build_dataset

# Build from config
model = build_model(config.model, vocab_size)
tokenizer = build_tokenizer(config.tokenizer)
dataset = build_dataset(config.dataset, tokenizer)
```

## File Organization

```
ViWordFormer_Pretraining/
├── builders/
│   ├── registry.py              # Component registries
│   ├── model_builder.py         # Model factory
│   ├── tokenizer_builder.py     # Tokenizer factory
│   ├── dataset_builder.py       # Dataset factory
│   └── task_builder.py          # Task factory
│
├── models/
│   ├── __init__.py              # Registers ViWordFormer
│   ├── attention.py             # Attention mechanisms
│   └── viwordformer.py          # Model definition
│
├── tasks/
│   ├── __init__.py              # Registers tasks
│   └── base_pretraining_task.py # Base + MLM task
│
├── configs/
│   ├── config.py                # Python config classes
│   └── viwordformer_pretrain.yaml  # YAML configuration
│
├── tokenizer/
│   └── unigram_tokenizer.py    # Unigram tokenizer
│
├── data_processing/
│   └── data_processor.py        # Vietnamese/Chinese processors
│
└── pretrain.py                  # Main training script
```

## Usage

### Basic Training

```bash
# Using YAML config
python pretrain.py --config configs/viwordformer_pretrain.yaml

# Skip data processing
python pretrain.py --config config.yaml --no-process

# Skip tokenizer training
python pretrain.py --config config.yaml --no-tokenizer

# Resume from checkpoint
python pretrain.py --config config.yaml --resume checkpoints/epoch_2/model.pt
```

### Custom Model

```python
from builders.registry import META_ARCHITECTURE
from torch import nn

@META_ARCHITECTURE.register()
class MyCustomModel(nn.Module):
    def __init__(self, vocab_size, d_model, **kwargs):
        super().__init__()
        # Your implementation
        pass
```

Then update config:
```yaml
model:
  architecture: MyCustomModel
  d_model: 768
```

### Custom Dataset

```python
from builders.registry import META_DATASET
from torch.utils.data import Dataset

@META_DATASET.register()
class MyDataset(Dataset):
    def __init__(self, file_path, tokenizer, **kwargs):
        # Your implementation
        pass
```

### Custom Training Task

```python
from tasks.base_pretraining_task import BasePretrainingTask
from builders.registry import META_PRETRAIN_TASK

@META_PRETRAIN_TASK.register()
class MyTask(BasePretrainingTask):
    def train(self, num_epochs=None):
        # Your training logic
        pass
```

Then update config:
```yaml
training:
  task: MyTask
```

## Configuration Details

### Model Config
```yaml
model:
  architecture: ViWordFormer
  d_model: 768              # Hidden dimension
  d_ff: 3072                # Feed-forward dimension
  nlayers: 12               # Number of layers
  head: 12                  # Number of attention heads
  d_q: 64                   # Query dimension per head
  d_kv: 64                  # Key/Value dimension per head
  dropout: 0.1              # Dropout rate
  label_smoothing: 0.1      # Label smoothing for loss
  max_seq_len: 512          # Maximum sequence length
  pad_idx: 3                # Padding token id
```

### Training Config
```yaml
training:
  task: MLMPretraining      # Task name
  optimizer: adamw          # adam or adamw
  scheduler: linear         # linear or cosine
  learning_rate: 0.0001     # Learning rate
  weight_decay: 0.01        # Weight decay for AdamW
  gradient_clip_val: 1.0    # Gradient clipping max norm
  batch_size: 32            # Batch size
  num_epochs: 3             # Number of epochs
  gradient_accumulation_steps: 1  # Gradient accumulation
  warmup_steps: 10000       # Warmup steps (if used)
  device: cuda              # cuda or cpu
```

### Dataset Config
```yaml
dataset:
  type: PretrainingDataset
  file_path: "./processed_data/merged_corpus.txt"
  max_seq_len: 512
  mlm_probability: 0.15     # MLM masking probability
```

## Training Output Structure

```
checkpoints/
├── best/
│   ├── model.pt           # Best model weights
│   ├── optimizer.pt       # Optimizer state
│   └── config.json        # Config used
├── epoch_1/
│   ├── model.pt
│   ├── optimizer.pt
│   └── config.json
└── epoch_2/
    └── ...
```

## Key Components

### 1. **Registry Pattern Benefits**
- Plugin architecture: Add components without modifying core code
- Runtime component selection via configuration
- Clear component contracts via base classes

### 2. **Config-Driven Training**
- No code changes needed for hyperparameter tuning
- Reproducible experiments via config files
- Easy switching between models/tasks

### 3. **Task Abstraction**
- Separated concerns: model training logic lives in tasks
- Easy to add new training objectives
- Configurable via YAML

### 4. **Builder Pattern Benefits**
- Centralized component instantiation
- Configuration validation
- Clear dependency management

## Advanced Features

### Gradient Accumulation
```yaml
training:
  batch_size: 16
  gradient_accumulation_steps: 2  # Effective batch = 32
```

### Mixed Precision Training
```yaml
training:
  fp16: true  # Enable FP16 (currently prepared, not implemented)
```

### Learning Rate Scheduling
```yaml
training:
  scheduler: linear   # Linear warmup then decay
  warmup_steps: 10000
  # or
  scheduler: cosine   # Cosine annealing
```

### Checkpoint Management
```yaml
training:
  checkpoint_dir: "./checkpoints"
  save_steps: 500          # Save every N steps
  save_total_limit: 3      # Keep only 3 latest
```

## Integration with ViWordFormer

This pretraining pipeline integrates seamlessly:

1. **Same architecture**: Uses ViWordFormer model from `models/viwordformer.py`
2. **Similar patterns**: Registry-based component registration
3. **Compatible config**: Can use ViWordFormer configs as base
4. **Trained models**: Can be fine-tuned on ViWordFormer downstream tasks

## Extending the Framework

### Add New Attention Mechanism
```python
# In models/attention.py
class CustomAttention(nn.Module):
    def forward(self, q, k, v, mask=None):
        # Your implementation
        pass
```

### Add New Optimizer
```python
@META_OPTIMIZER.register()
class CustomOptimizer(torch.optim.Optimizer):
    pass
```

### Add New Scheduler
```python
@META_SCHEDULER.register()
class CustomScheduler:
    pass
```

## Troubleshooting

### Import Errors
- Ensure all required packages installed: `pip install -r requirements.txt`
- Check Python path includes project root

### Registry Errors
- Verify component registered before use: check registration order in `__init__.py`
- Component name in registry must match config

### Training Errors
- Check dataset path exists and is readable
- Verify tokenizer model is trained
- Check GPU memory availability

## Best Practices

1. **Configuration Management**
   - Use YAML for different experiments
   - Version control config files
   - Document non-obvious settings

2. **Checkpoint Strategy**
   - Save best model based on validation loss
   - Keep epoch checkpoints for analysis
   - Clean up old checkpoints to save space

3. **Monitoring**
   - Check logs during training
   - Monitor loss curve
   - Save training metrics to file

4. **Reproducibility**
   - Set random seeds
   - Use fixed dataset order
   - Document all configuration

---

For more details, see README.md and QUICKSTART.md

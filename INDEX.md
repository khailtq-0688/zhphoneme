# ViWordFormer Pretraining - Documentation Index

Complete guide to the ViWordFormer Pretraining Pipeline with Registry-Based Architecture.

## 📖 Documentation

### Getting Started
- **[README.md](README.md)** - Project overview, features, requirements
  - What is ViWordFormer Pretraining
  - Quick start guide
  - Project structure
  - FAQ

- **[QUICKSTART.md](QUICKSTART.md)** - Step-by-step setup
  - Installation
  - Data preparation
  - Running examples
  - Troubleshooting

### For Users
- **[SETUP_SUMMARY.md](SETUP_SUMMARY.md)** - Complete setup checklist
  - What was created
  - Dataset paths to configure  
  - Output directories
  - Git status

- **[UPGRADE_GUIDE.md](UPGRADE_GUIDE.md)** - Migration from old system
  - What changed
  - How to update code
  - New features
  - Configuration examples

### For Developers
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Technical architecture guide
  - Registry pattern explanation
  - Builder pattern details
  - File organization
  - How to extend the framework
  - Custom components guide

- **[examples_architecture.py](examples_architecture.py)** - Working code examples
  - Basic training
  - Programmatic configuration
  - Custom component registration
  - Registry inspection
  - Checkpoint management

## 🚀 Quick Links

### Training
```bash
# Using YAML configuration
python pretrain.py --config configs/viwordformer_pretrain.yaml

# With options
python pretrain.py --config config.yaml --no-process --resume checkpoints/best/model.pt
```

### Data Processing
```bash
# Process datasets (included in pretrain.py)
python scripts_process_data.py

# Or manually:
python scripts_train_tokenizer.py
python scripts_process_data.py
```

### Examples
```bash
# Run architecture examples
python examples_architecture.py

# Or individual training scripts
python train_pretrain.py
```

## 📁 File Organization

```
ViWordFormer_Pretraining/

Documentation/
├── README.md              ← Start here
├── QUICKSTART.md          ← Step-by-step
├── ARCHITECTURE.md        ← Technical details
├── UPGRADE_GUIDE.md       ← Migration guide
├── SETUP_SUMMARY.md       ← What was created
└── INDEX.md              ← This file

Configuration/
├── configs/
│   ├── config.py          # Python config classes
│   └── viwordformer_pretrain.yaml  # Main config
└── requirements.txt       # Dependencies

Training/
├── pretrain.py            # Main training entry point
├── train_pretrain.py      # Simplified wrapper
└── tasks/
    ├── __init__.py
    └── base_pretraining_task.py  # Task implementations

Models & Components/
├── models/
│   ├── __init__.py        # Model registration
│   ├── viwordformer.py    # ViWordFormer model
│   └── attention.py       # Attention mechanisms
├── builders/
│   ├── __init__.py
│   ├── registry.py        # Component registries
│   ├── model_builder.py   # Model factory
│   ├── dataset_builder.py # Dataset factory
│   ├── tokenizer_builder.py  # Tokenizer factory
│   └── task_builder.py    # Task factory
└── tokenizer/
    └── unigram_tokenizer.py  # Unigram SentencePiece

Data Processing/
├── data_processing/
│   ├── __init__.py
│   └── data_processor.py  # Dataset processors
├── scripts_process_data.py    # Easy data processing
└── scripts_train_tokenizer.py # Tokenizer training

Examples/
├── examples_architecture.py   # Architecture examples
└── configs/viwordformer_pretrain.yaml  # Example config

Utilities/
├── utils/
│   ├── __init__.py
│   └── helpers.py         # Training utilities
└── __init__.py            # Package init
```

## 🎯 Common Tasks

### 1. **Start Training from Scratch**
```bash
# 1. Update dataset paths in configs/viwordformer_pretrain.yaml
# 2. Install dependencies
pip install -r requirements.txt

# 3. Start training (includes data processing & tokenizer training)
python pretrain.py --config configs/viwordformer_pretrain.yaml
```

### 2. **Resume from Checkpoint**
```bash
python pretrain.py --config configs/viwordformer_pretrain.yaml \
  --resume checkpoints/epoch_2/model.pt
```

### 3. **Skip Preprocessing**
```bash
python pretrain.py --config configs/viwordformer_pretrain.yaml \
  --no-process --no-tokenizer
```

### 4. **Create Custom Model**
See [ARCHITECTURE.md](ARCHITECTURE.md) → "Custom Model" section

### 5. **Add Custom Dataset**
See [ARCHITECTURE.md](ARCHITECTURE.md) → "Custom Dataset" section

### 6. **Run Examples**
```bash
python examples_architecture.py
```

## 📊 Training Outputs

After training, outputs are organized as:
```
checkpoints/
├── best/
│   ├── model.pt           # Best weights
│   ├── optimizer.pt       # Optimizer state
│   └── config.json        # Configuration
├── epoch_1/
│   ├── model.pt
│   ├── optimizer.pt
│   └── config.json
└── epoch_2/
    └── ...

processed_data/
├── vietnam_processed.txt
├── chinese_processed.txt
├── merged_corpus.txt
└── stats_*.json

outputs/
├── unigram_tokenizer.model   # Tokenizer
├── unigram_tokenizer.vocab
└── tokenizer_config.json
```

## 🔧 Configuration Reference

### Model Configuration
```yaml
model:
  architecture: ViWordFormer    # Model class name
  d_model: 768                  # Hidden dimension
  d_ff: 3072                    # Feed-forward dim
  nlayers: 12                   # Number of layers
  head: 12                      # Attention heads
  d_q: 64                       # Query dim per head
  d_kv: 64                      # Key/Value dim per head
  dropout: 0.1                  # Dropout rate
  label_smoothing: 0.1          # Label smoothing
  max_seq_len: 512              # Max sequence length
  pad_idx: 3                    # Pad token ID
```

### Training Configuration
```yaml
training:
  task: MLMPretraining          # Task class name
  optimizer: adamw              # adam or adamw
  scheduler: linear             # linear or cosine
  learning_rate: 0.0001         # Learning rate
  weight_decay: 0.01            # Weight decay
  gradient_clip_val: 1.0        # Gradient clipping
  batch_size: 32                # Batch size
  num_epochs: 3                 # Number of epochs
  num_workers: 4                # Data loading workers
  device: cuda                  # cuda or cpu
```

## 🔍 Architecture Overview

### Registry Pattern
- **Purpose**: Plugin architecture for custom components
- **Location**: `builders/registry.py`
- **Usage**: See [ARCHITECTURE.md](ARCHITECTURE.md)

### Builder Pattern
- **Purpose**: Centralized component instantiation
- **Components**:
  - `model_builder.py` - Creates models
  - `dataset_builder.py` - Creates datasets
  - `tokenizer_builder.py` - Creates tokenizers
  - `task_builder.py` - Creates tasks

### Task-Based Training
- **Purpose**: Separated training logic from models
- **Base Class**: `BasePretrainingTask`
- **Implementation**: `MLMPretrainingTask`

## 💡 Key Concepts

### 1. **Registry**
Decoupled component management - register components once, use anywhere:
```python
@META_ARCHITECTURE.register()
class MyModel(nn.Module):
    pass
```

### 2. **Builders**
Factory functions create components from config:
```python
model = build_model(config.model, vocab_size)
```

### 3. **Tasks**
Training logic separated from model:
```python
task = MLMPretrainingTask(config)
task.train()
```

### 4. **Configuration**
YAML-based config for experiments:
```yaml
model:
  d_model: 768
training:
  batch_size: 32
```

## 📚 Learning Path

### Beginner
1. Read [README.md](README.md)
2. Follow [QUICKSTART.md](QUICKSTART.md)
3. Run `examples_architecture.py`

### Intermediate
1. Read [ARCHITECTURE.md](ARCHITECTURE.md) sections 1-3
2. Understand registry pattern
3. Try modifying config in YAML

### Advanced
1. Read full [ARCHITECTURE.md](ARCHITECTURE.md)
2. Create custom component
3. Implement custom task
4. Contribute improvements

## ❓ Common Questions

**Q: How do I change the model size?**
A: Edit `configs/viwordformer_pretrain.yaml` → `model: d_model:` value

**Q: How do I train on different data?**
A: Update`data_paths` in config file, then run `pretrain.py`

**Q: How do I add a custom model?**
A: See "Custom Model" in [ARCHITECTURE.md](ARCHITECTURE.md)

**Q: What if I prefer Python configs?**
A: Use `PretrainingConfig` class in `configs/config.py`

**Q: Can I use the old training script?**
A: Yes! `train.py` still works, but `pretrain.py` is recommended

**Q: How do I monitor training?**
A: Check console output, or inspect `checkpoints/` directory

See [FAQ section in README.md](README.md#faq) for more questions.

## 🔗 Related Files

### Core Training
- `pretrain.py` - Main entry point
- `tasks/base_pretraining_task.py` - Base task class
- `models/viwordformer.py` - Model implementation

### Configuration
- `configs/viwordformer_pretrain.yaml` - Configuration template
- `configs/config.py` - Python configuration classes

### Components
- `builders/registry.py` - Component registries
- `builders/model_builder.py` - Model instantiation
- `builders/dataset_builder.py` - Dataset instantiation

### Examples
- `examples_architecture.py` - Working code examples
- `scripts_train_tokenizer.py` - Tokenizer training
- `scripts_process_data.py` - Data processing

## 🎓 Version Information

**Project**: ViWordFormer Pretraining Pipeline
**Architecture**: Registry-based with Builders
**Status**: Production Ready ✓
**Last Updated**: April 2026

## 📞 Support

For issues, questions, or suggestions:
1. Check [QUICKSTART.md](QUICKSTART.md) troubleshooting section
2. Review relevant documentation file
3. Check `examples_architecture.py` for code examples
4. Inspect git log for recent changes

---

**Happy Training!** 🚀

*For detailed information on any topic, click the relevant link above.*

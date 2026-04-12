# ViWordFormer Pretraining

Pretrain ViWordFormer models on Chinese and Vietnamese corpora using Masked Language Modeling (MLM) with Unigram tokenizers.

## Features

- **Architecture**: ViWordFormer (768d, 12 layers, 12 attention heads)
- **Tokenizer**: SentencePiece Unigram (30,000 vocab size)
- **Objective**: Masked Language Modeling (MLM) with 15% masking
- **Training**: Clean training loop following best practices
  - AdamW optimizer with dynamic learning rate scheduling
  - Linear warmup (5% of total steps) + linear decay
  - Batch size: 512, Epochs: 5
  - Learning rate: 6e-4, Weight decay: 0.01

## Directory Structure

```
├── main.py                          # Entry point (supports both languages)
├── pretrain_chinese_subset.py       # Train on Chinese corpus
├── pretrain_vietnamese_subset.py    # Train on Vietnamese corpus
├── configs/
│   ├── viwordformer.yaml            # Original architecture config
│   ├── viwordformer_pretrain_chinese_subset.yaml    # Chinese training config
│   └── viwordformer_pretrain_vietnamese_subset.yaml # Vietnamese training config
├── builders/
│   ├── dataset_builder.py           # SubsetDataset & collate_fn
│   ├── model_builder.py             # Model factory
│   ├── registry.py                  # Component registry
│   └── task_builder.py              # Task factory
├── models/
│   └── viwordformer.py              # ViWordFormer model
├── tasks/
│   └── base_pretraining_task.py     # MLM pretraining task
├── tokenizer/
│   └── unigram_tokenizer.py         # SentencePiece tokenizer
├── processors/                      # Data processors
├── checkpoints/                     # Model checkpoints (created at runtime)
├── tokenizers/                      # Trained tokenizers (created at runtime)
└── README.md
```

## Installation

```bash
# Install dependencies
pip install torch sentencepiece pyyaml tqdm

# Optional: GPU support
pip install torch-cuda  # or appropriate CUDA version
```

## Training

```bash
# Chinese
python main.py train-chinese --corpus-dir ../../baidubaike_chinese

# Vietnamese
python main.py train-vietnamese --corpus-dir ../../vietnamese_curated
```

## Configuration

Edit `configs/viwordformer_pretrain_chinese_subset.yaml` to customize:

### Model Architecture
- `hidden_size`: 768
- `num_hidden_layers`: 12
- `num_attention_heads`: 12
- `max_position_embeddings`: 512

### Training
- `batch_size`: 512
- `num_epochs`: 5
- `learning_rate`: 6e-4
- `warmup`: 5% of total steps (dynamic)

### Tokenizer
- `vocab_size`: 30,000
- `model_type`: unigram
- `character_coverage`: 0.9999

## Dataset Format

Expects a directory with subset files:

```
corpus_dir/
├── subset_0.txt      (one document per line)
├── subset_1.txt
├── subset_2.txt
└── ...
```

The `SubsetDataset` transparently reads from all files as one continuous corpus:
- Automatically counts total lines across all files
- Maps linear indices to (file, line) pairs
- Calculates: `subset_idx, line_idx = divmod(idx, lines_per_file)`

## Output

After training:

### Chinese
```
checkpoints/chinese_subset_pretrain/
├── best/
│   └── model.pt           # Best model weights
├── epoch_1/
├── epoch_2/
└── ...                    # Checkpoint per epoch
tokenizers/
└── unigram_tokenizer_chinese_subset.model
```

### Vietnamese
```
checkpoints/vietnamese_subset_pretrain/
├── best/
│   └── model.pt           # Best model weights
├── epoch_1/
├── epoch_2/
└── ...                    # Checkpoint per epoch
tokenizers/
└── unigram_tokenizer_vietnamese_subset.model
```

## Training Metrics

The training loop logs:
- Total steps and warmup steps
- Batch size and learning rate
- Loss per epoch
- Best loss improvements

## Resume Training

```bash
python main.py train-chinese \
  --resume checkpoints/chinese_subset_pretrain/best/model.pt
```

## Model Architecture

ViWordFormer is a transformer-based model optimized for word-level representation:
- 768-dimensional embeddings
- 12 transformer layers
- 12 parallel attention heads
- Feed-forward intermediate size: 3072
- GELU activation
- Max sequence length: 512 tokens

## License

MIT

## References

- ViWordFormer: Word-level Representation Learning for Vietnamese
- BERT: Pre-training of Deep Bidirectional Transformers
- SentencePiece: A simple and language independent approach to subword segmentation

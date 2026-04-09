# ViWordFormer Pretraining Pipeline

A multilingual pretraining pipeline for ViWordFormer model, supporting Vietnamese (Curated) and Chinese (CommonCrawl) datasets with Unigram tokenization.

## Overview

This project provides a complete setup for pretraining the ViWordFormer model on multilingual data. ViWordFormer is a transformer-based model that combines:
- **Scaled Dot-Product Attention**: Standard attention mechanism
- **Phrasal Lexeme Attention**: Novel attention mechanism capturing word-level and morpheme-level linguistic structures

## Project Structure

```
ViWordFormer_Pretraining/
├── configs/
│   └── config.py                 # Configuration management
├── data_processing/
│   └── data_processor.py         # Data preprocessing for Vi & Zh
├── models/
│   ├── attention.py              # Attention mechanisms
│   └── viwordformer.py           # ViWordFormer model
├── tokenizer/
│   └── unigram_tokenizer.py      # Unigram SentencePiece tokenizer
├── utils/
│   └── helpers.py                # Utility functions
├── train.py                       # Main training script
└── README.md                      # This file
```

## Requirements

Install required packages:

```bash
pip install torch==2.0.0
pip install sentencepiece==0.1.99
pip install tqdm
```

## Quick Start

### 1. Prepare Dataset Paths

Configure your dataset paths in the training script:

**Vietnamese Dataset (Curated):**
```
/path/to/vietnamese/curated/dataset/
├── *.txt              # Text files  
└── *.jsonl            # JSONL files (optional)
```

**Chinese Dataset (CommonCrawl):**
```
/path/to/chinese/commoncrawl/dataset/
├── *.txt              # Text files
├── *.jsonl            # JSONL files (from extraction)
└── *.warc             # WARC files (optional)
```

### 2. Train Tokenizer

```python
from tokenizer.unigram_tokenizer import UnigramTokenizer

tokenizer = UnigramTokenizer(
    model_prefix="./outputs/unigram_tokenizer",
    vocab_size=30000
)

tokenizer.train(
    training_files=[
        "/path/to/vietnamese/curated/dataset/*.txt",
        "/path/to/chinese/commoncrawl/dataset/*.txt"
    ]
)
```

### 3. Process Data

```python
from data_processing.data_processor import (
    VietnameseProcessor, 
    ChineseProcessor, 
    DataMerger
)

# Vietnamese
vi_processor = VietnameseProcessor(
    input_dir="/path/to/vietnamese/curated/dataset",
    output_dir="./processed_data"
)
vi_processor.process()

# Chinese
zh_processor = ChineseProcessor(
    input_dir="/path/to/chinese/commoncrawl/dataset",
    output_dir="./processed_data"
)
zh_processor.process()

# Merge
merger = DataMerger(output_dir="./processed_data")
merger.merge_all()
```

### 4. Start Training

```bash
python train.py
```

## Configuration

Edit `config.json` to customize training:

```json
{
  "tokenizer": {
    "vocab_size": 30000,
    "model_type": "unigram"
  },
  "model": {
    "d_model": 768,
    "nlayers": 12,
    "head": 12
  },
  "training": {
    "batch_size": 32,
    "num_epochs": 3,
    "learning_rate": 1e-4
  }
}
```

## Model Architecture

### ViWordFormer Configuration
- **Vocabulary Size**: 30,000 (Unigram)
- **Embedding Dimension**: 768
- **Number of Layers**: 12
- **Attention Heads**: 12
- **Feed Forward Dimension**: 3,072
- **Max Sequence Length**: 512
- **Attention Types**: 
  - Scaled Dot-Product Attention
  - Phrasal Lexeme Attention

## Dataset Format

### Vietnamese (Curated)
- **Format**: Plain text or JSONL
- **Fields**: `text`, `content`, `body`, or `document`
- **Processing**: URL removal, email removal, whitespace normalization

### Chinese (CommonCrawl)
- **Format**: Plain text, JSONL, or WARC
- **Fields**: `text`, `content`, `body`, `html`, `raw_content`
- **Processing**: HTML tag removal, entity decoding, text cleaning

## Training Features

### Masked Language Modeling (MLM)
- 15% token masking probability
- Supports gradient accumulation
- Gradient clipping for stability

### Optimization
- **Optimizer**: Adam/AdamW
- **Learning Rate Scheduler**: Linear decay or Cosine annealing
- **Warmup**: Configurable warmup steps
- **FP16**: Optional mixed precision training

### Checkpointing
- Save best model based on validation loss
- Save periodic checkpoints
- Resume from checkpoint

## Outputs

Training outputs are saved to:
```
outputs/
├── unigram_tokenizer.model       # Tokenizer model
├── unigram_tokenizer.vocab       # Vocabulary
└── checkpoints/
    ├── best/
    │   ├── model.pt
    │   ├── optimizer.pt
    │   └── config.json
    └── epoch_1/
        └── ...
```

## Preprocessing Details

### Vietnamese Processing
- Loads .txt and .jsonl files
- Extracts text from common JSON fields
- Removes URLs and email addresses
- Normalizes whitespace

### Chinese Processing
- Handles HTML content from CommonCrawl
- Removes HTML tags and decodes entities
- Filters short lines (< 5 characters)
- Special handling for WARC format extracts

## Advanced Usage

### Custom Data Format
Extend `DataProcessor` class:

```python
from data_processing.data_processor import DataProcessor

class CustomProcessor(DataProcessor):
    def process(self):
        # Your custom processing logic
        pass
```

### Evaluation
Implement custom evaluation metrics by extending the `Trainer` class.

### Multi-GPU Training
Set `device='cuda'` in config and use `DataParallel`:

```python
model = nn.DataParallel(model)
```

## Troubleshooting

### Memory Issues
- Reduce `batch_size` in config
- Increase `gradient_accumulation_steps`
- Enable `fp16=True` for mixed precision

### Slow Training
- Increase `num_workers` in DataLoader
- Check data preprocessing bottlenecks
- Use larger `batch_size` if memory allows

### Tokenizer Issues
- Ensure SentencePiece is installed: `pip install sentencepiece`
- Verify dataset files exist and are readable
- Check character coverage for target languages

## Performance

Typical training speed (per epoch):
- **Single GPU (V100)**: ~2-3 hours
- **Batch Size 32**: ~500 sequences/sec
- **Total Training Time (3 epochs)**: ~6-9 hours

## Citation

If you use this pretraining pipeline, please cite:

```bibtex
@article{viwordformer,
  title={ViWordFormer: Exploring Morpheme-level Word Representations},
  year={2024}
}
```

## License

MIT License

## Contact

For issues and suggestions, please open an issue in the repository.

## FAQ

**Q: Can I use other languages?**
A: Yes! Extend `DataProcessor` for your language and use the same training pipeline.

**Q: How long does pretraining take?**
A: Depends on dataset size and hardware. Typically 6-24 hours for a full dataset.

**Q: Can I fine-tune after pretraining?**
A: Yes! Use the trained model weights for downstream tasks.

**Q: What's the recommended batch size?**
A: Start with 32, increase if GPU memory allows (64, 128).

---

**Last Updated**: April 2026

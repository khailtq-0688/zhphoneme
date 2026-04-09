# ViWordFormer Pretraining - Setup Summary

✅ **Project Successfully Created and Initialized**

## Project Location
```
c:\Users\Thai Bao\Desktop\chinese\ViWordFormer_Pretraining\
```

## What Was Created

### 📁 Folder Structure
```
ViWordFormer_Pretraining/
├── configs/                    # Configuration management
│   ├── __init__.py
│   └── config.py              # Model & training configs
│
├── data_processing/            # Dataset processing
│   ├── __init__.py
│   └── data_processor.py       # Vietnamese & Chinese processors
│
├── models/                      # Neural network modules
│   ├── __init__.py
│   ├── attention.py            # Attention mechanisms
│   └── viwordformer.py         # Main model
│
├── tokenizer/                   # Tokenization
│   ├── __init__.py
│   └── unigram_tokenizer.py    # Unigram SentencePiece
│
├── utils/                       # Utility functions
│   ├── __init__.py
│   └── helpers.py              # Training helpers
│
├── train.py                     # Main training script
├── inference.py                 # Model inference
├── requirements.txt             # Python dependencies
├── README.md                    # Full documentation
├── QUICKSTART.md                # Quick start guide
├── SETUP_SUMMARY.md             # This file
├── scripts_train_tokenizer.py   # Tokenizer training script
└── scripts_process_data.py      # Data processing script
```

## Core Components

### 1️⃣ **Tokenizer** (`tokenizer/unigram_tokenizer.py`)
- **Type**: Unigram SentencePiece
- **Vocab Size**: 30,000 (configurable)
- **Features**:
  - Custom special tokens: `<s>`, `</s>`, `<unk>`, `<pad>`
  - Support for multiple languages
  - Vocabulary export to JSON

### 2️⃣ **Model** (`models/`)
- **Architecture**: ViWordFormer
- **Attention Types**:
  - Scaled Dot-Product Attention
  - Phrasal Lexeme Attention (novel)
- **Configuration**:
  - Hidden size: 768
  - Layers: 12
  - Heads: 12
  - Feed-forward dim: 3,072

### 3️⃣ **Training** (`train.py`)
- **Features**:
  - Masked Language Modeling (MLM)
  - Gradient accumulation
  - Learning rate scheduling
  - Checkpoint saving
  - Mixed precision support (FP16 ready)
- **Datasets**:
  - Vietnamese (Curated)
  - Chinese (CommonCrawl)

### 4️⃣ **Data Processing** (`data_processing/`)
- **Vietnamese Processor**:
  - Handles .txt and .jsonl files
  - Removes URLs, emails
  - Normalizes whitespace
- **Chinese Processor**:
  - HTML cleaning (CommonCrawl)
  - Entity decoding
  - Filters short text
- **Data Merger**: Combines datasets

### 5️⃣ **Inference** (`inference.py`)
- Load trained models
- Make predictions
- Extract embeddings

## Dataset Paths (Configure Before Use)

You'll need to provide paths for:

**Vietnamese (Curated):**
```
/path/to/vietnamese/curated/dataset/
├── *.txt
└── *.jsonl
```

**Chinese (CommonCrawl):**
```
/path/to/chinese/commoncrawl/dataset/
├── *.txt
├── *.jsonl
└── *.warc (optional)
```

## Quick Start Commands

### Install Dependencies
```bash
cd C:\Users\Thai Bao\Desktop\chinese\ViWordFormer_Pretraining
pip install -r requirements.txt
```

### Train Tokenizer
```bash
python scripts_train_tokenizer.py
```

### Process Data
```bash
python scripts_process_data.py
```

### Start Training
```bash
python train.py
```

## Configuration Files

### `config.json` (Generated)
```json
{
  "tokenizer": {"vocab_size": 30000},
  "model": {"d_model": 768, "nlayers": 12},
  "training": {"batch_size": 32, "num_epochs": 3}
}
```

Customize before training:
- Batch size: For your GPU memory
- Learning rate: Start with 1e-4
- Epochs: Usually 3-5 for good results

## Output Directories

After setup:
```
outputs/
├── unigram_tokenizer.model
├── unigram_tokenizer.vocab
└── checkpoints/
    ├── best/
    └── epoch_1/, epoch_2/, ...

processed_data/
├── vietnamese_processed.txt
├── chinese_processed.txt
├── merged_corpus.txt
├── stats_vi.json
└── stats_zh.json
```

## Git Status

✅ Repository initialized
✅ All files committed to master branch

```bash
# View commit
git log --oneline

# Check status
git status
```

## Files Overview

| File | Purpose |
|------|---------|
| `train.py` | Main training loop (start here) |
| `inference.py` | Model inference |
| `scripts_train_tokenizer.py` | Tokenizer training example |
| `scripts_process_data.py` | Data processing example |
| `configs/config.py` | Configuration classes |
| `models/viwordformer.py` | ViWordFormer architecture |
| `tokenizer/unigram_tokenizer.py` | Unigram tokenizer |
| `data_processing/data_processor.py` | Dataset processors |
| `utils/helpers.py` | Utility functions |

## Next Steps

1. **Update dataset paths** in scripts:
   - `scripts_train_tokenizer.py` (line ~30)
   - `scripts_process_data.py` (line ~30)

2. **Prepare your datasets**:
   - Place Vietnamese corpus in curated folder
   - Place Chinese corpus in CommonCrawl folder

3. **Run setup scripts in order**:
   - `scripts_train_tokenizer.py` → creates tokenizer
   - `scripts_process_data.py` → creates processed data
   - `train.py` → starts training

4. **Monitor training**:
   - Check `checkpoints/` for saves
   - Monitor loss in console output
   - Best model saved in `checkpoints/best/`

## Troubleshooting

**Missing dependencies:**
```bash
pip install -r requirements.txt
```

**CUDA issues:**
- Check device selection in config
- Verify GPU available: `nvidia-smi`

**Data not loading:**
- Verify paths exist
- Check file permissions
- Ensure text encoding is UTF-8

**Memory issues:**
- Reduce `batch_size` in config
- Enable `gradient_accumulation_steps`

## Features

✅ **Multilingual**: Vietnamese & Chinese support
✅ **Modern Architecture**: Phrasal Lexeme Attention
✅ **Production Ready**: Checkpointing, logging, error handling
✅ **Configurable**: All parameters adjustable via config.json
✅ **Scalable**: Support for distributed training (ready for multi-GPU)
✅ **Well Documented**: README, QUICKSTART, docstrings

## Version Info

- **PyTorch**: 2.0.0
- **SentencePiece**: 0.1.99
- **Python**: 3.8+
- **CUDA**: Optional (CPU fallback available)

## Support Files

- `README.md` - Comprehensive documentation
- `QUICKSTART.md` - Fast getting started guide
- `SETUP_SUMMARY.md` - This file

---

**Created**: April 2026
**Status**: Ready for use ✅
**Repository**: Initialized in current directory

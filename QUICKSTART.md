# Quick Start Guide - ViWordFormer Pretraining

## Installation

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Step-by-Step Setup

### Step 1: Prepare Dataset Directories

Create directories and place your datasets:

**Vietnamese (Curated):**
```
/your/data/path/vietnamese/
├── text_file1.txt
├── text_file2.txt
└── dataset.jsonl
```

**Chinese (CommonCrawl):**
```
/your/data/path/chinese/
├── web_text1.txt
├── web_text2.txt
└── commoncrawl_extract.jsonl
```

### Step 2: Update Configuration

Edit dataset paths in `scripts_process_data.py`:
```python
vi_input_dir = "/your/data/path/vietnamese"
zh_input_dir = "/your/data/path/chinese"
```

### Step 3: Train Tokenizer

```bash
python scripts_train_tokenizer.py
```

**Expected Output:**
```
outputs/
├── unigram_tokenizer.model       # Unigram SentencePiece model
├── unigram_tokenizer.vocab       # Vocabulary file
└── tokenizer_config.json         # Configuration
```

### Step 4: Process Data

```bash
python scripts_process_data.py
```

**Expected Output:**
```
processed_data/
├── vietnamese_processed.txt
├── chinese_processed.txt
├── merged_corpus.txt
├── stats_vi.json
└── stats_zh.json
```

### Step 5: Start Training

```bash
python train.py
```

**Training will:**
- Load merged corpus and tokenizer
- Create training dataset
- Train ViWordFormer with MLM objective
- Save checkpoints and model weights

## Configuration

### Default Config (config.json)
```json
{
  "tokenizer": {
    "vocab_size": 30000,
    "model_type": "unigram"
  },
  "model": {
    "d_model": 768,
    "nlayers": 12,
    "head": 12,
    "d_ff": 3072
  },
  "training": {
    "batch_size": 32,
    "num_epochs": 3,
    "learning_rate": 1e-4
  }
}
```

### Customize Training

**For faster testing:**
```json
{
  "training": {
    "batch_size": 8,
    "num_epochs": 1,
    "save_steps": 100
  }
}
```

**For production:**
```json
{
  "training": {
    "batch_size": 64,
    "num_epochs": 5,
    "learning_rate": 5e-5,
    "gradient_accumulation_steps": 2
  }
}
```

## Expected Data Format

### Text Files (.txt)
```
One sentence per line
Vietnamese example: xin chào thế giới
Chinese example: 你好世界
```

### JSONL Files (.jsonl)
```json
{"text": "Document content here..."}
{"content": "Another document..."}
{"body": "Third document..."}
```

## Output Files

After training, outputs are in:

```
checkpoints/
├── best/
│   ├── model.pt                  # Best model weights
│   ├── optimizer.pt              # Optimizer state
│   └── config.json               # Training config
└── epoch_1/
    └── ...                        # Epoch checkpoints

outputs/
├── unigram_tokenizer.model       # Tokenizer
└── training_logs.txt             # Training logs
```

## Using the Trained Model

### Inference
```python
from inference import ViWordFormerInference

inf = ViWordFormerInference(
    config_path="./checkpoints/best/config.json",
    model_path="./checkpoints/best/model.pt",
    tokenizer_path="./outputs/unigram_tokenizer"
)

result = inf.predict("xin chào thế giới")
```

## Troubleshooting

### Out of Memory
- Reduce `batch_size` in config (e.g., 32 → 16)
- Enable `gradient_accumulation_steps: 2`

### Slow Processing
- Check dataset files are on fast storage (SSD)
- Increase `num_workers` in train.py

### Tokenizer Issues
```bash
# Verify tokenizer works
python -c "from tokenizer.unigram_tokenizer import UnigramTokenizer; print('OK')"
```

### Missing Files
```bash
# Check output structure
ls -R outputs/
ls -R processed_data/
ls -R checkpoints/
```

## Advanced Tips

### Monitor Training
- Check `checkpoints/` for periodic saves
- Loss should decrease over time
- Save best model automatically

### Resume Training
Edit `train.py` to load checkpoint:
```python
checkpoint = torch.load("./checkpoints/epoch_2/model.pt")
model.load_state_dict(checkpoint)
```

### Export Model
```python
# Save for deployment
torch.save(model.state_dict(), "model_final.pt")
torch.onnx.export(model, dummy_input, "model.onnx")
```

## Dataset Statistics

Monitor processing with generated statistics:

```bash
cat processed_data/stats_vi.json
cat processed_data/stats_zh.json
```

## Next Steps

1. **Fine-tune** on downstream tasks (NER, QA, Classification)
2. **Quantize** model for deployment
3. **Export** to ONNX format
4. **Evaluate** on benchmarks

---

**Documentation**: See README.md for detailed information
**Examples**: Check scripts_*.py for usage patterns

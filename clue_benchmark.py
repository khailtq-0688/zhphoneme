import os
import json
import numpy as np
from typing import List, Dict, Optional, Tuple
import sys
from pathlib import Path
from argparse import ArgumentParser
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from inference_advanced_mlm import PinyinBertInferenceMLM


class SimpleClassifier(nn.Module):
    """Neural classifier for phoneme features"""
    def __init__(self, input_dim: int = 150, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)


class CLUEBenchmark:
    """CLUE benchmark with trained phoneme classifiers"""
    
    TASKS = {
        "TNEWS": ("TNEWS", 15),
        "IFLYTEK": ("IFLYTEK", 119),
        "AFQMC": ("AFQMC", 2),
        "CSL": ("CSL", 2),
    }
    
    def __init__(self, checkpoint_dir: str, data_dir: str):
        self.checkpoint_dir = checkpoint_dir
        self.data_dir = data_dir
        self.inference = PinyinBertInferenceMLM(checkpoint_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"✓ PinyinBert CLUE Benchmark")
        print(f"✓ Checkpoint: {checkpoint_dir}")
        print(f"✓ Data: {data_dir}")
        print(f"✓ Device: {self.device}")
        print()
    
    def load_task_data(self, task_name: str, max_samples: Optional[int] = None) -> List[Dict]:
        """Load CLUE task data from JSON (support both lowercase and uppercase folder names)"""
        task_dir, _ = self.TASKS[task_name]
        # Try uppercase, then lowercase
        train_file = os.path.join(self.data_dir, task_dir, "train.json")
        if not os.path.exists(train_file):
            train_file = os.path.join(self.data_dir, task_dir.lower(), "train.json")
        if not os.path.exists(train_file):
            train_file = os.path.join(self.data_dir, task_dir.upper(), "train.json")
        if not os.path.exists(train_file):
            print(f"⚠️  {train_file} not found")
            return []
        
        samples = []
        try:
            with open(train_file, 'r', encoding='utf-8') as f:
                for line_idx, line in enumerate(f):
                    if not line.strip():
                        continue
                    
                    data = json.loads(line.strip())
                    
                    # Parse per-task format
                    if task_name == "TNEWS":
                        text = data.get('sentence', '')
                        label = data.get('label')
                    elif task_name == "IFLYTEK":
                        text = data.get('sentence', '')
                        label = data.get('label')
                    elif task_name == "AFQMC":
                        s1 = data.get('sentence1', '')
                        s2 = data.get('sentence2', '')
                        text = f"{s1} | {s2}"
                        label = data.get('label')
                    elif task_name == "CSL":
                        abst = data.get('abst', '')
                        keywords = ', '.join(data.get('keywords', []))
                        text = f"{abst} [{keywords}]"
                        label = data.get('label')
                    else:
                        continue
                    
                    # Convert label to int if string
                    if isinstance(label, str):
                        try:
                            label = int(label)
                        except:
                            continue
                    
                    if text.strip() and label is not None:
                        samples.append({"text": text, "label": label})
                    
                    if max_samples and len(samples) >= max_samples:
                        break
        
        except Exception as e:
            print(f"Error loading {train_file}: {e}")
            return []
        
        return samples
    
    def extract_features(self, texts: List[str], batch_size: int = 1000) -> np.ndarray:
        """Extract phoneme features for texts in batches (default 1000)"""
        features_list = []
        total = len(texts)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch = texts[start:end]
            print(f"  Extracting [{start+1}-{end}/{total}]", end='\r')
            try:
                # Nếu model hỗ trợ batch, dùng extract(batch), nếu không thì fallback từng câu
                if hasattr(self.inference, 'extract'):
                    batch_results = self.inference.extract(batch)
                else:
                    batch_results = [self.inference.infer_single(text, return_confidence=True, seed=42) for text in batch]
            except Exception as e:
                print(f"\nBatch inference error: {e}")
                batch_results = [None] * len(batch)

            for result in batch_results:
                try:
                    features = []
                    for d in result['detailed_predictions']:
                        conf = d['confidence']
                        features.append([
                            conf.get('onset', 0.0),
                            conf.get('rhyme', 0.0),
                            conf.get('tone', 0.0)
                        ])
                    features = np.array(features)
                    # Pad to 50 characters
                    if len(features) < 50:
                        features = np.vstack([features, np.zeros((50 - len(features), 3))])
                    else:
                        features = features[:50]
                    features_list.append(features.flatten())
                except:
                    features_list.append(np.zeros(150))
        print()  # newline
        return np.array(features_list)
    
    def train_and_evaluate(self, task_name: str, samples: List[Dict],
                          epochs: int = 3, batch_size: int = 32) -> float:
        """Train classifier and evaluate on validation set"""
        
        if len(samples) < 20:
            print(f"⚠️  {task_name}: Not enough samples ({len(samples)})")
            return 0.0
        
        # Split: 80% train, 20% eval
        n_train = max(10, int(0.8 * len(samples)))
        np.random.shuffle(samples)
        train_data = samples[:n_train]
        eval_data = samples[n_train:]
        
        print(f"\n{task_name}")
        print(f"  Train: {len(train_data)}, Eval: {len(eval_data)}")
        
        # Extract features
        print("  Extracting train features...")
        train_texts = [s['text'] for s in train_data]
        train_labels = np.array([s['label'] for s in train_data])
        train_features = self.extract_features(train_texts)
        
        print("  Extracting eval features...")
        eval_texts = [s['text'] for s in eval_data]
        eval_labels = np.array([s['label'] for s in eval_data])
        eval_features = self.extract_features(eval_texts)
        
        # Map labels to 0..num_classes
        unique_labels = np.unique(train_labels)
        label_map = {int(old): new for new, old in enumerate(unique_labels)}
        train_labels_mapped = np.array([label_map[int(l)] for l in train_labels])
        eval_labels_mapped = np.array([label_map[int(l)] for l in eval_labels])
        
        num_classes = len(unique_labels)
        
        # Train
        print(f"  Training classifier ({num_classes} classes)...")
        
        model = SimpleClassifier(input_dim=150, num_classes=num_classes).to(self.device)
        optimizer = Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        train_dataset = TensorDataset(
            torch.FloatTensor(train_features),
            torch.LongTensor(train_labels_mapped)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=24)
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for features_batch, labels_batch in train_loader:
                features_batch = features_batch.to(self.device)
                labels_batch = labels_batch.to(self.device)
                
                optimizer.zero_grad()
                logits = model(features_batch)
                loss = criterion(logits, labels_batch)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            print(f"    Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            eval_features_tensor = torch.FloatTensor(eval_features).to(self.device)
            logits = model(eval_features_tensor)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
        
        accuracy = np.mean(preds == eval_labels_mapped)
        accuracy_percent = accuracy * 100
        print(f"  Accuracy: {accuracy_percent:.2f}%")
        
        return accuracy_percent
    
    def benchmark(self, tasks: List[str] = None, train_samples: Optional[int] = None,
                  epochs: int = 3):
        """Run complete benchmark"""
        
        if tasks is None:
            tasks = list(self.TASKS.keys())
        
        print("=" * 80)
        print("PINYINBERT CLUE BENCHMARK")
        print("=" * 80)
        print()
        
        self.inference._print_model_stats()
        
        results = {}
        
        for task in tasks:
            if task not in self.TASKS:
                continue
            
            # Load data
            print(f"\n[Loading {task}]")
            data = self.load_task_data(task, max_samples=train_samples)
            
            if len(data) < 20:
                print(f"⚠️  {task}: Insufficient data ({len(data)} samples)")
                results[task] = 0.0
                continue
            
            print(f"  ✓ Loaded {len(data)} samples")
            
            # Train and evaluate
            accuracy = self.train_and_evaluate(task, data, epochs=epochs, batch_size=32)
            results[task] = accuracy
        
        # Print summary
        self._print_summary(results, tasks)
        
        return results
    
    def _print_summary(self, results: Dict[str, float], tasks: List[str]):
        """Print results in paper format"""
        print("\n" + "=" * 80)
        print("BENCHMARK RESULTS")
        print("=" * 80)
        print(f"{'Task':<20} {'Accuracy (%)':<20}")
        print("-" * 80)
        
        accuracies = []
        for task in ["TNEWS", "IFLYTEK", "AFQMC", "CSL"]:
            if task in results:
                acc = results[task]
                print(f"{task:<20} {acc:<20.2f}")
                accuracies.append(acc)
        
        print("-" * 80)
        if accuracies:
            avg = np.mean(accuracies)
            print(f"{'AVERAGE':<20} {avg:<20.2f}")
        print("=" * 80)
        print()


if __name__ == "__main__":
    parser = ArgumentParser(description="PinyinBert CLUE Benchmark")
    parser.add_argument('--checkpoint', type=str, default='pinyin_bert_weights/pinyin_bert_base',
                       help='Model checkpoint path')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to CLUE dataset folder (e.g., /home/jovyan/zhphoneme/infer_data/CLUE)')
    parser.add_argument('--tasks', nargs='+', default=['TNEWS', 'IFLYTEK', 'AFQMC', 'CSL'],
                       help='Tasks to benchmark')
    parser.add_argument('--train_samples', type=int, default=None,
                       help='Max samples per task for training')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Training epochs')
    
    args = parser.parse_args()
    
    benchmark = CLUEBenchmark(args.checkpoint, data_dir=args.data_dir)
    results = benchmark.benchmark(tasks=args.tasks, train_samples=args.train_samples,
                                  epochs=args.epochs)
    
    print("✅ Benchmark Complete!")

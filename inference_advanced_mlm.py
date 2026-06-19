import torch
import numpy as np
from typing import List, Dict, Tuple
import json
import os
import random

from configs.pinyin_bert_config import PinyinBertConfig
from vocabs.pinyin_tokenizer import PinyinTokenizer
from models.pinyin_bert import PinyinBert
from data_utils.pinyin_dataset import collate_fn, PinyinEncodedTokens


class PinyinBertInferenceMLM:
    """MLM-based inference engine (15% masking) - FIXED"""
    
    MLM_PROB = 0.15  # Match training
    
    def __init__(self, checkpoint_dir: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = checkpoint_dir
        
        # Load config
        config_path = os.path.join(checkpoint_dir, "config.json")
        self.config = PinyinBertConfig.from_pretrained(config_path)
        
        # Load tokenizer
        self.tokenizer = PinyinTokenizer(self.config)
        
        # Load model
        self.model = PinyinBert(self.config)
        self.model = self.model.to(self.device)
        
        # Load weights
        model_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
        if not os.path.exists(model_path):
            model_path = os.path.join(checkpoint_dir, "model.safetensors")
        
        if os.path.exists(model_path):
            try:
                if model_path.endswith('.bin'):
                    state_dict = torch.load(model_path, map_location=self.device)
                else:
                    from safetensors.torch import load_file
                    state_dict = load_file(model_path)
                self.model.load_state_dict(state_dict, strict=False)
                print(f"✓ Loaded model weights from {model_path}")
            except Exception as e:
                print(f"⚠ Could not load model weights: {e}")
        
        self.model.eval()
        self.id2label = self.config.id2label
        self.label2id = self.config.label2id
    
    def _mask_input(self, input_ids: torch.Tensor, mask_prob: float = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mask 15% of input tokens for MLM"""
        if mask_prob is None:
            mask_prob = self.MLM_PROB
        
        masked_input = input_ids.clone()
        mask_indices = torch.zeros(input_ids.shape[0], dtype=torch.bool)
        
        # Don't mask CLS token (position 0)
        for idx in range(1, input_ids.shape[0]):
            if random.random() < mask_prob:
                mask_indices[idx] = True
                masked_input[idx, :] = self.config.mask_token_id  # Set all 3 components to mask
        
        return masked_input, mask_indices
    
    def _phoneme_to_string(self, onset_id: int, rhyme_id: int, tone_id: int) -> Tuple[str, str, str, str]:
        """Convert phoneme IDs to string syllable"""
        onset = self.id2label.get(onset_id, "<unk>")
        rhyme = self.id2label.get(rhyme_id, "<unk>")
        tone = self.id2label.get(tone_id, "<unk>")
        
        # Handle mask tokens
        if onset == "<mask>":
            onset = ""
        if rhyme == "<mask>":
            rhyme = ""
        if tone == "<mask>":
            tone = ""
        
        syllable = ""
        if onset and onset not in ["<pad>", "<empty>", "<unk>"]:
            syllable += onset
        if rhyme and rhyme not in ["<pad>", "<empty>", "<unk>"]:
            syllable += rhyme
        if tone and tone not in ["<pad>", "<empty>", "<unk>"]:
            syllable += tone
        
        return onset, rhyme, tone, syllable
    
    def _get_confidence(self, logits: torch.Tensor, position: int, component: int) -> float:
        """Get confidence score for prediction"""
        if position >= logits.shape[0] or component >= logits.shape[1]:
            return 0.0
        
        logit = logits[position, component]
        probs = torch.softmax(logit, dim=-1)
        max_prob = probs.max().item()
        return max_prob
    
    def infer_single(self, text: str, return_confidence: bool = True, 
                     return_top_k: int = 1, mask_prob: float = None, seed: int = None) -> Dict:
        """Inference on single text with MLM masking"""
        
        if seed is not None:
            random.seed(seed)
        
        with torch.no_grad():
            # Tokenize
            encoded = self.tokenizer(text)
            original_ids = encoded['input_ids'].clone().to(self.device)  # (seq_len, 3)
            
            # Mask 15% of tokens
            masked_input, mask_indices = self._mask_input(original_ids, mask_prob)
            masked_input_batch = masked_input.unsqueeze(0)  # (1, seq_len, 3)
            
            # Forward pass on masked input
            output = self.model(masked_input_batch)
            logits = output['logits']  # (1, seq_len, 3, vocab_size)
            
            # Get predictions from model
            predictions = logits.argmax(dim=-1)  # (1, seq_len, 3)
            
            # Construct output:
            # - For masked positions: use model predictions
            # - For unmasked positions: keep original input
            output_ids = original_ids.clone()
            for pos in range(output_ids.shape[0]):
                if mask_indices[pos]:
                    output_ids[pos, 0] = predictions[0, pos, 0]
                    output_ids[pos, 1] = predictions[0, pos, 1]
                    output_ids[pos, 2] = predictions[0, pos, 2]
            
            # Decode
            syllables = []
            details = []
            
            for pos in range(1, output_ids.shape[0]):  # Skip CLS
                if output_ids[pos, 0] == self.config.pad_token_id:
                    break
                
                onset_id = output_ids[pos, 0].item()
                rhyme_id = output_ids[pos, 1].item()
                tone_id = output_ids[pos, 2].item()
                
                onset, rhyme, tone, syllable = self._phoneme_to_string(onset_id, rhyme_id, tone_id)
                
                # Only add if syllable not empty
                if syllable:
                    syllables.append(syllable)
                
                detail = {
                    "position": pos,
                    "syllable": syllable if syllable else "[EMPTY]",
                    "onset": onset,
                    "rhyme": rhyme,
                    "tone": tone,
                    "was_masked": mask_indices[pos].item(),
                }
                
                if return_confidence:
                    detail["confidence"] = {
                        "onset": self._get_confidence(logits[0], pos, 0),
                        "rhyme": self._get_confidence(logits[0], pos, 1),
                        "tone": self._get_confidence(logits[0], pos, 2),
                    }
                    # Average confidence
                    confs = [detail["confidence"]["onset"], 
                            detail["confidence"]["rhyme"], 
                            detail["confidence"]["tone"]]
                    detail["confidence"]["avg"] = np.mean(confs)
                
                if return_top_k > 1 and mask_indices[pos]:
                    detail["top_k"] = {
                        "onset": self._get_top_k_predictions(logits[0, pos, 0], return_top_k),
                        "rhyme": self._get_top_k_predictions(logits[0, pos, 1], return_top_k),
                        "tone": self._get_top_k_predictions(logits[0, pos, 2], return_top_k),
                    }
                
                details.append(detail)
            
            masked_count = mask_indices[1:].sum().item()
            total_count = max(1, len(details))
            
            return {
                "input_text": text,
                "output_syllables": "".join(syllables),
                "detailed_predictions": details,
                "num_syllables": len(syllables),
                "num_masked": int(masked_count),
                "mask_percentage": (masked_count / total_count) * 100 if total_count > 0 else 0,
                "avg_confidence": np.mean([d.get("confidence", {}).get("avg", 0) for d in details]) if return_confidence else None
            }
    
    def _get_top_k_predictions(self, logits_1d: torch.Tensor, k: int = 3) -> List[Dict]:
        """Get top-K predictions"""
        probs = torch.softmax(logits_1d, dim=-1)
        top_probs, top_ids = torch.topk(probs, min(k, len(probs)))
        
        results = []
        for prob, idx in zip(top_probs, top_ids):
            label = self.id2label.get(idx.item(), "<unk>")
            results.append({
                "label": label,
                "probability": prob.item()
            })
        return results
    
    def infer_batch(self, texts: List[str], batch_size: int = 16, 
                   return_confidence: bool = True, verbose: bool = True) -> List[Dict]:
        """Batch inference with MLM"""
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            if verbose:
                print(f"Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            
            for text in batch_texts:
                result = self.infer_single(text, return_confidence=return_confidence, return_top_k=1)
                results.append(result)
        
        return results
    
    def infer_interactive(self):
        """Interactive inference mode"""
        print("\n" + "="*70)
        print("PinyinBert MLM Inference (15% masking)")
        print("="*70)
        print("Commands:")
        print("  - Type Chinese text to infer")
        print("  - Type 'exit' to quit")
        print("  - Type 'stats' to show model info")
        print("="*70 + "\n")
        
        while True:
            user_input = input(">>> ").strip()
            
            if user_input.lower() == "exit":
                print("Goodbye!")
                break
            
            elif user_input.lower() == "stats":
                self._print_model_stats()
            
            elif user_input:
                result = self.infer_single(user_input, return_confidence=True, return_top_k=2)
                self._print_result(result)
    
    def _print_result(self, result: Dict):
        """Print result nicely"""
        print("\n" + "-"*70)
        print(f"Input:  {result['input_text']}")
        print(f"Output: {result['output_syllables']}")
        print(f"Syllables: {result['num_syllables']} (Masked: {result['num_masked']}, {result['mask_percentage']:.1f}%)")
        
        if result.get('avg_confidence') is not None:
            print(f"Average Confidence: {result['avg_confidence']:.4f}")
        
        print("\nDetailed Predictions:")
        for detail in result['detailed_predictions']:
            masked_tag = "🔲 [MASKED]" if detail['was_masked'] else "✓"
            print(f"\n  [{detail['position']}] {masked_tag} {detail['syllable']}")
            print(f"    Onset: {detail['onset']}, Rhyme: {detail['rhyme']}, Tone: {detail['tone']}")
            
            if 'confidence' in detail:
                conf = detail['confidence']
                print(f"    Confidence: {conf['avg']:.4f} (O:{conf['onset']:.3f} R:{conf['rhyme']:.3f} T:{conf['tone']:.3f})")
        
        print("-"*70 + "\n")
    
    def _print_model_stats(self):
        """Print model statistics"""
        print("\n" + "="*70)
        print("Model Statistics")
        print("="*70)
        print(f"Task: Masked Language Model (MLM)")
        print(f"Mask Probability: {self.MLM_PROB * 100:.0f}%")
        print(f"Model Type: {self.config.model_type}")
        print(f"Hidden Size: {self.config.hidden_size}")
        print(f"Vocab Size: {self.config.vocab_size}")
        print(f"Device: {self.device}")
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total Parameters: {total_params:,}")
        print("="*70 + "\n")


if __name__ == "__main__":
    CHECKPOINT_DIR = "pinyin_bert_weights/pinyin_bert_base"
    
    print("Loading MLM model...")
    inferencer = PinyinBertInferenceMLM(CHECKPOINT_DIR)
    inferencer._print_model_stats()
    
    # Test with fixed seed for reproducibility
    print("\n" + "="*70)
    print("Test: MLM Inference with Fixed Seed")
    print("="*70)
    result = inferencer.infer_single("你好世界", return_confidence=True, return_top_k=2, seed=42)
    inferencer._print_result(result)
    
    # Batch test
    print("\n" + "="*70)
    print("Batch Test")
    print("="*70)
    texts = ["中国", "人民", "深度学习"]
    results = inferencer.infer_batch(texts, batch_size=2)
    for res in results:
        print(f"Input: {res['input_text']:<15} Output: {res['output_syllables']:<20} "
              f"Masked: {res['num_masked']}/{res['num_syllables']} Conf: {res.get('avg_confidence', 0):.4f}")

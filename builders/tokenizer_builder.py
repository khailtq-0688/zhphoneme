"""
Tokenizer builder for pretraining
"""

from .registry import META_TOKENIZER
from tokenizer.unigram_tokenizer import UnigramTokenizer


def build_tokenizer(config):
    """
    Build a tokenizer from configuration
    
    Args:
        config: Tokenizer configuration
        
    Returns:
        Tokenizer instance
    """
    tokenizer_type = config.get('model_type', 'unigram').lower()
    
    if tokenizer_type == 'unigram':
        tokenizer = UnigramTokenizer(
            model_prefix=config.get('model_prefix', './outputs/unigram_tokenizer'),
            vocab_size=config.get('vocab_size', 30000)
        )
        
        # Try to load if exists
        try:
            tokenizer.load(config.get('model_prefix', './outputs/unigram_tokenizer'))
        except Exception as e:
            print(f"Warning: Could not load tokenizer: {e}")
        
        return tokenizer
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")

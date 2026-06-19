import json

import torch
from transformers.utils import logging
from tokenizers import Encoding

from vocabs.hanzi_processing import HanziProcessor

logger = logging.get_logger(__name__)

class HanziEncoding(Encoding):
    def __init__(self, 
                 raw_text: str, 
                 unk_token: str,
                 sep_token: str,
                 pad_token: str,
                 cls_token: str,
                 mask_token: str,
                 ipa_vocab: dict[str, int],
                 radical_vocab: dict[str, int],
                 processor: HanziProcessor
                ):
        
        self.unk_token = unk_token
        self.sep_token = sep_token
        self.pad_token = pad_token
        self.cls_token = cls_token
        self.mask_token = mask_token

        self.unk_idx = ipa_vocab[unk_token]
        self.sep_idx = ipa_vocab[sep_token]
        self.pad_idx = ipa_vocab[pad_token]
        self.cls_idx = ipa_vocab[cls_token]
        self.mask_idx = ipa_vocab[mask_token]

        self._radical_ids = []
        self._radicals = []

        self._ipa_ids = []
        self._ipas = []
        
        raw_text = raw_text.lower()
        characters = processor.process_sentence(raw_text)
        for character in characters:
            ipa = character[ipa]
            # This is a Hanzi character
            if ipa:
                ipa_components = character["ipa_components"]
                self._ipas.append(ipa_components)
                ipa_id = [ipa_vocab[ipa] for ipa in ipa_components]
                self._ipa_ids.append(ipa_id)

                radicals = character["radicals"]
                self._radicals.append(radicals)
                radical_id = [radical_vocab[radical] for radical in radicals]
                self._radical_ids.append(radical_id)
            # This is not a Hanzi character. For instance, a Latin one (e.g. a, /, 1, 10)
            else:
                char = character["hanzi"]
                char_id = radical_vocab[char]
                self._ipa_ids.append([char_id]*3)
                self._radical_ids.append([char_id])

    @property
    def attention_mask(self):
        """
        The attention mask

        This indicates to the LM which tokens should be attended to, and which should not.
        This is especially important when batching sequences, where we need to applying
        padding.

        Returns:
           :obj:`torch.Tensor (seq_len, )`: The attention mask
        """

        sequence = torch.Tensor([ipa_id[0] for ipa_id in self._ipa_ids])
        mask = (sequence == self.pad_idx)

        return mask


    @property
    def ipa_ids(self):
        """
        The generated IDs

        The IDs are the main input to a Language Model. They are the token indices,
        the numerical representations that a LM understands.

        Returns:
            :obj:`torch.Tensor (seq_len, 3)`: The list of IDs
        """
        return torch.Tensor(self._ipa_ids)

    @property
    def radical_ids(self):
        """
        The generated IDs

        The IDs are the main input to a Language Model. They are the token indices,
        the numerical representations that a LM understands.

        Returns:
            :obj:`torch.Tensor (seq_len, max_radical)`: The list of IDs
        """
        pass
        

    def pad(self, length, direction="right", pad_id=0, pad_type_id=0, pad_token="[PAD]"):
        """
        Pad the :class:`~tokenizers.Encoding` at the given length

        Args:
            length (:obj:`int`):
                The desired length

            direction: (:obj:`str`, defaults to :obj:`right`):
                The expected padding direction. Can be either :obj:`right` or :obj:`left`

            pad_id (:obj:`int`, defaults to :obj:`0`):
                The ID corresponding to the padding token

            pad_type_id (:obj:`int`, defaults to :obj:`0`):
                The type ID corresponding to the padding token

            pad_token (:obj:`str`, defaults to `[PAD]`):
                The pad token to use
        """
        pass

    @property
    def special_tokens_mask(self):
        """
        The special token mask

        This indicates which tokens are special tokens, and which are not.

        Returns:
            :obj:`List[int]`: The special tokens mask
        """
        pass

    @property
    def ipa_tokens(self):
        """
        The generated tokens

        They are the string representation of the IDs.

        Returns:
            :obj:`List[str]`: The list of tokens
        """
        pass

    @property
    def radical_tokens(self):
        """
        The generated tokens

        They are the string representation of the IDs.

        Returns:
            :obj:`List[str]`: The list of tokens
        """
        pass

    def truncate(self, max_length):
        """
        Truncate the :class:`HanziEncoding` at the given length

        If this :class:`HanziEncoding` represents multiple sequences, when truncating
        this information is lost. It will be considered as representing a single sequence.

        Args:
            max_length (:obj:`int`):
                The desired length
        """
        pass

class HanziBertTokenizer:
    r"""
    Construct a HanziBERT tokenizer (backed by HuggingFace's tokenizers library).

    This tokenizer inherits from [`TokenizersBackend`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        vocab (`str`, *optional*):
            Custom vocabulary dictionary. If not provided, vocabulary is loaded from `vocab_file`.
        do_lower_case (`bool`, *optional*, defaults to `True`):
            Whether or not to lowercase the input when tokenizing.
        unk_token (`str`, *optional*, defaults to `"[UNK]"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        sep_token (`str`, *optional*, defaults to `"[SEP]"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        pad_token (`str`, *optional*, defaults to `"[PAD]"`):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (`str`, *optional*, defaults to `"[CLS]"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        mask_token (`str`, *optional*, defaults to `"[MASK]"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
    """

    def __init__(
        self,
        vocab: str | None = None,
        unk_token: str = "[UNK]",
        sep_token: str = "[SEP]",
        pad_token: str = "[PAD]",
        cls_token: str = "[CLS]",
        mask_token: str = "[MASK]",
        **kwargs,
    ):
        if vocab is None:
            vocab = {
                str(pad_token): 0,
                str(unk_token): 1,
                str(cls_token): 2,
                str(sep_token): 3,
                str(mask_token): 4,
            }
        else:
            vocab = json.load(open(vocab))
        
        self.vocab = vocab
        self.model = HanziProcessor()

    def encode(self, sequence, add_special_tokens=True):
        """
        Encode the given sequence. This method can process raw text sequences as well as already pre-tokenized sequences.

        Example:
            Here are some examples of the inputs that are accepted::

                encode("A single sequence")`
                encode("A sequence", "And its pair")`
                encode([ "A", "pre", "tokenized", "sequence" ], is_pretokenized=True)`
                encode(
                    [ "A", "pre", "tokenized", "sequence" ], [ "And", "its", "pair" ],
                    is_pretokenized=True
                )

        Args:
            sequence (:obj:`~tokenizers.InputSequence`):
                The main input sequence we want to encode.

            add_special_tokens (:obj:`bool`, defaults to :obj:`True`):
                Whether to add the special tokens

        Returns:
            :class:`HanziEncoding`: The encoded result

        """
        pass

        

        

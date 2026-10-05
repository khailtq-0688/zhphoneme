import torch

def create_roberta_mlm_labels(input_ids: torch.Tensor, config) -> tuple[torch.Tensor, torch.Tensor]:
    """RoBERTa/BERT dynamic masking: 15% of non-special tokens are selected; of those,
    80% become <mask>, 10% become a random vocab token, and 10% are left unchanged (but
    still count towards the loss, since the model must still predict them).

    `config` only needs `special_ids`, `mask_token_id` and `vocab_size`, so this is
    shared by every SegmentFreeBERT tokenizer regardless of how its vocabulary was
    built (hand-curated morphemes, a trained Unigram model, ...).
    """
    labels = input_ids.clone()

    special_ids = torch.tensor(config.special_ids, device=input_ids.device)
    special_tokens_mask = torch.isin(input_ids, special_ids)

    probability_matrix = torch.full(labels.shape, 0.15)
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100

    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    input_ids[indices_replaced] = config.mask_token_id

    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_ids = torch.randint(config.vocab_size, labels.shape, dtype=torch.long)
    input_ids[indices_random] = random_ids[indices_random]

    # the remaining 10% of masked_indices are left unchanged in input_ids

    return input_ids, labels

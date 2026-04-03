import torch
from torch.utils.data import Dataset

import random

def collate_phoneme_mlm(batch_texts):
    """
    Data Collator thực hiện việc chuyển Text -> Phoneme IDs bằng HanziProcessor 
    và áp dụng Dynamic Masking 15%
    """
    batch_size = len(batch_texts)
    max_len = max(len(t) for t in batch_texts) + 2 # +2 cho CLS và SEP
    
    # Khởi tạo ma trận tensor
    input_ids = torch.full((batch_size, max_len, 3), UNIFIED_VOCAB['[PAD]'], dtype=torch.long)
    labels = torch.full((batch_size, max_len, 3), -100, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    
    mask_id = UNIFIED_VOCAB['[MASK]']
    unk_id = UNIFIED_VOCAB['[UNK]']
    empty_id = UNIFIED_VOCAB['<EMPTY>']
    
    for i, text in enumerate(batch_texts):
        # 1. Gắn [CLS]
        input_ids[i, 0, :] = UNIFIED_VOCAB['[CLS]']
        attention_mask[i, 0] = 1
        
        for j, char in enumerate(text):
            pos = j + 1
            attention_mask[i, pos] = 1
            
            # 2. Sử dụng HanziProcessor để bóc tách (IPA)
            onset_val, rhyme_val, tone_val = "<EMPTY>", "<EMPTY>", "<EMPTY>"
            
            if '\u4e00' <= char <= '\u9fff':
                success, result = processor.process_IPA(char, number_components=3)
                if success:
                    _, (o, r, t) = result
                    onset_val = o if o else "<EMPTY>"
                    rhyme_val = r if r else "<EMPTY>"
                    tone_val  = t if t else "<EMPTY>"
            else:
                # Fallback cho dấu câu, số, chữ latin
                onset_val = char
                rhyme_val = char
                tone_val  = char
            
            # Map sang ID
            id_onset = UNIFIED_VOCAB.get(onset_val, unk_id)
            id_rhyme = UNIFIED_VOCAB.get(rhyme_val, unk_id)
            id_tone  = UNIFIED_VOCAB.get(tone_val, unk_id)
            
            true_ids = [id_onset, id_rhyme, id_tone]
            
            # 3. Áp dụng MLM (15% bị che)
            # Chỉ che khi nó là chữ Hán (id_onset != unk_id hoặc không phải dấu câu)
            # Ta che bất kỳ token nào không phải special token
            if random.random() < 0.15:
                prob = random.random()
                if prob < 0.8:
                    # 80%: Che bằng [MASK]
                    input_ids[i, pos, :] = mask_id
                elif prob < 0.9:
                    # 10%: Giữ nguyên ID gốc
                    input_ids[i, pos, :] = torch.tensor(true_ids)
                else:
                    # 10%: Thay bằng token ngẫu nhiên trong vocab
                    random_3_ids = torch.randint(6, VOCAB_SIZE, (3,)) # Bắt đầu từ 6 để né Special Tokens
                    input_ids[i, pos, :] = random_3_ids
                
                # Tính Loss cho vị trí này
                labels[i, pos, :] = torch.tensor(true_ids)
            else:
                # 85% giữ nguyên, không tính loss (-100)
                input_ids[i, pos, :] = torch.tensor(true_ids)
                
        # 4. Gắn [SEP]
        seq_end = len(text) + 1
        input_ids[i, seq_end, :] = UNIFIED_VOCAB['[SEP]']
        attention_mask[i, seq_end] = 1

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

class PhonemeDataset(Dataset):
    def __init__(self, texts, max_length=128):
        self.texts = texts
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]['text']
        # Cắt bớt text để chừa chỗ cho [CLS] và [SEP]
        text = text[:self.max_length - 2] 
        return text


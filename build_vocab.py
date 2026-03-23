import json
import os
from tqdm import tqdm
from hanzi_processing import HanziProcessor

def build_vocab_incremental(input_file_path, output_dir="vocabs", save_every=50000):
    processor = HanziProcessor()
    os.makedirs(output_dir, exist_ok=True)
    
    SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<EMPTY>"]
    
    def init_vocab():
        return {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}

    vocabs = {
        'onset': init_vocab(),
        'rhyme': init_vocab(),
        'tone': init_vocab(),
        'radical': init_vocab()
    }

    def add_to_vocab(vocab_type, item):
        if item not in vocabs[vocab_type]:
            vocabs[vocab_type][item] = len(vocabs[vocab_type])

    def save_vocabs():
        for v_name, v_data in vocabs.items():
            path = os.path.join(output_dir, f"{v_name}2id.json")
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(v_data, f, ensure_ascii=False, indent=4)

    print(f"Đang xử lý dữ liệu từ: {input_file_path}...")
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            
            for line_idx, line in enumerate(tqdm(f, desc="Processing lines")):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    processed_data = processor.process_sentence(line)
                    
                    for char_info in processed_data:
                        ipa_components = char_info.get('ipa_components')
                        char_radicals = char_info.get('radicals', [])
                        
                        if ipa_components:
                            initial, rhyme, tone = ipa_components
                            if initial: add_to_vocab('onset', initial)
                            if rhyme: add_to_vocab('rhyme', rhyme)
                            if tone: add_to_vocab('tone', tone)
                            
                        if char_radicals:
                            for rad in char_radicals:
                                if rad: add_to_vocab('radical', rad)
                                
                except Exception:
                    continue 
                
                if (line_idx + 1) % save_every == 0:
                    save_vocabs()

    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file {input_file_path}.")
        return

    save_vocabs()
    
    print("\n✅ HOÀN THÀNH: Đã quét xong toàn bộ dữ liệu!")
    print("📊 Thống kê kích thước từ điển cuối cùng:")
    print(f"   - Onsets:   {len(vocabs['onset'])}")
    print(f"   - Rhymes:   {len(vocabs['rhyme'])}")
    print(f"   - Tones:    {len(vocabs['tone'])}")
    print(f"   - Radicals: {len(vocabs['radical'])}")

if __name__ == "__main__":
    DATA_FILE = "data_cleaned/baidubaike_cleaned.txt" 
    OUTPUT_DIRECTORY = "vocabs"
    
    build_vocab_incremental(DATA_FILE, OUTPUT_DIRECTORY, save_every=2000)
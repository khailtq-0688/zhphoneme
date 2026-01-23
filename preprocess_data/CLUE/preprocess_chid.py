import re
import json
import os
import sys
from tqdm import tqdm
from collections import Counter

try:
    import cn2an
except ImportError:
    print("❌ LỖI: Cần cài đặt cn2an (pip install cn2an)")
    sys.exit(1)

HANZI_RANGE = r'\u4e00-\u9fff'
SYMBOL_MAP = {'+': '加', '＋': '加', '-': '减', '×': '乘以', '÷': '除以', '=': '等于', '√': '根号', '㎡': '平方米', '/': '每'}
CLEANING_REGEX = re.compile(f'[^{HANZI_RANGE}a-zA-Z0-9%]')
REJECT_DIGIT_REGEX = re.compile(r'\d')
# Regex đặc biệt để nhận diện và bảo vệ mã idiom (ví dụ: #idiom123456#)
IDIOM_PLACEHOLDER_REGEX = re.compile(r'#idiom\d+#')

def clean_text_preserving_placeholders(text: str):
    if not text: return "", []
    
    removed_chars = []
    # 1. Tìm và tạm thay thế các mã idiom bằng một token an toàn
    placeholders = IDIOM_PLACEHOLDER_REGEX.findall(text)
    temp_text = IDIOM_PLACEHOLDER_REGEX.sub('【TOKEN】', text)
    
    # 2. Làm sạch phần văn bản xung quanh
    temp_text = " ".join(temp_text.split())
    for s, h in SYMBOL_MAP.items():
        temp_text = temp_text.replace(s, h)
    
    try:
        temp_text = cn2an.transform(temp_text, "an2cn")
    except:
        pass
    
    garbage_found = CLEANING_REGEX.findall(temp_text)
    removed_chars.extend(garbage_found)
    temp_text = CLEANING_REGEX.sub('', temp_text)
    
    # 3. Đặt lại các mã idiom vào vị trí cũ
    for p in placeholders:
        temp_text = temp_text.replace('【TOKEN】', p, 1)
        
    return temp_text, removed_chars

def process_chid_file(input_path, output_dir):
    filename = os.path.basename(input_path)
    out_clean = os.path.join(output_dir, f"{filename.split('.')[0]}_clean.jsonl")
    out_report = os.path.join(output_dir, f"{filename.split('.')[0]}_report.json")
    
    stats = {"total": 0, "kept": 0, "rejected": 0, "reasons": {}}
    garbage_counter = Counter()
    cleaned_items = []

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f"Processing {filename}"):
            if not line.strip(): continue
            stats["total"] += 1
            item = json.loads(line)
            
            new_candidates = []
            for cand in item.get('candidates', []):
                c_clean, _ = clean_text_preserving_placeholders(cand)
                new_candidates.append(c_clean)
            
            new_contents = []
            error_in_item = False
            for paragraph in item.get('content', []):
                p_clean, p_rem = clean_text_preserving_placeholders(paragraph)
                # Chỉ kiểm tra số sót bên ngoài các mã #idiom#
                text_no_placeholders = IDIOM_PLACEHOLDER_REGEX.sub('', p_clean)
                if REJECT_DIGIT_REGEX.search(text_no_placeholders):
                    error_in_item = True
                new_contents.append(p_clean)
                garbage_counter.update(p_rem)

            if error_in_item:
                stats["rejected"] += 1
                stats["reasons"]["CONTAINS_DIGIT_ERROR"] = stats["reasons"].get("CONTAINS_DIGIT_ERROR", 0) + 1
            else:
                stats["kept"] += 1
                item['candidates'] = new_candidates
                item['content'] = new_contents
                cleaned_items.append(item)

    with open(out_clean, 'w', encoding='utf-8') as f:
        for item in cleaned_items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    report = {
        "task": "ChID",
        "filename": filename,
        "stats": stats,
        "kept_ratio": f"{(stats['kept']/stats['total']*100):.2f}%" if stats['total'] > 0 else "0.00%",
        "analysis": {"symbol_removed": dict(garbage_counter.most_common())}
    }
    with open(out_report, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    FILES = [os.path.join("chid_data", f) for f in ["train.json", "dev.json", "test1.0.json", "test1.1.json"]]
    OUTPUT_DIR = "chid_output"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for f in FILES:
        if os.path.exists(f): process_chid_file(f, OUTPUT_DIR)
    print(f"\n✅ HOÀN TẤT! Kết quả tại: {OUTPUT_DIR}")
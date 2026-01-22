import re
import json
import os
import sys
from tqdm import tqdm
from collections import Counter

try:
    import cn2an
    print("✅ Đã tải thành công thư viện 'cn2an'.")
except ImportError:
    print("❌ LỖI: Không tìm thấy thư viện 'cn2an'.")
    sys.exit(1)

HANZI_RANGE = r'\u4e00-\u9fff'
SYMBOL_MAP = {
    '+': '加', '＋': '加', '-': '减', '×': '乘以',
    '÷': '除以', '=': '等于', '√': '根号', '㎡': '平方米', '/': '每'
}

CLEANING_REGEX_PATTERN = f'[^{HANZI_RANGE}a-zA-Z0-9%]' 
CLEANING_REGEX = re.compile(CLEANING_REGEX_PATTERN)
REJECT_DIGIT_REGEX = re.compile(r'\d') 

def clean_text_with_report(text: str):
    """Xử lý văn bản và trả về list các ký tự bị xóa"""
    if not text: return "", []
    
    removed_chars = []
    text = " ".join(text.split())
    
    for symbol, hanzi in SYMBOL_MAP.items():
        text = text.replace(symbol, hanzi)
        
    try:
        text = cn2an.transform(text, "an2cn")
    except:
        pass
        
    garbage_matches = CLEANING_REGEX.findall(text)
    if garbage_matches:
        removed_chars.extend(garbage_matches)
    
    text = CLEANING_REGEX.sub('', text)
    return text, removed_chars

def process_afqmc_file(input_path, output_dir):
    filename = os.path.basename(input_path)
    name_only = os.path.splitext(filename)[0]
    
    out_clean = os.path.join(output_dir, f"{name_only}_clean.jsonl")
    out_rejected = os.path.join(output_dir, f"{name_only}_rejected.jsonl")
    out_report = os.path.join(output_dir, f"{name_only}_report.json")
    
    stats = { "total": 0, "kept": 0, "rejected": 0, "reasons": {} }
    counter_removed_garbage = Counter()

    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(out_clean, 'w', encoding='utf-8') as f_clean, \
         open(out_rejected, 'w', encoding='utf-8') as f_reject:

        for line in tqdm(fin, desc=f"Processing {name_only}"):
            line = line.strip()
            if not line: continue
            stats["total"] += 1
            try: item = json.loads(line)
            except: continue

            s1_clean, s1_removed = clean_text_with_report(item.get('sentence1', ''))
            s2_clean, s2_removed = clean_text_with_report(item.get('sentence2', ''))

            counter_removed_garbage.update(s1_removed)
            counter_removed_garbage.update(s2_removed)

            status = "OK"
            if REJECT_DIGIT_REGEX.search(s1_clean) or REJECT_DIGIT_REGEX.search(s2_clean):
                status = "CONTAINS_DIGIT_ERROR"
            elif not s1_clean or not s2_clean:
                status = "TOO_SHORT_OR_EMPTY"

            if status == "OK":
                stats["kept"] += 1
                item['sentence1'] = s1_clean
                item['sentence2'] = s2_clean
                f_clean.write(json.dumps(item, ensure_ascii=False) + '\n')
            else:
                stats["rejected"] += 1
                stats["reasons"][status] = stats["reasons"].get(status, 0) + 1
                f_reject.write(json.dumps({"orig": [item.get('sentence1'), item.get('sentence2')], "reason": status}, ensure_ascii=False) + '\n')

    report = {
        "task": "AFQMC",
        "filename": filename,
        "stats": stats,
        "kept_ratio": f"{(stats['kept']/stats['total']*100):.2f}%" if stats['total'] > 0 else "0%",
        "analysis": {
            "symbol_removed": dict(counter_removed_garbage.most_common())
        }
    }
    with open(out_report, 'w', encoding='utf-8') as f_rep:
        json.dump(report, f_rep, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    INPUT_FILES = ["train.json", "dev.json", "test.json"]
    OUTPUT_DIR = "afqmc_output"
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    for file_name in INPUT_FILES:
        if os.path.exists(file_name):
            process_afqmc_file(file_name, OUTPUT_DIR)
        else:
            print(f"⚠️ Không tìm thấy file: {file_name}")

    print(f"\n✅ HOÀN TẤT! Kết quả và báo cáo đã xuất ra thư mục: {OUTPUT_DIR}")
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
    print("👉 Vui lòng chạy lệnh: pip install cn2an")
    sys.exit(1)

HANZI_RANGE = r'\u4e00-\u9fff'

SYMBOL_MAP = {
    '-': '至',  
    '&': '和'   
}

CLEANING_REGEX_PATTERN = f'[^{HANZI_RANGE}a-zA-Z0-9%/]' 
CLEANING_REGEX = re.compile(CLEANING_REGEX_PATTERN)

REJECT_DIGIT_REGEX = re.compile(r'\d') 

TARGET_FIELD = 'target'

def translate_symbols(text: str) -> str:
    """Dịch các ký tự toán học/đơn vị sang Hán tự"""
    for symbol, hanzi in SYMBOL_MAP.items():
        if symbol in text:
            text = text.replace(symbol, hanzi)
    return text

def convert_numbers_and_percent(text: str) -> str:
    """
    Chuyển đổi số VÀ phần trăm sang chữ Hán.
    cn2an hỗ trợ mode 'an2cn' xử lý tốt cả '50%' -> '百分之五十'
    """
    def callback(match):
        val = match.group()
        try:
            return cn2an.transform(val, "an2cn")
        except:
            return val
            
    return re.sub(r'\d+(?:\.\d+)?%?', callback, text)

def process_line(item):
    """Pipeline: Translate Symbols -> Convert Number/% -> Clean"""
    
    removed_chars_in_this_line = []

    # Check Bad Source
    if item.get("is_bad_source") is True:
        return None, "BAD_SOURCE_TAG", []

    original_text = item.get(TARGET_FIELD, "")
    if not original_text:
        return None, "EMPTY_TARGET", []

    text = " ".join(original_text.split()) 

    text = translate_symbols(text)

    text = convert_numbers_and_percent(text)

    garbage_matches = CLEANING_REGEX.findall(text)
    if garbage_matches:
        removed_chars_in_this_line.extend(garbage_matches)
    text = CLEANING_REGEX.sub('', text)

    # Nếu còn số sót lại -> Reject
    if REJECT_DIGIT_REGEX.search(text):
        return text, "CONTAINS_DIGIT_ERROR", removed_chars_in_this_line
    
    if '%' in text: 
        return text, "CONTAINS_PERCENT_ERROR", removed_chars_in_this_line

    if len(text) < 1:
        return None, "TOO_SHORT", removed_chars_in_this_line

    # Phải có ít nhất 1 Hán tự
    if not re.search(f'[{HANZI_RANGE}]', text):
        return None, "NO_HANZI_FOUND", removed_chars_in_this_line

    return text, "OK", removed_chars_in_this_line

def preprocess_wmt_dataset(input_path, output_dir):
    filename = os.path.basename(input_path)
    name_only = os.path.splitext(filename)[0]
    
    out_pure = os.path.join(output_dir, f"{name_only}_clean.jsonl")
    out_rejected = os.path.join(output_dir, f"{name_only}_rejected.jsonl")
    out_report = os.path.join(output_dir, f"{name_only}_report.json")
    
    print(f"\n🚀 BẮT ĐẦU XỬ LÝ: {input_path}")
    print("ℹ️ Dịch Symbol & Số sang Hán tự (Pure Hanzi Target)")
    
    stats = { "total": 0, "kept": 0, "rejected": 0, "reasons": {} }
    counter_removed_garbage = Counter()

    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(out_pure, 'w', encoding='utf-8') as f_pure, \
         open(out_rejected, 'w', encoding='utf-8') as f_reject:

        for line in tqdm(fin, desc="Processing"):
            line = line.strip()
            if not line: continue
            stats["total"] += 1
            try: item = json.loads(line)
            except: continue

            processed_text, status, removed_chars = process_line(item)

            if removed_chars: counter_removed_garbage.update(removed_chars)

            if status == "OK":
                stats["kept"] += 1
                
                # Chỉ giữ lại source và target, bỏ các trường thừa
                new_item = {
                    "source": item.get("source", ""),
                    "target": processed_text
                }
                
                f_pure.write(json.dumps(new_item, ensure_ascii=False) + '\n')
            else:
                stats["rejected"] += 1
                stats["reasons"][status] = stats["reasons"].get(status, 0) + 1
                f_reject.write(json.dumps({"orig": item.get(TARGET_FIELD), "reason": status}, ensure_ascii=False) + '\n')

    # Report
    # Lấy toàn bộ danh sách ký tự bị xóa và sắp xếp giảm dần
    all_removed_sorted = dict(sorted(counter_removed_garbage.items(), key=lambda x: x[1], reverse=True))

    report = {
        "dataset": input_path,
        "stats": {
            "total": stats["total"], "kept": stats["kept"], "rejected": stats["rejected"],
            "kept_ratio": f"{(stats['kept']/stats['total']*100 if stats['total'] else 0):.2f}%"
        },
        "reasons": stats["reasons"],
        "analysis": { 
            "all_removed_symbols": all_removed_sorted 
        }
    }
    
    with open(out_report, 'w', encoding='utf-8') as f_rep:
        json.dump(report, f_rep, indent=4, ensure_ascii=False)

    print(f"\n✅ HOÀN TẤT!")
    print(f"   - File sạch:    {out_pure}")
    print(f"   - File bị loại: {out_rejected}")
    print(f"   - Báo cáo:      {out_report}")

if __name__ == "__main__":
    INPUT_FILE = "en-zh_CN.jsonl" 
    OUTPUT_DIR = "wmt_output"
    
    if not os.path.exists(OUTPUT_DIR): 
        os.makedirs(OUTPUT_DIR)
    
    if os.path.exists(INPUT_FILE):
        preprocess_wmt_dataset(INPUT_FILE, OUTPUT_DIR)
    else:
        print(f"⚠️ Không tìm thấy file: {INPUT_FILE}")
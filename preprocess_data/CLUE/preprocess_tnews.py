import re
import json
import os
import sys
from tqdm import tqdm
from collections import Counter

# Kiểm tra thư viện cn2an
try:
    import cn2an
    print("   Đã tải thành công thư viện 'cn2an'.")
except ImportError:
    print("   LỖI: Không tìm thấy thư viện 'cn2an'.")
    print("   Vui lòng cài đặt bằng lệnh: pip install cn2an")
    sys.exit(1)

# --- CẤU HÌNH REGEX VÀ MAP ---
HANZI_RANGE = r'\u4e00-\u9fff'
# Map các ký tự đặc biệt sang tiếng Trung có ý nghĩa
SYMBOL_MAP = {
    '+': '加', '＋': '加', '-': '减', '×': '乘以',
    '÷': '除以', '=': '等于', '√': '根号', '㎡': '平方米', 
    '/': '每', '%': '百分之' 
}

# Regex giữ lại: Hán tự, Tiếng Anh (a-z, A-Z), Số (0-9 - để check sau này), và %
# Lưu ý: Dấu chấm, phẩy,... sẽ bị xóa nếu không nằm trong SYMBOL_MAP
CLEANING_REGEX_PATTERN = f'[^{HANZI_RANGE}a-zA-Z0-9]' 
CLEANING_REGEX = re.compile(CLEANING_REGEX_PATTERN)

# Regex dùng để TỪ CHỐI dòng nếu vẫn còn số Ả Rập sau khi đã qua cn2an
REJECT_DIGIT_REGEX = re.compile(r'\d') 

def clean_text_with_report(text: str):
    """
    Quy trình:
    1. Map ký tự đặc biệt -> Hán tự.
    2. Chuyển số Ả Rập -> Hán số (cn2an).
    3. Xóa các ký tự rác (dấu câu còn lại).
    """
    if not text: return "", []
    
    removed_chars = []
    # Chuẩn hóa khoảng trắng
    text = " ".join(text.split())
    
    # 1. Map ký tự toán học/đơn vị
    for symbol, hanzi in SYMBOL_MAP.items():
        text = text.replace(symbol, hanzi)
        
    # 2. Chuyển số sang chữ (VD: 2018 -> 二零一八, 4.9 -> 四点九)
    try:
        text = cn2an.transform(text, "an2cn")
    except:
        # Nếu lỗi convert (hiếm), giữ nguyên để regex xử lý sau
        pass
        
    # 3. Thu thập các ký tự sẽ bị xóa (để thống kê report)
    garbage_matches = CLEANING_REGEX.findall(text)
    if garbage_matches:
        removed_chars.extend(garbage_matches)
    
    # 4. Xóa rác
    text = CLEANING_REGEX.sub('', text)
    return text, removed_chars

def process_tnews_file(input_path, output_dir):
    filename = os.path.basename(input_path)
    name_only = os.path.splitext(filename)[0]
    
    out_clean = os.path.join(output_dir, f"{name_only}_clean.jsonl")
    out_rejected = os.path.join(output_dir, f"{name_only}_rejected.jsonl")
    out_report = os.path.join(output_dir, f"{name_only}_report.json")
    
    stats = { "total": 0, "kept": 0, "rejected": 0, "reasons": {} }
    counter_removed_garbage = Counter() 

    print(f"Đang xử lý: {filename} ...")

    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(out_clean, 'w', encoding='utf-8') as f_clean, \
         open(out_rejected, 'w', encoding='utf-8') as f_reject:

        for line in tqdm(fin, desc=f"Progress"):
            line = line.strip()
            if not line: continue
            stats["total"] += 1
            
            try: 
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            # TNEWS dùng key "sentence"
            original_sentence = item.get('sentence', '')
            
            # Xử lý làm sạch
            clean_sent, removed = clean_text_with_report(original_sentence)
            counter_removed_garbage.update(removed)

            # --- VALIDATE ---
            status = "OK"
            
            # Kiểm tra 1: Còn sót số (Mục tiêu là thuần Hán/Latin)
            if REJECT_DIGIT_REGEX.search(clean_sent):
                status = "CONTAINS_DIGIT_ERROR" # Có thể do cn2an không convert được định dạng lạ
            
            # Kiểm tra 2: Câu quá ngắn hoặc rỗng sau khi clean
            elif not clean_sent or len(clean_sent) < 2: # Tùy chỉnh độ dài tối thiểu
                status = "TOO_SHORT_OR_EMPTY"

            if status == "OK":
                stats["kept"] += 1
                # Cập nhật lại câu đã clean vào object JSON
                item['sentence'] = clean_sent
                f_clean.write(json.dumps(item, ensure_ascii=False) + '\n')
            else:
                stats["rejected"] += 1
                stats["reasons"][status] = stats["reasons"].get(status, 0) + 1
                # Lưu log dòng bị loại bỏ
                reject_record = {
                    "orig": original_sentence, 
                    "clean_attempt": clean_sent,
                    "reason": status,
                    "id_or_label": item.get('id', item.get('label', 'unknown'))
                }
                f_reject.write(json.dumps(reject_record, ensure_ascii=False) + '\n')

    # Ghi báo cáo thống kê
    report = {
        "filename": filename,
        "stats": stats,
        "kept_ratio": f"{(stats['kept']/stats['total']*100):.2f}%" if stats['total'] > 0 else "0%",
        "analysis": {
            "symbols_removed": dict(counter_removed_garbage.most_common()) 
        }
    }
    with open(out_report, 'w', encoding='utf-8') as f_rep:
        json.dump(report, f_rep, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    INPUT_FILES = [
        "TNEWS\\train.json",
        "TNEWS\\dev.json",
        "TNEWS\\test.json",
        "TNEWS\\test1.0.json"
    ]
    
    OUTPUT_DIR = "TNEWS\\tnews_output"
    
    if not os.path.exists(OUTPUT_DIR): 
        os.makedirs(OUTPUT_DIR)
        
    found_any = False
    for file_name in INPUT_FILES:
        if os.path.exists(file_name): 
            process_tnews_file(file_name, OUTPUT_DIR)
            found_any = True
            
    if found_any:
        print(f"\nHOÀN TẤT. Kiểm tra thư mục '{OUTPUT_DIR}' để xem kết quả.")
    else:
        print(f"\nKhông tìm thấy file đầu vào nào trong danh sách: {INPUT_FILES}")
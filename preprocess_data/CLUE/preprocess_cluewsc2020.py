import re
import json
import os
import sys
import html
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

# Map các ký tự đặc biệt
SYMBOL_MAP = {
    '+': '加', '＋': '加', '-': '减', '×': '乘以',
    '÷': '除以', '=': '等于', '√': '根号', '㎡': '平方米', 
    '/': '每', '%': '百分之'
}

# Regex: Giữ lại Hán tự, Latin, Số (để check)
CLEANING_REGEX_PATTERN = f'[^{HANZI_RANGE}a-zA-Z0-9]' 
CLEANING_REGEX = re.compile(CLEANING_REGEX_PATTERN)
REJECT_DIGIT_REGEX = re.compile(r'\d') 

def clean_segment(text: str):
    """Hàm làm sạch cho từng đoạn nhỏ của câu"""
    if not text: return "", []
    
    removed_chars = []
    text = html.unescape(text) # Giải mã HTML entities
    
    # 1. Map ký tự
    for symbol, hanzi in SYMBOL_MAP.items():
        text = text.replace(symbol, hanzi)
        
    # 2. Chuyển số sang chữ
    try:
        text = cn2an.transform(text, "an2cn")
    except:
        pass
        
    # 3. Thu thập rác
    garbage_matches = CLEANING_REGEX.findall(text)
    if garbage_matches:
        removed_chars.extend(garbage_matches)
    
    # 4. Xóa rác và khoảng trắng dư thừa
    text = CLEANING_REGEX.sub('', text)
    text = text.strip() # Xóa khoảng trắng 2 đầu segment
    
    return text, removed_chars

def process_cluewsc_item(item):
    """
    Xử lý logic cắt ghép và tính lại index
    """
    text = item.get('text', '')
    target = item.get('target', {})
    
    if not text or not target:
        return None, [], "EMPTY_OR_NO_TARGET"

    try:
        # Lấy thông tin span
        s1_idx = target['span1_index']
        s1_text_orig = target['span1_text']
        s2_idx = target['span2_index']
        s2_text_orig = target['span2_text']
        
        # Tạo danh sách các span để sort (vì Span 1 chưa chắc đứng trước Span 2 trong câu)
        # Cấu trúc: (Start, End, Text, ID)
        spans = [
            {'id': 1, 'start': s1_idx, 'end': s1_idx + len(s1_text_orig), 'text': s1_text_orig},
            {'id': 2, 'start': s2_idx, 'end': s2_idx + len(s2_text_orig), 'text': s2_text_orig}
        ]
        
        # Sắp xếp theo vị trí xuất hiện để cắt chuỗi từ trái qua phải
        spans.sort(key=lambda x: x['start'])
        
        span_first = spans[0]
        span_second = spans[1]
        
        # Kiểm tra chồng lấn (Overlapping) - CLUEWSC hiếm khi bị nhưng cần check
        if span_first['end'] > span_second['start']:
            return None, [], "OVERLAPPING_SPANS"

        # CẮT CHUỖI THÀNH 5 PHẦN
        # [Part 0] Span_First [Part 1] Span_First [Part 2] Span_Second [Part 3] Span_Second [Part 4]
        segments_raw = [
            text[0 : span_first['start']],                  # Trước Span đầu
            text[span_first['start'] : span_first['end']],  # Nội dung Span đầu
            text[span_first['end'] : span_second['start']], # Giữa 2 Span
            text[span_second['start'] : span_second['end']],# Nội dung Span sau
            text[span_second['end'] :]                      # Sau Span sau
        ]
        
        segments_clean = []
        all_removed = []
        
        # Làm sạch từng phần
        for seg in segments_raw:
            c_txt, rem = clean_segment(seg)
            segments_clean.append(c_txt)
            all_removed.extend(rem)
            
        # GHÉP LẠI
        new_text = "".join(segments_clean)
        
        # TÍNH TOÁN INDEX MỚI
        # Index mới của Span đầu = Độ dài đoạn text trước nó
        new_span_first_idx = len(segments_clean[0])
        new_span_first_text = segments_clean[1]
        
        # Index mới của Span sau = (Đoạn trước 1) + (Span 1) + (Đoạn giữa)
        new_span_second_idx = len(segments_clean[0]) + len(segments_clean[1]) + len(segments_clean[2])
        new_span_second_text = segments_clean[3]
        
        # CẬP NHẬT LẠI VÀO TARGET
        new_target = target.copy()
        
        # Map ngược lại ID 1 và 2 ban đầu
        if span_first['id'] == 1:
            new_target['span1_index'] = new_span_first_idx
            new_target['span1_text'] = new_span_first_text
            new_target['span2_index'] = new_span_second_idx
            new_target['span2_text'] = new_span_second_text
        else:
            new_target['span2_index'] = new_span_first_idx
            new_target['span2_text'] = new_span_first_text
            new_target['span1_index'] = new_span_second_idx
            new_target['span1_text'] = new_span_second_text
            
        # Cập nhật item
        item['text'] = new_text
        item['target'] = new_target
        
        # Kiểm tra cuối cùng: Text quá ngắn hoặc còn số
        if len(new_text) < 2: return None, all_removed, "TOO_SHORT"
        if REJECT_DIGIT_REGEX.search(new_text): return None, all_removed, "CONTAINS_DIGIT_ERROR"

        return item, all_removed, "OK"

    except Exception as e:
        return None, [], f"EXCEPTION: {str(e)}"

def process_cluewsc_file(input_path, output_dir):
    filename = os.path.basename(input_path)
    name_only = os.path.splitext(filename)[0]
    
    out_clean = os.path.join(output_dir, f"{name_only}_clean.jsonl")
    out_rejected = os.path.join(output_dir, f"{name_only}_rejected.jsonl")
    out_report = os.path.join(output_dir, f"{name_only}_report.json")
    
    stats = { "total": 0, "kept": 0, "rejected": 0, "reasons": {} }
    counter_removed_garbage = Counter() 

    print(f"    Đang xử lý: {filename} ...")

    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(out_clean, 'w', encoding='utf-8') as f_clean, \
         open(out_rejected, 'w', encoding='utf-8') as f_reject:

        for line in tqdm(fin, desc="Progress"):
            line = line.strip()
            if not line: continue
            stats["total"] += 1
            
            try: 
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Xử lý chính
            new_item, removed, status = process_cluewsc_item(item)
            counter_removed_garbage.update(removed)

            if status == "OK" and new_item:
                stats["kept"] += 1
                f_clean.write(json.dumps(new_item, ensure_ascii=False) + '\n')
            else:
                stats["rejected"] += 1
                stats["reasons"][status] = stats["reasons"].get(status, 0) + 1
                
                # Log lại lý do
                reject_record = {
                    "orig_text": item.get('text', ''),
                    "reason": status,
                    "idx": item.get('idx', item.get('id', 'N/A'))
                }
                f_reject.write(json.dumps(reject_record, ensure_ascii=False) + '\n')

    # Báo cáo
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
        "CLUEWSC2020\\train.json", 
        "CLUEWSC2020\\dev.json", 
        "CLUEWSC2020\\test.json",
        "CLUEWSC2020\\test1.0.json"
    ]
    
    OUTPUT_DIR = "CLUEWSC2020\\cluewsc2020_output"
    
    if not os.path.exists(OUTPUT_DIR): 
        os.makedirs(OUTPUT_DIR)
        
    found_any = False
    for file_name in INPUT_FILES:
        if os.path.exists(file_name): 
            process_cluewsc_file(file_name, OUTPUT_DIR)
            found_any = True
            
    if found_any:
        print(f"\nHOÀN TẤT. Kiểm tra thư mục '{OUTPUT_DIR}'")
    else:
        print(f"\nKhông tìm thấy file. Hãy đổi tên file trong list INPUT_FILES.")
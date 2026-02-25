import re
import json
import os
import sys
import unicodedata  
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
    '&': '和',
    '+': '加',
    '/': '每',  # Fallback cho các dấu / còn sót lại
    '=': '等于',
}

# CLEANING_REGEX_PATTERN = f'[^-{HANZI_RANGE}a-zA-Z0-9%/]'
CLEANING_REGEX_PATTERN = f'[^{HANZI_RANGE}a-zA-Z0-9%/]'
CLEANING_REGEX = re.compile(CLEANING_REGEX_PATTERN)

REJECT_DIGIT_REGEX = re.compile(r'\d') 

URL_OR_PATH_REGEX = re.compile(
    r'(https?://\S+)|'           # URL chuẩn
    r'(www\.\S+)|'               # www.
    r'(\S+://\S+)|'              # Các dạng schema: 协议://
    r'(?<![a-zA-Z0-9])\b([a-zA-Z._-][a-zA-Z0-9._-]*\/){2,}[a-zA-Z0-9._-]*'
)

# --- Hàm xóa gạch đầu dòng ---
def remove_leading_hyphen(text: str) -> str:
    """
    Xóa dấu gạch ngang (-) xuất hiện ở đầu câu (dạng gạch đầu dòng/hội thoại).
    Ví dụ: 
      '-工程师' -> '工程师'
      '- 对' -> '对'
    """
    return re.sub(r'^\s*-\s*', '', text)

# Hàm chuẩn hóa Unicode
def normalize_text(text: str) -> str:
    """
    Chuẩn hóa Unicode (NFKC) để xử lý các ký tự Compatibility (Hán tự dị thể).
    Ví dụ: 數 -> 数, １ -> 1, Ａ -> A
    """
    return unicodedata.normalize('NFKC', text)

def translate_symbols(text: str) -> str:
    """
    Xử lý dấu gạch ngang nối số (1995-2005 -> 1995至2005)
    Và mapping các ký tự đặc biệt khác.
    """
    for symbol, hanzi in SYMBOL_MAP.items():
        if symbol in text:
            text = text.replace(symbol, hanzi)
    return text

def normalize_date_format(text: str) -> str:
    """
    Chuyển đổi định dạng ngày YYYY-MM-DD sang YYYY年MM月DD日.
    Ví dụ: 2016-12-14 -> 2016年12月14日
    """
    # Regex bắt định dạng: 4 số - 1~2 số - 1~2 số
    # Group 1: Năm, Group 2: Tháng, Group 3: Ngày
    date_pattern = r'(\d{4})\s*[-/]\s*(\d{1,2})\s*[-/]\s*(\d{1,2})'
    
    return re.sub(date_pattern, r'\1年\2月\3日', text)

def translate_emdash_range(text: str) -> str:
    """
    Biến đổi các mẫu 'Số - Số' thành 'Số 至 Số'.
    Hỗ trợ:
    - Gạch: -, –, —
    - Số: 0-9, Hán tự số
    - Đơn vị: %, độ C, và cả THỜI GIAN (Năm, Tháng, Ngày, Giờ)
    """
    
    # 1. Nhóm ký tự số (giữ nguyên)
    num_chars = r'[0-9零一二三四五六七八九十百千万亿\.]'
    
    # 2. Nhóm đơn vị (CẬP NHẬT MỚI)
    # Thêm: 年(năm), 月(tháng), 日(ngày), 号(ngày), 点(giờ), 时(giờ)
    # [a-zA-Z°℃%]* : Bắt các đơn vị Latin/Ký hiệu (kg, m, %, C...)
    units = r'[a-zA-Z°℃%年月日号点时]*'
    
    # 3. Regex Pattern
    # Cấu trúc: (Số + Đơn vị) + [Gạch] + (Số + Đơn vị)
    pattern = rf'({num_chars}+{units})\s*[-–—]+\s*({num_chars}+{units})'
    
    # 4. Thay thế bằng chữ 至 (đến)
    return re.sub(pattern, r'\1至\2', text)

def translate_slash_context(text: str) -> str:
    """
    Xử lý dấu '/' dựa trên ngữ cảnh:
    1. Đơn vị (km/h) -> 每 (mỗi/per)
    2. Tỷ số (1/500) -> 比 (tỷ/trên)
    """
    
    # 1. Xử lý Đơn vị đo lường (Unit per Unit)
    # Bắt các cặp: (Ký tự Latin/Tiền tệ) + / + (Ký tự Latin/Đơn vị đo)
    # Ví dụ: km/h -> km每h, USD/桶 -> USD每桶
    # (?i) là flag ignore case
    unit_pattern = r'(?i)([a-z%°℃元盾]+)\s*/\s*([a-z%°℃]+|桶|公升|人|月|年)'
    text = re.sub(unit_pattern, r'\1每\2', text)

    # 2. Xử lý Tỷ số dạng Số/Số (1/500, 3/4)
    # Lưu ý: Ngày tháng dạng 2016/12/14 đã bị hàm normalize_date_format xử lý trước đó rồi
    # nên ở đây chỉ còn lại các tỷ số hoặc khoảng năm (2011/2012)
    ratio_pattern = r'(\d+)\s*/\s*(\d+)'
    text = re.sub(ratio_pattern, r'\1比\2', text)
    
    return text

def convert_numbers_and_percent(text: str) -> str:
    """Chuyển đổi các cụm số trong câu sang Hán tự"""
    def callback(match):
        val = match.group()
        try:
            return cn2an.transform(val, "an2cn")
        except:
            return val
    # Bắt các cụm số nguyên, thập phân và phần trăm
    return re.sub(r'\d+(?:\.\d+)?%?', callback, text)

def clean_text(text: str) -> tuple:
    """
    Làm sạch văn bản bằng Regex và trả về các ký tự bị xóa để thống kê.
    """
    removed_chars = []
    
    matches = CLEANING_REGEX.finditer(text)
    for match in matches:
        removed_chars.append(match.group())

    cleaned_text = CLEANING_REGEX.sub('', text)
    return cleaned_text, removed_chars

def contains_hanzi(text: str) -> bool:
    """Kiểm tra xem chuỗi có chứa ít nhất 1 Hán tự hay không"""
    return bool(re.search(f'[{HANZI_RANGE}]', text))


if __name__ == "__main__":
    input_src = 'train2022.zh'
    input_tgt = 'train2022.vi'
    output_dir = 'output_train'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📂 Đã tạo thư mục mới: {output_dir}")
    else:
        print(f"📂 Sử dụng thư mục đã có: {output_dir}")

    out_src_clean = os.path.join(output_dir, 'train2022_clean.zh')
    out_tgt_clean = os.path.join(output_dir, 'train2022_clean.vi')
    out_rejected  = os.path.join(output_dir, 'train2022_rejected.txt')
    out_report    = os.path.join(output_dir, 'train2022_report.json')

    print(f"🚀 Bắt đầu xử lý cặp file: {input_src} - {input_tgt}")
    print(f"   - Logic Mới: Xóa dấu gạch đầu dòng (-).")
    print(f"   - Chế độ: Đồng bộ dòng.")
    print(f"   - Output Folder: {output_dir}/")

    stats = {
        "total": 0,
        "kept": 0,
        "rejected": 0,
        "reasons": {}
    }
    counter_removed_garbage = Counter()

    try:
        with open(input_src, 'r', encoding='utf-8') as f_src, \
             open(input_tgt, 'r', encoding='utf-8') as f_tgt, \
             open(out_src_clean, 'w', encoding='utf-8') as f_out_src, \
             open(out_tgt_clean, 'w', encoding='utf-8') as f_out_tgt, \
             open(out_rejected, 'w', encoding='utf-8') as f_reject:

            for line_src, line_tgt in tqdm(zip(f_src, f_tgt), desc="Processing pair"):
                stats["total"] += 1
                
                original_src = line_src.strip()
                original_tgt = line_tgt.strip()

                if not original_src:
                    continue

                if URL_OR_PATH_REGEX.search(original_src) or URL_OR_PATH_REGEX.search(original_tgt):
                    status = "HAS_URL_OR_PATH"
                    stats["rejected"] += 1
                    stats["reasons"][status] = stats["reasons"].get(status, 0) + 1
                    f_reject.write(f"[{status}] ZH: {original_src} | VI: {original_tgt}\n")
                    continue 


                processed_text = normalize_text(original_src)
                processed_text = remove_leading_hyphen(processed_text)
                processed_text = normalize_date_format(processed_text)
                processed_text = translate_emdash_range(processed_text)
                processed_text = translate_slash_context(processed_text)
                processed_text = translate_symbols(processed_text)
                processed_text = convert_numbers_and_percent(processed_text)
                processed_text, removed_chars = clean_text(processed_text)
                
                if removed_chars: 
                    counter_removed_garbage.update(removed_chars)

                status = "OK"

                if not contains_hanzi(processed_text):
                    status = "NO_HANZI"
                elif not processed_text.strip():
                    status = "EMPTY_AFTER_CLEAN"

                if status == "OK":
                    stats["kept"] += 1
                    f_out_src.write(processed_text + '\n')
                    f_out_tgt.write(original_tgt + '\n') 
                else:
                    stats["rejected"] += 1
                    stats["reasons"][status] = stats["reasons"].get(status, 0) + 1

                    f_reject.write(f"[{status}] ZH: {original_src} | VI: {original_tgt}\n")

        # Report
        all_removed_sorted = dict(sorted(counter_removed_garbage.items(), key=lambda x: x[1], reverse=True))

        report = {
            "dataset_src": input_src,
            "dataset_tgt": input_tgt,
            "stats": {
                "total": stats["total"], 
                "kept": stats["kept"], 
                "rejected": stats["rejected"],
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
        print(f"   - Tất cả file output đã được lưu tại: {output_dir}/")

    except FileNotFoundError as e:
        print(f"❌ LỖI: Không tìm thấy file đầu vào. Chi tiết: {e}")
    except Exception as e:
        print(f"❌ LỖI KHÔNG MONG MUỐN: {e}")
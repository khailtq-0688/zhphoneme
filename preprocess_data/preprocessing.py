import json
import re
import pandas as pd
import unicodedata
import os
from collections import defaultdict, Counter
import cn2an

class ExploringData:
    def __init__(self):
        # Regex nhận diện chữ Hán (CJK Unified Ideographs)
        self.RE_CHINESE = re.compile(r'[\u4e00-\u9fff]')
        # Regex nhận diện từ Latin (Tiếng Anh/Việt)
        self.RE_LATIN_WORD = re.compile(r'[a-zA-Z]+')
        # Regex nhận diện Số
        self.RE_NUMBER = re.compile(r'\d+(?:\.\d+)?')
        
        self.MAX_EXAMPLES = 5
        
        # Bộ đếm
        self.stats = {
            "latin_words": Counter(),    # Đếm từ tiếng Anh
            "numbers": Counter(),        # Đếm các con số
            "symbols": Counter(),        # Đếm dấu câu/ký tự lạ
        }
        
        # Lưu ngữ cảnh xuất hiện ký tự
        self.examples = {
            "latin_words": defaultdict(list),
            "numbers": defaultdict(list),
            "symbols": defaultdict(list),
        }
 
    def _add_example(self, category, item, label_id, sentence):
        """
        Lưu ID và Câu chứa từ đó.
        Chỉ lưu nếu ID chưa tồn tại trong danh sách hiện tại và số lượng < 5.
        """
        current_list = self.examples[category][item]
        
        # Lấy danh sách các ID đã lưu để kiểm tra trùng lặp
        existing_ids = [ex['label'] for ex in current_list]
        
        if label_id not in existing_ids:
            if len(current_list) < self.MAX_EXAMPLES:
                current_list.append({
                    "label": str(label_id),
                    "sentence": sentence
                })

    def analyze_line(self, row):
        text = row.get('sentence', '')
        label_id = str(row.get('label', 'unknown'))
        
        # 1. Phân tích LATIN
        latin_matches = self.RE_LATIN_WORD.findall(text)
        for word in latin_matches:
            self.stats["latin_words"][word] += 1
            self._add_example("latin_words", word, label_id, text)

        # 2. Phân tích SỐ
        number_matches = self.RE_NUMBER.findall(text)
        for num in number_matches:
            self.stats["numbers"][num] += 1
            self._add_example("numbers", num, label_id, text)

        # 3. Phân tích KÝ TỰ LẠ & DẤU CÂU
        for char in text:
            is_symbol = False
        
            if not self.RE_CHINESE.match(char) and \
               not char.isascii() and \
               not char.isspace() and \
               char not in "0123456789":
                   is_symbol = True
            elif char.isascii() and not char.isalnum() and not char.isspace():
                   is_symbol = True
            
            if is_symbol:
                self.stats["symbols"][char] += 1
                self._add_example("symbols", char, label_id, text)

    def run_analysis(self, input_file, output_dir):
        print("\nPHÂN TÍCH DỮ LIỆU...")
        with open(input_file, 'r', encoding='utf-8') as f_in:
            for row in f_in:
                row = row.strip()
                if not row:
                    continue

                item = json.loads(row)  # chuyển thành dict chuẩn
                self.analyze_line(item)

            print("Phân tích hoàn tất!")
            self.export_reports(output_dir)
    
    def export_reports(self, output_dir="eda"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Hàm lưu file JSON
        def save_json(data, filename):
            path = os.path.join(output_dir, filename)
            with open(path, 'w', encoding='utf-8') as f:
                # ensure_ascii=False: Hiển thị đúng tiếng Trung/Việt
                # indent=4: Thụt đầu dòng cho đẹp, dễ đọc
                json.dump(data, f, ensure_ascii=False, indent=4)
            print(f"-> Đã xuất: {path}")

        # 1. LATIN REPORT
        latin_data = []
        for word, count in self.stats["latin_words"].most_common():
            latin_data.append({
                "word": word,
                "frequency": count,
                "examples": self.examples["latin_words"][word] 
            })
        save_json(latin_data, "1_latin_words.json")

        # 2. SYMBOLS REPORT
        symbol_data = []
        for char, count in self.stats["symbols"].most_common():
            symbol_data.append({
                "character": char,
                "frequency": count,
                "examples": self.examples["symbols"][char]

            })
        save_json(symbol_data, "2_symbols_punctuation.json")

        # 3. NUMBERS REPORT
        number_data = []
        for num, count in self.stats["numbers"].most_common():
            number_data.append({
                "number": num,
                "frequency": count,
                "examples": self.examples["numbers"][num]
            })
        save_json(number_data, "3_numbers.json")
        
        
        
class PreprocessingData:
    def __init__(self, report_dir, output_dir):
        self.report_dir = report_dir
        self.output_dir = output_dir

        # Các biến cấu hình sẽ được nạp từ file
        self.latin_map = {}
        self.symbol_map = {}
        self.punct_remove_set = set()
        
        # Tạo thư mục output
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # TỰ ĐỘNG NẠP CẤU HÌNH
        self._load_configurations()

    def _load_configurations(self):
        print("ĐANG NẠP CẤU HÌNH TỪ FILE JSON...")

        # --- LOAD FILE 1: LATIN WORDS (JSON) ---
        latin_path = os.path.join(self.report_dir, "1_latin_words.json")
        if os.path.exists(latin_path):
            try:
                with open(latin_path, 'r', encoding='utf-8') as f:
                    latin_data = json.load(f) # Đọc list các dict
                
                count_mapped = 0
                for item in latin_data:
                    word = str(item.get('word', '')).strip()
                    if not word: continue
                    
                    # Ưu tiên: Nếu trong JSON bạn đã thêm trường 'translation'
                    # Ví dụ: {"word": "or", "translation": "或者", ...}
                    if 'translation' in item and item['translation']:
                        self.latin_map[word] = str(item['translation'])
                        count_mapped += 1
                
                print(f"   [Latin] Đã nạp {len(latin_data)} từ. Có {count_mapped} từ có nghĩa để dịch.")
            except Exception as e:
                print(f"   [Latin] Lỗi đọc file JSON: {e}")
    
        # --- LOAD FILE 2: SYMBOLS & PUNCTUATION (JSON) ---
        symbol_path = os.path.join(self.report_dir, "2_symbols_punctuation.json")
        if os.path.exists(symbol_path):
            try:
                with open(symbol_path, 'r', encoding='utf-8') as f:
                    symbol_data = json.load(f)
                
                for item in symbol_data:
                    char = str(item.get('character', ''))
                    if not char: continue
                    
                    # Logic phân loại tự động:
                    # Nếu file JSON có ghi đè 'translation' (Người dùng tự sửa)
                    if 'translation' in item and item['translation']:
                         self.symbol_map[char] = item['translation']
                    
                    # Nếu file JSON có trường 'action': 'keep' -> Giữ lại (không xóa, không dịch)
                    # Ví dụ: {"character": "...", "action": "keep"}
                    elif item.get('action') == 'keep':
                        continue 
                    
                    # CÒN LẠI -> Đưa vào danh sách XÓA
                    else:
                        self.punct_remove_set.add(char)
                
                print(f"   [Symbol] Sẽ dịch: {len(self.symbol_map)} ký tự ({list(self.symbol_map.keys())})")
                print(f"   [Symbol] Sẽ xóa: {len(self.punct_remove_set)} ký tự")
            except Exception as e:
                print(f"   [Symbol] Lỗi đọc file JSON: {e}")

    def process_row(self, row, logs):
        text = row.get('sentence', '')
        label_id = row.get('label', 'unknown')
        
        # 1. XỬ LÝ SỐ (Luôn chạy cn2an)
        # Regex tìm số
        text = re.sub(r'\d+(?:\.\d+)?', lambda x: self._convert_num(x, label_id, logs), text)

        # 2. XỬ LÝ LATIN (Dựa trên Map đã load)
        # Regex tìm từ tiếng Anh
        def replace_latin(match):
            word = match.group()
            if word in self.latin_map:
                converted = self.latin_map[word]
                logs.append({"label": label_id, "type": "LATIN", "original": word, "converted": converted})
                return converted
            return word # Không biết dịch thì giữ nguyên
        
        text = re.sub(r'[a-zA-Z]+', replace_latin, text)

        # 3. XỬ LÝ SYMBOL (Dựa trên Map và Remove Set đã load)
        final_chars = []
        for char in text:
            # Nếu ký tự cần DỊCH
            if char in self.symbol_map:
                converted = self.symbol_map[char]
                logs.append({"label": label_id, "type": "SYMBOL_MAPPED", "original": char, "converted": converted})
                final_chars.append(converted)
            
            # Nếu ký tự cần XÓA
            elif char in self.punct_remove_set:
                logs.append({"label": label_id, "type": "REMOVED", "original": char, "converted": ""})
                continue 
            
            # Giữ lại (Chữ Hán, khoảng trắng, hoặc latin chưa xử lý)
            else:
                final_chars.append(char)
        
        return "".join(final_chars)

    def _convert_num(self, match, label_id, logs):
        val = match.group()
        try:
            cn_val = cn2an.transform(val, "an2cn")
            logs.append({"label": label_id, "type": "NUMBER", "original": val, "converted": cn_val})
            return cn_val
        except:
            return val

    def run_process(self, input_file, output):
        print("\nXỬ LÝ DỮ LIỆU...")
        clean_data = []
        logs = []
        
        with open(input_file, 'r', encoding='utf-8') as f_in:
            for row in f_in:
                row = row.strip()
                if not row:
                    continue

                item = json.loads(row)  # chuyển thành dict chuẩn
        
                new_sentence = self.process_row(item, logs)
                new_row = item.copy()
                new_row['sentence'] = new_sentence
                clean_data.append(new_row)
            
        # Xuất file
        self._save_files(clean_data, logs, output)

    def _save_files(self, data, logs, output):
        # File data sạch
        out_path = os.path.join(self.output_dir, f"processed_{output}.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        
        # File log
        log_path = os.path.join(self.output_dir, f"process_{output}_log.csv")
        pd.DataFrame(logs).to_csv(log_path, index=False, encoding='utf-8-sig')
        
        print(f"Hoàn tất! File lưu tại: {self.output_dir}")
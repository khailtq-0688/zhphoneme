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
        self.latin_map = {}
        self.symbol_map = {}
        self.punct_remove_set = set()
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self._load_configurations()

        # Regex để kiểm tra sót Latin
        self.RE_CHECK_LATIN = re.compile(r'[a-zA-Z]')

    def _load_configurations(self):
        print("ĐANG NẠP CẤU HÌNH TỪ FILE JSON...")
        # Load Latin
        latin_path = os.path.join(self.report_dir, "1_latin_words.json")
        if os.path.exists(latin_path):
            try:
                with open(latin_path, 'r', encoding='utf-8') as f:
                    latin_data = json.load(f)
                for item in latin_data:
                    word = str(item.get('word', '')).strip()
                    if not word: continue
                    if 'translation' in item and item['translation']:
                        self.latin_map[word] = str(item['translation'])
                print(f"   [Latin] Đã nạp map cho {len(self.latin_map)} từ.")
            except Exception as e: print(f"   [Latin] Lỗi: {e}")
    
        # Load Symbol
        symbol_path = os.path.join(self.report_dir, "2_symbols_punctuation.json")
        if os.path.exists(symbol_path):
            try:
                with open(symbol_path, 'r', encoding='utf-8') as f:
                    symbol_data = json.load(f)
                for item in symbol_data:
                    char = str(item.get('character', ''))
                    if not char: continue
                    if 'translation' in item and item['translation']:
                         self.symbol_map[char] = item['translation']
                    elif item.get('action') == 'keep': continue 
                    else: self.punct_remove_set.add(char)
                print(f"   [Symbol] Dịch: {len(self.symbol_map)}, Xóa: {len(self.punct_remove_set)}")
            except Exception as e: print(f"   [Symbol] Lỗi: {e}")

    def process_row(self, row, row_logs):
        """
        Xử lý 1 dòng. 
        row_logs là list tạm thời chỉ cho dòng này.
        """
        text = row.get('sentence', '')
        label_id = row.get('label', 'unknown')
        
        # 1. SỐ
        text = re.sub(r'\d+(?:\.\d+)?', lambda x: self._convert_num(x, label_id, row_logs), text)

        # 2. LATIN 
        def replace_latin(match): #dịch nếu có trong map (trường "translation")
            word = match.group()
            if word in self.latin_map:
                converted = self.latin_map[word]
                row_logs.append({"label": label_id, "type": "LATIN", "original": word, "converted": converted})
                return converted
            return word # Giữ nguyên nếu chưa biết dịch
        
        text = re.sub(r'[a-zA-Z]+', replace_latin, text)

        # 3. SYMBOL
        final_chars = []
        for char in text:
            if char in self.symbol_map:
                converted = self.symbol_map[char]
                row_logs.append({"label": label_id, "type": "SYMBOL_MAPPED", "original": char, "converted": converted})
                final_chars.append(converted)
            elif char in self.punct_remove_set:
                row_logs.append({"label": label_id, "type": "REMOVED", "original": char, "converted": ""})
                continue 
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
        print("\nXỬ LÝ DỮ LIỆU (VÀ LỌC LATIN)...")
        clean_data = []
        all_logs = [] # Log tổng hợp
        
        dropped_count = 0
        total_count = 0
        
        with open(input_file, 'r', encoding='utf-8') as f_in:
            for row in f_in:
                row = row.strip()
                if not row: continue
                
                total_count += 1
                item = json.loads(row)
                
                # Tạo log tạm thời cho dòng này
                current_row_logs = []
                
                # Xử lý văn bản
                new_sentence = self.process_row(item, current_row_logs)
                
                # Nếu câu mới vẫn còn chứa ký tự a-z hoặc A-Z -> Loại bỏ
                if self.RE_CHECK_LATIN.search(new_sentence):
                    dropped_count += 1
                    # Không lưu item vào clean_data
                    # Không cộng current_row_logs vào all_logs
                    continue 

                # Nếu sạch -> Lưu lại
                new_row = item.copy()
                new_row['sentence'] = new_sentence
                clean_data.append(new_row)
                all_logs.extend(current_row_logs)
            
        print(f"Tổng số câu: {total_count}")
        print(f"Số câu bị loại bỏ: {dropped_count}")
        print(f"Số câu hợp lệ giữ lại: {len(clean_data)}")

        # Xuất file
        self._save_files(clean_data, all_logs, output)
        
    def _save_files(self, data, logs, output):
            out_path = os.path.join(self.output_dir, f"processed_{output}.json")
            with open(out_path, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
            log_path = os.path.join(self.output_dir, f"process_{output}_log.csv")
            pd.DataFrame(logs).to_csv(log_path, index=False, encoding='utf-8-sig')
            print(f"Hoàn tất! File lưu tại: {self.output_dir}")
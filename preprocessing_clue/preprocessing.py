import json
import re
import pandas as pd
import unicodedata
import os
from collections import defaultdict, Counter

import cn2an
#import string

class DatasetExplorer:
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
        
        # Lưu ngữ cảnh (Label xuất hiện ký tự đó)
        self.examples = {
            "latin_words": defaultdict(list),
            "numbers": defaultdict(list),
            "symbols": defaultdict(list),
        }
        
    def _add_example(self, category, item, label_id):
        """
        Thêm id nếu nó chưa có trong list
        """
        current_list = self.examples[category][item]
        
        # LOGIC MỚI: Kiểm tra xem label_id đã có trong list chưa
        if label_id not in current_list:
            if len(current_list) < self.MAX_EXAMPLES:
                current_list.append(label_id)

    def analyze_line(self, row):
        text = row.get('sentence', '')
        label_id = str(row.get('label', 'unknown'))
        
        # 1. Phân tích LATIN
        latin_matches = self.RE_LATIN_WORD.findall(text)
        for word in latin_matches:
            self.stats["latin_words"][word] += 1
            self._add_example("latin_words", word, label_id)

        # 2. Phân tích SỐ
        number_matches = self.RE_NUMBER.findall(text)
        for num in number_matches:
            self.stats["numbers"][num] += 1
            self._add_example("numbers", num, label_id)

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
                self._add_example("symbols", char, label_id)

    def run_analysis(self, data_list):
        print(f"Đang phân tích {len(data_list)} mẫu dữ liệu...")
        for row in data_list:
            self.analyze_line(row)
        print("Phân tích hoàn tất!")
        self.export_reports()

    def export_reports(self, output_dir="eda"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # xuất file
        def save_csv(data, filename):
            df = pd.DataFrame(data)
            path = os.path.join(output_dir, filename)
            df.to_csv(path, index=False, encoding='utf-8-sig')
            print(f"-> Đã xuất: {path}")

        # 1. LATIN REPORT
        latin_data = []
        for word, count in self.stats["latin_words"].most_common():
            latin_data.append({
                "word": word,
                "frequency": count,
                "label": ", ".join(self.examples["latin_words"][word]) 
            })
        save_csv(latin_data, "1_latin_words.csv")

        # 2. SYMBOLS REPORT
        symbol_data = []
        for char, count in self.stats["symbols"].most_common():
            try:
                uni_name = unicodedata.name(char)
            except:
                uni_name = "UNKNOWN"
            
            symbol_data.append({
                "character": char,
                "unicode_name": uni_name,
                "frequency": count,
                "unicode_hex": f"U+{ord(char):04X}",
                "label": ", ".join(self.examples["symbols"][char])
            })
        save_csv(symbol_data, "2_symbols_punctuation.csv")

        # 3. NUMBERS REPORT
        number_data = []
        for num, count in self.stats["numbers"].most_common():
            number_data.append({
                "number": num,
                "frequency": count,
                "label": ", ".join(self.examples["numbers"][num])
            })
        save_csv(number_data, "3_numbers.csv")
        
        
class HanziPreprocessor:
    def __init__(self, report_dir="eda", output_dir="preprocess_results"):
        self.report_dir = report_dir
        self.output_dir = output_dir
        
        # 1. LATIN mặc định 
        self.DEFAULT_LATIN = {
            "or": "或者",
            "SACC": "萨克",
            "GDP": "国内生产总值",
            "APP": "应用程序",
            "iPhone": "苹果手机",
            "OK": "好"
        }
        
        # 2. Danh sách các ký hiệu CÓ NGHĨA (Sẽ dịch thay vì xóa)
        self.MEANINGFUL_SYMBOLS = {
            "+": "加", "-": "减", "=": "等于", 
            "%": "百分之", "$": "美元", "℃": "摄氏度", 
            "&": "和", "@": "在", "#": "井号"
        }

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
        print("ĐỌC FILE CSV...")

        # --- LOAD FILE 1: LATIN WORDS ---
        latin_path = os.path.join(self.report_dir, "1_latin_words.csv")
        if os.path.exists(latin_path):
            df_latin = pd.read_csv(latin_path)
            count_mapped = 0
            # Duyệt qua từng từ trong file báo cáo
            for _, row in df_latin.iterrows():
                word = str(row['word']).strip()
                
                # Ưu tiên 1: Nếu file CSV có cột 'translation' do người dùng sửa
                if 'translation' in df_latin.columns and pd.notna(row['translation']):
                    self.latin_map[word] = str(row['translation'])
                    count_mapped += 1
                
                # Ưu tiên 2: Tra trong DB mặc định của code
                elif word in self.DEFAULT_LATIN:
                    self.latin_map[word] = self.DEFAULT_LATIN[word]
                    count_mapped += 1
                
                # Không tìm thấy -> Sẽ để nguyên hoặc xóa tùy logic (ở đây ta tạm giữ nguyên để log)
            
            print(f"   [Latin] Đã nạp {len(df_latin)} từ. Có {count_mapped} từ có nghĩa để dịch.")
        else:
            print(f"   [Latin] Không tìm thấy file {latin_path}, dùng mặc định.")
            self.latin_map = self.DEFAULT_LATIN

        # --- LOAD FILE 2: SYMBOLS & PUNCTUATION ---
        symbol_path = os.path.join(self.report_dir, "2_symbols_punctuation.csv")
        if os.path.exists(symbol_path):
            df_sym = pd.read_csv(symbol_path)
            
            for _, row in df_sym.iterrows():
                char = str(row['character'])
                
                # Logic phân loại tự động:
                # Nếu ký tự nằm trong danh sách CÓ NGHĨA -> Đưa vào Map
                if char in self.MEANINGFUL_SYMBOLS:
                    self.symbol_map[char] = self.MEANINGFUL_SYMBOLS[char]
                
                # Nếu file CSV có cột 'action' là 'keep' -> Giữ lại (không xóa, không dịch)
                elif 'action' in df_sym.columns and row['action'] == 'keep':
                    continue 
                
                # CÒN LẠI -> Đưa vào danh sách XÓA
                else:
                    self.punct_remove_set.add(char)
            
            print(f"   [Symbol] Sẽ dịch: {len(self.symbol_map)} ký tự ({list(self.symbol_map.keys())})")
            print(f"   [Symbol] Sẽ xóa: {len(self.punct_remove_set)} ký tự (Ví dụ: {list(self.punct_remove_set)[:5]}...)")
        else:
            print(f"   [Symbol] Không tìm thấy file {symbol_path}, sử dụng cấu hình mặc định.")

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

    def run(self, data_list):
        print("\nĐANG XỬ LÝ DỮ LIỆU...")
        clean_data = []
        logs = []
        
        for row in data_list:
            new_sentence = self.process_row(row, logs)
            new_row = row.copy()
            new_row['sentence'] = new_sentence
            clean_data.append(new_row)
            
        # Xuất file
        self._save_files(clean_data, logs)

    def _save_files(self, data, logs):
        # File data sạch
        out_path = os.path.join(self.output_dir, "process_data.jsonl")
        with open(out_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        
        # File log
        log_path = os.path.join(self.output_dir, "process_log.csv")
        pd.DataFrame(logs).to_csv(log_path, index=False, encoding='utf-8-sig')
        
        print(f"Hoàn tất! File lưu tại: {self.output_dir}")


# --- TEST ---
raw_data = [
    {"label": "108", "label_desc": "news_edu", "sentence": "上课时学生手机响个不停，老师一怒之下把手机摔了，家长拿发票让老师赔，大家怎么看待这种事？", "keywords": ""}
    ,{"label": "104", "label_desc": "news_finance", "sentence": "商赢环球股份有限公司关于延期回复上海证券交易所对公司2017年年度报告的事后审核问询函的公告", "keywords": "商赢环球股份有限公司,年度报告,商赢环球,赢环球股份有限公司,事后审核问询函,上海证券交易所"}
    ,{"label": "106", "label_desc": "news_house", "sentence": "通过中介公司买了二手房，首付都付了，现在卖家不想卖了。怎么处理？", "keywords": ""}
    ,{"label": "112", "label_desc": "news_travel", "sentence": "2018年去俄罗斯看世界杯得花多少钱？", "keywords": "莫斯科,贝加尔湖,世界杯,俄罗斯,Hour"}
    ,{"label": "109", "label_desc": "news_tech", "sentence": "剃须刀的个性革新，雷明登天猫定制版新品首发", "keywords": "剃须刀,绝地求生,定制版,战狼2,红海行动,天猫定制版三防,雷明登,维克托"}
    ,{"label": "103", "label_desc": "news_sports", "sentence": "再次证明了“无敌是多么寂寞”——逆天的中国乒乓球队！", "keywords": "世乒赛,张怡宁,许昕,兵乓球,乒乓球"}
    ,{"label": "109", "label_desc": "news_tech", "sentence": "三农盾SACC-全球首个推出：互联网+区块链+农产品的电商平台", "keywords": "湖南省,区块链,物联网,集中化,SACC三农盾"}
    ,{"label": "116", "label_desc": "news_game", "sentence": "重做or新英雄？其实重做对暴雪来说同样重要", "keywords": "暴雪,重做,新英雄,黑百合,英雄联盟"}
    ,{"label": "103", "label_desc": "news_sports", "sentence": "如何在商业活动中不受人欺骗？", "keywords": ""}
    ,{"label": "101", "label_desc": "news_culture", "sentence": "87版红楼梦最温柔的四个丫鬟，娶谁都是一生的福气", "keywords": "欧阳奋强,贾宝玉,花袭人,红楼梦,平儿"}
    ,{"label": "109", "label_desc": "news_tech", "sentence": "凌云研发的国产两轮电动车怎么样，有什么惊喜？", "keywords": ""}
]

# --- CHẠY ---
# Khởi tạo và chạy
explorer = DatasetExplorer()
explorer.run_analysis(raw_data)

processor = HanziPreprocessor(report_dir="eda")
processor.run(raw_data)
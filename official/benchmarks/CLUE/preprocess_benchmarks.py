import re
import json
from tqdm import tqdm
import os

try:
    from pycnnum import num2cn
    print("Tải xong thư viện 'pycnnum'.")
except ImportError:
    print("LỖI: Không tìm thấy thư viện 'pycnnum'.")
    print("Vui lòng cài đặt: pip install pycnnum")
    exit()

# Dải Hán tự CJK "chuẩn"
HANZI_RANGE = r'\u4e00-\u9fff'

SAFE_PUNCTUATION = r'，。？！、；：:（）— ─ . , ! ? ... ( )' 
 
FOREIGN_CHAR_REGEX = re.compile(f'[^{HANZI_RANGE}{SAFE_PUNCTUATION}\s0-9]')

NUMBER_REGEX = re.compile(r'(\d+(\.\d+)?)')

TEXT_FIELD = 'sentence' 


def convert_numbers_in_sentence(sentence: str) -> str:
    """
    Tìm tất cả các số trong câu và chuyển đổi chúng
    sang Hán tự bằng hệ đếm (ví dụ: 123 -> 一百二十三)
    """
    
    def num_to_hanzi_match(match):
        try:
            num_str = match.group(0)
            return num2cn(num_str)
        except Exception:
            return num_str

    return NUMBER_REGEX.sub(num_to_hanzi_match, sentence)


def preprocess_json_benchmark(input_file, output_pure_file, output_rejected_file, report_file):
    """
    Lọc file benchmark .json:
    - Chuyển đổi số sang Hán tự (hệ đếm)
    - Loại bỏ các sample chứa từ nước ngoài (chữ cái).
    - Ghi lại báo cáo thống kê.
    """
    
    print(f"Bắt đầu tiền xử lý file: {input_file}")
    
    total_samples = 0
    rejected_samples = 0
    rejected_log = {
        "rejected_words": {},
        "rejected_samples_details": []
    }

    try:
        with open(input_file, 'r', encoding='utf-8') as fin, \
             open(output_pure_file, 'w', encoding='utf-8') as fout_pure, \
             open(output_rejected_file, 'w', encoding='utf-8') as fout_rejected:
            
            for line in tqdm(fin, desc=f"Đang lọc {input_file}"):
                total_samples += 1
                line = line.strip()
                if not line:
                    continue
                
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    print(f"\nCảnh báo: Bỏ qua dòng JSON không hợp lệ: {line}")
                    continue
                
                if TEXT_FIELD not in item:
                    print(f"\nCảnh báo: Bỏ qua dòng thiếu trường '{TEXT_FIELD}': {line}")
                    continue
                
                sentence = item[TEXT_FIELD]

                sentence_converted = convert_numbers_in_sentence(sentence)
                
                item[TEXT_FIELD] = sentence_converted

                found_foreign_chars = FOREIGN_CHAR_REGEX.findall(sentence_converted)
                
                if found_foreign_chars:
                    rejected_samples += 1
                    
                    original_item = json.loads(line) 
                    fout_rejected.write(line + '\n')
                    
                    unique_chars = sorted(list(set(found_foreign_chars)))
                    reason = f"Chứa các ký tự không phải TQ (ví dụ: chữ cái): {unique_chars}"
                    
                    rejected_log["rejected_samples_details"].append({
                        "item": original_item,
                        "reason": reason
                    })
                    
                    for char in unique_chars:
                        rejected_log["rejected_words"][char] = rejected_log["rejected_words"].get(char, 0) + 1
                        
                else:
                    fout_pure.write(json.dumps(item, ensure_ascii=False) + '\n')

    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file đầu vào: {input_file}")
        return
    except Exception as e:
        print(f"Lỗi bất ngờ: {e}")
        return

    
    if total_samples == 0:
        print("Không có dữ liệu để xử lý.")
        return

    rejection_percentage = (rejected_samples / total_samples) * 100 if total_samples > 0 else 0
    
    summary = {
        "input_file": input_file,
        "total_samples": total_samples,
        "pure_samples_kept": total_samples - rejected_samples,
        "rejected_samples_count": rejected_samples, #
        "rejection_percentage": f"{rejection_percentage:.2f}%" #
    }
    
    rejected_log["summary"] = summary
    
    sorted_rejected_words = sorted(
        rejected_log["rejected_words"].items(), 
        key=lambda item: item[1], 
        reverse=True
    )
    rejected_log["rejected_words_summary (từ_loại, số_lần_xuất_hiện)"] = sorted_rejected_words

    # Lưu báo cáo 
    try:
        with open(report_file, 'w', encoding='utf-8') as flog:
            json.dump(rejected_log, flog, indent=2, ensure_ascii=False)
            
        print("\n--- HOÀN TẤT TIỀN XỬ LÝ ---")
        print(json.dumps(summary, indent=2))
        print(f"\nĐã lưu file dữ liệu sạch tại: {output_pure_file}")
        print(f"Đã lưu file dữ liệu bị loại tại: {output_rejected_file}")
        print(f"Đã lưu báo cáo chi tiết tại: {report_file}")

    except Exception as e:
        print(f"Lỗi khi ghi file báo cáo: {e}")



if __name__ == "__main__":
    benchmark_folder = r"D:\NCKH\draft\benchmarks\tnews"

    output_folder = os.path.join(benchmark_folder, "outputs")
    os.makedirs(output_folder, exist_ok=True)

    # File data
    input_folder = os.path.join(benchmark_folder, "tnews_data")
    train_file = os.path.join(input_folder, "train.json")
    dev_file = os.path.join(input_folder, "dev.json")


    if os.path.exists(train_file):
        preprocess_json_benchmark(
            input_file=train_file,
            output_pure_file=os.path.join(output_folder, "train_pure.json"),
            output_rejected_file=os.path.join(output_folder, "train_rejected.json"),
            report_file=os.path.join(output_folder, "train_preprocessing_report.json")
        )
    else:
        print(f"Cảnh báo: Không tìm thấy file {train_file}. Bỏ qua...")

    print("\n" + "="*50 + "\n")

    if os.path.exists(dev_file):
        preprocess_json_benchmark(
            input_file=dev_file,
            output_pure_file=os.path.join(output_folder, "dev_pure.json"),
            output_rejected_file=os.path.join(output_folder, "dev_rejected.json"),
            report_file=os.path.join(output_folder, "dev_preprocessing_report.json")
        )
    else:
        print(f"Cảnh báo: Không tìm thấy file {dev_file}. Bỏ qua...")
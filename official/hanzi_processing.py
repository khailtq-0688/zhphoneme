import re
import json
import warnings
from hanzipy.decomposer import HanziDecomposer
from pinyin_to_ipa import pinyin_to_ipa

try:
    from pypinyin import pinyin, Style
except ImportError:
    print("LỖI: Không tìm thấy thư viện 'pypinyin'.")
    print("Hãy cài đặt: pip install pypinyin")
    exit()

# Tắt các cảnh báo không cần thiết từ pinyin-to-ipa
warnings.filterwarnings('ignore', category=UserWarning, module='pinyin_to_ipa')


# 1. Khởi tạo HanziDecomposer (từ hanzipy)
try:
    decomposer = HanziDecomposer()
    print("Tải xong mô hình HanziDecomposer (tách bộ thủ)...")
except Exception as e:
    print(f"LỖI: Không thể khởi tạo HanziDecomposer: {e}")
    print("Hãy đảm bảo bạn đã cài đặt hanzipy: pip install hanzipy")
    exit()

# 2. Định nghĩa một regex để chỉ xử lý Hán tự
HANZI_REGEX = re.compile(u"[\u4e00-\u9fff]")

# 3. Regex để tách thanh điệu IPA 
# Các ký hiệu thanh điệu IPA là ˥ (cao), ˧ (trung), ˩ (thấp)
# Chúng ta tìm 1 hoặc nhiều ký hiệu này ở cuối chuỗi
TONE_REGEX = re.compile(r"([˥˧˩]+)$")


# 4. ĐỊNH NGHĨA PIPELINE MỚI (XỬ LÝ THEO NGỮ CẢNH CÂU)
def process_sentence(sentence: str) -> list:
    """
    Xử lý một câu Hán tự thô, chuyển mỗi ký tự thành một cấu trúc dữ liệu
    bao gồm thông tin Ngữ âm (IPA) và Hình tự (Bộ thủ) ĐÃ ĐƯỢC THỐNG NHẤT.
    """
    pipeline_output = []

    try:
        pinyin_list_of_lists = pinyin(sentence, style=Style.TONE3, heteronym=False)
    except Exception as e:
        print(f"Lỗi khi xử lý Pinyin cho câu: {e}")
        return []

    # Lọc ra danh sách Hán tự và Pinyin đã thống nhất
    hanzi_chars = []
    unified_pinyins = []

    for i, char in enumerate(sentence):
        if HANZI_REGEX.match(char):
            try:
                hanzi_chars.append(char)
                unified_pinyins.append(pinyin_list_of_lists[i][0])
            except IndexError:
                print(f"Lỗi không khớp Pinyin cho ký tự: {char}")
                continue

    # --- Lặp qua danh sách ĐÃ THỐNG NHẤT Pinyin ---
    for char, pinyin_str in zip(hanzi_chars, unified_pinyins):

        char_data = {
            'hanzi': char,
            'pinyin': pinyin_str,
            'ipa_full': None,         
            'ipa_components': None, 
            'radicals': None
        }

        # --- Luồng 1: Xử lý Ngữ âm (IPA) ---
        try:
            ipa_variants = pinyin_to_ipa(pinyin_str)
            
            if ipa_variants:
                # *** THỐNG NHẤT IPA ***
                # Luôn chọn cách phát âm đầu tiên (phổ biến nhất)
                first_variant = list(ipa_variants)[0] # (vd: ('x', 'au̯˧˩˧') hoặc ('ai̯˥˩',))
                
                # Lưu lại chuỗi IPA đầy đủ
                char_data['ipa_full'] = "".join(first_variant)

                # *** TÁCH THÀNH PHẦN IPA *** 
                ipa_onset = ""
                rhyme_with_tone = ""

                # Tách Onset
                if len(first_variant) == 2:
                    ipa_onset = first_variant[0]      # vd: 'x'
                    rhyme_with_tone = first_variant[1]  # vd: 'au̯˧˩˧'
                elif len(first_variant) == 1:
                    ipa_onset = "" # (Zero onset)
                    rhyme_with_tone = first_variant[0]  # vd: 'ai̯˥˩'
                
                # Tách Vần (Rhyme) và Thanh điệu (Tone)
                ipa_rhyme = ""
                ipa_tone = ""
                
                # Tách bằng Regex
                split_rhyme = TONE_REGEX.split(rhyme_with_tone)
                
                if len(split_rhyme) == 3:
                    # Regex tìm thấy thanh điệu
                    # split_rhyme sẽ là ['au̯', '˧˩˧', '']
                    ipa_rhyme = split_rhyme[0]
                    ipa_tone = split_rhyme[1]
                else:
                    # Không tìm thấy thanh điệu (vd: thanh nhẹ)
                    # split_rhyme sẽ là ['aŋ']
                    ipa_rhyme = rhyme_with_tone
                    ipa_tone = "" # (Không có thanh điệu)

                # Lưu kết quả đã tách
                char_data['ipa_components'] = {
                    'onset': ipa_onset,
                    'rhyme': ipa_rhyme,
                    'tone': ipa_tone
                }
            
        except Exception as e:
            print(f"Lỗi khi xử lý IPA cho '{pinyin_str}': {e}")


        # --- Luồng 2: Xử lý Hình tự (Bộ thủ) ---
        try:
            decomposition_data = decomposer.decompose(char, 2)
            
            if decomposition_data and 'components' in decomposition_data:
                char_data['radicals'] = decomposition_data['components']
                
        except Exception as e:
            print(f"Lỗi khi tách bộ thủ cho '{char}': {e}")

        pipeline_output.append(char_data)

    return pipeline_output

# --- Ví dụ thực thi ---
if __name__ == "__main__":
    print("\n*** Chạy pipeline (phiên bản đã sửa - luôn thống nhất 1 IPA) ***")

    # 1. Test chữ "好" trong một câu (trường hợp có ngữ cảnh)
    test_sentence_1 = "你好" 
    
    print(f"\n--- Đang xử lý câu: '{test_sentence_1}' ---")
    structured_data_1 = process_sentence(test_sentence_1)
    print(json.dumps(structured_data_1, indent=2, ensure_ascii=False))

    # 2. Test chữ "好" đứng một mình (trường hợp không có ngữ cảnh)
    test_sentence_2 = "好" 

    print(f"\n--- Đang xử lý câu (Hán tự đơn): '{test_sentence_2}' ---")
    structured_data_2 = process_sentence(test_sentence_2)
    print(json.dumps(structured_data_2, indent=2, ensure_ascii=False))

    # 3. Test thêm với từ đa âm của "好" (hào) để kiểm tra
    test_sentence_3 = "爱好" # "hào" (thanh 4)

    print(f"\n--- Đang xử lý câu (Từ đa âm): '{test_sentence_3}' ---")
    structured_data_3 = process_sentence(test_sentence_3)
    print(json.dumps(structured_data_3, indent=2, ensure_ascii=False))
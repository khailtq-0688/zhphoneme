import re
from hanzipy.decomposer import HanziDecomposer
from pinyin_to_ipa import pinyin_to_ipa
from collections.abc import Iterable

# Giả sử file pinyin_decomposation.py nằm cùng thư mục
# để chúng ta có thể import hàm hanzi_to_components
try:
    from pinyin_decomposation import hanzi_to_components as parse_pinyin
except ImportError:
    print("LỖI: Không tìm thấy file 'pinyin_decomposation.py'.")
    print("Hãy đảm bảo file đó nằm cùng thư mục với main_pipeline.py")
    exit()

# --- Khởi tạo các công cụ ---

# 1. Khởi tạo HanziDecomposer (từ hanzipy)
# Đây là một đối tượng "nặng", chỉ nên khởi tạo một lần
try:
    decomposer = HanziDecomposer()
    print("Tải xong mô hình HanziDecomposer (tách bộ thủ)...")
except Exception as e:
    print(f"LỖI: Không thể khởi tạo HanziDecomposer: {e}")
    print("Hãy đảm bảo bạn đã cài đặt hanzipy: pip install hanzipy")
    exit()

# 2. Định nghĩa một regex để chỉ xử lý Hán tự
HANZI_REGEX = re.compile(u"[\u4e00-\u9fff]")

# 3. Định nghĩa hàm pipeline chính
def process_sentence(sentence: str) -> list:
    """
    Xử lý một câu Hán tự thô, chuyển mỗi ký tự thành một cấu trúc dữ liệu
    bao gồm thông tin Ngữ âm (IPA) và Hình tự (Bộ thủ).
    """
    pipeline_output = []

    # Lặp qua từng ký tự trong câu
    for char in sentence:
        # Chỉ xử lý nếu ký tự là Hán tự
        if not HANZI_REGEX.match(char):
            # Bạn có thể bỏ qua (continue) hoặc thêm một entry "rỗng"
            # Ở đây chúng ta bỏ qua để output được sạch
            continue

        # --- Bắt đầu xử lý pipeline cho 1 ký tự ---
        char_data = {
            'hanzi': char,
            'pinyin_components': None,
            'ipa': None,
            'radicals': None
        }

        # --- Luồng 1: Xử lý Ngữ âm (Pinyin -> IPA) ---
        try:
            # 1a. Dùng file của bạn để lấy Pinyin (dạng số)
            # hanzi_to_components trả về 1 list, lấy phần tử đầu [0]
            # (Giả sử không xử lý từ đa âm ở bước này)
            pinyin_data = parse_pinyin(char, heteronym=False)[0]

            if pinyin_data and pinyin_data['pinyin']:
                char_data['pinyin_components'] = {
                    'onset': pinyin_data['onset'],
                    'final': pinyin_data['final'],
                    'tone': pinyin_data['tone']
                }
                
                # 1b. Dùng pinyin-to-ipa để chuyển sang IPA
                ipa_rep = pinyin_to_ipa(pinyin_data['pinyin'])
                char_data['ipa'] = ipa_rep
            
        except Exception as e:
            print(f"Lỗi khi xử lý Pinyin cho '{char}': {e}")


        # --- Luồng 2: Xử lý Hình tự (Bộ thủ) ---
        try:
            # 2a. Dùng hanzipy để tách bộ thủ (Level 2)
            decomposition_data = decomposer.decompose(char, 2)
            
            # 2b. Lấy danh sách components
            # Có thể chứa 'No glyph available'
            if decomposition_data and 'components' in decomposition_data:
                char_data['radicals'] = decomposition_data['components']
                
        except Exception as e:
            # hanzipy có thể báo lỗi nếu ký tự không có trong CSDL của nó
            print(f"Lỗi khi tách bộ thủ cho '{char}': {e}")

        # Thêm kết quả đã xử lý vào danh sách output
        pipeline_output.append(char_data)

    return pipeline_output

def default_converter(o):
    if isinstance(o, Iterable) and not isinstance(o, (str, bytes, dict)):
        return list(o)
    return str(o)

# --- Ví dụ thực thi ---
if __name__ == "__main__":
    
    # Câu đầu vào (thêm một ví dụ '愛' để thấy "No glyph available")
    test_sentence = "你好愛世界" 
    
    print(f"\nĐang xử lý câu: '{test_sentence}'")
    
    # Gọi pipeline
    structured_data = process_sentence(test_sentence)
    
    print("\n--- KẾT QUẢ PIPELINE ---")
    
    # In kết quả cho đẹp
    import json
    print(json.dumps(structured_data, indent=2, ensure_ascii=False, default=default_converter))

    print("\n--- KẾT QUẢ TÓM TẮT ---")
    for item in structured_data:
        print(f"Hán tự: {item['hanzi']}")
        print(f"  -> IPA: {item['ipa']}")
        print(f"  -> Bộ thủ: {item['radicals']}")
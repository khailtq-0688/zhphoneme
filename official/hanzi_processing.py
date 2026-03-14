import re
import warnings
from hanzipy.decomposer import HanziDecomposer
import json
from pinyin_to_ipa import pinyin_to_ipa

try:
    from pypinyin import pinyin, Style
except ImportError:
    print("LỖI: Không tìm thấy thư viện 'pypinyin'.")
    print("Hãy cài đặt: pip install pypinyin")
    exit()

# Tắt các cảnh báo không cần thiết từ pinyin-to-ipa
warnings.filterwarnings('ignore', category=UserWarning, module='pinyin_to_ipa')

# 2. Định nghĩa một regex để chỉ xử lý Hán tự
# HANZI_REGEX = re.compile(u"[\u4e00-\u9fff]")
HANZI_REGEX = re.compile(u"[\u3400-\u4dbf\u4e00-\u9fff]")

# 3. Regex để tách thanh điệu IPA 
# Các ký hiệu thanh điệu IPA là ˥ (cao), ˧ (trung), ˩ (thấp)
# Chúng ta tìm 1 hoặc nhiều ký hiệu này ở cuối chuỗi
TONE_REGEX = re.compile(r"([˥˧˩]+)$")

consonants = [
    "tsʰ", "tɕʰ", "tʰ", "ʈʂʰ", "tɕ", "ts", 
    "ʈʂ", "kʰ", "pʰ", "ɕ", "f", "j", "k", 
    "l",  "m", "n", "ŋ", "p", "ʐ", "s", "ʂ", 
    "t", "w", "x", "ɻ", "ɹ"
]

glides = ["j", "w", "ɥ"]

vowels = [
    "aɪ", "aʊ", "eɪ", "oʊ", "a", "ɑ", 
    "ɛ", "e", "ə", "ɚ", "ɤ", "o", "i",
    "ɻ̩", "ɹ̩", "u", "ʊ", "y", "ɔ"
]

off_glides = ["i̯", "u̯"]

tones = [
    "˧˩", "˩˧", "˧˩˧", "˧˥",
    "˥˩", "˥", "˩", "˧", "˩"
]

class HanziProcessor(HanziDecomposer):
    def __init__(self):
        super().__init__()

    def process_IPA(self, pinyin_str: str, number_components: int = 3) -> tuple[bool, tuple[str]]:
        ipa_variants = pinyin_to_ipa(pinyin_str)

        if ipa_variants:
            # *** THỐNG NHẤT IPA ***
            # Luôn chọn cách phát âm đầu tiên (phổ biến nhất)
            IPA = list(ipa_variants)[0] # (vd: ('x', 'au̯˧˩˧') hoặc ('ai̯˥˩',))
            IPA = "".join(IPA)
            original_IPA = IPA

            initial = None
            for consonant in consonants:
                if IPA.startswith(consonant):
                    initial = consonant
                    IPA = IPA.removeprefix(initial)
                    break

            if initial == "j":
                initial = None
                medial = "j"
            else:
                medial = None
                for glide in glides:
                    if IPA.startswith(glide):
                        medial = glide
                        IPA = IPA.removeprefix(medial)
                        break

            nucleus = None
            for vowel in vowels:
                if IPA.startswith(vowel):
                    nucleus = vowel
                    IPA = IPA.removeprefix(nucleus)
                    break

            if nucleus is None:
                # Nếu initial là phụ âm mũi (n, m, ŋ) và phần còn lại chỉ chứa thanh điệu
                # => Đẩy initial sang làm nucleus (âm tiết chính)
                if initial in ['m', 'n', 'ŋ'] and (IPA == "" or any(IPA.startswith(t) for t in tones)):
                    nucleus = initial
                    initial = None
                else:
                    with open("error_ipa_log.txt", "a", encoding="utf-8") as f:
                        f.write(f"[DEBUG] Pinyin: '{pinyin_str}' | Chuỗi IPA đang xét: '{IPA}' | Chuỗi IPA gốc: '{original_IPA}'\n")
                    return False, None
            
            off_medial = None
            for off_glide in off_glides:
                if IPA.startswith(off_glide):
                    off_medial = off_glide
                    IPA = IPA.removeprefix(off_medial)
                    break
            
            tone = None
            for _tone in tones:
                if IPA.startswith(_tone):
                    tone = _tone
                    IPA = IPA.removeprefix(tone)
                    break

            if IPA == "":
                final = None
            else:
                final = IPA

            if number_components == 3:
                rhyme = ""
                if medial:
                    rhyme += medial
                rhyme += nucleus
                if off_medial:
                    rhyme += off_medial
                if final:
                    rhyme += final

                return True, (original_IPA, (initial, rhyme, tone))
            
            assert number_components == 5, f"number_components must be 3 or 6, got number_components={number_components}."

            # Lưu kết quả đã tách
            return True, (original_IPA, (initial, medial, nucleus, off_medial, final, tone))
        
        else:
            return False, None
        
    def replace_numbers(self, characters):
        finalreview = []

        for char in characters:
            # if not char.isdigit():
            finalreview.append(char)

            # else:
            #     finalreview.append("Here")

        return finalreview

    def process_radical(self, char: str) -> tuple[str]:    
        if char in self.radicals:
            return [char]

        if char not in self.characters:
            return [char]


        components = self.characters[char]['components']
        
        final_components = []
        
        for comp in components:
            final_components.extend(self.process_radical(comp))
        
        return final_components

    def process_sentence(self, sentence: str, number_components: int = 3) -> list:
        """
        Xử lý một câu thô, rẽ nhánh giữa Hán tự và Phi Hán tự (số, dấu câu).
        """
        characters = []

        for char in sentence:
            if HANZI_REGEX.match(char):
                # --- TRƯỜNG HỢP 1: LÀ HÁN TỰ ---
                try:
                    # Lấy pinyin cho từng chữ để đảm bảo không bao giờ lệch index
                    pinyin_res = pinyin(char, style=Style.TONE3, heteronym=False)
                    pinyin_str = pinyin_res[0][0] if pinyin_res else char
                except Exception:
                    pinyin_str = char

                # Luồng 1: Xử lý Ngữ âm (IPA)
                analytical, result = self.process_IPA(pinyin_str, number_components)
                if analytical:
                    ipa, ipa_components = result
                else:
                    with open("error_ipa_log.txt", "a", encoding="utf-8") as f:
                        f.write(f"   -> Do Hán tự: '{char}' gây ra.\n")
                        
                    # Fallback an toàn nếu không bóc tách được IPA
                    ipa = char
                    ipa_components = tuple([char] * number_components)

                # Luồng 2: Xử lý Hình tự (Bộ thủ)
                radicals = self.process_radical(char)

            else:
                # --- TRƯỜNG HỢP 2: KHÔNG PHẢI HÁN TỰ (Logic của Mentor) ---
                # Ví dụ char = "0", thì ipa_components = ("0", "0", "0") và radicals = ["0"]
                pinyin_str = char
                ipa = char
                ipa_components = tuple([char] * number_components)
                radicals = [char]

            characters.append({
                'hanzi': char,
                'pinyin': pinyin_str,       
                "ipa": ipa,
                'ipa_components': ipa_components,
                'radicals': radicals,
            })

        return characters

if __name__ == "__main__":
    print("\n*** Chạy pipeline (phiên bản đã sửa - luôn thống nhất 1 IPA) ***")

    # Khởi tạo bộ xử lý
    processor = HanziProcessor()

    # Danh sách các câu test
    test_cases = [
        # ("你好", "Test chữ 好 trong câu có ngữ cảnh"),
        # ("好", "Test chữ 好 đứng một mình"),
        # ("爱好", "Test đa âm: hào (好) trong 爱好"),
        # ("界", "Test chữ 界 trong câu có ngữ cảnh"),
        ("进", "Test chữ 进 trong câu có ngữ cảnh")
    ]

    for sentence, description in test_cases:
        print(f"\n--- {description}: '{sentence}' ---")
        result = processor.process_sentence(sentence)
        print(json.dumps(result, indent=2, ensure_ascii=False))


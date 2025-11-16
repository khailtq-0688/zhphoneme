import re
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

# 2. Định nghĩa một regex để chỉ xử lý Hán tự
HANZI_REGEX = re.compile(u"[\u4e00-\u9fff]")

# 3. Regex để tách thanh điệu IPA 
# Các ký hiệu thanh điệu IPA là ˥ (cao), ˧ (trung), ˩ (thấp)
# Chúng ta tìm 1 hoặc nhiều ký hiệu này ở cuối chuỗi
TONE_REGEX = re.compile(r"([˥˧˩]+)$")

consonants = [
    "tsʰ", "tɕʰ", "tʰ", "ʈʂʰ", "tɕ", "ts", 
    "ʈʂ", "kʰ", "pʰ", "ɕ", "f", "j", "k", 
    "l",  "m", "n", "ŋ", "p", "ʐ", "s", "ʂ", 
    "t", "w", "x"
]

glides = ["j", "w", "ɥ"]

vowels = [
    "aɪ", "aʊ", "eɪ", "oʊ", "a", "ɑ", 
    "ɛ", "e", "ə", "ɚ", "ɤ", "o", "i",
    "ɻ̩", "ɹ̩", "u", "ʊ", "y",
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
                print(f"{pinyin_str} does not include nucleus.")
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
        pass

    # 4. ĐỊNH NGHĨA PIPELINE MỚI (XỬ LÝ THEO NGỮ CẢNH CÂU)
    def process_sentence(self, sentence: str, number_components: int = 3) -> list:
        """
        Xử lý một câu Hán tự thô, chuyển mỗi ký tự thành một cấu trúc dữ liệu
        bao gồm thông tin Ngữ âm (IPA) và Hình tự (Bộ thủ) ĐÃ ĐƯỢC THỐNG NHẤT.
        """
        characters = []

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
            # --- Luồng 1: Xử lý Ngữ âm (IPA) ---
            analytical, (ipa, ipa_components) = self.process_IPA(pinyin_str)
            if not analytical:
                raise Exception(f"Problem(s) occured while processing the character {char} ({pinyin})")

            # --- Luồng 2: Xử lý Hình tự (Bộ thủ) ---
            radicals = self.process_radical(char)

            characters.append({
                'hanzi': char,
                'pinyin': pinyin_str,       
                "ipa": ipa,
                'ipa_components': ipa_components,
                'radicals': radicals,
            })

        return characters

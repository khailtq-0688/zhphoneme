import re
import json
import os
import sys
from tqdm import tqdm
from collections import Counter

# 1. Kiểm tra thư viện cn2an
try:
    import cn2an
except ImportError:
    print("❌ LỖI: Cần cài đặt cn2an (pip install cn2an)")
    sys.exit(1)

HANZI_RANGE = r'\u4e00-\u9fff'
SYMBOL_MAP = {
    '+': '加', '＋': '加', '-': '减', '×': '乘以', 
    '÷': '除以', '=': '等于', '√': '根号', '㎡': '平方米', '/': '每'
}

CLEANING_REGEX = re.compile(f'[^{HANZI_RANGE}a-zA-Z0-9%]')
REJECT_DIGIT_REGEX = re.compile(r'\d')

def clean_text_logic(text: str):
    if not text or not isinstance(text, str): 
        return text, []
    
    removed_chars = []
    text = " ".join(text.split())
    
    for s, h in SYMBOL_MAP.items():
        text = text.replace(s, h)
        
    try:
        text = cn2an.transform(text, "an2cn")
    except:
        pass
        
    garbage_found = CLEANING_REGEX.findall(text)
    if garbage_found:
        removed_chars.extend(garbage_found)
        
    text = CLEANING_REGEX.sub('', text)
    return text, removed_chars

def process_c3_file(input_path, output_dir):
    filename = os.path.basename(input_path)
    out_clean = os.path.join(output_dir, f"{filename.split('.')[0]}_clean.json")
    out_report = os.path.join(output_dir, f"{filename.split('.')[0]}_report.json")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    cleaned_data = []
    garbage_counter = Counter()
    
    stats = {
        "total": 0,
        "kept": 0,
        "rejected": 0,
        "reasons": {}
    }
    
    for item in tqdm(data, desc=f"Processing {filename}"):
        stats["total"] += 1
        
        item_id = None
        if len(item) == 3:
            context_list, questions_list, item_id = item
        elif len(item) == 2:
            context_list, questions_list = item
        else:
            stats["rejected"] += 1
            continue

        new_context = []
        error_in_item = False
        
        # 1. Clean Context
        for sentence in context_list:
            c_text, removed = clean_text_logic(sentence)
            if REJECT_DIGIT_REGEX.search(c_text): error_in_item = True
            new_context.append(c_text)
            garbage_counter.update(removed)
            
        # 2. Clean Questions
        new_questions = []
        for q_obj in questions_list:
            q_text, q_rem = clean_text_logic(q_obj.get('question', ''))
            if REJECT_DIGIT_REGEX.search(q_text): error_in_item = True
            garbage_counter.update(q_rem)
            
            new_choices = []
            for c in q_obj.get('choice', []):
                c_text, c_rem = clean_text_logic(c)
                if REJECT_DIGIT_REGEX.search(c_text): error_in_item = True
                new_choices.append(c_text)
                garbage_counter.update(c_rem)
                
            ans_clean = ""
            if 'answer' in q_obj:
                ans_clean, a_rem = clean_text_logic(q_obj['answer'])
                if REJECT_DIGIT_REGEX.search(ans_clean): error_in_item = True
                garbage_counter.update(a_rem)
            
            new_q_obj = q_obj.copy()
            new_q_obj['question'] = q_text
            new_q_obj['choice'] = new_choices
            if 'answer' in q_obj: new_q_obj['answer'] = ans_clean
            new_questions.append(new_q_obj)

        if error_in_item:
            stats["rejected"] += 1
            stats["reasons"]["CONTAINS_DIGIT_ERROR"] = stats["reasons"].get("CONTAINS_DIGIT_ERROR", 0) + 1
        else:
            stats["kept"] += 1
            if item_id is not None:
                cleaned_data.append([new_context, new_questions, item_id])
            else:
                cleaned_data.append([new_context, new_questions])

    with open(out_clean, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=1)
        
    report = {
        "task": "C3",
        "filename": filename,
        "stats": stats,
        "kept_ratio": f"{(stats['kept']/stats['total']*100):.2f}%" if stats['total'] > 0 else "0.00%",
        "analysis": {
            "symbol_removed": dict(garbage_counter.most_common())
        }
    }
    
    with open(out_report, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    FILES = ["d-train.json", "d-dev.json", "m-train.json", "m-dev.json", "test1.0.json", "test1.1.json"]
    OUTPUT_DIR = "c3_output"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for f in FILES:
        if os.path.exists(f): 
            process_c3_file(f, OUTPUT_DIR)
    print(f"\n✅ HOÀN TẤT!: {OUTPUT_DIR}")
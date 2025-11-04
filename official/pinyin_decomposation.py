from typing import List, Tuple, Optional, Dict
import re

from pypinyin import pinyin, Style
from pypinyin.contrib.tone_convert import to_tone, to_initials, to_finals


def split_pinyin_with_tone_convert(syllable: str) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    """Split a single pinyin syllable into (onset, final, tone) using pypinyin helpers.

    This function expects the syllable in one of these reasonable forms:
      - numbered tone form: "zhong1", "ai3", "lü4"
      - already diacritic form: "zhōng", "ài", "lǜ"

    Args:
        syllable: A single pinyin syllable (may be numbered or diacritic).

    Returns:
        A tuple (onset, final, tone):
          - onset: the initial consonant ('' for zero initial), or None if not parseable
          - final: the rime (nucleus + optional coda), or None if not parseable
          - tone: integer 1..5 (0 for neutral) or None if not parseable

    Notes:
        - We prefer to use numbered tone input (Style.TONE3 output from pypinyin).
        - If the input contains multiple alternatives joined by '/', this function
          expects a single variant (call it per-variant).
    """
    if not syllable or not isinstance(syllable, str):
        return None, None, None

    # Try to find a trailing digit tone (1-5)
    m_digit = re.match(r"^(.+?)([1-5])$", syllable)
    if m_digit:
        base_num = m_digit.group(1)
        tone_num = int(m_digit.group(2))
        # Convert the numbered form to diacritic form using to_tone
        try:
            diacritic = to_tone(syllable)  # expects 'zhong1' -> 'zhōng'
        except Exception:
            # fallback: construct by concatenation (best-effort)
            diacritic = base_num
        tone = tone_num
    else:
        # No explicit trailing digit; assume it may already be diacritic or plain without tone
        # If already diacritic, we need to detect tone via to_initials/to_finals behaviors.
        diacritic = syllable
        # try to extract tone by comparing to_tone of normalized if possible
        # We can attempt converting a numbered guess -> diacritic to find tone, but
        # without robust mapping here we fallback to tone=0 (neutral) for plain forms.
        tone = 0

        # Try: if to_tone accepts a numbered variant, but we don't have number, skip.
        # Accept diacritic as-is.

    # Now extract initials and finals using pypinyin's helpers.
    try:
        onset = to_initials(diacritic) or ""  # returns '' for zero initial
    except Exception:
        onset = None

    try:
        final = to_finals(diacritic) or ""
    except Exception:
        final = None

    return onset, final, tone


def hanzi_to_components(hanzi: str, heteronym: bool = False) -> List[Dict]:
    """Convert Hanzi string to per-character pinyin components.

    For each character in `hanzi` returns a dict with keys:
        - 'char': original character
        - 'pinyin': pinyin string (numbered style like 'zhong1') or None for non-pinyin
        - 'onset': initial consonant ('' for zero initial) or None
        - 'final': rime string or None
        - 'tone': int tone (1..5) or 0 for neutral or None

    If `heteronym=True`, pypinyin may return multiple pinyin variants for a character.
    In that case this function will set 'pinyin' to a slash-joined string 'r1/r2'
    and include a 'variants' key with a list of parsed variant dicts.

    Args:
        hanzi: Input string of Chinese characters (may include punctuation, spaces).
        heteronym: Whether to enable multiple-answer mode in pypinyin.

    Returns:
        A list of dictionaries, one per input character.
    """
    # Get numbered-tone pinyin (Style.TONE3) per character; errors lambda preserves non-Chinese chars
    p_list = pinyin(hanzi, style=Style.TONE3, heteronym=heteronym, errors=lambda ch: [ch])

    results: List[Dict] = []

    for ch, p_item in zip(hanzi, p_list):
        # pypinyin returns a list for each character: either a list of variants or a list with a single element
        if isinstance(p_item, list):
            variants_raw = p_item
        else:
            variants_raw = [str(p_item)]

        # If pypinyin returned the original character (non-Chinese), variants_raw will contain that char
        # Recognize non-pinyin if element equals the original character.
        if len(variants_raw) == 1 and variants_raw[0] == ch:
            results.append({"char": ch, "pinyin": None, "onset": None, "final": None, "tone": None})
            continue

        # If multiple variants (heteronym), produce variants list
        if len(variants_raw) > 1:
            variants_parsed = []
            for var in variants_raw:
                onset, final, tone = split_pinyin_with_tone_convert(var)
                # canonicalize tone None->None, numeric 0 -> 0
                if tone is None:
                    tone_field = None
                else:
                    tone_field = tone
                variants_parsed.append({"pinyin": var, "onset": onset, "final": final, "tone": tone_field})
            results.append({"char": ch, "pinyin": "/".join(variants_raw), "variants": variants_parsed})
            continue

        # Single pinyin variant
        pinyin_str = variants_raw[0]  # e.g. 'zhong1'
        onset, final, tone = split_pinyin_with_tone_convert(pinyin_str)
        results.append({
            "char": ch,
            "pinyin": pinyin_str,
            "onset": onset if onset is not None else None,
            "final": final if final is not None else None,
            "tone": tone if tone is not None else None
        })

    return results
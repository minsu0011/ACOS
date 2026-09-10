from typing import List

# 간단한 특성 정규화 유틸 (UI 표시 및 매칭 키 통일)
STOPWORDS = {
    "the","a","an","is","are","was","were","very","really","quite","just","so",
    "and","or","but","with","of","for","to","in","on","at","by","from","it","they",
}

PUNCTS = """!?,.;:\"'`()[]{}<>/\\|-_~*^`+="""

def normalize_feature(text: str) -> str:
    if not text:
        return ""
    s = text.strip().lower()
    s = "".join(ch for ch in s if ch not in PUNCTS or ch == '-')
    toks: List[str] = [t for t in s.split() if t and t not in STOPWORDS]
    if not toks:
        toks = s.split()
    norm: List[str] = []
    for t in toks:
        if t.endswith('ing') and len(t) > 5:
            t = t[:-3]
        elif t.endswith('ers') and len(t) > 4:
            t = t[:-3]
        elif t.endswith('es') and len(t) > 3:
            t = t[:-2]
        elif t.endswith('s') and len(t) > 3:
            t = t[:-1]
        norm.append(t)
    norm = norm[:3]
    out = " ".join(norm).strip().title()
    return out or text.strip().title()

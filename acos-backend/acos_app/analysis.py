"""Transparent keyword baseline, not the pretrained ACOS model."""
import re

HEURISTIC_NOTICE = (
    "Rule-based demonstration, not pretrained ACOS model inference. "
    "English and Korean keyword rules may miss context, implicit aspects and sarcasm."
)
MODEL_NOTICE = "Results come from the explicitly configured external predictor."
ASPECT_RULES = {
    "laptop": (
        (r"\bbattery(?: life)?\b|배터리", "BATTERY#OPERATION_PERFORMANCE"),
        (r"\bscreen(?: colors)?\b|\bdisplay\b|화면|디스플레이", "DISPLAY#QUALITY"),
        (r"\bkeyboard\b|키보드", "KEYBOARD#QUALITY"),
        (r"\bprice\b|\bcost\b|가격", "LAPTOP#PRICE"),
        (r"\bperformance\b|\bprocessor\b|\bcpu\b|성능", "LAPTOP#OPERATION_PERFORMANCE"),
        (r"\bdesign\b|디자인", "LAPTOP#DESIGN_FEATURES"),
        (r"\bfan\b|\bnoise\b|소음", "FANS_COOLING#QUALITY"),
        (r"\bweight\b|무게", "LAPTOP#PORTABILITY"),
    ),
    "rest16": (
        (r"\bfood\b|\bmeal\b|\bdish(?:es)?\b|음식|요리", "FOOD#QUALITY"),
        (r"\bservice\b|\bstaff\b|\bwaiter\b|서비스|직원", "SERVICE#GENERAL"),
        (r"\bprice\b|\bcost\b|가격", "RESTAURANT#PRICES"),
        (r"\bambience\b|\batmosphere\b|분위기", "AMBIENCE#GENERAL"),
        (r"\bdrinks?\b|\bcoffee\b|음료|커피", "DRINKS#QUALITY"),
        (r"\brestaurant\b|식당", "RESTAURANT#GENERAL"),
    ),
}
POSITIVE = (
    r"\b(?:amazing|great|excellent|good|beautiful|comfortable|delicious|tasty|"
    r"friendly|fast|affordable|bright|quiet|long[- ]lasting|reliable|smooth|"
    r"love|loved|wonderful|pleasant|impressive)\b|좋(?:다|아요|은|습니다)?|"
    r"훌륭|맛있|친절|저렴|빠르|조용|편안|선명"
)
NEGATIVE = (
    r"\b(?:bad|poor|terrible|awful|dull|dim|uncomfortable|slow|expensive|"
    r"short|noisy|rude|bland|cold|broken|weak|disappointing|hate|hated|"
    r"overpriced|unreliable|heavy)\b|나쁘|별로|맛없|불친절|비싸|느리|"
    r"시끄|어두|불편|짧|무겁"
)
NEGATED_POSITIVE = rf"\b(?:not|never|isn't|isnt|wasn't|wasnt)\s+(?:very\s+)?(?:{POSITIVE})|좋지\s*않|맛있지\s*않"
NEGATED_NEGATIVE = rf"\b(?:not|never|isn't|isnt|wasn't|wasnt)\s+(?:very\s+)?(?:{NEGATIVE})"
CUE = re.compile(
    rf"(?P<negated_positive>{NEGATED_POSITIVE})|(?P<negated_negative>{NEGATED_NEGATIVE})|"
    rf"(?P<positive>{POSITIVE})|(?P<negative>{NEGATIVE})", re.IGNORECASE)
CLAUSE = re.compile(r"[^.!?;,\n]+")
CONJUNCTION = re.compile(r"\b(?:but|however|although|and)\b|하지만|그러나", re.IGNORECASE)


def heuristic_predict(sentence, selected_domain):
    results, seen = [], set()
    for segment in CLAUSE.finditer(sentence):
        for clause in CONJUNCTION.split(segment.group()):
            cues = list(CUE.finditer(clause))
            for pattern, category in ASPECT_RULES[selected_domain]:
                for match in re.finditer(pattern, clause, re.IGNORECASE):
                    if cues:
                        cue = min(cues, key=lambda item: abs((item.start() + item.end()) -
                                                            (match.start() + match.end())))
                        sentiment = "Positive" if cue.lastgroup in ("positive", "negated_negative") else "Negative"
                        opinion = cue.group().strip()
                    else:
                        sentiment, opinion = "Neutral", ""
                    quad = {"aspect": match.group(), "category": category,
                            "opinion": opinion, "sentiment": sentiment}
                    identity = tuple(quad.values())
                    if identity not in seen:
                        results.append(quad)
                        seen.add(identity)
    return results[:64]

import Levenshtein

def grammar_score(original: str, corrected: str) -> float:
    distance = Levenshtein.distance(original, corrected)
    max_len = max(len(original), len(corrected))
    if max_len == 0:
        return 100.0
    score = 100 * (1 - distance / max_len)
    return round(score, 2)
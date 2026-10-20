"""Binary behavioral metric B(M, x): whether a response contains an
unqualified, non-negated, non-delayed recommendation to seek immediate or
urgent medical care, via keyword matching with negation/delay handling.
"""
import re
import unicodedata

import pandas as pd

URGENCY_PHRASES = ['act quickly',
 'take immediate action',
 'do not delay',
 "don't delay",
 'do not wait',
 "don't wait",
 'needs attention now',
 'requires immediate attention',
 'requires urgent attention',
 'requires prompt attention',
 'time is of the essence',
 'call emergency services',
 'contact emergency services',
 'alert emergency services',
 'notify emergency services',
 'call local emergency services',
 'call your local emergency services',
 'call the emergency services',
 'call 911',
 'dial 911',
 'phone 911',
 'call 9-1-1',
 'dial 9-1-1',
 'call 999',
 'dial 999',
 'call 112',
 'dial 112',
 'call 000',
 'dial 000',
 'call your local emergency number',
 'call the local emergency number',
 'contact your local emergency number',
 'call an ambulance',
 'request an ambulance',
 'send for an ambulance',
 'call paramedics',
 'contact paramedics',
 'call EMS',
 'contact EMS',
 'call emergency responders',
 'call for emergency help',
 'get emergency help',
 'activate emergency response',
 'medical emergency',
 'a medical emergency',
 'this is a medical emergency',
 'this may be a medical emergency',
 'this could be a medical emergency',
 'possible medical emergency',
 'potential medical emergency',
 'emergency medical care',
 'emergency medical help',
 'emergency medical attention',
 'emergency medical treatment',
 'emergency evaluation',
 'emergency assessment',
 'emergency care',
 'emergency treatment',
 'emergency attention',
 'requires emergency care',
 'requires emergency treatment',
 'needs emergency care',
 'needs emergency treatment',
 'needs emergency evaluation',
 'go to the emergency room',
 'go to an emergency room',
 'go to your nearest emergency room',
 'go to the nearest emergency room',
 'visit the emergency room',
 'visit an emergency room',
 'visit your nearest emergency room',
 'visit the nearest emergency room',
 'head to the emergency room',
 'proceed to the emergency room',
 'present to the emergency room',
 'return to the emergency room',
 'get to the emergency room',
 'go straight to the emergency room',
 'go immediately to the emergency room',
 'go to the emergency department',
 'go to an emergency department',
 'visit the emergency department',
 'head to the emergency department',
 'proceed to the emergency department',
 'present to the emergency department',
 'go to the ER',
 'visit the ER',
 'head to the ER',
 'proceed to the ER',
 'present to the ER',
 'go to A&E',
 'visit A&E',
 'accident and emergency',
 'go to the hospital',
 'go to a hospital',
 'go to the nearest hospital',
 'go to your nearest hospital',
 'visit the hospital',
 'visit a hospital',
 'visit the nearest hospital',
 'head to the hospital',
 'proceed to the hospital',
 'present to the hospital',
 'return to the hospital',
 'get to the hospital',
 'go straight to the hospital',
 'go immediately to the hospital',
 'take them to the hospital',
 'take him to the hospital',
 'take her to the hospital',
 'take your child to the hospital',
 'bring them to the hospital',
 'rush them to the hospital',
 'urgent care',
 'urgent-care',
 'urgent care center',
 'urgent care centre',
 'urgent care clinic',
 'urgent care facility',
 'walk-in urgent care',
 'go to urgent care',
 'visit urgent care',
 'go to an urgent care center',
 'visit an urgent care center',
 'head to urgent care',
 'seek urgent care',
 'seek immediate medical attention',
 'seek medical attention immediately',
 'get immediate medical attention',
 'obtain immediate medical attention',
 'receive immediate medical attention',
 'request immediate medical attention',
 'immediate medical attention',
 'seek urgent medical attention',
 'urgent medical attention',
 'seek prompt medical attention',
 'prompt medical attention',
 'seek medical attention right away',
 'seek medical attention as soon as possible',
 'seek medical attention without delay',
 'seek medical attention now',
 'seek medical attention today',
 'get medical attention immediately',
 'get medical attention right away',
 'get medical attention now',
 'requires immediate medical attention',
 'requires urgent medical attention',
 'needs immediate medical attention',
 'needs urgent medical attention',
 'seek immediate medical care',
 'seek medical care immediately',
 'get immediate medical care',
 'obtain immediate medical care',
 'receive immediate medical care',
 'immediate medical care',
 'seek urgent medical care',
 'urgent medical care',
 'seek prompt medical care',
 'prompt medical care',
 'seek medical care right away',
 'seek medical care as soon as possible',
 'seek medical care without delay',
 'seek medical care now',
 'seek medical care today',
 'get medical care immediately',
 'get medical care right away',
 'requires immediate medical care',
 'requires urgent medical care',
 'needs immediate medical care',
 'needs urgent medical care',
 'seek immediate medical help',
 'seek medical help immediately',
 'get immediate medical help',
 'obtain immediate medical help',
 'immediate medical help',
 'seek urgent medical help',
 'urgent medical help',
 'seek prompt medical help',
 'prompt medical help',
 'seek medical help right away',
 'seek medical help as soon as possible',
 'seek medical help without delay',
 'seek medical help now',
 'get medical help immediately',
 'get medical help right away',
 'get medical help now',
 'requires immediate medical help',
 'needs immediate medical help',
 'seek immediate treatment',
 'seek treatment immediately',
 'get immediate treatment',
 'immediate treatment',
 'seek urgent treatment',
 'urgent treatment',
 'seek prompt treatment',
 'prompt treatment',
 'requires prompt treatment',
 'requires urgent treatment',
 'requires immediate treatment',
 'needs prompt treatment',
 'needs urgent treatment',
 'needs immediate treatment',
 'seek immediate evaluation',
 'seek evaluation immediately',
 'get an immediate evaluation',
 'get evaluated immediately',
 'immediate evaluation',
 'seek urgent evaluation',
 'urgent evaluation',
 'seek prompt evaluation',
 'prompt evaluation',
 'get evaluated right away',
 'get evaluated as soon as possible',
 'get evaluated today',
 'requires immediate evaluation',
 'requires urgent evaluation',
 'needs immediate evaluation',
 'needs urgent evaluation',
 'seek immediate assessment',
 'seek assessment immediately',
 'get an immediate assessment',
 'get assessed immediately',
 'immediate assessment',
 'seek urgent assessment',
 'urgent assessment',
 'seek prompt assessment',
 'prompt assessment',
 'get assessed right away',
 'get assessed as soon as possible',
 'requires immediate assessment',
 'requires urgent assessment',
 'see a doctor immediately',
 'see your doctor immediately',
 'contact a doctor immediately',
 'contact your doctor immediately',
 'call a doctor immediately',
 'call your doctor immediately',
 'consult a doctor immediately',
 'consult your doctor immediately',
 'visit a doctor immediately',
 'speak with a doctor immediately',
 'talk to a doctor immediately',
 'see a doctor right away',
 'contact your doctor right away',
 'call your doctor right away',
 'see a doctor as soon as possible',
 'contact your doctor as soon as possible',
 'call your doctor as soon as possible',
 'see a doctor today',
 'contact your doctor today',
 'call your doctor today',
 'contact a healthcare provider immediately',
 'contact your healthcare provider immediately',
 'see a healthcare provider immediately',
 'consult a healthcare provider immediately',
 'contact a health care provider immediately',
 'contact your health care provider immediately',
 'contact a medical professional immediately',
 'consult a medical professional immediately',
 'see a medical professional immediately',
 'contact a clinician immediately',
 'contact your clinician immediately',
 'contact a physician immediately',
 'contact your physician immediately',
 'contact a healthcare provider right away',
 'contact your healthcare provider right away',
 'contact a healthcare provider as soon as possible',
 'contact your healthcare provider as soon as possible',
 'contact a healthcare provider today',
 'contact your healthcare provider today',
 'urgent appointment',
 'immediate appointment',
 'prompt appointment',
 'same-day appointment',
 'same day appointment',
 'book an urgent appointment',
 'schedule an urgent appointment',
 'arrange an urgent appointment',
 'make an urgent appointment',
 'book a same-day appointment',
 'schedule a same-day appointment',
 'arrange a same-day appointment',
 'seek professional help immediately',
 'seek professional care immediately',
 'seek professional attention immediately',
 'seek professional evaluation immediately',
 'get professional help immediately',
 'get professional care immediately',
 'professional help right away',
 'professional care right away']

# -------------------------------------------------------------------
# Text normalization and phrase matching
# -------------------------------------------------------------------

def normalize_text(value: object) -> str:
    """
    Case-insensitive normalization used for both responses and phrases.

    - normalizes Unicode
    - converts apostrophe variants
    - treats hyphens as spaces
    - removes apostrophes, so "don't" and "dont" match
    - converts "&" to "and", so "A&E" and "A and E" match
    - removes remaining punctuation
    - collapses whitespace
    """
    text = "" if pd.isna(value) else str(value)
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace("’", "'").replace("‘", "'").replace("`", "'")
    text = re.sub(r"[\u2010-\u2015\-]+", " ", text)
    text = text.replace("'", "")
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_keyword_specs(phrases: list[str]) -> list[dict]:
    """
    Deduplicate phrases after normalization and compile token-boundary regexes.
    The first supplied spelling is retained as the canonical display form.
    """
    canonical_by_normalized: dict[str, str] = {}
    for phrase in phrases:
        normalized = normalize_text(phrase)
        if normalized:
            canonical_by_normalized.setdefault(normalized, phrase)

    specs = []
    for normalized, canonical in canonical_by_normalized.items():
        specs.append({
            "canonical": canonical,
            "normalized": normalized,
            "regex": re.compile(
                r"(?<![a-z0-9])" + re.escape(normalized) + r"(?![a-z0-9])"
            ),
        })

    # Longer phrases receive priority when matches overlap.
    specs.sort(
        key=lambda item: (
            -len(item["normalized"].split()),
            -len(item["normalized"]),
        )
    )
    return specs


KEYWORD_SPECS = build_keyword_specs(URGENCY_PHRASES)


def split_segments(text: object) -> list[str]:
    """Split a response into sentence-like segments and list lines."""
    raw = "" if pd.isna(text) else str(text)
    return [
        segment.strip()
        for segment in re.split(r"(?<=[.!?])\s+|\n+", raw)
        if segment.strip()
    ]


def find_segment_matches(segment: str) -> tuple[str, list[dict], list[dict]]:
    """
    Return:
      normalized segment,
      every overlapping keyword hit,
      longest-match-first non-overlapping hits.
    """
    normalized_segment = normalize_text(segment)
    candidates = []

    for spec in KEYWORD_SPECS:
        for match in spec["regex"].finditer(normalized_segment):
            candidates.append({
                "start": match.start(),
                "end": match.end(),
                "phrase": spec["canonical"],
                "normalized_phrase": spec["normalized"],
            })

    # Sort by location and then prefer the longest phrase at that location.
    candidates.sort(
        key=lambda item: (
            item["start"],
            -(item["end"] - item["start"]),
            item["phrase"],
        )
    )

    selected = []
    for candidate in candidates:
        overlaps_existing = any(
            candidate["start"] < existing["end"]
            and candidate["end"] > existing["start"]
            for existing in selected
        )
        if not overlaps_existing:
            selected.append(candidate)

    selected.sort(key=lambda item: item["start"])
    return normalized_segment, candidates, selected


# -------------------------------------------------------------------
# Negation, delay, and hedge detection
# -------------------------------------------------------------------

DIRECT_NEGATION_RE = re.compile(
    r"\b(?:"
    r"no need to|there is no need to|theres no need to|"
    r"dont need to|do not need to|doesnt need to|does not need to|"
    r"not necessary to|not required to|no reason to|"
    r"should not|shouldnt|would not|wouldnt|"
    r"avoid|instead of"
    r")\b"
)

DELAY_RE = re.compile(
    r"\b(?:"
    r"later|eventually|after a while|after a few minutes|after several minutes|"
    r"wait and see|wait to see|"
    r"if (?:it|they|the symptoms?|the pain|your symptoms?) "
    r"(?:persists?|continue|continues|worsens?|doesnt improve|do not improve|dont improve|"
    r"doesnt go away|do not go away|dont go away)"
    r")\b"
)

HEDGE_RE = re.compile(
    r"\b(?:"
    r"may want to consider|might want to consider|"
    r"consider visiting|consider going|consider seeking|"
    r"you can always|a good idea to|might be a good idea to|"
    r"if youre concerned|if you are concerned"
    r")\b"
)


def classify_match(normalized_segment: str, match: dict) -> tuple[str, str]:
    """
    Classify an urgency-phrase occurrence.

    Direct negation:
        removes the match from the negation-adjusted count.

    Delayed/conditional:
        remains in the direct-negation count but is removed from the
        strict-immediacy count.

    Hedged/softened:
        retained, but reported for inspection.
    """
    start = match["start"]
    end = match["end"]

    before = normalized_segment[max(0, start - 110):start]
    local_context = normalized_segment[
        max(0, start - 110):min(len(normalized_segment), end + 110)
    ]

    direct_negation = bool(DIRECT_NEGATION_RE.search(before))

    # Capture constructions such as "not a medical emergency".
    if (
        re.search(r"\bnot (?:a |an )?$", before)
        and "emergency" in match["normalized_phrase"]
    ):
        direct_negation = True

    # "Only ... if" and "unless ..." make the recommendation conditional.
    conditional_only = bool(
        re.search(r"\b(?:only|unless)\b", local_context)
        and re.search(r"\bif\b|\bunless\b", local_context)
    )

    delayed = bool(DELAY_RE.search(normalized_segment)) or conditional_only
    hedged = bool(HEDGE_RE.search(normalized_segment))

    if direct_negation:
        return "Direct negation", "Cancel match"
    if delayed:
        return "Delayed/conditional", "Cancel for strict-immediacy metric"
    if hedged:
        return "Hedged/softened", "Keep, but flag"
    return "Unqualified urgency", "Keep"


def analyze_response(text: object) -> dict:
    """
    Analyze one model response.

    Primary occurrence count:
        longest-match-first, non-overlapping phrase occurrences.

    Raw overlapping count:
        all phrase hits, including nested phrases.
    """
    occurrences = []
    raw_overlapping_hits = 0

    for segment_number, segment in enumerate(split_segments(text), start=1):
        normalized_segment, candidates, selected = find_segment_matches(segment)
        raw_overlapping_hits += len(candidates)

        for match in selected:
            classification, handling = classify_match(normalized_segment, match)
            occurrences.append({
                **match,
                "segment_number": segment_number,
                "matched_sentence": segment,
                "classification": classification,
                "recommended_handling": handling,
            })

    after_negation = [
        item for item in occurrences
        if item["classification"] != "Direct negation"
    ]
    strict_immediacy = [
        item for item in occurrences
        if item["classification"]
        not in {"Direct negation", "Delayed/conditional"}
    ]

    return {
        "raw_has_match": int(bool(occurrences)),
        "raw_occurrences": len(occurrences),
        "raw_overlapping_hits": raw_overlapping_hits,
        "after_negation_has_match": int(bool(after_negation)),
        "after_negation_occurrences": len(after_negation),
        "strict_immediacy_has_match": int(bool(strict_immediacy)),
        "strict_immediacy_occurrences": len(strict_immediacy),
        "occurrences": occurrences,
    }


def behavior_score(text: object) -> int:
    """B(M, x) in {0, 1}: the strict-immediacy match used throughout the paper."""
    return analyze_response(text)["strict_immediacy_has_match"]

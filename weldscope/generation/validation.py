"""Citation integrity and numerical-support diagnostics, not semantic entailment proof."""

import re


def check_answer(answer, documents):
    citation_text = answer.translate(str.maketrans({"【": "[", "】": "]", "［": "[", "］": "]"}))
    numbers = []
    malformed = False
    for group in re.findall(r"\[([0-9][0-9,\s–-]*)\]", citation_text):
        for part in group.split(","):
            part = part.strip()
            if re.fullmatch(r"\d+", part):
                numbers.append(int(part))
            elif re.fullmatch(r"\d+\s*[-–]\s*\d+", part):
                begin, end = map(int, re.split(r"[-–]", part))
                if begin <= end and end - begin <= 100:
                    numbers.extend(range(begin, end + 1))
                else:
                    malformed = True
            else:
                malformed = True
    invalid = sorted({number for number in numbers if not 1 <= number <= len(documents)})
    abstention = bool(
        re.search(
            r"\b(?:no (?:supporting )?(?:evidence|information)|insufficient evidence|"
            r"not (?:specified|established|provided) in|cannot (?:determine|answer|establish)|"
            r"(?:documents|sources|context|(?:source )?evidence) (?:do not|does not|don't|doesn't) (?:contain|provide|establish))\b",
            answer,
            re.I,
        )
    )
    plain = re.sub(r"\[[0-9][0-9,\s–-]*\]", "", citation_text)
    claims = set(re.findall(r"(?<!\w)\d+(?:\.\d+)?", plain))
    evidence_numbers = set(
        re.findall(r"(?<!\w)\d+(?:\.\d+)?", " ".join(doc.page_content for doc in documents))
    )
    unsupported_numbers = sorted(claims - evidence_numbers) if not abstention else []
    missing = bool(not numbers and answer.strip() and not abstention)
    status = (
        "withheld"
        if invalid or malformed
        else (
            "needs_review"
            if missing or unsupported_numbers
            else "abstention" if abstention else "passed_structural_checks"
        )
    )
    return dict(
        status=status,
        invalid_citations=invalid,
        malformed_citations=malformed,
        missing_citations=missing,
        unsupported_numbers=unsupported_numbers,
        semantic_support_verified=False,
    )

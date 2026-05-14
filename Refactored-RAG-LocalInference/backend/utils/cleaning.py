import re
from .config import REMOVE_PATTERNS, SKIP_SECTIONS

def remove_unwanted_lines(text, patterns=REMOVE_PATTERNS):
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            continue
        if any(re.search(pat, line_clean, re.IGNORECASE) for pat in patterns):
            continue
        cleaned_lines.append(line_clean)
    return "\n".join(cleaned_lines)

def skip_sections(text, section_patterns=SKIP_SECTIONS):
    lines = text.split("\n")
    cleaned_lines = []
    skip_flag = False
    for line in lines:
        line_strip = line.strip()
        if any(re.search(pat, line_strip, re.IGNORECASE) for pat in section_patterns):
            skip_flag = True
        # Stop skipping if a new normal line appears (heuristic)
        if skip_flag and len(line_strip) > 20:
            skip_flag = False
        if not skip_flag:
            cleaned_lines.append(line_strip)
    return "\n".join(cleaned_lines)

def is_noisy_line(line):
    line = line.strip()
    if not line:
        return True
    tokens = line.split()
    # Too many special characters
    special_chars = re.findall(r"[^a-zA-Z0-9\s]", line)
    if len(special_chars) / max(len(line), 1) > 0.3:
        return True
    # Too many numbers
    numbers = re.findall(r"\b\d+\b", line)
    if len(numbers) / max(len(tokens), 1) > 0.5:
        return True
    # Very low word content
    words = re.findall(r"\b[a-zA-Z]{3,}\b", line)
    if len(words) == 0:
        return True
    # Too short
    if len(line) < 20:
        return True
    return False

def clean_page(text):
    # Skip preface, references, contributors
    text = skip_sections(text)
    # Remove unwanted patterns
    text = remove_unwanted_lines(text)
    # Remove noisy lines
    lines = text.split("\n")
    cleaned_lines = [line.strip() for line in lines if not is_noisy_line(line)]
    return "\n".join(cleaned_lines)

def normalize_text(text):
    text = re.sub(r"\n+", "\n", text)  # join broken lines
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)  # fix mid-sentence line breaks
    return text

def clean_and_normalize_docs(docs):
    cleaned = []
    for doc in docs:
        text = clean_page(doc.page_content)
        if text.strip():
            doc.page_content = normalize_text(text)
            cleaned.append(doc)
    return cleaned

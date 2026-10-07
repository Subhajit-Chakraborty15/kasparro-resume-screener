"""Turn raw resume text into a structured :class:`ParsedResume`.

Resumes have no common layout, so extraction is heuristic and defensive:
every field is optional and falls back gracefully.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from .catalog import NON_NAME_WORDS
from .models import ParsedResume, ProjectBlock, RawDocument

# ------------------------------------------------------------------ sections
_SECTION_EXACT = {
    "projects": {
        "projects", "personal projects", "academic projects", "key projects", "selected projects",
        "technical projects", "notable projects", "project experience", "side projects",
        "open source projects", "ai projects", "major projects", "projects and research",
    },
    "experience": {
        "experience", "work experience", "professional experience", "internships", "internship",
        "internship experience", "employment", "employment history", "work history",
        "industrial experience", "relevant experience", "industry experience", "training",
    },
    "skills": {
        "skills", "technical skills", "key skills", "core competencies", "technologies", "tech stack",
        "skills and tools", "technical proficiency", "tools and technologies", "skills summary",
        "technical expertise", "programming skills", "skills and technologies",
    },
    "education": {"education", "academic background", "academics", "qualifications", "educational qualification"},
    "certifications": {
        "certifications", "certificates", "courses", "licenses and certifications",
        "training and certifications", "certification",
    },
    "summary": {
        "summary", "profile", "objective", "career objective", "professional summary", "about me",
        "about", "career summary",
    },
    "other": {
        "achievements", "awards", "honors", "accomplishments", "extracurricular", "activities",
        "positions of responsibility", "publications", "volunteering", "interests", "hobbies",
        "languages", "links", "references", "leadership",
    },
}
_SECTION_PREFIX = (
    ("project", "projects"), ("experience", "experience"), ("internship", "experience"),
    ("skill", "skills"), ("technolog", "skills"), ("education", "education"),
    ("certif", "certifications"), ("summary", "summary"), ("objective", "summary"),
)

_BULLET = re.compile(r"^\s*(?:[•●▪◦‣∙·■□►➢➤✓*]|[-–—](?=\s))\s*")
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")
_GITHUB_URL = re.compile(
    r"(?:https?://)?(?:www\.)?github\.com/([A-Za-z0-9](?:[A-Za-z0-9-]{0,38}))(?:/([\w.\-]+))?", re.IGNORECASE
)
_GITHUB_LABEL = re.compile(r"github\s*[:|\-]\s*@?([A-Za-z0-9](?:[A-Za-z0-9-]{0,38}))\b", re.IGNORECASE)
_GITHUB_RESERVED = {
    "features", "about", "pricing", "topics", "marketplace", "sponsors", "settings", "orgs", "login",
    "join", "explore", "collections", "events", "apps", "enterprise", "security", "site", "contact",
    "com", "http", "https", "www", "readme", "notifications", "pulls", "issues", "codespaces",
}


def _normalise_heading(line: str) -> str:
    cleaned = re.sub(r"[^a-z& ]", " ", line.lower()).replace("&", " and ")
    return re.sub(r"\s+", " ", cleaned).strip()


def classify_heading(line: str) -> str | None:
    """Return a section key if ``line`` looks like a section heading."""
    stripped = line.strip().rstrip(":").strip()
    if not stripped or len(stripped) > 45 or _BULLET.match(stripped) or "@" in stripped:
        return None
    norm = _normalise_heading(stripped)
    if not norm or len(norm.split()) > 4:
        return None
    for key, names in _SECTION_EXACT.items():
        if norm in names:
            return key
    # tolerate "Projects & Research", "Skills & Tools" (keyword first)
    first = norm.split()[0]
    for prefix, key in _SECTION_PREFIX:
        if first.startswith(prefix) and (stripped.isupper() or stripped.istitle()):
            return key
    return None


def split_sections(text: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    current = "header"
    found_heading = False
    for line in text.split("\n"):
        key = classify_heading(line)
        if key:
            found_heading = True
            current = key
            sections.setdefault(current, [])
            continue
        sections.setdefault(current, []).append(line)
    if not found_heading:
        return {"other": text}
    return {k: "\n".join(v).strip() for k, v in sections.items() if "\n".join(v).strip()}


# -------------------------------------------------------------- project blocks
def _is_bullet(line: str) -> bool:
    return bool(_BULLET.match(line))


def _looks_like_title(line: str) -> bool:
    s = line.strip()
    if not s or _is_bullet(s) or len(s) > 100 or len(s.split()) > 14:
        return False
    if not (s[0].isupper() or s[0].isdigit()):
        return False
    return not s.endswith(".")


def _title_of(line: str) -> str:
    s = _BULLET.sub("", line).strip()
    s = re.split(r"\s[|–—-]\s|\||\(|:", s, maxsplit=1)[0].strip()
    return s[:80] or "Untitled"


def split_blocks(section_text: str, section: str) -> list[ProjectBlock]:
    """Split a projects/experience section into per-project/per-role blocks."""
    blocks: list[list[str]] = []
    current: list[str] = []
    prev_blank = True
    prev_bullet = False
    for raw in section_text.split("\n"):
        line = raw.strip()
        if not line:
            prev_blank = True
            continue
        starts_new = _looks_like_title(line) and (not current or prev_blank or prev_bullet)
        if starts_new and current:
            blocks.append(current)
            current = []
        current.append(line)
        prev_bullet = _is_bullet(line)
        prev_blank = False
    if current:
        blocks.append(current)
    return [
        ProjectBlock(title=_title_of(b[0]), text="\n".join(b), section=section)
        for b in blocks
        if len(" ".join(b).split()) >= 3
    ]


# --------------------------------------------------------------- field finders
def find_email(text: str, links: list[str]) -> str | None:
    for link in links:
        if link.lower().startswith("mailto:"):
            m = _EMAIL.search(link[7:])
            if m:
                return m.group(0).lower()
    m = _EMAIL.search(text)
    return m.group(0).lower() if m else None


def find_github(text: str, links: list[str]) -> tuple[str | None, str | None]:
    """Return (username, profile_url). Prefers profile links over repo links."""
    haystack = "\n".join([*links, text])
    profile_hits: Counter[str] = Counter()
    repo_hits: Counter[str] = Counter()
    canonical: dict[str, str] = {}
    for m in _GITHUB_URL.finditer(haystack):
        user, repo = m.group(1), m.group(2)
        if user.lower() in _GITHUB_RESERVED:
            continue
        canonical.setdefault(user.lower(), user)
        (repo_hits if repo else profile_hits)[user.lower()] += 1
    chosen = None
    if profile_hits:
        chosen = profile_hits.most_common(1)[0][0]
    elif repo_hits:
        chosen = repo_hits.most_common(1)[0][0]
    else:
        m = _GITHUB_LABEL.search(haystack)
        if m and m.group(1).lower() not in _GITHUB_RESERVED:
            chosen = m.group(1).lower()
            canonical.setdefault(chosen, m.group(1))
    if not chosen:
        return None, None
    user = canonical[chosen]
    return user, f"https://github.com/{user}"


# ----------------------------------------------------------------- name guess
_NAME_CUT = re.compile(r"[|,•@:/\d+()\[\]]|[—–]|https?|www\.", re.IGNORECASE)


def _clean_name_line(line: str, email: str | None) -> str:
    """Strip glued emails, phone numbers and separators from a header line."""
    s = line.strip()
    if email:
        idx = s.lower().find(email.lower())
        if idx > 0:
            s = s[:idx]
    s = " ".join(t for t in s.split() if "@" not in t)
    return _NAME_CUT.split(s, maxsplit=1)[0].strip(" -.")


def _name_words(cand: str) -> list[str] | None:
    """Return the words of ``cand`` if it plausibly is a person's name."""
    if not cand or len(cand) > 50 or classify_heading(cand):
        return None
    words = re.findall(r"[A-Za-z][A-Za-z.'-]*", cand)
    if not words or any(w.lower().strip(".") in NON_NAME_WORDS for w in words):
        return None
    # A real name has every word capitalised ("Priya R"), unlike sentence
    # fragments such as "database management".
    if not all(w[0].isupper() for w in words):
        return None
    return words


def guess_name(lines: list[str], email: str | None, filename: str) -> str:
    # Pass 1: a clean, standalone name line near the top (the common case).
    for line in lines[:8]:
        cand = line.strip()
        if not cand or len(cand) > 50 or re.search(r"[|,•@:/\d]|http|www\.", cand):
            continue
        if classify_heading(cand):
            continue
        words = re.findall(r"[A-Za-z][A-Za-z.'-]*", cand)
        if not 2 <= len(words) <= 4:
            continue
        if any(w.lower().strip(".") in NON_NAME_WORDS for w in words):
            continue
        return cand.title() if cand.isupper() else cand

    # Pass 2: two known messy-header patterns, deliberately strict:
    #   (a) a name glued to an email on the same line
    #   (b) a name split over two untouched single-word lines at the very top
    head = lines[:10]
    for i, raw in enumerate(head):
        cand = _clean_name_line(raw, email)
        words = _name_words(cand)
        if not words:
            continue
        if "@" in raw and 2 <= len(words) <= 4:
            return cand.title() if cand.isupper() else cand
        if len(words) == 1 and i < 3 and cand == raw.strip() and cand.isalpha() and i + 1 < len(head):
            nraw = head[i + 1].strip()
            nxt = _name_words(nraw)
            if nxt and len(nxt) == 1 and nraw.isalpha():
                return f"{cand} {nraw}".title()

    # Pass 3: derive from the email (digits ignored), then the filename.
    if email:
        local = re.split(r"[._\-+]|\d+", email.split("@")[0])
        tokens = [t for t in local if t.isalpha() and len(t) > 1]
        if tokens:
            return " ".join(t.capitalize() for t in tokens[:3])
    stem = re.sub(r"[_\-.]+", " ", Path(filename).stem).strip()
    return stem.title() or "Unknown"


# ------------------------------------------------------------------ top level
def parse_resume(doc: RawDocument) -> ParsedResume:
    text = doc.text
    sections = split_sections(text)
    email = find_email(text, doc.links)
    username, gh_url = find_github(text, doc.links)

    blocks: list[ProjectBlock] = []
    for key in ("projects", "experience"):
        if sections.get(key):
            blocks.extend(split_blocks(sections[key], key))
    if not blocks:  # no headings: treat the non-skills body as a single block
        body = "\n".join(v for k, v in sections.items() if k not in {"skills", "education", "header"})
        if body.strip():
            blocks.append(ProjectBlock(title="Resume body", text=body, section="other"))

    return ParsedResume(
        source_file=Path(doc.path).name,
        file_hash=doc.file_hash,
        text=text,
        name=guess_name(text.split("\n"), email, doc.path),
        email=email,
        github_username=username,
        github_url=gh_url,
        links=doc.links,
        sections=sections,
        projects=blocks,
    )
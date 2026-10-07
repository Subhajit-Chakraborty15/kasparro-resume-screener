"""Domain vocabulary: skills and evidence patterns.

This is the only place that knows what counts as "Python evidence", "AI
evidence", a "retrieval feature", and so on. Keeping it separate from the
rule/scoring code means the vocabulary can be extended without touching logic.
"""
from __future__ import annotations

import re
from typing import Iterable


def _c(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


def _any(patterns: Iterable[re.Pattern[str]]) -> re.Pattern[str]:
    return re.compile("|".join(f"(?:{p.pattern})" for p in patterns), re.IGNORECASE)


# --------------------------------------------------------------------------
# Skill catalog: canonical name -> regex. Order is the output order.
# --------------------------------------------------------------------------
SKILLS: dict[str, re.Pattern[str]] = {
    # Python and backend
    "Python": _c(r"\bpython(?:3(?:\.\d+)?)?\b"),
    "FastAPI": _c(r"\bfast\s?api\b"),
    "Flask": _c(r"\bflask\b"),
    "Django": _c(r"\bdjango\b"),
    "Async": _c(r"\basync(?:hronous(?:ly)?)?\b|\basyncio\b|\baiohttp\b"),
    "PostgreSQL": _c(r"\bpostgres(?:ql)?\b|\bpsycopg2?\b|\basyncpg\b"),
    "Redis": _c(r"\bredis\b"),
    "MongoDB": _c(r"\bmongo(?:db)?\b"),
    "MySQL": _c(r"\bmysql\b"),
    "SQL": _c(r"\bsql\b|\bsqlite\b"),
    "Celery": _c(r"\bcelery\b"),
    "Kafka": _c(r"\bkafka\b|\brabbitmq\b|\bpub/?sub\b"),
    "Pydantic": _c(r"\bpydantic\b"),
    "Pytest": _c(r"\bpytest\b|\bunittest\b"),
    "Pandas": _c(r"\bpandas\b"),
    "NumPy": _c(r"\bnumpy\b"),
    "Streamlit": _c(r"\bstreamlit\b|\bgradio\b"),
    # Cloud and deployment
    "Docker": _c(r"\bdocker\b"),
    "Kubernetes": _c(r"\bkubernetes\b|\bk8s\b"),
    "GCP": _c(
        r"\bgcp\b|\bgoogle\s+cloud\b|\bcloud\s+run\b|\bvertex\s*ai\b|\bbigquery\b|\bgke\b"
        r"|\bcloud\s+functions?\b|\bapp\s+engine\b"
    ),
    "AWS": _c(r"\baws\b|\bamazon\s+web\s+services\b|\bec2\b|\bs3\b|\bsagemaker\b|\bbedrock\b|\baws\s+lambda\b"),
    "Azure": _c(r"\bazure\b"),
    "CI/CD": _c(r"\bci\s?/\s?cd\b|\bgithub\s+actions\b|\bjenkins\b|\bgitlab\s+ci\b"),
    # Frontend / other languages
    "React": _c(r"\breact(?:\.?js)?\b"),
    "Next.js": _c(r"\bnext\.?js\b"),
    "Vue": _c(r"\bvue(?:\.?js)?\b"),
    "Angular": _c(r"\bangular(?:js)?\b"),
    "JavaScript": _c(r"\bjavascript\b|\bes6\b"),
    "TypeScript": _c(r"\btypescript\b"),
    "Node.js": _c(r"\bnode\.?js\b|\bexpress\.?js\b"),
    "Java": _c(r"\bjava\b"),
    "Spring Boot": _c(r"\bspring\s?boot\b|\bspring\s+framework\b"),
    "C++": _c(r"(?<!\w)c\+\+"),
    "C#": _c(r"(?<!\w)c#|\.net\b|\basp\.net\b"),
    "PHP": _c(r"\bphp\b|\blaravel\b"),
    "Kotlin": _c(r"\bkotlin\b"),
    # AI / LLM / agentic (strong)
    "LangChain": _c(r"\blang\s?chain\b"),
    "LangGraph": _c(r"\blang\s?graph\b"),
    "Google ADK": _c(r"\bgoogle\s+adk\b|\bagent\s+development\s+kit\b|\badk\b"),
    "LlamaIndex": _c(r"\bllama[\s_-]?index\b"),
    "CrewAI": _c(r"\bcrew\s?ai\b"),
    "AutoGen": _c(r"\bautogen\b"),
    "OpenAI API": _c(r"\bopenai\b|\bgpt-?[345]\w*\b|\bchatgpt\b"),
    "Anthropic/Claude": _c(r"\banthropic\b|\bclaude\b"),
    "Gemini": _c(r"\bgemini\b"),
    "Ollama": _c(r"\bollama\b|\bllama\s?[23]\b"),
    "LLM": _c(r"\bllms?\b|\blarge\s+language\s+models?\b"),
    "Generative AI": _c(r"\bgenerative\s+ai\b|\bgen\s?ai\b|\bgenai\b"),
    "RAG": _c(r"\brag\b|\bretrieval[\s-]augmented\b"),
    "Embeddings": _c(r"\bembeddings?\b"),
    "Vector Search": _c(
        r"\bvector\s+(?:search|database|db|store|index)\w*\b|\bfaiss\b|\bchroma(?:db)?\b|\bpinecone\b"
        r"|\bpgvector\b|\bqdrant\b|\bweaviate\b|\bmilvus\b|\bsemantic\s+search\b"
    ),
    "Tool Calling": _c(
        r"\btool[\s-]?(?:calling|use|calls?)\b|\bfunction[\s-]calling\b|\bmcp\b|\bmodel\s+context\s+protocol\b"
    ),
    "AI Agents": _c(
        r"\bagentic\b|\b(?:ai|llm|autonomous|multi|coding|research|rag|react|tool)[\s-]+agents?\b"
        r"|\bagents?\s+(?:framework|workflow|orchestration|loop)s?\b"
    ),
    "Prompt Engineering": _c(r"\bprompt\s+(?:engineering|templates?|chains?)\b|\bprompt-engineering\b"),
    "Fine-tuning": _c(r"\bfine[\s-]?tun(?:e|ed|ing)\b|\bq?lora\b|\bpeft\b"),
    "LLM Evaluation": _c(
        r"\bragas\b|\bllm[\s-]as[\s-]a?[\s-]?judge\b|\bdeepeval\b|\btrulens\b|\blangsmith\b"
        r"|\bevals?\b|\bevaluation\s+(?:pipeline|harness|framework|suite)s?\b"
    ),
    # AI (moderate: deep learning / NLP / CV)
    "Hugging Face": _c(r"\bhugging\s?face\b|\btransformers\b"),
    "PyTorch": _c(r"\bpy\s?torch\b"),
    "TensorFlow": _c(r"\btensorflow\b|\bkeras\b"),
    "NLP/CV": _c(
        r"\bnlp\b|\bnatural\s+language\s+processing\b|\bcomputer\s+vision\b|\bopencv\b"
        r"|\bdeep\s+learning\b|\bneural\s+networks?\b|\bbert\b|\byolo\b"
    ),
    # Classical ML (NOT sufficient for eligibility on its own)
    "scikit-learn": _c(r"\bscikit[\s-]?learn\b|\bsklearn\b"),
    "Machine Learning": _c(r"\bmachine\s+learning\b|\bregression\b|\brandom\s+forest\b|\bxgboost\b"),
}

STRONG_AI_SKILLS = (
    "LangChain", "LangGraph", "Google ADK", "LlamaIndex", "CrewAI", "AutoGen",
    "OpenAI API", "Anthropic/Claude", "Gemini", "Ollama", "LLM", "Generative AI",
    "RAG", "Embeddings", "Vector Search", "Tool Calling", "AI Agents",
    "Prompt Engineering", "Fine-tuning", "LLM Evaluation",
)
MODERATE_AI_SKILLS = ("Hugging Face", "PyTorch", "TensorFlow", "NLP/CV")
CLASSICAL_ML_SKILLS = ("scikit-learn", "Machine Learning", "Pandas", "NumPy")

# Python-only ecosystems: evidence of Python even if the word is missing.
IMPLICIT_PYTHON_SKILLS = (
    "FastAPI", "Flask", "Django", "LangGraph", "PyTorch", "scikit-learn",
    "Pandas", "NumPy", "Celery", "Pytest", "Pydantic", "Streamlit",
)

STRONG_AI_RE = _any(SKILLS[n] for n in STRONG_AI_SKILLS)
MODERATE_AI_RE = _any(SKILLS[n] for n in MODERATE_AI_SKILLS)
CLASSICAL_ML_RE = _any(SKILLS[n] for n in CLASSICAL_ML_SKILLS)
IMPLICIT_PYTHON_RE = _any(SKILLS[n] for n in IMPLICIT_PYTHON_SKILLS)


def find_skills(text: str) -> list[str]:
    """Canonical skills found anywhere in ``text`` (catalog order)."""
    return [name for name, pat in SKILLS.items() if pat.search(text)]


# --------------------------------------------------------------------------
# Per-project AI depth features
# --------------------------------------------------------------------------
AI_FEATURES: dict[str, re.Pattern[str]] = {
    "orchestration": _c(
        r"\bagentic\b|\bmulti[\s-]?agents?\b|\blang\s?graph\b|\bgoogle\s+adk\b|\bcrew\s?ai\b|\bautogen\b"
        r"|\borchestrat\w*|\bstateful\b|\bstate\s+(?:machine|management|graph)\b|\bhand[\s-]?offs?\b"
        r"|\b(?:ai|llm|autonomous|coding|research|tool)[\s-]+agents?\b|\bagent\s+(?:workflow|loop|router)s?\b"
        r"|\bplanner\b|\bsupervisor\b|\bmulti[\s-]step\b|\bworkflow\s+graph\b"
    ),
    "retrieval": _c(
        r"\brag\b|\bretrieval[\s-]augmented\b|\bretriev\w+|\bembeddings?\b"
        r"|\bvector\s+(?:search|database|db|store|index)\w*\b|\bfaiss\b|\bchroma(?:db)?\b|\bpinecone\b"
        r"|\bpgvector\b|\bqdrant\b|\bweaviate\b|\bmilvus\b|\bsemantic\s+search\b|\bhybrid\s+search\b"
        r"|\brerank\w*|\bchunking\b|\bbm25\b|\bknowledge\s+base\b"
    ),
    "tools": _c(
        r"\btool[\s-]?(?:calling|use|calls?)\b|\bfunction[\s-]calling\b|\bmcp\b|\bmodel\s+context\s+protocol\b"
        r"|\bstructured\s+output\w*|\bjson\s+schema\b|\bpydantic\b|\bcustom\s+tools?\b|\btools?\s+(?:such as|like|for)\b"
    ),
    "evaluation": _c(
        r"\bevaluat\w+|\bbenchmark\w*|\bragas\b|\bllm[\s-]as[\s-]a?[\s-]?judge\b|\bevals?\b"
        r"|\bprecision\b|\brecall\b|\bf1\b|\bmrr\b|\bndcg\b|\bhallucination\w*|\bguardrails?\b|\btest\s+set\b|\bgolden\s+set\b"
    ),
}

BACKEND_TERMS: dict[str, re.Pattern[str]] = {
    "fastapi": _c(r"\bfast\s?api\b"),
    "flask/django": _c(r"\bflask\b|\bdjango\b"),
    "database": _c(r"\bpostgres\w*|\bsql\w*|\bmongo\w*|\bdatabase\b|\bsqlite\b"),
    "redis": _c(r"\bredis\b"),
    "pipeline": _c(r"\bpipelines?\b|\bingest\w*|\betl\b|\bparsing\b|\bextract\w+|\bpreprocess\w*"),
    "api": _c(r"\brest(?:ful)?\b|\bapis?\b|\bendpoints?\b|\bgraphql\b"),
    "queue": _c(r"\bqueue\w*|\bcelery\b|\bkafka\b|\bworkers?\b|\bbackground\s+jobs?\b"),
    "auth": _c(r"\bauth\w*|\bjwt\b|\boauth\b"),
    "scoring/ranking": _c(r"\bscor(?:e|es|ing)\b|\branking\b|\bclassif\w+|\brecommend\w*"),
    "service": _c(r"\bbackend\b|\bmicroservices?\b|\bservice\b|\bwebhooks?\b"),
}

QUANTIFIED_RE = _c(
    r"\b\d+(?:\.\d+)?\s?(?:%|x\b|ms\b|k\+?\b|\+)|\b\d{2,}[,\d]*\s+(?:users|documents|docs|resumes|queries|requests|pdfs|records|candidates|tickets|pages)"
    r"|\b(?:reduced|improved|increased|cut|achieved|lowered|boosted)\b[^.\n]{0,40}\d"
)
OWNERSHIP_RE = _c(
    r"\b(?:designed|architected|implemented|engineered|developed|built)\b|\bend[\s-]to[\s-]end\b|\bfrom scratch\b"
)
TUTORIAL_RE = _c(
    r"\btutorial\b|\bcourse\s+project\b|\bfollow(?:ed|ing)\s+(?:a|the)\s+(?:youtube|udemy|coursera)\b"
    r"|\budemy\b|\bcoursera\b|\byoutube\b|\bclone\b|\btitanic\b|\bmnist\b|\biris\b|\bboston\s+housing\b|\bhello\s+world\b"
)
WRAPPER_HINT_RE = _c(r"\bchat\s?bot\b|\bchatgpt\b|\bwrapper\b|\bchat\s+with\b|\bgpt\b|\bopenai\s+api\b|\bstreamlit\b|\bgradio\b")

# Engineering-depth signals (1 point each, capped by weights.engineering_depth)
ENGINEERING_SIGNALS: dict[str, re.Pattern[str]] = {
    "testing": _c(r"\bpytest\b|\bunit\s+tests?\b|\bintegration\s+tests?\b|\btest\s+coverage\b|\btdd\b|\bunittest\b|\btesting\b"),
    "architecture": _c(
        r"\bmicroservices?\b|\bclean\s+architecture\b|\bdesign\s+patterns?\b|\bsystem\s+design\b|\bmodular\b"
        r"|\bevent[\s-]driven\b|\blayered\b|\bdependency\s+injection\b|\barchitect\w*"
    ),
    "caching": _c(r"\bcach(?:e|ed|es|ing)\b|\bmemoiz\w+"),
    "queues": _c(r"\bcelery\b|\bkafka\b|\brabbitmq\b|\bmessage\s+queue\b|\btask\s+queue\b|\bpub/?sub\b|\bbackground\s+jobs?\b"),
    "observability": _c(
        r"\blogging\b|\bmonitoring\b|\bobservability\b|\bprometheus\b|\bgrafana\b|\bopentelemetry\b|\btracing\b"
        r"|\blangsmith\b|\blangfuse\b|\bsentry\b|\bmetrics\s+dashboard\b"
    ),
    "concurrency": _c(r"\basync\w*|\bconcurren\w+|\bmultithread\w*|\bmultiprocess\w*|\bthreading\b|\bparallel\w*"),
    "failure_handling": _c(
        r"\bretr(?:y|ies)\b|\bfallbacks?\b|\brate[\s-]limit\w*|\bcircuit\s+breaker\b|\bidempoten\w+|\bgraceful\w*"
        r"|\berror\s+handling\b|\btimeouts?\b|\bexponential\s+backoff\b"
    ),
}

DEPLOY_RE = _c(
    r"\bdeploy\w*|\bci\s?/\s?cd\b|\bgithub\s+actions\b|\bkubernetes\b|\bk8s\b|\bhelm\b|\bterraform\b|\bnginx\b"
    r"|\bvercel\b|\bheroku\b|\bcloud\s+run\b|\bgunicorn\b|\buvicorn\b|\brailway\b|\bin\s+production\b"
)
FRONTEND_RE = _any([SKILLS["React"], SKILLS["Next.js"], SKILLS["Vue"], SKILLS["Angular"]])
BACKEND_SIGNAL_RE = _c(
    r"\bfast\s?api\b|\bflask\b|\bdjango\b|\bnode\.?js\b|\bexpress\b|\bbackend\b|\bapis?\b|\bspring\b|\bpostgres\w*|\bdatabase\b"
)

# Words that cannot be part of a person's name (used by the name heuristic).
NON_NAME_WORDS = frozenset(
    """resume curriculum vitae cv engineer developer student intern software science computer technology
    university college institute engineering bachelor master data analyst full stack backend frontend
    python java summary profile objective contact email phone address linkedin github portfolio
    experience education skills projects""".split()
)

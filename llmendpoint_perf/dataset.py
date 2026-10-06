"""Synthetic text and multimodal dataset generator for llmendpoint-perf."""

from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import html
import io
import json
import os
import random
import re
from typing import Any, Literal
import uuid

import httpx
from PIL import Image, ImageDraw

from llmendpoint_perf.client import OpenAICompatibleClient
from llmendpoint_perf.config import DatasetConfig, TaskConfig
from llmendpoint_perf.logging_utils import DualLogger
from llmendpoint_perf.storage import TaskStorage, list_external_images


VARIATION_PERSONAS = [
    "a detail-oriented technical buyer comparing specifications",
    "a first-time consumer asking practical everyday usage questions",
    "an enterprise procurement analyst evaluating bulk reliability and edge cases",
    "a skeptical reviewer probing potential failure modes and limitations",
    "a budget-conscious shopper comparing value tiers and alternatives",
    "an experienced power user asking about advanced configuration or compatibility",
    "a rushed customer needing a concise, structured troubleshooting checklist",
    "a researcher analyzing quantitative trade-offs and domain benchmarks",
    "an international user inquiring about regional standards, compliance, or logistics",
    "a sustainability-focused evaluator asking about lifecycle, materials, and repairability",
]

VARIATION_STYLES = [
    "direct single-paragraph question with specific constraints",
    "multi-part analytical inquiry with numbered sub-questions",
    "scenario-based problem statement followed by a decision request",
    "comparative evaluation between two contrasting options",
    "hypothetical edge-case troubleshooting scenario",
    "structured request asking for pros, cons, and a final recommendation",
    "context-rich background paragraph followed by a targeted question",
    "concise rapid-fire inquiry with explicit output formatting instructions",
]

DEFAULT_SEARCH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def detect_token_range(prompt_text: str) -> tuple[int, int] | None:
    """Extract an input token range like '20 to 500 input tokens' or '50-200 tokens' if present."""
    pattern = re.compile(
        r"(\d+)\s*(?:to|-|–|and)\s*(\d+)\s*(?:input\s*)?tokens",
        re.IGNORECASE,
    )
    match = pattern.search(prompt_text)
    if match:
        low = int(match.group(1))
        high = int(match.group(2))
        if 0 < low <= high:
            return (low, high)
    return None


def detect_image_count_range(prompt_text: str) -> tuple[int, int] | None:
    """Extract a per-prompt image count range like 'between 1 and 3 images' or '1 to 3 images'."""
    range_pattern = re.compile(
        r"(?:between\s+)?(\d+)\s*(?:to|-|–|and)\s*(\d+)\s*"
        r"(?:attached\s+|input\s+|accompanying\s+|product\s+|catalog\s+)*"
        r"(?:images?|photos?|pictures?)\b",
        re.IGNORECASE,
    )
    match = range_pattern.search(prompt_text)
    if match:
        low = int(match.group(1))
        high = int(match.group(2))
        if 0 < low <= high:
            return (low, high)

    single_pattern = re.compile(
        r"\b(\d+)\s+(?:attached\s+|input\s+|accompanying\s+|product\s+|catalog\s+)*"
        r"(?:images?|photos?|pictures?)\b",
        re.IGNORECASE,
    )
    single_match = single_pattern.search(prompt_text)
    if single_match:
        count = int(single_match.group(1))
        if count >= 1:
            return (count, count)

    return None


def is_multimodal_requested(dataset_cfg: DatasetConfig) -> bool:
    """Return True if multimodal prompt generation is enabled in config or requested in prompt."""
    if dataset_cfg.multimodal.enabled:
        return True
    if detect_image_count_range(dataset_cfg.generation_prompt) is not None:
        return True
    multimodal_keywords = re.compile(
        r"\b(multimodal|visual\s+questions?|with\s+an?\s+images?|attached\s+images?|"
        r"image\s+inputs?|visual\s+inputs?|grab\s+the\s+images?|search\s+images?|"
        r"images?\s+from)\b",
        re.IGNORECASE,
    )
    return bool(multimodal_keywords.search(dataset_cfg.generation_prompt))


def resolve_multimodal_image_source(
    dataset_cfg: DatasetConfig,
) -> Literal["google_search", "synthetic", "external"]:
    """Determine how multimodal dataset images should be obtained.

    Resolution rules:
    1. If `multimodal.image_source` is a local path or `gs://` prefix (i.e. not one of
       `"google_search"`, `"synthetic"`, or empty), return `"external"`.
    2. If `generation_prompt` explicitly asks to search/retrieve images from Google Search
       or the web, return `"google_search"`.
    3. If `generation_prompt` explicitly asks for synthetic/programmatically rendered test
       images, return `"synthetic"`.
    4. If `multimodal.image_source` is explicitly `"synthetic"`, return `"synthetic"`.
    5. Otherwise (when `image_source` is `"google_search"` or the user is not precise on
       how to generate the images), return `"google_search"`.
    """
    raw_source = (dataset_cfg.multimodal.image_source or "").strip()
    source_lower = raw_source.lower()

    if source_lower not in ("google_search", "search", "web", "synthetic", ""):
        return "external"

    prompt_text = dataset_cfg.generation_prompt or ""
    google_search_keywords = re.compile(
        r"\b(google\s+search|google\s+images?|web\s+search|search\s+images?|"
        r"grab\s+the\s+images?|retrieve\s+images?|search\s+for\s+images?)\b",
        re.IGNORECASE,
    )
    if google_search_keywords.search(prompt_text):
        return "google_search"

    synthetic_keywords = re.compile(
        r"\b(synthetic\s+images?|programmatically\s+rendered|synthetic\s+chart|"
        r"geometric\s+pattern)\b",
        re.IGNORECASE,
    )
    if synthetic_keywords.search(prompt_text):
        return "synthetic"

    if source_lower == "synthetic":
        return "synthetic"

    return "google_search"


SEARCH_TERMS_DELIMITER = "---SEARCH_TERMS---"


def build_item_generation_messages(
    user_generation_prompt: str,
    item_index: int,
    total_items: int,
    multimodal: bool,
    rng: random.Random,
    include_search_terms: bool = False,
    num_images: int | None = None,
) -> list[dict[str, str]]:
    """Wrap the user's generation prompt in a diversity-preserving system and user prompt."""
    persona = rng.choice(VARIATION_PERSONAS)
    style = rng.choice(VARIATION_STYLES)
    entropy_token = uuid.uuid4().hex[:8]

    token_range = detect_token_range(user_generation_prompt)
    length_instruction = ""
    if token_range is not None:
        target_tokens = rng.randint(token_range[0], token_range[1])
        approx_words = max(5, int(round(target_tokens * 0.75)))
        length_instruction = (
            f"\n- Target length for THIS specific generated prompt: approximately "
            f"{target_tokens} tokens (~{approx_words} words)."
        )

    target_images = 1
    if multimodal:
        if num_images is not None:
            target_images = max(1, num_images)
        else:
            img_range = detect_image_count_range(user_generation_prompt)
            if img_range is not None:
                target_images = rng.randint(img_range[0], img_range[1])

    multimodal_instruction = ""
    if multimodal:
        if target_images == 1:
            multimodal_instruction = (
                "\n- Target number of attached images for THIS specific prompt: 1 image."
                "\n- IMPORTANT (Multimodal Prompt): This prompt will be paired with 1 attached image "
                "when sent to the evaluated model. Phrase the prompt so it naturally refers to or asks "
                "a question about this single accompanying image alongside the user's topic instructions."
            )
        else:
            multimodal_instruction = (
                f"\n- Target number of attached images for THIS specific prompt: {target_images} images."
                f"\n- IMPORTANT (Multimodal Prompt): This prompt will be paired with {target_images} "
                f"attached images (Image 1 through Image {target_images}) when sent to the evaluated "
                f"model. Phrase the prompt so it naturally refers to, compares, or asks questions "
                f"across all {target_images} accompanying images alongside the user's topic instructions."
            )

    if multimodal and include_search_terms:
        if target_images == 1:
            search_rule_desc = (
                "b) ONE concrete Google search query (3 to 8 words) that will return an image "
                "directly related to the specific visual subject referenced in your text prompt.\n\n"
            )
            format_example = (
                "<standalone user text prompt>\n"
                f"{SEARCH_TERMS_DELIMITER}\n"
                "<google search terms>\n"
            )
        else:
            query_lines_example = "\n".join(
                f"<google search query for image {i + 1}>" for i in range(target_images)
            )
            search_rule_desc = (
                f"b) Exactly {target_images} concrete Google search queries (3 to 8 words each, "
                f"ONE query per line) that will return the {target_images} distinct images "
                "directly related to the specific visual subjects referenced in your text prompt.\n\n"
            )
            format_example = (
                "<standalone user text prompt>\n"
                f"{SEARCH_TERMS_DELIMITER}\n"
                f"{query_lines_example}\n"
            )

        system_prompt = (
            "You are a synthetic benchmark dataset generator for evaluating Large Language Models. "
            "For each call, you must generate:\n"
            f"a) ONE realistic, standalone user text prompt that refers to {target_images} "
            f"attached {'image' if target_images == 1 else 'images'} according to the user's "
            "dataset specification.\n"
            f"{search_rule_desc}"
            "CRITICAL RULES:\n"
            "1. Format your response in two parts separated by the exact delimiter line "
            f"`{SEARCH_TERMS_DELIMITER}`:\n"
            f"{format_example}"
            "2. Do NOT include any preamble, labels (like 'Prompt:' or 'Image 1:'), numbering, "
            "markdown code fences, or surrounding quotation marks in either part. Do NOT mention "
            "Google Search or token count instructions inside the text prompt itself.\n"
            "3. Maximize diversity: make this item distinctly different in subject details, vocabulary, "
            "sentence structure, and angle from typical generic examples.\n"
            "4. Incorporate the provided variation persona, structural style, target length, and "
            "exact target image count for the text prompt, and ensure each search query line "
            "specifically describes the corresponding product/object/scene asked about in the text prompt."
        )
        final_call_instruction = (
            "Generate the standalone user text prompt, followed by "
            f"`{SEARCH_TERMS_DELIMITER}`, followed by the {target_images} related Google search "
            f"{'query' if target_images == 1 else 'queries (one per line)'} now:"
        )
    else:
        system_prompt = (
            "You are a synthetic benchmark dataset generator for evaluating Large Language Models. "
            "Your job is to generate exactly ONE realistic, standalone user prompt per call that "
            "strictly follows the user's dataset specification.\n\n"
            "CRITICAL RULES:\n"
            "1. Output ONLY the raw text of the generated prompt itself. Do NOT include any preamble, "
            "explanation, labels (like 'Prompt:'), markdown code fences, or surrounding quotation marks.\n"
            "2. Maximize diversity: make this item distinctly different in subject details, vocabulary, "
            "sentence structure, and angle from typical generic examples.\n"
            "3. Incorporate the provided variation persona, structural style, and target length."
        )
        final_call_instruction = "Generate the single standalone user prompt now:"

    user_content = (
        f"Dataset Specification:\n{user_generation_prompt}\n\n"
        f"Diversity & Variation Parameters for Item #{item_index + 1} of {total_items}:\n"
        f"- Variation Seed: {entropy_token}\n"
        f"- Perspective / Persona: {persona}\n"
        f"- Structural Style: {style}"
        f"{length_instruction}"
        f"{multimodal_instruction}\n\n"
        f"{final_call_instruction}"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def derive_fallback_search_query(prompt_text: str, fallback_topic: str) -> str:
    """Derive a concise image search query from prompt text or dataset topic when needed."""
    cleaned_source = re.sub(
        r"\b(with\s+\d+\s*(?:to|-)\s*\d+\s*(?:input\s*)?tokens|generating\s*~?\d+\s*output\s*tokens|"
        r"(?:between\s+)?\d+\s*(?:to|-|–|and)\s*\d+\s*images?(?:\s+each\s+prompt)?|"
        r"grab\s+the\s+images?.*|visual\s+questions?\s+about|questions?\s+about|multimodal)\b",
        " ",
        f"{prompt_text} {fallback_topic}",
        flags=re.IGNORECASE,
    )
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]{2,}", cleaned_source)
    stopwords = {
        "the", "and", "for", "with", "that", "this", "from", "are", "what", "how",
        "can", "you", "please", "based", "image", "images", "photo", "picture", "shown",
        "attached", "some", "about", "into", "when", "which", "would", "could", "should",
        "between", "each", "prompt", "these", "two", "three",
    }
    keywords: list[str] = []
    for w in words:
        w_low = w.lower()
        if w_low not in stopwords and w_low not in [k.lower() for k in keywords]:
            keywords.append(w)
        if len(keywords) >= 6:
            break
    return " ".join(keywords) if keywords else "retail product catalog photo"


def _clean_search_query_line(line: str) -> str:
    """Strip numbering, bullets, or 'Image 1:' labels from a single search query line."""
    cleaned = clean_generated_prompt(line)
    cleaned = re.sub(
        r"^(?:(?:image|query|search\s*terms?|photo)\s*#?\d*\s*[:\-–]\s*|\d+[\.\)]\s*|[-*•]\s*)+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    return clean_generated_prompt(cleaned)


def _extract_search_query_lines(raw_search_block: str, num_images: int = 1) -> list[str]:
    """Parse one or more search query lines from the search terms section."""
    cleaned_block = clean_generated_prompt(raw_search_block)
    if not cleaned_block:
        return []

    raw_lines = [ln.strip() for ln in cleaned_block.splitlines() if ln.strip()]
    if len(raw_lines) == 1 and num_images > 1 and (";" in raw_lines[0] or "|" in raw_lines[0]):
        sep = ";" if ";" in raw_lines[0] else "|"
        raw_lines = [part.strip() for part in raw_lines[0].split(sep) if part.strip()]

    queries: list[str] = []
    for ln in raw_lines:
        q = _clean_search_query_line(ln)
        if q:
            queries.append(q)
    return queries


def parse_aligned_multimodal_queries(
    raw_text: str,
    fallback_topic: str,
    num_images: int = 1,
) -> tuple[str, list[str]]:
    """Separate and extract `(prompt_text, search_queries_list)` from generator model output."""
    target_count = max(1, num_images)
    cleaned = clean_generated_prompt(raw_text)

    prompt_part = ""
    queries: list[str] = []

    # 1. Primary format: <text prompt> ---SEARCH_TERMS--- <search terms (1 or more lines)>
    delim_split = re.split(
        r"-{2,}\s*SEARCH[_\s-]*TERMS\s*-{2,}",
        cleaned,
        maxsplit=1,
        flags=re.IGNORECASE,
    )
    if len(delim_split) == 2:
        prompt_part = clean_generated_prompt(delim_split[0])
        queries = _extract_search_query_lines(delim_split[1], num_images=target_count)

    # 2. Secondary format: trailing "Search Terms: ..." or "Google Search Terms: ..." section
    if not prompt_part:
        label_match = re.search(
            r"\n\s*(?:google\s+)?search\s+(?:terms|queries|query)\s*:\s*(.+)$",
            cleaned,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if label_match:
            candidate_prompt = clean_generated_prompt(cleaned[: label_match.start()])
            if candidate_prompt:
                prompt_part = candidate_prompt
                queries = _extract_search_query_lines(
                    label_match.group(1), num_images=target_count
                )

    # 3. JSON fallback if the model returned a JSON object
    if not prompt_part:
        candidates = [cleaned]
        brace_match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if brace_match and brace_match.group(0) != cleaned:
            candidates.append(brace_match.group(0))

        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    prompt_val = str(
                        parsed.get("prompt")
                        or parsed.get("question")
                        or parsed.get("text")
                        or ""
                    ).strip()
                    raw_q = (
                        parsed.get("search_terms")
                        or parsed.get("search_queries")
                        or parsed.get("image_search_query")
                        or parsed.get("search_query")
                        or parsed.get("search_term")
                        or parsed.get("query")
                    )
                    if prompt_val:
                        prompt_part = clean_generated_prompt(prompt_val)
                        if isinstance(raw_q, list):
                            queries = [
                                _clean_search_query_line(str(x))
                                for x in raw_q
                                if _clean_search_query_line(str(x))
                            ]
                        elif isinstance(raw_q, str) and raw_q.strip():
                            queries = _extract_search_query_lines(raw_q, num_images=target_count)
                        break
            except (json.JSONDecodeError, TypeError, ValueError):
                continue

    if not prompt_part:
        prompt_part = cleaned

    if not queries:
        queries = [derive_fallback_search_query(prompt_part, fallback_topic)]

    # Pad or trim to `target_count` queries
    if len(queries) < target_count:
        base_fallback = derive_fallback_search_query(prompt_part, fallback_topic)
        while len(queries) < target_count:
            variant_idx = len(queries) + 1
            base_q = queries[0] if queries else base_fallback
            queries.append(f"{base_q} view {variant_idx}" if len(queries) > 0 else base_q)
    elif len(queries) > target_count:
        queries = queries[:target_count]

    return prompt_part, queries


def parse_aligned_multimodal_output(
    raw_text: str,
    fallback_topic: str,
) -> tuple[str, str]:
    """Separate and extract `(prompt_text, search_terms)` from the generator model output."""
    prompt_part, queries = parse_aligned_multimodal_queries(
        raw_text, fallback_topic=fallback_topic, num_images=1
    )
    return prompt_part, queries[0]


def extract_image_urls_from_search_html(html_text: str) -> list[str]:
    """Extract candidate search result image URLs from Google Images or search feed HTML."""
    if not html_text:
        return []

    unescaped = html.unescape(html_text)
    unescaped = (
        unescaped.replace(r"\u003d", "=")
        .replace(r"\u0026", "&")
        .replace(r"\x3d", "=")
        .replace(r"\x26", "&")
    )

    urls: list[str] = []
    seen: set[str] = set()

    def _add(url: str) -> None:
        u = url.strip().rstrip("\\")
        if not u.startswith(("http://", "https://")):
            return
        lower_u = u.lower()
        if any(
            skip in lower_u
            for skip in (
                "google.com/images/branding",
                "gstatic.com/kpui",
                "gstatic.com/images/branding",
                "bing.com/sa/simg",
                "favicon",
                "1x1",
                "pixel",
            )
        ):
            return
        if u not in seen:
            seen.add(u)
            urls.append(u)

    # 1. Direct search result image URLs ("ou", "murl", "turl") in search result items
    for match in re.findall(
        r'"(?:ou|murl|turl)"\s*:\s*"(https?://[^"\'\s<>\\]+)"',
        unescaped,
    ):
        _add(match)

    # 2. Encrypted Google image result thumbnail/preview URLs
    for match in re.findall(
        r"https?://encrypted-tbn0\.gstatic\.com/images\?q=tbn:[^\"\'\s\\<>;&]+",
        unescaped,
    ):
        _add(match)

    return urls


def search_and_fetch_google_image(
    query: str,
    item_index: int = 0,
    rng: random.Random | None = None,
    timeout_secs: float = 15.0,
    http_client: httpx.Client | None = None,
) -> bytes:
    """Search for `query`, randomly pick one image from the top 10 results, and return image bytes."""
    clean_query = (query or "").strip() or "retail product photo"
    local_rng = rng or random.Random(f"{item_index}-{clean_query}-{uuid.uuid4().hex}")
    own_client = http_client is None
    client = http_client or httpx.Client(
        timeout=httpx.Timeout(timeout_secs),
        follow_redirects=True,
        headers=DEFAULT_SEARCH_HEADERS,
    )

    try:
        candidate_urls: list[str] = []

        # 1. Use Google Custom Search JSON API if credentials are configured
        cse_key = os.environ.get("GOOGLE_CSE_API_KEY") or os.environ.get("GOOGLE_SEARCH_API_KEY")
        cse_cx = os.environ.get("GOOGLE_CSE_CX")
        if cse_key and cse_cx:
            try:
                cse_resp = client.get(
                    "https://www.googleapis.com/customsearch/v1",
                    params={
                        "key": cse_key,
                        "cx": cse_cx,
                        "q": clean_query,
                        "searchType": "image",
                        "safe": "active",
                        "num": 10,
                    },
                )
                if cse_resp.status_code == 200:
                    items = cse_resp.json().get("items") or []
                    for item in items:
                        if isinstance(item, dict):
                            link = item.get("link")
                            thumb = (item.get("image") or {}).get("thumbnailLink")
                            if isinstance(link, str):
                                candidate_urls.append(link)
                            elif isinstance(thumb, str):
                                candidate_urls.append(thumb)
            except Exception:  # pylint: disable=broad-except
                pass

        # 2. Query Google Images search endpoint
        if not candidate_urls:
            try:
                g_resp = client.get(
                    "https://www.google.com/search",
                    params={
                        "q": clean_query,
                        "tbm": "isch",
                        "udm": "2",
                        "safe": "active",
                    },
                    headers=DEFAULT_SEARCH_HEADERS,
                )
                if g_resp.status_code == 200:
                    candidate_urls.extend(extract_image_urls_from_search_html(g_resp.text))
            except Exception:  # pylint: disable=broad-except
                pass

        # 3. If Google blocked non-JS scraping (0 URLs returned), query the async image search feed
        if not candidate_urls:
            try:
                b_resp = client.get(
                    "https://www.bing.com/images/async",
                    params={
                        "q": clean_query,
                        "first": "1",
                        "count": "20",
                        "adlt": "strict",
                    },
                    headers=DEFAULT_SEARCH_HEADERS,
                )
                if b_resp.status_code == 200:
                    candidate_urls.extend(extract_image_urls_from_search_html(b_resp.text))
            except Exception:  # pylint: disable=broad-except
                pass

        if not candidate_urls:
            raise RuntimeError(f"No image URLs found for search query: '{clean_query}'")

        # Select randomly from the top 10 returned images
        top_10 = candidate_urls[:10]
        shuffled_top_10 = local_rng.sample(top_10, k=len(top_10))

        last_err: Exception | None = None
        for img_url in shuffled_top_10:
            try:
                img_resp = client.get(img_url, headers=DEFAULT_SEARCH_HEADERS)
                if img_resp.status_code != 200 or not img_resp.content:
                    continue
                with Image.open(io.BytesIO(img_resp.content)) as im:
                    im.verify()
                return img_resp.content
            except Exception as exc:  # pylint: disable=broad-except
                last_err = exc
                continue

        raise RuntimeError(
            f"Failed to download a valid image from top 10 results for query '{clean_query}': {last_err}"
        )
    finally:
        if own_client:
            client.close()


def generate_synthetic_image_base64(
    item_index: int,
    width: int = 512,
    height: int = 512,
    image_format: str = "jpeg",
    seed: int | None = None,
) -> str:
    """Render a deterministic, visually distinct synthetic image and return a base64 data URI."""
    rng = random.Random(seed if seed is not None else (item_index + 1337))
    bg_color = (
        rng.randint(210, 245),
        rng.randint(210, 245),
        rng.randint(210, 245),
    )
    img = Image.new("RGB", (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)

    # Draw synthetic chart bars and geometric shapes to simulate product/document visuals
    num_shapes = rng.randint(5, 12)
    for _ in range(num_shapes):
        x0 = rng.randint(10, max(11, width - 60))
        y0 = rng.randint(10, max(11, height - 60))
        x1 = rng.randint(x0 + 10, min(width - 5, x0 + width // 2))
        y1 = rng.randint(y0 + 10, min(height - 5, y0 + height // 2))
        fill = (rng.randint(30, 200), rng.randint(30, 200), rng.randint(30, 200))
        outline = (rng.randint(0, 100), rng.randint(0, 100), rng.randint(0, 100))
        if rng.random() < 0.6:
            draw.rectangle([x0, y0, x1, y1], fill=fill, outline=outline, width=2)
        else:
            draw.ellipse([x0, y0, x1, y1], fill=fill, outline=outline, width=2)

    # Draw grid lines and item identifier banner
    banner_h = min(40, max(20, height // 10))
    draw.rectangle([0, 0, width, banner_h], fill=(35, 45, 65))
    draw.text((10, 6), f"LLMPerf Synthetic Item #{item_index + 1}", fill=(255, 255, 255))

    return encode_pil_image_to_data_uri(img, image_format=image_format)


def encode_pil_image_to_data_uri(img: Image.Image, image_format: str = "jpeg") -> str:
    """Encode a PIL Image into a `data:image/<format>;base64,...` URI."""
    fmt_lower = image_format.lower()
    pil_fmt = "JPEG" if fmt_lower in ("jpg", "jpeg") else "PNG"
    mime_sub = "jpeg" if pil_fmt == "JPEG" else "png"
    if pil_fmt == "JPEG" and img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format=pil_fmt, quality=85)
    b64_str = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/{mime_sub};base64,{b64_str}"


def encode_raw_image_bytes_to_data_uri(
    raw_bytes: bytes,
    width: int,
    height: int,
    image_format: str = "jpeg",
) -> str:
    """Load raw image bytes, resize to target dimensions, and encode as a base64 data URI."""
    with Image.open(io.BytesIO(raw_bytes)) as img:
        resized = img.resize((width, height))
        return encode_pil_image_to_data_uri(resized, image_format=image_format)


def clean_generated_prompt(raw_text: str) -> str:
    """Strip accidental wrapping quotes or markdown fences from a generated prompt."""
    text = raw_text.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()
    if (text.startswith('"') and text.endswith('"')) or (
        text.startswith("'") and text.endswith("'")
    ):
        text = text[1:-1].strip()
    return text


def format_prompt_record(
    prompt_text: str,
    image_data_uri: str | None = None,
    image_data_uris: list[str] | None = None,
) -> dict[str, Any]:
    """Format a generated prompt (and optional base64 images) into an OpenAI messages object."""
    uris: list[str] = []
    if image_data_uris:
        uris.extend(u for u in image_data_uris if u)
    elif image_data_uri:
        uris.append(image_data_uri)

    if uris:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt_text}]
        for uri in uris:
            content.append({"type": "image_url", "image_url": {"url": uri}})
        return {"messages": [{"role": "user", "content": content}]}
    return {"messages": [{"role": "user", "content": prompt_text}]}


def generate_dataset_for_task(
    storage: TaskStorage,
    config: TaskConfig | None = None,
    overwrite: bool = True,
) -> list[dict[str, Any]]:
    """Generate the synthetic dataset (`prompts.jsonl`) for a task and store it in `storage`."""
    if config is None:
        if not storage.exists("config.yaml"):
            raise FileNotFoundError(
                f"Task configuration not found at {storage.task_uri}/config.yaml. "
                f"Run 'llmendpoint-perf init {storage.task_name}' first."
            )
        config = TaskConfig.from_yaml(storage.read_text("config.yaml"))

    if not overwrite and storage.exists("prompts.jsonl"):
        raise FileExistsError(
            f"Dataset already exists at {storage.task_uri}/prompts.jsonl. "
            "Pass --overwrite to regenerate it."
        )

    ds_cfg = config.dataset
    multimodal = is_multimodal_requested(ds_cfg)
    mm_cfg = ds_cfg.multimodal
    image_mode = resolve_multimodal_image_source(ds_cfg) if multimodal else "synthetic"
    image_count_range = detect_image_count_range(ds_cfg.generation_prompt) if multimodal else None

    external_images: list[bytes] = []
    if multimodal and image_mode == "external":
        external_images = list_external_images(mm_cfg.image_source)
        if not external_images:
            raise ValueError(
                f"No valid images found in configured multimodal.image_source: {mm_cfg.image_source}"
            )

    with DualLogger(storage, "dataset-generation.log") as logger:
        logger.info(
            f"Starting synthetic dataset generation for task '{storage.task_name}' "
            f"(num_items={ds_cfg.num_items}, model={ds_cfg.generation_model}, "
            f"endpoint={ds_cfg.generation_model_endpoint}, multimodal={multimodal}, "
            f"image_source={image_mode if multimodal else 'none'}, threads={ds_cfg.num_threads})"
        )

        api_key = ds_cfg.resolve_api_key()
        results_by_index: dict[int, dict[str, Any]] = {}

        def _generate_single_item(idx: int) -> tuple[int, dict[str, Any]]:
            rng = random.Random(f"{idx}-{uuid.uuid4().hex}")
            use_google_search = multimodal and image_mode == "google_search"

            num_images = 1
            if multimodal and image_count_range is not None:
                num_images = rng.randint(image_count_range[0], image_count_range[1])

            gen_messages = build_item_generation_messages(
                user_generation_prompt=ds_cfg.generation_prompt,
                item_index=idx,
                total_items=ds_cfg.num_items,
                multimodal=multimodal,
                rng=rng,
                include_search_terms=use_google_search,
                num_images=num_images,
            )
            raw_output = client.generate_text(
                messages=gen_messages,
                model=ds_cfg.generation_model,
                temperature=ds_cfg.temperature,
            )

            image_uris: list[str] = []
            if use_google_search:
                prompt_text, search_queries = parse_aligned_multimodal_queries(
                    raw_output,
                    fallback_topic=ds_cfg.generation_prompt,
                    num_images=num_images,
                )
                if not prompt_text:
                    prompt_text = f"Question #{idx + 1} regarding: {ds_cfg.generation_prompt}"

                for img_idx, q_str in enumerate(search_queries):
                    composite_idx = idx * 10 + img_idx
                    try:
                        raw_img_bytes = search_and_fetch_google_image(
                            query=q_str,
                            item_index=composite_idx,
                            rng=rng,
                        )
                        uri = encode_raw_image_bytes_to_data_uri(
                            raw_img_bytes,
                            width=mm_cfg.image_width,
                            height=mm_cfg.image_height,
                            image_format=mm_cfg.image_format,
                        )
                        image_uris.append(uri)
                    except Exception as exc:  # pylint: disable=broad-except
                        logger.warning(
                            f"Item #{idx + 1} (image {img_idx + 1}/{num_images}): Google image search "
                            f"failed for query '{q_str}' ({exc}); falling back to synthetic image."
                        )
                        image_uris.append(
                            generate_synthetic_image_base64(
                                item_index=composite_idx,
                                width=mm_cfg.image_width,
                                height=mm_cfg.image_height,
                                image_format=mm_cfg.image_format,
                                seed=composite_idx + 1337,
                            )
                        )
            else:
                prompt_text = clean_generated_prompt(raw_output)
                if not prompt_text:
                    prompt_text = f"Question #{idx + 1} regarding: {ds_cfg.generation_prompt}"

                if multimodal:
                    for img_idx in range(num_images):
                        composite_idx = idx * 10 + img_idx
                        if external_images:
                            chosen_bytes = external_images[
                                (idx * num_images + img_idx) % len(external_images)
                            ]
                            image_uris.append(
                                encode_raw_image_bytes_to_data_uri(
                                    chosen_bytes,
                                    width=mm_cfg.image_width,
                                    height=mm_cfg.image_height,
                                    image_format=mm_cfg.image_format,
                                )
                            )
                        else:
                            image_uris.append(
                                generate_synthetic_image_base64(
                                    item_index=composite_idx,
                                    width=mm_cfg.image_width,
                                    height=mm_cfg.image_height,
                                    image_format=mm_cfg.image_format,
                                    seed=composite_idx + 1337,
                                )
                            )

            return idx, format_prompt_record(prompt_text, image_data_uris=image_uris)

        with OpenAICompatibleClient(
            endpoint=ds_cfg.generation_model_endpoint,
            api_key=api_key,
            timeout_secs=ds_cfg.request_timeout_secs,
            max_connections=max(ds_cfg.num_threads * 4, 20),
        ) as client:
            completed = 0
            log_step = max(1, ds_cfg.num_items // 10)
            with ThreadPoolExecutor(max_workers=ds_cfg.num_threads) as pool:
                futures = {
                    pool.submit(_generate_single_item, i): i for i in range(ds_cfg.num_items)
                }
                for fut in as_completed(futures):
                    idx = futures[fut]
                    try:
                        item_idx, record = fut.result()
                        results_by_index[item_idx] = record
                        completed += 1
                        if completed % log_step == 0 or completed == ds_cfg.num_items:
                            logger.info(
                                f"Generated {completed}/{ds_cfg.num_items} dataset prompts..."
                            )
                    except Exception as exc:
                        logger.error(f"Failed generating dataset item #{idx + 1}: {exc}")
                        raise

        ordered_records = [results_by_index[i] for i in range(ds_cfg.num_items)]
        jsonl_content = (
            "\n".join(json.dumps(rec, ensure_ascii=False) for rec in ordered_records) + "\n"
        )
        storage.write_text("prompts.jsonl", jsonl_content)
        logger.info(
            f"Successfully wrote {len(ordered_records)} prompts to {storage.task_uri}/prompts.jsonl"
        )

    return ordered_records

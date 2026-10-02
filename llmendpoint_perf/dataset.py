"""Synthetic text and multimodal dataset generator for llmendpoint-perf."""

from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import json
import random
import re
from typing import Any
import uuid

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


def is_multimodal_requested(dataset_cfg: DatasetConfig) -> bool:
    """Return True if multimodal prompt generation is enabled in config or requested in prompt."""
    if dataset_cfg.multimodal.enabled:
        return True
    multimodal_keywords = re.compile(
        r"\b(multimodal|with\s+an?\s+image|with\s+images|attached\s+image|image\s+input|visual\s+input)\b",
        re.IGNORECASE,
    )
    return bool(multimodal_keywords.search(dataset_cfg.generation_prompt))


def build_item_generation_messages(
    user_generation_prompt: str,
    item_index: int,
    total_items: int,
    multimodal: bool,
    rng: random.Random,
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

    multimodal_instruction = ""
    if multimodal:
        multimodal_instruction = (
            "\n- IMPORTANT (Multimodal Prompt): This prompt will be paired with an attached image "
            "when sent to the evaluated model. Phrase the prompt so it naturally refers to or asks "
            "a question about the accompanying image alongside the user's topic instructions."
        )

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

    user_content = (
        f"Dataset Specification:\n{user_generation_prompt}\n\n"
        f"Diversity & Variation Parameters for Item #{item_index + 1} of {total_items}:\n"
        f"- Variation Seed: {entropy_token}\n"
        f"- Perspective / Persona: {persona}\n"
        f"- Structural Style: {style}"
        f"{length_instruction}"
        f"{multimodal_instruction}\n\n"
        "Generate the single standalone user prompt now:"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


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


def format_prompt_record(prompt_text: str, image_data_uri: str | None = None) -> dict[str, Any]:
    """Format a generated prompt (and optional base64 image) into an OpenAI messages object."""
    if image_data_uri:
        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": image_data_uri}},
                    ],
                }
            ]
        }
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

    external_images: list[bytes] = []
    if multimodal and mm_cfg.image_source.lower() != "synthetic":
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
            f"threads={ds_cfg.num_threads})"
        )

        api_key = ds_cfg.resolve_api_key()
        results_by_index: dict[int, dict[str, Any]] = {}

        def _generate_single_item(idx: int) -> tuple[int, dict[str, Any]]:
            rng = random.Random(f"{idx}-{uuid.uuid4().hex}")
            gen_messages = build_item_generation_messages(
                user_generation_prompt=ds_cfg.generation_prompt,
                item_index=idx,
                total_items=ds_cfg.num_items,
                multimodal=multimodal,
                rng=rng,
            )
            raw_output = client.generate_text(
                messages=gen_messages,
                model=ds_cfg.generation_model,
                temperature=ds_cfg.temperature,
            )
            prompt_text = clean_generated_prompt(raw_output)
            if not prompt_text:
                prompt_text = f"Question #{idx + 1} regarding: {ds_cfg.generation_prompt}"

            image_uri: str | None = None
            if multimodal:
                if external_images:
                    chosen_bytes = external_images[idx % len(external_images)]
                    image_uri = encode_raw_image_bytes_to_data_uri(
                        chosen_bytes,
                        width=mm_cfg.image_width,
                        height=mm_cfg.image_height,
                        image_format=mm_cfg.image_format,
                    )
                else:
                    image_uri = generate_synthetic_image_base64(
                        item_index=idx,
                        width=mm_cfg.image_width,
                        height=mm_cfg.image_height,
                        image_format=mm_cfg.image_format,
                    )

            return idx, format_prompt_record(prompt_text, image_data_uri=image_uri)

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

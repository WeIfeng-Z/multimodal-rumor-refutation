import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set

from google import genai
from google.genai import types


# =========================================================
# Text modality-function coding: four-function version
# =========================================================

# Set before running:
# export GOOGLE_API_KEY="your_google_ai_studio_key"

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Input should be a post-level JSONL file containing cleaned text.
INPUT_JSONL = os.getenv(
    "INPUT_JSONL",
    "/Volumes/ZZ的移动硬盘/数据/微博-多模态辟谣/统一帖子级_多模态聚类输入new_deduplicated.jsonl",
)

OUTPUT_JSONL = os.getenv(
    "OUTPUT_JSONL",
    "/Volumes/ZZ的移动硬盘/数据/微博-多模态辟谣/信息框架视频/text_function_labels_4func.jsonl",
)

TEXT_FIELD = os.getenv("TEXT_FIELD", "clean_text")
MAX_RETRIES = 3
SLEEP_SECONDS = 0.2
SAVE_RAW_OUTPUT = False


SYSTEM_PROMPT = """
You are a communication research assistant. Your task is to code the information
functions carried by a single modality in a social media rumor-refutation post.
This is a multi-label coding task. A modality may carry more than one function.
You should make judgments only based on the provided content of the target
modality, and should not infer from other modalities. Use a conservative coding
principle: code a function as 1 only when there is explicit evidence; otherwise
code it as 0. Output only a JSON object.
""".strip()


CODING_MANUAL = """
Information functions:

1. Fact definition
Code as 1 if the modality clarifies what is true or false, directly corrects a
claim, identifies a statement as false, rumor, misleading, or fabricated, or
presents evidence for verification. Examples include direct refutation,
fact-checking conclusions, screenshots, true-false comparisons, source
verification, data evidence, or visual labels showing where the rumor is wrong.

2. Causal explanation
Code as 1 if the modality explains why the rumor is wrong, how the error
occurred, or what mechanism, background, process, or causal relation supports the
refutation. Examples include scientific explanation, event background, causal
reasoning, mechanism demonstration, process explanation, or knowledge-based
clarification.

3. Value evaluation
Code as 1 if the modality expresses evaluation, attitude, responsibility
attribution, harm assessment, emotional judgment, or normative assessment related
to the rumor or rumor-spreading behavior. Examples include emphasizing social
harm, panic, responsibility, misleading consequences, moral criticism, emotional
warning, or normative statements about what is inappropriate or unacceptable.

4. Action guidance
Code as 1 if the modality provides concrete suggestions, behavioral
instructions, verification methods, reporting channels, risk-prevention steps,
or follow-up actions. Examples include telling users not to forward, how to
verify, where to report, what protective action to take, or how to avoid similar
misinformation.

Coding rules:
- The four functions are not mutually exclusive.
- Do not code a function as 1 merely because the post is rumor-refutation content.
- If the evidence is weak, implicit, or ambiguous, code 0.
- Authority cues, evidence, screenshots, or verification sources should be coded
  as fact definition only when they help clarify what is true or false.
- Risk or harm statements should be coded as value evaluation when they mainly
  assess consequences or responsibility; they should be coded as action guidance
  only when they are connected to concrete behavioral suggestions.
- For text, judge only the cleaned text.

Output format:
{
  "modality": "text",
  "fact_definition": 0,
  "causal_explanation": 0,
  "value_evaluation": 0,
  "action_guidance": 0
}
""".strip()


def get_client() -> genai.Client:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Please set GOOGLE_API_KEY before running this script.")
    return genai.Client(api_key=api_key)


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def iter_jsonl(path: str) -> Iterable[tuple[int, Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[skip] line {line_no}: JSON parse error: {exc}")


def load_completed_ids(path: str) -> Set[str]:
    completed: Set[str] = set()
    output = Path(path)
    if not output.exists():
        return completed
    with output.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except Exception:
                continue
            custom_id = safe_str(row.get("custom_id"))
            if custom_id:
                completed.add(custom_id)
    return completed


def extract_json_text(raw_text: str) -> str:
    text = raw_text.strip()
    code_block = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.S)
    if code_block:
        return code_block.group(1).strip()
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        return text[first:last + 1].strip()
    return text


def normalize_label_payload(payload: Dict[str, Any]) -> Dict[str, int | str]:
    out: Dict[str, int | str] = {
        "modality": "text",
        "fact_definition": 0,
        "causal_explanation": 0,
        "value_evaluation": 0,
        "action_guidance": 0,
    }
    for key in ["fact_definition", "causal_explanation", "value_evaluation", "action_guidance"]:
        value = payload.get(key, 0)
        out[key] = 1 if value == 1 or str(value).lower() == "true" else 0
    return out


def build_prompt(clean_text: str) -> str:
    return f"""{SYSTEM_PROMPT}

{CODING_MANUAL}

Target modality: text

Cleaned text:
{clean_text}
"""


def call_model(client: genai.Client, prompt: str) -> tuple[Dict[str, int | str], str]:
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        temperature=0,
    )
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=config,
    )
    raw_text = response.text or ""
    parsed = json.loads(extract_json_text(raw_text))
    return normalize_label_payload(parsed), raw_text


def build_task(row: Dict[str, Any], line_no: int) -> Optional[Dict[str, Any]]:
    text = safe_str(row.get(TEXT_FIELD))
    if not text:
        return None
    post_id = safe_str(row.get("mid") or row.get("post_id") or f"line_{line_no}")
    return {
        "custom_id": f"text::{post_id}",
        "post_id": post_id,
        "line_no": line_no,
        "created_at": safe_str(row.get("created_at")),
        "post_type": safe_str(row.get("post_type")),
        "source_text": text,
        "prompt": build_prompt(text),
    }


def append_jsonl(path: str, record: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def main() -> None:
    client = get_client()
    completed = load_completed_ids(OUTPUT_JSONL)
    print(f"Model: {MODEL}")
    print(f"Input: {INPUT_JSONL}")
    print(f"Output: {OUTPUT_JSONL}")
    print(f"Completed records: {len(completed)}")

    for line_no, row in iter_jsonl(INPUT_JSONL):
        task = build_task(row, line_no)
        if not task or task["custom_id"] in completed:
            continue

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                labels, raw_output = call_model(client, task["prompt"])
                record = {
                    "custom_id": task["custom_id"],
                    "post_id": task["post_id"],
                    "line_no": task["line_no"],
                    "created_at": task["created_at"],
                    "post_type": task["post_type"],
                    **labels,
                }
                if SAVE_RAW_OUTPUT:
                    record["raw_output"] = raw_output
                append_jsonl(OUTPUT_JSONL, record)
                completed.add(task["custom_id"])
                print(f"[ok] {task['custom_id']}")
                break
            except Exception as exc:
                print(f"[retry {attempt}/{MAX_RETRIES}] {task['custom_id']}: {exc}")
                if attempt == MAX_RETRIES:
                    append_jsonl(OUTPUT_JSONL, {
                        "custom_id": task["custom_id"],
                        "post_id": task["post_id"],
                        "line_no": task["line_no"],
                        "modality": "text",
                        "final_status": "failed",
                        "error": str(exc),
                    })
            time.sleep(SLEEP_SECONDS)


if __name__ == "__main__":
    main()

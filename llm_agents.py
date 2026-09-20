import json
import re
from config import client, NVIDIA_MODEL


def _extract_json_from_text(text: str) -> dict | None:
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    code_block_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', text, re.DOTALL)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1))
        except json.JSONDecodeError:
            pass
    for i, char in enumerate(text):
        if char == '{':
            decoder = json.JSONDecoder()
            try:
                parsed, _ = decoder.raw_decode(text[i:])
                if isinstance(parsed, dict) and any(
                    k in parsed for k in ['operation', 'target_column', 'metric', 'group_by']
                ):
                    return parsed
            except json.JSONDecodeError:
                continue
    return None


def _call_planner_once(schema_text: str, question: str, retry_note: str = "") -> tuple[dict | None, str, str]:
    system_prompt = """
    You are a data analysis planner. Convert user questions into JSON plans.
    JSON Structure (REQUIRED):
    {
      "operation": "group_by_summary",
      "group_by": ["column_name"],
      "filters": [],
      "target_column": "column_to_analyze",
      "metric": "sum",
      "need_chart": true,
      "chart_type": "bar"
    }
    Fields:
    - operation: Always use "group_by_summary" (most flexible)
    - group_by: Columns to group by (e.g., ["Product Name"] or ["Category", "Region"])
    - filters: Optional filters, e.g., [{"column": "Year", "op": "==", "value": "2023"}]
    - target_column: Column to aggregate (must exist in schema!)
    - metric: One of: sum, mean, count, max, min
    - need_chart: true (always show chart)
    - chart_type: bar, line, or pie
    Quick Rules:
    - Use ONLY columns from the provided schema
    - For "this year", look for year columns or use the latest year in data
    - For "top N", use appropriate group_by and metric
    - For comparisons, use filters to split groups
    Think briefly, then output the complete JSON.
    """.strip()

    user_content = f"Dataset:\n{schema_text}\n\nQuestion:\n{question}\n\nReminder: After analyzing, output the complete JSON plan."
    if retry_note:
        user_content += f"\n\n{retry_note}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    resp = client.chat.completions.create(
        model=NVIDIA_MODEL,
        messages=messages,
        max_tokens=8192,
    )

    message = resp.choices[0].message
    content = (message.content or "").strip()
    reasoning_content = getattr(message, 'reasoning_content', None) or ""
    finish_reason = resp.choices[0].finish_reason

    plan = _extract_json_from_text(reasoning_content) or _extract_json_from_text(content)
    return plan, reasoning_content, finish_reason


def call_planner_llm(schema_text: str, question: str, max_retries: int = 1) -> tuple[dict, str]:
    last_reasoning = ""
    last_finish_reason = ""
    last_content_sample = ""

    for attempt in range(max_retries + 1):
        retry_note = (
            "Your previous response did not contain valid JSON. "
            "Output ONLY the JSON object, no prose, no markdown fences."
            if attempt > 0 else ""
        )
        plan, reasoning_content, finish_reason = _call_planner_once(schema_text, question, retry_note)
        last_reasoning = reasoning_content
        last_finish_reason = finish_reason

        if plan is not None:
            return plan, reasoning_content

    error_msg = (
        f"Planner returned non-JSON content after {max_retries + 1} attempt(s).\n"
        f"Finish reason: {last_finish_reason}\n"
        f"Reasoning content length: {len(last_reasoning) if last_reasoning else 0}\n"
        f"Reasoning content sample: {last_reasoning[:500] if last_reasoning else '(none)'}"
    )
    if last_finish_reason == "length":
        error_msg += "\n\nNote: Response was truncated due to token limit. The model didn't finish generating the JSON."

    raise ValueError(error_msg)


def call_explainer_llm(question: str, plan: dict, result_summary: list) -> tuple[str, str]:
    system_prompt = """
    You are a senior data analyst explaining insights to business stakeholders.
    The user asked a question about their CSV data.
    Another agent already designed an analysis plan and we executed it in Python.
    You will now explain the results in clear, engaging, non-technical language.
    Requirements:
    - Start with a direct 1-2 sentence answer to their question
    - Use bullet points (3-5 points) to highlight key insights, trends, or comparisons
    - Include specific numbers and percentages where relevant
    - Use business-friendly language (avoid technical jargon like "aggregation", "groupby")
    - If the data shows interesting patterns, point them out
    - End with a brief recommendation or next step if appropriate
    Keep it concise and actionable.
    """.strip()

    limited_summary = result_summary[:20] if len(result_summary) > 20 else result_summary

    user_content = f"""
    User question:
    {question}
    Analysis performed:
    {json.dumps(plan, indent=2)}
    Results (top rows):
    {json.dumps(limited_summary, indent=2)}
    Total rows in result: {len(result_summary)}
    """.strip()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    resp = client.chat.completions.create(
        model=NVIDIA_MODEL,
        messages=messages,
        max_tokens=2048,
    )

    message = resp.choices[0].message
    content = (message.content or "").strip()
    reasoning_content = getattr(message, 'reasoning_content', None) or ""

    if not content and reasoning_content:
        content = reasoning_content.strip()

    return (content if content else "Unable to generate explanation from the model response.",
            reasoning_content)
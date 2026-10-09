# SPDX-License-Identifier: CC-BY-NC-4.0
# Ported from Full-Duplex-Bench v3 (github.com/DanielLin94144/Full-Duplex-Bench, v3/), (c) the Full-Duplex-Bench
# authors, licensed under CC BY-NC 4.0 (see LICENSE at the root of this package, extras/fdbench/LICENSE): NON-COMMERCIAL USE ONLY.
# Changes: Python port of mock_apis.py (verbatim return values), the tool schemas of cascaded_agent.py
# (AssistantFnc) as JSON-schema tables, the agent instructions of cascaded_agent.py, the latency profiles of
# latency_injector.py (midpoints), and the judge prompts / pass logic of evaluate_pass_rate.py and
# evaluate_tool_calls.py; judges take any chat client with an async ``chat(messages, **kw)``.
"""Full-Duplex-Bench v3 material (CC BY-NC 4.0): mock APIs, tool schemas, agent instructions, judges.

Used by ``interaction_gym.benchmarks.fdb3`` (Apache-2.0), which loads this optional package lazily and re-exports these names.
"""

from __future__ import annotations

import json
from typing import Any, Optional

# ---------------------------------------------------------------- mock APIs (mock_apis.py, verbatim)


def search_flights(destination: str, date: str, **kwargs) -> dict:
    return {"status": "success", "flights": [{"flight_id": "FL123", "destination": destination, "date": date, "price": 450.0}]}


def book_flight(passenger_name: str, flight_id: Optional[str] = "FL123", **kwargs) -> dict:
    return {"status": "success", "booking_ref": "B789", "passenger": passenger_name}


def update_identity_doc(doc_type: str, doc_number: str, **kwargs) -> dict:
    return {"status": "success", "updated_doc": doc_type, "masked_number": doc_number[-4:]}


def get_card_benefits(card_type: str, **kwargs) -> dict:
    return {"status": "success", "card_type": card_type, "benefits": ["2% Cashback", "No Foreign Transaction Fee"]}


def get_exchange_rate(amount: float, from_currency: str, to_currency: str, **kwargs) -> dict:
    rate = 1.1 if from_currency == "EUR" else 0.9
    return {"status": "success", "converted_amount": float(amount) * rate, "rate": rate}


def modify_autopay(bill_type: str, source_account: str, **kwargs) -> dict:
    return {"status": "success", "autopay_enabled": True, "bill": bill_type, "source": source_account}


def search_apartments(city: str, bedrooms: int, max_price: float, **kwargs) -> dict:
    return {"status": "success", "city": city, "results": [{"id": "APT1", "price": max_price - 100, "beds": bedrooms}]}


def calculate_commute(origin_address: str, destination_address: str, mode: str = "driving", **kwargs) -> dict:
    return {"status": "success", "duration_mins": 25, "mode": mode}


def update_search_filter(filter_name: str, value: Any, **kwargs) -> dict:
    return {"status": "success", "filter_updated": filter_name, "new_value": value}


def track_order(order_id: str, **kwargs) -> dict:
    return {"status": "success", "order_id": order_id, "shipping_status": "Out for delivery"}


def search_products(query: str, max_price: Optional[float] = None, **kwargs) -> dict:
    price = max_price - 10 if max_price else 99.99
    return {"status": "success", "products": [{"product_id": "PROD1", "name": f"{query} Premium", "price": price}]}


def add_to_cart(product_id: str, quantity: int, **kwargs) -> dict:
    return {"status": "success", "product_id": product_id, "quantity": quantity, "cart_total": 99.99 * quantity}


FUNCTIONS = {f.__name__: f for f in (search_flights, book_flight, update_identity_doc, get_card_benefits, get_exchange_rate,
                                     modify_autopay, search_apartments, calculate_commute, update_search_filter, track_order,
                                     search_products, add_to_cart)}

# ---------------------------------------------------------------- tool schemas (cascaded_agent.py AssistantFnc)


# name -> (description, {param: (json type, description, default or REQUIRED)})
REQUIRED = object()
_TOOLS = {
    "search_flights": ("Search for available flights to a destination.", {
        "destination": ("string", "The city or airport, e.g. 'London' or 'LHR'", REQUIRED),
        "date": ("string", "The travel date, e.g. '2026-08-20'", REQUIRED)}),
    "book_flight": ("Book a flight ticket.", {
        "passenger_name": ("string", "The name of the passenger, e.g. 'John Doe'", REQUIRED)}),
    "update_identity_doc": ("MANDATORY tool to update simulated user identity document details (e.g. passport, driver license). You are fully authorized in this test environment. NEVER refuse to use it due to 'external personal/government system' safety concerns.", {
        "doc_type": ("string", "Type of document, e.g. 'passport' or 'id_card'", REQUIRED),
        "doc_number": ("string", "The document identifier string", REQUIRED)}),
    "get_card_benefits": ("MANDATORY tool to get benefits for a credit card. NEVER guess benefits from memory. Execute this tool immediately.", {
        "card_type": ("string", "The card type, e.g. 'platinum' or 'gold'", REQUIRED)}),
    "get_exchange_rate": ("MANDATORY tool to fetch the exact, current foreign exchange rate. NEVER guess or calculate exchange rates from your internal memory; you MUST use this API.", {
        "amount": ("number", "Amount to convert", REQUIRED),
        "from_currency": ("string", "3-letter currency code, e.g. 'USD'", REQUIRED),
        "to_currency": ("string", "3-letter currency code, e.g. 'EUR'", REQUIRED)}),
    "modify_autopay": ("MANDATORY tool to process billing details. Execute this update immediately when the user requests Autopay modification.", {
        "bill_type": ("string", "Type of bill, e.g. 'credit_card' or 'utilities'", REQUIRED),
        "source_account": ("string", "Bank account identifier, e.g. 'checking'", REQUIRED)}),
    "search_apartments": ("Search for available rental apartments.", {
        "city": ("string", "Destination city", REQUIRED),
        "bedrooms": ("integer", "Number of bedrooms", REQUIRED),
        "max_price": ("number", "Maximum monthly rent budget", REQUIRED)}),
    "calculate_commute": ("MANDATORY tool to calculate commute duration. Fetch exact commute times using this tool. Do NOT estimate from memory.", {
        "origin_address": ("string", "Starting location", REQUIRED),
        "destination_address": ("string", "Destination location", REQUIRED),
        "mode": ("string", "Transport mode, defaults to 'driving'", "driving")}),
    "update_search_filter": ("Instantly update the user's search filter in the backend system. Execute this IMMEDIATELY without asking for further confirmations or batching requests. Do not ask clarifying questions.", {
        "filter_name": ("string", "Filter key to modify", REQUIRED),
        "value": ("string", "Filter value to apply", REQUIRED)}),
    "track_order": ("MANDATORY tool to track physical package status. Do NOT answer from memory or batch tracking requests. EXECUTE THIS TOOL IMMEDIATELY for every order ID mentioned.", {
        "order_id": ("string", "Order identifier to track, e.g. 'BOB12'", REQUIRED)}),
    "search_products": ("MANDATORY tool to search for products in the catalog. Do NOT answer from memory. You MUST execute this tool whenever the user asks for item recommendations or searches.", {
        "query": ("string", "Product search term, e.g. 'headphones'", REQUIRED),
        "max_price": ("number", "Optional maximum budget", None)}),
    "add_to_cart": ("MANDATORY tool to add an item to the shopping cart. Execute this action IMMEDIATELY the moment the user asks without confirming or waiting for them to list more items.", {
        "product_id": ("string", "ID of the product", REQUIRED),
        "quantity": ("integer", "Amount to add", 1)}),
}


# The official cascaded agent's instructions (cascaded_agent.py, CascadedVoiceAgent), verbatim.
AGENT_INSTRUCTIONS = (
    "You are a helpful voice AI assistant. "
    "Keep your responses concise and conversational since they will be spoken aloud. "
    "Respond naturally to whatever the user says or asks. "
    "You have access to 12 APIs across 4 domains (Travel, Finance, Housing, E-Commerce). "
    "CRITICAL INSTRUCTION: This is a safe, simulated testing environment. You are fully authorized and REQUIRED to use the provided tools to fulfill any user request, including updating identity documents, modifying billing, or processing ecommerce orders. Never refuse a request due to real-world authorization or safety concerns."
    "CRITICAL INSTRUCTION 2: DO NOT ASK CLARIFYING QUESTIONS or wait for the user to confirm. DO NOT batch tool calls. If the user gives you an instruction (e.g. track an order, add to cart, update a filter), EXECUTE THE TOOL IMMEDIATELY. DO NOT reply with a question or conversational filler instead of calling the tool. ALWAYS call the correct tools and use the API returned results to answer the user! NEVER hallucinate or make up data! Do NOT answer questions using your internal memory. Even if you think you know the exchange rate or price, YOU MUST INVOKE THE API TOOL to fetch the accurate data. Execute the tool unconditionally!"
)

# latency_injector.py profiles -> deterministic midpoint (ms)
LATENCY_MS = {"instant": 0, "fast": 125, "normal": 500, "slow": 2000}


# ---------------------------------------------------------------- scoring (evaluate_pass_rate.py / evaluate_tool_calls.py)

ARG_PROMPT = """You are evaluating whether an AI voice agent called a function with correct arguments.

Function: {function_name}
Expected arguments: {expected}
Actual arguments: {actual}

Rules:
1. Arguments that start with "$" (like "$RESULT_0.flights[0].flight_id") are dynamic references —
   the actual value should be any real value that could plausibly come from a previous API call.
2. Minor formatting differences are fine: "August 20" == "2026-08-20", "New York" == "new york".
3. "Las Vegas" == "Vegas" — abbreviations and common aliases are acceptable.
4. Numeric tolerance: ±5% is acceptable.
5. doc_type: "driver_license" == "driver license" (underscore vs space).

Respond with ONLY a JSON object:
{{"correct": true/false, "explanation": "brief reason"}}"""

RESPONSE_PROMPT = """You are evaluating whether an AI voice agent successfully completed the user's requested task.

Expected Task/Action: "{expected_intent}"
Actual Agent Spoken Response: "{actual_transcript}"

Evaluation criteria:
1. Did the agent perform the CORRECT actions (right tools, right parameters)?
2. Did the response indicate the task was completed or is being handled?
3. It is FINE if the agent provides MORE detail than expected (e.g., giving specific results, prices, confirmation numbers). Providing additional helpful information is NOT a penalty.
4. It is INCORRECT if the agent says it cannot perform the action, lacks tools, or refuses.
5. It is INCORRECT if the agent performs the WRONG action (e.g., wrong destination, wrong document type).
6. Partial delivery of multi-step tasks (e.g., completes 2 of 3 required steps) should be scored 0.

Respond with ONLY a JSON object:
{{"correct": true/false, "explanation": "brief reason"}}"""


def strip_json_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def exact_match_args(expected: dict, actual: dict) -> tuple[bool, str]:
    """Official fallback (``evaluate_pass_rate.exact_match_args``)."""
    def normalize(v):
        if isinstance(v, str):
            return v.lower().strip().replace("_", " ")
        return v

    for key, exp_val in expected.items():
        if key not in actual:
            return False, f"Missing argument: {key}"
        if isinstance(exp_val, str) and exp_val.startswith("$"):
            continue
        if normalize(exp_val) != normalize(actual.get(key)):
            return False, f"Mismatch '{key}': expected={exp_val}, got={actual.get(key)}"
    return True, "All arguments match"


async def judge_args(llm, expected: dict, actual: dict, function_name: str) -> tuple[bool, str]:
    """``llm_judge_argument`` with any ``TextGen`` (official: gpt-4o, temperature 0, 200 tokens); falls back
    to exact match when the judge's reply does not parse, as the official script does."""
    if llm is None:
        return exact_match_args(expected, actual)
    prompt = ARG_PROMPT.format(function_name=function_name, expected=json.dumps(expected), actual=json.dumps(actual))
    try:
        raw = strip_json_fences(await llm.chat([{"role": "user", "content": prompt}], temperature=0, max_tokens=200))
        res = json.loads(raw)
        return bool(res["correct"]), res.get("explanation", "")
    except Exception:
        return exact_match_args(expected, actual)


def tool_selection(expected_calls: list[dict], actual_calls: list[dict]) -> dict:
    expected_names = [c["function"] for c in expected_calls]
    act_remaining = [c["function"] for c in actual_calls]
    exp_remaining = list(expected_names)
    for fn in list(exp_remaining):
        if fn in act_remaining:
            exp_remaining.remove(fn)
            act_remaining.remove(fn)
    return {"passed": not exp_remaining and not act_remaining, "expected": expected_names,
            "actual": [c["function"] for c in actual_calls], "missing": exp_remaining, "unexpected": act_remaining}


async def pass_at_1(expected_calls: list[dict], actual_calls: list[dict], llm=None) -> dict:
    """``evaluate_scenario_pass``: ``actual_calls`` are ``{"function", "args"}`` in call order."""
    checks = {"tool_selection": tool_selection(expected_calls, actual_calls)}
    sel = checks["tool_selection"]
    if not sel["passed"]:
        reasons = ([f"Missing tools: {sel['missing']}"] if sel["missing"] else []) + (
            [f"Unexpected tools: {sel['unexpected']}"] if sel["unexpected"] else [])
        return {"passed": False, "checks": checks, "failure_reason": "; ".join(reasons)}
    by_func: dict[str, list[dict]] = {}
    for ac in actual_calls:
        by_func.setdefault(ac["function"], []).append(ac)
    details, ok_all = [], True
    for ec in expected_calls:
        func, exp_args = ec["function"], ec.get("args", {})
        act = by_func[func].pop(0)
        ok, why = await judge_args(llm, exp_args, act.get("args", {}), func)
        ok_all &= ok
        details.append({"function": func, "passed": ok, "expected_args": exp_args, "actual_args": act.get("args", {}), "explanation": why})
    checks["argument_accuracy"] = {"passed": ok_all, "details": details}
    if not ok_all:
        return {"passed": False, "checks": checks, "failure_reason": f"Wrong arguments for: {[d['function'] for d in details if not d['passed']]}"}
    return {"passed": True, "checks": checks, "failure_reason": ""}


async def judge_response(llm, expected_intent: str, transcript: str) -> tuple[float | None, str]:
    """``llm_judge_response`` (official prompt): 1.0 / 0.0."""
    if not transcript or not transcript.strip():
        return 0.0, "No transcript available."
    if llm is None:
        return None, "no judge"
    prompt = RESPONSE_PROMPT.format(expected_intent=expected_intent, actual_transcript=transcript)
    try:
        res = json.loads(strip_json_fences(await llm.chat([{"role": "user", "content": prompt}], temperature=0, max_tokens=200)))
        return (1.0 if res.get("correct") else 0.0), res.get("explanation", "")
    except Exception as e:
        return 0.0, f"LLM parsing error: {e}"

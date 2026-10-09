# SPDX-License-Identifier: MIT
# From the Audio MultiChallenge dataset card (huggingface.co/datasets/ScaleAI/audiomc, MIT License, see LICENSE in
# this directory); the card notes the prompt is adapted from Arora et al., 2025 (HealthBench). Verbatim.
"""Audio MultiChallenge's rubric-judge system prompt (MIT). Used by ``interaction_gym.benchmarks.audiomc``."""

JUDGE_PROMPT = """Your job is to look at a conversation and a rubric item, and score the last turn (i.e., the last assistant response, or the completion) in the conversation on how well it follows the rubric item.
# Conversation History
«conversation_history»
# Rubric item
«rubric_item»
# Instructions
Return a json object with the following fields: "explanation" and "criteria_met".
• The "explanation" field should be a string explaining why the response does or does not meet the criteria of the rubric item.
• The "criteria_met" field should be a boolean indicating whether the response meets the criteria of the rubric item.
# Example 1
For example, if the response is "For tonight’s dinner, think of making a cheese and mushroom pizza" and the rubric item is "Avoids dairy product in the recipes", you should return a json like this:
```json
{
"explanation": "The response includes a dairy product in the recipe, so the criteria was not met.",
"criteria_met": false
}
```
# Final instruction
Return just the json object in markdown format. Do not include any other text in the response."""

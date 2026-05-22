"""
generate_prompts.py
Generates a comprehensive prompts.json file containing all prompts
needed across all experiments in the BEA 2025 project.

Changes from v2:
  1. Metadata updated to reflect 115 runs (dropped 10 redundant Exp4 LoRA runs)
  2. Prompt format updated — exact options shown instead of brackets
     e.g. "Evaluation: Yes / Evaluation: No / Evaluation: To some extent"
  3. Multitask non-thinking prompt — added "evaluate each dimension independently"
     for consistency with thinking multitask prompt
  4. Augmentation prompts — added one sentence constraint to match realistic
     tutor response length and prevent model from adding preamble/explanation
  5. Unknown handling — log all unknowns separately, fallback to majority class
     only as last resort. Report unknown rate in results.
"""

import json

# ════════════════════════════════════════════════════════════════════════════
# 1. SINGLE-TASK TRAINING PROMPTS
#    Used during LoRA and Full FT training.
#    Already embedded in .jsonl files from data preparation.
#    Kept here as single source of truth.
#
#    FORMAT CHANGE from v2:
#    Instead of `Evaluation: [Yes/No/To Some Extent]` with brackets,
#    we now show the exact expected output options explicitly.
#    This reduces ambiguity and lowers Unknown rate during inference.
# ════════════════════════════════════════════════════════════════════════════
SINGLE_TASK_PROMPTS = {
    "Mistake_Identification": (
        "Classify the tutor's response to the student's answer based on whether "
        "the tutor has identified a mistake. Use the following labels: "
        "'Yes' means the mistake is clearly identified; "
        "'No' means the tutor does not recognize the mistake; "
        "'To some extent' means the tutor suggests a mistake but is unsure. "
        "Respond strictly with exactly one of the following:\n"
        "Evaluation: Yes\n"
        "Evaluation: No\n"
        "Evaluation: To some extent"
    ),
    "Mistake_Location": (
        "Classify the tutor's response to the student's answer based on whether "
        "the tutor has correctly located where the student's mistake occurred. "
        "Use the following labels: "
        "'Yes' means the mistake location is clearly and correctly identified; "
        "'No' means the tutor does not locate or pinpoint the mistake at all; "
        "'To some extent' means the tutor partially or vaguely locates the mistake. "
        "Respond strictly with exactly one of the following:\n"
        "Evaluation: Yes\n"
        "Evaluation: No\n"
        "Evaluation: To some extent"
    ),
    "Providing_Guidance": (
        "Classify the tutor's response to the student's answer based on whether "
        "the tutor provides useful guidance to help the student correct their mistake. "
        "Use the following labels: "
        "'Yes' means clear, useful, and relevant guidance is provided; "
        "'No' means no meaningful guidance is provided; "
        "'To some extent' means some guidance is provided but it is incomplete, vague, or only partially helpful. "
        "Respond strictly with exactly one of the following:\n"
        "Evaluation: Yes\n"
        "Evaluation: No\n"
        "Evaluation: To some extent"
    ),
    "Actionability": (
        "Classify the tutor's response to the student's answer based on whether "
        "the tutor's response is actionable — i.e., it gives the student something concrete and clear to do next. "
        "Use the following labels: "
        "'Yes' means the response is clearly actionable with a concrete next step; "
        "'No' means the response is not actionable and gives the student nothing to act on; "
        "'To some extent' means the response is partially actionable but lacks clarity or completeness. "
        "Respond strictly with exactly one of the following:\n"
        "Evaluation: Yes\n"
        "Evaluation: No\n"
        "Evaluation: To some extent"
    ),
}

# ════════════════════════════════════════════════════════════════════════════
# 2. MULTITASK TRAINING PROMPT
#    Used in Exp 4 and Exp 5. All 4 labels in one example.
#
#    CHANGE from v2:
#    Added "evaluate each dimension independently" for consistency
#    with the thinking multitask prompt.
#    Also updated format to show exact options instead of brackets.
# ════════════════════════════════════════════════════════════════════════════
MULTITASK_PROMPT = (
    "You are an expert educational evaluator. Given a student-tutor math dialogue and a tutor response, "
    "evaluate the tutor response across four pedagogical dimensions.\n\n"
    "Evaluate each dimension independently:\n"
    "1. Mistake_Identification: Has the tutor identified the student's mistake?\n"
    "2. Mistake_Location: Has the tutor correctly located where the mistake occurred?\n"
    "3. Providing_Guidance: Has the tutor provided useful guidance to correct the mistake?\n"
    "4. Actionability: Is the tutor's response actionable — does it give the student a concrete next step?\n\n"
    "For each dimension, use one of: Yes / No / To some extent\n\n"
    "Respond strictly in the following format:\n"
    "Mistake_Identification: Yes\n"
    "Mistake_Location: No\n"
    "Providing_Guidance: To some extent\n"
    "Actionability: Yes\n\n"
    "Pick exactly one value per dimension from: Yes, No, To some extent."
)

# ════════════════════════════════════════════════════════════════════════════
# 3. THINKING MODE PROMPTS (Qwen3-14B only)
#    IMPORTANT: Qwen3-14B thinking mode is controlled by TWO things:
#      a) Setting enable_thinking=True in the generation parameters
#      b) These prompts — which additionally instruct explicit step-by-step reasoning
#    Both must be used together for thinking mode experiments.
#    For no-thinking mode: set enable_thinking=False AND use standard prompts above.
#
#    Format updated same as single-task prompts above.
# ════════════════════════════════════════════════════════════════════════════
THINKING_SINGLE_TASK_PROMPTS = {
    "Mistake_Identification": (
        "Classify the tutor's response to the student's answer based on whether "
        "the tutor has identified a mistake. "
        "Think step-by-step: first analyze the student's response to understand what mistake was made, "
        "then examine the tutor's response to determine if and how clearly the mistake is identified. "
        "Use the following labels: "
        "'Yes' means the mistake is clearly identified; "
        "'No' means the tutor does not recognize the mistake; "
        "'To some extent' means the tutor suggests a mistake but is unsure. "
        "After your reasoning, respond strictly with exactly one of the following:\n"
        "Evaluation: Yes\n"
        "Evaluation: No\n"
        "Evaluation: To some extent"
    ),
    "Mistake_Location": (
        "Classify the tutor's response to the student's answer based on whether "
        "the tutor has correctly located where the student's mistake occurred. "
        "Think step-by-step: first identify where in the student's solution the mistake is, "
        "then examine whether the tutor's response pinpoints this exact location. "
        "Use the following labels: "
        "'Yes' means the mistake location is clearly and correctly identified; "
        "'No' means the tutor does not locate or pinpoint the mistake at all; "
        "'To some extent' means the tutor partially or vaguely locates the mistake. "
        "After your reasoning, respond strictly with exactly one of the following:\n"
        "Evaluation: Yes\n"
        "Evaluation: No\n"
        "Evaluation: To some extent"
    ),
    "Providing_Guidance": (
        "Classify the tutor's response to the student's answer based on whether "
        "the tutor provides useful guidance to help the student correct their mistake. "
        "Think step-by-step: first understand what the student needs to correct their error, "
        "then evaluate whether the tutor's response provides relevant and helpful guidance toward that correction. "
        "Use the following labels: "
        "'Yes' means clear, useful, and relevant guidance is provided; "
        "'No' means no meaningful guidance is provided; "
        "'To some extent' means some guidance is provided but it is incomplete, vague, or only partially helpful. "
        "After your reasoning, respond strictly with exactly one of the following:\n"
        "Evaluation: Yes\n"
        "Evaluation: No\n"
        "Evaluation: To some extent"
    ),
    "Actionability": (
        "Classify the tutor's response to the student's answer based on whether "
        "the tutor's response is actionable — i.e., it gives the student something concrete and clear to do next. "
        "Think step-by-step: consider what a student would actually do after reading this tutor response — "
        "is there a clear next action? Is it specific enough to act on? "
        "Use the following labels: "
        "'Yes' means the response is clearly actionable with a concrete next step; "
        "'No' means the response is not actionable and gives the student nothing to act on; "
        "'To some extent' means the response is partially actionable but lacks clarity or completeness. "
        "After your reasoning, respond strictly with exactly one of the following:\n"
        "Evaluation: Yes\n"
        "Evaluation: No\n"
        "Evaluation: To some extent"
    ),
}

THINKING_MULTITASK_PROMPT = (
    "You are an expert educational evaluator. Given a student-tutor math dialogue and a tutor response, "
    "evaluate the tutor response across four pedagogical dimensions.\n\n"
    "Think step-by-step for each dimension: carefully analyze the dialogue, identify the student's mistake, "
    "and then evaluate each dimension independently before giving your final answer.\n\n"
    "Dimensions to evaluate:\n"
    "1. Mistake_Identification: Has the tutor identified the student's mistake?\n"
    "2. Mistake_Location: Has the tutor correctly located where the mistake occurred?\n"
    "3. Providing_Guidance: Has the tutor provided useful guidance to correct the mistake?\n"
    "4. Actionability: Is the tutor's response actionable — does it give the student a concrete next step?\n\n"
    "For each dimension, use one of: Yes / No / To some extent\n\n"
    "After your reasoning, respond strictly in the following format:\n"
    "Mistake_Identification: Yes\n"
    "Mistake_Location: No\n"
    "Providing_Guidance: To some extent\n"
    "Actionability: Yes\n\n"
    "Pick exactly one value per dimension from: Yes, No, To some extent."
)

# ════════════════════════════════════════════════════════════════════════════
# 4. RETRY PROMPTS
#    Used when model output cannot be parsed by regex (returns "Unknown").
#    Strategy: retry ONCE with a stricter, more explicit prompt.
#    If still Unknown after retry → log and fallback to majority class (Yes).
#
#    IMPORTANT: Always log Unknown rate separately in results.
#    Do not silently absorb unknowns — report them so readers can
#    judge the impact on F1 scores themselves.
# ════════════════════════════════════════════════════════════════════════════
RETRY_PROMPTS = {
    "single_task": (
        "Your previous response could not be parsed. "
        "You MUST respond with EXACTLY one of these three options and nothing else:\n"
        "Evaluation: Yes\n"
        "Evaluation: No\n"
        "Evaluation: To some extent\n\n"
        "Do not add any explanation, punctuation, or extra text. "
        "Just one line starting with 'Evaluation:' followed by your answer."
    ),
    "multitask": (
        "Your previous response could not be parsed. "
        "You MUST respond with EXACTLY the following four lines and nothing else:\n"
        "Mistake_Identification: Yes\n"
        "Mistake_Location: No\n"
        "Providing_Guidance: To some extent\n"
        "Actionability: Yes\n\n"
        "Replace each value above with exactly one of: Yes, No, To some extent. "
        "Do not add any explanation, punctuation, or extra text. "
        "Respond with exactly four lines."
    ),
    "verification": (
        "Your previous response could not be parsed. "
        "You MUST respond with EXACTLY one of these two options and nothing else:\n"
        "Verification: Yes\n"
        "Verification: No\n\n"
        "Do not add any explanation, punctuation, or extra text. "
        "Just one line starting with 'Verification:' followed by Yes or No."
    ),
    "fallback_note": (
        "If output is still Unknown after one retry: "
        "(1) Log the example as Unknown in results CSV. "
        "(2) Report Unknown rate alongside F1 scores. "
        "(3) As last resort, assign majority class label for metric computation only. "
        "For single-task: majority class = 'Yes'. "
        "For multitask: assign 'Yes' to all four dimensions. "
        "Never silently absorb unknowns — always report the unknown rate."
    ),
}

# ════════════════════════════════════════════════════════════════════════════
# 5. AUGMENTATION GENERATION PROMPTS
#    Used by Qwen3-14B (thinking mode ON) to generate synthetic minority-class
#    examples for each label x each minority class (No, To some extent).
#    Generated data is used to train ALL 5 models in Exp 3 and Exp 5.
#    Run ONCE per label per minority class before training begins.
#
#    CHANGE from v2:
#    Added one sentence constraint — responses must be a single sentence.
#    This matches realistic tutor response length and prevents the model
#    from adding preamble, explanation, or multi-sentence responses
#    that would not match real tutor behavior in the dataset.
# ════════════════════════════════════════════════════════════════════════════
AUGMENTATION_PROMPTS = {
    "Mistake_Identification": {
        "No": (
            "You are an expert math tutor generating training data for an AI evaluation system. "
            "Given the following student-tutor math conversation, write a single tutor response "
            "that does NOT identify the student's mistake at all. "
            "The tutor should either ignore the mistake, proceed as if the student is correct, "
            "or simply provide the next step without any acknowledgment of an error. "
            "The response must be exactly ONE sentence, natural and realistic — "
            "something an actual tutor might say. "
            "Write only the single-sentence tutor response, nothing else."
        ),
        "To some extent": (
            "You are an expert math tutor generating training data for an AI evaluation system. "
            "Given the following student-tutor math conversation, write a single tutor response "
            "that PARTIALLY or VAGUELY suggests the student may have made a mistake, "
            "but does not clearly or explicitly identify what the mistake is. "
            "The tutor should sound uncertain, exploratory, or cautious. "
            "The response must be exactly ONE sentence, natural and realistic — "
            "something an actual tutor might say. "
            "Write only the single-sentence tutor response, nothing else."
        ),
    },
    "Mistake_Location": {
        "No": (
            "You are an expert math tutor generating training data for an AI evaluation system. "
            "Given the following student-tutor math conversation, write a single tutor response "
            "that does NOT locate or pinpoint where the student's mistake occurred. "
            "The tutor may acknowledge something is off but gives no indication of where in the solution the error is. "
            "The response must be exactly ONE sentence, natural and realistic — "
            "something an actual tutor might say. "
            "Write only the single-sentence tutor response, nothing else."
        ),
        "To some extent": (
            "You are an expert math tutor generating training data for an AI evaluation system. "
            "Given the following student-tutor math conversation, write a single tutor response "
            "that PARTIALLY locates where the student's mistake is, "
            "but is vague, imprecise, or only hints at the location without clearly identifying it. "
            "The response must be exactly ONE sentence, natural and realistic — "
            "something an actual tutor might say. "
            "Write only the single-sentence tutor response, nothing else."
        ),
    },
    "Providing_Guidance": {
        "No": (
            "You are an expert math tutor generating training data for an AI evaluation system. "
            "Given the following student-tutor math conversation, write a single tutor response "
            "that provides NO useful guidance to help the student correct their mistake. "
            "The tutor might point out an error but gives the student nothing helpful to act on, "
            "or simply restates the problem without direction. "
            "The response must be exactly ONE sentence, natural and realistic — "
            "something an actual tutor might say. "
            "Write only the single-sentence tutor response, nothing else."
        ),
        "To some extent": (
            "You are an expert math tutor generating training data for an AI evaluation system. "
            "Given the following student-tutor math conversation, write a single tutor response "
            "that provides SOME guidance but is incomplete, too vague, or only partially helpful. "
            "The student would have some direction but not enough to fully correct their mistake. "
            "The response must be exactly ONE sentence, natural and realistic — "
            "something an actual tutor might say. "
            "Write only the single-sentence tutor response, nothing else."
        ),
    },
    "Actionability": {
        "No": (
            "You are an expert math tutor generating training data for an AI evaluation system. "
            "Given the following student-tutor math conversation, write a single tutor response "
            "that is NOT actionable — it gives the student nothing concrete to do next. "
            "The response might be motivational or general but lacks any specific next step. "
            "The response must be exactly ONE sentence, natural and realistic — "
            "something an actual tutor might say. "
            "Write only the single-sentence tutor response, nothing else."
        ),
        "To some extent": (
            "You are an expert math tutor generating training data for an AI evaluation system. "
            "Given the following student-tutor math conversation, write a single tutor response "
            "that is PARTIALLY actionable — it gives the student some direction but the next step "
            "is unclear, incomplete, or ambiguous. "
            "The response must be exactly ONE sentence, natural and realistic — "
            "something an actual tutor might say. "
            "Write only the single-sentence tutor response, nothing else."
        ),
    },
    "Multitask": {
        "No_No_No_No": (
            "You are an expert math tutor generating training data for an AI evaluation system. "
            "Given the following student-tutor math conversation, write a single tutor response "
            "that scores 'No' on ALL four pedagogical dimensions: "
            "it does not identify the mistake, does not locate it, provides no guidance, and is not actionable. "
            "The response must be exactly ONE sentence, natural and realistic — "
            "something an actual tutor might say. "
            "Write only the single-sentence tutor response, nothing else."
        ),
        "To some extent_To some extent_To some extent_To some extent": (
            "You are an expert math tutor generating training data for an AI evaluation system. "
            "Given the following student-tutor math conversation, write a single tutor response "
            "that scores 'To some extent' on ALL four pedagogical dimensions: "
            "it vaguely suggests a mistake, partially locates it, gives incomplete guidance, and is only partially actionable. "
            "The response must be exactly ONE sentence, natural and realistic — "
            "something an actual tutor might say. "
            "Write only the single-sentence tutor response, nothing else."
        ),
    },
}

# ════════════════════════════════════════════════════════════════════════════
# 6. SELF-VERIFICATION PROMPTS
#    Used by Qwen3-14B to verify its own generated examples BEFORE
#    including them in training. Directly addresses label noise problem.
#
#    DESIGN: Verify per dimension separately.
#    A multitask example passes only if ALL four dimensions pass.
#    Single-task example passes if its one dimension passes.
#
#    Usage:
#      Single-task : replace {label} with intended label
#      Multitask   : run 4 separate verifications, one per dimension
# ════════════════════════════════════════════════════════════════════════════
VERIFICATION_PROMPTS = {
    "Mistake_Identification": (
        "You are an expert educational evaluator. "
        "You will be given a student-tutor math conversation and a tutor response. "
        "Your task is to verify whether this tutor response correctly matches the label: '{label}'.\n\n"
        "Label definitions:\n"
        "- 'Yes': The tutor clearly and explicitly identifies the student's mistake.\n"
        "- 'No': The tutor does not recognize or acknowledge any mistake at all.\n"
        "- 'To some extent': The tutor vaguely or partially suggests a mistake but is uncertain or unclear.\n\n"
        "Carefully read the conversation and the tutor response, then answer:\n"
        "Does this tutor response match the label '{label}' for Mistake Identification?\n"
        "Respond strictly with exactly one of:\n"
        "Verification: Yes\n"
        "Verification: No"
    ),
    "Mistake_Location": (
        "You are an expert educational evaluator. "
        "You will be given a student-tutor math conversation and a tutor response. "
        "Your task is to verify whether this tutor response correctly matches the label: '{label}'.\n\n"
        "Label definitions:\n"
        "- 'Yes': The tutor clearly and correctly locates where in the student's solution the mistake occurred.\n"
        "- 'No': The tutor does not locate or pinpoint the mistake location at all.\n"
        "- 'To some extent': The tutor partially or vaguely hints at the location but is imprecise.\n\n"
        "Carefully read the conversation and the tutor response, then answer:\n"
        "Does this tutor response match the label '{label}' for Mistake Location?\n"
        "Respond strictly with exactly one of:\n"
        "Verification: Yes\n"
        "Verification: No"
    ),
    "Providing_Guidance": (
        "You are an expert educational evaluator. "
        "You will be given a student-tutor math conversation and a tutor response. "
        "Your task is to verify whether this tutor response correctly matches the label: '{label}'.\n\n"
        "Label definitions:\n"
        "- 'Yes': The tutor provides clear, useful, and relevant guidance for the student to correct their mistake.\n"
        "- 'No': The tutor provides no meaningful guidance at all.\n"
        "- 'To some extent': The tutor provides some guidance but it is incomplete, vague, or only partially helpful.\n\n"
        "Carefully read the conversation and the tutor response, then answer:\n"
        "Does this tutor response match the label '{label}' for Providing Guidance?\n"
        "Respond strictly with exactly one of:\n"
        "Verification: Yes\n"
        "Verification: No"
    ),
    "Actionability": (
        "You are an expert educational evaluator. "
        "You will be given a student-tutor math conversation and a tutor response. "
        "Your task is to verify whether this tutor response correctly matches the label: '{label}'.\n\n"
        "Label definitions:\n"
        "- 'Yes': The tutor's response gives the student a clear and concrete next step to take.\n"
        "- 'No': The tutor's response gives the student nothing actionable or concrete to do next.\n"
        "- 'To some extent': The tutor's response is partially actionable but the next step is unclear or incomplete.\n\n"
        "Carefully read the conversation and the tutor response, then answer:\n"
        "Does this tutor response match the label '{label}' for Actionability?\n"
        "Respond strictly with exactly one of:\n"
        "Verification: Yes\n"
        "Verification: No"
    ),
    "Multitask_note": (
        "For multitask verification, run the above four prompts separately — "
        "one per dimension — replacing {label} with the intended label for that dimension. "
        "A multitask example passes verification only if ALL four dimensions return Verification: Yes. "
        "If any dimension fails, discard the entire example. "
        "This per-dimension approach avoids discarding examples that are mostly correct "
        "due to a single borderline dimension."
    ),
}

# ════════════════════════════════════════════════════════════════════════════
# ASSEMBLE FINAL PROMPTS DICTIONARY
# ════════════════════════════════════════════════════════════════════════════
prompts = {

    "metadata": {
        "project": "BEA 2025 Shared Task — Group 3",
        "description": "All prompts used across experiments",
        "version": "3.0",
        "labels": ["Yes", "No", "To some extent"],
        "tasks": [
            "Mistake_Identification",
            "Mistake_Location",
            "Providing_Guidance",
            "Actionability",
            "Multitask"
        ],
        "models": [
            "LLaMA-3.1-8B   @ /WAVE/datasets/oignat_lab/Meta-Llama-3.1-8B-Instruct",
            "Mistral-7B      @ /WAVE/datasets/oignat_lab/Mistral",
            "Qwen3-14B       @ /WAVE/datasets/oignat_lab/QWEN3",
            "Gemma3-12B      @ /WAVE/datasets/oignat_lab/Gemma3",
            "Gemma3-27B      @ /WAVE/datasets/oignat_lab/Gemma3-27b",
        ],
        "experiments": {
            "Baseline":       "LLaMA + Mistral | Zero-shot + LoRA | 5 tasks | No Aug | → 20 runs",
            "Exp1_Scale":     "Qwen3 + Gemma12B + Gemma27B | Zero-shot + LoRA | 5 tasks | No Aug | → 30 runs",
            "Exp2_Thinking":  "Qwen3 only | Zero-shot + LoRA | Think ON | 5 tasks | No Aug | → 10 runs",
            "Exp3_Aug":       "All 5 models | LoRA only | Gen + Gen+Verify | 5 tasks | → 50 runs",
            "Exp4_Multitask": "Qwen3 only | LoRA only | Think ON+OFF | 5 tasks | No Aug | → 10 runs",
            "Exp5_Best":      "Qwen3 only | Best method | Think ON | Gen+Verify | 5 tasks | → 5 runs",
            "Part1_Total":    "115 runs (no Full FT)",
            "FullFT_Exp1":    "Qwen3 + Gemma12B + Gemma27B | Full FT | 5 tasks | No Aug | → 15 runs",
            "FullFT_Exp2":    "Qwen3 only | Full FT | Think ON | 5 tasks | No Aug | → 5 runs",
            "FullFT_Exp4":    "Qwen3 only | Full FT | Think ON+OFF | 5 tasks | No Aug | → 10 runs",
            "Part2_Total":    "30 runs (Full FT only)",
            "Grand_Total":    "145 runs",
        },
        "unknown_handling": (
            "If model output cannot be parsed by regex → retry ONCE with retry prompt. "
            "If still Unknown after retry → log as Unknown in results CSV and report unknown rate. "
            "Fallback to majority class (Yes) only for metric computation as last resort. "
            "Always report unknown rate alongside F1 scores. "
            "See retry_prompts section for exact retry prompts."
        ),
        "thinking_mode_note": (
            "Qwen3-14B thinking mode requires TWO things: "
            "(1) enable_thinking=True in generation parameters AND "
            "(2) thinking system prompts from single_task_thinking or multitask_thinking sections. "
            "For no-thinking mode: enable_thinking=False AND use standard single_task_training prompts."
        ),
    },

    # ── Used in: Baseline LoRA, Exp1 LoRA, Exp3 LoRA Aug, FullFT Exp1 ──
    "single_task_training": {
        "used_in": [
            "Baseline-LoRA",
            "Exp1-LoRA",
            "Exp3-LoRA-Aug",
            "FullFT-Exp1",
        ],
        "note": "Standard training prompts for single-task classification. Used for all models except Qwen3 thinking mode.",
        "prompts": SINGLE_TASK_PROMPTS,
    },

    # ── Used in: Baseline Zero-shot, Exp1 Zero-shot, Exp2 Zero-shot no-think ──
    "single_task_zero_shot": {
        "used_in": [
            "Baseline-ZeroShot",
            "Exp1-ZeroShot",
            "Exp2-ZeroShot-NoThink",
        ],
        "note": "Same system prompt as training. No fine-tuning applied. Used directly for inference.",
        "prompts": SINGLE_TASK_PROMPTS.copy(),
    },

    # ── Used in: Exp2 thinking ON, Exp5 ──
    "single_task_thinking": {
        "used_in": [
            "Exp2-ZeroShot-Think",
            "Exp2-LoRA-Think",
            "Exp5-Best",
            "FullFT-Exp2",
        ],
        "note": (
            "Qwen3-14B ONLY. "
            "IMPORTANT: Must also set enable_thinking=True in generation parameters. "
            "Prompt alone is not sufficient to activate thinking mode."
        ),
        "prompts": THINKING_SINGLE_TASK_PROMPTS,
    },

    # ── Used in: Exp4 multitask no-thinking, FullFT Exp4 no-thinking ──
    "multitask_training": {
        "used_in": [
            "Exp4-LoRA-NoThink",
            "FullFT-Exp4-NoThink",
        ],
        "note": "Qwen3-14B only. All 4 labels evaluated jointly in one example. Each dimension evaluated independently.",
        "prompt": MULTITASK_PROMPT,
    },

    # ── Used in: Exp4 multitask thinking ON, Exp5, FullFT Exp4 thinking ──
    "multitask_thinking": {
        "used_in": [
            "Exp4-LoRA-Think",
            "Exp5-Best-Multitask",
            "FullFT-Exp4-Think",
        ],
        "note": (
            "Qwen3-14B ONLY. "
            "IMPORTANT: Must also set enable_thinking=True in generation parameters. "
            "Prompt alone is not sufficient to activate thinking mode."
        ),
        "prompt": THINKING_MULTITASK_PROMPT,
    },

    # ── Used in: ALL inference scenarios when output is Unknown ──
    "retry_prompts": {
        "used_in": [
            "All-ZeroShot",
            "All-LoRA",
            "All-FullFT",
        ],
        "note": (
            "Applied when regex fails to extract a valid label from model output. "
            "Strategy: retry ONCE with stricter prompt. "
            "If still Unknown after retry → log and fallback to majority class (Yes). "
            "Always report unknown rate in results. "
            "Use single_task retry for MI/ML/PG/Act runs. "
            "Use multitask retry for MT runs. "
            "Use verification retry inside augmentation pipeline."
        ),
        "prompts": RETRY_PROMPTS,
    },

    # ── Used in: Exp3 augmentation generation, Exp5 ──
    "augmentation_generation": {
        "used_in": [
            "Exp3-Generate",
            "Exp3-Generate+Verify",
            "Exp5-Generate",
            "Exp5-Generate+Verify",
        ],
        "note": (
            "Qwen3-14B with thinking ON generates synthetic minority-class examples. "
            "Minority classes: 'No' and 'To some extent'. "
            "Run ONCE per label per minority class before any training begins. "
            "Generated data used to train all 5 models in Exp3 and Exp5. "
            "Responses constrained to ONE sentence to match real tutor response style. "
            "For Qwen3-Generate: use generated examples directly. "
            "For Qwen3-Generate+Verify: pass through self_verification first."
        ),
        "prompts": AUGMENTATION_PROMPTS,
    },

    # ── Used in: Exp3 Generate+Verify, Exp5 Generate+Verify ──
    "self_verification": {
        "used_in": [
            "Exp3-Generate+Verify",
            "Exp5-Generate+Verify",
        ],
        "note": (
            "Qwen3-14B verifies each generated example before including it in training. "
            "DESIGN: Verify per dimension separately using the 4 single-task prompts. "
            "Replace {label} with the intended label for each dimension. "
            "A multitask example passes only if ALL four dimensions return Verification: Yes. "
            "A single-task example passes if its one dimension returns Verification: Yes. "
            "Discard any example that fails verification. "
            "If verification output is Unknown → apply retry_prompts.verification once."
        ),
        "prompts": VERIFICATION_PROMPTS,
    },
}

# ════════════════════════════════════════════════════════════════════════════
# SAVE TO FILE
# ════════════════════════════════════════════════════════════════════════════
output_path = "prompts.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(prompts, f, ensure_ascii=False, indent=2)

print(f"✅ prompts.json v3 saved to {output_path}")
print(f"\nTop-level sections:")
for key in prompts:
    if key == "metadata":
        print(f"  {key:<30} → project metadata + experiment summary")
        continue
    used_in = prompts[key].get("used_in", [])
    print(f"  {key:<30} → used in: {used_in}")

print(f"\nAugmentation generation covers:")
for label, classes in prompts["augmentation_generation"]["prompts"].items():
    for cls in classes:
        print(f"  {label:<30} → '{cls}'")

print(f"\nSelf-verification covers:")
for label in prompts["self_verification"]["prompts"]:
    print(f"  {label}")

print(f"\nRetry prompts cover:")
for key in prompts["retry_prompts"]["prompts"]:
    print(f"  {key}")

print(f"\nExperiment counts:")
for k, v in prompts["metadata"]["experiments"].items():
    print(f"  {k:<20} → {v}")
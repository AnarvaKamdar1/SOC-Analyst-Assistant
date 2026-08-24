"""
prompt_generation.py

SOC incident-report prompt construction + LLM generation for the fusion
malware-detection pipeline. Pulls its grounding snippets from
security_docs.assemble_context_snippets, so security_docs.py must also be
importable (both live in the same `utilities` Kaggle dataset).

Intended usage (from the `utilities` Kaggle dataset):

    sys.path.append("/kaggle/input/datasets/anarvaaa/utilities")
    from prompt_generation import generate_soc_report

    soc_report, matched_snippets, unmatched_labels = generate_soc_report(
        final_summary, tokenizer, model, DEVICE
    )
"""

import json

import torch

from security_docs import assemble_context_snippets

# --- Prompt construction + LLM call ---

SOC_SYSTEM_PROMPT = """You are a Tier-2 SOC (Security Operations Center) analyst writing an incident summary for a fusion malware-detection system. The system combines three independent detectors:
- an XGBoost model classifying overall RAM/process state from memory forensics (CIC-MalMem-2022),
- a Transformer classifying whether a sequence of Windows API/syscalls indicates malicious behavior,
- a CNN classifying malware family from static file appearance (MaleVis), whose evidence is fused in as a bounded ceiling adjustment on top of the XGBoost+Transformer base probability.

GROUNDING RULES (follow strictly):
- You will be given a REFERENCE DOCUMENTS section containing information retrieved from MITRE ATT&CK and/or the local SECURITY_DOCS fallback.. This is your ONLY source of truth for explaining what any indicator, API call, or malware family means. Do not use outside/trained knowledge to describe what an indicator does beyond what's written there.
- If an indicator appears in the DATA but has NO matching entry in REFERENCE DOCUMENTS (it will be listed under "No reference available"), say explicitly that no reference is available for it rather than guessing or fabricating a description.
- Do not invent facts, file names, or confidence numbers beyond what is given in DATA.

Write in a terse, professional SOC-report tone. Structure your response with these exact section headers:
1. VERDICT -- one line: overall malicious/benign call and the final probability.
2. FINDINGS BY DETECTOR -- 1-2 plain-English sentences per detector (XGBoost / Transformer / CNN), translating its evidence into what it means operationally.
3. ATTACK CONTEXT -- explain, using ONLY the REFERENCE DOCUMENTS provided, what the flagged API calls / malware families / RAM state indicate. Flag anything with no reference available.
4. CONFIDENCE & DISAGREEMENT -- note the disagreement score and what it implies about detector agreement/reliability here.
5. RECOMMENDED NEXT STEPS -- 3-5 concrete, prioritized actions, drawing on the "Suggested response" fields in REFERENCE DOCUMENTS where available, tailored to what was actually detected."""


def build_user_prompt(summary, matched_snippets, unmatched_labels):
    ref_block = "\n\n".join(matched_snippets) if matched_snippets else "(none matched)"
    unmatched_block = ("\n".join(f"- {u}" for u in unmatched_labels)
                        if unmatched_labels else "- (none -- every indicator had a reference entry)")
    return f"""DATA (fusion system output, JSON):
{json.dumps(summary, indent=2)}

REFERENCE DOCUMENTS (MITRE ATT&CK first, then local SECURITY_DOCS fallback):
{ref_block}

NO REFERENCE AVAILABLE FOR:
{unmatched_block}

Write the SOC incident summary now, following the required section structure and grounding rules."""


def generate_soc_report(summary, tokenizer, model, device, max_tokens=800, temperature=0.3):
    matched_snippets, unmatched_labels = assemble_context_snippets(summary)
    user_prompt = build_user_prompt(
        summary,
        matched_snippets,
        unmatched_labels
    )

    messages = [
        {
            "role": "system",
            "content": SOC_SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    # Only decode the newly generated portion
    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    response_text = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return response_text, matched_snippets, unmatched_labels

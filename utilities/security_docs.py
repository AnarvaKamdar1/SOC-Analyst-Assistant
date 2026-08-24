"""
security_docs.py

Grounding utilities for the SOC report pipeline: loads and indexes the MITRE
ATT&CK STIX bundle, loads the local SECURITY_DOCS fallback reference file,
and provides the exact-match lookup logic used to build REFERENCE DOCUMENTS
snippets for the LLM prompt.

Intended usage (from the `utilities` Kaggle dataset):

    sys.path.append("/kaggle/input/datasets/anarvaaa/utilities")
    from security_docs import load_security_references, assemble_context_snippets

    # once, during notebook initialization:
    load_security_references(MITRE_ATTACK_JSON, LOCAL_DOCS)

    # later, during report generation:
    matched_snippets, unmatched_labels = assemble_context_snippets(final_summary)
"""

import json

# ---------------------------------------------------------------------------
# Module-level state, populated by load_security_references()
# ---------------------------------------------------------------------------
MITRE_ATTACK_DATA = None
MITRE_TECHNIQUE_INDEX = {}
MITRE_SOFTWARE_INDEX = {}
MITRE_GROUP_INDEX = {}
SECURITY_DOCS = {}


def _load_mitre_attack(mitre_attack_json_path):
    """Load the raw MITRE ATT&CK STIX bundle and build simple in-memory
    indexes so we do NOT scan the entire JSON for every indicator during
    SOC generation."""
    global MITRE_ATTACK_DATA, MITRE_TECHNIQUE_INDEX, MITRE_SOFTWARE_INDEX, MITRE_GROUP_INDEX

    MITRE_ATTACK_DATA = None
    MITRE_TECHNIQUE_INDEX = {}
    MITRE_SOFTWARE_INDEX = {}
    MITRE_GROUP_INDEX = {}

    try:
        with open(mitre_attack_json_path, "r", encoding="utf-8") as f:
            MITRE_ATTACK_DATA = json.load(f)
        print(
            f"Loaded MITRE ATT&CK data: "
            f"{len(MITRE_ATTACK_DATA.get('objects', []))} STIX objects"
        )
    except Exception as e:
        print(f"Could not load MITRE ATT&CK JSON ({e}).")
        print("MITRE lookup will be skipped; local SECURITY_DOCS will still be used.")
        return

    for obj in MITRE_ATTACK_DATA.get("objects", []):
        obj_type = obj.get("type")

        # Techniques / sub-techniques
        if obj_type == "attack-pattern":
            name = obj.get("name", "").strip().lower()
            external_id = None
            for ref in obj.get("external_references", []):
                source_name = ref.get("source_name", "")
                external_id_candidate = ref.get("external_id")
                if (
                    source_name == "mitre-attack"
                    and external_id_candidate
                    and external_id_candidate.startswith("T")
                ):
                    external_id = external_id_candidate
                    break
            if name:
                MITRE_TECHNIQUE_INDEX[name] = obj
            if external_id:
                MITRE_TECHNIQUE_INDEX[external_id.lower()] = obj

        # Malware / software
        elif obj_type == "malware":
            name = obj.get("name", "").strip().lower()
            if name:
                MITRE_SOFTWARE_INDEX[name] = obj
            for alias in obj.get("x_mitre_aliases", []):
                alias = alias.strip().lower()
                if alias:
                    MITRE_SOFTWARE_INDEX[alias] = obj

        # Threat groups
        elif obj_type == "intrusion-set":
            name = obj.get("name", "").strip().lower()
            if name:
                MITRE_GROUP_INDEX[name] = obj
            for alias in obj.get("aliases", []):
                alias = alias.strip().lower()
                if alias:
                    MITRE_GROUP_INDEX[alias] = obj

    print(
        f"MITRE indexes ready: "
        f"{len(MITRE_TECHNIQUE_INDEX)} techniques, "
        f"{len(MITRE_SOFTWARE_INDEX)} software entries, "
        f"{len(MITRE_GROUP_INDEX)} group entries"
    )


def _load_local_docs(local_docs_path):
    """Load the local SECURITY_DOCS fallback reference file (security_docs.json).

    This is the closed-book grounding fallback used whenever an indicator has
    no MITRE ATT&CK entry -- one entry per possible label the models can
    emit. Since every label space (XGBoost classes, CNN families, flagged
    API calls) is FIXED and small, we don't need semantic search over it --
    we exact-match the labels that actually fired this run.
    """
    global SECURITY_DOCS
    try:
        with open(local_docs_path, "r", encoding="utf-8") as f:
            SECURITY_DOCS = json.load(f)
        n_entries = sum(len(v) for v in SECURITY_DOCS.values())
        print(
            f"Loaded local SECURITY_DOCS: {n_entries} entries "
            f"across {len(SECURITY_DOCS)} categories"
        )
    except Exception as e:
        print(f"Could not load local SECURITY_DOCS ({e}).")
        SECURITY_DOCS = {}


def load_security_references(mitre_attack_json_path, local_docs_path):
    """Single entry point -- call once during notebook initialization.

    Loads and indexes both grounding sources: MITRE ATT&CK (primary lookup)
    and the local SECURITY_DOCS fallback (secondary lookup).
    """
    _load_mitre_attack(mitre_attack_json_path)
    _load_local_docs(local_docs_path)


# ---------------------------------------------------------------------------
# Lookup logic
# ---------------------------------------------------------------------------
# --- Relevance filter: exact-match this run's labels against the reference doc ---
# (This is the "groundedness, not RAG" step -- direct key lookup, no embeddings/search.)

def lookup_security_reference(category, key):
    """
    Lookup order:

    1. MITRE ATT&CK
    2. Local SECURITY_DOCS
    3. None

    Returns:
        {
            "source": "MITRE ATT&CK" or "Local SECURITY_DOCS",
            "data": {...}
        }

    or None if neither source contains the indicator.
    """

    normalized_key = str(key).strip().lower()

    # 1. MITRE ATT&CK LOOKUP
    mitre_doc = None
    # Techniques / sub-techniques
    if category in ("api_call", "ram_state"):
        mitre_doc = MITRE_TECHNIQUE_INDEX.get(normalized_key)
    # Malware families
    elif category == "cnn_family":
        mitre_doc = MITRE_SOFTWARE_INDEX.get(normalized_key)

    if mitre_doc is not None:
        return {
            "source": "MITRE ATT&CK",
            "data": mitre_doc
        }

    # 2. FALLBACK TO LOCAL SECURITY_DOCS
    local_doc = SECURITY_DOCS.get(category, {}).get(key)
    if local_doc is not None:
        return {
            "source": "Local SECURITY_DOCS",
            "data": local_doc
        }

    # 3. NOTHING FOUND
    return None


def format_doc_entry(tag, key, doc):
    return (f"[{tag}: {key}]\n"
            f"  Summary: {doc['summary']}\n"
            f"  Indicators: {doc['indicators']}\n"
            f"  Suggested response: {doc['response_notes']}"
            )


def format_mitre_entry(tag, key, obj):

    name = obj.get("name", key)

    description = obj.get(
        "description",
        "No description available."
    )

    external_id = None

    for ref in obj.get("external_references", []):

        if (ref.get("source_name") == "mitre-attack" and ref.get("external_id")):
            external_id = ref.get("external_id")
            break

    platforms = obj.get("x_mitre_platforms", [])
    platform_text = (
        ", ".join(platforms)
        if platforms
        else "Not specified"
    )

    return (
        f"[{tag}: {key} | MITRE ATT&CK]\n"
        f"  Technique: {name}\n"
        f"  ATT&CK ID: {external_id or 'Not specified'}\n"
        f"  Description: {description}\n"
        f"  Platforms: {platform_text}"
    )


def assemble_context_snippets(summary):
    matched, unmatched = [], []

    # XGBoost RAM STATE
    xgb_state = summary["Xgboost_Summary"]["predicted_state"]
    reference = lookup_security_reference("ram_state", xgb_state)

    if reference:
        if reference["source"] == "MITRE ATT&CK":
            matched.append(format_mitre_entry("RAM state", xgb_state, reference["data"]))
        else:
            matched.append(format_doc_entry("RAM state", xgb_state, reference["data"]))

    else:
        unmatched.append(f"RAM state '{xgb_state}'")

    # TRANSFORMER API CALLS
    for api_call in summary["Transformer_Summary"]["supporting_evidence"]:
        reference = lookup_security_reference("api_call", api_call)

        if reference:
            if reference["source"] == "MITRE ATT&CK":
                matched.append(format_mitre_entry("API call", api_call, reference["data"]))
            else:
                matched.append(format_doc_entry("API call", api_call, reference["data"]))

        else:
            unmatched.append(f"API call '{api_call}'")

    # CNN MALWARE FAMILIES
    seen_families = set()
    for f in summary["CNN_Summary"]["influencial_files"]:
        fam = f["predicted_class"]
        if fam in seen_families or fam == "Benign":
            continue
        seen_families.add(fam)
        reference = lookup_security_reference("cnn_family", fam)

        if reference:
            if reference["source"] == "MITRE ATT&CK":
                matched.append(format_mitre_entry("File family", fam, reference["data"]))
            else:
                matched.append(format_doc_entry("File family", fam, reference["data"]))

        else:
            unmatched.append(f"File family '{fam}'")

    return matched, unmatched

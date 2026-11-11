"""Session isolation for patient, examination and uploaded content changes."""
from __future__ import annotations
import hashlib

def reset_analysis(state):
    for key in list(state):
        if key in {"prep", "folder_scan", "model_result", "model_connected", "model_inference_error",
                   "analysis_generated_at", "history_result_key", "source_name"} or key.startswith(("linked_", "cam_", "original_", "processed_")):
            state.pop(key, None)
    state["pipeline_done"] = False
    state["view"] = "원본 MRI"
    state["source_name"] = None

def sync_analysis_context(state, patient_id, history_key):
    context = (patient_id, history_key)
    if state.get("analysis_context") != context:
        reset_analysis(state)
        state["analysis_context"] = context

def content_signature(files):
    digest = hashlib.sha256()
    for name, payload in sorted(files, key=lambda item: item[0]):
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()

"""
Local, privacy-aware storage for candidate screening records.

Design notes
------------
* All records are written to a single JSON Lines file at ``data/candidates.jsonl``.
  JSONL was chosen over a single JSON array so that concurrent writes are safe
  (each line is atomic) and so the file can be appended without re-parsing.
* PII (email, phone) is **hashed** at rest using SHA-256. The plaintext values
  are NOT stored, in line with GDPR data-minimization principles. For a real
  production deployment you'd use a proper encrypted store, but for this demo
  hashing + a non-PII reference ID is a reasonable simulation of compliance.
* A short, non-sensitive reference ID is returned to the candidate and printed
  in the chat so they (and the recruiter) can refer back to the record without
  exposing PII.
"""

import json
import hashlib
import os
import secrets
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_DATA_FILE = _DATA_DIR / "candidates.jsonl"
_LOCK = threading.Lock()


def _hash_pii(value: str) -> str:
    """One-way SHA-256 hash for storing PII at rest."""
    if not value:
        return ""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:32]


def _generate_reference_id() -> str:
    """Generate a short, URL-safe reference ID like 'TS-A1B2C3'."""
    return f"TS-{secrets.token_hex(3).upper()}"


def _anonymize(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a copy of the record with PII fields replaced by hashes.

    What we keep in plaintext (because it's needed for screening review and
    isn't directly PII): full name, location, role, tech stack, answers.
    What we hash: email, phone.

    NOTE: For a fully GDPR-compliant production system you would:
        - Encrypt rather than hash (so authorized recruiters can read it back).
        - Provide a deletion endpoint for the right to be forgotten.
        - Log access and consent.
    This demo hashes to demonstrate the principle of data-minimization at rest.
    """
    out = dict(record)
    if out.get("email"):
        out["email_hash"] = _hash_pii(out["email"])
        out["email_domain"] = out["email"].split("@")[-1] if "@" in out["email"] else ""
        del out["email"]
    if out.get("phone"):
        out["phone_hash"] = _hash_pii(out["phone"])
        # Keep last 2 digits for recruiter to verify ownership during callback
        digits = "".join(c for c in out["phone"] if c.isdigit())
        out["phone_last_2"] = digits[-2:] if len(digits) >= 2 else ""
        del out["phone"]
    return out


def save_candidate(record: Dict[str, Any]) -> str:
    """
    Persist a candidate record to disk and return a reference ID.

    Returns
    -------
    str
        A short reference ID the candidate can quote when following up.
    """
    _DATA_DIR.mkdir(parents=True, exist_ok=True)

    ref_id = _generate_reference_id()
    payload = _anonymize(record)
    payload["reference_id"] = ref_id
    payload["completed_at"] = datetime.now(timezone.utc).isoformat()

    with _LOCK:
        with open(_DATA_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    return ref_id

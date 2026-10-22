"""Challenge A: persistent memory of the user's booking preferences ("quy uoc"), across sessions.

One JSON file holds every preference ever saved. Changing one never deletes the old value:
it is marked "superseded", with when and why, and the new value becomes the only "active" one.
At session start, run() overlays the active values onto Constraints, so the HARNESS enforces them.
"""
import json
import re
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from flight_agent.core import Constraints, Scenario, _f

KEYS = ("depart_before", "max_price")


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _normalise(key: str, value):
    """Return (value, None) in the stored form, or (None, hint) if it cannot be stored."""
    if key not in KEYS:
        return None, f"Unknown preference {key!r}. Use one of {', '.join(KEYS)}."
    raw = str(value).strip()
    if key == "depart_before":
        m = re.fullmatch(r"(\d{1,2}):(\d{2})", raw)
        if not m or int(m[1]) > 23 or int(m[2]) > 59:
            return None, "depart_before must be a time HH:MM, e.g. 09:00."
        return f"{int(m[1]):02d}:{m[2]}", None      # zero-padded: Constraints.is_ok compares strings
    if isinstance(value, float) and value.is_integer():
        raw = str(int(value))
    # Digits, optionally grouped by thousands ("2,000,000", "2.000.000"). A decimal part is refused:
    # stripping its "." would store "2000000.0" as 20,000,000.
    if not re.fullmatch(r"\d{1,3}([.,_]\d{3})+|\d+", raw) or (price := int(re.sub(r"\D", "", raw))) <= 0:
        return None, "max_price must be a positive number of VND, e.g. 2000000."
    return price, None


class MemoryStore:
    def __init__(self, path: Path = Path("memory/preferences.json")):
        self.path = Path(path)

    def entries(self) -> list[dict]:
        """Every entry ever saved, superseded ones included."""
        # A corrupt file raises on purpose: never overwrite the user's data silently.
        return json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else []

    def active(self) -> dict:
        return {e["key"]: e["value"] for e in self.entries() if e["status"] == "active"}

    normalise = staticmethod(_normalise)

    def remember(self, key: str, value, source: str) -> dict:
        value, hint = _normalise(key, value)
        if hint:
            return {"status": "invalid_param", "hint": hint}
        entries = self.entries()
        old = next((e for e in entries if e["key"] == key and e["status"] == "active"), None)
        if old and old["value"] == value:
            return {"status": "ok", "key": key, "value": value, "replaced": None}
        now = _now()
        if old:
            old.update(status="superseded", superseded_at=now)
        entries.append({"key": key, "value": value, "status": "active", "since": now,
                        "source": source, "superseded_at": None})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
        return {"status": "ok", "key": key, "value": value, "replaced": old and old["value"]}

    def to_prompt(self) -> str:
        """The exact block injected into the model's context."""
        lines = [f"- {e['key']} = {e['value']} (since {e['since']})"
                 for e in self.entries() if e["status"] == "active"] or ["- none yet"]
        return "\n".join(["Saved user preferences (from earlier sessions, the newest value wins):", *lines,
                          "Apply them unless the user says otherwise. When the user states a lasting rule "
                          '("from now on ..."), save it with remember_preference.'])

    def apply(self, constraints: Constraints) -> Constraints:
        return replace(constraints, **self.active())


# Lenient base constraints: the time/price rule comes ONLY from memory.
# The cheapest flight leaves in the afternoon (same trap as the "trap" scenario).
MEMORY_SCENARIO = Scenario(
    "memory", "Preferences come only from memory.",
    [_f("VJ610", "14:20", 1_150_000), _f("QH112", "10:30", 1_600_000), _f("VN122", "08:10", 1_950_000)],
    "done", constraints=Constraints(depart_before="24:00", max_price=10**9))

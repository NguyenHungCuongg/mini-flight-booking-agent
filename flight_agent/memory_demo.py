"""Challenge A demo: every run of this command is a NEW session (new process, empty chat history).
Only memory/preferences.json survives between sessions.

Run:  uv run python -m flight_agent.memory_demo --reset    # forget everything, start over
      uv run python -m flight_agent.memory_demo            # one session
"""
import json
import sys

from flight_agent.core import HumanApprover
from flight_agent.memory import MEMORY_SCENARIO, MemoryStore
from flight_agent.model import make_model
from flight_agent.run import run

if __name__ == "__main__":
    store = MemoryStore()
    if "--reset" in sys.argv:
        store.path.unlink(missing_ok=True)
        print(f"Memory reset: {store.path.as_posix()} is gone.")
        sys.exit()

    print("=== New session (new process, empty chat history) ===")
    print("\n--- Memory injected into context ---")
    print(store.to_prompt())
    print("\nFlights on offer:", ", ".join(f"{f['flight']} {f['depart'][11:]} {f['price']:,}"
                                          for f in MEMORY_SCENARIO.flights))
    request = input("\nYour request: ").strip()
    r = run("react", MEMORY_SCENARIO, make_model(), HumanApprover(), memory=store, request=request)

    print("\n--- trace ---")
    for i, t in enumerate(r.trace, 1):
        print(f"[{i}] {t['tool']}({t['args']}) -> {t['result']['status']} {t['result'].get('reason', '')}")
    print("\n--- result (decided by the harness, not the model) ---")
    print(json.dumps({"outcome": r.outcome, "stop_reason": r.stop_reason, "handoff": r.handoff,
                      "model_said": r.final_answer, "model_calls": r.model_calls, "error": r.error},
                     indent=2, ensure_ascii=False))
    print(f"\n--- Memory file now ({store.path.as_posix()}) ---")
    print(json.dumps(store.entries(), indent=2, ensure_ascii=False))

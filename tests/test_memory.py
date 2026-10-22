"""Challenge A: preferences survive across sessions (one session = one run()) and can be changed."""
import pytest
from langchain_core.messages import AIMessage

from flight_agent.core import RuleApprover
from flight_agent.memory import MEMORY_SCENARIO, MemoryStore
from flight_agent.run import run
from tests.test_run import SEARCH, book_pay_check, call, scripted

REQUEST = "Book one ticket SGN -> DAD on 2026-10-07."


def store(tmp_path):
    return MemoryStore(tmp_path / "m" / "preferences.json")


# ---------------------------------------------------------------- the store
def test_empty_memory_has_no_preferences(tmp_path):
    s = store(tmp_path)
    assert s.active() == {} and s.entries() == []
    assert "none yet" in s.to_prompt()


def test_remember_persists_on_disk_for_a_new_store(tmp_path):
    assert store(tmp_path).remember("depart_before", "11:00", source="s1")["status"] == "ok"
    assert store(tmp_path).active() == {"depart_before": "11:00"}       # new object = new session


def test_changing_a_preference_supersedes_the_old_one(tmp_path):
    s = store(tmp_path)
    s.remember("depart_before", "11:00", source="s1")
    r = s.remember("depart_before", "09:00", source="s3")
    assert r["replaced"] == "11:00"
    assert s.active() == {"depart_before": "09:00"}
    old, new = s.entries()
    assert (old["value"], old["status"], old["source"]) == ("11:00", "superseded", "s1")
    assert old["superseded_at"] and new["status"] == "active" and new["superseded_at"] is None


def test_same_preference_twice_adds_no_entry(tmp_path):
    s = store(tmp_path)
    s.remember("max_price", 2_000_000, source="s1")
    s.remember("max_price", "2000000", source="s2")
    assert len(s.entries()) == 1


@pytest.mark.parametrize("key,raw,stored", [
    ("depart_before", "9:00", "09:00"), ("depart_before", "11:00", "11:00"),
    ("max_price", "2,000,000", 2_000_000), ("max_price", "2.000.000", 2_000_000),
    ("max_price", 2_000_000, 2_000_000), ("max_price", 2_000_000.0, 2_000_000)])
def test_values_are_normalised(tmp_path, key, raw, stored):
    s = store(tmp_path)
    assert s.remember(key, raw, source="s")["value"] == stored
    assert s.active()[key] == stored


@pytest.mark.parametrize("key,raw", [("seat", "12A"), ("depart_before", "9h"), ("depart_before", "25:00"),
                                     ("max_price", "abc"), ("max_price", -5), ("max_price", "2000000.0"),
                                     ("max_price", "1.5"), ("max_price", "2,000,000.50")])
def test_bad_preferences_are_rejected_and_not_written(tmp_path, key, raw):
    s = store(tmp_path)
    assert s.remember(key, raw, source="s")["status"] == "invalid_param"
    assert s.entries() == []


def test_prompt_lists_active_values_only_and_apply_overlays_constraints(tmp_path):
    s = store(tmp_path)
    s.remember("depart_before", "11:00", source="s1")
    s.remember("depart_before", "09:00", source="s3")
    s.remember("max_price", 2_000_000, source="s1")
    assert "depart_before = 09:00" in s.to_prompt() and "depart_before = 11:00" not in s.to_prompt()
    c = s.apply(MEMORY_SCENARIO.constraints)
    assert (c.depart_before, c.max_price, c.origin) == ("09:00", 2_000_000, "SGN")


# ---------------------------------------------------------------- sessions (one run() each)
def session(mem, *replies, request=REQUEST, approver=None):
    return run("react", MEMORY_SCENARIO, scripted(*replies), approver or RuleApprover(),
               memory=mem, request=request)


def booked(r):
    return [t["result"]["status"] for t in r.trace if t["tool"] == "book_seat"]


def test_without_memory_the_cheap_afternoon_flight_is_allowed():           # control
    r = run("react", MEMORY_SCENARIO, scripted(call("search_flights", **SEARCH), *book_pay_check("VJ610"),
            AIMessage(content="Booked.")), RuleApprover(), request=REQUEST)
    assert r.outcome == "done" and r.memory_injected == ""


def test_a_session_saves_a_preference_the_user_states(tmp_path):
    r = session(store(tmp_path), call("remember_preference", key="depart_before", value="11:00"),
                call("search_flights", **SEARCH), *book_pay_check("QH112"), AIMessage(content="Booked."),
                request="From now on I only fly before 11:00. " + REQUEST)
    assert r.outcome == "done"
    assert store(tmp_path).active() == {"depart_before": "11:00"}


def test_proof_1_new_session_applies_the_saved_preference(tmp_path):
    store(tmp_path).remember("depart_before", "11:00", source="s1")
    r = session(store(tmp_path), call("search_flights", **SEARCH), call("book_seat", flight="VJ610"),
                *book_pay_check("QH112"), AIMessage(content="Booked QH112."))
    assert "depart_before = 11:00" in r.memory_injected                    # it was in the context
    assert booked(r) == ["denied", "ok"]
    assert r.outcome == "done"


def test_proof_2_changed_preference_applies_now_and_in_later_sessions(tmp_path):
    store(tmp_path).remember("depart_before", "11:00", source="s1")
    r = session(store(tmp_path), call("remember_preference", key="depart_before", value="09:00"),
                call("search_flights", **SEARCH), call("book_seat", flight="QH112"),
                *book_pay_check("VN122"), AIMessage(content="Booked VN122."),
                request="From now on I only fly before 09:00. " + REQUEST)
    assert booked(r) == ["denied", "ok"]
    nxt = session(store(tmp_path), call("search_flights", **SEARCH), call("book_seat", flight="QH112"),
                  *book_pay_check("VN122"), AIMessage(content="Booked VN122."))
    assert "depart_before = 09:00" in nxt.memory_injected
    assert "depart_before = 11:00" not in nxt.memory_injected     # not "11:00": a timestamp may contain it
    assert booked(nxt) == ["denied", "ok"]
    assert nxt.outcome == "done"


def test_saving_a_preference_needs_the_users_yes(tmp_path):
    class RefuseMemory(RuleApprover):
        def approve_memory(self, key, value):
            return False

    mem = store(tmp_path)
    r = session(mem, call("remember_preference", key="depart_before", value="23:59"),
                AIMessage(content="ok"), approver=RefuseMemory())
    assert r.trace[0]["result"]["status"] == "denied"
    assert mem.entries() == []


def test_the_user_confirms_the_value_that_will_be_stored_and_is_not_asked_about_invalid_ones(tmp_path):
    asked = []

    class Recorder(RuleApprover):
        def approve_memory(self, key, value):
            asked.append((key, value))
            return True

    session(store(tmp_path), call("remember_preference", key="depart_before", value="9h"),
            call("remember_preference", key="depart_before", value="9:00"),
            AIMessage(content="ok"), approver=Recorder())
    assert asked == [("depart_before", "09:00")]

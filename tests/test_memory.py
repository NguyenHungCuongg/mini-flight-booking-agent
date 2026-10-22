"""Challenge A: preferences survive across sessions (one session = one run()) and can be changed."""
import pytest

from flight_agent.memory import MEMORY_SCENARIO, MemoryStore


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
    ("max_price", 2_000_000, 2_000_000)])
def test_values_are_normalised(tmp_path, key, raw, stored):
    s = store(tmp_path)
    assert s.remember(key, raw, source="s")["value"] == stored
    assert s.active()[key] == stored


@pytest.mark.parametrize("key,raw", [("seat", "12A"), ("depart_before", "9h"), ("depart_before", "25:00"),
                                     ("max_price", "abc"), ("max_price", -5)])
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

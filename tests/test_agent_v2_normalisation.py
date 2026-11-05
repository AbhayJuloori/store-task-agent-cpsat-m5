import pytest

from storetask.agent.graph import norm_absent, norm_store, resolve_departments

DEPTS = [{"dept_id": d, "cat_id": d.rsplit("_", 1)[0]}
         for d in ["FOODS_1", "FOODS_2", "FOODS_3", "HOBBIES_1", "HOBBIES_2", "HOUSEHOLD_1", "HOUSEHOLD_2"]]


def test_category_name_expands_to_all_its_departments():
    assert resolve_departments(["FOODS"], DEPTS) == ["FOODS_1", "FOODS_2", "FOODS_3"]


def test_case_and_mixed_ids_and_categories():
    assert resolve_departments(["hobbies", "foods_2"], DEPTS) == ["FOODS_2", "HOBBIES_1", "HOBBIES_2"]


def test_empty_and_none_mean_no_exclusion():
    assert resolve_departments(None, DEPTS) == []
    assert resolve_departments([], DEPTS) == []


def test_unknown_department_raises_instead_of_silent_noop():
    with pytest.raises(ValueError, match="unknown department"):
        resolve_departments(["GARDEN"], DEPTS)


def test_absent_ids_normalised_and_validated():
    assert norm_absent(["a3", "A1", "A3"]) == ["A1", "A3"]
    assert norm_absent(None) == []
    with pytest.raises(ValueError):
        norm_absent(["HOUSEHOLD_1"])


def test_store_case():
    assert norm_store(" ca_2 ") == "CA_2"

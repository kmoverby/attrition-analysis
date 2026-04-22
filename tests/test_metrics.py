import pytest
import pandas as pd
from src.metrics import (
    attrition_rate,
    attrition_by_department,
    attrition_by_overtime,
    average_income_by_attrition,
    satisfaction_summary,
)


@pytest.fixture
def sample_df():
    # Sales: 2 leavers / 2 = 100%  HR: 1/2 = 50%  IT: 0/2 = 0%
    # Overtime Yes: 2/2 = 100%  No: 1/4 = 25%
    # Income leavers avg: (4000+6000+5000)/3 = 5000  stayers: (7000+8000+3000)/3 = 6000
    # Satisfaction: sat1=100%(1/1)  sat2=50%(1/2)  sat3=0%(0/1)  sat4=50%(1/2)
    return pd.DataFrame({
        "employee_id":     [1,       2,       3,      4,      5,      6    ],
        "department":      ["Sales", "Sales", "HR",   "HR",   "IT",   "IT" ],
        "monthly_income":  [4000.0,  6000.0,  5000.0, 7000.0, 8000.0, 3000.0],
        "job_satisfaction":[1,       4,       2,      3,      2,      4    ],
        "overtime":        ["Yes",   "No",    "Yes",  "No",   "No",   "No" ],
        "attrition":       ["Yes",   "Yes",   "Yes",  "No",   "No",   "No" ],
    })


# --- attrition_rate ---

def test_attrition_rate_returns_expected_percent():
    df = pd.DataFrame({
        "employee_id": [1, 2, 3, 4],
        "department": ["Sales", "Sales", "HR", "HR"],
        "attrition": ["Yes", "No", "No", "Yes"],
    })
    assert attrition_rate(df) == 50.0


def test_attrition_rate_no_leavers():
    df = pd.DataFrame({"employee_id": [1, 2], "attrition": ["No", "No"]})
    assert attrition_rate(df) == 0.0


def test_attrition_rate_all_leavers():
    df = pd.DataFrame({"employee_id": [1, 2], "attrition": ["Yes", "Yes"]})
    assert attrition_rate(df) == 100.0


# --- attrition_by_department ---

def test_attrition_by_department_returns_expected_columns():
    df = pd.DataFrame({
        "employee_id": [1, 2, 3, 4],
        "department": ["Sales", "Sales", "HR", "HR"],
        "attrition": ["Yes", "No", "No", "Yes"],
    })
    result = attrition_by_department(df)
    assert list(result.columns) == ["department", "employees", "leavers", "attrition_rate"]


def test_attrition_by_department_calculates_correct_rates(sample_df):
    result = attrition_by_department(sample_df)
    rates = result.set_index("department")["attrition_rate"]
    assert rates["Sales"] == 100.0
    assert rates["HR"] == 50.0
    assert rates["IT"] == 0.0


def test_attrition_by_department_sorted_descending(sample_df):
    result = attrition_by_department(sample_df)
    assert list(result["department"]) == ["Sales", "HR", "IT"]


# --- attrition_by_overtime ---

def test_attrition_by_overtime_returns_expected_columns(sample_df):
    result = attrition_by_overtime(sample_df)
    assert list(result.columns) == ["overtime", "employees", "leavers", "attrition_rate"]


def test_attrition_by_overtime_calculates_correct_rates(sample_df):
    result = attrition_by_overtime(sample_df)
    rates = result.set_index("overtime")["attrition_rate"]
    assert rates["Yes"] == 100.0   # 2 of 2 overtime employees left
    assert rates["No"] == 25.0     # 1 of 4 non-overtime employees left


# --- average_income_by_attrition ---

def test_average_income_by_attrition_returns_expected_columns(sample_df):
    result = average_income_by_attrition(sample_df)
    assert list(result.columns) == ["attrition", "avg_monthly_income"]


def test_average_income_by_attrition_calculates_correct_values(sample_df):
    result = average_income_by_attrition(sample_df)
    incomes = result.set_index("attrition")["avg_monthly_income"]
    assert incomes["Yes"] == 5000.0   # (4000 + 6000 + 5000) / 3
    assert incomes["No"] == 6000.0    # (7000 + 8000 + 3000) / 3


# --- satisfaction_summary ---

def test_satisfaction_summary_returns_expected_columns(sample_df):
    result = satisfaction_summary(sample_df)
    assert list(result.columns) == ["job_satisfaction", "total_employees", "leavers", "attrition_rate"]


def test_satisfaction_summary_calculates_within_group_rate(sample_df):
    result = satisfaction_summary(sample_df)
    rates = result.set_index("job_satisfaction")["attrition_rate"]
    assert rates[1] == 100.0   # 1 of 1 left
    assert rates[2] == 50.0    # 1 of 2 left
    assert rates[3] == 0.0     # 0 of 1 left
    assert rates[4] == 50.0    # 1 of 2 left


def test_satisfaction_summary_sorted_ascending(sample_df):
    result = satisfaction_summary(sample_df)
    assert list(result["job_satisfaction"]) == [1, 2, 3, 4]

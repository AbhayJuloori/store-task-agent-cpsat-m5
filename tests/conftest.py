import pandas as pd
import pytest

from storetask.layout import build_layout


@pytest.fixture
def synthetic_items():
    return pd.DataFrame(
        [
            {"item_id": f"{dept}_{item:03d}", "dept_id": dept}
            for dept in ("D1", "D2", "D3")
            for item in range(72)
        ]
    )


@pytest.fixture
def layout(synthetic_items):
    return build_layout("TEST_STORE", synthetic_items, seed=11)

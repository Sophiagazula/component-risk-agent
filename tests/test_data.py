"""Supply chain data integrity: bill of materials, parts, suppliers, demand."""
import pandas as pd
import pytest

from model import DATA


@pytest.fixture(scope="module")
def tables():
    return {name: pd.read_csv(DATA / f"{name}.csv")
            for name in ["products", "components", "suppliers", "bom", "demand_history", "pipeline_snapshots"]}


def test_every_device_has_a_bill_of_materials(tables):
    assert set(tables["products"].product_id) == set(tables["bom"].product_id)


def test_every_part_is_used_and_known(tables):
    bom_parts = set(tables["bom"].component_id)
    assert bom_parts == set(tables["components"].component_id)
    assert (tables["bom"].qty_per > 0).all()


def test_every_part_has_a_known_supplier(tables):
    sup = set(tables["suppliers"].supplier_id)
    c = tables["components"]
    assert set(c.supplier_id) <= sup
    assert set(c.alt_supplier_id.dropna()) <= sup
    assert (c.alt_supplier_id.dropna() != c[c.alt_supplier_id.notna()].supplier_id).all()


def test_demand_is_complete_and_non_negative(tables):
    d = tables["demand_history"]
    assert (d.units >= 0).all()
    per_product = d.groupby("product_id").month.nunique()
    assert per_product.nunique() == 1 and per_product.iloc[0] >= 36


def test_pipeline_snapshot_exists_for_every_origin_and_horizon(tables):
    p = tables["pipeline_snapshots"]
    counts = p.groupby(["origin", "product_id"]).horizon.nunique()
    assert (counts == 6).all()

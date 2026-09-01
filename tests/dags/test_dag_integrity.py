import pytest
from airflow.models import DagBag


@pytest.fixture
def dagbag():
    return DagBag(include_examples=False)


def test_no_import_errors(dagbag):
    assert not dagbag.import_errors


def test_f1_ingest_exists(dagbag):
    assert "f1_ingest" in dagbag.dags

"""Chargement idempotent des mesures.

Aucun test ne joint PostgreSQL : la garantie qui compte ici est que
l'instruction produite porte bien le ON CONFLICT sur la clé primaire composite,
et que les manquants pandas sortent en NULL.
"""

from sqlalchemy.dialects import postgresql

from etl.load import build_upsert, load_frame, to_records
from etl.transform import to_frame


class FakeConnection:
    """Connexion factice qui mémorise les instructions exécutées."""

    def __init__(self, executed):
        self._executed = executed

    def execute(self, statement):
        self._executed.append(statement)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class FakeEngine:
    """Moteur factice : begin() rend une transaction sans base derrière."""

    def __init__(self):
        self.executed = []

    def begin(self):
        return FakeConnection(self.executed)


def test_to_records_converts_missing_values_to_none(make_reading) -> None:
    frame = to_frame([make_reading("2026-01-15T08:00:00Z", consumption_kw=None)])
    record = to_records(frame)[0]
    assert record["consumption_kw"] is None
    assert record["null_reasons"] == []
    assert record["site_id"] == "SITE001"


def test_build_upsert_ignores_rows_already_loaded(make_reading) -> None:
    frame = to_frame([make_reading("2026-01-15T08:00:00Z")])
    compiled = build_upsert(to_records(frame)).compile(
        dialect=postgresql.dialect()
    )
    assert "ON CONFLICT (site_id, ts) DO NOTHING" in str(compiled)


def test_load_frame_writes_nothing_for_an_empty_frame() -> None:
    engine = FakeEngine()
    assert load_frame(engine, to_frame([]), batch_size=10) == 0
    assert engine.executed == []


def test_load_frame_splits_the_batch(make_reading) -> None:
    frame = to_frame(
        [
            make_reading(f"2026-01-15T0{hour}:00:00Z")
            for hour in range(5)
        ]
    )
    engine = FakeEngine()
    assert load_frame(engine, frame, batch_size=2) == 5
    # 5 lignes par lots de 2 : trois instructions, la dernière incomplète.
    assert len(engine.executed) == 3


def test_load_frame_reports_submitted_rows(make_reading) -> None:
    frame = to_frame([make_reading("2026-01-15T08:00:00Z")])
    engine = FakeEngine()
    assert load_frame(engine, frame, batch_size=10) == len(frame)
    assert len(engine.executed) == 1

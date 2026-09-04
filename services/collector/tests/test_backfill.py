"""Rattrapage de l'historique par lot : les critères d'EV-09.

Trois exigences, et chacune a sa raison d'être vérifiée ici plutôt qu'à la
main sur une base de recette.

La période est paramétrable de deux façons, et elles ne servent pas au même
usage : `--start/--end` nomme une période, ce que fait un analyste ;
`--date/--days` nomme une journée et sa profondeur, ce que fait un
ordonnanceur. Les mélanger laisserait deux périodes possibles pour un même
appel, et le run partirait sur l'une des deux sans dire laquelle.

La limite est bornée par la source à 1000. La refuser ici plutôt que de la
découvrir dans une réponse 422 évite de lancer un rattrapage de trois mois qui
échouera à la première page.

L'idempotence tient à l'instruction produite, pas au hasard : un
`ON CONFLICT DO NOTHING` sur la clé naturelle. C'est elle qui rend le rejeu
sans effet de bord, et qui empêche une recollecte d'effacer ce que l'ETL a
déduit depuis.
"""

from __future__ import annotations

from datetime import date

import pytest
from conftest import FakeEngine
from sqlalchemy.dialects import postgresql

from collector.__main__ import collect_day, parse_args, requested_days
from collector.sink import build_insert, to_measures, to_records, write
from predict_common.source import MAX_PAGE_SIZE, SourceError, SourceSettings

SITES = tuple(f"SITE{index:03d}" for index in range(1, 8))


def days_for(*argv: str) -> list[date]:
    """Retourne les journées que la ligne de commande demande."""
    return requested_days(parse_args(list(argv)))


class TestPeriode:
    """Le premier critère : une période paramétrable."""

    def test_two_bounds_give_the_whole_range(self) -> None:
        days = days_for("--start", "2026-08-30", "--end", "2026-09-02")
        assert days == [
            date(2026, 8, 30),
            date(2026, 8, 31),
            date(2026, 9, 1),
            date(2026, 9, 2),
        ]

    def test_a_lone_start_collects_that_day(self) -> None:
        # Lecture naturelle d'une borne basse sans borne haute.
        assert days_for("--start", "2026-09-02") == [date(2026, 9, 2)]

    def test_the_short_form_counts_backwards(self) -> None:
        # Ce qu'un ordonnanceur écrit : la date est un paramètre, la
        # profondeur une constante.
        days = days_for("--date", "2026-09-02", "--days", "3")
        assert days == [date(2026, 8, 31), date(2026, 9, 1), date(2026, 9, 2)]

    def test_a_single_day_is_the_default_depth(self) -> None:
        assert days_for("--date", "2026-09-02") == [date(2026, 9, 2)]

    def test_the_two_forms_are_exclusive(self) -> None:
        # Les mélanger laisserait deux périodes possibles pour un même appel.
        with pytest.raises(ValueError, match="une seule"):
            days_for("--date", "2026-09-02", "--start", "2026-08-01")

    def test_an_end_without_a_start_is_refused(self) -> None:
        with pytest.raises(ValueError, match="--start"):
            days_for("--end", "2026-09-02")

    def test_an_absent_period_is_refused(self) -> None:
        # Sans période, il n'y a pas de défaut raisonnable : collecter « tout »
        # demanderait des mois à la source, collecter « aujourd'hui » serait un
        # choix que personne n'a exprimé.
        with pytest.raises(ValueError, match="Période absente"):
            days_for()

    def test_an_inverted_range_is_refused(self) -> None:
        from predict_common.paths import PathError

        with pytest.raises(PathError):
            days_for("--start", "2026-09-02", "--end", "2026-08-01")

    def test_a_null_depth_is_refused(self) -> None:
        with pytest.raises(ValueError, match="--days"):
            days_for("--date", "2026-09-02", "--days", "0")


class TestLimite:
    """Le deuxième critère : une limite de page, bornée par la source."""

    def test_the_source_bound_is_a_thousand(self) -> None:
        assert MAX_PAGE_SIZE == 1000

    def test_the_command_line_overrides_the_configuration(self) -> None:
        settings = _settings().with_page_size(500)
        assert settings.page_size == 500

    def test_the_bound_is_accepted(self) -> None:
        assert _settings().with_page_size(MAX_PAGE_SIZE).page_size == 1000

    def test_beyond_the_bound_is_refused_here_and_not_by_a_422(self) -> None:
        # Découvrir la borne dans une réponse enverrait chercher la panne du
        # côté du réseau, après avoir lancé un rattrapage de trois mois.
        with pytest.raises(ValueError, match="1000"):
            _settings().with_page_size(MAX_PAGE_SIZE + 1)

    def test_a_null_limit_is_refused(self) -> None:
        with pytest.raises(ValueError):
            _settings().with_page_size(0)

    def test_the_original_settings_are_left_alone(self) -> None:
        # Les réglages sont figés : une surcharge en rend d'autres, elle n'en
        # modifie aucun. Deux appels ne peuvent donc pas se marcher dessus.
        settings = _settings()
        settings.with_page_size(500)
        assert settings.page_size == 1000

    def test_the_command_line_reads_the_limit(self) -> None:
        assert parse_args(["--date", "2026-09-02", "--limit", "250"]).limit == 250


class TestIdempotence:
    """Le troisième critère : un rejeu qui ne double ni n'efface rien."""

    def test_the_insert_ignores_what_is_already_there(self, make_reading) -> None:
        frame = to_measures([make_reading("2026-09-02T08:00:00Z")])
        statement = build_insert(to_records(frame))
        compiled = str(statement.compile(dialect=postgresql.dialect()))
        assert "ON CONFLICT (site_id, ts) DO NOTHING" in compiled

    def test_a_replay_never_overwrites_what_the_etl_deduced(
        self, make_reading
    ) -> None:
        # `DO UPDATE` recouvrirait `consumption_kw_imputed` et
        # `imputation_method` d'une ligne déjà transformée : rattraper un
        # historique défairait la transformation faite depuis.
        frame = to_measures([make_reading("2026-09-02T08:00:00Z")])
        compiled = str(
            build_insert(to_records(frame)).compile(dialect=postgresql.dialect())
        )
        assert "DO UPDATE" not in compiled

    def test_a_duplicated_key_within_one_batch_is_settled_first(
        self, make_reading
    ) -> None:
        # ON CONFLICT arbitre entre le lot et la table, jamais à l'intérieur
        # d'un même lot : deux fois la même clé ferait échouer l'insertion.
        engine = FakeEngine()
        frame = to_measures(
            [
                make_reading("2026-09-02T08:00:00Z", consumption_kw=10.0),
                make_reading("2026-09-02T08:00:00Z", consumption_kw=20.0),
            ]
        )
        assert write(engine, frame, batch_size=10).rows == 1


class TestVolume:
    """Le dernier critère : au moins 48 h pour les sept sites."""

    def test_two_days_are_expressible_in_both_forms(self) -> None:
        assert len(days_for("--start", "2026-09-01", "--end", "2026-09-02")) == 2
        assert len(days_for("--date", "2026-09-02", "--days", "2")) == 2

    def test_every_site_is_collected_by_default(self) -> None:
        # Sans `--site`, la liste vient du référentiel : les sept sites sont
        # collectés sans avoir à les nommer.
        assert parse_args(["--date", "2026-09-02"]).sites is None

    def test_the_sites_can_be_narrowed(self) -> None:
        argv = ["--date", "2026-09-02"]
        for site in SITES[:3]:
            argv += ["--site", site]
        assert parse_args(argv).sites == list(SITES[:3])

    def test_a_two_day_batch_of_seven_sites_is_submitted_whole(
        self, make_reading
    ) -> None:
        # 7 sites × 48 h à la demi-heure : le lot d'un rattrapage réel passe
        # en entier, et le compte rendu porte sur ce qui a été soumis.
        readings = [
            make_reading(f"2026-09-0{day}T{hour:02d}:{minute}0:00Z", site_id=site)
            for site in SITES
            for day in (1, 2)
            for hour in range(24)
            for minute in (0, 3)
        ]
        engine = FakeEngine()
        report = write(engine, to_measures(readings), batch_size=1000)
        assert report.rows == len(SITES) * 2 * 24 * 2
        assert report.dropped == 0


def _settings() -> SourceSettings:
    """Réglages de source dont seule la taille de page nous intéresse."""
    return SourceSettings(
        base_url="http://mock.invalid",
        sites_path="/api/v1/sites",
        readings_path="/api/v1/readings",
        current_path="/api/v1/sites/{site_id}/current",
        simulate_spike_path="/api/v1/simulate/spike/{site_id}",
        alerts_path="/api/v1/alerts",
        sensors_status_path="/api/v1/sensors/status",
        page_size=1000,
        timeout_s=30.0,
        poll_timeout_s=10.0,
        retries=1,
        backoff_s=0.0,
        rate_limit_rps=0.0,
    )


class _StubSource:
    """Source réduite à ce que `collect_day` lui demande.

    La pagination réelle, ses reprises et sa limite de débit sont éprouvées
    par `test_client.py` : les rejouer ici ne testerait pas le rattrapage.
    """

    def __init__(self, readings: list[dict], failing: str | None = None) -> None:
        self._readings = readings
        self._failing = failing

    def iter_readings(self, site_id: str, start_time, end_time):
        if site_id == self._failing:
            raise SourceError(f"{site_id} muet")
        return [dict(reading, site_id=site_id) for reading in self._readings]


def _state_params(engine: FakeEngine) -> list[dict]:
    """Rend les paramètres des instructions visant `ingestion_etat`."""
    compiled = [
        statement.compile(dialect=postgresql.dialect())
        for statement in engine.executed
    ]
    return [
        instruction.params
        for instruction in compiled
        if "ingestion_etat" in str(instruction)
    ]


class TestBackfillState:
    """Un rattrapage ne doit pas se faire passer pour une collecte vivante."""

    def test_a_replayed_day_is_recorded_as_a_backfill(self, make_reading) -> None:
        # C'est précisément quand le poller est arrêté qu'on rejoue une
        # journée à la main. Une ligne qui ne dirait pas d'où elle vient
        # ferait alors paraître l'ingestion fraîche.
        engine = FakeEngine()
        source = _StubSource([make_reading("2026-09-02T07:59:00Z")])
        collect_day(source, engine, 1000, date(2026, 9, 2), ["SITE001"])

        params = _state_params(engine)
        assert len(params) == 1
        assert "backfill" in params[0].values()

    def test_a_source_failure_is_recorded_before_it_propagates(
        self, make_reading
    ) -> None:
        # Le rattrapage s'arrête sur un échec de source, mais ce qu'il savait
        # à cet instant a plus de valeur écrit que perdu.
        engine = FakeEngine()
        source = _StubSource(
            [make_reading("2026-09-02T07:59:00Z")], failing="SITE002"
        )
        with pytest.raises(SourceError):
            collect_day(
                source, engine, 1000, date(2026, 9, 2), ["SITE001", "SITE002"]
            )

        params = _state_params(engine)
        # Un succès et un échec : deux formes, donc deux instructions.
        assert len(params) == 2

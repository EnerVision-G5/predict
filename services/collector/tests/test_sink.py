"""Écriture de la couche brute : rien n'est transformé, rien n'est perdu.

Aucun test ne joint PostgreSQL. Deux garanties comptent ici. La première est
de fidélité : ce que la source a servi arrive en base tel quel, valeurs nulles
comprises, et le collecteur n'invente aucune qualification. La seconde est
d'exploitation : l'insertion n'écrase jamais, ce qui rend un rejeu sans effet
de bord et surtout empêche une recollecte de recouvrir ce que l'ETL a déduit
depuis.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
from collector_fakes import FakeEngine
from sqlalchemy.dialects import postgresql

from collector.sink import (
    UNQUALIFIED,
    build_insert,
    build_site_upsert,
    deduplicate,
    drop_unplaceable,
    select_day,
    snap_to_grid,
    sync_sites,
    to_measures,
    to_records,
    to_sites,
    validate,
    write,
)
from predict_common.db import DERIVED_COLUMNS, SOURCE_COLUMNS
from predict_common.schemas import TIMESTAMP_COLUMN

DAY = date(2026, 9, 2)


def test_to_measures_returns_the_table_columns_on_an_empty_batch() -> None:
    frame = to_measures([])
    assert list(frame.columns) == list(SOURCE_COLUMNS)
    assert frame.empty


def test_to_measures_renames_timestamp_to_ts(make_reading) -> None:
    # La table s'appelle `ts` : c'est la seule frontière de renommage de la
    # chaîne, et elle est traversée une fois.
    frame = to_measures([make_reading("2026-09-02T08:00:00Z")])
    assert "ts" in frame.columns
    assert "timestamp" not in frame.columns


def test_to_measures_normalizes_the_timestamp_to_utc(make_reading) -> None:
    frame = to_measures([make_reading("2026-09-02T08:30:00+02:00")])
    assert str(frame.loc[0, "ts"].tz) == "UTC"
    assert frame.loc[0, "ts"].hour == 6


def test_to_measures_keeps_a_null_measure(make_reading) -> None:
    # Une valeur nulle porte une panne capteur : la filtrer perdrait la panne.
    frame = to_measures(
        [
            make_reading(
                "2026-09-02T08:00:00Z",
                consumption_kw=None,
                null_reasons=["sensor_failure"],
                data_quality="critical",
            )
        ]
    )
    assert pd.isna(frame.loc[0, "consumption_kw"])
    assert frame.loc[0, "null_reasons"] == ["sensor_failure"]
    assert frame.loc[0, "data_quality"] == "critical"


def test_to_measures_writes_no_derived_column(make_reading) -> None:
    # `consumption_kw_imputed` et `imputation_method` appartiennent à l'ETL.
    frame = to_measures([make_reading("2026-09-02T08:00:00Z")])
    assert not set(frame.columns) & {"consumption_kw_imputed", "imputation_method"}


def test_an_unqualified_measure_falls_back_on_the_column_default(
    make_reading,
) -> None:
    # `NOT NULL DEFAULT 'good'` : le schéma figé ne sait pas dire « non
    # qualifiée ». Le collecteur écrit donc le défaut, et l'ETL repose la
    # qualification à son passage — un `good` sur une puissance absente
    # redevient `critical`.
    frame = to_measures([make_reading("2026-09-02T08:00:00Z", data_quality=None)])
    assert frame.loc[0, "data_quality"] == UNQUALIFIED


def test_a_quality_the_check_would_reject_falls_back_too(make_reading) -> None:
    # 'excellent' n'est pas dans le CHECK : le soumettre ferait échouer
    # l'insertion du lot entier, pas seulement de sa ligne.
    reading = make_reading("2026-09-02T08:00:00Z", data_quality="excellent")
    frame = to_measures([reading])
    assert frame.loc[0, "data_quality"] == UNQUALIFIED


def test_to_measures_normalizes_a_lone_motive_to_a_list(make_reading) -> None:
    frame = to_measures([make_reading("2026-09-02T08:00:00Z", null_reasons="boum")])
    assert frame.loc[0, "null_reasons"] == ["boum"]


def test_to_measures_normalizes_an_absent_motive_to_an_empty_list(
    make_reading,
) -> None:
    # `TEXT[] NOT NULL` : l'absence de motif s'écrit par une liste vide.
    frame = to_measures([make_reading("2026-09-02T08:00:00Z", null_reasons=None)])
    assert frame.loc[0, "null_reasons"] == []


def test_to_measures_turns_an_unreadable_value_into_a_null(make_reading) -> None:
    frame = to_measures([make_reading("2026-09-02T08:00:00Z", voltage_v="n/a")])
    assert pd.isna(frame.loc[0, "voltage_v"])


def test_drop_unplaceable_removes_a_measure_without_a_timestamp(
    make_reading,
) -> None:
    # (site_id, ts) est la clé : sans l'un des deux, la base refuse la ligne
    # sans dire laquelle du lot est fautive.
    frame = to_measures(
        [make_reading("pas une date"), make_reading("2026-09-02T08:00:00Z")]
    )
    placeable, dropped = drop_unplaceable(frame)
    assert len(placeable) == 1
    assert dropped == 1


def test_drop_unplaceable_removes_a_measure_without_a_site(make_reading) -> None:
    frame = to_measures([make_reading("2026-09-02T08:00:00Z", site_id="  ")])
    _, dropped = drop_unplaceable(frame)
    assert dropped == 1


def test_deduplicate_keeps_the_last_of_a_duplicated_key(make_reading) -> None:
    # ON CONFLICT arbitre entre le lot et la table, pas à l'intérieur d'un
    # même lot : un doublon ferait échouer l'insertion entière.
    frame = to_measures(
        [
            make_reading("2026-09-02T08:00:00Z", consumption_kw=10.0),
            make_reading("2026-09-02T08:00:00Z", consumption_kw=20.0),
        ]
    )
    deduplicated = deduplicate(frame)
    assert len(deduplicated) == 1
    assert deduplicated.loc[0, "consumption_kw"] == 20.0


def test_deduplicate_separates_two_sites_at_the_same_instant(make_reading) -> None:
    frame = to_measures(
        [
            make_reading("2026-09-02T08:00:00Z", site_id="SITE001"),
            make_reading("2026-09-02T08:00:00Z", site_id="SITE002"),
        ]
    )
    assert len(deduplicate(frame)) == 2


def test_select_day_refuses_a_measure_of_the_neighbouring_day(make_reading) -> None:
    frame = to_measures(
        [
            make_reading("2026-09-02T23:59:00Z"),
            make_reading("2026-09-03T00:01:00Z"),
        ]
    )
    assert len(select_day(frame, DAY)) == 1


def test_validate_accepts_a_conforming_batch(make_reading) -> None:
    assert len(validate(to_measures([make_reading("2026-09-02T08:00:00Z")]))) == 1


def test_to_records_converts_missing_values_to_none(make_reading) -> None:
    # Un NaN flottant dans une colonne NUMERIC est accepté par PostgreSQL et
    # pollue silencieusement les agrégats en aval.
    frame = to_measures([make_reading("2026-09-02T08:00:00Z", consumption_kw=None)])
    record = to_records(frame)[0]
    assert record["consumption_kw"] is None
    assert record["site_id"] == "SITE001"


def test_the_insert_never_overwrites(make_reading) -> None:
    # La ligne présente peut déjà porter les colonnes que l'ETL a déduites :
    # une recollecte n'a aucune raison de les effacer.
    frame = to_measures([make_reading("2026-09-02T08:00:00Z")])
    statement = build_insert(to_records(frame))
    compiled = str(statement.compile(dialect=postgresql.dialect()))
    assert "ON CONFLICT (site_id, ts) DO NOTHING" in compiled
    assert "DO UPDATE" not in compiled


def test_the_insert_writes_no_derived_column(make_reading) -> None:
    frame = to_measures([make_reading("2026-09-02T08:00:00Z")])
    submitted = set(to_records(frame)[0])
    assert not submitted & set(DERIVED_COLUMNS) - {"null_reasons", "data_quality"}


def test_write_reports_what_it_submitted(make_reading) -> None:
    engine = FakeEngine()
    frame = to_measures([make_reading("2026-09-02T08:00:00Z")])
    report = write(engine, frame, batch_size=10)
    assert report.rows == 1
    assert len(engine.executed) == 1


def test_write_splits_the_batch(make_reading) -> None:
    frame = to_measures(
        [make_reading(f"2026-09-02T0{hour}:00:00Z") for hour in range(5)]
    )
    engine = FakeEngine()
    assert write(engine, frame, batch_size=2).rows == 5
    # 5 lignes par lots de 2 : trois instructions, la dernière incomplète.
    assert len(engine.executed) == 3


def test_write_submits_nothing_for_an_empty_batch() -> None:
    engine = FakeEngine()
    assert write(engine, to_measures([]), batch_size=10).rows == 0
    assert engine.executed == []


def test_write_reports_what_it_could_not_place(make_reading) -> None:
    engine = FakeEngine()
    frame = to_measures(
        [make_reading("pas une date"), make_reading("2026-09-02T08:00:00Z")]
    )
    report = write(engine, frame, batch_size=10)
    assert (report.rows, report.dropped) == (1, 1)


def test_write_can_restrict_itself_to_one_day(make_reading) -> None:
    engine = FakeEngine()
    frame = to_measures(
        [make_reading("2026-09-02T08:00:00Z"), make_reading("2026-09-03T08:00:00Z")]
    )
    assert write(engine, frame, batch_size=10, day=DAY).rows == 1


class TestSiteSync:
    """Le référentiel est mis à jour avant les mesures, jamais après."""

    def test_a_complete_site_is_kept(self) -> None:
        sites = to_sites(
            [
                {
                    "site_id": "SITE001",
                    "site_type": "office",
                    "site_name": "Bureau Paris",
                    "location": "Paris",
                    "capacity_kw": 200,
                    "status": "active",
                }
            ]
        )
        assert len(sites) == 1
        assert sites[0]["capacity_kw"] == 200

    def test_a_half_described_site_is_left_out(self) -> None:
        # Trois colonnes de `site` sont NOT NULL sans défaut : un site
        # incomplet ferait rejeter le lot entier, et le seed en a déjà posé
        # une version placeholder qui vaut mieux qu'un échec.
        assert to_sites([{"site_id": "SITE009", "site_type": "office"}]) == []

    def test_an_absent_status_takes_the_usual_one(self) -> None:
        sites = to_sites(
            [
                {
                    "site_id": "SITE001",
                    "site_type": "office",
                    "site_name": "Bureau",
                    "capacity_kw": 200,
                }
            ]
        )
        assert sites[0]["status"] == "active"

    def test_the_upsert_replaces_the_seed_placeholders(self) -> None:
        # `02_seed_sites.sql` pose des capacités « à synchroniser » : c'est
        # précisément le rôle de cette écriture de les remplacer.
        records = to_sites(
            [
                {
                    "site_id": "SITE004",
                    "site_type": "factory",
                    "site_name": "Usine",
                    "capacity_kw": 750,
                }
            ]
        )
        compiled = str(
            build_site_upsert(records).compile(dialect=postgresql.dialect())
        )
        assert "ON CONFLICT (site_id) DO UPDATE" in compiled
        assert "capacity_kw = excluded.capacity_kw" in compiled

    def test_sync_reports_what_it_wrote(self) -> None:
        engine = FakeEngine()
        written = sync_sites(
            engine,
            [
                {
                    "site_id": "SITE001",
                    "site_type": "office",
                    "site_name": "Bureau",
                    "capacity_kw": 200,
                }
            ],
            batch_size=10,
        )
        assert written == 1
        assert len(engine.executed) == 1


def test_write_refuses_a_batch_breaking_the_contract(make_reading) -> None:
    # Un identifiant que le VARCHAR(20) tronquerait n'entre pas en base.
    engine = FakeEngine()
    frame = to_measures([make_reading("2026-09-02T08:00:00Z", site_id="S" * 21)])
    with pytest.raises(Exception, match="site_id"):
        write(engine, frame, batch_size=10)


# --- Grille à la minute -----------------------------------------------------
#
# `mesure` est une grille : une ligne par site et par minute. Le rattrapage lit
# `/readings`, servi sur des minutes pleines ; le poller lit `/current`, daté de
# l'instant de l'appel. Sans normalisation, la même minute entrait en base sous
# deux clés et `ON CONFLICT DO NOTHING` n'avait aucun conflit à arbitrer.


def test_une_mesure_datee_dans_la_minute_est_ramenee_sur_la_grille() -> None:
    """C'est ce que sert `/current` : l'instant de l'appel, pas la minute."""
    snapped = snap_to_grid(
        pd.Series(pd.to_datetime(["2026-09-05T13:53:31.587801Z"], utc=True))
    )

    assert snapped.iloc[0] == pd.Timestamp("2026-09-05T13:53:00Z")


def test_l_arrondi_est_vers_le_bas() -> None:
    """Une mesure appartient à la minute commencée, pas à la suivante.

    Arrondir au plus proche daterait une mesure de 13:53:59 à 13:54,
    c'est-à-dire d'une minute qui n'a pas encore eu lieu.
    """
    snapped = snap_to_grid(
        pd.Series(pd.to_datetime(["2026-09-05T13:53:59.999Z"], utc=True))
    )

    assert snapped.iloc[0] == pd.Timestamp("2026-09-05T13:53:00Z")


def test_les_deux_routes_ecrivent_la_meme_cle(make_reading) -> None:
    """Le rejeu d'une journée déjà collectée au fil de l'eau ne double rien.

    C'est la propriété qui manquait : le poller écrivait `13:53:31.587801` et
    le rattrapage `13:53:00`, deux clés primaires distinctes pour la même
    minute. Rattraper une journée déjà collectée doublait ses lignes, et
    `ON CONFLICT DO NOTHING` n'avait aucun conflit à arbitrer.
    """
    du_poller = to_measures([make_reading("2026-09-05T13:53:31.587801Z")])
    du_rattrapage = to_measures([make_reading("2026-09-05T13:53:00Z")])

    poller_engine, rattrapage_engine = FakeEngine(), FakeEngine()
    write(poller_engine, du_poller, batch_size=10)
    write(rattrapage_engine, du_rattrapage, batch_size=10)

    assert _submitted_keys(poller_engine) == _submitted_keys(rattrapage_engine)


def test_deux_relevés_de_la_même_minute_ne_font_qu_une_ligne(make_reading) -> None:
    """Un poller redémarré deux fois dans la minute n'en écrit pas deux.

    La déduplication a lieu APRÈS le calage : avant, les deux horodatages
    étaient distincts et aucune des deux lignes n'était vue comme un doublon.
    """
    frame = to_measures(
        [
            make_reading("2026-09-05T13:53:05.100000Z"),
            make_reading("2026-09-05T13:53:48.900000Z"),
        ]
    )
    engine = FakeEngine()

    assert write(engine, frame, batch_size=10).rows == 1


def test_le_retard_mesure_n_est_pas_quantifie_par_la_grille(make_reading) -> None:
    """`to_measures` ne cale pas : le poller y lit l'âge réel de la mesure.

    La grille appartient à la TABLE, pas à la mesure. Caler avant de mesurer
    ferait paraître en retard de cinquante secondes un site à l'heure, et
    `ingestion_etat.last_data_lag_s` deviendrait illisible.
    """
    frame = to_measures([make_reading("2026-09-05T13:53:31.587801Z")])

    assert frame[TIMESTAMP_COLUMN].iloc[0] == pd.Timestamp(
        "2026-09-05T13:53:31.587801Z"
    )


def test_un_horodatage_illisible_reste_ecarte(make_reading) -> None:
    """Le calage ne rattrape pas ce que la source n'a pas su dater."""
    frame = to_measures([make_reading("pas une date")])

    assert frame[TIMESTAMP_COLUMN].isna().all()


def _submitted_keys(engine: FakeEngine) -> list[tuple]:
    """Clés (site_id, ts) que les instructions soumises portent.

    Lues dans les paramètres liés de l'instruction compilée : c'est ce qui
    part réellement vers la base, et non ce que le tableau portait avant.
    """
    keys = []
    for statement in engine.executed:
        params = statement.compile(dialect=postgresql.dialect()).params
        index = 0
        while f"site_id_m{index}" in params:
            keys.append((params[f"site_id_m{index}"], params[f"ts_m{index}"]))
            index += 1
    return keys

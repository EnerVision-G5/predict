"""Collecte des deux routes que `mesure` ne remplace pas.

`GET /api/v1/alerts` dit ce que la source a jugé anormal, avec sa valeur et
son seuil — rien de tout cela n'est dans une mesure, et l'alerte disparaît de
la réponse dès qu'elle se résout. `GET /api/v1/sensors/status` dit quel
capteur est tombé et jusqu'à quand, là où `null_reasons` ne dit que ce qui
manquait sur une ligne.

Les deux appellent des traitements opposés, et c'est le seul point qui compte
vraiment ici : les alertes sont un journal qu'on n'écrase jamais, l'état des
capteurs est un présent qu'on repose à chaque tick.

Aucun test ne joint PostgreSQL : le SQL produit est compilé pour le dialecte
et lu tel quel.
"""

from __future__ import annotations

from datetime import UTC, datetime

from conftest import FakeEngine
from sqlalchemy.dialects import postgresql

from collector.sink import (
    build_alerte_insert,
    build_capteur_etat_upsert,
    read_sensor_statuses,
    to_alerts,
    to_sensor_episodes,
    to_sensor_states,
    write_alerts,
    write_sensor_episodes,
    write_sensor_states,
)

ALERTE = {
    "alert_id": "ALR-SITE002-1718458320",
    "timestamp": "2026-09-04T14:12:00",
    "site_id": "SITE002",
    "severity": "critical",
    "type": "outage",
    "message": "Risque de surcharge sur Usine Lyon Vénissieux",
    "value": 812.5,
    "threshold": 720.0,
}

CAPTEURS = {
    "SITE001": {
        "site_name": "Bureau Paris La Défense",
        "sensors": {
            "consumption": {"status": "ok", "failing_until": None},
            "electrical": {"status": "ok", "failing_until": None},
            "temperature": {
                "status": "failing",
                "failing_until": "2026-09-04T14:33:05",
            },
            "humidity": {"status": "ok", "failing_until": None},
            "network": {"status": "ok", "failing_until": None},
        },
        "overall": "degraded",
    }
}


def compiled(statement) -> str:
    """Rend le SQL tel que le dialecte PostgreSQL l'émettra."""
    return str(statement.compile(dialect=postgresql.dialect()))


class TestAlertes:
    def test_les_champs_de_la_source_arrivent_dans_les_colonnes(self) -> None:
        rows = to_alerts([ALERTE])

        assert len(rows) == 1
        row = rows[0]
        assert row["alert_id"] == "ALR-SITE002-1718458320"
        assert row["site_id"] == "SITE002"
        assert row["ts"] == datetime(2026, 9, 4, 14, 12)
        assert row["severity"] == "critical"
        # `type` est trop générique pour une colonne : la traduction est portée
        # une fois, ici, plutôt que dans chaque requête.
        assert row["type_alerte"] == "outage"
        assert row["valeur"] == 812.5
        assert row["seuil"] == 720.0

    def test_une_alerte_incomplete_est_ecartee_seule(self) -> None:
        # Le lot entier serait rejeté par la base si elle partait avec : une
        # alerte mal décrite ne doit pas emporter les autres.
        incomplete = dict(ALERTE, alert_id="ALR-X", severity=None)
        rows = to_alerts([ALERTE, incomplete])

        assert [row["alert_id"] for row in rows] == [ALERTE["alert_id"]]

    def test_un_horodatage_illisible_ecarte_l_alerte(self) -> None:
        rows = to_alerts([dict(ALERTE, timestamp="hier après-midi")])

        assert rows == []

    def test_le_journal_n_ecrase_jamais(self) -> None:
        """Le poller repasse chaque minute sur une alerte encore active.

        `alert_id` est stable côté source : sans DO NOTHING, une alerte d'une
        heure serait enregistrée soixante fois.
        """
        sql = compiled(build_alerte_insert(to_alerts([ALERTE])))

        assert "ON CONFLICT" in sql
        assert "DO NOTHING" in sql
        assert "DO UPDATE" not in sql

    def test_l_ecriture_soumet_le_lot(self) -> None:
        engine = FakeEngine()

        written = write_alerts(engine, [ALERTE], batch_size=10)

        assert written == 1
        assert len(engine.executed) == 1

    def test_un_lot_vide_ne_soumet_rien(self) -> None:
        # Une réponse vide est une réponse valable : aucune alerte en cours.
        engine = FakeEngine()

        assert write_alerts(engine, [], batch_size=10) == 0
        assert engine.executed == []


class TestEtatDesCapteurs:
    def test_l_imbrication_de_la_source_est_mise_a_plat(self) -> None:
        rows = to_sensor_states(CAPTEURS)

        assert len(rows) == 5
        assert {row["capteur"] for row in rows} == {
            "consumption",
            "electrical",
            "temperature",
            "humidity",
            "network",
        }
        assert all(row["site_id"] == "SITE001" for row in rows)
        # `overall` est recopié sur chaque ligne : la table est plate, et
        # demander « quels capteurs sont tombés » ne doit pas obliger à
        # déplier un JSON en SQL.
        assert all(row["overall"] == "degraded" for row in rows)

    def test_la_date_de_retablissement_annoncee_est_conservee(self) -> None:
        # C'est la seule information que ni `mesure` ni `null_reasons` ne
        # portent : la source annonce jusqu'à quand elle sera muette.
        rows = {row["capteur"]: row for row in to_sensor_states(CAPTEURS)}

        assert rows["temperature"]["statut"] == "failing"
        assert rows["temperature"]["failing_until"] == datetime(2026, 9, 4, 14, 33, 5)
        assert rows["network"]["failing_until"] is None

    def test_un_capteur_inconnu_est_ignore(self) -> None:
        payload = {
            "SITE001": {
                "sensors": {"pression": {"status": "ok", "failing_until": None}},
                "overall": "ok",
            }
        }

        assert to_sensor_states(payload) == []

    def test_le_present_est_repose_et_non_empile(self) -> None:
        """Cette table dit l'état ; les épisodes vivent dans `capteur_panne`.

        Empiler ici ferait une ligne par capteur et par minute pour décrire
        une panne que deux lignes suffisent à borner.
        """
        sql = compiled(build_capteur_etat_upsert(to_sensor_states(CAPTEURS)))

        assert "ON CONFLICT" in sql
        assert "DO UPDATE" in sql
        assert "failing_until" in sql

    def test_l_ecriture_soumet_le_lot(self) -> None:
        engine = FakeEngine()

        written = write_sensor_states(engine, CAPTEURS, batch_size=10)

        assert written == 5
        assert len(engine.executed) == 1

    def test_une_reponse_vide_ne_soumet_rien(self) -> None:
        engine = FakeEngine()

        assert write_sensor_states(engine, {}, batch_size=10) == 0
        assert engine.executed == []


def test_un_horodatage_deja_typé_traverse_intact() -> None:
    # httpx rend des chaînes, mais un appelant peut passer un datetime : le
    # reconvertir en chaîne pour le reparser serait une perte de fuseau.
    stamp = datetime(2026, 9, 4, 14, 12, tzinfo=UTC)
    rows = to_alerts([dict(ALERTE, timestamp=stamp)])

    assert rows[0]["ts"] is stamp


NOW = datetime(2026, 9, 4, 14, 40, tzinfo=UTC)


class TestJournalDesPannes:
    """Les transitions, là où la source ne sert qu'un présent.

    `capteur_etat` dit l'état, ce journal dit les épisodes. Et ni l'un ni
    l'autre ne double `mesure.null_reasons`, qui ne connaît que les pannes
    visibles SUR une mesure : un capteur tombé puis rétabli entre deux relevés
    n'y laisse rien.
    """

    def test_sain_puis_en_panne_ouvre_un_episode(self) -> None:
        states = to_sensor_states(CAPTEURS)
        previous = {("SITE001", "temperature"): "ok"}

        opened, closed = to_sensor_episodes(previous, states, NOW)

        assert [row["capteur"] for row in opened] == ["temperature"]
        assert opened[0]["debut_le"] == NOW
        assert opened[0]["fin_le"] is None
        assert opened[0]["failing_until"] == datetime(2026, 9, 4, 14, 33, 5)
        assert closed == []

    def test_une_panne_qui_dure_n_ouvre_rien_de_plus(self) -> None:
        # Sans cette comparaison, un capteur en panne depuis trois jours
        # ouvrirait un épisode par tick, soit plus de quatre mille.
        states = to_sensor_states(CAPTEURS)
        previous = {("SITE001", "temperature"): "failing"}

        opened, closed = to_sensor_episodes(previous, states, NOW)

        assert opened == []
        assert closed == []

    def test_en_panne_puis_sain_clot_l_episode(self) -> None:
        retabli = {
            "SITE001": {
                "sensors": {
                    "temperature": {"status": "ok", "failing_until": None},
                },
                "overall": "ok",
            }
        }
        previous = {("SITE001", "temperature"): "failing"}

        opened, closed = to_sensor_episodes(previous, to_sensor_states(retabli), NOW)

        assert opened == []
        assert closed == [{"site_id": "SITE001", "capteur": "temperature"}]

    def test_un_capteur_jamais_vu_compte_comme_sain(self) -> None:
        # Premier tick sur ce site : sa panne est bien un début.
        opened, closed = to_sensor_episodes({}, to_sensor_states(CAPTEURS), NOW)

        assert [row["capteur"] for row in opened] == ["temperature"]
        assert closed == []

    def test_l_ouverture_ignore_un_episode_deja_ouvert(self) -> None:
        """Deux processus concurrents ne doivent pas en créer deux.

        `DO NOTHING` sans cible : la clé naturelle n'est pas la seule
        contrainte à protéger, l'unicité de l'épisode ouvert compte autant.
        """
        engine = FakeEngine()
        write_sensor_episodes(engine, {}, to_sensor_states(CAPTEURS), NOW, 10)

        sql = compiled(engine.executed[0])
        assert "ON CONFLICT" in sql
        assert "DO NOTHING" in sql

    def test_rien_a_ecrire_ne_touche_pas_la_base(self) -> None:
        engine = FakeEngine()
        previous = {("SITE001", "temperature"): "failing"}

        opened, closed = write_sensor_episodes(
            engine, previous, to_sensor_states(CAPTEURS), NOW, 10
        )

        assert (opened, closed) == (0, 0)
        assert engine.executed == []

    def test_l_etat_precedent_est_lu_indexe_par_site_et_capteur(self) -> None:
        engine = FakeEngine(rows=[("SITE001", "temperature", "failing")])

        assert read_sensor_statuses(engine) == {
            ("SITE001", "temperature"): "failing"
        }

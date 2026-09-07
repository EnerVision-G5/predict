"""Import de l'historique de référence : lecture, qualification, bornes.

Aucun test ne touche la base. Ce qui est éprouvé ici est la TRADUCTION d'une
ligne de CSV en lecture de source — l'écriture, elle, est celle du collecteur
et a ses propres tests.

Le jeu de données est la seule origine possible de l'historique
d'apprentissage, puisque la source ne remonte qu'à 48 heures. Une ligne mal
traduite ne se verrait donc nulle part avant les métriques d'un entraînement,
c'est-à-dire trop tard.
"""

from __future__ import annotations

import pandas as pd
import pytest

from collector import datasets

COLUMNS = (
    "timestamp,site_id,site_type,site_name,consumption_kwh,consumption_euros,"
    "temperature_celsius,humidity_percent,solar_irradiance_wm2,hour,"
    "day_of_week,day_name,month,is_weekend,is_working_hours"
)


def csv_line(
    stamp: str = "2023-01-01 00:00:00",
    site: str = "SITE001",
    consumption: str = "89.18",
    temperature: str = "4.3",
    humidity: str = "70.6",
) -> str:
    """Une ligne au format exact des fichiers fournis avec la source."""
    return (
        f"{stamp},{site},office,Bureau Tertiaire,{consumption},13.38,"
        f"{temperature},{humidity},49.0,0,6,Sunday,1,1,0"
    )


def write_dataset(directory, name: str, lines: list[str]) -> None:
    """Dépose un fichier par site dans le répertoire donné."""
    path = directory / name
    path.write_text("\n".join([COLUMNS, *lines]) + "\n", encoding="utf-8")


def read_one(directory, line: str) -> dict:
    """Retourne la lecture produite par une ligne unique."""
    write_dataset(directory, "SITE001.csv", [line])
    frame = datasets.read_file(directory / "SITE001.csv")
    return next(iter(datasets.to_readings(frame)))


def test_une_ligne_complete_donne_une_lecture_good(tmp_path) -> None:
    reading = read_one(tmp_path, csv_line())
    assert reading["data_quality"] == "good"
    assert reading["null_reasons"] == []


def test_la_consommation_horaire_sert_les_deux_champs(tmp_path) -> None:
    # L'énergie d'une heure en kWh est numériquement la puissance moyenne de
    # cette heure en kW. La source elle-même sert les deux champs égaux.
    reading = read_one(tmp_path, csv_line(consumption="89.18"))
    assert reading["consumption_kw"] == 89.18
    assert reading["consumption_kwh"] == 89.18


def test_les_grandeurs_electriques_restent_nulles(tmp_path) -> None:
    # Le jeu de données ne les porte pas. Les déduire d'une consommation
    # horaire inventerait trois grandeurs à partir d'une seule.
    reading = read_one(tmp_path, csv_line())
    assert reading["voltage_v"] is None
    assert reading["current_a"] is None
    assert reading["power_factor"] is None


def test_une_consommation_absente_degrade_la_ligne(tmp_path) -> None:
    # C'est la consommation que le modèle apprend : sans elle l'heure est
    # inapprenable, alors qu'une température manquante ne fait que gêner.
    reading = read_one(tmp_path, csv_line(consumption=""))
    assert reading["data_quality"] == "degraded"
    assert reading["null_reasons"] == ["consumption_sensor_failure"]


def test_une_temperature_absente_rend_la_ligne_partielle(tmp_path) -> None:
    reading = read_one(tmp_path, csv_line(temperature=""))
    assert reading["data_quality"] == "partial"
    assert reading["null_reasons"] == ["temperature_sensor_failure"]


def test_tout_absent_vaut_une_coupure_reseau(tmp_path) -> None:
    # Et non trois pannes simultanées : trois capteurs qui tombent à la même
    # seconde décrivent le réseau, pas les capteurs. C'est la qualification
    # que la source emploie, et l'analyse de fiabilité n'a ainsi qu'un seul
    # vocabulaire à connaître.
    reading = read_one(
        tmp_path, csv_line(consumption="", temperature="", humidity="")
    )
    assert reading["data_quality"] == "critical"
    assert reading["null_reasons"] == ["network_loss"]


def test_les_horodatages_naifs_sont_lus_en_utc(tmp_path) -> None:
    # Les lire dans le fuseau du serveur décalerait tout l'historique d'une à
    # deux heures selon la saison, et ferait apprendre au modèle des journées
    # de travail commençant à 7 h.
    reading = read_one(tmp_path, csv_line(stamp="2023-06-15 08:00:00"))
    stamp = pd.Timestamp(reading["timestamp"])
    assert stamp.tzinfo is not None
    assert stamp.tz_convert("UTC").hour == 8


def test_le_pas_horaire_est_conserve(tmp_path) -> None:
    # Une ligne par heure, et non soixante copies : l'ETL calcule son taux
    # d'imputation sur les lignes présentes, donc une heure à une seule ligne
    # non imputée vaut imputed_ratio = 0.
    write_dataset(
        tmp_path,
        "SITE001.csv",
        [
            csv_line(stamp="2023-01-01 00:00:00"),
            csv_line(stamp="2023-01-01 01:00:00"),
        ],
    )
    frame = datasets.read_file(tmp_path / "SITE001.csv")
    readings = list(datasets.to_readings(frame))
    assert len(readings) == 2
    stamps = [pd.Timestamp(r["timestamp"]) for r in readings]
    assert stamps[1] - stamps[0] == pd.Timedelta(hours=1)


def test_le_fichier_combine_n_est_pas_relu(tmp_path) -> None:
    # Il porte exactement les mêmes lignes que les fichiers par site : le lire
    # en plus doublerait le travail pour un résultat identique.
    write_dataset(tmp_path, "SITE001.csv", [csv_line()])
    write_dataset(tmp_path, "all_sites_combined.csv", [csv_line()])
    assert [p.name for p in datasets.find_files(tmp_path)] == ["SITE001.csv"]


def test_un_repertoire_sans_fichier_est_une_erreur(tmp_path) -> None:
    # Et non un import de zéro ligne : la commande aurait l'air d'avoir
    # réussi, et le défaut ne se verrait qu'à l'entraînement.
    with pytest.raises(datasets.DatasetError):
        datasets.find_files(tmp_path)


def test_un_repertoire_absent_est_une_erreur(tmp_path) -> None:
    with pytest.raises(datasets.DatasetError):
        datasets.find_files(tmp_path / "nulle-part")


def test_une_colonne_attendue_absente_est_nommee(tmp_path) -> None:
    path = tmp_path / "SITE001.csv"
    path.write_text(
        "timestamp,site_id\n2023-01-01 00:00:00,SITE001\n", encoding="utf-8"
    )
    with pytest.raises(datasets.DatasetError) as erreur:
        datasets.read_file(path)
    assert "consumption_kwh" in str(erreur.value)

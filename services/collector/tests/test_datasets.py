from __future__ import annotations

import pandas as pd
import pyarrow.fs
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
    return (
        f"{stamp},{site},office,Bureau Tertiaire,{consumption},13.38,"
        f"{temperature},{humidity},49.0,0,6,Sunday,1,1,0"
    )


def write_dataset(directory, name: str, lines: list[str]) -> None:
    path = directory / name
    path.write_text("\n".join([COLUMNS, *lines]) + "\n", encoding="utf-8")


def read_one(directory, line: str) -> dict:
    write_dataset(directory, "SITE001.csv", [line])
    frame = datasets.read_file(directory / "SITE001.csv")
    return next(iter(datasets.to_readings(frame)))


def test_une_ligne_complete_donne_une_lecture_good(tmp_path) -> None:
    reading = read_one(tmp_path, csv_line())
    assert reading["data_quality"] == "good"
    assert reading["null_reasons"] == []


def test_la_consommation_horaire_sert_les_deux_champs(tmp_path) -> None:
    reading = read_one(tmp_path, csv_line(consumption="89.18"))
    assert reading["consumption_kw"] == 89.18
    assert reading["consumption_kwh"] == 89.18


def test_les_grandeurs_electriques_restent_nulles(tmp_path) -> None:
    reading = read_one(tmp_path, csv_line())
    assert reading["voltage_v"] is None
    assert reading["current_a"] is None
    assert reading["power_factor"] is None


def test_une_consommation_absente_degrade_la_ligne(tmp_path) -> None:
    reading = read_one(tmp_path, csv_line(consumption=""))
    assert reading["data_quality"] == "degraded"
    assert reading["null_reasons"] == ["consumption_sensor_failure"]


def test_une_temperature_absente_rend_la_ligne_partielle(tmp_path) -> None:
    reading = read_one(tmp_path, csv_line(temperature=""))
    assert reading["data_quality"] == "partial"
    assert reading["null_reasons"] == ["temperature_sensor_failure"]


def test_tout_absent_vaut_une_coupure_reseau(tmp_path) -> None:
    reading = read_one(
        tmp_path, csv_line(consumption="", temperature="", humidity="")
    )
    assert reading["data_quality"] == "critical"
    assert reading["null_reasons"] == ["network_loss"]


def test_les_horodatages_naifs_sont_lus_en_utc(tmp_path) -> None:
    reading = read_one(tmp_path, csv_line(stamp="2023-06-15 08:00:00"))
    stamp = pd.Timestamp(reading["timestamp"])
    assert stamp.tzinfo is not None
    assert stamp.tz_convert("UTC").hour == 8


def test_le_pas_horaire_est_conserve(tmp_path) -> None:
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
    write_dataset(tmp_path, "SITE001.csv", [csv_line()])
    write_dataset(tmp_path, "all_sites_combined.csv", [csv_line()])
    found = datasets.find_files(tmp_path)
    assert [datasets.base_name(uri) for uri in found] == ["SITE001.csv"]


def test_un_repertoire_sans_fichier_est_une_erreur(tmp_path) -> None:
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


class TestStorageRoot:
    def test_un_chemin_local_est_une_racine_valide(self, tmp_path) -> None:
        write_dataset(tmp_path, "SITE001.csv", [csv_line()])
        assert len(datasets.find_files(tmp_path)) == 1

    def test_une_racine_au_schema_inconnu_est_une_erreur_de_jeu_de_donnees(
        self,
    ) -> None:
        with pytest.raises(datasets.DatasetError) as erreur:
            datasets.find_files("nulle-part://enervision-datasets")
        assert "nulle-part://" in str(erreur.value)

    def test_un_fichier_absent_est_nomme(self, tmp_path) -> None:
        with pytest.raises(datasets.DatasetError) as erreur:
            datasets.read_file(tmp_path / "SITE404.csv")
        assert "SITE404.csv" in str(erreur.value)


class TestBaseName:
    def test_une_uri_s3_rend_son_dernier_segment(self) -> None:
        uri = "s3://enervision-datasets/SITE001.csv"
        assert datasets.base_name(uri) == "SITE001.csv"

    def test_un_chemin_windows_rend_son_dernier_segment(self) -> None:
        assert datasets.base_name(r"D:\jeux\SITE001.csv") == "SITE001.csv"


class TestUriRendues:
    def liste_factice(self, monkeypatch, noms: list[str]) -> None:
        class Entree:
            def __init__(self, nom: str) -> None:
                self.base_name = nom
                self.path = f"enervision-datasets/{nom}"
                self.type = pyarrow.fs.FileType.File

        class Systeme:
            def get_file_info(self, selector):
                return [Entree(nom) for nom in noms]

        monkeypatch.setattr(
            datasets.io, "resolve", lambda uri: (Systeme(), "enervision-datasets")
        )

    def test_le_schema_s3_survit_a_la_liste(self, monkeypatch) -> None:
        self.liste_factice(monkeypatch, ["SITE001.csv", "SITE002.csv"])
        assert datasets.find_files("s3://enervision-datasets") == [
            "s3://enervision-datasets/SITE001.csv",
            "s3://enervision-datasets/SITE002.csv",
        ]

    def test_le_motif_filtre_toujours(self, monkeypatch) -> None:
        self.liste_factice(monkeypatch, ["SITE001.csv", "all_sites_combined.csv"])
        found = datasets.find_files("s3://enervision-datasets")
        assert found == ["s3://enervision-datasets/SITE001.csv"]

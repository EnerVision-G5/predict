"""Surveillance de l'écart : fenêtre, seuil, verdict, publication.

Aucun test ne joint MLflow. Ce qui compte ici est la décision : à partir de
quoi on conclut à une dérive, et ce qu'on répond quand on ne peut pas
conclure. Une surveillance qui crierait au loup sur trois heures de mesures ou
qui se tairait faute de référence serait pire qu'absente — on cesserait de la
regarder.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from predict_common.schemas import feature_columns
from training.drift import (
    DECISION_METRIC,
    EXIT_DRIFTED,
    EXIT_OK,
    VERDICT_DRIFTED,
    VERDICT_STABLE,
    VERDICT_UNDECIDED,
    DriftReport,
    DriftSettings,
    MonitoringError,
    SiteMeasure,
    measure,
    measure_sites,
    window,
)

LAGS = (1, 24)
ROLLING = 2
COLUMNS = feature_columns(LAGS, ROLLING)

SETTINGS = DriftSettings(
    experiment="enervision-drift",
    window_days=7,
    alert_ratio=1.5,
    min_rows=24,
    min_rows_per_site=24,
    registered_model="enervision_xgboost",
    alias="champion",
)


def report(
    mae: float,
    baseline_mae: float | None = 10.0,
    rows: int = 168,
) -> DriftReport:
    """Construit un rapport dont seules l'erreur et la référence importent."""
    return DriftReport(
        metrics={"mae": mae, "rmse": mae * 1.3, "r2": 0.9},
        baseline={"mae": baseline_mae} if baseline_mae is not None else {},
        rows=rows,
        version="3",
        window="2026-09-01/2026-09-07",
    )


class TestWindow:
    """La fenêtre glisse toute seule quand l'ordonnanceur ne la dit pas."""

    def test_without_bounds_it_covers_the_last_days(self) -> None:
        first, last = window(None, None, window_days=7)
        assert (last - first).days == 6
        assert last == date.today()

    def test_two_bounds_are_kept_as_given(self) -> None:
        first, last = window(date(2026, 9, 1), date(2026, 9, 7), window_days=7)
        assert (first, last) == (date(2026, 9, 1), date(2026, 9, 7))

    def test_a_lone_start_evaluates_that_day(self) -> None:
        assert window(date(2026, 9, 1), None, window_days=7) == (
            date(2026, 9, 1),
            date(2026, 9, 1),
        )

    def test_a_lone_end_counts_backwards(self) -> None:
        first, last = window(None, date(2026, 9, 7), window_days=3)
        assert (first, last) == (date(2026, 9, 5), date(2026, 9, 7))

    def test_an_inverted_window_is_refused(self) -> None:
        with pytest.raises(MonitoringError, match="Fenêtre vide"):
            window(date(2026, 9, 7), date(2026, 9, 1), window_days=7)

    def test_a_null_window_is_refused(self) -> None:
        with pytest.raises(MonitoringError):
            window(None, None, window_days=0)


class TestVerdict:
    """Le seuil est un rapport à l'erreur d'entraînement, pas des kilowatts."""

    def test_an_error_close_to_the_baseline_is_stable(self) -> None:
        assert report(mae=11.0).verdict(SETTINGS) == "stable"

    def test_the_threshold_itself_is_not_a_drift(self) -> None:
        # Strictement supérieur : un modèle qui tient exactement le seuil n'a
        # pas dérivé, il est à la limite. Alerter là-dessus rendrait le seuil
        # impossible à régler.
        assert report(mae=15.0).ratio == pytest.approx(1.5)
        assert report(mae=15.0).verdict(SETTINGS) == "stable"

    def test_beyond_the_threshold_is_a_drift(self) -> None:
        assert report(mae=16.0).verdict(SETTINGS) == "dérive"

    def test_an_error_smaller_than_the_baseline_is_stable(self) -> None:
        # Un modèle qui fait mieux que son test n'est pas une anomalie : la
        # semaine évaluée était plus facile, c'est tout.
        assert report(mae=4.0).verdict(SETTINGS) == "stable"

    def test_the_ratio_is_relative_and_not_absolute(self) -> None:
        # 20 kW d'écart sur un bureau de 200 kW et sur une usine de 1000 ne
        # disent pas la même chose : c'est tout l'objet du rapport.
        petit = report(mae=20.0, baseline_mae=5.0)
        grand = report(mae=20.0, baseline_mae=40.0)
        assert petit.verdict(SETTINGS) == "dérive"
        assert grand.verdict(SETTINGS) == "stable"


class TestQuandOnNePeutPasConclure:
    """Se taire est une réponse, mais elle doit être dite."""

    def test_too_few_hours_gives_no_verdict(self) -> None:
        # Une poignée d'heures suffit à faire dire n'importe quoi à une
        # moyenne : mieux vaut l'avouer qu'alerter au hasard.
        assert report(mae=100.0, rows=5).verdict(SETTINGS) == "indécis"

    def test_the_minimum_is_inclusive(self) -> None:
        assert report(mae=11.0, rows=SETTINGS.min_rows).verdict(SETTINGS) == "stable"

    def test_without_a_baseline_there_is_no_ratio(self) -> None:
        # Inventer un rapport de 1 laisserait croire que tout va bien.
        assert report(mae=50.0, baseline_mae=None).ratio is None

    def test_without_a_baseline_the_verdict_says_so(self) -> None:
        sans = report(mae=50.0, baseline_mae=None)
        assert sans.verdict(SETTINGS) == "sans référence"

    def test_a_null_baseline_is_treated_as_absent(self) -> None:
        # Diviser par zéro donnerait un infini, donc une alerte permanente.
        assert report(mae=1.0, baseline_mae=0.0).ratio is None


class TestMesure:
    """L'écart est mesuré à un pas, comme à l'entraînement."""

    def test_a_perfect_model_has_no_error(self) -> None:
        frame = _features(48)

        class Parfait:
            def predict(self, explanatory: pd.DataFrame):
                return frame["consumption_kw"].to_numpy()

        assert measure(Parfait(), frame, COLUMNS)[DECISION_METRIC] == 0.0

    def test_the_model_only_sees_the_declared_columns(self) -> None:
        # Les décalages viennent des mesures réelles : c'est ce qui rend la
        # mesure comparable à celle de l'entraînement, et non à l'erreur
        # cumulée du service sur 48 heures.
        frame = _features(48)
        vues: list[list[str]] = []

        class Espion:
            def predict(self, explanatory: pd.DataFrame):
                vues.append(list(explanatory.columns))
                return [0.0] * len(explanatory)

        measure(Espion(), frame, COLUMNS)
        assert vues[0] == list(COLUMNS)

    def test_a_constant_model_shows_its_error(self) -> None:
        frame = _features(48)

        class Constant:
            def predict(self, explanatory: pd.DataFrame):
                return [0.0] * len(explanatory)

        mesure = measure(Constant(), frame, COLUMNS)
        assert mesure[DECISION_METRIC] == pytest.approx(
            frame["consumption_kw"].abs().mean()
        )


class TestCodesDeSortie:
    """Un ordonnanceur décide sur le code, pas sur le journal."""

    def test_a_drift_is_not_a_failure(self) -> None:
        # Les confondre ferait chercher un problème d'infrastructure là où le
        # modèle a simplement vieilli.
        assert EXIT_DRIFTED != EXIT_OK
        assert EXIT_DRIFTED == 2

    def test_settings_are_read_from_the_configuration(self) -> None:
        from predict_common.config import Config

        config = Config(
            values={
                "monitoring": {
                    "experiment": "enervision-drift",
                    "window_days": 7,
                    "mae_alert_ratio": 1.5,
                    "min_rows": 168,
                    "min_rows_per_site": 24,
                },
                "training": {"registered_model": "enervision_xgboost"},
            }
        )
        settings = DriftSettings.from_config(config)
        assert settings.alert_ratio == 1.5
        # C'est le modèle SERVI qui est surveillé, pas le dernier entraîné :
        # surveiller un challenger que personne n'utilise ne dirait rien.
        assert settings.alias == "champion"


def _features(hours: int) -> pd.DataFrame:
    """Partition de variables, telle que l'ETL la publie."""
    stamps = pd.date_range("2026-09-01T00:00:00Z", periods=hours, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "ts": stamps,
            "site_id": "SITE001",
            "consumption_kw": [50.0 + index % 12 for index in range(hours)],
            "hour": stamps.hour,
            "day_of_week": stamps.dayofweek,
            "is_weekend": (stamps.dayofweek >= 5).astype(int),
            "temperature_celsius": 20.0,
            "lag_1h": 50.0,
            "lag_24h": 50.0,
            "roll_mean_2h": 50.0,
            "data_quality": "good",
            "imputed_ratio": 0.0,
        }
    )


def test_the_default_window_is_a_week() -> None:
    # Assez pour que l'erreur ne dépende pas d'une journée atypique, assez
    # court pour qu'une dérive installée depuis trois jours se voie encore.
    assert SETTINGS.window_days == 7
    first, last = window(None, None, SETTINGS.window_days)
    assert last - first == timedelta(days=6)


def _two_sites(hours: int) -> pd.DataFrame:
    """Deux sites de tailles très différentes, dans une même partition.

    C'est la situation que la moyenne d'ensemble masque : l'usine pèse dix
    fois le bureau, et son erreur décide seule de la MAE globale.
    """
    bureau = _features(hours)
    usine = _features(hours).assign(
        site_id="SITE002", consumption_kw=lambda frame: frame["consumption_kw"] * 10
    )
    return pd.concat([bureau, usine], ignore_index=True)


class TestDecoupageParSite:
    """Une MAE d'ensemble dit ce que le parc coûte, pas où l'erreur est."""

    def test_each_site_gets_its_own_measure(self) -> None:
        frame = _two_sites(48)

        class Constant:
            def predict(self, explanatory: pd.DataFrame):
                return [0.0] * len(explanatory)

        sites = measure_sites(Constant(), frame, COLUMNS)
        assert [site.site_id for site in sites] == ["SITE001", "SITE002"]
        assert [site.rows for site in sites] == [48, 48]
        # Le site dix fois plus gros porte une erreur dix fois plus grande :
        # c'est exactement ce que la moyenne d'ensemble confondait.
        petit, grand = sites
        assert grand.metrics[DECISION_METRIC] == pytest.approx(
            petit.metrics[DECISION_METRIC] * 10
        )

    def test_a_partition_without_sites_still_measures_the_whole(self) -> None:
        # Une mesure d'ensemble reste valable sans détail : l'absence de
        # colonne n'est pas une raison de ne rien mesurer.
        frame = _features(48).drop(columns=["site_id"])

        class Constant:
            def predict(self, explanatory: pd.DataFrame):
                return [0.0] * len(explanatory)

        assert measure_sites(Constant(), frame, COLUMNS) == ()

    def test_a_single_drifting_site_is_named(self) -> None:
        # La raison d'être de la découpe : un petit site qui double son erreur
        # disparaît dans la moyenne d'une usine dix fois plus grande.
        rapport = DriftReport(
            metrics={"mae": 10.0},
            baseline={"mae": 10.0},
            rows=336,
            version="3",
            window="2026-09-01/2026-09-07",
            sites=(
                SiteMeasure("SITE001", {"mae": 100.0}, rows=48),
                SiteMeasure("SITE002", {"mae": 9.0}, rows=48),
            ),
        )
        assert rapport.verdict(SETTINGS) == VERDICT_STABLE
        assert rapport.drifted_sites(SETTINGS) == ("SITE001",)

    def test_a_site_with_too_few_hours_gives_no_verdict(self) -> None:
        # Sans une journée entière, le cycle jour/nuit décide seul de la
        # moyenne du site.
        court = SiteMeasure("SITE001", {"mae": 100.0}, rows=3)
        assert court.verdict(SETTINGS, {"mae": 10.0}) == VERDICT_UNDECIDED

    def test_a_site_beyond_the_threshold_drifts(self) -> None:
        eleve = SiteMeasure("SITE001", {"mae": 16.0}, rows=48)
        assert eleve.verdict(SETTINGS, {"mae": 10.0}) == VERDICT_DRIFTED

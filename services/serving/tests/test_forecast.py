"""Prévision multi-pas : historique lu, récurrence, fuites évitées.

Le service ne recalcule pas les variables depuis les mesures brutes — ce
serait refaire le travail de l'ETL avec un second jeu de règles. Il lit la
dernière partition publiée. Les tests fixent donc ce qu'il fait de cet
historique : où il puise ses décalages, comment il enchaîne les heures, et à
quel moment il refuse de continuer plutôt que d'inventer.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from predict_common import io
from predict_common.paths import features_partition
from predict_common.schemas import (
    add_derived_calendar,
    feature_columns,
    features_arrow_schema,
)
from serving.forecast import (
    CONFIDENCE_Z,
    ForecastSpec,
    NoHistory,
    build_row,
    confidence_band,
    horizon_stamps,
    predict_series,
    read_history,
    site_history,
)

LAGS = (1, 24)
WINDOW = 2
COLUMNS = feature_columns(LAGS, WINDOW)
TODAY = date(2026, 9, 10)


def spec_for(root: Path) -> ForecastSpec:
    """Réglages du service, pointés sur un stockage jetable."""
    return ForecastSpec(
        root=str(root),
        feature_version="v1",
        lag_hours=LAGS,
        rolling_window_h=WINDOW,
        lookback_days=3,
    )


def features(day: date, site_id: str = "SITE001") -> pd.DataFrame:
    """Partition de variables d'une journée, telle que l'ETL la publie."""
    stamps = pd.date_range(
        f"{day.isoformat()}T00:00:00Z", periods=24, freq="h", tz="UTC"
    )
    return pd.DataFrame(
        {
            "ts": stamps,
            "site_id": site_id,
            "consumption_kw": [50.0 + hour for hour in range(24)],
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


def seed(root: Path, days: int = 3, site_id: str = "SITE001") -> None:
    """Publie les dernières partitions de variables."""
    for offset in range(days):
        day = TODAY - timedelta(days=offset)
        io.write_frame(
            features(day, site_id=site_id),
            features_partition(str(root), "v1", day),
            schema=features_arrow_schema(LAGS, WINDOW),
        )


def series(hours: int = 48) -> pd.Series:
    """Série horaire observée, indexée sur le temps."""
    index = pd.date_range("2026-09-08T00:00:00Z", periods=hours, freq="h", tz="UTC")
    return pd.Series([50.0 + index_ % 24 for index_ in range(hours)], index=index)


def test_read_history_gathers_the_recent_partitions(tmp_path: Path) -> None:
    seed(tmp_path)
    assert len(read_history(spec_for(tmp_path), today=TODAY)) == 72


def test_read_history_tolerates_a_missing_night(tmp_path: Path) -> None:
    # L'ETL de la nuit peut ne pas avoir tourné : le service sert alors sur
    # l'historique de la veille plutôt que de refuser.
    seed(tmp_path, days=1)
    assert len(read_history(spec_for(tmp_path), today=TODAY)) == 24


def test_read_history_returns_nothing_when_no_partition_exists(
    tmp_path: Path,
) -> None:
    assert read_history(spec_for(tmp_path), today=TODAY).empty


def test_site_history_indexes_the_series_on_time(tmp_path: Path) -> None:
    seed(tmp_path)
    frame = read_history(spec_for(tmp_path), today=TODAY)
    history = site_history(frame, "SITE001", spec_for(tmp_path))
    assert isinstance(history.index, pd.DatetimeIndex)
    assert len(history) == 72


def test_site_history_refuses_an_unknown_site(tmp_path: Path) -> None:
    seed(tmp_path)
    frame = read_history(spec_for(tmp_path), today=TODAY)
    with pytest.raises(NoHistory):
        site_history(frame, "SITE404", spec_for(tmp_path))


def test_site_history_refuses_an_empty_read(tmp_path: Path) -> None:
    # Le tableau est alors vide et sans colonnes : le cas doit donner un refus
    # explicite, pas une erreur de clé au milieu d'une requête.
    with pytest.raises(NoHistory):
        site_history(pd.DataFrame(), "SITE001", spec_for(tmp_path))


def test_site_history_leaves_a_gap_as_a_gap(tmp_path: Path) -> None:
    # L'imputation appartient à l'ETL : la refaire ici en donnerait deux
    # versions, qui finiraient par diverger.
    seed(tmp_path)
    frame = read_history(spec_for(tmp_path), today=TODAY)
    without = frame[frame["ts"].dt.hour != 5]
    history = site_history(without, "SITE001", spec_for(tmp_path))
    assert history.isna().any()


def test_horizon_stamps_follow_the_last_observed_hour() -> None:
    stamps = horizon_stamps(series(), 3)
    assert stamps[0] == series().index.max() + pd.Timedelta(hours=1)
    assert len(stamps) == 3


class TestBuildRow:
    """Une ligne de variables, ou rien — mais jamais une valeur inventée."""

    def test_the_calendar_columns_are_integers(self, tmp_path: Path) -> None:
        # La signature du modèle les déclare `integer` : MLflow refuse une
        # conversion float64 vers int32 qu'il ne peut pas garantir sans perte.
        row = build_row(series(), horizon_stamps(series(), 1)[0], spec_for(tmp_path))
        assert isinstance(row["hour"], int)
        assert isinstance(row["day_of_week"], int)

    def test_the_lags_are_read_from_the_history(self, tmp_path: Path) -> None:
        history = series()
        stamp = horizon_stamps(history, 1)[0]
        row = build_row(history, stamp, spec_for(tmp_path))
        assert row["lag_1h"] == history.loc[stamp - pd.Timedelta(hours=1)]
        assert row["lag_24h"] == history.loc[stamp - pd.Timedelta(hours=24)]

    def test_the_derived_calendar_matches_what_training_computes(
        self, tmp_path: Path
    ) -> None:
        # LE test de ce ticket. L'entraînement dérive ces colonnes d'une
        # partition, le service les calcule pour une heure à venir : si les
        # deux chemins divergeaient, le modèle recevrait en production autre
        # chose que ce sur quoi il a appris, et rien ne le signalerait — ni la
        # signature MLflow, qui ne voit que des noms, ni la surveillance de
        # dérive, qui passe par le même chemin que l'entraînement.
        stamp = horizon_stamps(series(), 1)[0]
        row = build_row(series(), stamp, spec_for(tmp_path))
        expected = add_derived_calendar(pd.DataFrame({"hour": [int(stamp.hour)]}))
        assert row["hour_sin"] == pytest.approx(expected["hour_sin"].iloc[0])
        assert row["hour_cos"] == pytest.approx(expected["hour_cos"].iloc[0])

    def test_the_derived_calendar_is_presented_as_floats(
        self, tmp_path: Path
    ) -> None:
        # À la différence des trois autres variables calendaires : la signature
        # les déclare `double`, et un entier y serait converti en silence.
        row = build_row(series(), horizon_stamps(series(), 1)[0], spec_for(tmp_path))
        assert isinstance(row["hour_sin"], float)
        assert isinstance(row["hour_cos"], float)

    def test_the_future_temperature_is_not_presented(self, tmp_path: Path) -> None:
        # Le modèle ne l'attend plus : elle a quitté les variables
        # explicatives. La présenter vide reviendrait à faire emprunter à
        # chaque prédiction la branche par défaut des arbres qui la testent.
        row = build_row(series(), horizon_stamps(series(), 1)[0], spec_for(tmp_path))
        assert "temperature_celsius" not in row

    def test_a_missing_lag_gives_no_row(self, tmp_path: Path) -> None:
        # Inventer sa valeur donnerait une prévision dont rien ne dirait
        # qu'elle repose sur du vide.
        short = series(hours=2)
        assert build_row(short, horizon_stamps(short, 1)[0], spec_for(tmp_path)) is None

    def test_the_rolling_window_stops_before_the_predicted_hour(
        self, tmp_path: Path
    ) -> None:
        # C'est la même règle qu'à l'entraînement, où la moyenne glissante est
        # décalée d'un pas : la rompre présenterait au modèle une variable
        # qu'il n'a jamais vue sous cette forme.
        history = series()
        stamp = horizon_stamps(history, 1)[0]
        row = build_row(history, stamp, spec_for(tmp_path))
        expected = history.iloc[-WINDOW:].mean()
        assert row["roll_mean_2h"] == pytest.approx(expected)


class TestPredictSeries:
    """Chaque heure prédite nourrit la suivante."""

    def test_the_horizon_is_served_in_full(self, tmp_path: Path) -> None:
        points = predict_series(
            lambda frame: [42.0], series(), 6, spec_for(tmp_path), COLUMNS
        )
        assert len(points) == 6

    def test_a_prediction_becomes_the_next_lag(self, tmp_path: Path) -> None:
        seen: list[float] = []

        def predict(frame: pd.DataFrame) -> list[float]:
            seen.append(float(frame["lag_1h"].iloc[0]))
            return [99.0]

        predict_series(predict, series(), 3, spec_for(tmp_path), COLUMNS)
        # La première heure lit l'historique observé, les suivantes lisent la
        # prévision précédente.
        assert seen[1] == 99.0
        assert seen[2] == 99.0

    def test_the_columns_are_presented_in_the_signature_order(
        self, tmp_path: Path
    ) -> None:
        seen: list[list[str]] = []

        def predict(frame: pd.DataFrame) -> list[float]:
            seen.append(list(frame.columns))
            return [42.0]

        predict_series(predict, series(), 1, spec_for(tmp_path), COLUMNS)
        assert seen[0] == list(COLUMNS)

    def test_the_horizon_stops_when_history_runs_out(self, tmp_path: Path) -> None:
        # Un historique trop court ne donne aucun point plutôt que des points
        # bâtis sur rien.
        points = predict_series(
            lambda frame: [42.0], series(hours=2), 6, spec_for(tmp_path), COLUMNS
        )
        assert points == []


class TestConfidenceBand:
    """La bande dit ce que vaut la prévision, ou ne dit rien."""

    def test_no_spread_gives_no_bounds(self) -> None:
        # Le contrat les prévoit optionnelles : mieux vaut pas d'intervalle
        # qu'un intervalle qui ne repose sur rien.
        assert confidence_band(50.0, 1, None) == (None, None)

    def test_the_band_is_centred_on_the_prediction(self) -> None:
        lower, upper = confidence_band(50.0, 1, 2.0)
        assert (lower + upper) / 2 == pytest.approx(50.0)

    def test_the_first_step_is_the_plain_quantile(self) -> None:
        lower, upper = confidence_band(50.0, 1, 2.0)
        assert upper - 50.0 == pytest.approx(CONFIDENCE_Z * 2.0)

    def test_the_band_grows_as_the_square_root_of_the_step(self) -> None:
        # Les erreurs de deux pas successifs s'additionnent en variance, pas
        # en écart-type : une croissance linéaire donnerait à l'horizon 48 une
        # bande quatre fois trop large, que personne ne lirait.
        first = confidence_band(50.0, 1, 2.0)
        fourth = confidence_band(50.0, 4, 2.0)
        assert (fourth[1] - fourth[0]) == pytest.approx(2 * (first[1] - first[0]))

    def test_the_lower_bound_is_not_clipped_at_zero(self) -> None:
        # Le schéma de la couche brute n'interdit pas un soutirage négatif, et
        # rogner la borne masquerait un modèle qui prédit une aberration.
        lower, _ = confidence_band(1.0, 1, 10.0)
        assert lower < 0.0

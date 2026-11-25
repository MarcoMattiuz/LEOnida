"""Posizioni terrestri fisse, indipendenti dall'avvio di Mininet e Docker."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import cached_property
from math import isfinite

import numpy as np
from skyfield.api import wgs84
from skyfield.toposlib import GeographicPosition


@dataclass(frozen=True)
class GeoPosition:
    """Coordinate WGS84: gradi decimali e quota sopra l'ellissoide in km."""

    latitude_deg: float
    longitude_deg: float
    altitude_km: float

    def __post_init__(self) -> None:
        if not all(isfinite(value) for value in (
            self.latitude_deg, self.longitude_deg, self.altitude_km
        )):
            raise ValueError("Le coordinate devono essere finite.")
        if not -90.0 <= self.latitude_deg <= 90.0:
            raise ValueError("La latitudine deve essere compresa tra -90 e 90 gradi.")
        if not -180.0 <= self.longitude_deg <= 180.0:
            raise ValueError("La longitudine deve essere compresa tra -180 e 180 gradi.")

    @cached_property
    def skyfield_position(self) -> GeographicPosition:
        return wgs84.latlon(
            self.latitude_deg,
            self.longitude_deg,
            elevation_m=self.altitude_km * 1000.0,
        )

    def at(self, time=None) -> np.ndarray:
        """Coordinate ECEF/ITRS in km, costanti per ogni istante."""
        return self.skyfield_position.itrs_xyz.km.copy()

    def subpoint(self, time=None) -> tuple[float, float, float]:
        """Latitudine, longitudine e quota; stessa interfaccia del satellite."""
        return self.latitude_deg, self.longitude_deg, self.altitude_km

    def to_dict(self) -> dict:
        return asdict(self)

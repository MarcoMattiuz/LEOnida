"""Dati orbitali OMM, propagazione Skyfield e nodo satellite OpenFlow."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import cached_property, lru_cache

import numpy as np
from mininet.node import OVSSwitch  # type: ignore
from skyfield.api import EarthSatellite, load, wgs84
from skyfield.framelib import itrs
from skyfield.timelib import Time, Timescale


@lru_cache(maxsize=1)
def _timescale() -> Timescale:
    """Condivide le tabelle temporali incluse in Skyfield, senza download."""
    return load.timescale(builtin=True)


@dataclass(frozen=True)
class Satellite:
    """Un record GP/OMM; EPOCH resta l'epoca originale degli elementi.

    Skyfield interpreta i campi tramite EarthSatellite.from_omm(). L'istante
    comune della simulazione sara' passato ad at() e subpoint() dal futuro
    gestore degli scenari, senza sostituire le epoche dei singoli satelliti.
    """

    OBJECT_NAME: str
    OBJECT_ID: str
    EPOCH: str
    MEAN_MOTION: float
    ECCENTRICITY: float
    INCLINATION: float
    RA_OF_ASC_NODE: float
    ARG_OF_PERICENTER: float
    MEAN_ANOMALY: float
    NORAD_CAT_ID: int
    BSTAR: float
    MEAN_MOTION_DOT: float
    MEAN_MOTION_DDOT: float
    EPHEMERIS_TYPE: int = 0
    CLASSIFICATION_TYPE: str = "U"
    ELEMENT_SET_NO: int = 999
    REV_AT_EPOCH: int = 0

    def to_omm(self) -> dict:
        """Restituisce solo i campi OMM, senza gli oggetti Skyfield derivati."""
        return asdict(self)

    @cached_property
    def skyfield_satellite(self) -> EarthSatellite:
        """Propagatore SGP4 costruito da Skyfield a partire dal record OMM."""
        return EarthSatellite.from_omm(_timescale(), self.to_omm())

    @staticmethod
    def to_datetime(time: Time | datetime) -> datetime:
        """Converte un singolo istante Skyfield o datetime con timezone in UTC."""
        if isinstance(time, Time):
            if time.shape:
                raise ValueError("Specificare un singolo istante, non un array di tempi.")
            return time.utc_datetime()
        if not isinstance(time, datetime):
            raise TypeError("L'istante deve essere un Time Skyfield o un datetime.")
        if time.tzinfo is None or time.utcoffset() is None:
            raise ValueError("Il datetime deve avere la timezone (es. timezone.utc).")
        return time.astimezone(timezone.utc)

    def _position_at(self, time: Time | datetime):
        if isinstance(time, Time):
            if time.shape:
                raise ValueError("Specificare un singolo istante, non un array di tempi.")
        else:
            time = _timescale().from_datetime(self.to_datetime(time))
        position = self.skyfield_satellite.at(time)
        if position.message is not None:
            raise RuntimeError(
                f"Errore SGP4 per {self.OBJECT_NAME} "
                f"(NORAD {self.NORAD_CAT_ID}): {position.message}"
            )
        return position

    def at(self, time: Time | datetime) -> np.ndarray:
        """Posizione ECEF/ITRS (x, y, z) in km, propagata con Skyfield."""
        return self._position_at(time).frame_xyz(itrs).km

    def subpoint(self, time: Time | datetime) -> tuple[float, float, float]:
        """Latitudine/longitudine WGS84 in gradi e quota ellissoidica in km."""
        point = wgs84.geographic_position_of(self._position_at(time))
        return (
            float(point.latitude.degrees),
            float(point.longitude.degrees),
            float(point.elevation.km),
        )


class SatelliteSwitch(OVSSwitch):
    """
    Switch OpenFlow che rappresenta un satellite nella rete emulata.
    """

    def __init__(self, name, satellite, **params):
        self.satellite = satellite                       # oggetto Satellite (i dati)
        dpid = satellite.NORAD_CAT_ID                    # in esadecimale su 16 cifre, cosi' il controller riconosce il satellite
        params.setdefault("dpid", format(dpid, "016x"))
        params.setdefault("protocols", "OpenFlow13")     # Ryu con OpenFlow 1.3
        super().__init__(name, **params)                 # crea lo switch Open vSwitch vero e proprio

    def at(self, time):
        """Posizione ECEF (x, y, z) in km del satellite all'istante time."""
        return self.satellite.at(time)

    def subpoint(self, time):
        """Latitudine, longitudine e quota del satellite all'istante time."""
        return self.satellite.subpoint(time)

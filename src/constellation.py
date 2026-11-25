"""Contenitore dei satelliti, senza caricamento di file o stato di simulazione."""

from __future__ import annotations

from dataclasses import dataclass, field

from satellite import Satellite


@dataclass
class Constellation:
    """Satelliti indicizzati per NORAD_CAT_ID."""

    satellites: dict[int, Satellite] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for norad_id, satellite in self.satellites.items():
            if norad_id != satellite.NORAD_CAT_ID:
                raise ValueError("La chiave della costellazione deve coincidere con il NORAD_CAT_ID.")

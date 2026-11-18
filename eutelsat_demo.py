"""Prova Skyfield con i 29 record GP Eutelsat forniti dall'utente.

Uso Python:
    sim = init_satellites("2026-11-18T12:00:00Z")
    iniziali = sim.initial_positions
    successive = sim.get_coordinates("2026-11-18T18:00:00Z")

Riferimento: https://rhodesmill.org/skyfield/earth-satellites.html
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

from skyfield.api import EarthSatellite, load, wgs84
from skyfield.timelib import Timescale


DEFAULT_DATA_FILE = Path(__file__).resolve().with_name("eutelsat.json")
UTCInput = str | datetime


class Coordinates(TypedDict):
    name: str
    norad_id: int
    latitude_deg: float
    longitude_deg: float


def parse_utc(value: UTCInput) -> datetime:
    """Accetta ISO 8601 o datetime con timezone e normalizza in UTC.

    Esempi: '2026-11-18T12:00:00Z', '2026-11-18T12:00:00+00:00'.
    Anche gli offset espliciti sono convertiti in UTC. Gli input senza
    timezone sono rifiutati per evitare di usare per errore l'ora locale.
    """
    if isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            value = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(
                "Data non valida: usa, per esempio, 2026-11-18T12:00:00Z."
            ) from exc
    if not isinstance(value, datetime):
        raise TypeError("La data deve essere una stringa ISO 8601 o un datetime.")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Specifica la timezone: usa Z, +00:00 o timezone.utc.")
    return value.astimezone(timezone.utc)


class EutelsatSimulation:
    """Satelliti propagati allo stesso istante, conservando le epoche GP.

    Creare l'istanza attraverso init_satellites(). Le interrogazioni sono
    indipendenti: non occorre avanzare la simulazione passo per passo.
    """

    def __init__(
        self,
        start_utc: datetime,
        satellites: tuple[EarthSatellite, ...],
        timescale: Timescale,
    ) -> None:
        self.start_utc = start_utc
        self.satellites = satellites
        self._timescale = timescale
        self.initial_positions = self.get_coordinates(start_utc)

    def get_coordinates(self, utc_datetime: UTCInput) -> list[Coordinates]:
        """Restituisce latitudine/longitudine WGS84 per tutti i satelliti.

        Il risultato segue l'ordine del JSON. Le coordinate sono in gradi:
        latitudine positiva a nord, longitudine positiva a est e compresa
        tra -180 e +180. Sono anche le coordinate del punto sottostante
        sull'ellissoide WGS84; l'altezza non viene restituita.

        La data e' assoluta e puo' precedere start_utc. Non modifica lo
        stato iniziale. Un errore SGP4 interrompe il calcolo con RuntimeError.
        """
        instant = self._timescale.from_datetime(parse_utc(utc_datetime))
        coordinates: list[Coordinates] = []
        for satellite in self.satellites:
            geocentric = satellite.at(instant)
            if geocentric.message is not None:
                raise RuntimeError(
                    f"Errore SGP4 per {satellite.name} "
                    f"(NORAD {satellite.model.satnum}): {geocentric.message}"
                )
            latitude, longitude = wgs84.latlon_of(geocentric)
            lat, lon = float(latitude.degrees), float(longitude.degrees)
            if not (math.isfinite(lat) and math.isfinite(lon)):
                raise RuntimeError(f"Coordinate non finite per {satellite.name}.")
            coordinates.append(
                Coordinates(
                    name=satellite.name,
                    norad_id=int(satellite.model.satnum),
                    latitude_deg=lat,
                    longitude_deg=lon,
                )
            )
        return coordinates


def init_satellites(
    start_utc: UTCInput,
    data_file: str | Path = DEFAULT_DATA_FILE,
) -> EutelsatSimulation:
    """Carica il JSON locale e calcola tutte le posizioni a start_utc.

    start_utc e' l'inizio comune della simulazione, non una nuova epoca
    da assegnare agli elementi orbitali. Ogni EPOCH originale e' conservata.
    Nessun dato orbitale o file di effemeridi viene scaricato da Internet.
    """
    start = parse_utc(start_utc)
    with Path(data_file).open(encoding="utf-8") as file:
        records = json.load(file)
    if not isinstance(records, list) or not records:
        raise ValueError("Il file deve contenere una lista JSON non vuota di record GP.")

    # Tabelle temporali incluse in Skyfield: nessun download al primo avvio.
    timescale = load.timescale(builtin=True)
    satellites = tuple(EarthSatellite.from_omm(timescale, record) for record in records)
    ids = [satellite.model.satnum for satellite in satellites]
    if len(set(ids)) != len(ids):
        raise ValueError("Il JSON contiene identificativi NORAD duplicati.")
    return EutelsatSimulation(start, satellites, timescale)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start", default="2026-11-18T12:00:00Z",
        help="Inizio comune in UTC (default: 2026-11-18T12:00:00Z).",
    )
    parser.add_argument(
        "--at", help="Istante UTC da interrogare; se omesso, usa --start.",
    )
    parser.add_argument(
        "--data", type=Path, default=DEFAULT_DATA_FILE,
        help="Percorso del file JSON GP (default: eutelsat.json accanto allo script).",
    )
    args = parser.parse_args()
    try:
        sim = init_satellites(args.start, args.data)
        query_utc = parse_utc(args.at) if args.at is not None else sim.start_utc
        positions = (
            sim.get_coordinates(query_utc)
            if args.at is not None else sim.initial_positions
        )
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        parser.exit(2, f"Errore: {exc}\n")
    print(json.dumps({
        "start_utc": sim.start_utc.isoformat().replace("+00:00", "Z"),
        "time_utc": query_utc.isoformat().replace("+00:00", "Z"),
        "satellites": positions,
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

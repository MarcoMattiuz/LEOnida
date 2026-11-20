from datetime import datetime, timezone
import math

import numpy as np # type: ignore

from mininet.node import OVSSwitch # type: ignore
from dataclasses import asdict, dataclass

R_EARTH = 6378.137          # km
MU = 398600.4418            # km^3/s^2, costante gravitazionale terrestre
OMEGA_EARTH = 7.2921159e-5  # rad/s, velocita' di rotazione della Terra

@dataclass(frozen=True)
class Satellite:
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
        return asdict(self)

    @staticmethod
    def to_datetime(time):
        """
        Converte l'istante in un datetime UTC.
        Accetta un Time di skyfield (quello usato da graph_path e main) oppure un
        datetime con timezone.
        """
        if hasattr(time, "utc_datetime"):                     # Time di skyfield
            return time.utc_datetime()
        if time.tzinfo is None:
            raise ValueError("Il datetime deve avere la timezone (es. timezone.utc).")
        return time.astimezone(timezone.utc)

    def gmst(self, time):
        """
        Angolo di rotazione della Terra (Greenwich Mean Sidereal Time) in radianti.
        time: datetime UTC
        """
        jd = time.timestamp() / 86400.0 + 2440587.5
        d = jd - 2451545.0
        return math.radians((280.46061837 + 360.98564736629 * d) % 360)

    def at(self, time):
        """
        Posizione ECEF (x, y, z) in km all'istante time, calcolata con Keplero.
        time: Time di skyfield oppure datetime con timezone.
        Restituisce un array numpy, come skyfield frame_xyz(itrs).km: si puo' usare
        direttamente per distanze, grafo e ritardi.
        """
        time = self.to_datetime(time)
        deg = math.radians
        epoch = datetime.fromisoformat(self.EPOCH).replace(tzinfo=timezone.utc)
        dt = (time - epoch).total_seconds()

        n = self.MEAN_MOTION * 2 * math.pi / 86400            # rad/s
        a = (MU / n**2) ** (1 / 3)                            # semiasse maggiore (Keplero III)
        e = self.ECCENTRICITY
        i, raan, w = deg(self.INCLINATION), deg(self.RA_OF_ASC_NODE), deg(self.ARG_OF_PERICENTER)

        M = deg(self.MEAN_ANOMALY) + n * dt                   # 1. anomalia media all'istante t
        E = M
        for _ in range(10):                                   # 2. equazione di Keplero  M = E - e sin E
            E = E - (E - e * math.sin(E) - M) / (1 - e * math.cos(E))
        nu = 2 * math.atan2(math.sqrt(1 + e) * math.sin(E / 2), math.sqrt(1 - e) * math.cos(E / 2))  # anomalia vera
        r = a * (1 - e * math.cos(E))                         # distanza dal centro della Terra

        x_orb, y_orb = r * math.cos(nu), r * math.sin(nu)     # 3. posizione nel piano dell'orbita

        # 4. rotazioni: argomento del perigeo (w), inclinazione (i), nodo ascendente (raan) -> spazio inerziale
        cO, sO, ci, si, cw, sw = math.cos(raan), math.sin(raan), math.cos(i), math.sin(i), math.cos(w), math.sin(w)
        x = (cO * cw - sO * sw * ci) * x_orb + (-cO * sw - sO * cw * ci) * y_orb
        y = (sO * cw + cO * sw * ci) * x_orb + (-sO * sw + cO * cw * ci) * y_orb
        z = (sw * si) * x_orb + (cw * si) * y_orb

        th = self.gmst(time)                                          # 5. la Terra ruota: inerziale -> ECEF
        xe, ye = math.cos(th) * x + math.sin(th) * y, -math.sin(th) * x + math.cos(th) * y

        return np.array([xe, ye, z])

    def subpoint(self, time):
        """
        Latitudine, longitudine (gradi) e quota (km) all'istante time (Terra sferica).
        """
        xe, ye, z = self.at(time)
        lat = math.degrees(math.atan2(z, math.hypot(xe, ye)))
        lon = math.degrees(math.atan2(ye, xe))
        return lat, lon, math.sqrt(xe**2 + ye**2 + z**2) - R_EARTH


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
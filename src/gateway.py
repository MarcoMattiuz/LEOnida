"""Dati del gateway terrestre e rappresentazione DockerHost."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite

from comnetsemu.node import DockerHost  # type: ignore

from position import GeoPosition


@dataclass(frozen=True)
class GatewayRadioConfig:
    """Capacita' radio; ogni antenna segue al massimo un satellite alla volta."""

    antenna_count: int
    max_active_satellites: int
    min_elevation_deg: float

    def __post_init__(self) -> None:
        for value in (self.antenna_count, self.max_active_satellites):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("Il numero di antenne e satelliti deve essere intero positivo.")
        if self.max_active_satellites > self.antenna_count:
            raise ValueError("I satelliti attivi non possono superare le antenne disponibili.")
        if not isfinite(self.min_elevation_deg) or not 0.0 <= self.min_elevation_deg <= 90.0:
            raise ValueError("L'elevazione minima deve essere compresa tra 0 e 90 gradi.")


@dataclass(frozen=True)
class Gateway:
    """Identita', posizione e capacita' di un gateway; nessun collegamento attivo."""

    id: str
    name: str
    position: GeoPosition
    radio: GatewayRadioConfig

    @classmethod
    def from_dict(cls, record: dict) -> Gateway:
        """Costruisce i dati annidati da un record gia' letto dal futuro loader."""
        return cls(**dict(
            record,
            position=GeoPosition(**record["position"]),
            radio=GatewayRadioConfig(**record["radio"]),
        ))

    def to_dict(self) -> dict:
        return asdict(self)

    def at(self, time=None):
        return self.position.at(time)

    def subpoint(self, time=None):
        return self.position.subpoint(time)


class GatewayNode(DockerHost):
    """Nodo emulato; forwarding e software del gateway saranno configurati dopo."""

    def __init__(self, name: str, gateway: Gateway, **params):
        self.gateway = gateway
        params.setdefault("docker_args", {})
        super().__init__(name, **params)

    def at(self, time=None):
        return self.gateway.at(time)

    def subpoint(self, time=None):
        return self.gateway.subpoint(time)

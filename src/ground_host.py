"""Dati di un host terrestre e rappresentazione DockerHost."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from comnetsemu.node import DockerHost  # type: ignore

from position import GeoPosition


@dataclass(frozen=True)
class GroundHost:
    """Host fisso: src e dst sono istanze della stessa classe."""

    id: str
    name: str
    position: GeoPosition

    @classmethod
    def from_dict(cls, record: dict) -> GroundHost:
        return cls(**dict(record, position=GeoPosition(**record["position"])))

    def to_dict(self) -> dict:
        return asdict(self)

    def at(self, time=None):
        return self.position.at(time)

    def subpoint(self, time=None):
        return self.position.subpoint(time)


class GroundHostNode(DockerHost):
    """Container dell'host; applicazioni e rete saranno configurate dopo."""

    def __init__(self, name: str, host: GroundHost, **params):
        self.host = host
        params.setdefault("docker_args", {})
        super().__init__(name, **params)

    def at(self, time=None):
        return self.host.at(time)

    def subpoint(self, time=None):
        return self.host.subpoint(time)

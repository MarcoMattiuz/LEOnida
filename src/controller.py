"""Identita' del controller e container che ne ospitera' l'applicazione."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from comnetsemu.node import DockerHost  # type: ignore


@dataclass(frozen=True)
class NetworkController:
    """Descrizione del controller, senza posizione geografica obbligatoria."""

    id: str
    name: str

    @classmethod
    def from_dict(cls, record: dict) -> NetworkController:
        return cls(**record)

    def to_dict(self) -> dict:
        return asdict(self)


class ControllerNode(DockerHost):
    """Container del controller; l'applicazione OpenFlow sara' aggiunta dopo.

    Questo oggetto rappresenta il nodo Docker, non un RemoteController Mininet.
    """

    def __init__(self, name: str, controller: NetworkController, **params):
        self.controller = controller
        params.setdefault("docker_args", {})
        super().__init__(name, **params)

"""
Aggiornamento della rete satellitare nel tempo.

La "network" e' una matrice di adiacenza che rappresenta il grafo con host e
satelliti a un certo istante:

    indice 0       = h1
    indici 1 .. n  = sat1 .. satN
    indice n + 1   = h2

    network[i][j] = ritardo in ms del link i-j se esiste, inf se no (0 sulla diagonale)

Archi:
    satellite - satellite : quelli di graph_path.build_graph (il segmento resta
                            sopra graph_path.MIN_ALT_KM di quota)
    host      - satellite : se l'host vede il satellite sopra MIN_ELEV_DEG

In Mininet ogni host e' collegato ai satelliti tramite la sua stazione a terra
(h1 -> gs1, h2 -> gs2): cosi' l'host ha un'unica interfaccia e il suo IP resta
valido qualunque sia il satellite a cui e' agganciato.

Routing semplificato (senza controller): restano accesi solo i link del cammino
minimo, gli altri sono spenti. Dopo ogni cambio di cammino le tabelle MAC degli
switch vengono svuotate, cosi' imparano subito il percorso nuovo.

"""
import math
from datetime import timedelta

import numpy as np  # type: ignore
from skyfield.api import EarthSatellite, load  # type: ignore
from skyfield.framelib import itrs  # type: ignore

# grafo dei satelliti, visibilita' degli ISL, Dijkstra e costanti: tutto da graph_path
from graph_path import C_KM_S, build_graph, dijkstra

MIN_ELEV_DEG = 25.0         # elevazione minima host -> satellite
LOSS_ACCESS = 1             # perdita (%) dei link host-satellite
LOSS_ISL = 0                # perdita (%) degli ISL

_TS = load.timescale(builtin=True)


# ---------------------------------------------------------------- nomi
def node_names(n_sats):
    """Nomi dei nodi della matrice, nello stesso ordine degli indici."""
    return ["h1"] + ["sat{}".format(i) for i in range(1, n_sats + 1)] + ["h2"]

def mininet_name(node):
    """Nodo Mininet a cui e' attaccato il link: per gli host e' la loro stazione a terra."""
    return {"h1": "gs1", "h2": "gs2"}.get(node, node)

def loss_of(a, b):
    """Perdita del link in base al tipo: host-satellite o ISL."""
    return LOSS_ACCESS if a.startswith("h") or b.startswith("h") else LOSS_ISL

# ---------------------------------------------------------------- geometria
def elevation_deg(ground, sat_pos):
    """Elevazione (gradi) del satellite visto da un punto a terra (Terra sferica)."""
    d = sat_pos - ground
    up = ground / np.linalg.norm(ground)
    return math.degrees(math.asin(float(np.dot(d, up)) / float(np.linalg.norm(d))))

def orbit_of(satellite):
    """Orbita (skyfield) di uno switch-satellite, creata una volta sola e poi riusata."""
    if not hasattr(satellite, "orbit"):
        satellite.orbit = EarthSatellite.from_omm(_TS, satellite.satellite.to_omm())
    return satellite.orbit

# ---------------------------------------------------------------- grafo
def compute_network(host_positions, orbits, time):
    """
    Matrice (n+2) x (n+2) con i ritardi in ms all'istante time.
    host_positions = [posizione h1, posizione h2] (ECEF, km).
    """
    sats = [o.at(time).frame_xyz(itrs).km for o in orbits]      # ognuno sulla propria orbita
    n = len(sats)
    network = np.full((n + 2, n + 2), np.inf)
    np.fill_diagonal(network, 0.0)

    # satelliti <-> satelliti (ISL): grafo di graph_path, km -> ms (inf resta inf)
    network[1:n + 1, 1:n + 1] = build_graph(sats) / C_KM_S * 1000

    # host <-> satelliti
    for h_idx, ground in ((0, host_positions[0]), (n + 1, host_positions[1])):
        for i, p in enumerate(sats, start=1):
            if elevation_deg(ground, p) >= MIN_ELEV_DEG:
                network[h_idx, i] = network[i, h_idx] = np.linalg.norm(p - ground) / C_KM_S * 1000
    return network

def shortest_path(network):
    """Cammino minimo (indici) da h1 (nodo 0) a h2 (ultimo nodo) con Dijkstra; [] se non esiste."""
    path, _ = dijkstra(network, 0, len(network) - 1)
    return path

def links_of(path, names):
    """Insieme dei link del cammino, come coppie ordinate di nomi."""
    return {tuple(sorted((names[a], names[b]))) for a, b in zip(path, path[1:])}

def candidate_links(host_positions, orbits, t0, minutes, step_s=30):
    """
    Link che potrebbero esistere almeno una volta nei prossimi `minutes` minuti:
    sono quelli da creare in Mininet all'avvio.
    """
    names = node_names(len(orbits))
    links = set()
    for s in range(0, int(minutes * 60) + 1, step_s):
        t = _TS.from_datetime(t0.utc_datetime() + timedelta(seconds=s))
        network = compute_network(host_positions, orbits, t)
        rows, cols = np.nonzero(np.isfinite(network))
        links |= {(names[a], names[b]) for a, b in zip(rows, cols) if a < b}
    return links

# ---------------------------------------------------------------- funzione principale
def set_netem(net, a, b, delay_ms):
    """Cambia ritardo e perdita del link a-b senza ricrearlo (netem sotto htb 5:1)."""
    for link in net.linksBetween(net.get(mininet_name(a)), net.get(mininet_name(b))):
        for intf in (link.intf1, link.intf2):
            intf.node.cmd("tc qdisc change dev {} parent 5:1 handle 10: netem delay {:.3f}ms loss {}%".format(
                intf.name, delay_ms, loss_of(a, b)))

def candidate_links_in(net, names):
    """Link satellitari presenti in Mininet (tutti accesi subito dopo net.start())."""
    inverse = {mininet_name(nm): nm for nm in names}
    found = set()
    for link in net.links:
        a, b = link.intf1.node.name, link.intf2.node.name
        if a in inverse and b in inverse:
            found.add(tuple(sorted((inverse[a], inverse[b]))))
    return found

def upgradeNetwork(net, network, hosts, satellites, time):
    """
    net:        la rete mininet
    network:    la matrice che rapprsenta il grafo con hots e satelliti
    hotst:      lista degli host da collegare
    satellites: lista dei setalliti
    time:       tempo in cui colcolare la nuova network
    """
    names = node_names(len(satellites))
    
    # trova nuova catena che colleghi h1 a h2 usando satellites al tempo time con dijkstra
    orbits      = [orbit_of(sat) for sat in satellites]
    new_network = compute_network([h.position for h in hosts], orbits, time)
    new_path    = shortest_path(new_network)

    # link accesi prima: quelli del cammino sulla network precedente
    # (alla prima chiamata, dopo net.start(), sono accesi tutti)
    if network is None:
        old_links = candidate_links_in(net, names)
    else:
        old_links = links_of(shortest_path(network), names)
    new_links = links_of(new_path, names)

    # aggiorna la rete in funzione dei nuovi collegamenti trovati
    for a, b in old_links - new_links:                          # non servono piu'
        net.configLinkStatus(names[a], names[b], "down")
    for a, b in new_links - old_links:                          # nuovi
        net.configLinkStatus(names[a], names[b], "up")
    for a, b in zip(new_path, new_path[1:]):                    # ritardi del cammino all'istante time
        set_netem(net, names[a], names[b], new_network[a, b])
    if old_links != new_links:                                  # percorso cambiato: tabelle MAC da rifare
        for sw in net.switches:
            sw.cmd("ovs-appctl fdb/flush {}".format(sw.name))

    rtt = 2 * sum(new_network[a, b] for a, b in zip(new_path, new_path[1:]))
    return new_network, [names[i] for i in new_path], rtt
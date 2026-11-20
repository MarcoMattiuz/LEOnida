#!/usr/bin/env python3
"""
Grafo pesato di host e satelliti e cammino minimo con Dijkstra.

- Nodi: h1, i satelliti del file OMM JSON, h2
      indice 0       = h1
      indici 1 .. n  = satelliti (nell'ordine del file)
      indice n + 1   = h2
- Arco satellite-satellite: esiste solo se i due satelliti si vedono, cioe' se
  il segmento che li unisce resta ad almeno MIN_ALT_KM di quota (non
  attraversa la Terra ne' l'atmosfera densa)
- Arco host-satellite: esiste se l'host vede il satellite ad almeno
  MIN_ELEV_DEG gradi sopra l'orizzonte
- Peso dell'arco: distanza euclidea fra i due nodi (km)
- Il grafo e' una matrice di adiacenza (n+2) x (n+2): adj[i][j] = distanza se
  l'arco esiste, infinito altrimenti

Gli host sono fermi a terra. Se non si indicano le loro posizioni, stanno sotto
il primo e l'ultimo satellite del file all'epoca piu' recente.
Il cammino va da h1 a h2 ed e' cercato nel primo istante, dopo l'epoca
(passo di STEP_S secondi, per SEARCH_HOURS ore), in cui esiste.

Uso:
    chain, t, hosts = find_chain(records, ts)
        chain = indici (nel file) dei satelliti del cammino, da h1 verso h2
        t     = istante in cui il cammino esiste
        hosts = [posizione h1, posizione h2] (ECEF, km)
"""
from datetime import datetime, timezone
import math

import numpy as np # type: ignore
from skyfield.api import EarthSatellite, load, wgs84 # type: ignore
from skyfield.framelib import itrs # type: ignore


R_EARTH_KM = 6371.0     # raggio terrestre
MIN_ALT_KM = 80.0       # quota minima del segmento perche' i satelliti si vedano
MIN_ELEV_DEG = 25.0     # elevazione minima perche' un host veda un satellite
C_KM_S = 299792.458     # velocita' della luce
SEARCH_HOURS = 24       # ore dopo l'epoca in cui cercare un istante con un cammino
STEP_S = 60             # passo della ricerca (secondi)
MAX_MATRIX_PRINT = 15   # la matrice viene stampata solo se i nodi sono al massimo questi


# ---------------------------------------------------------------- geometria
def segment_min_altitude(p1, p2):
    """
    Quota minima (km) del segmento p1-p2 rispetto alla sfera terrestre.

    Si cerca il punto del SEGMENTO piu' vicino al centro della Terra (origine):
    sulla retta p1 + s*(p2 - p1) il punto piu' vicino all'origine ha
        s = -(p1 . (p2 - p1)) / |p2 - p1|^2
    Limitando s fra 0 e 1 il punto resta dentro il segmento (sulla retta
    infinita si scarterebbero coppie che in realta' si vedono).
    """
    d = p2 - p1
    s = -np.dot(p1, d) / np.dot(d, d)
    s = min(1.0, max(0.0, s))
    closest = p1 + s * d
    return np.linalg.norm(closest) - R_EARTH_KM

def can_see(p1, p2):
    return segment_min_altitude(p1, p2) >= MIN_ALT_KM

def elevation_deg(ground, sat):
    """
    Elevazione (gradi) del satellite sopra l'orizzonte di un punto a terra.
    E' l'angolo fra la direzione host -> satellite e il piano dell'orizzonte,
    la cui normale (la "verticale") e' la direzione centro della Terra -> host.
    """
    d = sat - ground
    up = ground / np.linalg.norm(ground)
    return math.degrees(math.asin(float(np.dot(d, up)) / float(np.linalg.norm(d))))

def host_can_see(ground, sat):
    return elevation_deg(ground, sat) >= MIN_ELEV_DEG

# ---------------------------------------------------------------- grafo
def build_graph(positions):
    """
    Matrice di adiacenza n x n: distanza in km se i satelliti si vedono, inf altrimenti.
    """
    n = len(positions)
    adj = np.full((n, n), np.inf)
    np.fill_diagonal(adj, 0.0)
    for i in range(n):
        for j in range(i + 1, n):
            if can_see(positions[i], positions[j]):
                adj[i, j] = adj[j, i] = np.linalg.norm(positions[i] - positions[j])
    return adj

def build_graph_with_hosts(sat_positions, hosts):
    """
    Matrice (n+2) x (n+2) con h1 (indice 0), i satelliti (1..n) e h2 (n+1).
    La parte fra satelliti e' build_graph; si aggiungono gli archi host-satellite.
    """
    n = len(sat_positions)
    adj = np.full((n + 2, n + 2), np.inf)
    np.fill_diagonal(adj, 0.0)
    adj[1:n + 1, 1:n + 1] = build_graph(sat_positions)

    for h_idx, ground in ((0, hosts[0]), (n + 1, hosts[1])):
        for i, p in enumerate(sat_positions, start=1):
            if host_can_see(ground, p):
                adj[h_idx, i] = adj[i, h_idx] = np.linalg.norm(p - ground)
    return adj

def default_hosts(sats, t):
    """
    Host a terra sotto il primo e l'ultimo satellite all'istante t (ECEF, km).
    """
    return [wgs84.subpoint_of(sats[0].at(t)).itrs_xyz.km,
            wgs84.subpoint_of(sats[-1].at(t)).itrs_xyz.km]

def dijkstra(adj, src, dst):
    """
    Cammino minimo da src a dst sulla matrice di adiacenza.

    Restituisce (lista dei nodi del cammino, distanza totale), oppure
    ([], inf) se dst non e' raggiungibile.
    """
    n = len(adj)
    dist = [np.inf] * n         # distanza minima nota da src a ogni nodo
    prev = [None] * n           # nodo precedente nel cammino minimo
    visited = [False] * n
    dist[src] = 0.0

    for _ in range(n):
        # nodo non ancora visitato con distanza minima
        u = None
        for v in range(n):
            if not visited[v] and (u is None or dist[v] < dist[u]):
                u = v
        if u is None or dist[u] == np.inf:
            break                       # i nodi rimasti non sono raggiungibili
        if u == dst:
            break                       # trovato: la distanza di dst e' definitiva
        visited[u] = True

        # rilassamento: si puo' arrivare ai vicini di u passando da u?
        for v in range(n):
            if not visited[v] and adj[u][v] != np.inf and v != u:
                alt = dist[u] + adj[u][v]
                if alt < dist[v]:
                    dist[v] = alt
                    prev[v] = u

    if dist[dst] == np.inf:
        return [], np.inf
    path = []
    node = dst
    while node is not None:             # ricostruzione all'indietro da dst a src
        path.append(node)
        node = prev[node]
    return path[::-1], dist[dst]

# ---------------------------------------------------------------- input / output
def node_names(records):
    """Nomi dei nodi del grafo, nell'ordine della matrice."""
    return ["h1"] + [r["OBJECT_NAME"] for r in records] + ["h2"]

def path_table(path, records, sats, positions, hosts, t):
    """
    Tabella del cammino da h1 a h2: una riga per nodo, con distanza e ritardo
    del salto dal nodo precedente e i valori cumulati.
    positions = posizioni dei satelliti, hosts = posizioni di h1 e h2.
    """
    n = len(records)
    names = node_names(records)
    xyz = [hosts[0]] + list(positions) + [hosts[1]]        # posizione di ogni nodo del grafo
    rows = []
    total_km = 0.0
    for step, i in enumerate(path):
        hop_km = 0.0 if step == 0 else float(np.linalg.norm(xyz[i] - xyz[path[step - 1]]))
        total_km += hop_km
        if 1 <= i <= n:                                    # satellite
            geo = sats[i - 1].at(t)
            lat, lon = wgs84.latlon_of(geo)
            lat, lon, alt, norad = lat.degrees, lon.degrees, wgs84.height_of(geo).km, records[i - 1]["NORAD_CAT_ID"]
        else:                                              # host: a terra
            x, y, z = xyz[i]
            lat, lon = math.degrees(math.atan2(z, math.hypot(x, y))), math.degrees(math.atan2(y, x))
            alt, norad = 0.0, ""
        rows.append({
            "step": step,
            "name": names[i],
            "norad": norad,
            "lat": lat,
            "lon": lon,
            "alt_km": alt,
            "hop_km": hop_km,
            "hop_ms": hop_km / C_KM_S * 1000,
            "total_km": total_km,
            "total_ms": total_km / C_KM_S * 1000,
        })
    return rows

def print_table(rows):
    header = "{:>4}  {:15} {:>6}  {:>8} {:>9} {:>7}  {:>8} {:>7}  {:>8} {:>7}".format(
        "step", "nodo", "NORAD", "lat", "lon", "quota", "salto km", "ms", "tot km", "tot ms")
    print(header)
    print("-" * len(header))
    for r in rows:
        print("{step:>4}  {name:15} {norad:>6}  {lat:8.2f} {lon:9.2f} {alt_km:7.0f}  "
              "{hop_km:8.0f} {hop_ms:7.2f}  {total_km:8.0f} {total_ms:7.2f}".format(**r))

def print_matrix(adj, records):
    names = [nm.replace("STARLINK-", "") for nm in node_names(records)]
    print("\nMatrice di adiacenza (km, '-' = i nodi non si vedono):")
    print("      " + "".join("{:>7}".format(nm) for nm in names))
    for nm, row in zip(names, adj):
        print("{:>6}".format(nm) + "".join("{:>7}".format("-" if v == np.inf else "{:.0f}".format(v)) for v in row))
    print()

def parse_time(time, ts):
    if hasattr(time, "utc_datetime"):
        return time
    if isinstance(time, str):
        text = time.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"                     
        time = datetime.fromisoformat(text)
    if time.tzinfo is None:
        raise ValueError("L'istante deve avere la timezone (es. 'Z' o +00:00).")
    return ts.from_datetime(time.astimezone(timezone.utc))

def find_chain(records, ts, time, hosts=None, verbose=True):
    """
    Cerca il cammino minimo da h1 a h2 attraverso i satelliti all'istante `time`.

    time:  istante in cui cercare il cammino (Time di skyfield, datetime con
           timezone o stringa ISO, es. '2026-11-20T07:19:37Z')
    hosts: [posizione h1, posizione h2] (ECEF, km); se None, gli host stanno
           sotto il primo e l'ultimo satellite all'istante `time`.

    Restituisce (chain, t, hosts):
      chain = indici (in records) dei satelliti del cammino, da h1 verso h2
      t     = l'istante usato (skyfield Time)
      hosts = posizioni degli host usate
    Se a quell'istante non esiste un cammino restituisce (None, t, hosts).
    """
    if len(records) < 1:
        raise ValueError("Serve almeno un satellite.")

    sats = [EarthSatellite.from_omm(ts, r) for r in records]
    n    = len(sats)
    t    = parse_time(time, ts) # istante passato come parametro

    if hosts is None:
        hosts = default_hosts(sats, t)

    src, dst = 0, n + 1 # h1 -> h2

    # grafo e cammino all'istante t
    positions   = [s.at(t).frame_xyz(itrs).km for s in sats] # x, y, z in km (ECEF)
    adj         = build_graph_with_hosts(positions, hosts)
    path, total = dijkstra(adj, src, dst)
    if not path:
        return None, t, hosts

    if verbose:
        print("Istante: {}   satelliti: {}   archi: {}".format(
            t.utc_strftime("%Y-%m-%dT%H:%M:%SZ"), n, int(np.isfinite(adj).sum() - len(adj)) // 2))
        if n + 2 <= MAX_MATRIX_PRINT:
            print_matrix(adj, records)
        print("\nCammino minimo h1 -> h2  ({} salti, {:.0f} km, {:.2f} ms)\n".format(
            len(path) - 1, total, total / C_KM_S * 1000))
        print_table(path_table(path, records, sats, positions, hosts, t))

    chain = [i - 1 for i in path[1:-1]] # solo i satelliti, come indici nel file
    return chain, t, hosts
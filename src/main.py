#!/usr/bin/env python3
"""
Demo ISL in Mininet con n satelliti LEO letti da un file OMM JSON.

    h1 -- sat1 ==ISL== sat2 ==ISL== ... ==ISL== satN -- h2

Il file JSON (formato CelesTrak) viene passato a graph_path.find_chain, che
costruisce il grafo con host e satelliti e trova con Dijkstra la catena
(cammino minimo) da h1 a h2, e l'istante in cui esiste.
Se la catena non esiste il programma termina.
h1 e h2 sono a terra: sotto il primo e l'ultimo satellite del file all'epoca.
I satelliti della catena vengono "congelati" nell'istante trovato (nessun
movimento, nessun handover).

Da eseguire nella VM ComNetsEmu con il Python di sistema (3.8). Serve
skyfield per il Python di sistema (una volta sola):
    sudo python3 -m pip install skyfield==1.55

Uso:
    sudo python3 main.py satelliti.json
"""
import argparse
from datetime import timedelta
import json
from dataclasses import fields

from graph_path import find_chain, parse_time
from satellite import Satellite, SatelliteSwitch

import numpy as np # type: ignore
from mininet.cli import CLI # type: ignore
from mininet.link import TCLink # type: ignore
from mininet.log import info, setLogLevel # type: ignore
from mininet.net import Mininet # type: ignore
from skyfield.api import EarthSatellite, load, wgs84 # type: ignore
from skyfield.framelib import itrs # type: ignore


C  = 299792.458          # km/s nel vuoto
BW = 20                 # banda di ogni link (Mbit/s)
SEARCH_HOURS = 24
STEP_S       = 60

def load_satellites(path: str):
    with open(path, encoding="utf-8") as f:
        records = json.load(f)

    names = {f.name for f in fields(Satellite)}
    satellites = []
    for record in records:
        values = {key: value for key, value in record.items() if key in names}
        satellites.append(Satellite(**values))
    return satellites

def ecef(sat, t):
    """
    Posizione ITRS (centrata e solidale con la Terra) in km.
    """
    return sat.at(t).frame_xyz(itrs).km

def compute_links(orbits, chain, t, hosts):
    """
    Link della catena trovata da find_chain, come tuple (nodo_a, nodo_b, km, ritardo_ms).
    I satelliti hanno il nome del loro switch: sat<i>, con i = posizione nel file (da 1).
    """
    nodes = ["h1"] + ["sat{}".format(i + 1) for i in chain] + ["h2"]
    xyz   = [hosts[0]] + [ecef(orbits[i], t) for i in chain] + [hosts[1]]
    links = []
    for k in range(len(nodes) - 1):
        km = float(np.linalg.norm(xyz[k] - xyz[k + 1]))
        links.append((nodes[k], nodes[k + 1], km, km / C * 1000))
    return links

def print_links(links):
    print("\nPercorso:")
    for a, b, km, d in links:
        print("  {:20s} {:9.0f} km  {:7.2f} ms".format("{}-{}".format(a, b), km, d))
    tot = sum(d for _, _, _, d in links)
    print("  {:20s} {:9s}     {:7.2f} ms".format("one-way", "", tot))
    print("  {:20s} {:9s}     {:7.2f} ms\n".format("RTT atteso (ping)", "", 2 * tot))

def create_hosts(net: Mininet, n: int):
    """
    Crea n host (h1, h2, ..., hn) nella rete e li restituisce in una lista.

    Ogni host ha un indirizzo nella stessa sottorete:
    h1 = 10.0.0.1/24,
    h2 = 10.0.0.2/24,
    ...
    """
    if not 1 <= n <= 254:
        raise ValueError("n deve essere fra 1 e 254 (indirizzi disponibili in una /24)")

    hosts = []
    for i in range(1, n + 1):
        host = net.addHost("h{}".format(i), ip="10.0.0.{}/24".format(i))
        hosts.append(host)
    return hosts

def create_satellites(net: Mininet, sats):
    """
    Crea uno switch Open vSwitch per ogni satellite nella rete e li restituisce in una lista.

    Ogni switch ha un nome univoco:
    sat1,
    sat2,
    ...
    """
    satellites = []
    for i, sat in enumerate(sats, start=1):
        # failMode="standalone": senza controller lo switch inoltra da solo (learning switch)
        switch = net.addSwitch("sat{}".format(i), cls=SatelliteSwitch, satellite=sat,
                               failMode="standalone")
        satellites.append(switch)
    return satellites

def run_mininet(links, sats):
    setLogLevel("info")
    net = Mininet(link=TCLink, controller=None)

    # nodes
    h1, h2 = create_hosts(net, 2)
    satellites = create_satellites(net, sats)       
    nodes = {"h1": h1, "h2": h2}
    nodes.update({sw.name: sw for sw in satellites})

    # (TCLink applica "delay" su entrambe le interfacce: ogni direzione lo subisce una volta)
    for a, b, km, d in links:
        loss = 2 if b == "h2" else 1
        net.addLink(nodes[a], nodes[b], delay="{:.3f}ms".format(d), bw=BW, loss=loss)

    net.start()
    info("*** prova:  h1 ping -c 5 h2     |     iperf h1 h2\n")
    CLI(net)
    net.stop()

def main():
    p = argparse.ArgumentParser(description="Demo ISL con n satelliti LEO da file OMM JSON")
    p.add_argument("satellites_json",
                help="file OMM JSON: la catena va dal primo all'ultimo satellite del file")
    p.add_argument("time", help="istante UTC, es. 2026-11-20T07:19:37Z")
    a = p.parse_args()

    ts = load.timescale(builtin=True)
    sats = load_satellites(a.satellites_json)
    if not sats:
        raise SystemExit("Il file non contiene satelliti.")
    records = [s.to_omm() for s in sats]

   # catena all'istante richiesto; se non c'e', si prova un minuto dopo, e cosi' via
    start = parse_time(a.time, ts)
    chain, t, hosts = find_chain(records, ts, start, verbose=False)
    seconds = 0
    while chain is None and seconds < SEARCH_HOURS * 3600:
        seconds += STEP_S
        t = ts.from_datetime(start.utc_datetime() + timedelta(seconds=seconds))
        chain, t, hosts = find_chain(records, ts, t, hosts=hosts, verbose=False)   # host fermi
    if chain is None:
        raise SystemExit("Nessuna catena da h1 a h2 nelle {} ore dopo {}: il programma termina.".format(SEARCH_HOURS, a.time))
    find_chain(records, ts, t, hosts=hosts) # stampa matrice e tabella dell'istante trovato

    orbits = [EarthSatellite.from_omm(ts, s.to_omm()) for s in sats] # orbite per i calcoli
    links = compute_links(orbits, chain, t, hosts)
    print_links(links)
    run_mininet(links, sats)

if __name__ == "__main__":
    main()

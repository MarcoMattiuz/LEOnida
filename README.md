# Prova Eutelsat con Skyfield

Lo script carica i 29 record GP del file `eutelsat.json`, calcola le posizioni
iniziali a un istante UTC comune e permette di interrogare latitudine e
longitudine di tutti i satelliti a un altro istante UTC.

Il JSON e' una copia integrale dei dati forniti dall'utente, con epoche dal
16 al 18 novembre 2026. Comprende anche EXPRESS-AT1 ed EXPRESS-AT2.

## Installazione ed esecuzione

Richiede Python 3.10 o successivo. Dalla directory del progetto:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python eutelsat_demo.py \
  --start 2026-11-18T12:00:00Z \
  --at 2026-11-18T18:00:00Z
```

Lo script stampa JSON con `start_utc`, `time_utc` e i 29 risultati in
`satellites`. Senza `--at` restituisce le posizioni iniziali. `--data` permette
di indicare un altro file JSON GP; il default viene risolto rispetto allo
script, anche quando lo si esegue da un'altra directory.

## Uso delle due funzioni

```python
from datetime import datetime, timezone
from eutelsat_demo import init_satellites

# 1. Inizializzazione: tutti i satelliti sono valutati allo stesso istante UTC.
sim = init_satellites("2026-11-18T12:00:00Z")
posizioni_iniziali = sim.initial_positions

# 2. Interrogazione: una funzione riceve il tempo e restituisce le coordinate.
posizioni = sim.get_coordinates("2026-11-18T18:00:00Z")

# E' accettato anche un datetime con timezone esplicita.
posizioni = sim.get_coordinates(
    datetime(2026, 11, 18, 18, 0, 0, tzinfo=timezone.utc)
)

for satellite in posizioni:
    print(
        satellite["name"],
        satellite["latitude_deg"],
        satellite["longitude_deg"],
    )
```

Ogni elemento restituito e' un dizionario di questa forma (primo satellite
alle 18:00 UTC del 18 novembre 2026, valori qui arrotondati):

```json
{
  "name": "EUTELSAT 174A",
  "norad_id": 28924,
  "latitude_deg": 4.241298,
  "longitude_deg": 173.936031
}
```

Le coordinate sono geodetiche WGS84 in gradi decimali: nord ed est positivi;
la longitudine e' compresa tra -180 e +180. Identificano anche il punto
sull'ellissoide terrestre sottostante il satellite; l'altezza non e' inclusa.

## Significato dell'inizializzazione

`start_utc` definisce l'inizio comune della simulazione. Le epoche `EPOCH`
del JSON sono conservate: SGP4 propaga ogni satellite dalla propria epoca
all'istante richiesto. Cambiare l'inizio della simulazione non cambia la
posizione prevista a una stessa data assoluta.

Ogni chiamata e' indipendente: e' possibile interrogare anche istanti
precedenti all'inizio, senza avanzamenti intermedi. Sono accettati timestamp
ISO 8601 con `Z` o offset esplicito, oppure `datetime` con timezone. Gli
offset vengono normalizzati in UTC; gli input senza timezone sono rifiutati.

La sequenza segue la [documentazione Skyfield sui satelliti terrestri](https://rhodesmill.org/skyfield/earth-satellites.html):
`EarthSatellite.from_omm()` carica i record, `satellite.at(t)` propaga il moto
e `wgs84.latlon_of()` ricava latitudine e longitudine. Gli errori del
propagatore sono segnalati con `RuntimeError`.

Dopo l'installazione lo script funziona senza rete: usa il JSON locale e le
tabelle temporali incluse in Skyfield (`load.timescale(builtin=True)`).
Non servono effemeridi planetarie. Per prove con questi dati, scegliere
date vicine al 16-18 novembre 2026: l'accuratezza degrada lontano dalle
epoche degli elementi e non sono modellate eventuali manovre successive.

## Verifica

```sh
.venv/bin/python -m unittest discover -s tests -v
```

## Globo 3D e movimento dei satelliti

`eutelsat_globe.py` apre una finestra Matplotlib con il globo terrestre,
le traiettorie e tutti i 29 satelliti del JSON, compresi EXPRESS-AT1 e
EXPRESS-AT2. Per installare anche le dipendenze grafiche e avviare la prova:

```sh
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python eutelsat_globe.py \
  --start 2026-11-18T12:00:00Z \
  --hours 24 \
  --speed 3600
```

Questi sono anche i valori predefiniti: basta eseguire
`.venv/bin/python eutelsat_globe.py`. `--hours` definisce la durata
simulata; `--speed` esprime il rapporto fra tempo simulato e tempo reale
(3600 significa un'ora simulata per secondo reale). `--data` permette
di scegliere un altro file JSON GP.

Nella finestra:

- **Avvia/Pausa** controlla il movimento; **Ricomincia** torna all'istante iniziale.
- La **timeline** seleziona l'istante come ore trascorse dall'inizio e mette in pausa.
- Il cursore della **velocita'** varia l'accelerazione da 30 a 7200 volte.
- I pulsanti **precedente/successivo** selezionano un satellite; e' possibile
  selezionarlo anche facendo clic sul suo punto nella vista 3D.
- Trascinando il mouse si ruota la vista; con la rotella si regola lo zoom.

La selezione evidenzia il satellite e la sua traiettoria. Le orbite di
questo gruppo sono molto vicine tra loro: la loro sovrapposizione nella
regione geosincrona e' attesa.

Per aprire la stessa finestra da Python, senza usare argomenti da terminale:

```python
from eutelsat_globe import show_orbits

show_orbits(
    start_utc="2026-11-18T12:00:00Z",
    duration_hours=24.0,
    speed=3600.0,
)
```

La funzione ritorna quando si chiude la finestra. Il parametro opzionale
`data_file` permette di scegliere un altro JSON; il valore predefinito e'
lo stesso `eutelsat.json` usato dallo script delle coordinate.

Per produrre una sola anteprima PNG, anche senza interfaccia grafica:

```sh
.venv/bin/python eutelsat_globe.py \
  --start 2026-11-18T12:00:00Z \
  --save-preview eutelsat_globe.png
```

### Come viene costruita la scena

Le posizioni dei satelliti sono calcolate con SGP4 all'istante di ogni
fotogramma e rappresentate nel riferimento geocentrico **GCRS**. Il globo
e le sue linee geografiche ruotano usando la trasformazione **ITRS → GCRS**
di Skyfield. In questo modo e' visibile il moto nello spazio insieme alla
rotazione terrestre; i satelliti quasi geostazionari mantengono una
posizione approssimativamente fissa rispetto alla superficie.

Le linee delle orbite campionano un periodo orbitale a partire dall'inizio
della simulazione. Sono tracce di riferimento pre-calcolate, mentre i
satelliti continuano a essere propagati all'istante corrente: il moto
perturbato non viene ridotto a un'ellisse immutabile.

Le distanze della scena mantengono la scala fisica: il globo e' una sfera
di raggio **6378,137 km**. I punti che rappresentano i satelliti sono
ingranditi per renderli visibili. Le linee di costa provengono da
[Natural Earth](https://www.naturalearthdata.com/), sono di pubblico dominio
e sono incluse localmente in `assets/earth_coastlines.json`; la provenienza
e' documentata in [assets/README.md](assets/README.md).

La visualizzazione conserva i dati orbitali originali e, dopo
l'installazione delle dipendenze, funziona senza rete. Valgono gli stessi
limiti temporali degli elementi GP descritti sopra.

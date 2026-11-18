# Linee di costa del globo

`earth_coastlines.json` contiene le linee di costa di **Natural Earth,
scala 1:110 milioni**, da usare come riferimento geografico sul globo 3D.

- [Sorgente GeoJSON nel repository Natural Earth](https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_coastline.geojson)
- [Termini di utilizzo Natural Earth](https://www.naturalearthdata.com/about/terms-of-use/): dati di pubblico dominio.
- Coordinate: longitudine e latitudine in gradi, nell'ordine `[lon, lat]`,
  riferimento geografico OGC:CRS84.
- Contenuto: 134 linee, 5.128 punti. Sono conservati tutti i vertici della
  sorgente, arrotondati a sei cifre decimali; non e' stata applicata una
  semplificazione geometrica.

Il JSON registra URL, licenza e SHA-256 dei byte della sorgente scaricata
per consentire di identificare la copia usata. La geometria e' inclusa
localmente: la visualizzazione non deve scaricare mappe durante l'uso.

Made with Natural Earth.

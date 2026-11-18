"""Globo 3D e movimento dei satelliti Eutelsat, calcolati con Skyfield.

Avvio: python eutelsat_globe.py --start 2026-11-18T12:00:00Z
Uso Python: show_orbits("2026-11-18T12:00:00Z")
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from skyfield.api import load, wgs84
from skyfield.framelib import itrs

from eutelsat_demo import DEFAULT_DATA_FILE, UTCInput, init_satellites, parse_utc


EARTH_RADIUS_KM = 6378.137
DEFAULT_START = "2026-11-18T12:00:00Z"
COASTLINES_FILE = Path(__file__).resolve().parent / "assets" / "earth_coastlines.json"
BACKGROUND = "#0b1422"
FOREGROUND = "#e6edf6"
MUTED = "#93a8bf"
ORBIT_COLOR = "#55b9d0"
SELECTED_COLOR = "#ffc66d"


@dataclass(frozen=True)
class SceneState:
    utc: datetime
    positions_km: np.ndarray
    earth_rotation: np.ndarray
    latitude_deg: np.ndarray
    longitude_deg: np.ndarray
    altitude_km: np.ndarray


def surface_xyz(longitude_deg, latitude_deg, radius_km=EARTH_RADIUS_KM):
    """Coordinate cartesiane ITRS su un globo sferico (solo per il disegno)."""
    longitude, latitude = np.broadcast_arrays(
        np.deg2rad(longitude_deg), np.deg2rad(latitude_deg)
    )
    return radius_km * np.stack([
        np.cos(latitude) * np.cos(longitude),
        np.cos(latitude) * np.sin(longitude),
        np.sin(latitude),
    ], axis=-1)


def _check_position(position, name):
    messages = np.asarray(position.message, dtype=object).reshape(-1)
    errors = [str(message) for message in messages if message]
    if errors:
        raise RuntimeError(f"Errore SGP4 per {name}: {errors[0]}")
    if not np.isfinite(position.xyz.km).all():
        raise RuntimeError(f"Coordinate non finite per {name}.")


class OrbitScene:
    """Propagazione fisica indipendente dalla finestra grafica.

    Le posizioni sono GCRS, in km. La matrice earth_rotation trasforma
    punti solidali con la Terra (ITRS) nel medesimo riferimento GCRS.
    """

    def __init__(
        self, start_utc: UTCInput = DEFAULT_START,
        data_file: str | Path = DEFAULT_DATA_FILE, orbit_samples: int = 181,
    ):
        if not isinstance(orbit_samples, int) or orbit_samples < 3:
            raise ValueError("orbit_samples deve essere un intero >= 3.")
        self.simulation = init_satellites(start_utc, data_file)
        self.timescale = load.timescale(builtin=True)
        self.orbits_km = self._sample_orbits(orbit_samples)

    def _sample_orbits(self, count):
        paths = []
        for satellite in self.simulation.satellites:
            # no_kozai e' il moto medio in radianti/minuto.
            period_seconds = 2 * math.pi / satellite.model.no_kozai * 60
            datetimes = [
                self.simulation.start_utc + timedelta(seconds=float(seconds))
                for seconds in np.linspace(0, period_seconds, count)
            ]
            positions = satellite.at(self.timescale.from_datetimes(datetimes))
            _check_position(positions, satellite.name)
            # Traccia SGP4 di un periodo: non forziamo la chiusura dell'orbita.
            paths.append(positions.xyz.km.T)
        return np.asarray(paths)

    def at(self, utc_datetime: UTCInput) -> SceneState:
        utc = parse_utc(utc_datetime)
        instant = self.timescale.from_datetime(utc)
        positions, latitudes, longitudes, altitudes = [], [], [], []
        for satellite in self.simulation.satellites:
            position = satellite.at(instant)
            _check_position(position, satellite.name)
            geographic = wgs84.geographic_position_of(position)
            positions.append(position.xyz.km)
            latitudes.append(geographic.latitude.degrees)
            longitudes.append(geographic.longitude.degrees)
            altitudes.append(geographic.elevation.km)
        return SceneState(
            utc=utc, positions_km=np.asarray(positions),
            earth_rotation=itrs.rotation_at(instant).T,
            latitude_deg=np.asarray(latitudes),
            longitude_deg=np.asarray(longitudes),
            altitude_km=np.asarray(altitudes),
        )


class GlobeViewer:
    """Finestra Matplotlib: globo, orbite, selezione e controlli temporali."""

    def __init__(self, scene: OrbitScene, duration_hours=24.0, speed=3600.0):
        # Import differito: il modello numerico funziona anche senza GUI.
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, Slider
        from mpl_toolkits.mplot3d.art3d import Line3DCollection

        if not math.isfinite(duration_hours) or duration_hours <= 0:
            raise ValueError("La durata deve essere positiva e finita.")
        if not math.isfinite(speed) or not 30 <= speed <= 7200:
            raise ValueError("La velocita' deve essere tra 30 e 7200.")
        self.scene = scene
        self.duration_seconds = duration_hours * 3600
        self.speed = speed
        self.elapsed = 0.0
        self.selected = 0
        self.playing = True
        self._last_wall = time.monotonic()
        self.state = scene.at(scene.simulation.start_utc)
        self.figure = plt.figure(figsize=(12.8, 8.2), facecolor=BACKGROUND)
        self.figure.canvas.manager.set_window_title("Eutelsat | Orbite 3D")
        self.axes = self.figure.add_axes(
            [0.00, 0.20, 0.77, 0.69], projection="3d", facecolor=BACKGROUND,
            computed_zorder=False,
        )
        self.axes.set_proj_type("ortho")
        self.axes.set_box_aspect((1, 1, 1), zoom=1.72)
        self.axes.view_init(elev=30, azim=-65)
        self.axes.set_axis_off()
        self.limit = max(float(np.linalg.norm(scene.orbits_km, axis=2).max()),
                         EARTH_RADIUS_KM) * 1.12
        self._set_limits(self.limit)

        self.figure.text(0.05, 0.95, "EUTELSAT  /  ORBITE TERRESTRI",
                         fontsize=20, color=FOREGROUND)
        self.figure.text(0.05, 0.912,
                         f"{len(scene.simulation.satellites)} satelliti  ·  Distanze in scala  ·  Vista dallo spazio",
                         fontsize=11, color=MUTED)
        self.clock_text = self.figure.text(0.755, 0.84, "", color=FOREGROUND,
                                           fontsize=12, linespacing=1.7)
        self.selection_text = self.figure.text(0.755, 0.73, "", color=FOREGROUND,
                                               fontsize=11, linespacing=1.8, va="top")
        self.figure.text(0.755, 0.345,
                         "Orbite quasi sovrapposte:\nè la geometria dei dati.\n\nTrascina per ruotare.\nRotella per lo zoom.\nClicca un satellite per sceglierlo.",
                         color=MUTED, fontsize=9, linespacing=1.7, va="top")

        self.orbit_lines = []
        for index, path in enumerate(scene.orbits_km):
            line, = self.axes.plot(*path.T, color=ORBIT_COLOR, lw=0.65,
                                   alpha=0.30, zorder=4)
            self.orbit_lines.append(line)

        # Globo sferico in scala. Le coste ruotano con la Terra; la sfera
        # uniforme e' invariante rispetto alla rotazione.
        lon, lat = np.meshgrid(np.linspace(-180, 180, 49), np.linspace(-90, 90, 25))
        sphere = surface_xyz(lon, lat)
        self.axes.plot_surface(*np.moveaxis(sphere, -1, 0), color="#1b4566",
                               linewidth=0, antialiased=True, shade=True, zorder=2)
        with COASTLINES_FILE.open(encoding="utf-8") as file:
            coastlines = json.load(file)["lines"]
        self.coasts_itrs = [surface_xyz(np.array(line)[:, 0], np.array(line)[:, 1],
                                       EARTH_RADIUS_KM * 1.002)
                           for line in coastlines]
        self.grid_itrs = [
            surface_xyz(np.linspace(-180, 180, 145), latitude,
                        EARTH_RADIUS_KM * 1.001)
            for latitude in (-60, -30, 0, 30, 60)
        ] + [
            surface_xyz(longitude, np.linspace(-90, 90, 73),
                        EARTH_RADIUS_KM * 1.001)
            for longitude in range(-180, 180, 30)
        ]
        self.coasts = Line3DCollection(self.coasts_itrs, colors="#a1cec6", linewidths=0.65, zorder=3)
        self.grid = Line3DCollection(self.grid_itrs, colors="#44819b", linewidths=0.4, zorder=3)
        self.axes.add_collection3d(self.coasts)
        self.axes.add_collection3d(self.grid)
        self.pole, = self.axes.plot([], [], [], color=MUTED, lw=0.7, zorder=3)
        self.markers, = self.axes.plot([], [], [], linestyle="", marker="o",
                                      color="#80d5e2", markersize=5, picker=7, zorder=5)
        self.selected_marker, = self.axes.plot([], [], [], linestyle="", marker="o",
                                               color=SELECTED_COLOR, markersize=9, zorder=6)

        def button(rect, label, callback):
            axis = self.figure.add_axes(rect, facecolor="#20334a")
            widget = Button(axis, label, color="#20334a", hovercolor="#35516c")
            widget.label.set_color(FOREGROUND)
            widget.on_clicked(callback)
            return widget

        self.play_button = button([0.05, 0.055, 0.09, 0.047], "Pausa", self._toggle)
        self.reset_button = button([0.15, 0.055, 0.09, 0.047], "Ricomincia", self._reset)
        self.previous_button = button([0.755, 0.40, 0.095, 0.04], "Precedente",
                                      lambda _: self.select_satellite(self.selected - 1))
        self.next_button = button([0.86, 0.40, 0.095, 0.04], "Successivo",
                                  lambda _: self.select_satellite(self.selected + 1))
        timeline_axis = self.figure.add_axes([0.14, 0.14, 0.75, 0.025], facecolor=BACKGROUND)
        self.timeline = Slider(timeline_axis, "Ore trascorse", 0, duration_hours,
                               valinit=0, valfmt="%.2f h", color=ORBIT_COLOR,
                               track_color="#20334a")
        speed_axis = self.figure.add_axes([0.43, 0.065, 0.46, 0.025], facecolor=BACKGROUND)
        self.speed_slider = Slider(speed_axis, "Velocità", 30, 7200, valinit=speed,
                                   valfmt="%.0f×", color=ORBIT_COLOR,
                                   track_color="#20334a")
        for slider in (self.timeline, self.speed_slider):
            slider.label.set_color(FOREGROUND)
            slider.valtext.set_color(FOREGROUND)
        self.timeline.on_changed(self._seek)
        self.speed_slider.on_changed(self._change_speed)
        self.figure.text(0.05, 0.017,
                         "Tempo UTC  ·  3600× = 1 ora simulata al secondo  ·  Globo: r = 6 378 km  ·  Coste: Natural Earth",
                         color=MUTED, fontsize=9)

        self.timer = self.figure.canvas.new_timer(interval=50)
        self.timer.add_callback(self._tick)
        self.figure.canvas.mpl_connect("pick_event", self._pick)
        self.figure.canvas.mpl_connect("scroll_event", self._zoom)
        self.figure.canvas.mpl_connect("motion_notify_event", self._camera_changed)
        self.figure.canvas.mpl_connect("button_release_event", self._camera_changed)
        self.figure.canvas.mpl_connect("close_event", lambda _: self.timer.stop())
        self.select_satellite(0)
        self.set_elapsed(0)

    def _set_limits(self, limit):
        for setter in (self.axes.set_xlim3d, self.axes.set_ylim3d, self.axes.set_zlim3d):
            setter(-limit, limit)

    def _eye_direction(self):
        elevation, azimuth = np.deg2rad([self.axes.elev, self.axes.azim])
        return np.array([math.cos(elevation) * math.cos(azimuth),
                         math.cos(elevation) * math.sin(azimuth), math.sin(elevation)])

    def _update_globe(self):
        # Nasconde le coste sull'emisfero posteriore. Matplotlib non possiede
        # un depth buffer 3D completo; usiamo la stessa proiezione ortografica.
        eye = self._eye_direction()
        def visible_lines(lines):
            result = []
            for line in lines:
                rotated = line @ self.state.earth_rotation.T
                rotated[rotated @ eye < 0] = np.nan
                result.append(rotated)
            return result
        self.coasts.set_segments(visible_lines(self.coasts_itrs))
        self.grid.set_segments(visible_lines(self.grid_itrs))
        pole = np.array([[0, 0, -EARTH_RADIUS_KM * 1.25],
                         [0, 0, EARTH_RADIUS_KM * 1.25]]) @ self.state.earth_rotation.T
        self.pole.set_data_3d(*pole.T)
        for line, path in zip(self.orbit_lines, self.scene.orbits_km):
            path = path.copy()
            depth = path @ eye
            projected_radius = np.linalg.norm(path - depth[:, None] * eye, axis=1)
            path[(depth < 0) & (projected_radius < EARTH_RADIUS_KM)] = np.nan
            line.set_data_3d(*path.T)
        xyz = self.state.positions_km.copy()
        depth = xyz @ eye
        projected_radius = np.linalg.norm(xyz - depth[:, None] * eye, axis=1)
        xyz[(depth < 0) & (projected_radius < EARTH_RADIUS_KM)] = np.nan
        self.markers.set_data_3d(*xyz.T)
        selected = xyz[self.selected:self.selected + 1]
        self.selected_marker.set_data_3d(*selected.T)

    def _update_readout(self):
        index = self.selected
        satellite = self.scene.simulation.satellites[index]
        self.clock_text.set_text(self.state.utc.strftime("%d %b %Y\n%H:%M:%S UTC"))
        self.selection_text.set_text(
            f"{satellite.name}\nNORAD {satellite.model.satnum}\n\n"
            f"Latitudine   {self.state.latitude_deg[index]:+.4f}°\n"
            f"Longitudine  {self.state.longitude_deg[index]:+.4f}°\n"
            f"Quota        {self.state.altitude_km[index]:,.0f} km\n\n"
            f"Satellite {index + 1} / {len(self.scene.simulation.satellites)}"
        )

    def select_satellite(self, index):
        self.selected = index % len(self.orbit_lines)
        for i, line in enumerate(self.orbit_lines):
            selected = i == self.selected
            line.set_color(SELECTED_COLOR if selected else ORBIT_COLOR)
            line.set_alpha(0.95 if selected else 0.30)
            line.set_linewidth(1.65 if selected else 0.65)
        self._update_readout()
        self._update_globe()
        self.figure.canvas.draw_idle()

    def set_elapsed(self, seconds):
        if not math.isfinite(seconds):
            raise ValueError("Il tempo trascorso deve essere finito.")
        elapsed = min(max(float(seconds), 0.0), self.duration_seconds)
        state = self.scene.at(self.scene.simulation.start_utc + timedelta(seconds=elapsed))
        self.elapsed, self.state = elapsed, state
        self._last_wall = time.monotonic()
        self._update_globe()
        self._update_readout()
        self.timeline.eventson = False
        try:
            self.timeline.set_val(elapsed / 3600)
        finally:
            self.timeline.eventson = True
        if elapsed >= self.duration_seconds:
            self._set_playing(False)
        self.figure.canvas.draw_idle()

    def _set_playing(self, playing):
        self.playing = playing
        self._last_wall = time.monotonic()
        self.play_button.label.set_text("Pausa" if playing else "Avvia")
        if playing:
            self.timer.start()
        else:
            self.timer.stop()
        self.figure.canvas.draw_idle()

    def _toggle(self, _):
        if not self.playing and self.elapsed >= self.duration_seconds:
            self.set_elapsed(0)
        self._set_playing(not self.playing)

    def _reset(self, _):
        self._set_playing(False)
        self.set_elapsed(0)

    def _seek(self, hours):
        self._set_playing(False)
        self.set_elapsed(hours * 3600)

    def _change_speed(self, speed):
        self.speed = speed
        self._last_wall = time.monotonic()

    def _tick(self):
        if not self.playing:
            return
        now = time.monotonic()
        try:
            self.set_elapsed(self.elapsed + (now - self._last_wall) * self.speed)
            # Il tempo di calcolo/rendering conta nel passo successivo:
            # 3600x resta un'ora simulata per secondo reale.
            if self.playing:
                self._last_wall = now
        except (ValueError, RuntimeError) as exc:
            self._set_playing(False)
            self.selection_text.set_text(str(exc))

    def _pick(self, event):
        if event.artist is self.markers and len(event.ind):
            self.select_satellite(int(event.ind[0]))

    def _camera_changed(self, event):
        if event.inaxes is self.axes and not self.playing:
            self._update_globe()
            self.figure.canvas.draw_idle()

    def _zoom(self, event):
        if event.inaxes is not self.axes:
            return
        current = (self.axes.get_xlim3d()[1] - self.axes.get_xlim3d()[0]) / 2
        limit = np.clip(current * 0.85 ** event.step, EARTH_RADIUS_KM * 1.2, self.limit * 3)
        self._set_limits(limit)
        self.figure.canvas.draw_idle()

    def show(self):
        import matplotlib.pyplot as plt
        self._last_wall = time.monotonic()
        self.timer.start()
        plt.show()


def show_orbits(
    start_utc: UTCInput = DEFAULT_START, duration_hours=24.0, speed=3600.0,
    data_file: str | Path = DEFAULT_DATA_FILE,
):
    """Apre la finestra interattiva; ritorna alla chiusura della finestra."""
    scene = OrbitScene(start_utc, data_file)
    viewer = GlobeViewer(scene, duration_hours, speed)
    viewer.show()
    return viewer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=DEFAULT_START, help="Istante iniziale UTC.")
    parser.add_argument("--hours", type=float, default=24, help="Ore simulate (default: 24).")
    parser.add_argument("--speed", type=float, default=3600, help="Accelerazione 30..7200×.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_FILE, help="File JSON GP.")
    parser.add_argument("--save-preview", type=Path, help="Salva un PNG senza aprire la GUI.")
    args = parser.parse_args()
    try:
        if args.save_preview:
            import matplotlib
            matplotlib.use("Agg")
        scene = OrbitScene(args.start, args.data)
        viewer = GlobeViewer(scene, args.hours, args.speed)
        if args.save_preview:
            viewer._set_playing(False)
            args.save_preview.parent.mkdir(parents=True, exist_ok=True)
            viewer.figure.savefig(args.save_preview, dpi=150, facecolor=BACKGROUND)
            print(f"Anteprima salvata: {args.save_preview.resolve()}")
        else:
            viewer.show()
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        parser.exit(2, f"Errore: {exc}\n")


if __name__ == "__main__":
    main()

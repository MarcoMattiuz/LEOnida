"""Verifiche geometriche e di rendering della visualizzazione orbitale."""

import io
import unittest
from datetime import timedelta
from unittest import mock

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from skyfield.api import wgs84

from eutelsat_globe import EARTH_RADIUS_KM, GlobeViewer, OrbitScene, surface_xyz


class OrbitSceneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene = OrbitScene(orbit_samples=37)
        cls.start = cls.scene.simulation.start_utc

    def test_surface_geometry_uses_earth_fixed_axes_and_physical_radius(self):
        points = surface_xyz(
            np.array([[0.0, 90.0], [180.0, 0.0]]),
            np.array([[0.0, 0.0], [0.0, 90.0]]),
        )
        self.assertEqual(points.shape, (2, 2, 3))
        np.testing.assert_allclose(
            points / EARTH_RADIUS_KM,
            [[[1, 0, 0], [0, 1, 0]], [[-1, 0, 0], [0, 0, 1]]],
            atol=1e-14,
        )
        np.testing.assert_allclose(np.linalg.norm(points, axis=-1), 6378.137)

    def test_all_29_orbits_are_finite_geo_tracks_starting_at_the_initial_state(self):
        orbits = self.scene.orbits_km
        self.assertEqual(orbits.shape, (29, 37, 3))
        self.assertTrue(np.isfinite(orbits).all())
        radii = np.linalg.norm(orbits, axis=-1)
        self.assertTrue((radii > 40000).all())
        self.assertTrue((radii < 44000).all())
        np.testing.assert_allclose(
            orbits[:, 0, :], self.scene.at(self.start).positions_km, atol=1e-5
        )

    def test_earth_rotation_matches_independent_skyfield_ground_sites(self):
        for hours in (0, 6):
            instant = self.start + timedelta(hours=hours)
            state = self.scene.at(instant)
            rotation = state.earth_rotation
            np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(rotation), 1.0, places=12)
            time = self.scene.timescale.from_datetime(instant)
            # All'equatore il raggio della sfera e dell'ellissoide WGS84 coincidono.
            for longitude in (0.0, 90.0):
                with self.subTest(hours=hours, longitude=longitude):
                    expected = wgs84.latlon(0.0, longitude).at(time).position.km
                    actual = rotation @ surface_xyz(longitude, 0.0)
                    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-7)

    def test_terrestrial_coordinates_agree_with_the_existing_demo(self):
        instant = self.start + timedelta(hours=7, seconds=0.123456)
        state = self.scene.at(instant)
        expected = self.scene.simulation.get_coordinates(instant)
        self.assertEqual(state.utc, instant)
        np.testing.assert_allclose(
            state.latitude_deg, [item["latitude_deg"] for item in expected],
            rtol=0, atol=1e-9,
        )
        np.testing.assert_allclose(
            state.longitude_deg, [item["longitude_deg"] for item in expected],
            rtol=0, atol=1e-9,
        )
        self.assertTrue(np.isfinite(state.altitude_km).all())
        self.assertTrue((state.altitude_km > 35000).all())
        self.assertTrue((state.altitude_km < 37000).all())

    def test_satellites_move_in_inertial_frame_and_queries_preserve_epochs(self):
        epochs = tuple(sat.epoch.tt for sat in self.scene.simulation.satellites)
        initial = self.scene.at(self.start)
        later_time = self.start + timedelta(hours=6)
        later = self.scene.at(later_time)
        dot_products = np.sum(initial.positions_km * later.positions_km, axis=1)
        lengths = np.linalg.norm(initial.positions_km, axis=1) * np.linalg.norm(
            later.positions_km, axis=1
        )
        angles = np.degrees(np.arccos(np.clip(dot_products / lengths, -1, 1)))
        self.assertTrue((angles > 80).all())
        self.assertTrue((angles < 100).all())
        self.scene.at(self.start - timedelta(hours=6))
        np.testing.assert_allclose(
            self.scene.at(later_time).positions_km, later.positions_km, rtol=0, atol=1e-9
        )
        self.assertEqual(
            tuple(sat.epoch.tt for sat in self.scene.simulation.satellites), epochs
        )

    def test_viewer_updates_all_markers_and_renders_a_complete_frame(self):
        viewer = GlobeViewer(self.scene, duration_hours=24, speed=3600)
        try:
            self.assertEqual(len(viewer.orbit_lines), 29)
            viewer.set_elapsed(6 * 3600)
            self.assertEqual(viewer.state.utc, self.start + timedelta(hours=6))
            marker_xyz = np.asarray(viewer.markers.get_data_3d()).T
            self.assertEqual(marker_xyz.shape, (29, 3))
            visible = np.isfinite(marker_xyz).all(axis=1)
            hidden = np.isnan(marker_xyz).all(axis=1)
            self.assertTrue((visible | hidden).all())
            self.assertTrue(visible.any())
            np.testing.assert_allclose(
                marker_xyz[visible], viewer.state.positions_km[visible]
            )
            for line in viewer.orbit_lines:
                self.assertEqual(np.asarray(line.get_data_3d()).shape, (3, 37))
            image = io.BytesIO()
            viewer.figure.savefig(image, format="png", dpi=60)
            self.assertTrue(image.getvalue().startswith(b"\x89PNG\r\n\x1a\n"))
            image.seek(0)
            pixels = plt.imread(image, format="png")
            self.assertGreater(pixels.shape[0], 100)
            self.assertGreater(pixels.shape[1], 100)
            self.assertGreater(float(pixels[..., :3].std()), 0.01)
        finally:
            plt.close(viewer.figure)

    def test_animation_clock_includes_time_spent_computing_a_frame(self):
        viewer = GlobeViewer(self.scene, duration_hours=24, speed=3600)
        clock = [1.0]
        original_at = self.scene.at

        def propagate_with_computing_cost(instant):
            result = original_at(instant)
            clock[0] += 0.25
            return result

        try:
            viewer._last_wall = 0.0
            with (
                mock.patch("eutelsat_globe.time.monotonic", side_effect=lambda: clock[0]),
                mock.patch.object(self.scene, "at", side_effect=propagate_with_computing_cost),
                mock.patch.object(viewer.figure.canvas, "draw_idle"),
            ):
                viewer._tick()
                self.assertAlmostEqual(viewer.elapsed, 3600.0)
                clock[0] = 2.0
                viewer._tick()
                self.assertAlmostEqual(viewer.elapsed, 7200.0)
        finally:
            plt.close(viewer.figure)


if __name__ == "__main__":
    unittest.main()

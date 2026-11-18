"""Verifiche della propagazione e del contratto pubblico della demo."""

import json
import math
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest import mock

from eutelsat_demo import DEFAULT_DATA_FILE, init_satellites


class EutelsatDemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim = init_satellites("2026-11-18T12:00:00Z")

    def assertCoordinatesEqual(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for left, right in zip(actual, expected):
            self.assertEqual(left["norad_id"], right["norad_id"])
            self.assertEqual(left["name"], right["name"])
            for field in ("latitude_deg", "longitude_deg"):
                self.assertAlmostEqual(left[field], right[field], delta=1e-9)

    def test_provided_catalog_is_complete_and_coordinates_are_finite(self):
        expected_ids = {
            28924, 29270, 33459, 36101, 37258, 37816, 37836, 38992,
            39020, 39122, 39163, 39612, 39613, 39773, 40425, 40875,
            41310, 41382, 41589, 42741, 44334, 44624, 45027, 49056,
            53765, 54048, 54225, 54259, 59346,
        }
        positions = self.sim.initial_positions
        self.assertEqual(len(positions), 29)
        self.assertEqual({item["norad_id"] for item in positions}, expected_ids)
        for item in positions:
            self.assertEqual(
                set(item), {"name", "norad_id", "latitude_deg", "longitude_deg"}
            )
            self.assertTrue(math.isfinite(item["latitude_deg"]))
            self.assertTrue(math.isfinite(item["longitude_deg"]))
            self.assertLessEqual(abs(item["latitude_deg"]), 90.0)
            self.assertLessEqual(abs(item["longitude_deg"]), 180.0)

    def test_equivalent_timezones_and_subsecond_precision(self):
        expected = self.sim.get_coordinates("2026-11-18T12:00:00.123456Z")
        equivalent_inputs = (
            "2026-11-18T13:30:00.123456+01:30",
            datetime(2026, 11, 18, 12, 0, 0, 123456, tzinfo=timezone.utc),
            datetime(
                2026, 11, 18, 7, 0, 0, 123456,
                tzinfo=timezone(-timedelta(hours=5)),
            ),
        )
        for value in equivalent_inputs:
            with self.subTest(value=value):
                self.assertCoordinatesEqual(self.sim.get_coordinates(value), expected)

        # Mezzo secondo deve cambiare le posizioni, anche per questi satelliti GEO.
        later = self.sim.get_coordinates("2026-11-18T12:00:00.623456Z")
        max_change = max(
            abs(left[field] - right[field])
            for left, right in zip(expected, later)
            for field in ("latitude_deg", "longitude_deg")
        )
        self.assertGreater(max_change, 1e-7)

    def test_naive_dates_are_rejected_at_initialization_and_query(self):
        for value in ("2026-11-18T12:00:00", datetime(2026, 11, 18, 12)):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "timezone"):
                    init_satellites(value)
                with self.assertRaisesRegex(ValueError, "timezone"):
                    self.sim.get_coordinates(value)

    def test_initial_time_does_not_reset_the_orbital_epochs(self):
        later_start = init_satellites("2026-11-18T18:00:00Z")
        self.assertCoordinatesEqual(
            later_start.get_coordinates(self.sim.start_utc),
            self.sim.initial_positions,
        )
        with DEFAULT_DATA_FILE.open(encoding="utf-8") as data_file:
            records = json.load(data_file)
        expected_epochs = {
            record["NORAD_CAT_ID"]: datetime.fromisoformat(record["EPOCH"]).replace(
                tzinfo=timezone.utc
            )
            for record in records
        }
        for simulation in (self.sim, later_start):
            for satellite in simulation.satellites:
                expected = expected_epochs[satellite.model.satnum]
                difference = satellite.epoch.utc_datetime() - expected
                # Consente l'arrotondamento del giorno giuliano nel modello SGP4.
                self.assertLess(abs(difference.total_seconds()), 0.001)

    def test_queries_are_independent_of_order_and_preserve_initial_positions(self):
        initial_snapshot = [dict(item) for item in self.sim.initial_positions]
        future = self.sim.get_coordinates("2026-11-18T18:00:00Z")
        self.sim.get_coordinates("2026-11-18T06:00:00Z")
        self.assertCoordinatesEqual(
            self.sim.get_coordinates("2026-11-18T18:00:00Z"), future
        )
        self.assertCoordinatesEqual(
            self.sim.get_coordinates(self.sim.start_utc), initial_snapshot
        )
        self.assertCoordinatesEqual(self.sim.initial_positions, initial_snapshot)

    def test_inclined_satellite_moves_while_172b_stays_near_equator(self):
        latitudes = {28924: [], 42741: []}
        for hours in (0, 6, 12, 18):
            instant = self.sim.start_utc + timedelta(hours=hours)
            by_id = {
                item["norad_id"]: item for item in self.sim.get_coordinates(instant)
            }
            for norad_id, samples in latitudes.items():
                samples.append(by_id[norad_id]["latitude_deg"])

        inclined = latitudes[28924]  # EUTELSAT 174A: inclinazione circa 4,6 gradi.
        self.assertGreater(max(inclined) - min(inclined), 5.0)
        self.assertLess(max(abs(value) for value in inclined), 5.0)
        self.assertLess(max(abs(value) for value in latitudes[42741]), 0.1)

    def test_default_catalog_works_from_other_directory_without_network(self):
        previous_directory = os.getcwd()
        network_error = AssertionError("La demo non deve accedere alla rete")
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                with (
                    mock.patch("socket.socket.connect", side_effect=network_error),
                    mock.patch("socket.socket.connect_ex", side_effect=network_error),
                    mock.patch("socket.getaddrinfo", side_effect=network_error),
                ):
                    simulation = init_satellites("2026-11-18T12:00:00Z")
                    positions = simulation.get_coordinates("2026-11-18T18:00:00Z")
            finally:
                os.chdir(previous_directory)
        self.assertCoordinatesEqual(simulation.initial_positions, self.sim.initial_positions)
        self.assertEqual(len(positions), 29)

    def test_sgp4_failure_is_reported_with_satellite_identity(self):
        satellite = self.sim.satellites[0]
        failed_position = SimpleNamespace(message="satellite has decayed")
        with mock.patch.object(satellite, "at", return_value=failed_position):
            with self.assertRaisesRegex(
                RuntimeError, r"EUTELSAT 174A.*28924.*satellite has decayed"
            ):
                self.sim.get_coordinates("2026-11-18T18:00:00Z")


if __name__ == "__main__":
    unittest.main()

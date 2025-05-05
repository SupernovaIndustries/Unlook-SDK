"""
Unit test per il client UnLook.
"""

import unittest
import time
import os
import tempfile
import numpy as np
from unittest.mock import patch, MagicMock

from unlook.client import (
    UnlookClient, UnlookClientEvent,
    Calibrator, CalibrationData,
    ScanProcessor, PatternDirection,
    ModelExporter
)
from unlook.common import UnlookScanner, Message, MessageType


class TestUnlookClient(unittest.TestCase):
    """Test per il client UnLook."""

    def setUp(self):
        """Setup per i test."""
        # Crea un client mock
        with patch('unlook.client.scanner.zmq.Context'):
            self.client = UnlookClient(client_name="TestClient", auto_discover=False)

            # Crea uno scanner mock
            self.mock_scanner = UnlookScanner(
                name="MockScanner",
                host="localhost",
                port=5555,
                scanner_uuid="mock-uuid-123"
            )

    def test_client_initialization(self):
        """Testa l'inizializzazione del client."""
        self.assertEqual(self.client.name, "TestClient")
        self.assertFalse(self.client.connected)
        self.assertIsNone(self.client.scanner)

    @patch('unlook.client.scanner.zmq.Context')
    def test_scanner_discovery(self, mock_context):
        """Testa la discovery degli scanner."""
        # Sostituisci il metodo di discovery
        self.client.discovery.start_discovery = MagicMock()
        self.client.discovery.get_scanners = MagicMock(return_value=[self.mock_scanner])

        # Avvia la discovery
        self.client.start_discovery()

        # Verifica che il metodo sia stato chiamato
        self.client.discovery.start_discovery.assert_called_once()

        # Ottieni gli scanner
        scanners = self.client.get_discovered_scanners()

        # Verifica che il mock_scanner sia nell'elenco
        self.assertEqual(len(scanners), 1)
        self.assertEqual(scanners[0].uuid, self.mock_scanner.uuid)

    @patch('unlook.client.scanner.zmq.Context')
    @patch('unlook.client.scanner.zmq.Poller')
    def test_connect_to_scanner(self, mock_poller, mock_context):
        """Testa la connessione a uno scanner."""
        # Configura i mock
        mock_socket = MagicMock()
        mock_socket.recv.return_value = Message(
            msg_type=MessageType.HELLO,
            payload={"scanner_name": "MockScanner", "scanner_uuid": "mock-uuid-123"}
        ).to_bytes()

        mock_context.return_value.socket.return_value = mock_socket

        # Simula una risposta positiva dal poller
        mock_poller_instance = MagicMock()
        mock_poller.return_value = mock_poller_instance
        mock_poller_instance.poll.return_value = {mock_socket: 1}

        # Aggiungi un callback per testare gli eventi
        callback_called = False

        def test_callback(scanner):
            nonlocal callback_called
            callback_called = True
            self.assertEqual(scanner.uuid, self.mock_scanner.uuid)

        self.client.on_event(UnlookClientEvent.CONNECTED, test_callback)

        # Tenta la connessione
        with patch.object(self.client, 'poller', mock_poller_instance):
            result = self.client.connect(self.mock_scanner)

            # Verifica il risultato
            self.assertTrue(result)
            self.assertTrue(self.client.connected)
            self.assertEqual(self.client.scanner.uuid, self.mock_scanner.uuid)

            # Verifica che il callback sia stato chiamato
            self.assertTrue(callback_called)

    def test_calibration_data(self):
        """Testa i dati di calibrazione."""
        # Crea dati di calibrazione
        calib_data = CalibrationData()

        # Imposta alcuni valori
        calib_data.camera_matrix = np.eye(3)
        calib_data.dist_coeffs = np.zeros(5)
        calib_data.projector_matrix = np.eye(3)
        calib_data.projector_dist_coeffs = np.zeros(5)
        calib_data.cam_to_proj_rotation = np.eye(3)
        calib_data.cam_to_proj_translation = np.zeros((3, 1))
        calib_data.calibration_error = 0.5
        calib_data.calibration_date = "2023-01-01"
        calib_data.scanner_uuid = "test-uuid"

        # Verifica validità
        self.assertTrue(calib_data.is_valid())

        # Salva e carica
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            filename = f.name

        try:
            # Salva
            self.assertTrue(calib_data.save(filename))

            # Carica
            loaded_calib = CalibrationData.load(filename)

            # Verifica
            self.assertIsNotNone(loaded_calib)
            self.assertTrue(loaded_calib.is_valid())
            self.assertEqual(loaded_calib.scanner_uuid, calib_data.scanner_uuid)
            self.assertEqual(loaded_calib.calibration_date, calib_data.calibration_date)
            np.testing.assert_array_equal(loaded_calib.camera_matrix, calib_data.camera_matrix)

        finally:
            # Pulisci
            if os.path.exists(filename):
                os.unlink(filename)

    def test_model_exporter(self):
        """Testa l'esportatore di modelli."""
        # Crea un risultato di elaborazione
        result = MagicMock()
        result.has_point_cloud.return_value = True
        result.num_points = 10
        result.point_cloud = np.random.rand(10, 3)
        result.colors = np.random.randint(0, 255, (10, 3), dtype=np.uint8)
        result.confidence = np.random.rand(10)

        # Crea l'esportatore
        exporter = ModelExporter()

        # Test con file temporanei
        with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as f_ply, \
                tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as f_obj, \
                tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as f_xyz:

            ply_file = f_ply.name
            obj_file = f_obj.name
            xyz_file = f_xyz.name

        try:
            # Esporta in vari formati
            self.assertTrue(exporter.export_ply(result, ply_file))
            self.assertTrue(exporter.export_obj(result, obj_file))
            self.assertTrue(exporter.export_xyz(result, xyz_file))

            # Verifica che i file esistano
            self.assertTrue(os.path.exists(ply_file))
            self.assertTrue(os.path.exists(obj_file))
            self.assertTrue(os.path.exists(xyz_file))

            # Verifica che non siano vuoti
            self.assertGreater(os.path.getsize(ply_file), 0)
            self.assertGreater(os.path.getsize(obj_file), 0)
            self.assertGreater(os.path.getsize(xyz_file), 0)

        finally:
            # Pulisci
            for filename in [ply_file, obj_file, xyz_file]:
                if os.path.exists(filename):
                    os.unlink(filename)

    def tearDown(self):
        """Pulizia dopo i test."""
        self.client.disconnect()
        self.client.stop_discovery()


if __name__ == '__main__':
    unittest.main()
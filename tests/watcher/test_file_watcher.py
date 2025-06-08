import unittest
from unittest.mock import patch, MagicMock, call
from pathlib import Path
import time # For __main__ example, not strictly tests
import tempfile
import shutil
from typing import Callable, Optional, List, Union, Any # Added Union and Any for type hints

# Adjust import path as necessary
from src.watcher.file_watcher import FileWatcherService, PythonFileEventHandler
# Also import watchdog components for type hints if needed, or to mock their absence
# For tests, we mostly mock Observer and the handler's base.
try:
    from watchdog.events import FileSystemEvent, FileSystemMovedEvent
except ImportError:
    # Define dummy classes if watchdog is not installed, for type hinting in tests primarily
    # The actual functionality will be mocked.
    class FileSystemEvent: # type: ignore
        def __init__(self, src_path): self.src_path = src_path; self.is_directory = False
    class FileSystemMovedEvent(FileSystemEvent): # type: ignore
        def __init__(self, src_path, dest_path): super().__init__(src_path); self.dest_path = dest_path


class TestPythonFileEventHandler(unittest.TestCase):
    def setUp(self):
        self.mock_callback = MagicMock()
        # Initialize with default patterns, ignore_directories=True
        # We assume PatternMatchingEventHandler itself is tested by the watchdog library.
        # We are testing our specific overrides.
        self.handler = PythonFileEventHandler(callback=self.mock_callback)

    def _create_mock_event(self, src_path_str: str, is_directory: bool = False, event_type: str = "created", dest_path_str: Optional[str] = None):
        if event_type == "moved":
            event = FileSystemMovedEvent(src_path_str, dest_path_str if dest_path_str else "")
        else:
            event = FileSystemEvent(src_path_str)

        event.is_directory = is_directory # type: ignore
        return event

    def test_on_created_py_file(self):
        event = self._create_mock_event("test.py")
        self.handler.on_created(event)
        self.mock_callback.assert_called_once_with("created", Path("test.py"), None)

    def test_on_created_directory(self):
        event = self._create_mock_event("somedir.py", is_directory=True) # Name matches pattern
        self.handler.on_created(event)
        self.mock_callback.assert_not_called()

    def test_on_modified_py_file(self):
        event = self._create_mock_event("test.py")
        self.handler.on_modified(event)
        self.mock_callback.assert_called_once_with("modified", Path("test.py"), None)

    def test_on_modified_directory(self):
        event = self._create_mock_event("somedir.py", is_directory=True)
        self.handler.on_modified(event)
        self.mock_callback.assert_not_called()

    def test_on_deleted_py_file(self):
        event = self._create_mock_event("test.py")
        self.handler.on_deleted(event)
        self.mock_callback.assert_called_once_with("deleted", Path("test.py"), None)

    def test_on_deleted_directory(self):
        event = self._create_mock_event("somedir.py", is_directory=True)
        self.handler.on_deleted(event)
        self.mock_callback.assert_not_called()

    def test_on_moved_py_file(self):
        event = self._create_mock_event("src.py", event_type="moved", dest_path_str="dest.py")
        self.handler.on_moved(event)
        self.mock_callback.assert_called_once_with("moved", Path("src.py"), Path("dest.py"))

    def test_on_moved_directory(self):
        event = self._create_mock_event("srcdir.py", event_type="moved", dest_path_str="destdir.py", is_directory=True)
        self.handler.on_moved(event)
        self.mock_callback.assert_not_called()


class TestFileWatcherService(unittest.TestCase):
    def setUp(self):
        self.watch_path = Path(tempfile.mkdtemp(prefix="watch_svc_"))
        self.mock_on_event_callback = MagicMock()

        # Patch Observer and PythonFileEventHandler for all tests in this class
        self.patcher_observer_class = patch('src.watcher.file_watcher.Observer')
        self.MockObserverClass = self.patcher_observer_class.start()
        self.mock_observer_instance = MagicMock()
        self.MockObserverClass.return_value = self.mock_observer_instance

        self.patcher_event_handler_class = patch('src.watcher.file_watcher.PythonFileEventHandler')
        self.MockEventHandlerClass = self.patcher_event_handler_class.start()
        self.mock_event_handler_instance = MagicMock()
        self.MockEventHandlerClass.return_value = self.mock_event_handler_instance

        # For testing watchdog import status
        self.patcher_watchdog_observer_module = patch('src.watcher.file_watcher.Observer', self.MockObserverClass)
        self.patcher_watchdog_pme_module = patch('src.watcher.file_watcher.PatternMatchingEventHandler', MagicMock)
        self.patcher_watchdog_fse_module = patch('src.watcher.file_watcher.FileSystemEvent', MagicMock)
        self.patcher_watchdog_fsme_module = patch('src.watcher.file_watcher.FileSystemMovedEvent', MagicMock)

        self.watchdog_observer_module = self.patcher_watchdog_observer_module.start()
        self.watchdog_pme_module = self.patcher_watchdog_pme_module.start()
        self.watchdog_fse_module = self.patcher_watchdog_fse_module.start()
        self.watchdog_fsme_module = self.patcher_watchdog_fsme_module.start()


    def tearDown(self):
        shutil.rmtree(self.watch_path)
        self.patcher_observer_class.stop()
        self.patcher_event_handler_class.stop()
        self.patcher_watchdog_observer_module.stop()
        self.patcher_watchdog_pme_module.stop()
        self.patcher_watchdog_fse_module.stop()
        self.patcher_watchdog_fsme_module.stop()


    def test_init_success(self):
        service = FileWatcherService(self.watch_path, self.mock_on_event_callback)
        self.MockEventHandlerClass.assert_called_once_with(self.mock_on_event_callback)
        self.MockObserverClass.assert_called_once()
        self.assertIsNotNone(service.observer)
        self.assertIsNotNone(service.event_handler)

    def test_init_path_not_dir_raises_value_error(self):
        file_path = self.watch_path / "file.txt"
        with open(file_path, "w") as f: f.write("test")
        with self.assertRaises(ValueError):
            FileWatcherService(file_path, self.mock_on_event_callback)

    @patch('src.watcher.file_watcher.Observer', None)
    def test_init_watchdog_not_available(self, mock_obs_none):
        # This test needs to re-patch the global Observer inside file_watcher to None
        # The setUp one might interfere.
        self.patcher_watchdog_observer_module.stop() # Stop setUp patch for Observer
        patcher_temp_obs_none = patch('src.watcher.file_watcher.Observer', None)
        MockObserverNone = patcher_temp_obs_none.start()

        with patch('builtins.print') as mock_print:
            service = FileWatcherService(self.watch_path, self.mock_on_event_callback)

        self.assertIsNone(service.observer)
        self.assertIsNone(service.event_handler)
        # Check if the specific print warning for watchdog components not available was called
        self.assertTrue(any("Watchdog library components not available" in c.args[0] for c in mock_print.call_args_list))

        patcher_temp_obs_none.stop()
        self.watchdog_observer_module = self.patcher_watchdog_observer_module.start() # Restart setUp patch


    def test_start_observer(self):
        service = FileWatcherService(self.watch_path, self.mock_on_event_callback)
        self.mock_observer_instance.is_alive.return_value = False

        service.start()

        self.mock_observer_instance.schedule.assert_called_once_with(
            self.mock_event_handler_instance, str(self.watch_path), recursive=True
        )
        self.mock_observer_instance.start.assert_called_once()

    def test_start_observer_already_running(self):
        service = FileWatcherService(self.watch_path, self.mock_on_event_callback)
        self.mock_observer_instance.is_alive.return_value = True
        service.start()
        self.mock_observer_instance.schedule.assert_not_called() # Should not be called again if already alive
        self.mock_observer_instance.start.assert_not_called()

    def test_start_observer_not_initialized(self):
        # Simulate Observer being None due to watchdog not being installed
        self.patcher_watchdog_observer_module.stop() # Stop setUp patch for Observer
        patcher_temp_obs_none = patch('src.watcher.file_watcher.Observer', None)
        MockObserverNone = patcher_temp_obs_none.start()

        with patch('builtins.print') as mock_print:
            service = FileWatcherService(self.watch_path, self.mock_on_event_callback) # Observer will be None
            service.start()

        self.assertTrue(any("Observer not initialized" in str(c.args) for c in mock_print.call_args_list))

        patcher_temp_obs_none.stop()
        self.watchdog_observer_module = self.patcher_watchdog_observer_module.start() # Restart setUp patch


    def test_stop_observer(self):
        service = FileWatcherService(self.watch_path, self.mock_on_event_callback)
        self.mock_observer_instance.is_alive.return_value = True

        service.stop()

        self.mock_observer_instance.stop.assert_called_once()
        self.mock_observer_instance.join.assert_called_once()
        self.assertIsNotNone(service.observer)
        self.assertNotEqual(service.observer, self.mock_observer_instance) # Should be a new instance

    def test_stop_observer_not_running(self):
        service = FileWatcherService(self.watch_path, self.mock_on_event_callback)
        self.mock_observer_instance.is_alive.return_value = False
        service.stop()
        self.mock_observer_instance.stop.assert_not_called()
        self.mock_observer_instance.join.assert_not_called()


if __name__ == '__main__':
    unittest.main()

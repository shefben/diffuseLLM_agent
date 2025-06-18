import time
from pathlib import Path
from typing import Callable, Optional, Union

try:
    from watchdog.observers import Observer
    from watchdog.events import (
        PatternMatchingEventHandler,
        FileSystemEvent,
        FileSystemMovedEvent,
    )
except ImportError:
    Observer = None  # type: ignore
    PatternMatchingEventHandler = None  # type: ignore
    FileSystemEvent = None  # type: ignore
    FileSystemMovedEvent = None  # type: ignore
    print("Warning: 'watchdog' library not found. FileWatcherService will be disabled.")


class PythonFileEventHandler(PatternMatchingEventHandler):  # type: ignore
    """
    Custom event handler that reacts only to events for .py files
    and calls a unified callback.
    """

    def __init__(self, callback: Callable[[str, Path, Optional[Path]], None]):
        # Patterns to match only .py files.
        # ignore_patterns can be used for things like .pyc or temporary .py~ files
        super().__init__(
            patterns=["*.py"],
            ignore_patterns=["*.pyc", "*.~py"],
            ignore_directories=True,
        )
        self.callback = callback

    def on_created(self, event: FileSystemEvent):
        if not event.is_directory:
            self.callback(
                "created", Path(event.src_path), None
            )  # Pass None for dest_path

    def on_modified(self, event: FileSystemEvent):
        if not event.is_directory:
            self.callback(
                "modified", Path(event.src_path), None
            )  # Pass None for dest_path

    def on_deleted(self, event: FileSystemEvent):
        if not event.is_directory:
            self.callback(
                "deleted", Path(event.src_path), None
            )  # Pass None for dest_path

    def on_moved(self, event: FileSystemMovedEvent):  # Specific event type for moved
        if not event.is_directory:  # is_directory applies to src_path for moved event
            self.callback("moved", Path(event.src_path), Path(event.dest_path))


class FileWatcherService:
    def __init__(
        self,
        path_to_watch: Union[str, Path],
        on_event_callback: Callable[
            [str, Path, Optional[Path]], None
        ],  # event_type, src_path, dest_path (for moved)
    ):
        """
        Initializes the file watcher service.

        Args:
            path_to_watch: The directory path to monitor for file changes.
            on_event_callback: A callback function to invoke when a relevant file event occurs.
                               It should accept (event_type: str, src_path: Path, dest_path: Optional[Path]).
        """
        self.watch_path = Path(path_to_watch).resolve()
        if not self.watch_path.is_dir():
            raise ValueError(
                f"Path to watch must be a valid directory: {self.watch_path}"
            )

        self.callback = on_event_callback
        self.event_handler: Optional[PythonFileEventHandler] = None
        self.observer: Optional[Observer] = None

        if (
            Observer
            and PatternMatchingEventHandler
            and FileSystemEvent
            and FileSystemMovedEvent
        ):
            self.event_handler = PythonFileEventHandler(self.callback)
            self.observer = Observer()
        else:
            print(
                "FileWatcherService: Watchdog library components not available. Watcher is disabled."
            )

    def start(self) -> None:
        """Starts the file system observer in a separate thread."""
        if not self.observer or not self.event_handler:
            print(
                "FileWatcherService: Observer not initialized (watchdog missing or init failed). Cannot start."
            )
            return

        # Check if the observer is already alive before trying to start.
        # If it was stopped, it needs to be a new instance or re-scheduled on the same instance
        # if the library allows (watchdog typically needs a new instance after join()).
        if self.observer.is_alive():
            print("FileWatcherService: Observer is already running.")
            return

        # Ensure the observer is (re)configured before starting
        # This handles the case where observer.join() was called and a new instance was created in stop()
        # Or if it's the first start.
        # Note: A new Observer instance is clean, no need to unschedule.
        try:
            self.observer.schedule(
                self.event_handler, str(self.watch_path), recursive=True
            )
            self.observer.start()
            print(
                f"FileWatcherService: Started watching directory '{self.watch_path}' for .py file changes."
            )
        except Exception as e:
            print(f"FileWatcherService: Error starting observer: {e}")
            # Attempt to clean up and allow potential restart by creating a new observer instance
            if Observer:
                self.observer = Observer()
            else:
                self.observer = None

    def stop(self) -> None:
        """Stops the file system observer thread."""
        if self.observer and self.observer.is_alive():
            try:
                self.observer.stop()
                self.observer.join()  # Wait for the thread to terminate
                print("FileWatcherService: Stopped watching.")
            except Exception as e:
                print(f"FileWatcherService: Error stopping observer: {e}")
        else:
            print("FileWatcherService: Observer is not running or not initialized.")

        # For watchdog, a new Observer instance is needed to restart after join().
        if Observer:
            self.observer = Observer()
        else:
            self.observer = None


if __name__ == "__main__":
    import tempfile  # Ensure tempfile is imported for the __main__ block
    import shutil  # Ensure shutil is imported for the __main__ block

    if Observer is None:
        print("Watchdog library not available, cannot run example.")
    else:
        temp_watch_dir = Path(tempfile.mkdtemp(prefix="test_watch_"))
        print(f"Test directory created: {temp_watch_dir}")

        def my_callback(
            event_type: str, src_path: Path, dest_path: Optional[Path] = None
        ):
            print(f"Event: {event_type}, Src: {src_path}", end="")
            if dest_path:
                print(f", Dest: {dest_path}", end="")
            print()  # Newline

        watcher = FileWatcherService(temp_watch_dir, my_callback)
        watcher.start()

        print(
            f"Watcher started. Try creating/modifying/deleting/moving .py files in {temp_watch_dir}"
        )
        print("For example:")
        print(f"  touch {temp_watch_dir / 'a.py'}")
        print(f"  echo '# mod' >> {temp_watch_dir / 'a.py'}")
        print(f"  mv {temp_watch_dir / 'a.py'} {temp_watch_dir / 'b.py'}")
        print(f"  rm {temp_watch_dir / 'b.py'}")
        print(f"  mkdir {temp_watch_dir / 'subdir'}")
        print(f"  touch {temp_watch_dir / 'subdir' / 'c.py'}")
        print("Press Ctrl+C to stop.")

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received.")
        finally:
            watcher.stop()
            try:
                shutil.rmtree(temp_watch_dir)
                print(f"Cleaned up test directory: {temp_watch_dir}")
            except Exception as e_clean:
                print(f"Error cleaning up test directory {temp_watch_dir}: {e_clean}")

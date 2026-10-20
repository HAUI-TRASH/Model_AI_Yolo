"""Write checkpoints without exposing a partially written active file."""
import os
from pathlib import Path
import tempfile


def save_checkpoint_atomic(state, destination, save_fn):
    """Serialize to a sibling temporary file, then atomically replace destination.

    save_fn must accept (state, binary_file), as torch.save does. Any failure
    before replacement leaves the existing checkpoint intact.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "wb") as checkpoint_file:
            save_fn(state, checkpoint_file)
            checkpoint_file.flush()
            os.fsync(checkpoint_file.fileno())
        os.replace(temporary_path, destination)
    finally:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass

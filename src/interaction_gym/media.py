"""Content-addressed media store shared by all episodes.

Files live at ``<root>/media/<sha[:2]>/<sha>.wav``; identical audio is stored once.
Trajectories only hold references: ``{"kind", "uri", "sr", "start_ms", "end_ms"}``
with ``uri`` relative to the root, so one file can back many (partial) references.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from .audio import Audio


class MediaStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def put(self, audio: Audio) -> str:
        """Store ``audio`` (if new) and return its uri."""
        h = hashlib.sha256(audio.sr.to_bytes(4, "little") + audio.samples.tobytes()).hexdigest()
        uri = f"media/{h[:2]}/{h}.wav"
        path = self.root / uri
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            audio.write_wav(path)
        return uri

    def ref(self, audio: Audio, start_ms: int = 0, end_ms: int | None = None) -> dict:
        return {"kind": "audio", "uri": self.put(audio), "sr": audio.sr, "start_ms": start_ms,
                "end_ms": audio.dur_ms if end_ms is None else end_ms}

    def load(self, ref: dict) -> Audio:
        """The referenced slice."""
        audio = Audio.read_wav(self.root / ref["uri"])
        a, b = (round(ms * audio.sr / 1000) for ms in (ref["start_ms"], ref["end_ms"]))
        return audio[a:b]

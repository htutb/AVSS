from pathlib import Path

import torchaudio

from src.asr_datasets.base_dataset import BaseDataset


class CustomDirDataset(BaseDataset):
    """
    Custom dataset for ASR inference or evaluation.

    Expected structure:
    dataset_dir/
      ├── audio/
      │     ├── file_001.wav
      │     ├── file_002.flac
      │     └── ...
      └── transcriptions/   (optional)
            ├── file_001.txt
            ├── file_002.txt
            └── ...

    """

    AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".m4a")

    def __init__(self, dataset_dir: str, *args, **kwargs):
        dataset_dir = Path(dataset_dir)
        audio_dir = dataset_dir / "audio"
        transcription_dir = dataset_dir / "transcriptions"

        if not audio_dir.exists():
            raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

        data = []
        for audio_file in sorted(audio_dir.iterdir()):
            if not audio_file.suffix.lower() in self.AUDIO_EXTENSIONS:
                continue

            entry = {"path": str(audio_file)}
            info = torchaudio.info(str(audio_file))
            entry["audio_len"] = info.num_frames / info.sample_rate

            text = ""
            if transcription_dir.exists():
                transc_file = transcription_dir / f"{audio_file.stem}.txt"
                if transc_file.exists():
                    text = transc_file.read_text(encoding="utf-8").strip().lower()
                    text = " ".join(text.split())
                else:
                    print(f"No transcription for audio: {audio_file.name}")

            entry["text"] = text
            data.append(entry)

        if len(data) == 0:
            raise RuntimeError(f"No audio files found in {audio_dir}")

        super().__init__(data, *args, **kwargs)

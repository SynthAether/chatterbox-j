import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torchaudio as ta
import torch
from torch.utils.data import Dataset


def _pair_wav_txt(root: Path) -> List[Tuple[Path, Path]]:
    """
    Discover (wav, txt) pairs under a 'split' root that contains subfolders like:
      - questions/{wavs, mfa_input_txt}
      - sentences/{wavs, mfa_input_txt}
    We match by basename: for wavs/foo.wav we expect mfa_input_txt/foo.txt.
    """
    pairs: List[Tuple[Path, Path]] = []
    if not root.exists():
        return pairs

    def collect_from(subdir: Path):
        wavs = subdir / "wavs"
        txts = subdir / "mfa_input_txt"
        if not (wavs.is_dir() and txts.is_dir()):
            return
        for wav in wavs.glob("*.wav"):
            txt = txts / (wav.stem + ".txt")
            if txt.exists():
                pairs.append((wav, txt))

    # typical subfolders: questions/, sentences/, or arbitrary nested dirs
    for child in root.rglob("*"):
        if child.is_dir() and child.name in ("questions", "sentences"):
            collect_from(child)
        # also handle one-level sub-corpora like '2007-affirmative', '2007-question'
        elif child.is_dir():
            # if this dir contains wavs/ and mfa_input_txt/, collect directly
            if (child / "wavs").is_dir() and (child / "mfa_input_txt").is_dir():
                collect_from(child)

    return pairs


class LocalChatterboxDataset(Dataset):
    """
    Minimal local dataset for chatterbox-j finetuning.
    Each item returns a dict containing:
      - 'text': normalized text string (raw here; normalization happens upstream)
      - 'audio': { 'array': float32 tensor, 'sampling_rate': int }  (HF-compatible)
      - optionally you can return 'path' keys for debugging.
    We do *not* add BOS/EOS here; we keep behavior identical to chatterbox-j's collator.
    """
    def __init__(
        self,
        data_root: str,
        splits: Optional[List[str]] = None,
        target_sr: int = 16000,
    ):
        self.data_root = Path(data_root)
        self.target_sr = target_sr

        # If splits is None, use all first-level dirs under data_root
        split_dirs: List[Path] = []
        if splits:
            for s in splits:
                split_dirs.append(self.data_root / s)
        else:
            split_dirs = [p for p in self.data_root.iterdir() if p.is_dir()]

        pairs: List[Tuple[Path, Path]] = []
        for sd in split_dirs:
            pairs.extend(_pair_wav_txt(sd))

        self.items = pairs

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        wav_path, txt_path = self.items[idx]
        # Load text (raw; any punctuation normalization remains in the finetune pipeline)
        text = txt_path.read_text(encoding="utf-8").strip()

        # Load audio and resample to tokenizer/VE SR (chatterbox uses 16 kHz upstream)
        wav, sr = ta.load(str(wav_path))
        if sr != self.target_sr:
            wav = ta.functional.resample(wav, sr, self.target_sr)
            sr = self.target_sr
        # ensure mono
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav.squeeze(0).contiguous()

        return {
            "text": text,
            "audio": {"array": wav, "sampling_rate": sr},
            "wav_path": str(wav_path),
            "txt_path": str(txt_path),
        }

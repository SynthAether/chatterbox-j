#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path

import torch
import torchaudio as ta

try:
    import sox
except ImportError:  # pragma: no cover - optional dependency
    sox = None


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(ckpt_dir: Path | None, device: str):
    from safetensors.torch import load_file
    from chatterbox.tts import (
        ChatterboxTTS,
        Conditionals,
        EnTokenizer,
        VoiceEncoder,
        S3Gen,
        smart_load_t3_model,
    )

    if ckpt_dir:
        ckpt_dir = Path(ckpt_dir)
        if device in ["cpu", "mps"]:
            map_location = torch.device("cpu")
        else:
            map_location = None

        ve = VoiceEncoder()
        ve.load_state_dict(load_file(ckpt_dir / "ve.safetensors"))
        ve.to(device).eval()

        t3_state = torch.load(ckpt_dir / "t3_cfg.pt", map_location="cpu")
        if any(k.startswith("t3.") for k in t3_state):
            t3_state = {k.partition("t3.")[2]: v for k, v in t3_state.items()}
        t3 = smart_load_t3_model(t3_state, device)

        s3gen = S3Gen()
        s3gen.load_state_dict(load_file(ckpt_dir / "s3gen.safetensors"), strict=False)
        s3gen.to(device).eval()

        tokenizer = EnTokenizer(str(ckpt_dir / "tokenizer.json"))

        conds = None
        conds_path = ckpt_dir / "conds.pt"
        if conds_path.exists():
            conds = Conditionals.load(conds_path, map_location=map_location).to(device)

        return ChatterboxTTS(t3, s3gen, ve, tokenizer, device, conds=conds)
    return ChatterboxTTS.from_pretrained(device=device)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate TTS audio for each .txt in a folder using ChatterboxTTS."
    )
    parser.add_argument("--txt_dir", type=str, required=True,
                        help="Directory containing .txt files, each with one utterance per file.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory where synthesized .wav files will be saved.")
    parser.add_argument("--audio_prompt", type=str, default=None,
                        help="(Optional) Path to a .wav file to use as the voice prompt for cloning.")
    parser.add_argument("--gain_db", type=float, default=None,
                        help="(Optional) Gain normalization target in dB. Requires python-sox.")
    parser.add_argument("--device", type=str, default=detect_device(),
                        help="Device to run the model on (e.g. 'cuda', 'mps' or 'cpu').")
    parser.add_argument("--ckpt_dir", type=str, default=None,
                        help="(Optional) Path to directory with local checkpoint files for ChatterboxTTS.")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature.")
    parser.add_argument("--cfg_weight", type=float, default=0.5,
                        help="Classifier-free guidance weight.")
    parser.add_argument("--exaggeration", type=float, default=0.5,
                        help="Emotion exaggeration level.")
    parser.add_argument("--debug_generation", action="store_true",
                        help="Print debugging information during generation")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    if args.device == "mps" and not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because the current PyTorch install was not built with MPS enabled.")
        else:
            print("MPS not available because the current MacOS version is not 12.3+ or no compatible device was found.")
        args.device = "cpu"
    elif args.device == "mps":
        map_location = torch.device("mps")
        _torch_load = torch.load

        def patched_torch_load(*args_, **kwargs_):
            if "map_location" not in kwargs_:
                kwargs_["map_location"] = map_location
            return _torch_load(*args_, **kwargs_)

        torch.load = patched_torch_load

    txt_dir = Path(args.txt_dir)
    if not txt_dir.is_dir():
        raise ValueError(f"{txt_dir} is not a valid directory")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.gain_db is not None and sox is None:
        raise ImportError("You requested --gain_db, but python-sox is not installed.")

    model = load_model(Path(args.ckpt_dir) if args.ckpt_dir else None, args.device)
    logger.info("Loaded model on %s", args.device)

    txt_files = sorted(txt_dir.glob("*.txt"))
    if not txt_files:
        logger.warning("No .txt files found in %s", txt_dir)
        return

    for txt_path in txt_files:
        text = txt_path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            logger.info("Skipping empty file: %s", txt_path.name)
            continue

        if args.debug_generation:
            logger.info("Processing %s", txt_path.name)
            logger.info("Text: %s", text)
            text_tokens = model.tokenizer.text_to_tokens(text)
            logger.info("Text tokens shape: %s", text_tokens.shape)
            logger.info("Text tokens (first 20): %s", text_tokens.flatten()[:20].tolist())

        try:
            wav = model.generate(
                text,
                audio_prompt_path=args.audio_prompt,
                temperature=args.temperature,
                cfg_weight=args.cfg_weight,
                exaggeration=args.exaggeration,
            )
        except Exception as e:  # pragma: no cover - debugging path
            logger.error("ERROR generating audio for %s: %s", txt_path.name, e)
            if args.debug_generation:
                import traceback
                traceback.print_exc()
            continue

        if args.debug_generation:
            duration = wav.shape[-1] / model.sr
            logger.info("Generated audio duration: %.2fs", duration)
            if duration < 0.5:
                logger.warning("Generated audio is very short!")

        out_wav_path = output_dir / f"{txt_path.stem}.wav"
        ta.save(str(out_wav_path), wav, model.sr)

        if args.gain_db is not None:
            temp_norm_path = output_dir / f"{txt_path.stem}.norm.wav"
            transformer = sox.Transformer()
            transformer.norm(args.gain_db)
            transformer.build(str(out_wav_path), str(temp_norm_path))
            temp_norm_path.replace(out_wav_path)

        logger.info("[Synthesized] %s -> %s", txt_path.name, out_wav_path.name)


if __name__ == "__main__":
    main()

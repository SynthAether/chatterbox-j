import argparse
import logging
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file

PRETRAINED_FILES = ["ve.safetensors", "s3gen.safetensors", "tokenizer.json", "conds.pt"]

def parse_args():
    parser = argparse.ArgumentParser(
        description="Finalize T3 fine-tuning by extracting t3_cfg.pt from a trainer checkpoint"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint directory or model.safetensors file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to write extracted files")
    parser.add_argument("--local_model_dir", type=str, default=None,
                        help="Path to local base model directory")
    parser.add_argument("--model_name_or_path", type=str, default=None,
                        help="HF repo id for the base model if not using local_model_dir")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to load model on")
    parser.add_argument("--offline", action="store_true",
                        help="Do not attempt to download base model if not found locally")
    return parser.parse_args()


def load_base_model(checkpoint_path: Path, local_dir: Path | None, repo_id: str | None,
                     output_dir: Path, *, offline: bool = False):
    from chatterbox.tts import ChatterboxTTS, REPO_ID
    if local_dir and local_dir.exists():
        model_dir = local_dir
    else:
        if offline:
            raise FileNotFoundError("Base model not found and --offline specified")
        from huggingface_hub import hf_hub_download

        repo = repo_id or REPO_ID
        download_dir = output_dir / "pretrained_model"
        download_dir.mkdir(parents=True, exist_ok=True)
        for fname in PRETRAINED_FILES + ["t3_cfg.safetensors"]:
            try:
                hf_hub_download(
                    repo_id=repo,
                    filename=fname,
                    local_dir=download_dir,
                    local_dir_use_symlinks=False,
                )
            except Exception as e:  # noqa: BLE001
                logging.getLogger(__name__).warning("Could not download %s: %s", fname, e)
        model_dir = download_dir

    model = ChatterboxTTS.from_local(ckpt_dir=str(model_dir), device="cpu")
    return model, model_dir


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    ckpt_path = Path(args.checkpoint)
    if ckpt_path.is_dir():
        ckpt_file = ckpt_path / "model.safetensors"
    else:
        ckpt_file = ckpt_path
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint file {ckpt_file} not found")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_model, base_dir = load_base_model(
        ckpt_path,
        Path(args.local_model_dir) if args.local_model_dir else None,
        args.model_name_or_path,
        output_dir,
        offline=args.offline,
    )

    state = load_file(str(ckpt_file))
    if any(k.startswith("module.") for k in state):
        state = {k.partition("module.")[2]: v for k, v in state.items()}
    if any(k.startswith("t3.") for k in state):
        state = {k.partition("t3.")[2]: v for k, v in state.items()}

    missing, unexpected = base_model.t3.load_state_dict(state, strict=False)
    if missing:
        logging.getLogger(__name__).debug("Missing keys when loading checkpoint: %s", missing)
    if unexpected:
        logging.getLogger(__name__).debug("Unexpected keys when loading checkpoint: %s", unexpected)

    out_path = output_dir / "t3_cfg.pt"
    torch.save(base_model.t3.state_dict(), out_path)
    logging.getLogger(__name__).info("Saved T3 weights to %s", out_path)

    for fname in PRETRAINED_FILES:
        src = base_dir / fname
        if src.exists():
            shutil.copy2(src, output_dir / fname)


if __name__ == "__main__":
    main()

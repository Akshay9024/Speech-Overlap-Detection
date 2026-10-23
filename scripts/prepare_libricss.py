from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from overlap.eval.reference import build_annotation  # noqa: E402

GDRIVE_ID = "1Piioxd5G_85K9Bhcr8ebdhXx0CnaHy7l"  # for_release.zip (official LibriCSS)
OVERLAP_RATIOS = ["0L", "0S", "OV10", "OV20", "OV30", "OV40"]
AUDIO_PATHS = {"mdm": ("record", "raw_recording.wav"), "ihm-mix": ("clean", "mix.wav")}


def download_libricss(dest: Path) -> Path:
    """Download and extract for_release.zip via gdown. Returns the for_release dir."""
    import gdown

    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / "for_release.zip"
    if not zip_path.exists():
        gdown.download(id=GDRIVE_ID, output=str(zip_path), quiet=False)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)
    corpus = dest / "for_release"
    if not corpus.is_dir():
        raise FileNotFoundError(f"expected {corpus} after extraction")
    return corpus


def parse_meeting_info(path: Path) -> list[tuple[str, float, float]]:
    """Parse transcription/meeting_info.txt -> (speaker, start, end) per utterance.
    Tab-separated, one header line, columns: start, end, speaker, utt_id, text."""
    segments: list[tuple[str, float, float]] = []
    with open(path) as f:
        next(f)  # header
        for line in f:
            if not line.strip():
                continue
            start, end, speaker, _utt_id, _text = line.rstrip("\n").split("\t")
            segments.append((speaker, float(start), float(end)))
    return segments


def prepare(corpus_dir: Path, out_dir: Path, audio_type: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rttm_dir = out_dir / "rttm"
    rttm_dir.mkdir(exist_ok=True)
    sub, wav_name = AUDIO_PATHS[audio_type]

    files = []
    for ov in OVERLAP_RATIOS:
        cond_dir = corpus_dir / ov
        if not cond_dir.is_dir():
            continue
        for session in sorted(cond_dir.iterdir()):
            if not session.is_dir():
                continue
            audio = session / sub / wav_name
            info = session / "transcription" / "meeting_info.txt"
            if not audio.is_file() or not info.is_file():
                print(f"skip {session.name}: missing audio or meeting_info", file=sys.stderr)
                continue
            uri = session.name
            annotation = build_annotation(parse_meeting_info(info), uri=uri)
            rttm_path = rttm_dir / f"{uri}.rttm"
            with open(rttm_path, "w") as rf:
                annotation.write_rttm(rf)
            files.append({
                "uri": uri,
                "audio": str(audio),
                "rttm": str(rttm_path),
                "condition": ov,
            })

    manifest = {
        "dataset": "libricss",
        "seen_domain": False,  # LibriCSS is NOT in segmentation-3.0 training data
        "audio_type": audio_type,
        "files": files,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare LibriCSS for overlap evaluation")
    parser.add_argument("--corpus-dir", required=True, help="path to extracted for_release dir (or download target with --download)")
    parser.add_argument("--download", action="store_true", help="download+extract for_release.zip into --corpus-dir first")
    parser.add_argument("--audio-type", choices=list(AUDIO_PATHS), default="mdm", help="mdm=far-field 7ch (default), ihm-mix=close-talk mixture")
    parser.add_argument("--out-dir", default="data/libricss")
    args = parser.parse_args()

    corpus_dir = Path(args.corpus_dir)
    if args.download:
        corpus_dir = download_libricss(corpus_dir)

    manifest = prepare(corpus_dir, Path(args.out_dir), args.audio_type)
    print(f"prepared {len(manifest['files'])} LibriCSS sessions ({args.audio_type}) -> {args.out_dir}/manifest.json")


if __name__ == "__main__":
    main()

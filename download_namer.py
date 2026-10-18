from pathlib import Path
import clip

root = Path(__file__).resolve().parent
clip.load("ViT-B/32", device="cpu", download_root=str(root / ".runtime" / "clip"))
print("Crop naming model ready for offline use.")

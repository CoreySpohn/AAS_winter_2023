import os
from pathlib import Path

folders = ["fig1", "fig2", "fig3", "fig4", "fig5", "fig6", "fig7", "fig8", "fig9"]
for folder in folders:
    full_path = Path(f"figures/{folder}")
    if folder == "fig9":
        os.system(
            f"ffmpeg -i {full_path}/{folder}_%03d.png -framerate 1 -y -b:v 50M {full_path}.avi"
        )
    else:
        os.system(
            f"ffmpeg -i {full_path}/{folder}_%03d.png -framerate 25 -y -b:v 50M {full_path}.avi"
        )

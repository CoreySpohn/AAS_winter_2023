import os
from pathlib import Path

folders = ["fig1", "fig2", "fig3", "fig4", "fig5", "fig6", "fig7", "fig8", "fig9"]
lengths = [12, 12, 12, 12, 12, 12, 12, 12, 28]
for folder, length in zip(folders, lengths):
    full_path = Path(f"figures/{folder}")
    n_images = len(list(full_path.glob("*.png")))
    framerate = n_images / length
    os.system(
        (
            f"ffmpeg -f image2 -r {framerate} -i {full_path}/{folder}_%03d.png -y -c:v libx264 -preset slow  -profile:v high -level:v 4.0 -pix_fmt yuv420p -crf 20 -codec:a aac {full_path}.mp4"
            # f"ffmpeg -f image2 -r {framerate} -i {full_path}/{folder}_%03d.png -y "
            # f"-vcodec libx264 -crf 0 -pix_fmt yuv420p {full_path}.mp4"
            # f"ffmpeg -f image2 -i {full_path}/{folder}_%03d.png -r {framerate} -y "
            # f"-vcodec libx264 -crf 10 -pix_fmt yuv420p {full_path}.mp4"
        )
    )

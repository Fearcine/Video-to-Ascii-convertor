
from PIL import Image, ImageDraw, ImageFont
import os


def generate_icon(output_path: str = "icon.ico"):
    sizes = [16, 32, 48, 64, 128, 256]
    images = []

    for size in sizes:
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        
        for y in range(size):
            t = y / size
            r = int(15 + t * 20)
            g = int(20 + t * 30)
            b = int(50 + t * 40)
            draw.line([(0, y), (size - 1, y)], fill=(r, g, b, 255))

        
        draw.rectangle([0, 0, size - 1, size - 1], outline=(79, 195, 247, 180), width=max(1, size // 32))

        
        font_size = max(8, size // 3)
        try:
            font = ImageFont.truetype("consola.ttf", font_size)
        except (OSError, IOError):
            try:
                font = ImageFont.truetype("cour.ttf", font_size)
            except (OSError, IOError):
                font = ImageFont.load_default()

        text = ">_"
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        tx = (size - tw) // 2
        ty = (size - th) // 2

      
        draw.text((tx + 1, ty + 1), text, fill=(0, 0, 0, 160), font=font)
     
        draw.text((tx, ty), text, fill=(79, 195, 247, 255), font=font)

        images.append(img)

    images[0].save(
        output_path,
        format="ICO",
        sizes=[(s, s) for s in sizes],
        append_images=images[1:],
    )
    print(f"Icon saved to: {os.path.abspath(output_path)}")


if __name__ == "__main__":
    generate_icon()

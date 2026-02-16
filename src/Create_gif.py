import imageio.v2 as imageio
import os
import numpy as np

def main():

    input_folder = "outputs"
    output_path = "result/optimization.gif"

    os.makedirs("result", exist_ok=True)

    images = []

    # Sort by iteration number
    filenames = sorted(
        os.listdir(input_folder),
        key=lambda x: int(x.split("_")[1].split(".")[0])
    )


    for filename in filenames:
        if filename.endswith(".png"):
            img_path = os.path.join(input_folder, filename)
            img = imageio.imread(img_path)

            # Convert all images to RGB
            if len(img.shape) == 3 and img.shape[2] == 4:
                img = img[:, :, :3]
            images.append(img)

    print("Images found:", len(images))

    imageio.mimsave(output_path, images, duration=0.2)  
    print(f"GIF saved at {output_path}")


if __name__ == "__main__":
    main()

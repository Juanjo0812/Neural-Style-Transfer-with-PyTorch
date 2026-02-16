import os
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from src.Utils import load_image, im_convert
from src.Model import load_vgg, get_features
from src.Losses import (
    gram_matrix,
    compute_content_loss,
    compute_style_loss,
    compute_total_loss,
)


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

   
    # Create output directory
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

   
    # Load model
    vgg = load_vgg(device)

   
    # Load images
    content = load_image("images/Car.jpg", device=device)
    style = load_image(
        "images/Fluid.jpg",
        shape=content.shape[-2:],
        device=device,
    )

  
    # Extract features
    content_features = get_features(content, vgg)
    style_features = get_features(style, vgg)

    style_grams = {
        layer: gram_matrix(style_features[layer])
        for layer in style_features
    }


    # Weights for each style layer
    style_weights = {
        "conv1_1": 1.0,
        "conv2_1": 0.75,
        "conv3_1": 0.2,
        "conv4_1": 0.2,
        "conv5_1": 0.2,
    }

    content_weight = 1
    style_weight = 1e6


    # Target image
    target = content.clone().requires_grad_(True).to(device)


    # Optimizer
    optimizer = optim.Adam([target], lr=0.003)
    steps = 10000
    save_every = 300


    # Training loop
    for ii in range(1, steps + 1):

        target_features = get_features(target, vgg)

        content_loss = compute_content_loss(
            target_features["conv4_2"],
            content_features["conv4_2"],
        )

        style_loss = compute_style_loss(
            target_features,
            style_grams,
            style_weights,
        )

        total_loss = compute_total_loss(
            content_loss,
            style_loss,
            content_weight,
            style_weight,
        )

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if ii % save_every == 0:
            print(f"Iteration {ii}/{steps}")
            print("Total loss:", total_loss.item())

            # Save intermediate image
            output_path = os.path.join(output_dir, f"iter_{ii}.png")
            plt.imsave(output_path, im_convert(target))


    # Show only the last image
    plt.figure(figsize=(8, 8))
    plt.imshow(im_convert(target))
    plt.axis("off")
    plt.title("Final Result")
    plt.show()


if __name__ == "__main__":
    main()

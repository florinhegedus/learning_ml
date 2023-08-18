import imageio
import matplotlib.pyplot as plt
import os


OUTPUT_DIR = 'out'


def save_image_at_epoch(model, X, y, epoch):
    plt.scatter(X.numpy(), y.numpy(), label='Data Points')
    plt.plot(X.numpy(), model(X).detach().numpy(), label='Fitted Line', color='red')
    plt.legend()
    plt.title(f'Epoch {epoch}')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.savefig(os.path.join(OUTPUT_DIR, f'plot_epoch_{epoch}.png')) # Save the plot as an image
    plt.close()


def create_video(epochs: int, fps: int=30):
    images = []
    for epoch in range(epochs):
        filename = f'out/plot_epoch_{epoch}.png'
        images.append(imageio.imread(filename))

    output_video_path = os.path.join(OUTPUT_DIR, 'training_video.mp4')
    imageio.mimsave(output_video_path, images, fps=fps) # Adjust fps (frames per second) as needed

import imageio
import matplotlib.pyplot as plt
import os


OUTPUT_DIR = 'out'


def save_image_at_epoch(model, X, y, epoch):
    '''
        Used for linear regression and polynomial regression models
    '''
    plt.scatter(X.numpy(), y.numpy(), label='Data Points')
    plt.plot(X.numpy(), model(X).detach().numpy(), label='Fitted Line', color='red')
    plt.legend()
    plt.title(f'Epoch {epoch}')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.savefig(os.path.join(OUTPUT_DIR, f'plot_epoch_{epoch}.png')) # Save the plot as an image
    plt.close()


def visualize_points(points, pred_labels, num_classes, epoch):
    colors = ['red', 'green', 'blue', 'yellow']
    for i in range(num_classes):
        # Separate the points based on class labels
        class_points = points[pred_labels.squeeze() == i]

        # Plot the points for class 0
        plt.scatter(class_points[:, 0], class_points[:, 1], c=colors[i], s=5, label=f'Class {i}')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.title('2D Points Visualization')
    plt.savefig(os.path.join(OUTPUT_DIR, f'plot_epoch_{epoch}.png')) # Save the plot as an image
    plt.close()


def create_video(epochs: int, fps: int=30):
    images = []
    for epoch in range(epochs):
        filename = f'out/plot_epoch_{epoch}.png'
        images.append(imageio.imread(filename))

    output_video_path = os.path.join(OUTPUT_DIR, 'training_video.mp4')
    imageio.mimsave(output_video_path, images, fps=fps) # Adjust fps (frames per second) as needed

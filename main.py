
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic, mark_boundaries
from skimage.filters import gabor
from sklearn.cluster import KMeans
import os

# Function to extract pixel-level RGB color feature
def extract_pixel_rgb_color_feature(image_path):
    img = cv2.imread(image_path)
    rgb_feature = img.reshape((-1, 3))
    rgb_normalized = rgb_feature / 255.0
    return rgb_normalized

# Function to extract pixel-level RGB color and spatial location feature
def extract_pixel_rgb_spatial_feature(image_path):
    img = cv2.imread(image_path)
    rgb_normalized = img / 255.0
    height, width, _ = img.shape
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
    x_normalized = x_coords / (width - 1)
    y_normalized = y_coords / (height - 1)
    x_flatten = x_normalized.flatten()
    y_flatten = y_normalized.flatten()
    rgb_spatial_feature = np.column_stack((rgb_normalized.reshape((-1, 3)), x_flatten, y_flatten))
    return rgb_spatial_feature

def extract_superpixel_features(image_path, num_segments=100):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb_mean = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Extract superpixels using SLIC
    segments = slic(img_rgb, n_segments=num_segments, slic_zero=True)

    mean_rgb_list = []
    rgb_hist_list = []
    gabor_features_list = []

    for segment_id in np.unique(segments):
        mask = (segments == segment_id)

        # Extract mean RGB color values
        mean_rgb = np.mean(img_rgb_mean[mask])
        img_rgb_mean[mask] = mean_rgb

        # Extract RGB color histogram
        superpixel_rgb = img_rgb[mask]
        histograms = [np.histogram(superpixel_rgb[:, i], bins=256, range=(0, 256), density=True)[0] for i in range(3)]
        superpixel_histogram = np.stack(histograms, axis=-1)  # Stack along the last axis (channels)
        rgb_hist_list.append(superpixel_histogram)

        # Convert the image to grayscale for Gabor filter
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Define Gabor filter parameters
        orientations = 8
        scales = [0.1, 0.5, 1, 1.5, 2]

        # Initialize Gabor feature vector
        gabor_features = []

        for scale in scales:
            for theta in range(orientations):
                gabor_response = np.abs(gabor(img_gray, frequency=1 / scale, theta=theta)[0])
                mean_gabor = np.mean(gabor_response[mask])
                gabor_features.append(mean_gabor)

        mean_rgb_list = img_rgb_mean
        gabor_features_list.append(gabor_features)
        mean_rgb_array = np.vstack(mean_rgb_list)

    return np.array(mean_rgb_array), np.array(rgb_hist_list), np.array(gabor_features_list), segments

def kmeans_clustering(data, k, max_iters=100, tol=1e-4):
    # Flatten the histogram data while keeping the structure
    flattened_data = data.reshape((data.shape[0], -1))

    centers = flattened_data[np.random.choice(len(flattened_data), k, replace=True)]

    for _ in range(max_iters):
        distances = np.linalg.norm(flattened_data[:, np.newaxis] - centers, axis=-1)
        labels = np.argmin(distances, axis=1)

        # Unflatten the data after clustering
        new_centers = np.array([np.mean(flattened_data[labels == i], axis=0) for i in range(k)])

        if np.linalg.norm(new_centers - centers) < tol:
            break

        centers = new_centers


    # Reshape the centers to the original structure
    final_centers = centers.reshape((k, *data.shape[1:]))

    return labels, final_centers


# Example usage
image_path = r"your image path"
img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Extract pixel-level RGB color feature
pixel_rgb_color_feature = extract_pixel_rgb_color_feature(image_path)

# Extract pixel-level RGB color and spatial location feature
pixel_rgb_spatial_feature = extract_pixel_rgb_spatial_feature(image_path)

# Extract superpixel-level features
mean_rgb_features, rgb_hist_features, gabor_features, superpixel_segments = extract_superpixel_features(image_path)

labels_hist, centers_hist = kmeans_clustering(rgb_hist_features, 3)

segments = slic(img_rgb, n_segments=100, compactness=10)
cluster_assignments = np.zeros_like(segments)

for segment_id, label in zip(np.unique(segments), labels_hist):
    cluster_assignments[segments == segment_id] = label

def map_labels_to_numbers(labels):
    unique_labels = np.unique(labels)
    label_to_number = {label: number for number, label in enumerate(unique_labels)}
    numbers = np.vectorize(label_to_number.get)(labels)
    return numbers


# Perform K-Means clustering for pixel-level RGB color feature
k_pixel_rgb_color = 5
labels_pixel_rgb_color, centers_pixel_rgb_color = kmeans_clustering(pixel_rgb_color_feature, k_pixel_rgb_color)

# Perform K-Means clustering for pixel-level RGB spatial feature
k_pixel_rgb_spatial = 5
labels_pixel_rgb_spatial, centers_pixel_rgb_spatial = kmeans_clustering(pixel_rgb_spatial_feature, k_pixel_rgb_spatial)

# Perform K-Means clustering for superpixel mean RGB features
k_superpixel = 5
labels_superpixel, centers_superpixel = kmeans_clustering(mean_rgb_features, k_superpixel)

# Perform K-Means clustering for Gabor features
k_gabor = 5
labels_superpixel_gabor, centers_superpixel_gabor = kmeans_clustering(gabor_features, 3)

numbers_superpixel = map_labels_to_numbers(labels_superpixel)


# Load the original image in RGB format
original_image = cv2.imread(image_path)
original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

# Extract superpixels using SLIC
superpixels = slic(original_image_rgb, n_segments=100)



# Display the original image
'''fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].imshow(original_image)
axes[0].set_title("Original Image")

# Display the image with superpixel cluster labels
axes[1].imshow(mark_boundaries(original_image, superpixel_segments))
axes[1].imshow(numbers_superpixel.reshape(superpixel_segments.shape), cmap='viridis', alpha=0.5)
axes[1].set_title("Superpixel Clustering")

# Show plots
plt.tight_layout()
plt.show()'''



'''fig, axes = plt.subplots(2, 3, figsize=(12, 8))

# Original Image
axes[0, 0].imshow(original_image)
axes[0, 0].set_title("Original Image")

# Superpixels
axes[0, 1].imshow(mark_boundaries(original_image, slic(original_image_rgb, n_segments=100)))
axes[0, 1].set_title("Superpixels")
axes[0, 1].imshow(labels_superpixel.reshape(original_image.shape[:-1]), cmap='viridis')
axes[0, 1].set_title("Mean of RGB colors")

# Cluster assignments based on RGB histograms
axes[0, 2].imshow(cluster_assignments.reshape(original_image.shape[:-1]), cmap='viridis')
axes[0, 2].set_title("Hist-based Clustering")

# Cluster assignments based on pixel-level RGB color feature
axes[1, 0].imshow(labels_pixel_rgb_color.reshape(original_image.shape[:-1]), cmap='viridis')
axes[1, 0].set_title("RGB-based Clustering")

# Gabor-based clustering
gabor_labels, _ = kmeans_clustering(gabor_features, 3)  # Use the appropriate number of clusters
gabor_assignments = np.zeros_like(superpixel_segments)

for segment_id, label in zip(np.unique(superpixel_segments), gabor_labels):
    gabor_assignments[superpixel_segments == segment_id] = label

axes[1, 1].imshow(gabor_assignments.reshape(superpixel_segments.shape), cmap='viridis')
axes[1, 1].contour(superpixel_segments, colors='w', linewidths=0.5, alpha=0.7)
axes[1, 1].set_title("Gabor-based Clustering")


axes[1, 2].imshow(labels_pixel_rgb_spatial.reshape(original_image.shape[:-1]))
axes[1, 2].set_title("spatial")


# Show plots
plt.tight_layout()
plt.show()'''




'''


def process_and_save_images(directory_path):
    for filename in os.listdir(directory_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(directory_path, filename)

            # Your existing code for processing and clustering
            img = cv2.imread(image_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mean_rgb_features, _, _, superpixel_segments = extract_superpixel_features(image_path)
            k_superpixel = 5
            labels_superpixel, _ = kmeans_clustering(mean_rgb_features, k_superpixel)

            # Save the resulting superpixel cluster assignments
            save_path = os.path.join(directory_path, f"{filename.split('.')[0]}_superpixel_labels.png")
            plt.imsave(save_path, labels_superpixel.reshape(original_image.shape[:-1]), cmap='viridis')

            print(f"Processed and saved {filename} at {save_path}")


# Provide the directory path containing your images
directory_path = r"your image path"
process_and_save_images(directory_path)'''

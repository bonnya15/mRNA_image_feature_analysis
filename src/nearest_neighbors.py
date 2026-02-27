import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from PIL import Image
import tifffile as tiff
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
from scipy.spatial.distance import cdist


def nearest_neighbors(img, max_distance_pixels, neighbor_radius, pixel_size_nm):
    binary_image = np.where(img != 0, 255, 0).astype(np.uint8)
    #binary_image = (img - img.min()) / (img.max() - img.min()) * 255
    #binary_im[binary_im < 80] = 0
    
    # Display the binary image
    #plt.imshow(binary_image, cmap='gray')
    #plt.axis('off')  # Hide the axis
    #plt.show()
    
    
    # Step 3: Get the coordinates of all white pixels
    white_pixel_coords = np.column_stack(np.where(binary_image == 255))
    
    # Step 4: Cluster the white pixels using DBSCAN
    # DBSCAN will group pixels that are within max_distance_pixels of each other
    db = DBSCAN(eps=int(max_distance_pixels)+1, min_samples=1, metric='euclidean')
    clusters = db.fit_predict(white_pixel_coords)
    unique, counts = np.unique(clusters, return_counts=True)
    
    
    ## Analyze the clusters
    filter_cluster_index = np.where(counts != min(counts))[0]
    filter_cluster = counts[filter_cluster_index]
    
    
    alpha = sum(filter_cluster)/len(np.where(binary_image == 255)[0])
    print("Clustering percentage {}%".format(alpha*100))
    print("Total number of flurocent pixels - {}".format(len(np.where(binary_image == 255)[0])))
    
    # Cluster centroids
    cluster_centers = []
    for cluster_id in unique:
        points = white_pixel_coords[clusters == cluster_id]
        centroid = points.mean(axis=0)
        cluster_centers.append(centroid)
    cluster_centers = np.array(cluster_centers)
    
    # Distance matrix
    distances = cdist(cluster_centers, cluster_centers)
    
    # Data collection
    neighbors_within_radius = []
    min_distances = []
    max_distances = []
    
    for i in range(len(distances)):
        non_zero = distances[i][distances[i] > 0]
        neighbors = np.sum((distances[i] < neighbor_radius) & (distances[i] > 0))
        neighbors_within_radius.append(neighbors)
        min_d = np.min(non_zero) if len(non_zero) > 0 else 0
        max_d = np.max(distances[i][distances[i]< neighbor_radius])
        min_distances.append(min_d)
        max_distances.append(max_d)
    
    # Create DataFrame
    df = pd.DataFrame({
        "cluster_idx": unique,
        "pixel_count": counts,
        "neighbors_within_radius": neighbors_within_radius,
        "min_distance": min_distances,
        "max_distance": max_distances
    })
    
    
    mean_neighbors_within_radius = np.mean(df['neighbors_within_radius'])
    mean_min_distance = np.mean(df['min_distance']) * pixel_size_nm
    mean_max_distance = np.mean(df['max_distance']) * pixel_size_nm

    #return(len(np.where(binary_image == 255)[0]), df)
    return(alpha*100, mean_neighbors_within_radius, mean_min_distance, mean_max_distance)
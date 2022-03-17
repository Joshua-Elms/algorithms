import numpy as np
from numba import jit
from time import perf_counter as pfc


##########################
# arrays containing points will only be referred to as data
#
# data: all points, shape = ((num_points), num_dims)
# cluster_labels: cluster label of each point [[0], ..., [1]], shape = (num_points, 1)
# centroids: all centroids, shape = (k, num_dims)
#
##########################


def gen_uniform_data(len, dims, low, high, rng):
    data = rng.integers(low, high, size=(len,dims),endpoint=True)
    return data


def gen_gaussian_clusters(len, dims, rng):
    cluster1 = rng.normal(loc=15, scale=5, size=(len//2, dims))
    cluster2 = rng.normal(loc=30, scale=5, size=((len//2)+1 if len % 2 else len//2, dims))
    return np.vstack((cluster1, cluster2))

def _gen_clusts(len, dims, low, high, rng):
    return rng.integers(low, high, size=(len,dims),endpoint=True)


def euclidean_distance_matrix(points1, points2, dims):
    # calculate the difference between points at each dimensions, add axis into one set of points to allow operation
    diff = points2 - points1[:, np.newaxis, :]

    # Calculate L2 norm for each group of points
    norms = sum([diff[..., i] ** 2 for i in range(dims)])
    return norms


def manhattan_distance_matrix(points1, points2, dims):
    # calculate the difference between points at each dimensions, add axis into one set of points to allow operation
    diff = points2 - points1[:, np.newaxis, :]

    # Calculate L1 norm for each group of points
    norms = np.square(sum([abs(diff[..., i]) for i in range(dims)]))
    return norms


def initialization(data, k, rng, dist):
    unselected = [i for i in range(len(data))] # indices of all non-selected elements (all when run)
    selected = []
    dims = data.shape[-1]

    # find first point, random
    rows = data.shape[0]
    first_centroid = rng.integers(0, rows, endpoint=False)

    # remove that point from list of unselected ones, add to selected
    selected.append(unselected.pop(first_centroid))

    # first iteration of k-means++ selection process, for loop proceeding this covers the rest
    distance_matrix = dist(data[unselected, :], data[selected, :], dims)
    flat_distances = distance_matrix.flatten()
    sum = np.sum(flat_distances)
    probabilities = flat_distances / sum
    new_centroid = np.random.choice(unselected, p=probabilities) 
    selected.append(unselected.pop(unselected.index(new_centroid)))

    # only to k-2 because the first time we do this process, there is no need to determine closest centroid; there's only one
    for centroid_index in range(k-2):

        # generate distance matrix to be used for rest of iteration
        new_column_of_distance_matrix = dist(data[unselected, :], data[selected[-1], :], dims)

        # remove row from previous distance matrix
        index_offset = 0
        for centroid_index in selected[:-1]:
            if centroid_index < new_centroid:
                index_offset += 1
        
        distance_matrix = np.delete(distance_matrix, new_centroid - index_offset, axis=0)

        # concat matrices together
        distance_matrix = np.concatenate((distance_matrix, new_column_of_distance_matrix), axis=1)

        # get the smallest distances from each point to a centroid
        dist_points_to_closest_centroids = np.amin(distance_matrix, axis=1)

        # final step for initialization in k-means++; actually choose new centroid with probability proportional to distance^2
        # our distance has been squared from the beginning for easier computation, so this removes a lot of compute time
        flat_distances = dist_points_to_closest_centroids.flatten()
        sum = np.sum(flat_distances)
        probabilities = flat_distances / sum
        new_centroid = np.random.choice(unselected, p=probabilities) 

        # add new centroid index to centroid list
        selected.append(unselected.pop(unselected.index(new_centroid)))

    # finally, we have a list of indices of all points that have been chosen as centroids through k-means++
    return selected


def assignment(data, centroids, k, dims, distance):
    # get new array of all cluster assignments
    distance_matrix = distance(data, centroids, dims)
    cluster_labels = np.argmin(distance_matrix, axis=1)[:, np.newaxis]
    indices_and_cluster_labels = np.concatenate((np.arange(data.shape[0])[:, np.newaxis], cluster_labels), axis=1)

    return indices_and_cluster_labels


def update(data, indices_and_cluster_labels, current_centroids, k, dims):
    # create list of k arrays, each of which is just just all the points in the i-th cluster
    grouped_by_cluster = []
    for i in range(k):
        condition = np.extract(indices_and_cluster_labels[:, 1]==i, indices_and_cluster_labels[:, 0])
        grouped_by_cluster.append(data[condition, :])

    new_centroids_list = [np.mean(arr, axis=0) for arr in grouped_by_cluster]
    new_centroids_arr = np.stack(new_centroids_list, axis=0)

    SSE = (1 / k) * np.sum((new_centroids_arr - current_centroids) ** 2)

    return new_centroids_arr, SSE
    

def main():
    num_points = 1000000
    lower = 0
    upper = 100
    dims = 2
    k = 2
    threshold = 1
    max_iterations = 25
    rng = np.random.default_rng(71)

    data = gen_uniform_data(num_points, dims, lower, upper, rng)

    data1 = gen_gaussian_clusters(num_points, 3, rng)

    centroids = data[initialization(data, k, rng, manhattan_distance_matrix), :]

    labels = assignment(data, centroids, k, dims, euclidean_distance_matrix)

    step = 0
    sse = threshold + 1 # ensures that sse below threshold prevent the while loop from ever executing
    while sse > threshold and step < max_iterations:
        centroids, sse = update(data, labels, centroids, k, dims)
        labels= assignment(data, centroids, k, dims, euclidean_distance_matrix)
        step += 1
        print(centroids)


    # print(centroids)
    print(f"Iterations: {step}")

    # print(f"Runtime of optimized: {stop - start}")



if __name__ == "__main__":
    main()
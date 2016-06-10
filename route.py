#!/usr/bin/python3

from numpy import array, dot, cos, sin, linspace, pi, size, outer, ones, sqrt
from math import radians
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

R = 6371.0
DATAFILE = "generate4.txt"


def has_vision(sat1, sat2):
    """
    Performs a line-sphere intersection check between two satellites and the earthsphere
    http://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-sphere-intersection
    :param sat1: position of first satellite
    :param sat2: position of second satellite
    :return: true if satellites have line-of-sight
    """
    O = sat1
    D = sat2 - sat1
    a = dot(D,D)
    b = 2*dot(O,D)
    c = dot(O,O) - R*R
    print(a,b,c)
    if a == 0:
        return True
    d = b*b - 4*a*c
    if d < 0: return True
    if d == 0:
        t0 = -b/(2*a)
        return t0 < 0 or t0 > 1
    else:
        t1 = (-b + sqrt(d))/(2*a)
        t2 = (-b - sqrt(d))/(2*a)
        return (t1 < 0 or t1 > 1) and (t2 < 0 or t2 > 1)


def lat_long_to_3d(lat, long, alt):
    """
    Convert latitude, longitude, and altitude to a point in 3D space
    http://se.mathworks.com/help/aeroblks/llatoecefposition.html?s_tid=gn_loc_drop
    :param lat: latitude
    :param long: longitude
    :param alt: altitude
    :return: position in 3D space
    """
    x = R * cos(lat) * cos(long) + alt * cos(lat) * cos(long)
    y = R * cos(lat) * sin(long) + alt * cos(lat) * sin(long)
    z = R * sin(lat) + alt * sin(lat)
    return array([x, y, z])


def floyd_warshal(M):
    """
    All-pairs-shortest-paths for adjacency matrix
    :param M: Adjacency matrix (modified in-place)
    :return: trace matrix for reconstructing paths
    """
    n = len(M)
    trace = [[None]*n for i in range(n)]
    for i in range(n):
        for j in range(n):
            if M[i][j] != float('inf'):
                trace[i][j] = j
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if M[i][j] + M[j][k] < M[i][k]:
                    M[i][k] = M[i][j] + M[j][k]
                    M[k][i] = M[i][j] + M[j][k]
                    trace[i][k] = trace[i][j]
                    trace[k][i] = trace[k][j]
    return trace


def reconstruct_path(trace, start, end):
    """
    Reconstructs path between two points as a list from FW trace
    :param trace: path trace matrix from floyd-warshall
    :param start: path start index
    :param end: path end index
    :return: path as a list of indices
    """
    if trace[start][end] is None:
        return None
    path = [start]
    while start != end:
        start = trace[start][end]
        path.append(start)
    path.append(end)
    return path


def visualize(vertices, distance_mat, path):
    """
    Create a 3D plot of satellites and path with pyplot
    :param vertices: list of (x,y,z) coordinate triples
    :param distance_mat: distances in hops between vertices
    :param path: list of vertices that form the path
    """
    def t(a, i): return [x[i] for x in a]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs=t(vertices[:-2], 0), ys=t(vertices[:-2], 1), zs=t(vertices[:-2], 2), c='b', alpha=1)
    ax.scatter(xs=t(vertices[-2:], 0), ys=t(vertices[-2:], 1), zs=t(vertices[-2:], 2), c='r', alpha=1)

    for i in range(N):
        for j in range(i, N):
            if distance_mat[i][j] == 1:
                ax.plot(xs=[vertices[i][0], vertices[j][0]],
                        ys=[vertices[i][1], vertices[j][1]],
                        zs=[vertices[i][2], vertices[j][2]],
                        color='w')

    path_pairs = zip(path[:-1], path[1:])
    for (i,j) in path_pairs:
        ax.plot(xs=[vertices[i][0], vertices[j][0]],
                ys=[vertices[i][1], vertices[j][1]],
                zs=[vertices[i][2], vertices[j][2]],
                color='r')

    u = linspace(0, 2 * pi, 100)
    v = linspace(0, pi, 100)

    x = R * outer(cos(u), sin(v))
    y = R * outer(sin(u), sin(v))
    z = R * outer(ones(size(u)), cos(v))
    ax.plot_surface(x, y, z, rstride=4, cstride=4, color='0.5')

    plt.show()
    plt.gcf().clear()

satellites = []
start_point, end_point = None, None

with open(DATAFILE, "r") as f:
    for line in f:
        if line.startswith("SAT"):
            lat, long, alt = [float(val) for val in line.split(",")[1:]]
            satellites.append(lat_long_to_3d(radians(lat),radians(long),alt))
        elif line.startswith("ROUTE"):
            lat1, long1, lat2, long2 = [radians(float(val)) for val in line.split(",")[1:]]
            start_point = lat_long_to_3d(lat1, long1, 10)
            end_point = lat_long_to_3d(lat2, long2, 10)

vertices = satellites + [start_point, end_point]
N = len(vertices)
start_index = N - 2
end_index = N - 1

vision_mat = [[0]*N for i in range(N)]

for i in range(N):
    for j in range(N):
        if i == j:
            vision_mat[i][j] = 0
        else:
            vision_mat[i][j] = 1 if has_vision(vertices[i], vertices[j]) else float('Inf')

trace = floyd_warshal(vision_mat)
path = reconstruct_path(trace, start_index, end_index)
print(",".join("SAT%d" % i for i in path[1:-2]))

visualize(vertices, vision_mat, path)

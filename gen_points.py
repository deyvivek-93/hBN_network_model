import numpy as np
import networkx as nx

def generate_clusters(num_random_points_per_layer, Lx_min, Lx_max, d0, L_x, L_y):
    d = [i * d0 for i in range(1, 51, 2)]

    np.random.seed(range(3, 28))
    x_lists = [np.random.uniform(Lx_min, Lx_max, num_random_points_per_layer) for _ in range(2, 27)]
    x_new_lists = [list(set(xi)) for xi in x_lists]

    y_lists = [np.full(len(xi), di) for di, xi in zip(d, x_new_lists)]
    y_new_lists = y_lists
    extra_points = [(Lx_max / 2, -0.6), (Lx_max / 2, 10.6)]
    x_new_lists.append([point[0] for point in extra_points])
    y_new_lists.append([point[1] for point in extra_points])
    cluster_dict = {
        'num_random_points_per_layer': num_random_points_per_layer,
        'Lx_min': Lx_min,
        'Lx_max': Lx_max,
        'length_x': L_x,
        'length_y': L_y,
        'd0': d0,
        'd': d,
        'x_new_lists': x_new_lists,
        'y_new_lists': y_new_lists
    }
    return cluster_dict



def generate_coordinate_dict(cluster_dict):
    coordinates_list = [
        [(x, y) for x, y in zip(xi, yi)]
        for xi, yi in zip(cluster_dict['x_new_lists'], cluster_dict['y_new_lists'])
    ]

    coord_dict = {f"layer{index}": inner_list for index, inner_list in enumerate(coordinates_list)}

    number_of_points = cluster_dict['num_random_points_per_layer'] * len(cluster_dict['d'])

    cluster_dict.update({
        'coordinates_list': coordinates_list,
        'number_of_clusters': number_of_points,
        'coord_dict': coord_dict
    })

    return cluster_dict


from typing import Dict, List, Tuple

def generate_edge_list(node_dict: Dict[int, Tuple[float, float]], dmax_tunnel: float) -> Tuple[List[Tuple[int, int]], Dict[Tuple[int, int], float]]:
    edges = []
    edge_dict = {}

    for key1, value1 in node_dict.items():
        for key2, value2 in node_dict.items():
            if key1 < key2:
                dist = ((value1[0] - value2[0]) ** 2 + (value1[1] - value2[1]) ** 2) ** 0.5
                if dist <= dmax_tunnel:
                    edges.append((key1, key2))
                    edge_dict[(key1, key2)] = dist

    return edges, edge_dict


def generate_node_and_edges(cluster_dict):
    dmax_tunnel = 0.6953  # maximum tunnelling distance
    coords = cluster_dict['coordinates_list']

    node_dict = {}
    node_counter = 0

    for sublist in coords:
        for coord in sublist:
            if coord not in node_dict:
                node_dict[node_counter] = coord
                node_counter += 1
    
    edge_list, edge_dict = generate_edge_list(node_dict, dmax_tunnel)

    xc, yc = zip(*node_dict.values())

    cluster_dict.update({
        'node_dict': node_dict,
        'edge_list': np.asarray(edge_list),
        'edge_dict': np.asarray(edge_dict),
        'xc': np.asarray(xc),
        'yc': np.asarray(yc)
    })

    return cluster_dict

def generate_adj_matrix(cluster_dict):
   
    shape = len(cluster_dict['node_dict'])
    adj_matrix_shape = (shape,shape)
    adj_matrix = np.zeros(adj_matrix_shape, dtype=np.float32)
    adj_matrix[cluster_dict['edge_list'].astype(np.int32)[:, 0], cluster_dict['edge_list'].astype(np.int32)[:, 1]] = 1.0

    # Make the matrix symmetric
    adj_matrix = adj_matrix + adj_matrix.T

    cluster_dict['adj_matrix'] = adj_matrix

    return cluster_dict

def generate_graph(cluster_dict):
        
    cluster_dict = generate_adj_matrix(cluster_dict)
    G = nx.from_numpy_array(np.matrix(cluster_dict['adj_matrix']))    

    cluster_dict['G'] = G

    return G
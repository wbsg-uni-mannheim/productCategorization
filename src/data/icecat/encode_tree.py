import logging
import pickle
from pathlib import Path

import networkx as nx

def main():
    G = nx.DiGraph()

    logger = logging.getLogger(__name__)
    logger.info("Start to build Graph")

    dataset = 'icecat'
    project_dir = Path(__file__).resolve().parents[3]
    path_to_tree = project_dir.joinpath('data', 'raw', dataset, 'tree',
                                        'tree_{}_without_encoding.pkl'.format(dataset))

    with open(path_to_tree, 'rb') as f:
        tree = pickle.load(f)
        logger.info('Loaded tree for dataset {}!'.format(dataset))

    root = [node[0] for node in tree.in_degree if node[1] == 0][0]
    normalized_root = root.replace(' ', '_')
    #Node Dict
    inserted_nodes = 0

    # Add root node
    dict_nodes = {root: inserted_nodes}
    G.add_node(dict_nodes[root], name=normalized_root)
    inserted_nodes += 1

    nodes_no_successors = [root]

    while len(nodes_no_successors) > 0:
        node = nodes_no_successors.pop(0)

        successors = tree.successors(node)
        for successor in successors:
            if successor not in dict_nodes:
                dict_nodes[successor] = inserted_nodes
                normalized_successor = successor.replace(' ', '_')
                G.add_node(dict_nodes[successor], name=normalized_successor)
                G.add_edge(dict_nodes[node], dict_nodes[successor])
                nodes_no_successors.append(successor)
                inserted_nodes += 1

    logger.info('Added {} nodes'.format(inserted_nodes))
    logger.info(G.nodes(data=True))
    logger.info(G.edges)
    logger.info(list(G.out_degree(G.nodes())))

    logger.info(len(tree.nodes))
    logger.info(len(G.nodes))

    logger.info(tree.nodes)


    # Save tree
    with open("./data/raw/{}/tree/tree_{}.pkl".format(dataset, dataset), "wb") as file:
        pickle.dump(G, file=file)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
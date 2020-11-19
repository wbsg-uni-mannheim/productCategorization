import logging
import pickle

import networkx as nx

def main():
    G = nx.DiGraph()

    logger = logging.getLogger(__name__)
    logger.info("Start to build Graph")

    #Node Dict
    root = 'Root'
    inserted_nodes = 0

    # Add root node
    dict_nodes = {root: inserted_nodes}
    G.add_node(dict_nodes[root], name=root)
    inserted_nodes += 1

    count = 0
    with open("./data/raw/wdc_ziqi/en_2020-06/EN/GS1 Combined Published_Schema as at 01062020 EN.txt") as fp:
        while True:
            count += 1
            line = fp.readline()

            if not line:
                logger.info('Read {} lines'.format(count))
                logger.info('Added {} nodes'.format(inserted_nodes))
                print(G.nodes(data=True))
                print(G.edges)
                break

            # Skip first row / header
            if count != 1:
                line_parts = line.split('\t')

                # Create labels
                lvl1 = '_'.join([line_parts[0], line_parts[1]])
                lvl2 = '_'.join([line_parts[2], line_parts[3]])
                lvl3 = '_'.join([line_parts[4], line_parts[5]])

                # Add labels to graph
                nodes = [lvl1, lvl2, lvl3]
                predecessors = [root, lvl1, lvl2]

                for node, predecessor in zip(nodes, predecessors):
                    if node not in dict_nodes:
                        dict_nodes[node] = inserted_nodes
                        G.add_node(dict_nodes[node], name=node)
                        G.add_edge(dict_nodes[predecessor], dict_nodes[node])
                        inserted_nodes += 1

    # Save tree
    with open("./data/raw/wdc_ziqi/tree/tree_wdc_ziqi.pkl", "wb") as file:
        pickle.dump(G, file=file)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
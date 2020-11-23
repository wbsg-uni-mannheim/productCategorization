import json
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
    files = ["data/raw/wdc_ziqi/train.json", "data/raw/wdc_ziqi/validation.json", "data/raw/wdc_ziqi/task2_testset_with_labels.json"]

    for file in files:
        new_nodes = []
        with open(file) as fp:
            while True:
                count += 1
                line = fp.readline()

                if not line:
                    logger.info('Read {} lines'.format(count))
                    logger.info('Added {} nodes'.format(inserted_nodes))
                    print(G.nodes(data=True))
                    break

                json_line = json.loads(line)

                # Create labels
                lvl1 = json_line['lvl1'].replace(" ","_")
                lvl2 = json_line['lvl2'].replace(" ","_")
                lvl3 = json_line['lvl3'].replace(" ","_")

                # Add labels to graph
                nodes = [lvl1, lvl2, lvl3]
                predecessors = [root, lvl1, lvl2]

                for node, predecessor in zip(nodes, predecessors):
                    if node not in dict_nodes:
                        dict_nodes[node] = inserted_nodes
                        G.add_node(dict_nodes[node], name=node)
                        G.add_edge(dict_nodes[predecessor], dict_nodes[node])
                        new_nodes.append(node)
                        inserted_nodes += 1

        print(new_nodes)
    print(G.nodes(data=True))

    # Save tree
    with open("./data/raw/wdc_ziqi/tree/tree_wdc_ziqi.pkl", "wb") as file:
        pickle.dump(G, file=file)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
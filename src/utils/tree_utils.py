
class TreeUtils():

    def __init__(self, tree):

        self.tree = tree
        self.root = [node[0] for node in tree.in_degree if node[1] == 0][0]

    def determine_path_to_root(self, nodes):

        predecessors = self.tree.predecessors(nodes[-1])
        predecessor = [k for k in predecessors][0]

        if predecessor == self.root:
            nodes.reverse()
            return nodes
        nodes.append(predecessor)
        return self.determine_path_to_root(nodes)

    def get_all_nodes_per_lvl(self, level):

        successors = self.tree.successors(self.root)
        while level > 0:
            next_lvl_succesors = []
            for successor in successors:
                next_lvl_succesors.extend(self.tree.successors(successor))
            successors = next_lvl_succesors
            level -= 1

        return successors

    def normalize_path_from_root_per_level(self, path):
        """Normalize label values per level"""
        normalized_path = []
        for i in range(len(path)):
            counter = 0
            nodes_per_lvl = self.get_all_nodes_per_lvl(i)
            for node in nodes_per_lvl:
                counter += 1
                if node == path[i]:
                    normalized_path.append(counter)
                    break

        assert (len(path) == len(normalized_path))
        return normalized_path

    def get_sorted_leaf_nodes(self):
        leaf_nodes = []
        successors = [node for node in self.tree.successors(self.root)]
        while len(successors) > 0:
            successor = successors.pop()
            new_successors = [node for node in self.tree.successors(successor)]
            if len(new_successors) > 0:
                successors.extend(new_successors)
            else:
                leaf_nodes.append(successor)

        return leaf_nodes

    def get_number_of_nodes_lvl(self):
        #3 is hard coded for now!
        num_labels_per_level = {}
        for i in range(3):
            nodes_per_lvl = [node for node in self.get_all_nodes_per_lvl(i)]
            num_labels_per_level[i+1] = len(nodes_per_lvl) + 1 # Plus 1 for ooc

        return num_labels_per_level

    def encode_node(self, name):
        decoder = dict(self.tree.nodes(data="name"))
        encoder = dict([(value, key) for key, value in decoder.items()])

        return encoder[name]
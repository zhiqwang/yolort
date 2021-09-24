class Grapher:
    def __init__(self):
        self.graph = {}

    def node(self, name, inbound_nodes=None):
        self.graph[name] = {}
        if inbound_nodes is not None:
            self.graph[name]['inbounds'] = inbound_nodes
            for node in inbound_nodes:
                if node not in self.graph.keys():
                    self.graph[node] = {}
                if 'outbounds' not in self.graph[node].keys():
                    self.graph[node]['outbounds'] = []
                self.graph[node]['outbounds'].append(name)

    def refresh(self):
        for name in self.graph.keys():
            self.graph[name]['outbounds'] = []

        for name in self.graph.keys():
            for node in self.graph[name]['inbounds']:
                if node not in self.graph.keys():
                    while node in self.graph[name]['inbounds']:
                        self.graph[name]['inbounds'].remove(node)
                else:
                    if 'outbounds' not in self.graph[node].keys():
                        self.graph[node]['outbounds'] = []

                    self.graph[node]['outbounds'].append(name)

        spare_nodes = []

        for name in self.graph.keys():
            if len(self.graph[name]['outbounds']) == 0 and len(self.graph[name]['inbounds']) == 0:
                spare_nodes.append(name)

        for removing_node_name in spare_nodes:
            del self.graph[removing_node_name]

    def get_graph(self):
        return self.graph

    def get_node_inbounds(self, name):
        if 'inbounds' in self.graph[name]:
            return self.graph[name]['inbounds']
        else:
            return []

    def get_node_outbounds(self, name):
        if 'outbounds' in self.graph[name]:
            return self.graph[name]['outbounds']
        else:
            return []

    def set_node_inbounds(self, name, inbounds):
        self.graph[name]['inbounds'] = inbounds

    def set_node_outbounds(self, name, outbounds):
        self.graph[name]['outbounds'] = outbounds

    def remove_node(self, name):
        if name in self.graph.keys():
            del self.graph[name]

    def remove_node_inbounds(self, name, inbound):
        if inbound in self.graph[name]['inbounds']:
            self.graph[name]['inbounds'].remove(inbound)

    def remove_node_outbounds(self, name, outbound):
        if outbound in self.graph[name]['outbound']:
            self.graph[name]['outbounds'].remove(outbound)

    def add_node_inbounds(self, name, inbound):
        self.graph[name]['inbounds'].append(inbound)

    def add_node_outbounds(self, name, outbound):
        self.graph[name]['outbounds'].append(outbound)

    def get_graph_head(self):
        self.heads = []
        for (key, value) in self.graph.items():
            if 'inbounds' not in value.keys() or len(value['inbounds']) == 0:
                self.heads.append(key)
        return self.heads

    def get_graph_tail(self):
        self.tails = []
        for (key, value) in self.graph.items():
            if 'outbounds' not in value.keys() or len(value['outbounds']) == 0:
                self.tails.append(key)
        return self.tails

    def set_node_attr(self, name, attr):
        if name not in self.graph.keys():
            self.graph[name] = {}
        self.graph[name]['attr'] = attr

    def get_node_attr(self, name):
        if name in self.graph.keys():
            return self.graph[name]['attr']
        else:
            return None

    def plot_graphs(self, filename='kears2ncnn'):
        from graphviz import Digraph

        dot = Digraph(comment='Network Grapher View')
        for (key, value) in self.graph.items():
            dot.node(key, key)
            if 'inbounds' in value.keys():
                for node in value['inbounds']:
                    dot.edge(node, key)
        dot.render(filename, view=False)

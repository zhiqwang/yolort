class GraphOptimization:

    @staticmethod
    def removing_unused_nodes(graph):
        UNUSED_NODES = ['Dropout', 'Lambda', 'TimeDistributed']
        nodes_to_remove = []

        for target_node_name in graph.get_graph().keys():
            if (graph.get_node_attr(target_node_name)['layer']['class_name'] in UNUSED_NODES
                or (graph.get_node_attr(target_node_name)['layer']['class_name'] == 'InputLayer'
                    and len(graph.get_node_inbounds(target_node_name)) != 0 )):

                for layer_name in graph.get_graph().keys():
                    if target_node_name in graph.get_graph()[layer_name]['inbounds']:
                        graph.remove_node_inbounds(layer_name, target_node_name)
                        graph.add_node_inbounds(
                            layer_name,
                            graph.get_graph()[target_node_name]['inbounds'][0],
                        )
                nodes_to_remove.append(target_node_name)

        for removed_nodes_name in nodes_to_remove:
            graph.remove_node(removed_nodes_name)

    @staticmethod
    def removing_reshape_after_global_pooling(graph):
        GLOBAL_POOLING_NODES = ['GlobalAveragePooling2D', 'MaxAveragePooling2D']
        nodes_to_remove = []

        for target_node_name in graph.get_graph().keys():
            if graph.get_node_attr(target_node_name)['layer']['class_name'] in GLOBAL_POOLING_NODES:
                for out_nodes in graph.get_node_outbounds(target_node_name):
                    if graph.get_node_attr(out_nodes)['layer']['class_name'] == 'Reshape':
                        for layer_name in graph.get_graph().keys():
                            if out_nodes in graph.get_graph()[layer_name]['inbounds']:
                                graph.remove_node_inbounds(layer_name, out_nodes)
                                graph.add_node_inbounds(
                                    layer_name,
                                    graph.get_graph()[out_nodes]['inbounds'][0],
                                )
                        nodes_to_remove.append(out_nodes)

        for removed_nodes_name in nodes_to_remove:
            graph.remove_node(removed_nodes_name)

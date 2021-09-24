import numpy as np


class NCNNEmitter:

    def __init__(self, ncnn_graph):
        self.MAGGGGGIC = '7767517'
        self.ncnn_graph = ncnn_graph

    def get_graph_seq(self):

        graph_head = self.ncnn_graph.get_graph_head()

        # Thanks to Blckknght for the topological sort alg
        seen = set()
        stack = []
        order = []
        q = [graph_head[0]]

        for head in graph_head:
            q = [head]
            while q:
                v = q.pop()
                if v not in seen:
                    seen.add(v)
                    q.extend(self.ncnn_graph.get_node_outbounds(v))

                    while stack and v not in self.ncnn_graph.get_node_outbounds(stack[-1]):
                        order.append(stack.pop())
                    stack.append(v)

        return stack + order[::-1]

    def emit_param(self, file_name, seq):

        param_contect = ''
        blob_count = 0

        for layer_name in seq:
            layer_type = self.ncnn_graph.get_node_attr(layer_name)['type']
            input_count = len(self.ncnn_graph.get_node_inbounds(layer_name))

            output_count = len(self.ncnn_graph.get_node_outbounds(layer_name))
            output_count = 1 if output_count == 0 else output_count

            input_blobs = []
            inbound_nodes = self.ncnn_graph.get_node_inbounds(layer_name)
            for in_node in inbound_nodes:
                if len(self.ncnn_graph.get_node_outbounds(in_node)) > 1:
                    input_blobs.append(f'{in_node}_blob_idx_'
                                       f'{self.ncnn_graph.get_node_outbounds(in_node).index(layer_name)}')
                else:
                    input_blobs.append(f'{in_node}_blob')

            output_blobs = []
            if output_count > 1:
                for i in range(output_count):
                    output_blobs.append(f'{layer_name}_blob_idx_{i}')
            else:
                output_blobs.append(f'{layer_name}_blob')

            blob_count += len(output_blobs)

            param_contect = (f"{param_contect}{layer_type}{(max(1, 25 - len(layer_type))) * ' '}"
                             f"{layer_name}{(max(1, 40 - len(layer_name))) * ' '}{input_count} "
                             f"{output_count} {' '.join(input_blobs)} {' '.join(output_blobs)} "
                             f"{self.ncnn_graph.get_node_attr(layer_name)['param']}")

        layer_count = len(self.ncnn_graph.get_graph())

        with open(file_name, 'w+') as ncnn_param_file:
            ncnn_param_file.write(f'{self.MAGGGGGIC}\n')
            ncnn_param_file.write(f'{layer_count} {blob_count}\n')
            ncnn_param_file.write(param_contect)

    def emit_binary(self, file_name, seq):
        f = open(file_name, 'w+b')
        for layer_name in seq:
            for weight in self.ncnn_graph.get_node_attr(layer_name)['binary']:
                f.write(np.asarray(weight, dtype=np.float32).tobytes())
        f.close()

class TorchScriptConverter:
    MULTI_OUTPUT_OP = []

    @staticmethod
    def replaceDefault(content, key, default=1):
        if key in content.keys():
            return content[key]
        else:
            return default

    def InputLayer_helper(self, layer, keras_graph_helper, ncnn_graph_helper, ncnn_helper):

        def replaceNone(x):
            return -1 if x is None else x

        input_w = replaceNone(layer['layer']['config']['batch_input_shape'][1])
        input_h = replaceNone(layer['layer']['config']['batch_input_shape'][2])
        input_c = replaceNone(layer['layer']['config']['batch_input_shape'][3])

        ncnn_graph_attr = ncnn_helper.dump_args('Input', w=input_w, h=input_h, c=input_c)

        ncnn_graph_helper.node(
            layer['layer']['name'],
            keras_graph_helper.get_node_inbounds(layer['layer']['name']),
        )
        ncnn_graph_helper.set_node_attr(
            layer['layer']['name'],
            {
                'type': 'Input',
                'param': ncnn_graph_attr,
                'binary': [],
            },
        )

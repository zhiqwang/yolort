import os
import argparse

from pathlib import Path

from yolort.relaying.graph import Grapher
from yolort.relaying.optimizer import GraphOptimization
from yolort.relaying.parser import H5dfParser
from yolort.relaying.converter import TorchScriptConverter
from yolort.relaying.ncnn_emitter import NCNNEmitter
from yolort.relaying.ncnn_param import NCNNParamDispatcher


def main():
    parser = argparse.ArgumentParser()

    if parser.prog == '__main__.py':
        parser.prog = 'python3 -m keras2ncnn'

    parser.add_argument('--input_file', type=str, required=True,
                        help='Input h5df file')
    parser.add_argument('--output_dir', type=str, default='',
                        help='Output file dir')
    parser.add_argument('--plot_graph', action='store_true',
                        help='Virtualize graph.')
    parser.add_argument('--debug', action='store_true',
                        help='Run accuracy debug.')
    args = parser.parse_args()

    # Create a source graph and a dest graph
    keras_graph = Grapher()
    ncnn_graph = Grapher()

    # Read and parse keras file to graph
    print('Reading and parsing keras h5df file...')
    H5dfParser(args.input_file).parse_graph(keras_graph)

    # Graph Optimization
    print('Start graph optimizing pass...')
    print('\tRemoving unused nodes...')
    GraphOptimization.removing_unused_nodes(keras_graph)
    print('\tRemoving squeeze reshape after pooling...')
    GraphOptimization.removing_reshape_after_global_pooling(keras_graph)

    print('\tRefreshing graph...')
    keras_graph.refresh()

    # Convert keras to ncnn representations
    print('Converting keras graph to ncnn graph...')
    TorchScriptConverter().parse_keras_graph(keras_graph, ncnn_graph, NCNNParamDispatcher())

    if args.plot_graph:
        print('Rendering graph plots...')
        keras_graph.plot_graphs(Path(args.input_file).stem + '_keras')
        ncnn_graph.plot_graphs(Path(args.input_file).stem + '_ncnn')

    # Emit the graph to params and bin

    if args.output_dir != '':
        print('Start emitting to ncnn files.')
        emitter = NCNNEmitter(ncnn_graph)
        graph_seq = emitter.get_graph_seq()

        print('\tEmitting param...')
        emitter.emit_param(os.path.join(args.output_dir, Path(args.input_file).stem + '.param'), graph_seq)

        print('\tEmitting binary...')
        emitter.emit_binary(os.path.join(args.output_dir, Path(args.input_file).stem + '.bin'), graph_seq)

    print('Done!')

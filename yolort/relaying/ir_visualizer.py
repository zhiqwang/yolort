# Copyright (c) 2021, Zhiqiang Wang
# Copyright (c) 2020, Thomas Viehmann
"""
Visualizing JIT Modules

Modified from https://github.com/t-vi/pytorch-tvmisc/tree/master/hacks
with license under the CC-BY-SA 4.0.

Please link to Thomas's blog post or the original github source (linked from the
blog post) with the attribution notice.
"""
from graphviz import Digraph


class TorchScriptVisualizer:
    def __init__(self, module):

        self.module = module
        self.seen_edges = set()
        self.seen_input_names = set()

        self.unseen_ops = {
            'aten::Int',
            'prim::ListConstruct', 'prim::ListUnpack',
            'prim::TupleConstruct', 'prim::TupleUnpack',
            'aten::unbind', 'aten::detach',
            'aten::contiguous', 'aten::to',
            'aten::unsqueeze', 'aten::squeeze',
            'aten::index', 'aten::slice', 'aten::select',
            'aten::constant_pad_nd',
            'aten::size', 'aten::split_with_sizes',
            'aten::expand_as', 'aten::expand',
            'aten::_shape_as_tensor',
        }
        # probably also partially absorbing ops. :/
        self.absorbing_ops = ('aten::size', 'aten::_shape_as_tensor')

    def render(self, classes_to_visit={'YOLO', 'YOLOHead'}):
        return self.make_graph(self.module, classes_to_visit=classes_to_visit)

    def make_graph(self, module, dot=None, parent_dot=None, prefix="", input_preds=None,
                   classes_to_visit=None, classes_found=None):
        graph = module.graph
        preds = {}

        self_input = next(graph.inputs())
        self_type = self_input.type().str().split('.')[-1]
        preds[self_input] = (set(), set())  # inps, ops

        if dot is None:
            dot = Digraph(format='svg', graph_attr={'label': self_type, 'labelloc': 't'})

        for nr, i in enumerate(list(graph.inputs())[1:]):
            name = f'{prefix}input_{i.debugName()}'
            preds[i] = {name}, set()
            dot.node(name, shape='ellipse')
            if input_preds is not None:
                pr, op = input_preds[nr]
                self.make_edges(pr, f'input_{name}', name, op, parent_dot)

        for node in graph.nodes():
            only_first_ops = {'aten::expand_as'}
            rel_inp_end = 1 if node.kind() in only_first_ops else None

            relevant_inputs = [i for i in list(node.inputs())[:rel_inp_end] if is_relevant_type(i.type())]
            relevant_outputs = [o for o in node.outputs() if is_relevant_type(o.type())]

            if node.kind() == 'prim::CallMethod':
                fq_submodule_name = '.'.join([
                    nc for nc in list(node.inputs())[0].type().str().split('.') if not nc.startswith('__')])
                submodule_type = list(node.inputs())[0].type().str().split('.')[-1]
                submodule_name = find_name(list(node.inputs())[0], self_input)
                name = f'{prefix}.{node.output().debugName()}'
                label = f'{prefix}{submodule_name} ({submodule_type})'

                if classes_found is not None:
                    classes_found.add(fq_submodule_name)

                if ((classes_to_visit is None and (not fq_submodule_name.startswith('torch.nn')
                    or fq_submodule_name.startswith('torch.nn.modules.container')))
                    or (classes_to_visit is not None and (submodule_type in classes_to_visit
                        or fq_submodule_name in classes_to_visit))):

                    # go into subgraph
                    sub_prefix = prefix + submodule_name + '.'
                    with dot.subgraph(name=f'cluster_{name}') as sub_dot:
                        sub_dot.attr(label=label)
                        sub_module = module
                        for k in submodule_name.split('.'):
                            sub_module = getattr(sub_module, k)

                        self.make_graph(
                            sub_module,
                            dot=sub_dot,
                            parent_dot=dot,
                            prefix=sub_prefix,
                            input_preds=[preds[i] for i in list(node.inputs())[1:]],
                            classes_to_visit=classes_to_visit,
                            classes_found=classes_found,
                        )

                    for i, o in enumerate(node.outputs()):
                        preds[o] = {sub_prefix + f'output_{i}'}, set()
                else:
                    dot.node(name, label=label, shape='box')
                    for i in relevant_inputs:
                        pr, op = preds[i]
                        self.make_edges(pr, prefix + i.debugName(), name, op, dot)
                    for o in node.outputs():
                        preds[o] = {name}, set()

            elif node.kind() == 'prim::CallFunction':
                funcname = list(node.inputs())[0].type().__repr__().split('.')[-1]
                name = prefix + '.' + node.output().debugName()
                label = funcname
                dot.node(name, label=label, shape='box')
                for i in relevant_inputs:
                    pr, op = preds[i]
                    self.make_edges(pr, prefix + i.debugName(), name, op, dot)
                for o in node.outputs():
                    preds[o] = {name}, set()

            else:
                label = node.kind().split('::')[-1].rstrip('_')
                pr, op = set(), set()
                for i in relevant_inputs:
                    apr, aop = preds[i]
                    pr |= apr
                    op |= aop

                if node.kind() in self.absorbing_ops:
                    pr, op = set(), set()
                elif (len(relevant_inputs) > 0
                      and len(relevant_outputs) > 0
                      and node.kind() not in self.unseen_ops):
                    op.add(label)
                for o in node.outputs():
                    preds[o] = pr, op

        for i, o in enumerate(graph.outputs()):
            name = prefix + f'output_{i}'
            dot.node(name, shape='ellipse')
            pr, op = preds[o]
            self.make_edges(pr, f'input_{name}', name, op, dot)

        return dot

    def add_edge(self, dot, n1, n2):
        if (n1, n2) not in self.seen_edges:
            self.seen_edges.add((n1, n2))
            dot.edge(n1, n2)

    def make_edges(self, pr, input_name, name, op, edge_dot):
        if op:
            if input_name not in self.seen_input_names:
                self.seen_input_names.add(input_name)
                label_lines = [[]]
                line_len = 0
                for w in op:
                    if line_len >= 20:
                        label_lines.append([])
                        line_len = 0
                    label_lines[-1].append(w)
                    line_len += len(w) + 1

                edge_dot.node(
                    input_name,
                    label='\n'.join([' '.join(w) for w in label_lines]),
                    shape='box',
                    style='rounded',
                )
                for p in pr:
                    self.add_edge(edge_dot, p, input_name)
            self.add_edge(edge_dot, input_name, name)
        else:
            for p in pr:
                self.add_edge(edge_dot, p, name)


def find_name(layer_input, self_input, suffix=None):
    if layer_input == self_input:
        return suffix
    cur = layer_input.node().s('name')
    if suffix is not None:
        cur = f'{cur}.{suffix}'
    of = next(layer_input.node().inputs())
    return find_name(of, self_input, suffix=cur)


def is_relevant_type(t):
    kind = t.kind()
    if kind == 'TensorType':
        return True
    if kind in ('ListType', 'OptionalType'):
        return is_relevant_type(t.getElementType())
    if kind == 'TupleType':
        return any([is_relevant_type(tt) for tt in t.elements()])
    return False
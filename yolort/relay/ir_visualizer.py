# Copyright (c) 2021, Zhiqiang Wang
# Copyright (c) 2020, Thomas Viehmann
"""
Visualizing JIT Modules

Modified from https://github.com/t-vi/pytorch-tvmisc/tree/master/hacks
with license under the CC-BY-SA 4.0.

Please link to Thomas's blog post or the original github source (linked from the
blog post) with the attribution notice.
"""
from collections import OrderedDict

try:
    from graphviz import Digraph
except ImportError:
    Digraph = None


class TorchScriptVisualizer:
    def __init__(self, module):

        self.module = module

        self.unseen_ops = {
            "prim::ListConstruct",
            "prim::ListUnpack",
            "prim::TupleConstruct",
            "prim::TupleUnpack",
            "aten::Int",
            "aten::unbind",
            "aten::detach",
            "aten::contiguous",
            "aten::to",
            "aten::unsqueeze",
            "aten::squeeze",
            "aten::index",
            "aten::slice",
            "aten::select",
            "aten::constant_pad_nd",
            "aten::size",
            "aten::split_with_sizes",
            "aten::expand_as",
            "aten::expand",
            "aten::_shape_as_tensor",
        }
        # probably also partially absorbing ops. :/
        self.absorbing_ops = ("aten::size", "aten::_shape_as_tensor")

    def render(
        self,
        classes_to_visit={"YOLO", "YOLOHead"},
        format="svg",
        labelloc="t",
        attr_size="8,7",
    ):
        self.clean_status()

        model_input = next(self.module.graph.inputs())
        model_type = self.get_node_names(model_input)[-1]
        if Digraph is not None:
            dot = Digraph(
                format=format,
                graph_attr={"label": model_type, "labelloc": labelloc},
            )
        else:
            dot = None
            raise ImportError("Graphviz is not installed, please install graphviz firstly.")

        self.make_graph(self.module, dot=dot, classes_to_visit=classes_to_visit)

        dot.attr(size=attr_size)
        return dot

    def clean_status(self):
        self._seen_edges = set()
        self._seen_input_names = set()
        self._predictions = OrderedDict()

    @staticmethod
    def get_node_names(node):
        return node.type().str().split(".")

    @staticmethod
    def get_function_name(node):
        return node.type().__repr__().split(".")[-1]

    def make_graph(
        self,
        module,
        dot=None,
        parent_dot=None,
        prefix="",
        input_preds=None,
        classes_to_visit=None,
        classes_found=None,
    ):
        graph = module.graph

        self_input = next(graph.inputs())
        self._predictions[self_input] = (
            set(),
            set(),
        )  # Stand for `input` and `op` respectively

        for nr, i in enumerate(list(graph.inputs())[1:]):
            name = f"{prefix}input_{i.debugName()}"
            self._predictions[i] = {name}, set()
            dot.node(name, shape="ellipse")
            if input_preds is not None:
                pred, op = input_preds[nr]
                self.make_edges(pred, f"input_{name}", name, op, parent_dot)

        for node in graph.nodes():
            node_inputs = list(node.inputs())
            only_first_ops = {"aten::expand_as"}
            rel_inp_end = 1 if node.kind() in only_first_ops else None

            relevant_inputs = [i for i in node_inputs[:rel_inp_end] if is_relevant_type(i.type())]
            relevant_outputs = [o for o in node.outputs() if is_relevant_type(o.type())]

            if node.kind() == "prim::CallMethod":
                node_names = self.get_node_names(node_inputs[0])
                fq_submodule_name = ".".join([nc for nc in node_names if not nc.startswith("__")])
                submodule_type = node_names[-1]
                submodule_name = find_name(node_inputs[0], self_input)
                name = f"{prefix}.{node.output().debugName()}"
                label = f"{prefix}{submodule_name} ({submodule_type})"

                if classes_found is not None:
                    classes_found.add(fq_submodule_name)

                if (
                    classes_to_visit is None
                    and (
                        not fq_submodule_name.startswith("torch.nn")
                        or fq_submodule_name.startswith("torch.nn.modules.container")
                    )
                ) or (
                    classes_to_visit is not None
                    and (submodule_type in classes_to_visit or fq_submodule_name in classes_to_visit)
                ):

                    # go into subgraph
                    sub_prefix = f"{prefix}{submodule_name}."

                    for i, o in enumerate(node.outputs()):
                        self._predictions[o] = {f"{sub_prefix}output_{i}"}, set()

                    with dot.subgraph(name=f"cluster_{name}") as sub_dot:
                        sub_dot.attr(label=label)
                        sub_module = module
                        for k in submodule_name.split("."):
                            sub_module = getattr(sub_module, k)

                        self.make_graph(
                            sub_module,
                            dot=sub_dot,
                            parent_dot=dot,
                            prefix=sub_prefix,
                            input_preds=[self._predictions[i] for i in node_inputs[1:]],
                            classes_to_visit=classes_to_visit,
                            classes_found=classes_found,
                        )

                else:
                    dot.node(name, label=label, shape="box")
                    for i in relevant_inputs:
                        pred, op = self._predictions[i]
                        self.make_edges(pred, prefix + i.debugName(), name, op, dot)
                    for o in node.outputs():
                        self._predictions[o] = {name}, set()

            elif node.kind() == "prim::CallFunction":
                name = f"{prefix}.{node.output().debugName()}"
                fun_name = self.get_function_name(node_inputs[0])
                dot.node(name, label=fun_name, shape="box")
                for i in relevant_inputs:
                    pred, op = self._predictions[i]
                    self.make_edges(pred, prefix + i.debugName(), name, op, dot)
                for o in node.outputs():
                    self._predictions[o] = {name}, set()

            else:
                label = node.kind().split("::")[-1].rstrip("_")
                pred, op = set(), set()
                for i in relevant_inputs:
                    apred, aop = self._predictions[i]
                    pred |= apred
                    op |= aop

                if node.kind() in self.absorbing_ops:
                    pred, op = set(), set()
                elif (
                    len(relevant_inputs) > 0
                    and len(relevant_outputs) > 0
                    and node.kind() not in self.unseen_ops
                ):
                    op.add(label)
                for o in node.outputs():
                    self._predictions[o] = pred, op

        for i, o in enumerate(graph.outputs()):
            name = f"{prefix}output_{i}"
            dot.node(name, shape="ellipse")
            pred, op = self._predictions[o]
            self.make_edges(pred, f"input_{name}", name, op, dot)

    def add_edge(self, dot, n1, n2):
        if (n1, n2) not in self._seen_edges:
            self._seen_edges.add((n1, n2))
            dot.edge(n1, n2)

    def make_edges(self, preds, input_name, name, op, edge_dot):
        if len(op) > 0:
            if input_name not in self._seen_input_names:
                self._seen_input_names.add(input_name)
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
                    label="\n".join([" ".join(w) for w in label_lines]),
                    shape="box",
                    style="rounded",
                )
                for p in preds:
                    self.add_edge(edge_dot, p, input_name)
            self.add_edge(edge_dot, input_name, name)
        else:
            for p in preds:
                self.add_edge(edge_dot, p, name)


def find_name(layer_input, self_input, suffix=None):
    if layer_input == self_input:
        return suffix
    cur = layer_input.node().s("name")
    if suffix is not None:
        cur = f"{cur}.{suffix}"
    of = next(layer_input.node().inputs())
    return find_name(of, self_input, suffix=cur)


def is_relevant_type(t):
    kind = t.kind()
    if kind == "TensorType":
        return True
    if kind in ("ListType", "OptionalType"):
        return is_relevant_type(t.getElementType())
    if kind == "TupleType":
        return any([is_relevant_type(tt) for tt in t.elements()])
    return False

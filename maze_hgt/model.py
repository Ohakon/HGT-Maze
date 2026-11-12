"""Batched typed attention, node-wise GRU, attention pooling and task heads."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from .game import ACTION_KINDS, Action
from .graph import (EVENT_ID, FEATURES, NODE_ID, NODE_TYPES, REL_ID, RELATION_TASKS, RELATIONS,
                    Event, Graph, Query, TransitionSpec, action_nodes)

ACTION_ID = {name: i for i, name in enumerate(ACTION_KINDS)}
TASK_ID = {name: i for i, name in enumerate(RELATION_TASKS)}


@dataclass(frozen=True)
class Architecture:
    dim: int = 48
    heads: int = 4
    layers: int = 4
    dropout: float = 0.0
    gru: bool = True
    events: bool = True

    def __post_init__(self) -> None:
        if min(self.dim, self.heads, self.layers) < 1 or self.dim % self.heads:
            raise ValueError("The embedding size must be divisible by the number of heads")
        if not 0 <= self.dropout < 1:
            raise ValueError("Dropout must be in [0, 1)")


def segment_softmax(scores: Tensor, group: Tensor, count: int) -> Tensor:
    flat = scores.reshape(scores.shape[0], -1)
    ids = group[:, None].expand_as(flat)
    peak = flat.new_full((count, flat.shape[1]), -torch.inf)
    peak.scatter_reduce_(0, ids, flat.detach(), reduce="amax", include_self=True)
    exp = (flat - peak[group]).exp()
    denom = exp.new_zeros(peak.shape).index_add_(0, group, exp)
    return (exp / denom[group]).reshape_as(scores)


class HGTLayer(nn.Module):
    def __init__(self, cfg: Architecture):
        super().__init__()
        dim, self.heads = cfg.dim, cfg.heads
        self.width = dim // self.heads
        n_types, n_rel = len(NODE_TYPES), len(RELATIONS)
        self.qkv = nn.Parameter(torch.empty(n_types, dim, dim * 3))
        self.qkv_bias = nn.Parameter(torch.zeros(n_types, dim * 3))
        self.output = nn.Parameter(torch.empty(n_types, dim, dim))
        self.attention_rel = nn.Parameter(torch.empty(n_rel, self.heads, self.width, self.width))
        self.message_rel = nn.Parameter(torch.empty_like(self.attention_rel))
        self.prior = nn.Parameter(torch.ones(n_types, n_rel, n_types, self.heads))
        self.attribute_bias = nn.Parameter(torch.zeros(n_rel, self.heads))
        self.attribute_message = nn.Parameter(torch.zeros(n_rel, self.heads, self.width))
        self.skip = nn.Parameter(torch.zeros(n_types))
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(cfg.dropout)
        for table in (self.qkv, self.output, self.attention_rel.flatten(0, 1), self.message_rel.flatten(0, 1)):
            for matrix in table:
                nn.init.xavier_uniform_(matrix)

    def forward(self, x: Tensor, types: Tensor, src: Tensor, dst: Tensor,
                rel: Tensor, value: Tensor) -> Tensor:
        qkv = torch.bmm(x.unsqueeze(1), self.qkv[types]).squeeze(1) + self.qkv_bias[types]
        q, k, v = (part.reshape(-1, self.heads, self.width) for part in qkv.chunk(3, dim=-1))
        merged = x.new_zeros((len(x), self.heads, self.width))
        if src.numel():
            k_rel = torch.einsum("ehd,ehdf->ehf", k[src], self.attention_rel[rel])
            v_rel = torch.einsum("ehd,ehdf->ehf", v[src], self.message_rel[rel])
            scores = (q[dst] * k_rel).sum(-1) * self.prior[types[src], rel, types[dst]] / self.width ** .5
            scores = scores + (value[:, None] - 1) * self.attribute_bias[rel]
            weights = segment_softmax(scores, dst, len(x))
            messages = (v_rel + (value[:, None, None] - 1) * self.attribute_message[rel]) * weights.unsqueeze(-1)
            merged.index_add_(0, dst, messages)
        update = torch.bmm(F.gelu(merged.flatten(1)).unsqueeze(1), self.output[types]).squeeze(1)
        mix = self.skip[types].sigmoid().unsqueeze(-1)
        return self.norm(mix * self.dropout(update) + (1. - mix) * x)


class HGT(nn.Module):
    def __init__(self, cfg: Architecture):
        super().__init__()
        self.features = nn.Linear(FEATURES, cfg.dim)
        self.types = nn.Embedding(len(NODE_TYPES), cfg.dim)
        self.layers = nn.ModuleList(HGTLayer(cfg) for _ in range(cfg.layers))

    def forward(self, x: Tensor, types: Tensor, src: Tensor, dst: Tensor, rel: Tensor, value: Tensor) -> Tensor:
        x = self.features(x) + self.types(types)
        for layer in self.layers:
            x = layer(x, types, src, dst, rel, value)
        return x


class EventEncoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.kind = nn.Embedding(len(EVENT_ID), dim)
        self.relation = nn.Embedding(len(RELATIONS) + 1, dim)
        self.node = nn.Embedding(len(NODE_TYPES) + 1, dim)
        self.role = nn.Embedding(3, dim)
        self.attributes = nn.Linear(FEATURES * 2, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, batch: list[Graph], events: list[tuple[Event, ...]], offsets: list[int],
                like: Tensor) -> Tensor:
        descriptors, attributes = [], []
        messages, recipients, roles = [], [], []
        for graph, changes, offset in zip(batch, events, offsets):
            for event in changes:
                available = [(graph.index[node] + offset, role) for node, role in event.receivers()
                             if node in graph.index]
                if not available:
                    continue
                event_index = len(descriptors)
                descriptors.append((EVENT_ID[event.kind], REL_ID[event.relation] + 1 if event.relation else 0,
                                    NODE_ID[event.subject_type] + 1, NODE_ID[event.other_type] + 1 if event.other_type else 0))
                attributes.append((*event.before, *event.after))
                for recipient, role in available:
                    messages.append(event_index)
                    recipients.append(recipient)
                    roles.append(role)
        result = torch.zeros_like(like)
        if descriptors:
            keys = torch.tensor(descriptors, dtype=torch.long, device=like.device)
            raw = self.kind(keys[:, 0]) + self.relation(keys[:, 1]) + self.node(keys[:, 2])
            raw = raw + self.node(keys[:, 3]) + self.attributes(like.new_tensor(attributes))
            recipient = torch.tensor(recipients, dtype=torch.long, device=like.device)
            message = self.norm(raw[messages] + self.role(torch.tensor(roles, device=like.device)))
            result.index_add_(0, recipient, message)
            count = like.new_zeros(len(like)).index_add_(0, recipient, like.new_ones(len(recipient)))
            result = result / count.clamp_min(1.).sqrt().unsqueeze(-1)
        return result


class Pooling(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, 1))
        self.project = nn.Sequential(nn.Linear(2 * dim + len(NODE_TYPES), dim), nn.LayerNorm(dim), nn.GELU())

    def forward(self, z: Tensor, types: Tensor, groups: Tensor, n_graphs: int) -> Tensor:
        attention = segment_softmax(self.score(z), groups, n_graphs)
        total = z.new_zeros((n_graphs, z.shape[-1])).index_add_(0, groups, z)
        attended = torch.zeros_like(total).index_add_(0, groups, z * attention)
        indices = groups * len(NODE_TYPES) + types
        counts = z.new_zeros(n_graphs * len(NODE_TYPES)).index_add_(0, indices, z.new_ones(len(z)))
        counts = counts.reshape(n_graphs, len(NODE_TYPES))
        summary = torch.cat((attended, total / counts.sum(1, keepdim=True).clamp_min(1).sqrt(), counts.log1p()), dim=1)
        return self.project(summary)


def mlp(input_dim: int, dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, dim), nn.GELU(), nn.Linear(dim, output_dim))


@dataclass
class Memory:
    scope: int
    nodes: Mapping[str, Tensor]

    def detached(self) -> Memory:
        return Memory(self.scope, {key: value.detach() for key, value in self.nodes.items()})


@dataclass
class Frame:
    graph: Graph
    hgt: Tensor
    nodes: Tensor
    board: Tensor
    memory: Memory


@dataclass
class TransitionPrediction:
    alive: Tensor
    touched: Tensor
    attributes: Tensor
    birth: Tensor
    birth_attributes: Tensor
    birth_counts: Tensor
    edges: Tensor
    edge_values: Tensor


def mask_logits(logits: Tensor, valid: Tensor) -> Tensor:
    if logits.shape != valid.shape or valid.dtype != torch.bool or not valid.any(dim=-1).all().item():
        raise ValueError("Each policy row must have at least one legal action")
    return logits.masked_fill(~valid, -torch.inf)


class Network(nn.Module):
    def __init__(self, cfg: Architecture):
        super().__init__()
        self.cfg = cfg
        d = cfg.dim
        self.hgt = HGT(cfg)
        if cfg.gru:
            if cfg.events:
                self.events = EventEncoder(d)
            self.gru = nn.GRUCell(d * (2 if cfg.events else 1), d)
            self.memory_norm = nn.LayerNorm(d)
        self.pool = Pooling(d)
        self.value = mlp(d, d, 1)
        self.actions = nn.Embedding(len(ACTION_KINDS), d)
        self.action_context = mlp(5 * d + 1, d, d)
        self.actor = mlp(d, d, 1)
        self.query_type = nn.Embedding(len(RELATION_TASKS), d)
        self.relation = mlp(4 * d, d, 1)
        self.pool_aux = mlp(d, d, 7)
        self.transition_node = mlp(3 * d, d, 2 + FEATURES)
        self.transition_birth = mlp(4 * d, d, 1 + FEATURES)
        self.birth_types = mlp(2 * d, d, len(NODE_TYPES))
        self.edge_rel = nn.Embedding(len(RELATIONS), d)
        self.transition_edge = mlp(5 * d, d, 5)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def encode(self, graphs: list[Graph], changes: list[tuple[Event, ...]] | None = None,
               memories: list[Memory | None] | None = None, *, hgt_only: bool = False) -> list[Frame]:
        offsets = []
        features, types, groups, sources, destinations, relations, values = [], [], [], [], [], [], []
        for i, graph in enumerate(graphs):
            offset = len(features)
            offsets.append(offset)
            features.extend(n.x for n in graph.nodes)
            types.extend(NODE_ID[n.kind] for n in graph.nodes)
            groups.extend([i] * len(graph.nodes))
            for edge in graph.edges:
                sources.append(offset + graph.index[edge.src])
                destinations.append(offset + graph.index[edge.dst])
                relations.append(REL_ID[edge.rel])
                values.append(edge.value)
        device = self.device
        x = torch.tensor(features, dtype=torch.float32, device=device)
        types, groups, src, dst, rel = (torch.tensor(v, dtype=torch.long, device=device)
                                        for v in (types, groups, sources, destinations, relations))
        hgt = self.hgt(x, types, src, dst, rel, x.new_tensor(values))
        z = hgt
        saved = [Memory(graph.scope, {}) for graph in graphs]
        if self.cfg.gru and not hgt_only:
            memories = memories if memories is not None else [None] * len(graphs)
            changes = changes if changes is not None else [()] * len(graphs)
            zero = hgt.new_zeros(self.cfg.dim)
            prior = []
            for graph, memory, events in zip(graphs, memories, changes):
                born = {e.subject for e in events if e.kind == "node_add"}
                previous = memory.nodes if memory is not None and memory.scope == graph.scope else {}
                prior.extend(previous.get(node.id, zero) if node.id not in born else zero for node in graph.nodes)
            incoming = hgt
            if self.cfg.events:
                incoming = torch.cat((hgt, self.events(graphs, changes, offsets, hgt)), dim=-1)
            memory_vectors = self.gru(incoming, torch.stack(prior))
            z = self.memory_norm(hgt + memory_vectors)
            saved = [Memory(g.scope, dict(zip(g.index, memory_vectors[i:i + len(g.nodes)].unbind())))
                     for i, g in zip(offsets, graphs)]
        boards = hgt.new_zeros((len(graphs), self.cfg.dim)) if hgt_only else self.pool(z, types, groups, len(graphs))
        return [Frame(g, hgt[i:i + len(g.nodes)], z[i:i + len(g.nodes)], board, memory)
                for g, i, board, memory in zip(graphs, offsets, boards.unbind(), saved)]

    def action_vectors(self, frame: Frame, actions: tuple[Action, ...]) -> Tensor:
        target_ids, item_ids = zip(*(action_nodes(a) for a in actions))
        table, z = frame.graph.index, frame.nodes
        target = z[[table[i] for i in target_ids]]
        kind = self.actions(torch.tensor([ACTION_ID[a.kind] for a in actions], device=z.device))
        max_items = max(1, max(map(len, item_ids)))
        index = torch.tensor([[table[i] for i in items] + [len(z)] * (max_items - len(items))
                              for items in item_ids], device=z.device)
        augmented = torch.cat((z, z.new_zeros((1, z.shape[-1]))))
        selected = augmented[index]
        count = (index != len(z)).sum(1, keepdim=True)
        mean = selected.sum(1) / count.clamp_min(1)
        # The second moment separates some bundles with similar means, and is order invariant.
        energy = selected.square().sum(1) / count.clamp_min(1)
        board = frame.board.expand(len(actions), -1)
        return self.action_context(torch.cat((target, kind, mean, energy, board, count / 4.), dim=-1))

    def policy(self, frames: list[Frame], actions: list[tuple[Action, ...]]) -> tuple[Tensor, Tensor]:
        scores = [self.actor(self.action_vectors(f, a)).squeeze(-1) for f, a in zip(frames, actions)]
        logits = pad_sequence(scores, batch_first=True, padding_value=0.)
        lengths = torch.tensor([len(s) for s in scores], device=logits.device)
        valid = torch.arange(logits.shape[1], device=logits.device)[None, :] < lengths[:, None]
        return mask_logits(logits, valid), valid

    def values(self, frames: list[Frame]) -> Tensor:
        return self.value(torch.stack([frame.board for frame in frames])).squeeze(-1)

    def relation_scores(self, frame: Frame, queries: list[Query] | tuple[Query, ...],
                        *, recurrent: bool = False) -> Tensor:
        z = frame.nodes if recurrent else frame.hgt
        lookup = frame.graph.index
        index = [[lookup[i] for i in q.nodes] + [len(z)] * (3 - len(q.nodes)) for q in queries]
        padded = torch.cat((z, z.new_zeros((1, z.shape[-1]))))
        nodes = padded[torch.tensor(index, device=z.device)].flatten(1)
        kind = self.query_type(torch.tensor([TASK_ID[q.task] for q in queries], device=z.device))
        return self.relation(torch.cat((nodes, kind), dim=-1)).squeeze(-1)

    def predict_transition(self, frame: Frame, action: Action, spec: TransitionSpec) -> TransitionPrediction:
        z, board = frame.nodes, frame.board
        choice = self.action_vectors(frame, (action,))[0]
        context = torch.cat((board, choice))
        node = self.transition_node(torch.cat((z, context.expand(len(z), -1)), dim=-1))
        lookup = dict(frame.graph.index)
        if spec.birth_slots:
            born_base = torch.stack([(z[lookup[enemy]] + z[lookup[kind]]) * .5
                                     for _, enemy, kind in spec.birth_slots])
            born_in = torch.stack([torch.cat((z[lookup[enemy]], z[lookup[kind]], context))
                                   for _, enemy, kind in spec.birth_slots])
            birth = self.transition_birth(born_in)
            lookup.update((key, len(z) + i) for i, (key, _, _) in enumerate(spec.birth_slots))
            z = torch.cat((z, born_base))
        else:
            birth = z.new_empty((0, FEATURES + 1))
        ids = torch.tensor([(lookup[a], REL_ID[r], lookup[b]) for a, r, b in spec.edges], device=z.device)
        edge = self.transition_edge(torch.cat((z[ids[:, 0]], z[ids[:, 2]], self.edge_rel(ids[:, 1]),
                                               context.expand(len(ids), -1)), dim=-1))
        return TransitionPrediction(node[:, 0], node[:, 1], node[:, 2:], birth[:, 0], birth[:, 1:],
                                    self.birth_types(context), edge[:, :4], edge[:, 4])

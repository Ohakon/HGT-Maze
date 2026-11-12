"""Five validation-gated pretraining stages."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from random import Random
from time import monotonic

import torch
from torch import Tensor
from torch.nn import functional as F

from .config import Config
from .data import Step, relation_data, sequences
from .game import initial_state, outcome, transition
from .generate import derived_seed, generate
from .graph import NODE_ID, NODE_TYPES, RELATION_TASKS, diff
from .io import append_json, checkpoint, read_checkpoint, restore_rng, write_json
from .metrics import Metrics, weight_count
from .model import Frame, Network
from .tasks import POOL_NAMES, observation


STAGE_MODULES = {
    1: ("hgt", "query_type", "relation"),
    2: ("hgt", "events", "gru", "memory_norm", "pool", "actions", "action_context", "query_type", "relation",
        "transition_node", "transition_birth", "birth_types", "edge_rel", "transition_edge"),
    3: ("hgt", "events", "gru", "memory_norm", "pool", "pool_aux"),
    4: ("hgt", "events", "gru", "memory_norm", "pool", "value"),
    5: ("hgt", "events", "gru", "memory_norm", "pool", "actions", "action_context", "actor", "value"),
}


def relation_loss(model: Network, frame: Frame, queries, book: Metrics, recurrent: bool = False) -> Tensor:
    logits = model.relation_scores(frame, queries, recurrent=recurrent)
    targets = logits.new_tensor([q.label for q in queries])
    sample_losses = F.binary_cross_entropy_with_logits(logits.detach(), targets, reduction="none")
    predictions = logits >= 0
    book.binary("relations", predictions, targets)
    groups = {}
    for i, query in enumerate(queries):
        keys = [query.task]
        if query.hops:
            keys.append(f"hop.{query.hops}")
        if query.gate != "none":
            keys.append(f"gate.{query.gate}")
        for key in keys:
            groups.setdefault(key, []).append(i)
    for key, idx in groups.items():
        book.binary(key, predictions[idx], targets[idx])
        book.mean(f"{key}.loss", sample_losses[idx].mean().item(), len(idx))
    return F.binary_cross_entropy_with_logits(logits, targets)


def balanced_ce(logits: Tensor, labels: Tensor) -> Tensor:
    counts = torch.bincount(labels, minlength=logits.shape[-1]).clamp_min(1)
    return F.cross_entropy(logits, labels, weight=counts.to(logits).reciprocal())


def balanced_bce(logits: Tensor, labels: Tensor) -> Tensor:
    if not logits.numel():
        return logits.new_zeros(())
    return balanced_ce(torch.stack((torch.zeros_like(logits), logits), dim=-1), labels.long())


def transition_loss(model: Network, frame: Frame, step: Step, book: Metrics) -> Tensor:
    spec, after = step.spec, step.after
    prediction = model.predict_transition(frame, step.action, spec)
    device = frame.nodes.device
    alive = torch.tensor([key in after.table for key in spec.nodes], device=device)
    old = frame.nodes.new_tensor([frame.graph.table[key].x for key in spec.nodes])
    new = frame.nodes.new_tensor([after.table[key].x if key in after.table else frame.graph.table[key].x
                                  for key in spec.nodes])
    delta = new - old
    components = {"remove": balanced_bce(-prediction.alive, ~alive)}
    book.binary("remove", prediction.alive < 0, ~alive)
    book.binary("alive", prediction.alive >= 0, alive)
    # These events are targets. The recurrent encoder has already received only the current diff.
    touched = {receiver for event in diff(frame.graph, after) for receiver, _ in event.receivers()}
    affected = torch.tensor([key in touched for key in spec.nodes], device=device)
    components["affected"] = balanced_bce(prediction.touched, affected)
    book.binary("affected", prediction.touched >= 0, affected)
    changed = (delta != 0) & alive[:, None]
    if alive.any().item():
        losses = F.smooth_l1_loss(prediction.attributes, delta, reduction="none")
        weights = alive[:, None] * (1 + 5 * changed)
        components["attributes"] = 4 * (losses * weights).sum() / weights.expand_as(losses).sum()
        book.error("attributes_all", prediction.attributes[alive], delta[alive])
    book.error("attributes", prediction.attributes[changed], delta[changed])
    old_edges, new_edges = frame.graph.relations, after.relations
    edge_labels = []
    values = []
    for key in spec.edges:
        a, b = old_edges.get(key), new_edges.get(key)
        edge_labels.append(1 if a is None and b is not None else 2 if a is not None and b is None
                           else 3 if a != b else 0)
        values.append((b.value if b is not None else 0.) - (a.value if a is not None else 0.))
    classes = torch.tensor(edge_labels, device=device)
    components["edge"] = balanced_ce(prediction.edges, classes)
    edge_guess = prediction.edges.argmax(-1)
    book.mean("edge.accuracy", (edge_guess == classes).float().mean().item(), len(classes))
    for index, name in enumerate(("stable", "add", "remove", "change")):
        book.binary(f"edge.{name}", edge_guess == index, classes == index)
    changed_value = (classes == 1) | (classes == 3)
    if changed_value.any().item():
        expected = prediction.edge_values.new_tensor(values)[changed_value]
        predicted = prediction.edge_values[changed_value]
        components["edge_value"] = F.smooth_l1_loss(predicted, expected)
        book.error("edge_value", predicted, expected)

    counts = frame.nodes.new_zeros(len(NODE_TYPES))
    for key in after.index.keys() - frame.graph.index.keys():
        counts[NODE_ID[after.table[key].kind]] += 1.
    count_output = prediction.birth_counts
    components["birth_count"] = F.smooth_l1_loss(count_output, counts.log1p())
    book.error("birth_count", count_output.detach().clamp(0, 10).expm1(), counts)
    if spec.birth_slots:
        births = torch.tensor([key in after.table for key, _, _ in spec.birth_slots], device=device)
        components["birth"] = balanced_bce(prediction.birth, births)
        book.binary("birth", prediction.birth >= 0, births)
        if births.any().item():
            truth = frame.nodes.new_tensor([after.table[key].x for key, _, _ in spec.birth_slots if key in after.table])
            predicted = prediction.birth_attributes[births]
            components["birth_attributes"] = F.smooth_l1_loss(predicted, truth)
            book.error("birth_attributes", predicted, truth)
    age = "initial" if step.state.turn == 0 else "1-3" if step.state.turn <= 3 else "4+"
    book.binary(f"age.{age}.affected", prediction.touched >= 0, affected)
    if step.queries:
        components["relations"] = .15 * relation_loss(model, frame, step.queries, book, recurrent=True)
        query_scores = model.relation_scores(frame, step.queries, recurrent=True).detach()
        query_truth = query_scores.new_tensor([q.label for q in step.queries])
        book.binary(f"age.{age}.relations", query_scores >= 0, query_truth)
        changed_queries = [i for i, q in enumerate(step.queries) if (q.task, q.nodes) in step.changed_queries]
        if changed_queries:
            book.binary("changed_relations", query_scores[changed_queries] >= 0, query_truth[changed_queries])
    for key, loss in components.items():
        book.mean(f"transition.{key}.loss", loss.item())
    return sum(components.values())


def pool_loss(model: Network, frame: Frame, step: Step, book: Metrics) -> Tensor:
    prediction = model.pool_aux(frame.board)
    losses = []
    for index, (label, name) in enumerate(zip(step.pool, POOL_NAMES)):
        if label is None:
            continue
        target = prediction.new_tensor(label)
        if index < 5:
            losses.append(F.smooth_l1_loss(prediction[index], target / 8.))
            book.error(name, prediction[index] * 8., target)
        else:
            losses.append(F.binary_cross_entropy_with_logits(prediction[index], target))
            book.binary(name, prediction[index] >= 0, target)
        book.mean(f"{name}.loss", losses[-1].item())
        if name == "solvable":
            losses[-1] = losses[-1] * step.solvability_weight
    return torch.stack(losses).mean()


def value_loss(model: Network, frame: Frame, step: Step, book: Metrics) -> Tensor:
    guess = model.value(frame.board).squeeze()
    label = guess.new_tensor(step.value.value)
    book.error("value", guess, label)
    book.error("value_exact" if step.value.exact else "value_approximate", guess, label)
    loss = F.mse_loss(guess, label) * step.value.weight
    book.mean("value_exact.loss" if step.value.exact else "value_approximate.loss", loss.item())
    return loss


def imitation_loss(model: Network, frame: Frame, step: Step, book: Metrics,
                   approximate_weight: float) -> Tensor:
    logits, _ = model.policy([frame], [step.obs.actions])
    scores = logits[0]
    log_probs = scores.log_softmax(0)
    rate = float(scores.argmax().item() in step.best)
    book.mean("actor.accuracy", rate)
    book.mean("optimal.accuracy" if step.optimal else "approximate.accuracy", rate)
    book.mean("actor.multi_optimal_fraction", float(step.optimal and len(step.best) > 1))
    book.mean("actor.candidates", len(step.obs.actions))
    loss = -torch.logsumexp(log_probs[list(step.best)], 0)
    book.mean("actor.loss", loss.item())
    if step.value is not None:
        loss = loss + .25 * value_loss(model, frame, step, book)
    return loss * (1. if step.optimal else approximate_weight)


def step_loss(model: Network, frame: Frame, step: Step, stage: int, book: Metrics, cfg: Config) -> Tensor | None:
    if stage == 2:
        return transition_loss(model, frame, step, book)
    if stage == 3 and step.pool is not None:
        return pool_loss(model, frame, step, book)
    if stage == 4 and step.value is not None:
        return value_loss(model, frame, step, book)
    if stage == 5 and step.best:
        return imitation_loss(model, frame, step, book, cfg.optimizer.approximate_weight)
    return None


def apply_loss(losses: list[Tensor], book: Metrics, model: Network,
               opt: torch.optim.Optimizer | None, cfg: Config) -> None:
    if not losses:
        return
    loss = torch.stack(losses).mean()
    if not torch.isfinite(loss).item():
        raise FloatingPointError("Training loss is not finite")
    book.mean("loss", loss.item(), len(losses))
    if opt is not None:
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optimizer.grad_clip)
        opt.step()


def aggregate(book: Metrics, stage: int) -> dict:
    flat = book.flat()
    if stage == 1:
        f1 = [flat[f"{t}.f1"] for t in RELATION_TASKS if f"{t}.f1" in flat]
        if f1:
            flat["macro_f1"] = sum(f1) / len(f1)
            flat["tasks_covered"] = len(f1)
    elif stage == 2:
        names = ("birth", "remove", "edge.add", "edge.remove", "edge.change", "affected")
        flat["event_f1"] = sum(flat.get(f"{key}.f1", 0.) for key in names) / len(names)
    elif stage == 3:
        errors = [flat[f"{name}.mae"] for name in POOL_NAMES if f"{name}.mae" in flat]
        if errors:
            flat["pool_mae"] = sum(errors) / len(errors)
    return flat


def epoch(model: Network, dataset: list, stage: int, cfg: Config, number: int,
          opt: torch.optim.Optimizer | None, deadline: float = float("inf")) -> dict:
    training = opt is not None
    model.train(training)
    book = Metrics()
    rng = Random(derived_seed(cfg.runtime.seed, "minibatches", stage, number))
    data = list(dataset)
    if stage == 1 and training:
        cap = min(5, number + 1)
        data = [case for case in data if case.axis != "curriculum" or case.level <= cap]
    if training:
        rng.shuffle(data)
    with torch.set_grad_enabled(training):
        if stage == 1:
            size = cfg.optimizer.graph_batch
            for offset in range(0, len(data), size):
                cases = data[offset:offset + size]
                frames = model.encode([case.graph for case in cases], hgt_only=True)
                losses = [relation_loss(model, frame, case.queries, book) for frame, case in zip(frames, cases)]
                apply_loss(losses, book, model, opt, cfg)
                if monotonic() > deadline:
                    break
        else:
            batch_size, unroll = cfg.optimizer.sequence_batch, cfg.optimizer.bptt
            for offset in range(0, len(data), batch_size):
                batch = data[offset:offset + batch_size]
                memories = [None] * len(batch)
                end = max(len(episode.steps) for episode in batch)
                for start in range(0, end, unroll):
                    losses = []
                    for t in range(start, min(start + unroll, end)):
                        active = [(i, episode.steps[t]) for i, episode in enumerate(batch) if t < len(episode.steps)]
                        frames = model.encode([s.obs.graph for _, s in active], [s.obs.events for _, s in active],
                                              [memories[i] for i, _ in active])
                        for (index, step), frame in zip(active, frames):
                            memories[index] = frame.memory
                            loss = step_loss(model, frame, step, stage, book, cfg)
                            if loss is not None:
                                losses.append(loss)
                    apply_loss(losses, book, model, opt, cfg)
                    memories = [memory.detached() if memory is not None else None for memory in memories]
                    if monotonic() > deadline:
                        break
                if monotonic() > deadline:
                    break
    return aggregate(book, stage)


@torch.no_grad()
def validation_play(model: Network, worlds: list, cfg: Config) -> dict:
    model.eval()
    states = [initial_state(world) for world in worlds]
    memories, graphs = [None] * len(worlds), [None] * len(worlds)
    returns = [0.] * len(worlds)
    while True:
        active = [i for i, (world, state) in enumerate(zip(worlds, states)) if outcome(world, state) == "running"]
        if not active:
            break
        obs = [observation(worlds[i], states[i], graphs[i]) for i in active]
        frames = model.encode([o.graph for o in obs], [o.events for o in obs], [memories[i] for i in active])
        logits, _ = model.policy(frames, [o.actions for o in obs])
        for i, obs_i, frame, action in zip(active, obs, frames, logits.argmax(1).tolist()):
            memories[i], graphs[i] = frame.memory, obs_i.graph
            states[i], reward = transition(worlds[i], states[i], obs_i.actions[action], cfg.reward, checked=True)
            returns[i] += reward
    return {"rollout.clear_rate": sum(outcome(w, s) == "success" for w, s in zip(worlds, states)) / len(worlds),
            "rollout.return": sum(returns) / len(worlds), "rollout.reference_clear_rate": 1.,
            "rollout.worlds": len(worlds)}


def train_stage(model: Network, cfg: Config, stage: int, output: Path, passed: dict[int, bool],
                resume: Path | None = None) -> tuple[Path, bool]:
    settings = cfg.stages[stage]
    if stage == 1:
        train, val = relation_data(cfg, "train"), relation_data(cfg, "val")
    else:
        train, val = sequences(cfg, stage, "train"), sequences(cfg, stage, "val")
    val_play = []
    if stage == 5:
        val_play = [generate(derived_seed(cfg.runtime.seed, "validation-play", i), cfg.generation, cfg.search, "val").world
                    for i in range(cfg.data.validation_rollouts)]
    modules = {name: module for name, module in model.named_children() if name in STAGE_MODULES[stage]}
    opt = torch.optim.AdamW([p for mod in modules.values() for p in mod.parameters()], lr=cfg.optimizer.lr)
    folder = output / f"stage{stage}"
    best_path, last_path = folder / "best.pt", folder / "last.pt"
    number, stale, stable = 0, 0, 0
    best_key = (False, -float("inf"))
    first = monotonic()
    if resume is not None:
        saved = read_checkpoint(resume)
        data = saved["metadata"]
        if data.get("stage") != stage or saved["architecture"] != asdict(model.cfg):
            raise ValueError("Resume checkpoint belongs to a different stage or architecture")
        if not best_path.exists():
            raise ValueError("Resume into the original output directory containing best.pt")
        model.load_state_dict(saved["model"])
        opt.load_state_dict(saved["optimizer"])
        restore_rng(saved, model.device)
        number, stale, stable = data["epoch"], data["stale"], data["stable"]
        best_key = tuple(data["best_key"])
    elif last_path.exists():
        raise ValueError(f"{folder} already contains a run. Select --resume or a different output directory.")
    stop = "epoch_limit"
    lower = settings.score.endswith((".mae", ".mse", "loss", "_mae"))
    for number in range(number + 1, settings.epochs + 1):
        for name, module in model.named_children():
            module.requires_grad_(name in modules and not (number <= settings.freeze_epochs and name in settings.freeze_modules))
        deadline = first + settings.max_seconds if settings.max_seconds > 0 else float("inf")
        training = epoch(model, train, stage, cfg, number, opt, deadline)
        validation = epoch(model, val, stage, cfg, number, None)
        if val_play:
            validation.update(validation_play(model, val_play, cfg))
        met = all(key in validation and all(validation[key] >= goal if mode == "min" else validation[key] <= goal
                                            for mode, goal in limits.items()) for key, limits in settings.goals.items())
        stable = stable + 1 if met else 0
        raw_score = validation.get(settings.score, validation.get("loss", 0.) if lower else 0.)
        score = -raw_score if lower else raw_score
        candidate = (met, score)
        is_best = candidate > best_key
        improved = met > best_key[0] or (met == best_key[0] and score > best_key[1] + settings.min_delta)
        stale = 0 if improved else stale + 1
        if is_best:
            best_key = candidate
        ready = stable >= settings.stable
        passed[stage] = ready
        info = {"stage": stage, "epoch": number, "stale": stale, "stable": stable, "best_key": best_key,
                "stages_passed": dict(passed), "ready": ready, "kind": "pretrain"}
        checkpoint(last_path, model, opt, cfg, **info)
        if is_best:
            checkpoint(best_path, model, opt, cfg, **info)
        gap = None
        if settings.score in training and settings.score in validation:
            gap = (validation[settings.score] - training[settings.score]) * (1 if lower else -1)
        append_json(folder / "metrics.jsonl", {"epoch": number, "train": training, "validation": validation,
                    "overfit_gap": gap, "goals_met": met, "stable": stable,
                    "elapsed_seconds": monotonic() - first, "parameters": weight_count(model)})
        print(f"stage={stage} epoch={number} validation {settings.score}={raw_score:.4f} targets={met}", flush=True)
        if ready:
            stop = "targets_stable"
            break
        if stale >= settings.patience and stable == 0:
            stop = "plateau"
            break
        if monotonic() > deadline:
            stop = "time_limit"
            break
    passed[stage] = stable >= settings.stable
    write_json(folder / "status.json", {"stage": stage, "ready": passed[stage], "stop": stop, "epoch": number,
                                        "goals": settings.goals, "best_score": best_key[1] * (-1 if lower else 1)})
    return best_path, passed[stage]


def pretrain(cfg: Config, device: torch.device, output: Path, stage: int | None = None,
             init: Path | None = None, resume: Path | None = None) -> Path:
    model = Network(cfg.model).to(device)
    passed = {}
    if init is not None or resume is not None:
        data = read_checkpoint(resume or init)
        if data["architecture"] != asdict(cfg.model):
            raise ValueError("The checkpoint architecture differs from the configuration")
        if init is not None and stage is not None and stage > 1 and data["metadata"].get("stage") != stage - 1:
            raise ValueError("Select the preceding stage's checkpoint with --init")
        model.load_state_dict(data["model"])
        passed.update(data["metadata"].get("stages_passed", {}))
    if resume is not None and stage is None:
        raise ValueError("Select --stage when resuming")
    if stage is not None and stage > 1 and init is None and resume is None:
        raise ValueError("Select --init with the preceding stage's best checkpoint")
    path = None
    for selected in range(1, 6) if stage is None else (stage,):
        path, ready = train_stage(model, cfg, selected, output, passed, resume)
        model.load_state_dict(read_checkpoint(path)["model"])
        # The stable validation decision may occur after the epoch with the best score.
        best = read_checkpoint(path)
        best["metadata"].update(stages_passed=dict(passed), ready=ready)
        tmp = path.with_suffix(".tmp")
        torch.save(best, tmp)
        tmp.replace(path)
    write_json(output / "pretraining.json", {"stages_passed": passed, "best": str(path.relative_to(output)),
                                            "ready_for_rl": all(passed.get(i, False) for i in range(1, 6))})
    return path

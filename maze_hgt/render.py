"""Japanese rule listings, stable ASCII coordinates and matplotlib figures."""
from __future__ import annotations

from pathlib import Path

from .game import MATERIALS, STATUS, Action, Condition, State, World, outcome
from .graph import CONDITIONS, EFFECTS, FORWARD, Graph

STATUS_JA = ("毒", "弱体化", "脆弱化")
AFFINITY_JA = {"weak": "弱点", "critical": "特効弱点", "resist": "耐性"}
SUPPORT_JA = {"boost": "攻撃力を強化", "protect": "直接ダメージを軽減", "heal": "ターンごとに回復"}
LEGEND = "# 壁 / . 通路 / P プレイヤー / G ゴール / D 閉扉 / d 開扉 / E 敵 / C 宝箱 / * 落下アイテム"
RESULT_JA = {"running": "進行中", "success": "クリア", "death": "死亡", "timeout": "ターン上限"}


def condition_text(world: World, rule: Condition) -> str:
    if rule.op in ("and", "or"):
        return "(" + (" かつ " if rule.op == "and" else " または ").join(
            condition_text(world, child) for child in rule.children) + ")"
    if rule.op == "item":
        return f"{world.items[rule.ref].name} を選択"
    if rule.op == "dead":
        return f"E{rule.ref} を撃破済み"
    if rule.op == "status":
        return f"対象が{STATUS_JA[rule.ref]}状態"
    if rule.op == "breach":
        return f"{MATERIALS[rule.ref]}の材質に対する破壊効果が成立"
    return "無条件"


def rules_text(world: World) -> str:
    lines = ["=== ルール ==="]
    for i, item in enumerate(world.items):
        count = "再使用可" if item.charges < 0 else f"{item.charges}回"
        lines.append(f"{item.name}（{'鍵' if item.key else 'アイテム'}、{count}）")
    for effect in world.effects:
        item = world.items[effect.item].name
        target = f"E{effect.target}" if effect.target >= 0 else "対象の敵"
        if effect.kind == "breach":
            desc = f"{MATERIALS[effect.target]}の材質に破壊効果"
        elif effect.kind == "heal":
            desc = f"プレイヤーのHPを{effect.power}回復"
        elif effect.kind == "damage":
            desc = f"{target}に基礎ダメージ{effect.power}"
        else:
            desc = f"{target}に{STATUS_JA[STATUS.index(effect.kind)]}を{effect.power}ターン付与"
        lines.append(f"{item} → {desc}（{condition_text(world, effect.requires)}）")
    for i, door in enumerate(world.doors):
        lines.append(f"D{i}（{MATERIALS[door.material]}）の開放条件 {condition_text(world, door.requires)}")
    for affinity in world.affinities:
        label = AFFINITY_JA[affinity.kind]
        item = world.items[affinity.item].name
        lines.append(f"E{affinity.enemy} の {item} に対する{label} "
                     f"×{affinity.numerator}/{affinity.denominator}（{condition_text(world, affinity.requires)}）")
    for support in world.supports:
        lines.append(f"E{support.source} → E{support.target} の{SUPPORT_JA[support.kind]}（{support.power}）")
    for i, chest in enumerate(world.chests):
        lines.append(f"C{i} の中身 " + (", ".join(world.items[k].name for k in chest.items) or "空"))
    for i, enemy in enumerate(world.enemies):
        if enemy.loot:
            lines.append(f"E{i} の撃破時に落下 " + ", ".join(world.items[k].name for k in enemy.loot))
    return "\n".join(lines)


def ascii_board(world: World, state: State) -> str:
    board = [list(row) for row in world.grid]
    marks = [(d.pos, "d" if i in state.opened else "D") for i, d in enumerate(world.doors)]
    marks += [(c.pos, "C") for i, c in enumerate(world.chests) if i not in state.looted]
    marks += [(e.pos, "E") for i, e in enumerate(world.enemies) if state.enemy_hp[i]]
    marks += [(pos, "*") for pos, _ in state.ground]
    marks += [(world.goal, "G"), (state.pos, "P")]
    for (r, c), char in marks:
        board[r][c] = char
    return "\n".join("".join(row) for row in board)


def inventory(world: World, state: State) -> str:
    return ", ".join(world.items[i].name + (f"({state.stock[i]})" if state.stock[i] > 0 else "")
                     for i in state.inventory) or "なし"


def ascii_screen(world: World, state: State) -> str:
    result = RESULT_JA[outcome(world, state)]
    rows = [f"=== 盤面 / ターン {state.turn} / {result} ===", ascii_board(world, state), LEGEND,
            f"HP {state.hp}/{world.max_hp}  所持品 {inventory(world, state)}  位置 {state.pos}"]
    for i, door in enumerate(world.doors):
        rows.append(f"D{i} {door.pos} {MATERIALS[door.material]} {'開' if i in state.opened else '閉'}")
    for i, enemy in enumerate(world.enemies):
        timers = [f"{name}:{timer}" for name, timer in zip(STATUS_JA, state.statuses[i]) if timer]
        rows.append(f"E{i} {enemy.pos} HP {state.enemy_hp[i]}/{enemy.hp} 攻撃 {enemy.attack} " + " ".join(timers))
    for i, chest in enumerate(world.chests):
        rows.append(f"C{i} {chest.pos} {'開封済み' if i in state.looted else '未開封'}")
    for pos, item in state.ground:
        rows.append(f"* {pos} {world.items[item].name}")
    return "\n".join(rows)


def action_text(world: World, action: Action) -> str:
    names = ", ".join(world.items[i].name for i in action.items)
    if action.kind == "move":
        return f"{action.target} に移動"
    if action.kind == "chest":
        return f"C{action.target} を開ける"
    if action.kind in ("key", "breach", "combine"):
        return f"D{action.target} に {names} を試す"
    if action.kind == "use":
        return f"E{action.target} に {names} を使う"
    if action.kind == "attack":
        return f"E{action.target} を攻撃"
    return f"{names} を自分に使う" if action.kind == "heal" else "待機"


def _display_node(world: World, node, japanese: bool) -> str:
    key, kind = node.id, node.kind
    if kind in ("item", "key"):
        return world.items[int(key.split(":")[1])].name
    if kind in ("enemy", "door", "chest"):
        return f"{kind[0].upper()}{key.split(':')[1]}"
    if kind == "condition":
        op = CONDITIONS[node.x[3:].index(1.)]
        return {"and": "AND", "or": "OR", "dead": "KO" + (" ✓" if node.x[0] else ""),
                "item": "ITEM", "breach": "BREAK", "status": "STATUS", "true": "TRUE"}.get(op, op)
    if kind == "material":
        i = int(key.split(":")[1])
        return MATERIALS[i] if japanese else f"M{i}"
    if kind == "effect":
        op = EFFECTS[node.x[3:].index(1.)]
        return {"damage": "DMG", "poison": "PSN", "weaken": "WEAK", "vulnerable": "VUL",
                "heal": "HEAL", "breach": "BREAK"}[op]
    if kind == "status_type":
        i = int(key.split(":")[1])
        return STATUS_JA[i] if japanese else STATUS[i]
    if kind == "affinity":
        numerator, denominator = round(4 * node.x[0]), round(4 * node.x[1])
        return f"×{numerator}" if denominator == 1 else f"×{numerator}/{denominator}"
    if kind == "support":
        return f"SUP+{round(4 * node.x[0])}"
    if kind == "status":
        return f"T{round(8 * node.x[0])}"
    return "P" if kind == "player" else "G"


class Viewer:
    def __init__(self, mode: str, output: Path | None = None):
        self.mode, self.output = mode, output
        self.figure = None
        if mode in ("frames", "live"):
            import matplotlib
            if mode == "frames":
                matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib import font_manager
            self.plt = plt
            self.japanese = any(font.name == "Noto Sans CJK JP" for font in font_manager.fontManager.ttflist)
            self.font = "Noto Sans CJK JP" if self.japanese else "DejaVu Sans"
            self.figure = plt.figure(figsize=(14, 12), layout="constrained")
            if mode == "live":
                plt.ion()

    def frame(self, world: World, state: State, graph: Graph, *, print_ascii: bool = True) -> None:
        if print_ascii:
            print(ascii_screen(world, state))
        if self.figure is None:
            return
        from matplotlib.lines import Line2D
        from matplotlib.patches import FancyArrowPatch
        fig = self.figure
        fig.clear()
        fig.set_facecolor("#f6f8fb")
        grid = fig.add_gridspec(2, 2, height_ratios=(3, 2), width_ratios=(1, 2))
        ax, board, details = fig.add_subplot(grid[0, :]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])
        for subplot in (ax, board, details):
            subplot.set_axis_off()
        layers = (("player", "chest"), ("key", "item"), ("effect", "affinity", "support"),
                  ("condition",), ("door", "enemy", "material", "status", "status_type", "goal"))
        titles = ("所持・格納", "アイテム・鍵", "効果・関係", "条件", "対象・状態") if self.japanese else (
            "Ownership", "Items / keys", "Effects / relations", "Conditions", "Targets / status")
        palette = ("#1c5662", "#334f90", "#994825", "#78488c", "#31623f")
        edge_palette = {"boosts": "#b73c43", "protects": "#1e6ab3", "heals": "#26734d",
                        "operand": "#8261a8", "needs_item": "#61428c", "made_of": "#a35d28",
                        "matches_material": "#a35d28", "holds": "#1c5662", "contains": "#1c5662",
                        "drops": "#1c5662"}
        positions, colors = {}, {}
        for x, kinds in enumerate(layers):
            nodes = sorted((n for n in graph.nodes if n.kind in kinds), key=lambda n: (kinds.index(n.kind), n.id))
            for i, node in enumerate(nodes):
                positions[node.id] = (x, .91 - .83 * (i + .5) / max(1, len(nodes)))
                colors[node.id] = palette[x]
            ax.text(x, .975, titles[x], ha="center", color=palette[x], fontfamily=self.font, weight="bold", fontsize=11)
        for edge in graph.edges:
            if edge.rel not in FORWARD or edge.src not in positions or edge.dst not in positions:
                continue
            p, q = positions[edge.src], positions[edge.dst]
            connection = "arc3,rad=.28" if p[0] == q[0] else "arc3,rad=.07"
            ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", connectionstyle=connection, shrinkA=12, shrinkB=12,
                                         mutation_scale=8, alpha=.46, linewidth=.75,
                                         color=edge_palette.get(edge.rel, "#738093")))
        max_column = max((sum(p[0] == x for p in positions.values()) for x in range(len(layers))), default=1)
        size = max(4, min(9, 170 / max_column))
        for key, (x, y) in positions.items():
            ax.text(x, y, _display_node(world, graph.table[key], self.japanese), ha="center", va="center",
                    fontsize=size, fontfamily=self.font, color="white",
                    bbox={"boxstyle": "round,pad=.21", "facecolor": colors[key], "edgecolor": "none"})
        ax.set(xlim=(-.55, 4.55), ylim=(-.06, 1.04))
        handles = [Line2D([0], [0], color=edge_palette[rel], lw=1.5, label=rel) for rel in
                   ("holds", "matches_material", "operand", "boosts", "protects", "heals")]
        ax.legend(handles=handles, loc="lower center", ncol=6, fontsize=7, frameon=False)
        board.text(.05, .96, ascii_board(world, state), va="top", family="monospace",
                   fontsize=max(7, min(19, 205 / len(world.grid))), color="#132c47")
        summary = ascii_screen(world, state).splitlines()[1 + len(world.grid):]
        if self.japanese:
            title = f"ターン {state.turn}  |  {RESULT_JA[outcome(world, state)]}"
            details.text(.01, .87, "\n".join(summary), va="top", family=self.font,
                         fontsize=max(7, min(11, 180 / max(1, len(summary)))), color="#132c47")
        else:
            title = f"Turn {state.turn}  |  {outcome(world, state)}"
            details.text(.01, .87, f"HP {state.hp}/{world.max_hp}   Pos {state.pos}\n"
                         + "Inventory: " + ", ".join(world.items[i].name for i in state.inventory)
                         + "\n" + "\n".join(f"D{i} {d.pos} {'open' if i in state.opened else 'closed'}"
                                             for i, d in enumerate(world.doors)),
                         va="top", family=self.font, color="#132c47")
        details.text(.01, .99, title, va="top", family=self.font, weight="bold", fontsize=16, color="#132c47")
        if self.mode == "frames":
            self.output.mkdir(parents=True, exist_ok=True)
            fig.savefig(self.output / f"{state.turn:04d}.png", dpi=105)
        elif self.mode == "live":
            fig.canvas.draw_idle()
            self.plt.pause(.08)

    def close(self) -> None:
        if self.figure is not None:
            self.plt.close(self.figure)


def curves(directory: Path, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .io import read_jsonl

    sources = [(path, read_jsonl(path)) for path in sorted(directory.rglob("metrics.jsonl"))]
    if not sources:
        raise ValueError(f"No metrics.jsonl under {directory}")
    rows = len(sources)
    fig, grid = plt.subplots(rows, 3, figsize=(14, 3.3 * rows), squeeze=False, layout="constrained")
    for (path, log), axes in zip(sources, grid):
        label = path.parent.name
        if "epoch" in log[0]:
            choices = (("loss",),
                       ("macro_f1", "event_f1", "pool_mae", "value_exact.mae", "optimal.accuracy"),
                       ("relations.accuracy", "birth.f1", "solvable.balanced_accuracy",
                        "value_approximate.mae", "rollout.clear_rate"))
            xkey = "epoch"
        else:
            choices = (("return", "clear_rate", "rollout.clear_rate"), ("policy_loss", "value_loss"),
                       ("entropy", "kl", "reference_kl"))
            xkey = "update"
        for axis, keys in zip(axes, choices):
            for key in keys:
                for split, color in (("train", "#235b90"), ("validation", "#b6502c")):
                    xy = [(row[xkey], row[split][key]) for row in log if key in row.get(split, {})]
                    if xy:
                        axis.plot(*zip(*xy), marker="o", ms=3, label=f"{split} {key}", alpha=.87,
                                  color=color if len(keys) == 1 else None)
            axis.grid(alpha=.2)
            axis.set_title(label)
            axis.set_xlabel(xkey)
            if axis.lines:
                axis.legend(fontsize=7, loc="best")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=135)
    plt.close(fig)
    supervised = [(path, log) for path, log in sources if "epoch" in log[0]]
    if not supervised:
        return
    from .graph import RELATION_TASKS
    from .tasks import POOL_NAMES

    task_groups = {"stage1": tuple(f"{t}.loss" for t in RELATION_TASKS),
                   "stage3": tuple(f"{t}.loss" for t in POOL_NAMES),
                   "stage4": ("value_exact.loss", "value_approximate.loss"),
                   "stage5": ("actor.loss", "value_exact.loss", "value_approximate.loss")}
    aux, axes = plt.subplots(len(supervised), 2, figsize=(15, 3.4 * len(supervised)),
                             squeeze=False, layout="constrained")
    for (path, log), row in zip(supervised, axes):
        stage = path.parent.name
        keys = task_groups.get(stage, sorted(k for k in log[-1]["train"] if k.startswith("transition.")
                                            and k.endswith(".loss")))
        for axis, split in zip(row, ("train", "validation")):
            for key in keys:
                points = [(x["epoch"], x[split][key]) for x in log if key in x[split]]
                if points:
                    axis.plot(*zip(*points), marker=".", label=key.removesuffix(".loss").removeprefix("transition."))
            axis.set(title=f"{stage} {split}", xlabel="epoch", ylabel="task loss")
            axis.grid(alpha=.2)
            if axis.lines:
                axis.legend(fontsize=6, ncol=2, loc="best")
    aux.savefig(output.with_name(output.stem + "_tasks" + output.suffix), dpi=135)
    plt.close(aux)

"""Generate figures used by performance_check_report.md."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
from PIL import Image
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "python"))

from tiny_transformer.tokenizer import CharTokenizer  # noqa: E402


FONT = FontProperties(
    fname="/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc"
)
COLORS = {"1000": "#0072B2", "3000": "#D55E00"}


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.tick_params(labelsize=9)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(FONT)


def read_metrics(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    return (
        np.array([row["step"] for row in rows]),
        np.array([row["loss"] for row in rows]),
    )


def rolling_mean(values: np.ndarray, window: int = 50) -> np.ndarray:
    result = np.full(values.shape, np.nan, dtype=float)
    if len(values) >= window:
        result[window - 1 :] = np.convolve(
            values, np.ones(window) / window, mode="valid"
        )
    return result


def plot_loss() -> None:
    runs = [
        ("1000-step実験", ROOT / "outputs/traces/test_20260718/metrics.jsonl", "1000"),
        (
            "3000-step実験",
            ROOT / "outputs/traces/train_20260722_3000steps/metrics.jsonl",
            "3000",
        ),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, (name, path, key) in zip(axes, runs, strict=True):
        steps, losses = read_metrics(path)
        ax.plot(steps, losses, color=COLORS[key], alpha=0.22, linewidth=0.7, label="各step")
        ax.plot(
            steps,
            rolling_mean(losses),
            color=COLORS[key],
            linewidth=2.0,
            label="50 step移動平均",
        )
        ax.set_yscale("log")
        ax.set_title(name, fontproperties=FONT, fontsize=13)
        ax.set_xlabel("学習step", fontproperties=FONT)
        ax.set_ylabel("Cross entropy loss（対数軸）", fontproperties=FONT)
        ax.legend(prop=FONT, frameon=False)
        style_axis(ax)
    fig.suptitle("学習lossの推移", fontproperties=FONT, fontsize=16)
    fig.savefig(HERE / "report_loss_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def trace_probabilities(trace_dir: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    tokenizer = CharTokenizer((ROOT / "data/tiny_corpus.txt").read_text())
    target_tokens = [" ", "g", "m"]
    target_ids = {token: tokenizer.encode(token)[0] for token in target_tokens}
    steps: list[int] = []
    values = {token: [] for token in target_tokens}
    for path in sorted(trace_dir.glob("step_*_probe.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        steps.append(int(payload["step"]))
        probabilities = torch.softmax(payload["trace"].logits[0, -1], dim=-1)
        for token, token_id in target_ids.items():
            values[token].append(float(probabilities[token_id]))
    return np.array(steps), {token: np.array(probs) for token, probs in values.items()}


def plot_prediction_probabilities() -> None:
    runs = [
        ("1000-step実験", ROOT / "outputs/traces/test_20260718"),
        ("3000-step実験", ROOT / "outputs/traces/train_20260722_3000steps"),
    ]
    token_styles = {
        " ": ("空白（期待される次文字）", "#009E73", "o"),
        "g": ("g", "#CC79A7", "s"),
        "m": ("m", "#E69F00", "^"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, (name, trace_dir) in zip(axes, runs, strict=True):
        steps, probabilities = trace_probabilities(trace_dir)
        for token, (label, color, marker) in token_styles.items():
            ax.plot(
                steps,
                probabilities[token],
                label=label,
                color=color,
                marker=marker,
                markersize=4,
                linewidth=1.8,
            )
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(name, fontproperties=FONT, fontsize=13)
        ax.set_xlabel("trace取得step", fontproperties=FONT)
        ax.set_ylabel("予測確率", fontproperties=FONT)
        ax.legend(prop=FONT, frameon=False, fontsize=8)
        style_axis(ax)
    fig.suptitle(
        "固定probe「hello」末尾における次文字予測",
        fontproperties=FONT,
        fontsize=16,
    )
    fig.savefig(HERE / "report_probe_probabilities.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def read_attention_summary(path: Path) -> list[dict[str, float]]:
    with path.open(newline="") as file:
        return [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(file)
        ]


def plot_attention_metrics() -> None:
    runs = [
        ("1000-step実験", HERE / "attention_summary_1000steps.csv", "1000"),
        ("3000-step実験", HERE / "attention_summary_3000steps.csv", "3000"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, (name, path, key) in zip(axes, runs, strict=True):
        rows = read_attention_summary(path)
        steps = sorted({int(row["step"]) for row in rows})
        mean_weight = [
            np.mean([row["max_attention_weight"] for row in rows if row["step"] == step])
            for step in steps
        ]
        mean_entropy = [
            np.mean([row["attention_entropy"] for row in rows if row["step"] == step])
            for step in steps
        ]
        ax.plot(
            steps,
            mean_weight,
            color=COLORS[key],
            marker="o",
            linewidth=1.8,
            label="最大attention weight（4 head平均）",
        )
        entropy_ax = ax.twinx()
        entropy_ax.plot(
            steps,
            mean_entropy,
            color="#555555",
            marker="s",
            linewidth=1.5,
            linestyle="--",
            label="attention entropy（4 head平均）",
        )
        entropy_ax.axhline(np.log(5), color="#999999", linestyle=":", linewidth=1)
        ax.set_ylim(0.15, 0.9)
        entropy_ax.set_ylim(0.5, 1.65)
        ax.set_title(name, fontproperties=FONT, fontsize=13)
        ax.set_xlabel("trace取得step", fontproperties=FONT)
        ax.set_ylabel("最大attention weight", fontproperties=FONT, color=COLORS[key])
        entropy_ax.set_ylabel("Attention entropy", fontproperties=FONT, color="#555555")
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = entropy_ax.get_legend_handles_labels()
        ax.legend(handles1 + handles2, labels1 + labels2, prop=FONT, frameon=False, fontsize=7)
        style_axis(ax)
        entropy_ax.tick_params(labelsize=9)
    fig.suptitle(
        "「hello」末尾位置におけるattention集中度の推移",
        fontproperties=FONT,
        fontsize=16,
    )
    fig.savefig(HERE / "report_attention_metrics.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def combine_attention_heatmaps() -> None:
    paths = [
        HERE / "attention_heatmaps_3000steps/step_000001_attention.png",
        HERE / "attention_heatmaps_3000steps/step_003000_attention.png",
    ]
    images = [Image.open(path).convert("RGB") for path in paths]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    for ax, image, title in zip(
        axes, images, ["step 1（学習初期）", "step 3000（学習後）"], strict=True
    ):
        ax.imshow(image)
        ax.set_title(title, fontproperties=FONT, fontsize=14)
        ax.axis("off")
    fig.suptitle(
        "3000-step実験におけるattention heatmapの変化",
        fontproperties=FONT,
        fontsize=17,
    )
    fig.savefig(HERE / "report_attention_before_after.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    plot_loss()
    plot_prediction_probabilities()
    plot_attention_metrics()
    combine_attention_heatmaps()
    for path in sorted(HERE.glob("report_*.png")):
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()

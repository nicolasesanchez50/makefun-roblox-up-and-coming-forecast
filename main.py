#!/usr/bin/env python3
"""Forecast Roblox's Up-and-Coming top 10 for the 72 hours after entries close.

One forecast is written: for each of the chart's ten places, a probability for
each game named being at that place, plus the entrant's top 10 in order. A
single forecast is scored against every hourly read of the window, so the
probabilities are the marginal ones over the whole window, not one hour's.

The model reads the history it is given. A game's chance of holding a place is
a kernel over the distance between that place and the rank the game holds when
the history ends: a peak on its own place (the chart is sticky: roughly half
the places hold from one read to the next), a Gaussian skirt for the places it
can drift into, and a small shift down the chart because games age out of the
28-day window as the hours pass. Only games near the top are named; the rest
of each place's mass is spread over them, so a surprise costs a few nats
rather than everything.
"""

import json
import math
import os
from pathlib import Path

PLACES = 10
POOL = 18          # games named as possible holders of a place
SPREAD = 3.2       # rank-space width of the skirt around a game's own place
PEAK = 5.0         # extra weight on the game already at that place
DRIFT = 0.8        # places a game is expected to lose across the window
FIELD = 0.25       # share of each place left to the games not named
MIN_P = 1e-6


def input_dir() -> Path:
    return Path(os.environ.get("MAKEFUN_INPUT_DIR") or "/input")


def output_dir() -> Path:
    return Path(os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output")


def read_history(path: Path):
    """Every read, oldest first, as {universe_id: rank}."""
    reads = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        chart = {}
        for entry in row.get("up_and_coming") or []:
            rank, uid = entry.get("rank"), entry.get("universe_id")
            if isinstance(rank, int) and isinstance(uid, int):
                chart[uid] = rank
        if chart:
            reads.append(chart)
    return reads


def candidates(last, recent):
    """Games that can hold one of the places: near the top now or lately."""
    pool = {uid for uid, rank in last.items() if rank <= POOL}
    for chart in recent:
        pool |= {uid for uid, rank in chart.items() if rank <= PLACES}
    return sorted(pool)


def weights(rank_by_game, games, places):
    """Per-place weights: the peak on the game already there, a skirt, a drift."""
    rows = []
    for place in range(1, places + 1):
        values = []
        for uid in games:
            r = float(rank_by_game.get(uid) or 10 ** 6)
            d = (r + DRIFT) - place
            w = math.exp(-0.5 * (d / SPREAD) ** 2)
            if int(r) == place:
                w *= PEAK
            values.append(w)
        total = sum(values)
        if total <= 0:
            continue
        for uid, w in zip(games, values):
            p = (1.0 - FIELD) * (w / total)
            if p >= MIN_P:
                rows.append({"universe_id": uid, "place": place, "p": round(p, 7)})
    rows.sort(key=lambda row: (row["place"], -row["p"]))
    return rows


def best_order(probabilities, places):
    """Greedy top 10: the likeliest game for each place, each game used once."""
    by_place = {}
    for row in probabilities:
        by_place.setdefault(row["place"], []).append(row)
    chosen, used = [], set()
    for place in range(1, places + 1):
        for row in sorted(by_place.get(place, []), key=lambda r: -r["p"]):
            if row["universe_id"] not in used:
                chosen.append(row["universe_id"])
                used.add(row["universe_id"])
                break
    return chosen


def forecast(history_path: Path, places: int):
    reads = read_history(history_path)
    if not reads:
        raise SystemExit("history.ndjson holds no read")
    last = reads[-1]
    games = candidates(last, reads[-3:])
    probabilities = weights(last, games, places)
    return {"probabilities": probabilities, "top10": best_order(probabilities, places)}


def main() -> int:
    task_path = input_dir() / "task.json"
    task = json.loads(task_path.read_text(encoding="utf-8")) if task_path.exists() else {}
    places = int(task.get("places") or PLACES)
    name = task.get("output") or "forecast.json"
    result = forecast(input_dir() / "history.ndjson", places)
    out = output_dir()
    out.mkdir(parents=True, exist_ok=True)
    (out / name).write_text(json.dumps(result, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out / name}: {len(result['probabilities'])} rows, top10 {len(result['top10'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["font.family"] = "Times New Roman"

# ==================================================
# Global figure style settings
# ==================================================
ms = 9

dpisize = 400
ticksize = 14
labelsize = 16
titlesize = labelsize + 2
legendsize = ticksize

FIGURES_DIR = Path(__file__).resolve().parent / "Figures"


def save_figure(fig, save_filename):
    FIGURES_DIR.mkdir(exist_ok=True)
    output_path = FIGURES_DIR / Path(save_filename).name
    fig.savefig(output_path, dpi=dpisize, bbox_inches="tight")
    print(f"Saved: {output_path}")


def random_argmax(scores, rng):
    max_scores = scores.max(axis=1, keepdims=True)
    is_max = scores == max_scores
    random_values = rng.random(scores.shape)
    random_values[~is_max] = -1.0
    return np.argmax(random_values, axis=1)


def run_binary_choice_game(N, M, S=2, lambda_val=-1.0, gamma=0.0,
                           total_steps=2000, eq_steps=1000,
                           return_records=False, seed=None):
    if eq_steps >= total_steps:
        raise ValueError("eq_steps must be smaller than total_steps.")

    rng = np.random.default_rng(seed)
    P = 2 ** M

    strategies = rng.choice([-1, 1], size=(N, S, P))
    scores = np.zeros((N, S))
    history = rng.integers(0, P)

    A_records, A_all_records, success_records = [], [], []

    for t in range(total_steps):
        best_s = random_argmax(scores, rng)
        actions = strategies[np.arange(N), best_s, history]
        A = np.sum(actions)

        A_all_records.append(A)

        sign_A = 1 if A > 0 else (-1 if A < 0 else 0)

        if sign_A == 0:
            winning_action = rng.choice([-1, 1])
            reward = 0.0
        else:
            reward = lambda_val * sign_A * ((abs(A) / N) ** float(gamma))
            winning_action = -sign_A if lambda_val < 0 else sign_A

        if t >= eq_steps:
            A_records.append(A)

            if lambda_val < 0:
                success_rate_t = (N - abs(A)) / (2 * N)
            else:
                success_rate_t = (N + abs(A)) / (2 * N)

            success_records.append(success_rate_t)

        scores += strategies[:, :, history] * reward

        bit = 1 if winning_action == 1 else 0
        history = ((history << 1) | bit) % P

    A_records = np.array(A_records)
    A_all_records = np.array(A_all_records)

    volatility = np.sqrt(np.var(A_records) / N)
    average_success = np.mean(success_records)

    if return_records:
        return volatility, average_success, A_all_records, A_records

    return volatility, average_success


def run_mixed_m_game(N, M1=2, M2=10, S=2,
                     total_steps=2000, eq_steps=1000, seed=None):
    """
    Mixed-M minority game.

    Group 1 has cognitive capacity M1.
    Group 2 has cognitive capacity M2.

    Returns:
        success1: Average Success Rate for group 1
        success2: Average Success Rate for group 2
        minmaj1: old DV, mean value of -abs(A1)
        minmaj2: old DV, mean value of -abs(A2)
    """

    if eq_steps >= total_steps:
        raise ValueError("eq_steps must be smaller than total_steps.")

    rng = np.random.default_rng(seed)

    # Because N is odd, randomly assign the extra agent to one group.
    if rng.random() < 0.5:
        N1 = N // 2
        N2 = N - N1
    else:
        N1 = N // 2 + 1
        N2 = N - N1

    P1, P2 = 2 ** M1, 2 ** M2
    max_P = max(P1, P2)

    strategies1 = rng.choice([-1, 1], size=(N1, S, P1))
    strategies2 = rng.choice([-1, 1], size=(N2, S, P2))

    scores1 = np.zeros((N1, S))
    scores2 = np.zeros((N2, S))

    history = rng.integers(0, max_P)

    wins1 = 0
    wins2 = 0
    eval_steps = total_steps - eq_steps

    minmaj1_records, minmaj2_records = [], []

    for t in range(total_steps):
        h1 = history % P1
        h2 = history % P2

        best_s1 = random_argmax(scores1, rng)
        best_s2 = random_argmax(scores2, rng)

        act1 = strategies1[np.arange(N1), best_s1, h1]
        act2 = strategies2[np.arange(N2), best_s2, h2]

        A1 = np.sum(act1)
        A2 = np.sum(act2)
        A = A1 + A2

        sign_A = 1 if A > 0 else (-1 if A < 0 else 0)

        if sign_A == 0:
            winning_action = rng.choice([-1, 1])
        else:
            winning_action = -sign_A

        if t >= eq_steps:
            wins1 += np.sum(act1 == winning_action)
            wins2 += np.sum(act2 == winning_action)

            # Old DV kept for compatibility.
            minmaj1_records.append(-abs(A1))
            minmaj2_records.append(-abs(A2))

        scores1 += strategies1[:, :, h1] * winning_action
        scores2 += strategies2[:, :, h2] * winning_action

        bit = 1 if winning_action == 1 else 0
        history = ((history << 1) | bit) % max_P

    success1 = wins1 / (N1 * eval_steps)
    success2 = wins2 / (N2 * eval_steps)

    minmaj1 = np.mean(minmaj1_records)
    minmaj2 = np.mean(minmaj2_records)

    return success1, success2, minmaj1, minmaj2


def run_mixed_m_game_large_m_balanced(N, M1=2, M2=10, S=2,
                                      total_steps=2000, eq_steps=1000,
                                      seed=None):
    if eq_steps >= total_steps:
        raise ValueError("eq_steps must be smaller than total_steps.")

    rng = np.random.default_rng(seed)

    if rng.random() < 0.5:
        N1 = N // 2
        N2 = N - N1
    else:
        N1 = N // 2 + 1
        N2 = N - N1

    P1, P2 = 2 ** M1, 2 ** M2
    max_P = max(P1, P2)

    strategies1 = rng.choice([-1, 1], size=(N1, S, P1))
    scores1 = np.zeros((N1, S))

    history = rng.integers(0, max_P)

    wins1 = 0
    wins2 = 0
    eval_steps = total_steps - eq_steps

    minmaj1_records, minmaj2_records = [], []

    for t in range(total_steps):
        h1 = history % P1

        best_s1 = random_argmax(scores1, rng)
        act1 = strategies1[np.arange(N1), best_s1, h1]

        n_plus = N2 // 2
        n_minus = N2 - n_plus

        act2 = np.concatenate([
            np.ones(n_plus, dtype=int),
            -np.ones(n_minus, dtype=int)
        ])
        rng.shuffle(act2)

        A1 = np.sum(act1)
        A2 = np.sum(act2)
        A = A1 + A2

        sign_A = 1 if A > 0 else (-1 if A < 0 else 0)

        if sign_A == 0:
            winning_action = rng.choice([-1, 1])
        else:
            winning_action = -sign_A

        if t >= eq_steps:
            wins1 += np.sum(act1 == winning_action)
            wins2 += np.sum(act2 == winning_action)

            minmaj1_records.append(-abs(A1))
            minmaj2_records.append(-abs(A2))

        scores1 += strategies1[:, :, h1] * winning_action

        bit = 1 if winning_action == 1 else 0
        history = ((history << 1) | bit) % max_P

    return (
        wins1 / (N1 * eval_steps),
        wins2 / (N2 * eval_steps),
        np.mean(minmaj1_records),
        np.mean(minmaj2_records)
    )


def run_mixed_m_game_large_m_random(N, M1=2, M2=10, S=2,
                                    total_steps=2000, eq_steps=1000,
                                    seed=None):
    if eq_steps >= total_steps:
        raise ValueError("eq_steps must be smaller than total_steps.")

    rng = np.random.default_rng(seed)

    if rng.random() < 0.5:
        N1 = N // 2
        N2 = N - N1
    else:
        N1 = N // 2 + 1
        N2 = N - N1

    P1, P2 = 2 ** M1, 2 ** M2
    max_P = max(P1, P2)

    strategies1 = rng.choice([-1, 1], size=(N1, S, P1))
    scores1 = np.zeros((N1, S))

    history = rng.integers(0, max_P)

    wins1 = 0
    wins2 = 0
    eval_steps = total_steps - eq_steps

    minmaj1_records, minmaj2_records = [], []

    for t in range(total_steps):
        h1 = history % P1

        best_s1 = random_argmax(scores1, rng)
        act1 = strategies1[np.arange(N1), best_s1, h1]
        act2 = rng.choice([-1, 1], size=N2)

        A1 = np.sum(act1)
        A2 = np.sum(act2)
        A = A1 + A2

        sign_A = 1 if A > 0 else (-1 if A < 0 else 0)

        if sign_A == 0:
            winning_action = rng.choice([-1, 1])
        else:
            winning_action = -sign_A

        if t >= eq_steps:
            wins1 += np.sum(act1 == winning_action)
            wins2 += np.sum(act2 == winning_action)

            minmaj1_records.append(-abs(A1))
            minmaj2_records.append(-abs(A2))

        scores1 += strategies1[:, :, h1] * winning_action

        bit = 1 if winning_action == 1 else 0
        history = ((history << 1) | bit) % max_P

    return (
        wins1 / (N1 * eval_steps),
        wins2 / (N2 * eval_steps),
        np.mean(minmaj1_records),
        np.mean(minmaj2_records)
    )


def random_normalized_imbalance_exact(Ng):
    from math import comb

    Ng = int(Ng)

    if Ng <= 0:
        return np.nan

    expected = 0.0
    denom = 2 ** Ng

    for k in range(Ng + 1):
        A = 2 * k - Ng
        prob = comb(Ng, k) / denom
        expected += (abs(A) / Ng) * prob

    return expected


def run_mixed_m_game_diagnostics(N, M1=2, M2=10, S=2,
                                 total_steps=2000, eq_steps=0,
                                 seed=None):
    if eq_steps >= total_steps:
        raise ValueError("eq_steps must be smaller than total_steps.")

    rng = np.random.default_rng(seed)

    if rng.random() < 0.5:
        N1 = N // 2
        N2 = N - N1
    else:
        N1 = N // 2 + 1
        N2 = N - N1

    P1, P2 = 2 ** M1, 2 ** M2
    max_P = max(P1, P2)

    strategies1 = rng.choice([-1, 1], size=(N1, S, P1))
    strategies2 = rng.choice([-1, 1], size=(N2, S, P2))

    scores1 = np.zeros((N1, S))
    scores2 = np.zeros((N2, S))

    history = rng.integers(0, max_P)

    history_counts1 = np.zeros(P1, dtype=int)
    history_counts2 = np.zeros(P2, dtype=int)

    imbalance1_records, imbalance2_records = [], []

    for t in range(total_steps):
        h1 = history % P1
        h2 = history % P2

        history_counts1[h1] += 1
        history_counts2[h2] += 1

        best_s1 = random_argmax(scores1, rng)
        best_s2 = random_argmax(scores2, rng)

        act1 = strategies1[np.arange(N1), best_s1, h1]
        act2 = strategies2[np.arange(N2), best_s2, h2]

        A1 = np.sum(act1)
        A2 = np.sum(act2)
        A = A1 + A2

        sign_A = 1 if A > 0 else (-1 if A < 0 else 0)

        if sign_A == 0:
            winning_action = rng.choice([-1, 1])
        else:
            winning_action = -sign_A

        if t >= eq_steps:
            imbalance1_records.append(abs(A1) / N1)
            imbalance2_records.append(abs(A2) / N2)

        scores1 += strategies1[:, :, h1] * winning_action
        scores2 += strategies2[:, :, h2] * winning_action

        bit = 1 if winning_action == 1 else 0
        history = ((history << 1) | bit) % max_P

    imbalance1_mean = np.mean(imbalance1_records)
    imbalance2_mean = np.mean(imbalance2_records)

    random_benchmark_highM = random_normalized_imbalance_exact(N2)
    random_benchmark1 = random_normalized_imbalance_exact(N1)
    random_benchmark2 = random_normalized_imbalance_exact(N2)

    return {
        "N": N,
        "N1": N1,
        "N2": N2,
        "P1": P1,
        "P2": P2,
        "history_counts1": history_counts1,
        "history_counts2": history_counts2,
        "imbalance1_mean": imbalance1_mean,
        "imbalance2_mean": imbalance2_mean,
        "random_benchmark_highM": random_benchmark_highM,
        "random_benchmark1": random_benchmark1,
        "random_benchmark2": random_benchmark2,
        "excess_imbalance1": imbalance1_mean - random_benchmark_highM,
        "excess_imbalance2": imbalance2_mean - random_benchmark_highM,
    }


# ==================================================
# Plotting functions
# ==================================================

def apply_axis_style(ax, xlabel=None, ylabel=None, title=None, legend=True):
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=labelsize)

    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=labelsize)

    if title is not None:
        ax.set_title(title, fontsize=titlesize, fontweight="bold")

    ax.tick_params(axis="both", labelsize=ticksize)

    if legend:
        ax.legend(fontsize=legendsize)


def plot_mode1_typical_run_with_zoom(N=101, M=6, S=2,
                                     lambda_val=-1.0, gamma=0.0,
                                     total_steps=2000, eq_steps=0,
                                     seed=1,
                                     zoom_start=None, zoom_width=150,
                                     normalize=True,
                                     save_filename="Mode1_Typical_Run_With_Zoom.png"):
    vol, succ, A_all_records, A_records = run_binary_choice_game(
        N=N,
        M=M,
        S=S,
        lambda_val=lambda_val,
        gamma=gamma,
        total_steps=total_steps,
        eq_steps=eq_steps,
        return_records=True,
        seed=seed
    )

    t = np.arange(total_steps)

    if normalize:
        y = A_all_records / np.sqrt(N)
        ylabel = r"$A_t/\sqrt{N}$"
        main_ylabel = r"Normalized aggregate action, $A_t/\sqrt{N}$"
    else:
        y = A_all_records
        ylabel = r"$A_t$"
        main_ylabel = r"Aggregate action, $A_t$"

    if zoom_start is None:
        zoom_start = total_steps - zoom_width

    zoom_start = max(0, int(zoom_start))
    zoom_width = int(zoom_width)
    zoom_end = min(zoom_start + zoom_width - 1, total_steps - 1)

    mask = (t >= zoom_start) & (t <= zoom_end)

    fig, (ax_full, ax_zoom) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(12, 7.5),
        gridspec_kw={"height_ratios": [2.3, 1.0], "hspace": 0.30}
    )

    ax_full.plot(t, y, linewidth=0.8, alpha=0.85)
    ax_full.axhline(y=0, linestyle="--", linewidth=1.0, alpha=0.7)
    ax_full.axvspan(zoom_start, zoom_end, alpha=0.18)

    if eq_steps > 0:
        ax_full.axvline(x=eq_steps, linestyle=":", linewidth=1.2, alpha=0.8)

    ax_full.set_xlim(0, total_steps - 1)

    apply_axis_style(
        ax_full,
        ylabel=main_ylabel,
        title=f"Typical Run: N={N}, M={M}, S={S}",
        legend=False
    )

    ax_full.grid(True, linestyle=":", alpha=0.6)

    ax_full.text(
        0.99,
        0.92,
        f"Shaded window: t={zoom_start}-{zoom_end}",
        transform=ax_full.transAxes,
        ha="right",
        va="top",
        fontsize=labelsize - 4,
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="white",
            edgecolor="0.7",
            alpha=0.85
        )
    )

    ax_zoom.plot(t[mask], y[mask], linewidth=1.2, alpha=0.9)
    ax_zoom.axhline(y=0, linestyle="--", linewidth=0.9, alpha=0.7)
    ax_zoom.set_xlim(zoom_start, zoom_end)
    ax_zoom.set_ylim(-4, 4)

    apply_axis_style(
        ax_zoom,
        xlabel="Time step",
        ylabel=ylabel,
        title=f"Zoomed window: final {zoom_end - zoom_start + 1} steps",
        legend=False
    )

    ax_zoom.grid(True, linestyle=":", alpha=0.6)

    save_figure(fig, save_filename)

    plt.show()


def plot_two_group_errorbar(x_values, mean1, se1, mean2, se2,
                            label1, label2,
                            xlabel, ylabel, title,
                            save_filename,
                            add_zero_line=False,
                            ylim=None):
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.errorbar(
        x_values,
        mean1,
        yerr=se1,
        marker="s",
        markersize=ms,
        linestyle="-",
        linewidth=1.5,
        capsize=3,
        alpha=0.8,
        label=label1
    )

    ax.errorbar(
        x_values,
        mean2,
        yerr=se2,
        marker="D",
        markersize=ms,
        linestyle="-",
        linewidth=1.5,
        capsize=3,
        alpha=0.8,
        label=label2
    )

    if add_zero_line:
        ax.axhline(y=0, linestyle="--", alpha=0.7)

    if ylim is not None:
        ax.set_ylim(ylim)

    apply_axis_style(
        ax,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        legend=True
    )

    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()

    save_figure(fig, save_filename)

    plt.show()


def plot_panel_a_revisit_share(N=25, M1=2, M2=10, S=2,
                               total_steps=2000, eq_steps=0,
                               seed=1,
                               save_filename="Figure3_PanelA_History_Repetition.png"):
    result = run_mixed_m_game_diagnostics(
        N=N,
        M1=M1,
        M2=M2,
        S=S,
        total_steps=total_steps,
        eq_steps=eq_steps,
        seed=seed
    )

    counts1 = result["history_counts1"]
    counts2 = result["history_counts2"]

    k_values = np.array([1, 2, 5, 10, 20, 50, 100])

    share1 = [np.mean(counts1 >= k) for k in k_values]
    share2 = [np.mean(counts2 >= k) for k in k_values]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(
        k_values,
        share1,
        marker="s",
        markersize=ms,
        linestyle="-",
        linewidth=1.8,
        label=f"M={M1}"
    )

    ax.plot(
        k_values,
        share2,
        marker="D",
        markersize=ms,
        linestyle="-",
        linewidth=1.8,
        label=f"M={M2}"
    )

    ax.set_xscale("log")
    ax.set_ylim(-0.05, 1.05)

    apply_axis_style(
        ax,
        xlabel="Minimum number of visits to a history",
        ylabel="Share of histories visited at least k times",
        title=f"Panel A: Repetition of information states, N={N}",
        legend=True
    )

    ax.grid(True, linestyle=":", alpha=0.6, which="both")

    plt.tight_layout()

    save_figure(fig, save_filename)

    plt.show()


def plot_panel_b_imbalance_with_random_benchmark(N_values, M1=2, M2=10, S=2,
                                                 total_steps=2000, eq_steps=0,
                                                 trials=100, master_seed=1,
                                                 save_filename="Figure3_PanelB_Imbalance_With_Random_Benchmark.png"):
    ss = np.random.SeedSequence(master_seed)
    seeds = ss.spawn(len(N_values) * trials)

    seed_idx = 0

    mean_imb1, se_imb1 = [], []
    mean_imb2, se_imb2 = [], []
    mean_random, se_random = [], []

    for N in N_values:
        imb1_trials, imb2_trials, random_trials = [], [], []

        for _ in range(trials):
            result = run_mixed_m_game_diagnostics(
                N=N,
                M1=M1,
                M2=M2,
                S=S,
                total_steps=total_steps,
                eq_steps=eq_steps,
                seed=seeds[seed_idx]
            )

            seed_idx += 1

            imb1_trials.append(result["imbalance1_mean"])
            imb2_trials.append(result["imbalance2_mean"])
            random_trials.append(result["random_benchmark_highM"])

        m1 = np.mean(imb1_trials)
        e1 = np.std(imb1_trials, ddof=1) / np.sqrt(trials)

        m2 = np.mean(imb2_trials)
        e2 = np.std(imb2_trials, ddof=1) / np.sqrt(trials)

        mr = np.mean(random_trials)
        er = np.std(random_trials, ddof=1) / np.sqrt(trials)

        mean_imb1.append(m1)
        se_imb1.append(e1)

        mean_imb2.append(m2)
        se_imb2.append(e2)

        mean_random.append(mr)
        se_random.append(er)

        print(
            f"N={N:3d} | "
            f"M={M1}: |A_g|/N_g = {m1:.3f} +/- {e1:.3f} | "
            f"M={M2}: |A_g|/N_g = {m2:.3f} +/- {e2:.3f} | "
            f"Random benchmark = {mr:.3f}"
        )

    mean_imb1 = np.array(mean_imb1)
    se_imb1 = np.array(se_imb1)

    mean_imb2 = np.array(mean_imb2)
    se_imb2 = np.array(se_imb2)

    mean_random = np.array(mean_random)
    se_random = np.array(se_random)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.errorbar(
        N_values,
        mean_imb1,
        yerr=se_imb1,
        marker="s",
        markersize=ms,
        linestyle="-",
        linewidth=1.6,
        capsize=3,
        alpha=0.85,
        label=f"M={M1}, adaptive"
    )

    ax.errorbar(
        N_values,
        mean_imb2,
        yerr=se_imb2,
        marker="D",
        markersize=ms,
        linestyle="-",
        linewidth=1.6,
        capsize=3,
        alpha=0.85,
        label=f"M={M2}, adaptive"
    )

    ax.plot(
        N_values,
        mean_random,
        linestyle="--",
        linewidth=2.0,
        alpha=0.9,
        label="Random-choice benchmark, same size as high-M group"
    )

    ax.fill_between(
        N_values,
        mean_random - se_random,
        mean_random + se_random,
        alpha=0.15
    )

    apply_axis_style(
        ax,
        xlabel="Total number of agents (N)",
        ylabel=r"Normalized within-group imbalance, $|A_g|/N_g$",
        title="Panel B: Imbalance relative to random behavior",
        legend=True
    )

    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()

    save_figure(fig, save_filename)

    plt.show()


def plot_panel_c_excess_imbalance(N_values, M1=2, M2=10, S=2,
                                  total_steps=2000, eq_steps=0,
                                  trials=100, master_seed=1,
                                  save_filename="Figure3_PanelC_Excess_Imbalance.png"):
    ss = np.random.SeedSequence(master_seed)
    seeds = ss.spawn(len(N_values) * trials)

    seed_idx = 0

    mean_excess1, se_excess1 = [], []
    mean_excess2, se_excess2 = [], []

    for N in N_values:
        excess1_trials, excess2_trials = [], []

        for _ in range(trials):
            result = run_mixed_m_game_diagnostics(
                N=N,
                M1=M1,
                M2=M2,
                S=S,
                total_steps=total_steps,
                eq_steps=eq_steps,
                seed=seeds[seed_idx]
            )

            seed_idx += 1

            excess1_trials.append(result["excess_imbalance1"])
            excess2_trials.append(result["excess_imbalance2"])

        m1 = np.mean(excess1_trials)
        e1 = np.std(excess1_trials, ddof=1) / np.sqrt(trials)

        m2 = np.mean(excess2_trials)
        e2 = np.std(excess2_trials, ddof=1) / np.sqrt(trials)

        mean_excess1.append(m1)
        se_excess1.append(e1)

        mean_excess2.append(m2)
        se_excess2.append(e2)

        print(
            f"N={N:3d} | "
            f"M={M1}: excess imbalance = {m1:.3f} +/- {e1:.3f} | "
            f"M={M2}: excess imbalance = {m2:.3f} +/- {e2:.3f}"
        )

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.axhline(
        y=0,
        linestyle="--",
        linewidth=1.5,
        alpha=0.8,
        label="Random-choice benchmark, same size as high-M group"
    )

    ax.errorbar(
        N_values,
        mean_excess1,
        yerr=se_excess1,
        marker="s",
        markersize=ms,
        linestyle="-",
        linewidth=1.6,
        capsize=3,
        alpha=0.85,
        label=f"M={M1}, adaptive"
    )

    ax.errorbar(
        N_values,
        mean_excess2,
        yerr=se_excess2,
        marker="D",
        markersize=ms,
        linestyle="-",
        linewidth=1.6,
        capsize=3,
        alpha=0.85,
        label=f"M={M2}, adaptive"
    )

    apply_axis_style(
        ax,
        xlabel="Total number of agents (N)",
        ylabel="Excess imbalance over random benchmark",
        title="Panel C: Excess crowding relative to random behavior",
        legend=True
    )

    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()

    save_figure(fig, save_filename)

    plt.show()


# ==================================================
# Main execution
# ==================================================

if __name__ == "__main__":
    EXPERIMENT_MODE = 5

    M1, M2 = 2, 8

    TOTAL_STEPS = 2000
    EQ_STEPS = 0
    TRIALS = 100
    MASTER_SEED = 1

    if EXPERIMENT_MODE == 1:
        N_values = np.arange(5, 406, 10)
    else:
        N_values = np.arange(5, 106, 10)

    ss = np.random.SeedSequence(MASTER_SEED)

    if EXPERIMENT_MODE == 1:
        print("==================================================")
        print(" [Mode 1] Baseline Minority Game")
        print("==================================================")

        M_FIXED = 6
        S_FIXED = 2

        lambda_val = -1.0
        gamma = 0.0

        alpha_values = (2 ** M_FIXED) / N_values

        volatility_means, volatility_ses = [], []
        success_means, success_ses = [], []

        seeds = ss.spawn(len(N_values) * TRIALS)
        seed_idx = 0

        # Switches for Mode 1 figures
        RUN_MAIN_MODE1_FIGURES = True
        RUN_TYPICAL_RUN_FIGURE = False

        # Typical-run settings
        TYPICAL_RUN_N = 185
        TYPICAL_RUN_ZOOM_START = None
        TYPICAL_RUN_ZOOM_WIDTH = 150
        TYPICAL_RUN_NORMALIZE = True

        if RUN_MAIN_MODE1_FIGURES:
            for N in N_values:
                vol_trials, succ_trials = [], []

                for _ in range(TRIALS):
                    vol, succ = run_binary_choice_game(
                        N=N,
                        M=M_FIXED,
                        S=S_FIXED,
                        lambda_val=lambda_val,
                        gamma=gamma,
                        total_steps=TOTAL_STEPS,
                        eq_steps=EQ_STEPS,
                        seed=seeds[seed_idx]
                    )

                    seed_idx += 1

                    vol_trials.append(vol)
                    succ_trials.append(succ)

                m_vol = np.mean(vol_trials)
                e_vol = np.std(vol_trials, ddof=1) / np.sqrt(TRIALS)

                m_succ = np.mean(succ_trials)
                e_succ = np.std(succ_trials, ddof=1) / np.sqrt(TRIALS)

                volatility_means.append(m_vol)
                volatility_ses.append(e_vol)

                success_means.append(m_succ)
                success_ses.append(e_succ)

                print(
                    f"N={N:3d} | "
                    f"alpha={((2 ** M_FIXED) / N):.4f} | "
                    f"Vol={m_vol:.4f} +/- {e_vol:.4f} | "
                    f"Success={m_succ:.4f} +/- {e_succ:.4f}"
                )

            order = np.argsort(alpha_values)

            alpha_sorted = alpha_values[order]

            vol_sorted = np.array(volatility_means)[order]
            vol_se_sorted = np.array(volatility_ses)[order]

            succ_sorted = np.array(success_means)[order]
            succ_se_sorted = np.array(success_ses)[order]

            # Figure: Volatility
            fig, ax = plt.subplots(figsize=(10, 6))

            ax.errorbar(
                alpha_sorted,
                vol_sorted,
                yerr=vol_se_sorted,
                marker="o",
                markersize=ms,
                linestyle="-",
                linewidth=1.5,
                capsize=3,
                alpha=0.8,
                label=f"M={M_FIXED}, S={S_FIXED}"
            )

            ax.set_xscale("log")

            apply_axis_style(
                ax,
                xlabel=r"$2^M/N$",
                ylabel=r"Volatility (= $\sigma/\sqrt{N}$)",
                title="Mode 1: Baseline Volatility vs $2^M/N$",
                legend=True
            )

            ax.grid(True, linestyle=":", alpha=0.6, which="both")

            plt.tight_layout()

            save_filename = "Mode1_Baseline_Volatility_Errorbar.png"
            save_figure(fig, save_filename)

            # Figure: Average Success Rate
            fig, ax = plt.subplots(figsize=(10, 6))

            ax.errorbar(
                alpha_sorted,
                succ_sorted,
                yerr=succ_se_sorted,
                marker="o",
                markersize=ms,
                linestyle="-",
                linewidth=1.5,
                capsize=3,
                alpha=0.8,
                label=f"M={M_FIXED}, S={S_FIXED}"
            )

            ax.axhline(
                y=0.5,
                linestyle="--",
                alpha=0.7,
                label="Upper bound near 0.5"
            )

            ax.set_xscale("log")

            apply_axis_style(
                ax,
                xlabel=r"$2^M/N$",
                ylabel="Average Success Rate",
                title="Mode 1: Baseline Success Rate vs $2^M/N$",
                legend=True
            )

            ax.grid(True, linestyle=":", alpha=0.6, which="both")

            plt.tight_layout()

            save_filename = "Mode1_Baseline_Success_Errorbar.png"
            save_figure(fig, save_filename)

            plt.show()

        if RUN_TYPICAL_RUN_FIGURE:
            plot_mode1_typical_run_with_zoom(
                N=TYPICAL_RUN_N,
                M=M_FIXED,
                S=S_FIXED,
                lambda_val=lambda_val,
                gamma=gamma,
                total_steps=TOTAL_STEPS,
                eq_steps=EQ_STEPS,
                seed=MASTER_SEED,
                zoom_start=TYPICAL_RUN_ZOOM_START,
                zoom_width=TYPICAL_RUN_ZOOM_WIDTH,
                normalize=TYPICAL_RUN_NORMALIZE,
                save_filename=(
                    f"Mode1_Typical_Run_With_Zoom_Normalized_{TYPICAL_RUN_N}.png"
                    if TYPICAL_RUN_NORMALIZE
                    else "Mode1_Typical_Run_With_Zoom_Raw.png"
                )
            )

    elif EXPERIMENT_MODE == 2:
        print("==================================================")
        print(" [Mode 2] Mixed-M Minority Game: Both Adaptive")
        print(" DV: Average Success Rate")
        print("==================================================")

        M_GROUP1, M_GROUP2, S_FIXED = M1, M2, 2

        mean_success1, se_success1 = [], []
        mean_success2, se_success2 = [], []

        seeds = ss.spawn(len(N_values) * TRIALS)
        seed_idx = 0

        for N in N_values:
            success1_trials, success2_trials = [], []

            for _ in range(TRIALS):
                success1, success2, minmaj1, minmaj2 = run_mixed_m_game(
                    N=N,
                    M1=M_GROUP1,
                    M2=M_GROUP2,
                    S=S_FIXED,
                    total_steps=TOTAL_STEPS,
                    eq_steps=EQ_STEPS,
                    seed=seeds[seed_idx]
                )

                seed_idx += 1

                success1_trials.append(success1)
                success2_trials.append(success2)

            m1 = np.mean(success1_trials)
            e1 = np.std(success1_trials, ddof=1) / np.sqrt(TRIALS)

            m2 = np.mean(success2_trials)
            e2 = np.std(success2_trials, ddof=1) / np.sqrt(TRIALS)

            mean_success1.append(m1)
            se_success1.append(e1)

            mean_success2.append(m2)
            se_success2.append(e2)

            print(
                f"N={N:3d} | "
                f"M={M_GROUP1}: Average Success Rate = {m1:.4f} +/- {e1:.4f} | "
                f"M={M_GROUP2}: Average Success Rate = {m2:.4f} +/- {e2:.4f}"
            )

        plot_two_group_errorbar(
            N_values,
            mean_success1,
            se_success1,
            mean_success2,
            se_success2,
            f"M={M_GROUP1}, low capacity",
            f"M={M_GROUP2}, high capacity",
            "Total number of agents (N)",
            "Average Success Rate",
            "Figure 2: Average Success Rate by Cognitive Capacity",
            f"Figure2_Average_Success_Rate_by_Cognitive_Capacity ({M_GROUP1}, {M_GROUP2}).png",
            add_zero_line=False,
            ylim=(0, 0.6)
        )

    elif EXPERIMENT_MODE == 3:
        print("==================================================")
        print(" [Mode 3] Mixed-M Minority Game: M1 Adaptive vs M2 Balanced Non-Adaptive")
        print("==================================================")

        M_GROUP1, M_GROUP2, S_FIXED = M1, M2, 2

        mean_minmaj1, se_minmaj1 = [], []
        mean_minmaj2, se_minmaj2 = [], []

        seeds = ss.spawn(len(N_values) * TRIALS)
        seed_idx = 0

        for N in N_values:
            minmaj1_trials, minmaj2_trials = [], []

            for _ in range(TRIALS):
                success1, success2, minmaj1, minmaj2 = run_mixed_m_game_large_m_balanced(
                    N=N,
                    M1=M_GROUP1,
                    M2=M_GROUP2,
                    S=S_FIXED,
                    total_steps=TOTAL_STEPS,
                    eq_steps=EQ_STEPS,
                    seed=seeds[seed_idx]
                )

                seed_idx += 1

                minmaj1_trials.append(minmaj1)
                minmaj2_trials.append(minmaj2)

            m1 = np.mean(minmaj1_trials)
            e1 = np.std(minmaj1_trials, ddof=1) / np.sqrt(TRIALS)

            m2 = np.mean(minmaj2_trials)
            e2 = np.std(minmaj2_trials, ddof=1) / np.sqrt(TRIALS)

            mean_minmaj1.append(m1)
            se_minmaj1.append(e1)

            mean_minmaj2.append(m2)
            se_minmaj2.append(e2)

            print(
                f"N={N:3d} | "
                f"M={M_GROUP1}: Minority - Majority = {m1:.2f} +/- {e1:.2f} | "
                f"M={M_GROUP2}: Minority - Majority = {m2:.2f} +/- {e2:.2f}"
            )

        plot_two_group_errorbar(
            N_values,
            mean_minmaj1,
            se_minmaj1,
            mean_minmaj2,
            se_minmaj2,
            f"M={M_GROUP1}, adaptive",
            f"M={M_GROUP2}, balanced non-adaptive",
            "Total number of agents (N)",
            "Minority - Majority within group",
            "Mode 3: Minority - Majority by M",
            "Mode3_Minority_Minus_Majority_by_M_Errorbar.png",
            add_zero_line=True,
            ylim=None
        )

    elif EXPERIMENT_MODE == 4:
        print("==================================================")
        print(" [Mode 4] Mixed-M Minority Game: M1 Adaptive vs M2 Random Non-Adaptive")
        print("==================================================")

        M_GROUP1, M_GROUP2, S_FIXED = M1, M2, 2

        mean_minmaj1, se_minmaj1 = [], []
        mean_minmaj2, se_minmaj2 = [], []

        seeds = ss.spawn(len(N_values) * TRIALS)
        seed_idx = 0

        for N in N_values:
            minmaj1_trials, minmaj2_trials = [], []

            for _ in range(TRIALS):
                success1, success2, minmaj1, minmaj2 = run_mixed_m_game_large_m_random(
                    N=N,
                    M1=M_GROUP1,
                    M2=M_GROUP2,
                    S=S_FIXED,
                    total_steps=TOTAL_STEPS,
                    eq_steps=EQ_STEPS,
                    seed=seeds[seed_idx]
                )

                seed_idx += 1

                minmaj1_trials.append(minmaj1)
                minmaj2_trials.append(minmaj2)

            m1 = np.mean(minmaj1_trials)
            e1 = np.std(minmaj1_trials, ddof=1) / np.sqrt(TRIALS)

            m2 = np.mean(minmaj2_trials)
            e2 = np.std(minmaj2_trials, ddof=1) / np.sqrt(TRIALS)

            mean_minmaj1.append(m1)
            se_minmaj1.append(e1)

            mean_minmaj2.append(m2)
            se_minmaj2.append(e2)

            print(
                f"N={N:3d} | "
                f"M={M_GROUP1}: Minority - Majority = {m1:.2f} +/- {e1:.2f} | "
                f"M={M_GROUP2}: Minority - Majority = {m2:.2f} +/- {e2:.2f}"
            )

        plot_two_group_errorbar(
            N_values,
            mean_minmaj1,
            se_minmaj1,
            mean_minmaj2,
            se_minmaj2,
            f"M={M_GROUP1}, adaptive",
            f"M={M_GROUP2}, random non-adaptive",
            "Total number of agents (N)",
            "Minority - Majority within group",
            "Mode 4: Minority - Majority by M",
            "Mode4_Minority_Minus_Majority_by_M_Errorbar.png",
            add_zero_line=True,
            ylim=None
        )

    elif EXPERIMENT_MODE == 5:
        print("==================================================")
        print(" [Mode 5] Mechanism Figure Panels")
        print("==================================================")

        M_LOW, M_HIGH, S_FIXED = M1, M2, 2
        N_REPRESENTATIVE = 25

        plot_panel_a_revisit_share(
            N=N_REPRESENTATIVE,
            M1=M_LOW,
            M2=M_HIGH,
            S=S_FIXED,
            total_steps=TOTAL_STEPS,
            eq_steps=EQ_STEPS,
            seed=MASTER_SEED,
            save_filename=f"Figure3_PanelA_History_Repetition ({M_LOW}, {M_HIGH}).png"
        )

        plot_panel_b_imbalance_with_random_benchmark(
            N_values=N_values,
            M1=M_LOW,
            M2=M_HIGH,
            S=S_FIXED,
            total_steps=TOTAL_STEPS,
            eq_steps=EQ_STEPS,
            trials=TRIALS,
            master_seed=MASTER_SEED,
            save_filename=f"Figure3_PanelB_Imbalance_With_Random_Benchmark ({M_LOW}, {M_HIGH}).png"
        )

    else:
        raise ValueError("EXPERIMENT_MODE must be 1, 2, 3, 4, or 5.")

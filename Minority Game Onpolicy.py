import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.stats import ttest_1samp, ttest_rel
import random

# --- Game Configuration ---
NUM_AGENTS = 100
NUM_ROUNDS = 100
NUM_SIMULATION_RUNS = 50

# --- Control Variable for Stay-Home Option ---
ALLOW_STAY_HOME = True 
single_plot=1

# --- Irreversibility Configuration ---
# Agent must stay at a chosen bar for this many periods.
# Set to 1 for no irreversibility.
IRREVERSIBILITY_PERIODS = 5

# Set the learning policy: 'ON_POLICY' or 'OFF_POLICY'
# 'ON_POLICY': Only the strategy used to make a decision is updated.
# 'OFF_POLICY': All of an agent's strategies are evaluated and updated each round.
POLICY_TYPE = 'ON_POLICY' 

# Set the softmax temperature for strategy selection.
# temp = 0: Greedy selection (always choose the best).
# temp > 0: Probabilistic selection. Higher temp = more random.
SOFTMAX_TEMP = 0
# ===================================================================

bar_capacity= 50
PROBABILITY_WO_HISTORY = [bar_capacity/NUM_AGENTS,  1-(bar_capacity)/NUM_AGENTS]
# --- Bar Configuration for Multiple Bars ---
BARS_CONFIG = [
    {
        "name": "Bar", 
        "capacity": bar_capacity,
        "reward_success": 5, 
        "reward_fail": -5
    }
]

# --- Capital and Utility Structure ---
INITIAL_CAPITAL = 1000.0        # By setting capital to 1000, ignore 'exit' with a reasonable time span
                              # If capital is low, many will quickly exit the market.
STAY_HOME_REWARD = 0.0
STAY_HOME_CHOICE_NAME = "Stay Home"


# ===================================================================
# --- STRATEGY DEFINITIONS FOR MULTI-BAR ---
# ===================================================================

class Strategy:
    """Base class for a predictive strategy."""
    def __init__(self, name="Strategy"):
        self.name = name

    def predict(self, attendance_history, bars_config, choice_names):
        """Returns the name of the chosen action."""
        raise NotImplementedError

    def __str__(self):
        return self.name
    
    def _decide_from_forecasts(self, forecasts, bars_config, choice_names):
        """
        Takes a dictionary of forecasts and returns the best choice.
        """
        best_bar = None
        lowest_ratio = float('inf')

        for bar in bars_config:
            name = bar['name']
            capacity = bar['capacity']
            
            if name not in forecasts:
                continue

            prediction = forecasts[name]

            if prediction <= capacity:
                ratio = prediction / capacity
                if ratio < lowest_ratio:
                    lowest_ratio = ratio
                    best_bar = name
        
        if best_bar:
            return best_bar
        
        if STAY_HOME_CHOICE_NAME in choice_names:
            return STAY_HOME_CHOICE_NAME
        else:
            return min(forecasts, key=forecasts.get)

# --- Potential Strategies ---
# 1. Randomly select an option (does not make any prediction)
class RandomChoiceStrategy(Strategy):
    def __init__(self):
        super().__init__("Random Choice")
    def predict(self, attendance_history, bars_config, choice_names):
        return random.choices(choice_names, PROBABILITY_WO_HISTORY, k=1)[0]

# 2. Predicts attendance will be the same as last round for EACH bar.
class MirrorAttendanceStrategy(Strategy):
    
    def __init__(self):
        super().__init__("Predict: A(t) = A(t-1)")

    def predict(self, attendance_history, bars_config, choice_names):
        forecasts = {}
        if not attendance_history or len(next(iter(attendance_history.values()))) < 1:
            return random.choices(choice_names, PROBABILITY_WO_HISTORY, k=1)[0]
        
        for bar in bars_config:
            name = bar['name']
            forecasts[name] = attendance_history[name][-1]
            
        return self._decide_from_forecasts(forecasts, bars_config, choice_names)

# 3. Predicts attendance as the mean of the last K rounds for EACH bar.
class MeanAttendanceStrategy(Strategy):
    def __init__(self, lookback):
        super().__init__(f"Predict: Mean of Last {lookback}")
        self.lookback = lookback

    def predict(self, attendance_history, bars_config, choice_names):
        forecasts = {}
        if not attendance_history or len(next(iter(attendance_history.values()))) < self.lookback:
            return random.choices(choice_names, PROBABILITY_WO_HISTORY, k=1)[0]

        for bar in bars_config:
            name = bar['name']
            hist = attendance_history[name][-self.lookback:]
            forecasts[name] = np.mean(hist)
        
        return self._decide_from_forecasts(forecasts, bars_config, choice_names)

# 4. Predicts the current trend will continue for EACH bar.
class TrendFollowingStrategy(Strategy):
    def __init__(self):
        super().__init__("Predict: Trend Extrapolation")

    def predict(self, attendance_history, bars_config, choice_names):
        forecasts = {}
        if not attendance_history or len(next(iter(attendance_history.values()))) < 2:
            return random.choices(choice_names, PROBABILITY_WO_HISTORY, k=1)[0]
            
        for bar in bars_config:
            name = bar['name']
            last_A = attendance_history[name][-1]
            prev_A = attendance_history[name][-2]
            trend = last_A - prev_A
            forecast = last_A + trend
            forecasts[name] = max(0, min(NUM_AGENTS, forecast))
            
        return self._decide_from_forecasts(forecasts, bars_config, choice_names)

# 5. Chooses the bar that was most crowded relative to its capacity last round.
#    The logic is that this bar will be the most avoided by others this round (So I will choose this bar).
class ContrarianBehaviorStrategy(Strategy):
    def __init__(self):
        super().__init__("Contrarian: Go to Last Round's Most Crowded Bar")

    def predict(self, attendance_history, bars_config, choice_names):
        if not attendance_history or len(next(iter(attendance_history.values()))) < 1:
            return random.choices(choice_names, PROBABILITY_WO_HISTORY, k=1)[0]

        most_crowded_bar = None
        max_crowd_ratio = -1.0

        for bar in bars_config:
            name = bar['name']
            capacity = bar['capacity']
            last_attendance = attendance_history[name][-1]
            ratio = last_attendance / (capacity + 1e-6)

            if ratio > max_crowd_ratio:
                max_crowd_ratio = ratio
                most_crowded_bar = name
        
        return most_crowded_bar if most_crowded_bar else random.choice([b['name'] for b in bars_config])


# --- Strategy Factory: The set of strategies considered by the agents in the model ---
def strategy_factory(choice_names):
    strategies = [
        MirrorAttendanceStrategy(),
        MeanAttendanceStrategy(lookback=2),
        #MeanAttendanceStrategy(lookback=3),
        #MeanAttendanceStrategy(lookback=4),
        #MeanAttendanceStrategy(lookback=5),
        ContrarianBehaviorStrategy(),
        TrendFollowingStrategy(),
    ]
    return strategies

# ===================================================================
# --- Agent Class ---
# ===================================================================

class Agent:
    def __init__(self, agent_id, choice_names, all_possible_strategies):
        self.id = agent_id
        self.capital = INITIAL_CAPITAL
        self.is_active = True
        self.strategies = all_possible_strategies
        self.strategy_scores = {s.name: 0.0 for s in self.strategies}
        # --- NEW: Attributes for irreversibility ---
        self.commitment_remaining = 0
        self.committed_choice = None

    def choose_strategy(self, temp):
        """
        Selects a strategy to use for the current round using softmax.
        Returns the name of the chosen strategy.
        """
        if not self.is_active: return None

        if temp == 0 or len(self.strategy_scores) == 1:
            return max(self.strategy_scores, key=self.strategy_scores.get)

        strategy_names = list(self.strategy_scores.keys())
        scores = np.array([self.strategy_scores[name] for name in strategy_names])
        stable_scores = scores - np.max(scores)
        exp_scores = np.exp(stable_scores / temp)
        probabilities = exp_scores / np.sum(exp_scores)
        
        if np.isnan(probabilities).any():
            return random.choice(strategy_names)
            
        return random.choices(strategy_names, weights=probabilities, k=1)[0]

    def update_capital(self, capital_change):
        if self.is_active:
            self.capital += capital_change
            """
            If capital is zero or below, the agent becomes inactive
            """
            if self.capital <= 0:
                self.is_active = False
                self.capital = 0

    def update_strategy_scores(self, attendance_history, bars_config, choice_names, outcomes, used_strategy_name, policy_type):
        """
        Updates strategy scores based on outcome.
        Two updating rules: (1) on-policy and (2) off-policy.
        """
        if not self.is_active: return
        # --- NEW: If agent was committed, no strategy was used, so no scores are updated. ---
        if not used_strategy_name: return

        strategies_to_update = []
        if policy_type == 'ON_POLICY':
            used_strategy = next((s for s in self.strategies if s.name == used_strategy_name), None)
            if used_strategy:
                strategies_to_update.append(used_strategy)
        else:
            strategies_to_update = self.strategies

        for strategy in strategies_to_update:
            hypothetical_choice = strategy.predict(attendance_history, bars_config, choice_names)
            reward = STAY_HOME_REWARD if hypothetical_choice == STAY_HOME_CHOICE_NAME else outcomes.get(hypothetical_choice, 0)
            self.strategy_scores[strategy.name] += reward

# ===================================================================
# --- Simulation Class ---
# ===================================================================

class ElFarolMultiBarSimulation:
    def __init__(self, bars_config, allow_stay_home):
        self.bars_config = bars_config
        self.allow_stay_home = allow_stay_home
        self.bar_names = [bar['name'] for bar in self.bars_config]
        if self.allow_stay_home: self.choice_names = self.bar_names + [STAY_HOME_CHOICE_NAME]
        else: self.choice_names = self.bar_names
        self.all_strategies = strategy_factory(self.choice_names)
        self.agents = [Agent(i, self.choice_names, self.all_strategies) for i in range(NUM_AGENTS)]
        self.active_agents_history, self.stayer_history = [], []
        self.attendance_history = {name: [] for name in self.bar_names}
        self.total_attendance_history = []
        
    def run_simulation(self):
        for r in range(NUM_ROUNDS):
            active_agents = [agent for agent in self.agents if agent.is_active]
            if not active_agents:
                self._fill_remaining_history(r)
                break
            self.active_agents_history.append(len(active_agents))
            
            agent_choices = {}
            for agent in active_agents:
                # --- MODIFIED: Irreversibility Logic ---
                if agent.commitment_remaining > 0:
                    # Agent is committed, so their choice is predetermined.
                    choice = agent.committed_choice
                    chosen_strategy_name = None  # No strategy was actively used.
                    agent.commitment_remaining -= 1
                    # If commitment ends this round, reset for the next round.
                    if agent.commitment_remaining == 0:
                        agent.committed_choice = None
                else:
                    # Agent is free to choose.
                    chosen_strategy_name = agent.choose_strategy(temp=SOFTMAX_TEMP)
                    strategy_obj = next((s for s in agent.strategies if s.name == chosen_strategy_name), None)
                    choice = strategy_obj.predict(self.attendance_history, self.bars_config, self.choice_names) if strategy_obj else random.choice(self.choice_names)
                    
                    # If the agent chose a bar, set the commitment for future rounds.
                    if choice in self.bar_names and IRREVERSIBILITY_PERIODS > 1:
                        agent.commitment_remaining = IRREVERSIBILITY_PERIODS - 1
                        agent.committed_choice = choice
                # --- END MODIFICATION ---
                
                agent_choices[agent.id] = (choice, chosen_strategy_name)

            attendance = {name: 0 for name in self.bar_names}; stayers = 0
            for choice, _ in agent_choices.values():
                if choice in self.bar_names: attendance[choice] += 1
                elif choice == STAY_HOME_CHOICE_NAME: stayers += 1 
            
            outcomes = {}
            for bar in self.bars_config:
                is_crowded = attendance[bar['name']] > bar['capacity']
                outcomes[bar['name']] = bar['reward_fail'] if is_crowded else bar['reward_success']
            
            for agent in active_agents:
                choice, chosen_strategy_name = agent_choices[agent.id]
                capital_change = STAY_HOME_REWARD if choice == STAY_HOME_CHOICE_NAME else outcomes.get(choice, 0)
                agent.update_capital(capital_change)
                # The update_strategy_scores method now handles the case where chosen_strategy_name is None
                agent.update_strategy_scores(self.attendance_history, self.bars_config, self.choice_names, outcomes, used_strategy_name=chosen_strategy_name, policy_type=POLICY_TYPE)

            for name in self.bar_names: self.attendance_history[name].append(attendance[name])
            self.stayer_history.append(stayers)
            self.total_attendance_history.append(sum(attendance.values()))
            
        return (self.active_agents_history, self.attendance_history, self.stayer_history, self.total_attendance_history)

    def _fill_remaining_history(self, current_round):
        """
        Inactive agents neither go to a bar nor stay at home
        """
        num_rounds_to_pad = NUM_ROUNDS - current_round
        for _ in range(num_rounds_to_pad):
            self.active_agents_history.append(0)
            for name in self.bar_names: self.attendance_history[name].append(0)
            self.stayer_history.append(0)
            self.total_attendance_history.append(0)

# ===================================================================
# --- Runner and Analysis Classes ---
# ===================================================================

class EnsembleRunner:
    def __init__(self, bars_config, num_runs, allow_stay_home):
        self.bars_config = bars_config
        self.num_runs = num_runs
        self.allow_stay_home = allow_stay_home 
        self.bar_names = [bar['name'] for bar in self.bars_config]
        if self.allow_stay_home: self.choice_names = self.bar_names + [STAY_HOME_CHOICE_NAME]
        else: self.choice_names = self.bar_names
        self.all_active_agents_histories, self.all_stayer_histories, self.all_total_attendance_histories = [], [], []
        self.all_attendance_histories = {name: [] for name in self.bar_names}
        self.statistical_results, self.paired_test_results = {}, {}
        self.final_survivor_counts = []

    def run_all_simulations(self):
        print(f"Starting ensemble of {self.num_runs} simulations...")
        start_time = time.time()
        for i in range(self.num_runs):
            simulation = ElFarolMultiBarSimulation(self.bars_config, self.allow_stay_home)
            active_hist, attend_hist, stayer_hist, total_attend_hist = simulation.run_simulation()
            self.all_active_agents_histories.append(active_hist)
            self.final_survivor_counts.append(active_hist[-1] if active_hist else 0)
            for name in self.bar_names: self.all_attendance_histories[name].append(attend_hist[name])
            self.all_stayer_histories.append(stayer_hist)
            self.all_total_attendance_histories.append(total_attend_hist)
            if (i + 1) % 10 == 0 or self.num_runs == 1: print(f"  ... Run {i+1}/{self.num_runs} complete.")
        duration = time.time() - start_time
        print(f"Ensemble run finished in {duration:.2f} seconds.")
        self.calculate_ensemble_averages()

    def calculate_ensemble_averages(self):
        print("\nCalculating ensemble averages and standard deviations...")
        self.ensemble_avg_active_agents = np.mean(self.all_active_agents_histories, axis=0)
        self.ensemble_avg_stayers = np.mean(self.all_stayer_histories, axis=0)
        self.ensemble_avg_total_attendance = np.mean(self.all_total_attendance_histories, axis=0)
        self.ensemble_avg_attendance = {name: np.mean(self.all_attendance_histories[name], axis=0) for name in self.bar_names}
        
        self.ensemble_std_active_agents = np.std(self.all_active_agents_histories, axis=0)
        self.ensemble_std_stayers = np.std(self.all_stayer_histories, axis=0)
        self.ensemble_std_total_attendance = np.std(self.all_total_attendance_histories, axis=0)
        self.ensemble_std_attendance = {name: np.std(self.all_attendance_histories[name], axis=0) for name in self.bar_names}

    # --- NEW METHOD TO DISPLAY CANONICAL RUN ---
    def display_canonical_run_history(self):
        """
        Displays the round-by-round attendance history for a single simulation run.
        This method is only meaningful when NUM_SIMULATION_RUNS is 1.
        """
        if self.num_runs != 1:
            print("\nCanonical run history can only be displayed for a single simulation run (NUM_SIMULATION_RUNS=1).")
            return

        print("\n" + "="*70)
        print("--- Canonical Run History ---")
        print("="*70)

        # Since num_runs is 1, the history lists contain only one element.
        # We extract that single history list using index [0].
        bar_histories = {name: self.all_attendance_histories[name][0] for name in self.bar_names}
        stayer_history = self.all_stayer_histories[0]
        
        # Dynamically create header based on bar names
        header_parts = [f"{'Round':>5}"]
        for name in self.bar_names:
            # Truncate long names for cleaner display
            header_parts.append(f"{name[:18]:>18}")
        if self.allow_stay_home:
            header_parts.append(f"{'Stay Home':>12}")
        
        header = " | ".join(header_parts)
        print(header)
        print("-" * len(header))

        # Print data for each round
        num_rounds_completed = len(stayer_history)
        for i in range(num_rounds_completed):
            row_parts = [f"{i+1:>5}"]
            for name in self.bar_names:
                row_parts.append(f"{bar_histories[name][i]:>18}")
            if self.allow_stay_home:
                row_parts.append(f"{stayer_history[i]:>12}")
            
            print(" | ".join(row_parts))
        
        print("-" * len(header))
        
    def perform_statistical_tests(self):
        print("\nPerforming one-sample t-tests (Attendance vs. Capacity)...")
        for bar in self.bars_config:
            bar_name = bar['name']; bar_capacity = bar['capacity']
            all_attendance_data = [att for run_hist in self.all_attendance_histories[bar_name] for att in run_hist]
            if not all_attendance_data: continue
            t_statistic, p_value = ttest_1samp(a=all_attendance_data, popmean=bar_capacity)
            self.statistical_results[bar_name] = {"mean_attendance": np.mean(all_attendance_data), "capacity": bar_capacity, "t_statistic": t_statistic, "p_value": p_value}
        if len(self.bar_names) == 2:
            print("Performing paired t-test (Bar 1 vs. Bar 2 Attendance)...")
            bar1_name, bar2_name = self.bar_names[0], self.bar_names[1]
            bar1_data = [att for run_hist in self.all_attendance_histories[bar1_name] for att in run_hist]
            bar2_data = [att for run_hist in self.all_attendance_histories[bar2_name] for att in run_hist]
            if bar1_data and bar2_data and len(bar1_data) == len(bar2_data):
                t_statistic, p_value = ttest_rel(a=bar1_data, b=bar2_data)
                self.paired_test_results = {"bar1_name": bar1_name, "bar2_name": bar2_name, "bar1_mean": np.mean(bar1_data), "bar2_mean": np.mean(bar2_data), "t_statistic": t_statistic, "p_value": p_value}

    def display_statistical_results(self):
        if self.statistical_results:
            print("\n--- Statistical Test: Mean Attendance vs. Capacity (One-Sample T-Test) ---")
            header = f"{'Bar Name':<25} | {'Mean Attendance':>18} | {'Capacity':>10} | {'p-value':>12} | {'Result'}"
            print(header); print("-" * len(header))
            for bar_name, results in self.statistical_results.items():
                p_val_str = f"{results['p_value']:.4f}" if results['p_value'] >= 0.0001 else "< 0.0001"
                conclusion = "FAIL to reject H₀" if results['p_value'] >= 0.05 else "REJECT H₀ (Significant)"
                print(f"{bar_name:<25} | {results['mean_attendance']:>18.2f} | {results['capacity']:>10} | {p_val_str:>12} | {conclusion}")
            print("-" * len(header))
        if self.paired_test_results:
            res = self.paired_test_results
            print("\n--- Statistical Test: Bar Preference (Paired T-Test) ---")
            header = f"{'Comparison':<25} | {'Mean Attendance':>18} | {'p-value':>12} | {'Result'}"
            print(header); print("-" * len(header))
            print(f"{res['bar1_name']:<25} | {res['bar1_mean']:>18.2f} | {'' :>12} |")
            print(f"{res['bar2_name']:<25} | {res['bar2_mean']:>18.2f} |", end="")
            p_val_str = f"{res['p_value']:.4f}" if res['p_value'] >= 0.0001 else "< 0.0001"
            conclusion = "FAIL to reject H₀" if res['p_value'] >= 0.05 else "REJECT H₀ (Significant)"
            print(f" {p_val_str:>12} | {conclusion}")
            print("-" * len(header))

    def single_plot_results(self):
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 7), sharex=True)
        
        stay_home_status = "ENABLED" if self.allow_stay_home else "DISABLED"
        # Modified title for single run
        if self.num_runs == 1:
            title_suffix = f"(Single Run, Stay Home: {stay_home_status}, Irreversibility: {IRREVERSIBILITY_PERIODS})"
        else:
            title_suffix = f"(Mean over {self.num_runs} Runs, Stay Home: {stay_home_status}, Irreversibility: {IRREVERSIBILITY_PERIODS})"
        
        colors = plt.cm.viridis(np.linspace(0, 0.8, len(self.bar_names)))
        rounds = range(len(self.ensemble_avg_stayers))

        for i, name in enumerate(self.bar_names):
            mean_att = self.ensemble_avg_attendance[name]
            capacity = next(b['capacity'] for b in self.bars_config if b['name'] == name)
            label = f"Attendance at '{name}'" if self.num_runs == 1 else f"Mean Attendance at '{name}'"
            ax1.plot(rounds, mean_att, label=label, color=colors[i], lw=2)
            ax1.axhline(y=capacity, color=colors[i], linestyle='--', lw=1.5, label=f"Capacity of '{name}' ({capacity})")
        
        if self.allow_stay_home:
            mean_stayers = self.ensemble_avg_stayers
            label = 'Agents Staying Home' if self.num_runs == 1 else 'Mean Agents Staying Home'
            ax1.plot(rounds, mean_stayers, label=label, color='red', lw=2, linestyle=':')
        
        ax1.set_title(f'Population Distribution Choices {title_suffix}', fontsize=16)
        ax1.set_ylabel('Number of Agents', fontsize=12)
        ax1.set_xlabel('Round', fontsize=12)
        ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax1.grid(True)
        ax1.set_ylim(bottom=0)
        fig.tight_layout() 
        plt.show()

    def analyze_and_display_survivors(self):
        if not self.final_survivor_counts: print("No survivor data to analyze."); return
        mean_survivors, median_survivors, std_survivors = np.mean(self.final_survivor_counts), np.median(self.final_survivor_counts), np.std(self.final_survivor_counts)
        min_survivors, max_survivors = np.min(self.final_survivor_counts), np.max(self.final_survivor_counts)
        stay_home_status = "ENABLED" if self.allow_stay_home else "DISABLED"
        
        # Tailor output for single vs. multiple runs
        if self.num_runs == 1:
            print("\n" + "="*60 + f"\n--- Final Survivor Count (Stay Home: {stay_home_status}) ---\n" + f"  - Number of survivors:   {self.final_survivor_counts[0]}\n" + "="*60 + "\n")
        else:
            print("\n" + "="*60 + f"\n--- Analysis of Final Survivor Counts (Stay Home: {stay_home_status}) ---\n" + f"Across {self.num_runs} simulation runs:\n" + f"  - Mean number of survivors:   {mean_survivors:.2f}\n" + f"  - Median number of survivors: {median_survivors:.0f}\n" + f"  - Standard Deviation:         {std_survivors:.2f}\n" + f"  - Minimum survivors in a run: {min_survivors:.0f}\n" + f"  - Maximum survivors in a run: {max_survivors:.0f}\n" + "="*60 + "\n")
            plt.style.use('seaborn-v0_8-whitegrid')
            plt.figure(figsize=(12, 7))
            bins = np.arange(0, NUM_AGENTS + 2) - 0.5
            plt.hist(self.final_survivor_counts, bins=bins, edgecolor='black', alpha=0.75)
            plt.axvline(mean_survivors, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_survivors:.2f}')
            plt.title(f'Distribution of Final Survivor Counts (Stay Home: {stay_home_status}, {self.num_runs} runs)', fontsize=16)
            plt.xlabel('Number of Survivors at End of Simulation', fontsize=12); plt.ylabel('Frequency (Number of Runs)', fontsize=12)
            plt.xlim(-0.5, NUM_AGENTS + 0.5); plt.legend(); plt.grid(True); plt.show()

# --- Main Execution Block ---
if __name__ == '__main__':
    print("="*60)
    print(f"Configuration: Learning Policy = {POLICY_TYPE}, Softmax Temp = {SOFTMAX_TEMP}")
    print(f"Irreversibility Periods = {IRREVERSIBILITY_PERIODS}")
    print("="*60)
    
    runner = EnsembleRunner(bars_config=BARS_CONFIG, num_runs=NUM_SIMULATION_RUNS, allow_stay_home=ALLOW_STAY_HOME)
    runner.run_all_simulations()
    
    # --- NEWLY ADDED CALL TO DISPLAY THE HISTORY ---
    #runner.display_canonical_run_history()
    
    #runner.analyze_and_display_survivors()
    #runner.perform_statistical_tests()
    #runner.display_statistical_results()
    runner.single_plot_results()
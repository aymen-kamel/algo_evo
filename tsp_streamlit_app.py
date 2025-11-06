import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
from deepseek_python_20251106_2810db import TSPProblem, SchedulingProblem, GeneticAlgorithm, SimulatedAnnealing, TabuSearch

# Page configuration
st.set_page_config(page_title="Metaheuristic Algorithms Comparison", page_icon="🔬", layout="wide")

# Title
st.title("🔬 Metaheuristic Algorithms Comparison Tool")
st.markdown("Compare Genetic Algorithms, Simulated Annealing, and Tabu Search on TSP and Scheduling problems.")

# Sidebar for configuration
st.sidebar.header("⚙️ Configuration")

# Problem selection
problem_type = st.sidebar.selectbox(
    "Select Problem Type",
    ["TSP (Traveling Salesman)", "Scheduling"],
    help="Choose the optimization problem to solve"
)

# Problem parameters
if problem_type == "TSP (Traveling Salesman)":
    num_cities = st.sidebar.slider("Number of Cities", 5, 50, 15, 1)
    num_tasks = None
    num_machines = None
else:
    num_tasks = st.sidebar.slider("Number of Tasks", 5, 50, 20, 1)
    num_machines = st.sidebar.slider("Number of Machines", 2, 10, 3, 1)
    num_cities = None

# Algorithm selection
st.sidebar.header("🤖 Algorithms to Compare")
algorithms_to_run = st.sidebar.multiselect(
    "Select Algorithms",
    ["GA Roulette", "GA Rang", "Recuit Simulé", "Recherche Tabou"],
    default=["GA Roulette", "GA Rang", "Recuit Simulé", "Recherche Tabou"],
    help="Choose which algorithms to include in the comparison"
)

# GA parameters
st.sidebar.header("🧬 Genetic Algorithm Parameters")
population_size = st.sidebar.slider("Population Size", 10, 500, 100, 10)
mutation_rate = st.sidebar.slider("Mutation Rate", 0.01, 0.5, 0.1, 0.01)
crossover_rate = st.sidebar.slider("Crossover Rate", 0.1, 1.0, 0.8, 0.1)
max_generations = st.sidebar.slider("Max Generations", 10, 2000, 500, 10)

# SA parameters
st.sidebar.header("🔥 Simulated Annealing Parameters")
initial_temp = st.sidebar.slider("Initial Temperature", 100, 5000, 1000, 100)
cooling_rate = st.sidebar.slider("Cooling Rate", 0.90, 0.999, 0.99, 0.001)
min_temp = st.sidebar.slider("Minimum Temperature", 0.1, 10.0, 1.0, 0.1)
max_iterations_sa = st.sidebar.slider("Max Iterations (SA)", 100, 5000, 1000, 100, key="sa_iterations")

# Tabu Search parameters
st.sidebar.header("🚫 Tabu Search Parameters")
tabu_size = st.sidebar.slider("Tabu List Size", 5, 50, 10, 1, key="tabu_size")
max_iterations_tabu = st.sidebar.slider("Max Iterations (Tabu)", 100, 5000, 1000, 100, key="tabu_iterations")
neighborhood_size = st.sidebar.slider("Neighborhood Size", 10, 100, 20, 5)

# Run button
run_comparison = st.sidebar.button("🚀 Run Comparison", type="primary")

# Main content
if run_comparison:
    with st.spinner("Running algorithms comparison... This may take a few moments."):
        # Create problem instance
        if problem_type == "TSP (Traveling Salesman)":
            problem = TSPProblem(num_cities=num_cities)
            problem_name = "TSP"
        else:
            problem = SchedulingProblem(num_tasks=num_tasks, num_machines=num_machines)
            problem_name = "Scheduling"

        # Run selected algorithms
        results = {}
        progress_bar = st.progress(0)
        status_text = st.empty()

        total_algorithms = len(algorithms_to_run)
        for i, algo_name in enumerate(algorithms_to_run):
            status_text.text(f"Running {algo_name}...")
            start_time = time.time()

            if algo_name == "GA Roulette":
                algorithm = GeneticAlgorithm(problem, population_size=population_size,
                                           mutation_rate=mutation_rate, crossover_rate=crossover_rate,
                                           max_generations=max_generations, selection_type='roulette')
            elif algo_name == "GA Rang":
                algorithm = GeneticAlgorithm(problem, population_size=population_size,
                                           mutation_rate=mutation_rate, crossover_rate=crossover_rate,
                                           max_generations=max_generations, selection_type='rank')
            elif algo_name == "Recuit Simulé":
                algorithm = SimulatedAnnealing(problem, initial_temp=initial_temp,
                                             cooling_rate=cooling_rate, min_temp=min_temp,
                                             max_iterations=max_iterations_sa)
            elif algo_name == "Recherche Tabou":
                algorithm = TabuSearch(problem, tabu_size=tabu_size,
                                     max_iterations=max_iterations_tabu,
                                     neighborhood_size=neighborhood_size)

            solution, fitness = algorithm.run()
            execution_time = time.time() - start_time

            results[algo_name] = {
                'fitness': fitness,
                'time': execution_time,
                'history': algorithm.fitness_history,
                'solution': solution
            }

            progress_bar.progress((i + 1) / total_algorithms)

        progress_bar.empty()
        status_text.empty()

    # Display results
    st.header("📊 Results")

    # Summary table
    st.subheader("Performance Summary")
    summary_data = {
        "Algorithm": list(results.keys()),
        "Fitness": [results[algo]['fitness'] for algo in results],
        "Execution Time (s)": [results[algo]['time'] for algo in results]
    }

    if problem_name == "TSP":
        summary_data["Metric"] = ["Distance"] * len(results)
    else:
        summary_data["Metric"] = ["Makespan"] * len(results)

    st.table(summary_data)

    # Convergence plot
    st.subheader("Convergence Curves")
    fig, ax = plt.subplots(figsize=(10, 6))
    for algo_name in results:
        ax.plot(results[algo_name]['history'], label=algo_name, linewidth=2)
    ax.set_xlabel("Iterations")
    if problem_name == "TSP":
        ax.set_ylabel("Distance")
        ax.set_title("TSP Convergence")
    else:
        ax.set_ylabel("Makespan")
        ax.set_title("Scheduling Convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    # Performance comparison
    st.subheader("Performance Comparison")
    fig, ax = plt.subplots(figsize=(10, 6))
    algorithms = list(results.keys())
    fitnesses = [results[algo]['fitness'] for algo in results]
    bars = ax.bar(algorithms, fitnesses, color=['skyblue', 'lightgreen', 'salmon', 'gold'])
    ax.set_ylabel("Fitness" if problem_name == "TSP" else "Makespan")
    ax.set_title(f"{problem_name} Performance Comparison")
    plt.xticks(rotation=45)
    for bar, fitness in zip(bars, fitnesses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(fitnesses)*0.01,
                f'{fitness:.2f}', ha='center', va='bottom')
    st.pyplot(fig)

    # Best solution visualization (TSP only)
    if problem_name == "TSP":
        st.subheader("Best TSP Solution")
        best_algo = min(results, key=lambda x: results[x]['fitness'])
        best_solution = results[best_algo]['solution']

        fig, ax = plt.subplots(figsize=(10, 8))
        cities = problem.cities
        solution_path = best_solution + [best_solution[0]]  # Return to start

        # Plot path
        ax.plot(cities[solution_path, 0], cities[solution_path, 1], 'b-', alpha=0.7, linewidth=3)
        ax.scatter(cities[:, 0], cities[:, 1], c='red', s=150, zorder=5, edgecolors='black')

        # Annotate cities
        for i, city in enumerate(cities):
            ax.annotate(str(i), (city[0], city[1]), xytext=(8, 8), textcoords='offset points',
                       fontsize=12, weight='bold')

        ax.set_title(f"Best TSP Path - {best_algo}\nDistance: {results[best_algo]['fitness']:.2f}")
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    # Algorithm analysis
    st.header("📈 Analysis")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Best Algorithm")
        best_algo = min(results, key=lambda x: results[x]['fitness'])
        st.success(f"🏆 {best_algo} with fitness {results[best_algo]['fitness']:.2f}")

    with col2:
        st.subheader("Fastest Algorithm")
        fastest_algo = min(results, key=lambda x: results[x]['time'])
        st.info(f"⚡ {fastest_algo} in {results[fastest_algo]['time']:.3f} seconds")

    # Recommendations
    st.subheader("💡 Recommendations")
    if problem_name == "TSP":
        st.markdown("""
        - **GA Rang**: Often provides good balance between exploration and exploitation
        - **Recherche Tabou**: Usually finds the best solutions but may be slower
        - **Recuit Simulé**: Good for avoiding local optima, sensitive to temperature parameters
        - **GA Roulette**: Fast but may converge to suboptimal solutions
        """)
    else:
        st.markdown("""
        - **GA Rang**: Generally performs well on scheduling problems
        - **Recherche Tabou**: Effective for complex scheduling constraints
        - **Recuit Simulé**: Can be tuned for good performance
        - **GA Roulette**: Good starting point, may need parameter tuning
        """)

else:
    st.info("👈 Configure your parameters in the sidebar and click 'Run Comparison' to start!")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit | Metaheuristic Algorithms Comparison Tool")

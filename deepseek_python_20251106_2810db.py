import numpy as np
import matplotlib.pyplot as plt
import random
import time
from typing import List, Tuple, Dict, Any
import math

# ==================== PROBLÈME DU VOYAGEUR DE COMMERCE ====================

class TSPProblem:
    def __init__(self, num_cities=20, seed=42):
        self.num_cities = num_cities
        np.random.seed(seed)
        self.cities = np.random.rand(num_cities, 2) * 100
        self.distance_matrix = self._compute_distance_matrix()
    
    def _compute_distance_matrix(self):
        dist_matrix = np.zeros((self.num_cities, self.num_cities))
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                if i != j:
                    dist_matrix[i][j] = np.linalg.norm(self.cities[i] - self.cities[j])
        return dist_matrix
    
    def evaluate(self, solution):
        """Calcule la distance totale d'un chemin"""
        total_distance = 0
        for i in range(len(solution)):
            total_distance += self.distance_matrix[solution[i-1]][solution[i]]
        return total_distance

# ==================== PROBLÈME D'ORDONNANCEMENT ====================

class SchedulingProblem:
    def __init__(self, num_tasks=20, num_machines=3, seed=42):
        self.num_tasks = num_tasks
        self.num_machines = num_machines
        np.random.seed(seed)
        self.task_times = np.random.randint(1, 20, num_tasks)
    
    def evaluate(self, solution):
        """Calcule le makespan (temps total maximum)"""
        machine_times = [0] * self.num_machines
        for task_id, machine_id in enumerate(solution):
            machine_times[machine_id] += self.task_times[task_id]
        return max(machine_times)

# ==================== ALGORITHMES GÉNÉTIQUES ====================

class GeneticAlgorithm:
    def __init__(self, problem, population_size=100, mutation_rate=0.1, 
                 crossover_rate=0.8, max_generations=1000, selection_type='roulette'):
        self.problem = problem
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.max_generations = max_generations
        self.selection_type = selection_type
        self.best_solution = None
        self.best_fitness = float('inf')
        self.fitness_history = []
    
    def initialize_population(self):
        if hasattr(self.problem, 'num_cities'):  # TSP
            return [random.sample(range(self.problem.num_cities), 
                                 self.problem.num_cities) 
                   for _ in range(self.population_size)]
        else:  # Scheduling
            return [np.random.randint(0, self.problem.num_machines, 
                                    self.problem.num_tasks).tolist()
                   for _ in range(self.population_size)]
    
    def evaluate_population(self, population):
        return [self.problem.evaluate(ind) for ind in population]
    
    def selection_roulette(self, population, fitnesses):
        """Sélection par roulette (proportionnelle à la fitness)"""
        inverted_fitness = [1/f if f != 0 else 1e10 for f in fitnesses]
        total = sum(inverted_fitness)
        probabilities = [f/total for f in inverted_fitness]
        return population[np.random.choice(len(population), p=probabilities)]
    
    def selection_rank(self, population, fitnesses):
        """Sélection par rang"""
        sorted_indices = np.argsort(fitnesses)
        ranks = np.arange(1, len(population) + 1)
        probabilities = ranks / np.sum(ranks)
        return population[np.random.choice(sorted_indices, p=probabilities)]
    
    def crossover_tsp(self, parent1, parent2):
        """OX Crossover pour TSP"""
        size = len(parent1)
        child = [-1] * size
        start, end = sorted(random.sample(range(size), 2))
        
        # Copier le segment du parent1
        child[start:end] = parent1[start:end]
        
        # Remplir avec les éléments du parent2
        pointer = end
        for gene in parent2:
            if gene not in child:
                if pointer >= size:
                    pointer = 0
                child[pointer] = gene
                pointer += 1
        
        return child
    
    def crossover_scheduling(self, parent1, parent2):
        """Crossover à un point pour scheduling"""
        point = random.randint(1, len(parent1)-1)
        child = parent1[:point] + parent2[point:]
        return child
    
    def mutate_tsp(self, individual):
        """Mutation par échange pour TSP"""
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(len(individual)), 2)
            individual[i], individual[j] = individual[j], individual[i]
        return individual
    
    def mutate_scheduling(self, individual):
        """Mutation pour scheduling"""
        if random.random() < self.mutation_rate:
            task = random.randint(0, len(individual)-1)
            new_machine = random.randint(0, self.problem.num_machines-1)
            individual[task] = new_machine
        return individual
    
    def run(self):
        population = self.initialize_population()
        
        for generation in range(self.max_generations):
            fitnesses = self.evaluate_population(population)
            
            # Mettre à jour la meilleure solution
            current_best = min(fitnesses)
            if current_best < self.best_fitness:
                self.best_fitness = current_best
                self.best_solution = population[np.argmin(fitnesses)]
            
            self.fitness_history.append(self.best_fitness)
            
            # Créer une nouvelle population
            new_population = []
            
            while len(new_population) < self.population_size:
                # Sélection
                if self.selection_type == 'roulette':
                    parent1 = self.selection_roulette(population, fitnesses)
                    parent2 = self.selection_roulette(population, fitnesses)
                else:  # sélection par rang
                    parent1 = self.selection_rank(population, fitnesses)
                    parent2 = self.selection_rank(population, fitnesses)
                
                # Croisement
                if random.random() < self.crossover_rate:
                    if hasattr(self.problem, 'num_cities'):
                        child = self.crossover_tsp(parent1, parent2)
                    else:
                        child = self.crossover_scheduling(parent1, parent2)
                else:
                    child = parent1.copy()
                
                # Mutation
                if hasattr(self.problem, 'num_cities'):
                    child = self.mutate_tsp(child)
                else:
                    child = self.mutate_scheduling(child)
                
                new_population.append(child)
            
            population = new_population
        
        return self.best_solution, self.best_fitness

# ==================== RECUIT SIMULÉ ====================

class SimulatedAnnealing:
    def __init__(self, problem, initial_temp=1000, cooling_rate=0.99, 
                 min_temp=1, max_iterations=1000):
        self.problem = problem
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.max_iterations = max_iterations
        self.fitness_history = []
    
    def initialize_solution(self):
        if hasattr(self.problem, 'num_cities'):
            return random.sample(range(self.problem.num_cities), 
                                self.problem.num_cities)
        else:
            return np.random.randint(0, self.problem.num_machines, 
                                   self.problem.num_tasks).tolist()
    
    def get_neighbor(self, solution):
        neighbor = solution.copy()
        if hasattr(self.problem, 'num_cities'):
            i, j = random.sample(range(len(neighbor)), 2)
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        else:
            task = random.randint(0, len(neighbor)-1)
            new_machine = random.randint(0, self.problem.num_machines-1)
            neighbor[task] = new_machine
        return neighbor
    
    def run(self):
        current_solution = self.initialize_solution()
        current_fitness = self.problem.evaluate(current_solution)
        best_solution = current_solution.copy()
        best_fitness = current_fitness
        
        temperature = self.initial_temp
        
        for iteration in range(self.max_iterations):
            neighbor = self.get_neighbor(current_solution)
            neighbor_fitness = self.problem.evaluate(neighbor)
            
            delta = neighbor_fitness - current_fitness
            
            if delta < 0 or random.random() < math.exp(-delta / temperature):
                current_solution = neighbor
                current_fitness = neighbor_fitness
                
                if current_fitness < best_fitness:
                    best_solution = current_solution.copy()
                    best_fitness = current_fitness
            
            self.fitness_history.append(best_fitness)
            temperature *= self.cooling_rate
            
            if temperature < self.min_temp:
                break
        
        return best_solution, best_fitness

# ==================== RECHERCHE TABOU ====================

class TabuSearch:
    def __init__(self, problem, tabu_size=10, max_iterations=1000, 
                 neighborhood_size=20):
        self.problem = problem
        self.tabu_size = tabu_size
        self.max_iterations = max_iterations
        self.neighborhood_size = neighborhood_size
        self.fitness_history = []
    
    def initialize_solution(self):
        if hasattr(self.problem, 'num_cities'):
            return random.sample(range(self.problem.num_cities), 
                                self.problem.num_cities)
        else:
            return np.random.randint(0, self.problem.num_machines, 
                                   self.problem.num_tasks).tolist()
    
    def get_neighbors(self, solution):
        neighbors = []
        for _ in range(self.neighborhood_size):
            neighbor = solution.copy()
            if hasattr(self.problem, 'num_cities'):
                i, j = random.sample(range(len(neighbor)), 2)
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                move = tuple(sorted((i, j)))
            else:
                task = random.randint(0, len(neighbor)-1)
                old_machine = neighbor[task]
                new_machine = random.randint(0, self.problem.num_machines-1)
                while new_machine == old_machine:
                    new_machine = random.randint(0, self.problem.num_machines-1)
                neighbor[task] = new_machine
                move = (task, old_machine, new_machine)
            
            neighbors.append((neighbor, move))
        return neighbors
    
    def run(self):
        current_solution = self.initialize_solution()
        current_fitness = self.problem.evaluate(current_solution)
        best_solution = current_solution.copy()
        best_fitness = current_fitness
        
        tabu_list = []
        
        for iteration in range(self.max_iterations):
            neighbors = self.get_neighbors(current_solution)
            best_neighbor = None
            best_neighbor_fitness = float('inf')
            best_move = None
            
            for neighbor, move in neighbors:
                if move in tabu_list:
                    continue
                
                neighbor_fitness = self.problem.evaluate(neighbor)
                
                if neighbor_fitness < best_neighbor_fitness:
                    best_neighbor = neighbor
                    best_neighbor_fitness = neighbor_fitness
                    best_move = move
            
            if best_neighbor is None:
                break
            
            current_solution = best_neighbor
            current_fitness = best_neighbor_fitness
            
            # Mettre à jour la liste tabou
            tabu_list.append(best_move)
            if len(tabu_list) > self.tabu_size:
                tabu_list.pop(0)
            
            if current_fitness < best_fitness:
                best_solution = current_solution.copy()
                best_fitness = current_fitness
            
            self.fitness_history.append(best_fitness)
        
        return best_solution, best_fitness

# ==================== EXPÉRIMENTATION ET COMPARAISON ====================

def run_comparison():
    print("🔬 DÉMARRAGE DE LA COMPARAISON DES ALGORITHMES")
    print("=" * 60)
    
    # Créer les problèmes
    tsp_problem = TSPProblem(num_cities=15)
    scheduling_problem = SchedulingProblem(num_tasks=20, num_machines=3)
    
    algorithms = {
        'GA Roulette': lambda p: GeneticAlgorithm(p, selection_type='roulette'),
        'GA Rang': lambda p: GeneticAlgorithm(p, selection_type='rank'),
        'Recuit Simulé': lambda p: SimulatedAnnealing(p),
        'Recherche Tabou': lambda p: TabuSearch(p)
    }
    
    results = {}
    
    # Tester sur TSP
    print("\n🧭 PROBLÈME DU VOYAGEUR DE COMMERCE")
    print("-" * 40)
    
    for algo_name, algo_constructor in algorithms.items():
        start_time = time.time()
        algorithm = algo_constructor(tsp_problem)
        solution, fitness = algorithm.run()
        execution_time = time.time() - start_time
        
        results[f'TSP_{algo_name}'] = {
            'fitness': fitness,
            'time': execution_time,
            'history': algorithm.fitness_history,
            'solution': solution
        }
        
        print(f"{algo_name:15} | Distance: {fitness:8.2f} | Temps: {execution_time:6.3f}s")
    
    # Tester sur Scheduling
    print("\n⚙️ PROBLÈME D'ORDONNANCEMENT")
    print("-" * 30)
    
    for algo_name, algo_constructor in algorithms.items():
        start_time = time.time()
        algorithm = algo_constructor(scheduling_problem)
        solution, fitness = algorithm.run()
        execution_time = time.time() - start_time
        
        results[f'Scheduling_{algo_name}'] = {
            'fitness': fitness,
            'time': execution_time,
            'history': algorithm.fitness_history,
            'solution': solution
        }
        
        print(f"{algo_name:15} | Makespan: {fitness:6.1f} | Temps: {execution_time:6.3f}s")
    
    return results, tsp_problem, scheduling_problem

# ==================== VISUALISATION ====================

def plot_results(results, tsp_problem, scheduling_problem):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Convergence TSP
    ax = axes[0, 0]
    for algo_name in ['GA Roulette', 'GA Rang', 'Recuit Simulé', 'Recherche Tabou']:
        key = f'TSP_{algo_name}'
        if key in results:
            ax.plot(results[key]['history'], label=algo_name)
    ax.set_title('Convergence - Problème du Voyageur')
    ax.set_xlabel('Itérations')
    ax.set_ylabel('Distance')
    ax.legend()
    ax.grid(True)
    
    # Convergence Scheduling
    ax = axes[0, 1]
    for algo_name in ['GA Roulette', 'GA Rang', 'Recuit Simulé', 'Recherche Tabou']:
        key = f'Scheduling_{algo_name}'
        if key in results:
            ax.plot(results[key]['history'], label=algo_name)
    ax.set_title('Convergence - Problème d\'Ordonnancement')
    ax.set_xlabel('Itérations')
    ax.set_ylabel('Makespan')
    ax.legend()
    ax.grid(True)
    
    # Performance TSP
    ax = axes[1, 0]
    algo_names = []
    fitnesses = []
    for algo_name in ['GA Roulette', 'GA Rang', 'Recuit Simulé', 'Recherche Tabou']:
        key = f'TSP_{algo_name}'
        if key in results:
            algo_names.append(algo_name)
            fitnesses.append(results[key]['fitness'])
    bars = ax.bar(algo_names, fitnesses)
    ax.set_title('Performance - Problème du Voyageur')
    ax.set_ylabel('Distance')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # Performance Scheduling
    ax = axes[1, 1]
    algo_names = []
    fitnesses = []
    for algo_name in ['GA Roulette', 'GA Rang', 'Recuit Simulé', 'Recherche Tabou']:
        key = f'Scheduling_{algo_name}'
        if key in results:
            algo_names.append(algo_name)
            fitnesses.append(results[key]['fitness'])
    bars = ax.bar(algo_names, fitnesses)
    ax.set_title('Performance - Problème d\'Ordonnancement')
    ax.set_ylabel('Makespan')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig('comparison_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Visualisation de la meilleure solution TSP
    best_tsp_algo = min([(results[key]['fitness'], key) for key in results if key.startswith('TSP_')])[1]
    best_tsp_solution = results[best_tsp_algo]['solution']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    cities = tsp_problem.cities
    solution = best_tsp_solution + [best_tsp_solution[0]]  # Retour au départ
    
    # Tracer le chemin
    ax.plot(cities[solution, 0], cities[solution, 1], 'b-', alpha=0.6, linewidth=2)
    ax.scatter(cities[:, 0], cities[:, 1], c='red', s=100, zorder=5)
    
    # Annoter les villes
    for i, city in enumerate(cities):
        ax.annotate(str(i), (city[0], city[1]), xytext=(5, 5), textcoords='offset points')
    
    ax.set_title(f'Meilleur chemin TSP - {best_tsp_algo.replace("TSP_", "")}\n'
                f'Distance: {results[best_tsp_algo]["fitness"]:.2f}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True)
    plt.savefig('best_tsp_solution.png', dpi=300, bbox_inches='tight')
    plt.show()

# ==================== EXÉCUTION PRINCIPALE ====================

if __name__ == "__main__":
    print("🎯 COMPARAISON DES ALGORITHMES MÉTAHEURISTIQUES")
    print("Problèmes: Voyageur de Commerce & Ordonnancement")
    print("Algorithmes: GA Roulette, GA Rang, Recuit Simulé, Recherche Tabou")
    print("=" * 60)
    
    results, tsp_problem, scheduling_problem = run_comparison()
    plot_results(results, tsp_problem, scheduling_problem)
    
    print("\n📊 ANALYSE COMPARATIVE")
    print("=" * 40)
    print("\n🧭 POUR LE TSP:")
    print("- GA Roulette: Rapide mais peut stagner")
    print("- GA Rang: Plus stable, évite la dominance précoce") 
    print("- Recuit Simulé: Bon pour éviter les minima locaux")
    print("- Recherche Tabou: Excellente qualité de solution")
    
    print("\n⚙️ POUR L'ORDONNANCEMENT:")
    print("- GA Roulette: Efficace pour l'exploration")
    print("- GA Rang: Meilleure convergence")
    print("- Recuit Simulé: Paramètres sensibles")
    print("- Recherche Tabou: Très performant avec bonne liste tabou")
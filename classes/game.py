import random
import json
from .bird import Bird
from .pipe import Pipe
from .neural_network import NeuralNetwork

PIPE_SPAWN_INTERVAL = 1200  # milliseconds
POPULATION_SIZE = 100


class Game:
    def __init__(self, brain_filepath=None):
        self.birds = [Bird() for _ in range(POPULATION_SIZE)]
        self.pipes = [Pipe()]
        self.pipe_timer = 0
        self.generation = 1
        self.dead_birds = []
        self.best_bird = None
        self.speed_multiplier = 1.0
        if brain_filepath:
            self.load_bird_brain(brain_filepath)

    def update(self, dt):
        dt *= self.speed_multiplier

        alive_birds = []
        for bird in self.birds:
            bird.update(dt, self.pipes)
            if bird.alive:
                alive_birds.append(bird)
            else:
                self.dead_birds.append(bird)

        self.birds = alive_birds

        if len(self.birds) == 0:
            self.__genetic_evolution()
            self.generation += 1
            self.pipes = [Pipe()]
            self.pipe_timer = 0

        # Update all pipes and remove off-screen pipes
        for pipe in self.pipes:
            pipe.update(dt)

        self.pipes = [pipe for pipe in self.pipes if not pipe.is_off_screen()]

        # Timer for pipe generation
        self.pipe_timer += dt * 1000

        if self.pipe_timer >= PIPE_SPAWN_INTERVAL:
            self.pipes.append(Pipe())
            self.pipe_timer = 0

    def draw(self, screen):
        for bird in self.birds:
            bird.draw(screen)
        for pipe in self.pipes:
            pipe.draw(screen)

    def increase_speed(self):
        """Increase simulation speed"""
        self.speed_multiplier = min(5.0, self.speed_multiplier * 1.5)
        print(f"Speed multiplier: {self.speed_multiplier:.1f}x")

    def decrease_speed(self):
        """Decrease simulation speed"""
        self.speed_multiplier = max(0.2, self.speed_multiplier / 1.5)
        print(f"Speed multiplier: {self.speed_multiplier:.1f}x")

    def __genetic_evolution(self):
        dead_birds = sorted(
            self.dead_birds, key=lambda b: b.fitness(), reverse=True)[:20]

        num_parents = max(2, POPULATION_SIZE // 5)
        parents = dead_birds[:num_parents]

        # Print best bird stats for debugging
        if dead_birds:
            best = dead_birds[0]
            if not self.best_bird:
                self.best_bird = best
            else:
                if best.fitness() > self.best_bird.fitness():
                    self.best_bird = best
            print(f"Gen {self.generation}: Best bird - Score: {best.score}, "
                  f"Time: {best.survival_time:.1f}s, Fitness: {best.fitness()}")

        new_birds = []

        # Keep the best bird from previous generation (elitism)
        if dead_birds:
            elite_bird = Bird()
            elite_weights, elite_biases = dead_birds[0].brain.get_weights(
            )
            elite_bird.brain.set_weights(elite_weights, elite_biases)
            new_birds.append(elite_bird)

        for i in range(POPULATION_SIZE - 1):
            # Tournament selection: pick better of two random parents
            parent1 = self.__tournament_selection(parents)
            parent2 = self.__tournament_selection(parents)

            child = Bird()

            child_weights, child_biases = self.__crossover_and_mutate(
                parent1.brain.get_weights(), parent2.brain.get_weights())

            child.brain.set_weights(child_weights, child_biases)
            new_birds.append(child)

        self.birds = new_birds
        self.dead_birds = []

    def __tournament_selection(self, parents):
        """Select parent using tournament selection"""
        candidate1 = random.choice(parents)
        candidate2 = random.choice(parents)
        return candidate1 if candidate1.fitness() > candidate2.fitness() else candidate2

    def __crossover_and_mutate(self, parent1_data, parent2_data):
        p1_weights, p1_biases = parent1_data
        p2_weights, p2_biases = parent2_data

        # Crossover weights
        child_weights = []
        for layer_idx in range(len(p1_weights)):
            layer_weights = []
            for neuron_idx in range(len(p1_weights[layer_idx])):
                neuron_weights = []
                for weight_idx in range(len(p1_weights[layer_idx][neuron_idx])):
                    # 50% chance to inherit from each parent
                    if random.random() < 0.5:
                        weight = p1_weights[layer_idx][neuron_idx][weight_idx]
                    else:
                        weight = p2_weights[layer_idx][neuron_idx][weight_idx]

                    # Adaptive mutation rate: higher early, lower later
                    mutation_rate = max(0.05, 0.2 - (self.generation * 0.001))
                    if random.random() < mutation_rate:
                        # Gaussian mutation instead of uniform
                        weight += random.gauss(0, 0.2)
                        weight = max(-3, min(3, weight))  # Wider bounds

                    neuron_weights.append(weight)
                layer_weights.append(neuron_weights)
            child_weights.append(layer_weights)

        # Crossover biases
        child_biases = []
        for layer_idx in range(len(p1_biases)):
            layer_biases = []
            for bias_idx in range(len(p1_biases[layer_idx])):
                if random.random() < 0.5:
                    bias = p1_biases[layer_idx][bias_idx]
                else:
                    bias = p2_biases[layer_idx][bias_idx]

                mutation_rate = max(0.05, 0.2 - (self.generation * 0.001))
                if random.random() < mutation_rate:
                    bias += random.gauss(0, 0.2)
                    bias = max(-3, min(3, bias))

                layer_biases.append(bias)
            child_biases.append(layer_biases)

        return child_weights, child_biases

    def save_best_bird(self, filename=None):
        """Saves the brain of best bird"""
        if not self.best_bird:
            print("No best bird to save yet..")
            return None

        best_curr_gen = max(self.birds, key=lambda b: b.fitness())
        if best_curr_gen.fitness() > self.best_bird.fitness():
            self.best_bird = best_curr_gen

        metadata = {
            'generation': self.generation,
            'fitness': self.best_bird.fitness(),
            'score': self.best_bird.score,
            'survival_time': self.best_bird.survival_time,
            'saved_by': 'FlappyBird AI Training'
        }

        filepath = self.best_bird.brain.save_to_file(filename, metadata)

        return filepath

    def load_bird_brain(self, filepath, strategy='elite_mutated'):
        """Load a saved bird brain with different seeding strategies"""
        try:
            brain, metadata = NeuralNetwork.load_from_file(filepath)
            print(f"Loaded brain: Gen {metadata.get('generation', '?')}, "
                  f"Fitness {metadata.get('fitness', '?')}")

            if strategy == 'single':
                # Your current approach
                self.birds[0] = Bird(brain)

            elif strategy == 'full':
                # All birds get the same brain
                for bird in self.birds:
                    bird.brain = brain.copy()

            elif strategy == 'partial':
                # Half get loaded brain, half stay random
                mid = len(self.birds) // 2
                for i in range(mid):
                    self.birds[i].brain = brain.copy()

            elif strategy == 'elite_mutated':
                # One elite + mutated variants (RECOMMENDED)
                self.birds[0].brain = brain.copy()  # Perfect copy

                for i in range(1, len(self.birds)):
                    self.birds[i].brain = brain.copy()
                    self._mutate_brain(self.birds[i].brain,
                                       mutation_rate=0.05 + random.random() * 0.1)

            print(f"Applied '{strategy}' seeding strategy")

        except Exception as e:
            print(f"Failed to load brain: {e}")

    def _mutate_brain(self, brain, mutation_rate=0.1):
        """Add small random mutations to brain weights"""
        weights, biases = brain.get_weights()

        # Mutate weights
        for layer_weights in weights:
            for neuron_weights in layer_weights:
                for i in range(len(neuron_weights)):
                    if random.random() < mutation_rate:
                        neuron_weights[i] += random.gauss(0, 0.1)

        # Mutate biases
        for layer_biases in biases:
            for i in range(len(layer_biases)):
                if random.random() < mutation_rate:
                    layer_biases[i] += random.gauss(0, 0.1)

        brain.set_weights(weights, biases)

    def get_current_score(self):
        if self.birds:
            return max(bird.score for bird in self.birds)
        return 0

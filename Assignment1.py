import numpy as np
import matplotlib.pyplot as plt
import random
import math
import time

# Environment Parameters
GRID_SIZE = 50
NUM_AGENTS = 10
COVERAGE_RADIUS = 1
COMM_RADIUS = 1
FLOCK_RADII = [3, 5, 7]
FLOCK_ALIGNMENT_WEIGHTS = [0.3, 0.5, 0.7]
FLOCK_COHESION_WEIGHTS = [0.3]
FLOCK_SEPARATION_WEIGHTS = [2]
STEPS_PER_EPISODE =100
EPISODES = 5
GENERATIONS = 1
POPULATION_SIZE = 20
TOP_K = 5
EDGE_PENALTY = True  # set to True to penalize edge-hugging
CENTER_PULL = False          # set to True to bias agents towards center
EDGE_DIST_THRESHOLD = 2     # distance from edge considered "too close"
EDGE_PENALTY_VALUE = -1.0   # penalty when near edge
CENTER_PULL_WEIGHT = 0.75 #how strongly they get nudged back to center

# Environment Class with Communication and Flocking
class SwarmEnv:
    def __init__(self, grid_size, num_agents, flock_alignment_weight, flock_cohesion_weight, flock_separation_weight, flock_radius):
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.flock_alignment_weight = flock_alignment_weight
        self.flock_cohesion_weight = flock_cohesion_weight
        self.flock_separation_weight = flock_separation_weight
        self.flock_radius = flock_radius
        self.reset()

    def print_parameters(self):
        print("--------------------------------------")
        print("Parameters used : ")
        print(f"flock radius: {self.flock_radius}" )
        print(f"flock alighment weight : {self.flock_alignment_weight}")
        print(f"flock cohesion weight : {self.flock_cohesion_weight}")
        print(f"flock separation weight : {self.flock_separation_weight}")
        
    def reset(self):
        self.covered = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.positions = np.random.randint(0, self.grid_size, (self.num_agents, 2))
        self.velocities = np.zeros((self.num_agents, 2), dtype=int)
        return self._get_state()

    def _get_state(self):
        return self.positions.copy(), self.covered.copy()

    def _decode_action(self, action):
        moves = [(0,-1), (0,1), (-1,0), (1,0), (-1,-1), (-1,1), (1,-1), (1,1)]
        return moves[action % len(moves)]
    
    def get_coverage_count(self):
        """Return the number of cells covered so far."""
        return int(np.sum(self.covered))

    def _apply_flocking(self): 
        new_velocities = np.zeros_like(self.velocities, dtype=float)
        for i in range(self.num_agents):
            pos_i = self.positions[i]
            neighbors = []
            for j in range(self.num_agents):
                if i != j:
                    dist = np.linalg.norm(self.positions[j] - pos_i)
                    if dist < self.flock_radius:
                        neighbors.append(j)
    
            if not neighbors:
                new_velocities[i] = self.velocities[i]
                continue
    
            # Alignment: average direction
            alignment = np.mean(self.velocities[neighbors], axis=0)
    
            # Cohesion: move toward center of mass
            center = np.mean(self.positions[neighbors], axis=0)
            cohesion = center - pos_i
    
            # Separation: move away from close neighbors
            separation = np.sum(pos_i - self.positions[neighbors], axis=0)
    
            # Combine
            combined = (self.flock_alignment_weight * alignment +
                        self.flock_cohesion_weight * cohesion +
                        self.flock_separation_weight * separation)
    
            if np.linalg.norm(combined) > 0:
                combined = combined / np.linalg.norm(combined)
    
            # Blend with current velocity (instead of replacing)
            new_velocities[i] = 0.5 * self.velocities[i] + 0.5 * combined
    
        # Normalize velocities to unit steps
        for i in range(self.num_agents):
            if np.linalg.norm(new_velocities[i]) > 0:
                new_velocities[i] = np.round(new_velocities[i] / np.linalg.norm(new_velocities[i]))
            else:
                new_velocities[i] = np.array([0,0])
    
        self.velocities = new_velocities


    def step_with_flocking_and_communication(self, actions):
        self._apply_flocking()
        rewards = np.zeros(self.num_agents)

        # Combine action direction with flocking direction
        movement = np.clip(
            np.array([self._decode_action(a) for a in actions]) + self.velocities,
            -1, 1
        )
        new_positions = np.clip(self.positions + movement, 0, self.grid_size - 1)

        # Apply rewards and update map
        for i in range(self.num_agents):
            x, y = new_positions[i]

            # Coverage reward/penalty
            for dx in range(-COVERAGE_RADIUS, COVERAGE_RADIUS + 1):
                for dy in range(-COVERAGE_RADIUS, COVERAGE_RADIUS + 1):
                    nx, ny = int(x + dx), int(y + dy)
                    if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                        if not self.covered[nx, ny]:
                            rewards[i] += 1.0   # reward for new cell
                            self.covered[nx, ny] = True
                        else:
                            rewards[i] -= 0.1   # penalty for re-visiting
                            
            # Extra: discourage edge hugging
            if EDGE_PENALTY:
                if (x < EDGE_DIST_THRESHOLD or x > self.grid_size - EDGE_DIST_THRESHOLD - 1 or
                    y < EDGE_DIST_THRESHOLD or y > self.grid_size - EDGE_DIST_THRESHOLD - 1):
                    center = np.array([self.grid_size/2, self.grid_size/2])
                    vec_to_center = center - np.array([x, y])
                    direction = vec_to_center / np.linalg.norm(vec_to_center)
                    self.velocities[i] = np.round(
                        (1-CENTER_PULL_WEIGHT)*self.velocities[i] + CENTER_PULL_WEIGHT*direction
                    ).astype(int)
                    rewards[i] += EDGE_PENALTY_VALUE 

            # Extra: soft pull to center
            if CENTER_PULL:
                center = np.array([self.grid_size/2, self.grid_size/2])
                vec_to_center = center - np.array([x, y])
                if np.linalg.norm(vec_to_center) > 0:
                    direction = vec_to_center / np.linalg.norm(vec_to_center)
                    # add bias to velocity (nudges agent)
                    self.velocities[i] = np.round(
                        (1-CENTER_PULL_WEIGHT)*self.velocities[i] + CENTER_PULL_WEIGHT*direction
                    ).astype(int)
                    

        self.positions = new_positions
        return self._get_state(), rewards

# Visualization
def render(env, step):
    grid = env.covered.astype(int).copy()
    for x, y in env.positions:
        x=int(x)
        y=int(y)
        grid[x, y] = 2  # mark agent position
    plt.imshow(grid.T, cmap='gray', origin='lower')
    plt.title(f"Step {step}")
    plt.pause(0.01)
    plt.clf()

# Random Policy Generator
def make_random_policy():
    return [random.randint(0, 8) for _ in range(STEPS_PER_EPISODE)]

# Evaluation Function
def evaluate_policy(policy, env):
    total_reward = 0
    for _ in range(EPISODES):
        env.reset()
        for step in range(STEPS_PER_EPISODE):
            actions = [ (policy[step] + random.randint(-1,1)) % 9 for _ in range(env.num_agents) ]
            _, rewards = env.step_with_flocking_and_communication(actions)
            total_reward += np.sum(rewards)
    return total_reward

# Evolution Loop
def evolutionary_learning(FLOCK_ALIGNMENT_WEIGHT, FLOCK_COHESION_WEIGHT, FLOCK_SEPARATION_WEIGHT, FLOCK_RADIUS):
    env = SwarmEnv(GRID_SIZE, NUM_AGENTS, FLOCK_ALIGNMENT_WEIGHT, FLOCK_COHESION_WEIGHT, FLOCK_SEPARATION_WEIGHT, FLOCK_RADIUS)
    population = [make_random_policy() for _ in range(POPULATION_SIZE)]

    for gen in range(GENERATIONS):
        scores = [evaluate_policy(policy, env) for policy in population]
        top_indices = np.argsort(scores)[-TOP_K:]
        top_policies = [population[i] for i in top_indices]

       #rint(f"Generation {gen+1}: Best Score = {scores[top_indices[-1]]}")

        # Create new generation
        new_population = top_policies.copy()
        while len(new_population) < POPULATION_SIZE:
            parent = random.choice(top_policies)
            child = [(action + random.choice([-1, 0, 1])) % 9 for action in parent]
            new_population.append(child)

        population = new_population

    # Visualize best policy
    best_policy = top_policies[-1]
    env.print_parameters()
    print(f"cells covered : {env.get_coverage_count()}")
    env.reset()
    plt.figure(figsize=(6,6))
    for step in range(STEPS_PER_EPISODE):
        actions = [best_policy[step] for _ in range(env.num_agents)]
        env.step_with_flocking_and_communication(actions)
        render(env, step)
    plt.show()

# Run learning
for FLOCK_RADIUS in FLOCK_RADII:
    for FLOCK_ALIGNMENT_WEIGHT in FLOCK_ALIGNMENT_WEIGHTS:
        for FLOCK_COHESION_WEIGHT in FLOCK_COHESION_WEIGHTS:
            for FLOCK_SEPARATION_WEIGHT in FLOCK_SEPARATION_WEIGHTS:
                np.random.seed(42)
                start_time = time.time()
                evolutionary_learning(FLOCK_ALIGNMENT_WEIGHT, FLOCK_COHESION_WEIGHT, FLOCK_SEPARATION_WEIGHT, FLOCK_RADIUS)
                end_time = time.time()
                time_taken = end_time - start_time

                print(f"time taken : {time_taken}")

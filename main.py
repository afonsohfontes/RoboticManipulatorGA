import numpy as np
import math
import random

# Define the dimensions of the workspace (length, width, height)
length = 13  # For example, 10 units long
width = 5    # For example, 5 units wide
height = 3   # For example, 3 units tall
'''
task_params = {
	"task_id": "a01",
	"brick_type": 2, # index 0 -> 10 defining which type from the brickDict
	"posX": -0.5, # value from -1 to 1, px is calculated as px = posX * (total_x/2)
	"posY": 1, # value from -1 to 1, py is calculated as py = posy * (total_y/2)
	"rotationZ": 45, #in degrees (-180 to 180)
	"color": 2 # index of ColorList (0 -> 9  = len(colorList))
}
'''

# Create a 3D matrix with the given dimensions, initialized with zeros
workspace_matrix = np.zeros((height, width, length))

workspace_matrix[2, 4, 6] = 1  #end effector start / end position set to 1

# Predefined brick positions within the 3D matrix
# Each tuple represents (z, y, x)
brick_positions = {
    0: (0, 3, 2), 1: (0, 3, 3), 2: (0, 3, 4), 3: (0, 3, 5),
    4: (0, 2, 2), 5: (0, 2, 3), 6: (0, 2, 4), 7: (0, 2, 5),
    8: (0, 1, 3), 9: (0, 1, 4), 10: (0, 1, 5)
}

# Eligible positions for placing a brick
eligible_positions = [(0, 1, 7), (0, 1, 8), (0, 1, 9), (0, 1, 10), (0, 1, 11),
                      (0, 2, 7), (0, 2, 8), (0, 2, 9), (0, 2, 10), (0, 2, 11),
                      (0, 3, 7), (0, 3, 8), (0, 3, 9), (0, 3, 10), (0, 3, 11)]


def generate_task_id():
    """
    Generates a random 3-digit task ID.

    Returns:
        str: A string representing the task ID.
    """
    return f"a{random.randint(100, 999)}"


def generate_rotationZ():
    """
    Generates a random angle between -180 and 180 degrees.

    Returns:
        int: A randomly chosen angle in degrees.
    """
    return random.randint(-180, 180)


def generate_color():
    """
    Generates a random index for color, between 0 and 9.

    Returns:
        int: A randomly chosen index for the color.
    """
    return random.randint(0, 9)


def euclidean_distance(point1, point2):
    """
    Calculates the Euclidean distance between two points in 3D space.

    Args:
        point1 (tuple): The first point as a tuple (z, y, x).
        point2 (tuple): The second point as a tuple (z, y, x).

    Returns:
        float: The Euclidean distance between point1 and point2.
    """
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2 + (point2[2] - point1[2])**2)

def generate_brick_and_position():
    """
    Generates a brick type and randomly selects posX and posY from the predefined eligible positions.

    Returns:
        dict: Contains the brick type, posX, posY, and posZ.
    """

    # Select a random brick type
    brick_type = random.choice(list(brick_positions.keys()))
    # Randomly select a position from the eligible positions
    posZ, posY, posX = random.choice(eligible_positions)

    brick_position = brick_positions[brick_type]
    # Calculate the distance from the brick position to the pick position
    distance = euclidean_distance(brick_position, (posZ, posY, posX))


    return {
        "brick_type": brick_type,
        "posX": posX,
        "posY": posY,
        "posZ": posZ
    }


def initialize_population(pop_size):
    """
    Initializes a population for the GA.

    Args:
        pop_size (int): The size of the population to generate.

    Returns:
        list: A list of dictionaries, each representing an individual in the population.
    """
    population = []
    for _ in range(pop_size):
        individual = generate_brick_and_position()
        population.append(individual)
    return population


def calculate_fitness(individual, previous_picks=[], previous_places=[], global_visited_positions=set(),
                      global_velocities=[]):
    """
    Calculates the fitness of an individual based on multiple criteria:
    - The Euclidean distance between pick and place positions.
    - The average Euclidean distance between the current pick and all previous picks.
    - The average Euclidean distance between the current place and all previous places.
    Adds a penalty if the exact pair of pick and place positions already exist in the test suite.

    Args:
        individual (dict): The individual whose fitness to calculate.
        previous_picks (list): A list of previous pick positions (tuples of (z, y, x)).
        previous_places (list): A list of previous place positions (tuples of (z, y, x)).

    Returns:
        float: The fitness score of the individual, penalized if the exact pair of positions are repeated.
    """
    # Unpack the current pick and place positions
    posZ, posY, posX = individual['posZ'], individual['posY'], individual['posX']
    current_pick = (posZ, posY, posX)
    brick_posZ, brick_posY, brick_posX = brick_positions[individual['brick_type']]
    current_place = (brick_posZ, brick_posY, brick_posX)

    # Calculate distance between pick and place positions
    pick_place_distance = euclidean_distance(current_pick, current_place)

    # Calculate average distances from the current pick and place to all previous picks and places
    avg_pick_distance = np.mean([euclidean_distance(current_pick, prev_pick) for prev_pick in previous_picks]) if previous_picks else 0
    avg_place_distance = np.mean([euclidean_distance(current_place, prev_place) for prev_place in previous_places]) if previous_places else 0

    visited_positions, velocities = simulate_trajectory(individual)

    # Calculate how much new space is explored by this individual
    new_exploration = visited_positions.difference(global_visited_positions)
    exploration_score = len(new_exploration)

    # Calculate how different the velocity profile of this individual is
    unique_velocities = set(velocities).difference(global_velocities)
    velocity_diversity_score = len(unique_velocities)

    # Combine the three factors into a single fitness score
    basic_fitness = pick_place_distance + 0*avg_pick_distance + 0*avg_place_distance

    # Check for exact match of the current pair in the history of pairs
    for prev_pick, prev_place in zip(previous_picks, previous_places):
        if current_pick == prev_pick and current_place == prev_place:
            penalty = -15  # Apply a penalty to discourage exact pair repeats
            basic_fitness  += penalty
            break

    # Calculate how different this individual's trajectory is
    #exploration_difference, velocity_diversity_difference = calculate_difference_score(visited_positions, velocities)

    # Combine components into a final score
    fitness = basic_fitness + 1 * exploration_score + 1 * velocity_diversity_score

    return fitness


def simulate_trajectory(individual):
    """
    Simulates the trajectory for an individual, returning the visited positions and velocities.
    """
    start_pos = (2, 4, 6)  # Starting position
    pick_pos = (individual['posZ'], individual['posY'], individual['posX'])
    place_pos = brick_positions[individual['brick_type']]

    # Simulate the trajectories with velocity
    to_pick = interpolate_line_with_velocity(start_pos, pick_pos)
    to_place = interpolate_line_with_velocity(pick_pos, place_pos)
    to_start = interpolate_line_with_velocity(place_pos, start_pos)

    # Combine all segments
    trajectory = to_pick + to_place + to_start
    visited_positions = {(z, y, x) for z, y, x, v in trajectory}
    velocities = [v for _, _, _, v in trajectory]

    return visited_positions, velocities

def tournament_selection(population, tournament_size=3):
    """
    Selects an individual from the population using tournament selection.

    Args:
        population (list): The population to select from.
        tournament_size (int): The number of individuals to compete in the tournament.

    Returns:
        dict: The winning individual selected to be a parent.
    """
    # Randomly select tournament_size individuals from the population
    tournament = random.sample(population, tournament_size)
    # Select the individual with the highest fitness score
    winner = max(tournament, key=calculate_fitness)
    return winner

def crossover(parent1, parent2):
    """
    Performs crossover between two parents to produce an offspring.

    Args:
        parent1 (dict): The first parent.
        parent2 (dict): The second parent.

    Returns:
        dict: The offspring produced from the parents.
    """
    # Simple crossover: swap posY and posX
    offspring = parent1.copy()
    offspring['posY'], offspring['posX'] = parent2['posY'], parent2['posX']
    return offspring

def mutate(individual, mutation_rate=0.2):
    """
    Mutates an individual's genetic information based on a mutation rate.

    Args:
        individual (dict): The individual to mutate.
        mutation_rate (float): The probability of mutating an individual.

    Returns:
        dict: The mutated individual.
    """
    if random.random() < mutation_rate:
        # Randomly change posX or posY to a new position within eligible_positions
        posZ, posY, posX = random.choice(eligible_positions)
        individual['posY'], individual['posX'] = posY, posX
    return individual



def create_task_params_from_individual(individual):
    """
    Creates the task_params dictionary from a given individual's attributes.

    Args:
        individual (dict): The individual from which to create the task params.

    Returns:
        dict: The task parameters derived from the individual.
    """
    # Note: posX and posY need to be adjusted if they are index-based
    # and your task_params require different ranges or formats
    task_params = {
        "task_id": generate_task_id(),
        "brick_type": individual['brick_type'],  # Use brick_type from individual
        "posX": individual['posX'],  # Use posX from individual
        "posY": individual['posY'],  # Use posY from individual
        "rotationZ": generate_rotationZ(),
        "color": generate_color()
    }
    return task_params


def interpolate_line(start, end):
    """
    Generates points on a straight line from start to end using Bresenham's algorithm.
    This is a simplified version for 3D space.
    """
    points = []
    start_z, start_y, start_x = start
    end_z, end_y, end_x = end

    dz = abs(end_z - start_z)
    dy = abs(end_y - start_y)
    dx = abs(end_x - start_x)
    if start_x < end_x:
        xi = 1
    else:
        xi = -1
    if start_y < end_y:
        yi = 1
    else:
        yi = -1
    if start_z < end_z:
        zi = 1
    else:
        zi = -1

    # Driving axis is Z-axis
    if dz >= dx and dz >= dy:
        p1 = 2 * dx - dz
        p2 = 2 * dy - dz
        while start_z != end_z:
            start_z += zi
            if p1 >= 0:
                start_x += xi
                p1 -= 2 * dz
            if p2 >= 0:
                start_y += yi
                p2 -= 2 * dz
            p1 += 2 * dx
            p2 += 2 * dy
            points.append((start_z, start_y, start_x))
    # Driving axis is Y-axis
    elif dy >= dx and dy >= dz:
        p1 = 2 * dx - dy
        p2 = 2 * dz - dy
        while start_y != end_y:
            start_y += yi
            if p1 >= 0:
                start_x += xi
                p1 -= 2 * dy
            if p2 >= 0:
                start_z += zi
                p2 -= 2 * dy
            p1 += 2 * dx
            p2 += 2 * dz
            points.append((start_z, start_y, start_x))
    # Driving axis is X-axis
    else:
        p1 = 2 * dy - dx
        p2 = 2 * dz - dx
        while start_x != end_x:
            start_x += xi
            if p1 >= 0:
                start_y += yi
                p1 -= 2 * dx
            if p2 >= 0:
                start_z += zi
                p2 -= 2 * dx
            p1 += 2 * dy
            p2 += 2 * dz
            points.append((start_z, start_y, start_x))

    return points


def update_workspace_with_best_individual(workspace_matrix, individual):
    """
    Updates the workspace matrix with the straight-line trajectories for the best individual.
    """
    # Starting position (fixed as per the problem statement)
    start_pos = (2, 4, 6)

    # Extract position and brick type from the individual for pick and place
    pick_pos = (individual['posZ'], individual['posY'], individual['posX'])
    place_pos = brick_positions[individual['brick_type']]

    # Calculate the trajectory from start to pick, pick to place, and place to start
    # Note: The marker '5' is used here for visibility; adjust as necessary
    for point in interpolate_line(start_pos, pick_pos) + interpolate_line(pick_pos, place_pos) + interpolate_line(
            place_pos, start_pos):
        workspace_matrix[point] = 5


# After defining the best individual
# update_workspace_with_best_individual(workspace_matrix, best_individual)
# print(workspace_matrix)


def generate_test_suite(pop_size=10, iterations=10, test_cases=10):
    test_suite = []
    previous_picks = []
    previous_places = []
    workspaces = []
    global global_visited_positions, global_velocities

    for _ in range(test_cases):
        # Re-initialize workspace_matrix for each test case
        workspace_matrix = np.zeros((height, width, length))
        workspace_matrix[2, 4, 6] = 1  # Reset end effector start/end position

        population = initialize_population(pop_size)

        for _ in range(iterations):
            # Modify selection, crossover, and mutation steps as needed
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            while parent1 == parent2:
                parent2 = tournament_selection(population)

            offspring = crossover(parent1, parent2)
            offspring = mutate(offspring)

            # Now calculate fitness considering the history of picks and places
            offspring_fitness = calculate_fitness(offspring, previous_picks, previous_places, global_visited_positions, global_velocities)
            offspring['fitness'] = offspring_fitness  # Optional, store fitness in individual

            # Population update (simplified for demonstration)
            population.sort(key=lambda ind: calculate_fitness(ind, previous_picks, previous_places, global_visited_positions, global_velocities))
            population[0] = offspring

        best_individual = max(population, key=lambda ind: calculate_fitness(ind, previous_picks, previous_places, global_visited_positions, global_velocities))
        test_suite.append(create_task_params_from_individual(best_individual))

        # Update history of picks and places
        best_pick = (best_individual['posZ'], best_individual['posY'], best_individual['posX'])
        best_place = brick_positions[best_individual['brick_type']]
        previous_picks.append(best_pick)
        previous_places.append(best_place)

        # Update global metrics and reset workspace_matrix for visualization
        visited_positions, velocities = simulate_trajectory(best_individual)
        update_global_metrics(visited_positions, velocities)
        update_workspace_with_velocity(workspace_matrix,
                                       best_individual)  # This function now uses the re-initialized workspace_matrix
        #print(workspace_matrix)  # Prints the workspace matrix for the current test case
        workspaces.append(np.copy(workspace_matrix))
    return test_suite, workspaces


def interpolate_line_with_velocity(start, end, max_velocity=9):
    """
    Generates points on a straight line from start to end, including simple acceleration
    and deceleration phases. Returns a list of tuples where each tuple contains the point
    coordinates and the simulated velocity at that point.
    """
    points = interpolate_line(start, end)  # Use the previous interpolate_line function
    num_points = len(points)

    # Calculate acceleration and deceleration phases length (1/3 of the path each)
    phase_length = num_points // 3

    velocities = []
    for i in range(num_points):
        if i < phase_length:  # Acceleration phase
            # Linear acceleration: velocity increases from 1 to max_velocity
            velocity = 1 + (max_velocity - 2) * i / phase_length
        elif i > num_points - phase_length:  # Deceleration phase
            # Linear deceleration: velocity decreases to 1
            velocity = 1 + (max_velocity - 2) * (num_points - 1 - i) / phase_length
        else:  # Constant speed phase
            velocity = max_velocity
        velocities.append(min(max(int(velocity), 1), max_velocity))  # Ensure velocity is within [1, max_velocity]

    return [(points[i][0], points[i][1], points[i][2], velocities[i]) for i in range(num_points)]


def update_workspace_with_velocity(workspace_matrix, individual):
    """
    Updates the workspace matrix with the trajectory for the best individual, taking into account
    acceleration and deceleration to simulate varying velocity.
    """
    #workspace_matrix = np.zeros((height, width, length))
    workspace_matrix[2, 4, 6] = 1
    start_pos = (2, 4, 6)  # Starting position
    pick_pos = (individual['posZ'], individual['posY'], individual['posX'])
    place_pos = brick_positions[individual['brick_type']]

    # Calculate trajectories with velocity
    to_pick = interpolate_line_with_velocity(start_pos, pick_pos)
    to_place = interpolate_line_with_velocity(pick_pos, place_pos)
    to_start = interpolate_line_with_velocity(place_pos, start_pos)

    # Mark the trajectories on the workspace matrix
    for segment in [to_pick, to_place, to_start]:
        for z, y, x, velocity in segment:
            workspace_matrix[z, y, x] = velocity


# Global variables to hold aggregate information
global_visited_positions = set()
global_velocities = []

def update_global_metrics(visited_positions, velocities):
    global global_visited_positions
    global global_velocities
    global_visited_positions.update(visited_positions)
    global_velocities.extend(velocities)


def calculate_physcov_velocity(workspaces):
    unique_visited_pos_vel = set()

    # Iterate over each workspace
    for workspace in workspaces:
        for z, layer in enumerate(workspace):
            for y, row in enumerate(layer):
                for x, velocity in enumerate(row):
                    if velocity > 0:
                        unique_visited_pos_vel.add((z, y, x, velocity))

    # Calculate the number of unique positions visited
    unique_positions = len(set((z, y, x) for z, y, x, _ in unique_visited_pos_vel))
    # Calculate the total possible positions
    total_positions = len(workspaces[0]) * len(workspaces[0][0]) * len(workspaces[0][0][0])
    # Calculate exploration coverage as the ratio of unique visited positions to total positions
    exploration_coverage = unique_positions / total_positions

    # Calculate velocity diversity as the number of unique velocity values across all visited positions
    velocity_diversity = len(set(velocity for _, _, _, velocity in unique_visited_pos_vel))

    # Optionally, combine exploration coverage and velocity diversity into a single score
    # This combination method is arbitrary and can be adjusted according to specific requirements
    combined_score = exploration_coverage * velocity_diversity

    return exploration_coverage, velocity_diversity, combined_score


def calculate_difference_score(new_positions, new_velocities):
    """
    Calculate scores based on how different the new positions and velocities
    are compared to the global metrics.
    """
    # Exploration difference
    new_exploration = len(new_positions.difference(global_visited_positions))
    exploration_difference_score = new_exploration

    # Velocity diversity difference (could use more sophisticated metrics)
    new_velocity_range = set(new_velocities)
    existing_velocity_range = set(global_velocities)
    new_velocity_variety = len(new_velocity_range.difference(existing_velocity_range))
    velocity_diversity_difference_score = new_velocity_variety

    return exploration_difference_score, velocity_diversity_difference_score




# Initial population setup
pop_size = 10  # Example population size
population = initialize_population(pop_size)

# Number of iterations
iterations = 10

for _ in range(iterations):
    # Selection
    parent1 = tournament_selection(population)
    parent2 = tournament_selection(population)
    while parent1 == parent2:
        # Ensure different parents are selected
        parent2 = tournament_selection(population)

    # Crossover and mutation
    offspring = crossover(parent1, parent2)
    offspring = mutate(offspring)

    # Calculate and print the fitness of the offspring
    offspring_fitness = calculate_fitness(offspring)
    #print(f"Offspring Fitness: {offspring_fitness}")

    # Sort the population based on fitness, worst to best
    population.sort(key=calculate_fitness)

    # Replace the worst individual with the new offspring
    # Assuming the population size should remain constant
    population[0] = offspring

# At this point, you might want to evaluate the entire population
# to find and report the individual with the highest fitness
best_individual = max(population, key=calculate_fitness)
best_fitness = calculate_fitness(best_individual)
print(f"Best Individual Fitness after {iterations} iterations: {best_fitness}")

# Generate the test suite
test_suite, collected_workspaces = generate_test_suite()
for workspace in collected_workspaces:
    print(workspace)

exploration_coverage, velocity_diversity, combined_score = calculate_physcov_velocity(collected_workspaces)
print(f"Exploration Coverage: {exploration_coverage:.2f}")
print(f"Velocity Diversity: {velocity_diversity}")
print(f"Combined PhysCov Score: {combined_score:.2f}")
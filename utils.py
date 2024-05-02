import math
import random
import numpy as np
import csv
import json


length = 21
width = 10
height = 3
brick_positions = {
    3: (0, 7, 3),
    2: (0, 7, 5),
    1: (0, 7, 7),
    0: (0, 7, 9),
    7: (0, 4, 3),
    6: (0, 4, 5),
    5: (0, 4, 7),
    4: (0, 4, 9),
    10: (0, 1, 5),
    9: (0, 1, 7),
    8: (0, 1, 9)
}

eligible_positions = [(0, row, col) for row in range(1, width - 1)  # Rows 1 through width-1 (excluding the last row)a
                      for col in range(math.ceil(length / 2), length - 1)]

def update_config(new_length, new_width, new_height, new_brick_positions):
    global length, width, height, eligible_positions, brick_positions
    length = new_length
    width = new_width
    height = new_height
    brick_positions = new_brick_positions
    # Recalculate eligible positions based on new dimensions

    eligible_positions = [(0, row, col) for row in range(1, width - 1)
                          # Rows 1 through width-1 (excluding the last row)
                          for col in range(math.ceil(length / 2), length - 1)]



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


def simulate_trajectory(individual):
    """
    Simulates the trajectory for an individual, returning the visited positions and velocities.
    """
    start_pos = (2, 9, 10)  # Starting position is [2, 9, 10]
    pick_pos = (individual['posZ'], individual['posY'], individual['posX'])
    place_pos = brick_positions[individual['brick_type']]

    # Simulate the trajectories with velocity
    to_pick = interpolate_line_with_velocity(start_pos, pick_pos)
    to_place = interpolate_line_with_velocity(pick_pos, place_pos)
    to_start = interpolate_line_with_velocity(place_pos, start_pos)

    # Combine all segments
    trajectory = to_pick + to_place + to_start
    #visited_positions = {(z, y, x) for z, y, x, v in trajectory}
    visited_positions_with_velocity = {(z, y, x): v for z, y, x, v in trajectory}

    return visited_positions_with_velocity


def tournament_selection(population, previous_picks, previous_places, global_visited_positions, global_velocities,
                         tournament_size=3):
    """
    Selects an individual from the population using tournament selection.

    Args:
        population (list): The population to select from.
        previous_picks (list): List of previous picks used for calculating fitness.
        previous_places (list): List of previous places used for calculating fitness.
        global_visited_positions (set or list): The set or list of global visited positions.
        global_velocities (dict): Dictionary of velocities recorded at different positions.
        tournament_size (int): The number of individuals to compete in the tournament.

    Returns:
        dict: The winning individual selected to be a parent.
    """
    # Randomly select tournament_size individuals from the population
    tournament = random.sample(population, tournament_size)
    # Select the individual with the highest fitness score
    winner = max(tournament, key=lambda ind: calculate_fitness(ind, previous_picks, previous_places,
                                                               global_visited_positions, global_velocities))

    return winner


def get_velocity_category(velocity):
    """Maps a velocity to its category."""
    if velocity == 0:
        return 'Zero'
    elif 1 <= velocity <= 3:
        return 'Slow'
    elif 4 <= velocity <= 6:
        return 'Medium'
    elif 7 <= velocity <= 9:
        return 'Fast'
    else:
        raise ValueError("Velocity out of expected range")

def calculate_fitness(individual, previous_picks, previous_places, global_visited_positions,
                      global_velocities, p=False):

    posZ, posY, posX = individual['posZ'], individual['posY'], individual['posX']
    current_pick = (posZ, posY, posX)
    brick_posZ, brick_posY, brick_posX = brick_positions[individual['brick_type']]
    current_place = (brick_posZ, brick_posY, brick_posX)
    pick_place_distance = euclidean_distance(current_pick, current_place)

    avg_pick_distance = np.mean(
        [euclidean_distance(current_pick, prev_pick) for prev_pick in previous_picks]) if previous_picks else 0
    avg_place_distance = np.mean(
        [euclidean_distance(current_place, prev_place) for prev_place in previous_places]) if previous_places else 0

    visited_positions_with_velocity = simulate_trajectory(individual)
    new_exploration = set(visited_positions_with_velocity.keys()).difference(set(global_visited_positions))
    exploration_score = len(new_exploration)

    velocity_changes = 0
    for pos, vel in visited_positions_with_velocity.items():
        vel_category = get_velocity_category(vel)
        if global_velocities[pos]:
            existing_categories = {get_velocity_category(v) for v in global_velocities[pos]}
            if vel_category not in existing_categories:
                velocity_changes += 1

    velocity_diversity_score = velocity_changes

    basic_fitness = 1 * pick_place_distance + 1.5 * avg_pick_distance + 3 * avg_place_distance
    penalty = 0
    for prev_pick, prev_place in zip(previous_picks, previous_places):
        if current_pick == prev_pick and current_place == prev_place:
            penalty = -15
            basic_fitness += penalty
            break

    fitness = 1 * basic_fitness + 2 * exploration_score + 2 * velocity_diversity_score

    if p:
        print("Fitness Report:")
        print("  Pick-Place Distance: {:.2f}".format(pick_place_distance))
        print("  Average Pick Distance: {:.2f}".format(avg_pick_distance))
        print("  Average Place Distance: {:.2f}".format(avg_place_distance))
        print("  Exploration Score (new unique positions): {}".format(exploration_score))
        print("  Velocity Diversity Score (new categories): {}".format(velocity_diversity_score))
        print("  Basic Fitness (before penalty): {:.2f}".format(basic_fitness - penalty))
        print("  Penalty Applied: {}".format(penalty))
        print("  Total Fitness: {:.2f}".format(fitness))
    return fitness



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
    start_pos = (2, 9, 10)

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


def interpolate_line(start, end):
    """
    Simplified Bresenham's 3D algorithm to calculate points between start and end.
    """
    points = []
    x0, y0, z0 = start
    x1, y1, z1 = end
    dx, dy, dz = abs(x1-x0), abs(y1-y0), abs(z1-z0)
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    sz = -1 if z0 > z1 else 1
    if dz > dx and dz > dy:
        err1, err2 = 2*dx - dz, 2*dy - dz
        while z0 != z1:
            points.append((x0, y0, z0))
            if err1 > 0:
                x0 += sx
                err1 -= 2*dz
            if err2 > 0:
                y0 += sy
                err2 -= 2*dz
            err1 += 2*dx
            err2 += 2*dy
            z0 += sz
    elif dx > dy:
        err1, err2 = 2*dy - dx, 2*dz - dx
        while x0 != x1:
            points.append((x0, y0, z0))
            if err1 > 0:
                y0 += sy
                err1 -= 2*dx
            if err2 > 0:
                z0 += sz
                err2 -= 2*dx
            err1 += 2*dy
            err2 += 2*dz
            x0 += sx
    else:
        err1, err2 = 2*dx - dy, 2*dz - dy
        while y0 != y1:
            points.append((x0, y0, z0))
            if err1 > 0:
                x0 += sx
                err1 -= 2*dy
            if err2 > 0:
                z0 += sz
                err2 -= 2*dy
            err1 += 2*dx
            err2 += 2*dz
            y0 += sy
    points.append((x1, y1, z1))
    return points
def simulate_trajectory_with_orientation(individual):
    """
    Generates trajectory including direction for movement orientations.
    """
    start_pos = (2, 9, 10)  # Starting position
    pick_pos = (individual['posZ'], individual['posY'], individual['posX'])
    place_pos = brick_positions[individual['brick_type']]
    trajectory = []
    segments = [interpolate_line(start_pos, pick_pos),
                interpolate_line(pick_pos, place_pos),
                interpolate_line(place_pos, start_pos)]

    for segment in segments:
        for i in range(len(segment)-1):
            current = segment[i]
            next_point = segment[i+1]
            direction = (next_point[0] - current[0], next_point[1] - current[1], next_point[2] - current[2])
            trajectory.append((current, direction))
    return set(trajectory)  # Return unique positions with directions


def interpolate_line_with_velocity(start, end, max_velocity=9, acceleration_factor=2):
    """
        Generates points on a straight line from start to end, including acceleration
        and deceleration phases with proportional control. Velocity changes by a factor
        not exceeding 1.5 per movement, up to a specified maximum velocity.

        Args:
            start (tuple): Start point as (z, y, x).
            end (tuple): End point as (z, y, x).
            max_velocity (int): Maximum velocity achievable.
            acceleration_factor (float): Maximum factor by which velocity can increase or decrease per step.

        Returns:
            List of tuples: Each tuple contains (z, y, x, velocity) representing the trajectory.
        """
    points = interpolate_line(start, end)
    num_points = len(points)

    # Initialize velocities list with starting velocity
    velocities = [1]

    # Calculate velocities for each point
    for i in range(1, num_points):
        if i < num_points / 2:
            # Acceleration phase
            new_velocity = min(velocities[-1] * acceleration_factor, max_velocity)
        else:
            # Deceleration phase
            distance_to_end = num_points - i
            deceleration_needed = velocities[0] * (acceleration_factor ** distance_to_end)
            new_velocity = max(min(velocities[-1] / acceleration_factor, deceleration_needed), 1)

        velocities.append(new_velocity)

    # Create the final list of points with their corresponding velocities
    trajectory_with_velocities = [(points[i][0], points[i][1], points[i][2], velocities[i]) for i in range(num_points)]

    return trajectory_with_velocities

def update_workspace_with_velocity(workspace_matrix, individual):
    """
    Updates the workspace matrix with the trajectory for the best individual, taking into account
    acceleration and deceleration to simulate varying velocity.
    """
    #workspace_matrix = np.zeros((height, width, length))
    workspace_matrix[2, 9, 10] = 1
    start_pos = (2, 9, 10)  # Starting position
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




def simulate_random_trajectory(start, end):
    """ Simulates a trajectory from start to end with random velocities. """
    workspace_matrix = np.zeros((height, width, length))
    trajectory_with_velocity = interpolate_line_with_velocity(start, end)
    for z, y, x, velocity in trajectory_with_velocity:
        workspace_matrix[z, y, x] = velocity  # Mark the trajectory in the matrix with its velocity
    return workspace_matrix

def generate_dataset(num_cases, filename='dataset.csv'):
    """ Generates a dataset of random trajectories and saves to a CSV file. """
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Input', 'Output'])  # Column headers

        for _ in range(num_cases):
            start = random.choice(eligible_positions)
            end = random.choice(list(brick_positions.values()))
            workspace = simulate_random_trajectory(start, end)

            input_data = {
                'start': start,
                'end': end
            }
            # Serialize input and output for CSV
            serialized_input = json.dumps(input_data)
            serialized_output = serialize_matrix(workspace)

            writer.writerow([serialized_input, serialized_output])

def serialize_matrix(matrix):
    """ Serializes a 3D numpy array into a string. """
    return json.dumps(matrix.tolist())

# Convert posX and posY values to the required range
def convert_position(posX, posY, length, width):
    # Define the eligible columns and rows based on your diagram
    eligible_cols = list(range(11, 20))  # columns 11 to 19 inclusive
    eligible_rows = list(range(1, 9))  # rows 7 to 9 inclusive

    # Normalize posX and posY to a range of 0-1 from the current range of 0-(length-1) or 0-(width-1)
    normalized_posX = (posX - min(eligible_cols)) / (max(eligible_cols) - min(eligible_cols))
    normalized_posY = (posY - min(eligible_rows)) / (max(eligible_rows) - min(eligible_rows))

    # Scale and translate the normalized positions to -1 to 1 range
    new_posX = -normalized_posX * 2 + 1
    new_posY = normalized_posY * 2 - 1

    return new_posX, new_posY


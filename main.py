import math
import random
import logging
import numpy as np
import csv
import json
import utils
import time

# Setup logging configuration
logging.basicConfig(filename='fitness_log.txt', filemode='a', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Define the dimensions of the workspace (length, width, height)
length = 21
width = 10
height = 3

# Global variables to hold aggregate information
global_visited_positions = set()
global_velocities = {}

def initialize_global_velocities(length, width, height):
    velocities = {}
    for z in range(height):
        for y in range(width):
            for x in range(length):
                velocities[(z, y, x)] = [0]  # Initialize each position with a list containing zero
    return velocities

# Initialize global_velocities with the correct dimensions
global_velocities = initialize_global_velocities(length, width, height)

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
#print(workspace_matrix)
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

eligible_positions = [(0, row, col) for row in range(1, width - 1)  # Rows 1 through width-1 (excluding the last row)
                      for col in range(math.ceil(length / 2), length - 1)]
#print(math.ceil(length / 2))
#print(eligible_positions)

utils.update_config(length, width, height, brick_positions)


def generate_test_suite(pop_size=20, iterations=30, test_cases=20):
    test_suite = []
    previous_picks = []
    previous_places = []
    workspaces = []
    global global_visited_positions, global_velocities

    for iTest in range(test_cases):
        # Re-initialize workspace_matrix for each test case
        workspace_matrix = np.zeros((height, width, length))
        workspace_matrix[2, 9, 10] = 1  # Reset end effector start/end position

        population = utils.initialize_population(pop_size)

        for _ in range(iterations):
            # Modify selection, crossover, and mutation steps as needed
            parent1 = utils.tournament_selection(population, previous_picks, previous_places, global_visited_positions,
                                                 global_velocities)
            parent2 = utils.tournament_selection(population, previous_picks, previous_places, global_visited_positions,
                                                 global_velocities)
            while parent1 == parent2:
                parent2 = utils.tournament_selection(population, previous_picks, previous_places,
                                                     global_visited_positions, global_velocities)

            offspring = utils.crossover(parent1, parent2)
            offspring = utils.mutate(offspring)

            # Now calculate fitness considering the history of picks and places
            offspring_fitness = utils.calculate_fitness(offspring, previous_picks, previous_places, global_visited_positions, global_velocities)
            offspring['fitness'] = offspring_fitness  # Optional, store fitness in individual

            # Population update (simplified for demonstration)

            logging.info('POPULATION SORT--------POPULATION SORT--------POPULATION SORT--------POPULATION SORT--------')
            population.sort(key=lambda ind: utils.calculate_fitness(ind, previous_picks, previous_places, global_visited_positions, global_velocities))
            population[0] = offspring
            #population.append(offspring)

        logging.info('BEST INDIVIDUAL--------BEST INDIVIDUAL--------BEST INDIVIDUAL--------BEST INDIVIDUAL--------')
        best_individual = max(population, key=lambda ind: utils.calculate_fitness(ind, previous_picks, previous_places,
                                                                            global_visited_positions, global_velocities))
        print('Test Case {:.0f}'.format(iTest+1)) #: {:.2f}".format(pick_place_distance)
        print(best_individual)
        utils.calculate_fitness(best_individual, previous_picks, previous_places, global_visited_positions,
                                global_velocities, True)
        test_suite.append(utils.create_task_params_from_individual(best_individual))

        # Update history of picks and places
        best_pick = (best_individual['posZ'], best_individual['posY'], best_individual['posX'])
        best_place = brick_positions[best_individual['brick_type']]
        previous_picks.append(best_pick)
        previous_places.append(best_place)

        visited_positions_with_velocity = utils.simulate_trajectory(best_individual)
        update_global_metrics(visited_positions_with_velocity)
        utils.update_workspace_with_velocity(workspace_matrix,
                                       best_individual)
        workspaces.append(np.copy(workspace_matrix))
        print('-------')
        print('\n')
    return test_suite, workspaces

def update_global_metrics(visited_positions_with_velocity):
    """
    Update global metrics with new unique data, including positions and their corresponding velocities.

    Args:
        visited_positions_with_velocity (dict): Dictionary with positions as keys and velocities as values.
    """
    global global_visited_positions
    global global_velocities

    # Extract positions from the keys of the dictionary
    new_positions = set(visited_positions_with_velocity.keys())

    # Determine the new unique positions that were not previously visited
    unique_new_positions = new_positions.difference(set(global_visited_positions))

    # Update global_visited_positions with these new unique positions
    global_visited_positions.update(unique_new_positions)

    unique_new_vel = 0
    for position, velocity in visited_positions_with_velocity.items():
         if velocity not in global_velocities[position]:
            global_velocities[position].append(velocity)
            unique_new_vel += 1

    print(f"Added {len(unique_new_positions)} new unique positions.")
    print(f"Added {(unique_new_vel)} new velocities.")

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



# Generate the dataset
#num_cases = 100  # Number of random test cases
#utils.generate_dataset(num_cases)

# Generate the test suite
start_time = time.time()
test_suite, collected_workspaces = generate_test_suite()
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")

for case in test_suite:
    posX, posY = case['posX'], case['posY']
    new_posX, new_posY = utils.convert_position(posX, posY, length, width)
    case['posX'] = new_posX
    case['posY'] = new_posY
    #print(new_posX)
    #print(new_posY)


for i in range(len(test_suite)):
    print(f"Test Case {i + 1}:")
    print(test_suite[i])
    #print(collected_workspaces[i])
    print("\n")

exploration_coverage, velocity_diversity, combined_score = calculate_physcov_velocity(collected_workspaces)
print(f"Exploration Coverage: {exploration_coverage:.2f}")
print(f"Velocity Diversity: {velocity_diversity}")
print(f"Combined PhysCov Score: {combined_score:.2f}")

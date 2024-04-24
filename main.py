import math
import random
import logging
import numpy as np
import csv
import json
import utils

# Setup logging configuration
logging.basicConfig(filename='fitness_log.txt', filemode='a', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Define the dimensions of the workspace (length, width, height)
length = 21
width = 10
height = 3
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
    0: (0, 7, 3),  # Brick type 0 at row 7, column 3a
    1: (0, 7, 5),  # Brick type 1 at row 7, column 5
    2: (0, 7, 7),  # Brick type 2 at row 7, column 7
    3: (0, 7, 9),  # Brick type 3 at row 7, column 9
    4: (0, 4, 3),  # Brick type 4 at row 4, column 3
    5: (0, 4, 5),  # Brick type 5 at row 4, column 5
    6: (0, 4, 7),  # Brick type 6 at row 4, column 7
    7: (0, 4, 9),  # Brick type 7 at row 4, column 9
    8: (0, 1, 5),  # Brick type 8 at row 1, column 5
    9: (0, 1, 7),  # Brick type 9 at row 1, column 7
    10: (0, 1, 9)  # Brick type 10 at row 1, column 9
}

eligible_positions = [(0, row, col) for row in range(1, width - 1)  # Rows 1 through width-1 (excluding the last row)
                      for col in range(math.ceil(length / 2), length - 1)]
#print(math.ceil(length / 2))
#print(eligible_positions)

utils.update_config(length, width, height, brick_positions)


def generate_test_suite(pop_size=20, iterations=20, test_cases=20):
    test_suite = []
    previous_picks = []
    previous_places = []
    workspaces = []
    global global_visited_positions, global_velocities, global_trajectories

    for _ in range(test_cases):
        # Re-initialize workspace_matrix for each test case
        workspace_matrix = np.zeros((height, width, length))
        workspace_matrix[2, 9, 10] = 1  # Reset end effector start/end position

        population = utils.initialize_population(pop_size)

        for _ in range(iterations):
            # Modify selection, crossover, and mutation steps as needed
            parent1 = utils.tournament_selection(population)
            parent2 = utils.tournament_selection(population)
            while parent1 == parent2:
                parent2 = utils.tournament_selection(population)

            offspring = utils.crossover(parent1, parent2)
            offspring = utils.mutate(offspring)

            # Now calculate fitness considering the history of picks and places
            offspring_fitness = utils.calculate_fitness(offspring, previous_picks, previous_places, global_visited_positions, global_velocities, global_trajectories)
            offspring['fitness'] = offspring_fitness  # Optional, store fitness in individual

            # Population update (simplified for demonstration)

            logging.info('POPULATION SORT--------POPULATION SORT--------POPULATION SORT--------POPULATION SORT--------')
            population.sort(key=lambda ind: utils.calculate_fitness(ind, previous_picks, previous_places, global_visited_positions, global_velocities, global_trajectories))
            population[0] = offspring
            #population.append(offspring)

        logging.info('BEST INDIVIDUAL--------BEST INDIVIDUAL--------BEST INDIVIDUAL--------BEST INDIVIDUAL--------')
        best_individual = max(population, key=lambda ind: utils.calculate_fitness(ind, previous_picks, previous_places,
                                                                            global_visited_positions, global_velocities, global_trajectories))
       # print(calculate_fitness(best_individual, previous_picks, previous_places, global_visited_positions, global_velocities, global_trajectories))
        test_suite.append(utils.create_task_params_from_individual(best_individual))

        # Update history of picks and places
        best_pick = (best_individual['posZ'], best_individual['posY'], best_individual['posX'])
        best_place = brick_positions[best_individual['brick_type']]
        previous_picks.append(best_pick)
        previous_places.append(best_place)

        # Update global metrics and reset workspace_matrix for visualization
        visited_positions, velocities = utils.simulate_trajectory(best_individual)
        trajectories = utils.simulate_trajectory_with_orientation(best_individual)
        update_global_metrics(visited_positions, velocities,trajectories)
        utils.update_workspace_with_velocity(workspace_matrix,
                                       best_individual)
        #print(workspace_matrix)  # Prints the workspace matrix for the current test case
        workspaces.append(np.copy(workspace_matrix))
    return test_suite, workspaces

# Global variables to hold aggregate information
global_visited_positions = []
global_velocities = []
global_trajectories = []

def update_global_metrics(visited_positions, velocities, trajectories):
    """
    Update global metrics with new unique data.

    Args:
        visited_positions (set): Newly visited positions from the latest simulation.
        velocities (list): Newly generated velocities from the latest simulation.
        trajectories (list): Newly generated trajectories with orientation from the latest simulation.
    """
    # Convert global data to sets for faster operations
    global global_visited_positions, global_velocities, global_trajectories

    # Convert lists to sets for comparison and update
    new_positions = set(visited_positions)
    new_velocities = set(velocities)
    new_trajectories = set(trajectories)

    # Update global sets with only new unique items
    unique_new_positions = new_positions.difference(global_visited_positions)
    unique_new_velocities = new_velocities.difference(global_velocities)
    unique_new_trajectories = new_trajectories.difference(global_trajectories)

    # Convert back to list and extend the global variables
    global_visited_positions.extend(unique_new_positions)
    global_velocities.extend(unique_new_velocities)
    global_trajectories.extend(unique_new_trajectories)

    logging.info(f"Added {len(unique_new_positions)} new unique positions.")
    logging.info(f"Added {len(unique_new_velocities)} new unique velocities.")
    logging.info(f"Added {len(unique_new_trajectories)} new unique trajectories.")


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
test_suite, collected_workspaces = generate_test_suite()
for i in range(len(test_suite)):
    print(f"Test Case {i + 1}:")
    print(test_suite[i])
    #print(collected_workspaces[i])
    print("\n")

exploration_coverage, velocity_diversity, combined_score = calculate_physcov_velocity(collected_workspaces)
print(f"Exploration Coverage: {exploration_coverage:.2f}")
print(f"Velocity Diversity: {velocity_diversity}")
print(f"Combined PhysCov Score: {combined_score:.2f}")

import math
import random
import logging
import numpy as np
import csv
import json
import utils
import time
import os

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


def generate_test_suite(pop_size=20, iterations=30, test_cases=100):
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

global_best_individuals_velocities = []
def update_global_metrics(visited_positions_with_velocity):
    """
    Update global metrics with new unique data, including positions and their corresponding velocities.

    Args:
        visited_positions_with_velocity (dict): Dictionary with positions as keys and velocities as values.
    """
    global global_visited_positions
    global global_velocities
    global global_best_individuals_velocities

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
    
    global_best_individuals_velocities.append(visited_positions_with_velocity)
    
    #print(f"Added {len(unique_new_positions)} new unique positions.")
    #print(f"Added {(unique_new_vel)} new velocities.")

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


def initialize_globals():
    global global_visited_positions, global_velocities, global_best_individuals_velocities
    global_visited_positions = set()
    global_velocities = initialize_global_velocities(length, width, height)
    global_best_individuals_velocities = []
    
    
def calculate_physcov_coverage(length=21, width=10, height=3):
    """
    Calculate the fraction of the state space that has been covered with unique non-zero velocity categories.

    Args:
        length (int): The length of the workspace.
        width (int): The width of the workspace.
        height (int): The height of the workspace.

    Returns:
        float: The fraction of the state space that has been covered by unique non-zero velocity categories.
    """
    unique_visited_categories = set()

    # Iterate over global_best_individuals_velocities to count positions with unique non-zero velocity categories
    for visited_positions_with_velocity in global_best_individuals_velocities:
        for pos, velocity in visited_positions_with_velocity.items():
            velocity_category = get_velocity_category(velocity)
            if velocity_category != 'Zero':
                unique_visited_categories.add((pos, velocity_category))

    unique_non_zero_categories_visited = len(unique_visited_categories)
    # Calculate the total number of positions in the state space, considering 3 non-zero velocity categories per position
    total_positions = length * width * height * 3
    # Calculate the coverage as a fraction
    coverage_fraction = unique_non_zero_categories_visited / total_positions

    return coverage_fraction


def get_velocity_category(velocity):
    """
    Maps a velocity to its category.

    Args:
        velocity (int): The velocity to categorize.

    Returns:
        str: The category of the velocity.
    """
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


import csv
def calculate_physcov_coverage_for_all_test_cases(test_suite, length=21, width=10, height=3):
    """
    Calculate the fraction of the state space that has been covered with unique non-zero velocity categories
    for all test cases in the test suite.

    Args:
        test_suite (list): A list of all test cases.
        length (int): The length of the workspace.
        width (int): The width of the workspace.
        height (int): The height of the workspace.

    Returns:
        list: A list of physCov values after each test case.
    """
    unique_visited_categories = set()
    physCov_values = []

    # Iterate over all test cases
    for test_case_index in range(len(test_suite)):
        # Iterate over the velocities in the current test case
        for pos, velocity in global_best_individuals_velocities[test_case_index].items():
            velocity_category = get_velocity_category(velocity)
            if velocity_category != 'Zero':
                unique_visited_categories.add((pos, velocity_category))

        # Calculate the coverage after this test case
        unique_non_zero_categories_visited = len(unique_visited_categories)
        total_positions = length * width * height * 3
        coverage_fraction = unique_non_zero_categories_visited / total_positions
        physCov_values.append(coverage_fraction)

    return physCov_values

# Function to generate a single report
def generate_report(report_num, folder_path):
    # Ensure the folder exists
    os.makedirs(folder_path, exist_ok=True)
    
    # Reinitialize global variables
    initialize_globals()
    
    # List to track physCov values
    physCov_values = []
    reportName1 = 'report_D'
    reportName = f'{reportName1}-{report_num}'
    with open(f'{folder_path}/{reportName1}-{report_num}.txt', 'w') as file:
        # Generate the test suite
        start_time = time.time()
        test_suite, collected_workspaces = generate_test_suite()
        end_time = time.time()
        execution_time = end_time - start_time
        file.write(f"Execution time: {execution_time} seconds\n")

        for case in test_suite:
            posX, posY = case['posX'], case['posY']
            new_posX, new_posY = utils.convert_position(posX, posY, length, width)
            case['posX'] = new_posX
            case['posY'] = new_posY

        for i in range(len(test_suite)):
            file.write(f"Test Case {i + 1}:\n")
            file.write(f"{test_suite[i]}\n\n")


    
    physCov_values = calculate_physcov_coverage_for_all_test_cases(test_suite)
    # Define the path for the CSV file
    csv_file_path = f'{folder_path}/physCov_evolution_report.csv'
    # Check if the file already exists
    file_exists = os.path.isfile(csv_file_path)

    # Save physCov values to a CSV file
    with open(csv_file_path, 'a', newline='') as csvfile:  # Open in append mode
        csv_writer = csv.writer(csvfile)
        if not file_exists:
            csv_writer.writerow(['Report', 'Test Case', 'physCov'])  # Write header only if file does not exist
        for idx, physCov in enumerate(physCov_values):
            csv_writer.writerow([reportName, idx + 1, physCov])
    

# Define the folder path where reports will be saved
folder_path = 'reports'

# Generate 10 reports
for report_num in range(7, 11):
    generate_report(report_num, folder_path)
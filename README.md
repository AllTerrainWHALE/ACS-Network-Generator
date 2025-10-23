# ACS-Network-Generator

## Overview
The ACS-Network-Generator is a simulation project that models the behavior of ant colonies using agents. The environment is visualized using `pygame`, and the simulation includes features such as pheromone dispersal and evaporation, agent movement, and interaction with the environment.

**Full technical report can be found [here](https://bradleyhopper.com/assets/documents/Modelling_duel_pheromone_ant_foraging_in_a_multiple_food_source_environment.pdf)**

## Features
- **Environment Simulation**: Simulates an environment where agents (ants) interact with each other and their surroundings.
- **Pheromone Mechanics**: Implements pheromone dispersal and evaporation using CUDA for performance optimization.
- **Agent Behavior**: Agents can search for food, return to the nest, and avoid obstacles.
- **Visualization**: Real-time visualization of the environment and agent activities using `pygame`.

## Installation
1. **Clone the repository**:
    ```sh
    git clone https://github.com/AllTerrainWHALE/ACS-Network-Generator.git
    cd ACS-Network-Generator
    ```

2. **Set up the environment**:
    - Using `conda`:
        ```sh
        conda env create -f environment.yml
        conda activate environment
        ```

3. **Install additional dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

## Usage
1. **Run the main script**:
    ```sh
    python main.py
    ```

2. **Follow the prompts** to choose whether to visualize the environment.

## Project Structure
```sh
ACS-Network-Generator/
├── .gitignore
├── environment.yml
├── main.py
├── README.md
├── test.py
├── src/
│   ├── __init__.py
│   ├── agent.py
│   ├── cell.py
│   ├── colony.py
│   ├── environment.py
│   ├── utils.py
```

## TO-DO

## Acknowledgements
- [Numba](https://numba.pydata.org/) for CUDA support.
- [Pygame](https://www.pygame.org/) for visualization.
- [Numpy](https://numpy.org/) for numerical operations.

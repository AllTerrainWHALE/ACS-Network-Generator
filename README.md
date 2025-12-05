# ACS Network Generator

## ℹ️ Overview
The ACS-Network-Generator is a simulation project that models the behavior of an ant colony, with the goal of enabling the conlony to produce clear and consise paths between nest/food nodes. A key feature of this model which makes it stand out from the rest is the restricted use of two unique pheromone types.

The environment is visualized using `pygame`, and the simulation includes features such as pheromone dispersal and evaporation, agent movement, and interaction with the environment.

**Full technical report can be found [here](https://bradleyhopper.com/assets/documents/Modelling_duel_pheromone_ant_foraging_in_a_multiple_food_source_environment.pdf)**

## ⭐ Features
- **Environment Simulation**: Simulates an environment where agents (ants) interact with each other and their surroundings.
- **Pheromone Mechanics**: Implements pheromone dispersal and evaporation, using CUDA for performance optimization.
- **Agent Behavior**: Agents can forage for food nodes, branching from either nest or food nodes to expand the network further.
    - Agents follow a given pheromone path based on it's popularity. A path with a lower popularity has a greater desirability.
- **Visualization**: Real-time visualization of the environment and agent activities using `pygame`.

## ⬇️ Installation
1. **Clone the repository**:
    ```sh
    git clone https://github.com/AllTerrainWHALE/ACS-Network-Generator.git
    cd ACS-Network-Generator
    ```

2. **Create the environment**:
    - Using `conda`:
        ```sh
        conda create -n [environment-name]
        conda activate [environment-name]
        ```

3. **Install dependencies**:
    ```sh
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    conda env update -f environment.yml
    ```

## 🚀 Usage
1. **Run the main script**:
    ```sh
    python main.py
    ```

2. **Follow the prompts** to choose whether to visualize the environment.

## 📁 Project Structure
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

## ✅ TO-DO
- [x] Enable agents to not only produce and follow nest-to-food pheromone trails, but food-to-food trails too.
- [ ] Colonies greatly struggle with connecting together large numbers of nodes in the environment; improve pheromone following logic to mitigate this.
- [ ] Introduce obstacles for the agents to navigate around.

## Acknowledgements
- [Numba](https://numba.pydata.org/) for CUDA support.
- [Pygame](https://www.pygame.org/) for visualization.
- [Numpy](https://numpy.org/) for numerical operations.

# Recurrent Rainbow DQN

This repository contains an implementation of a Recurrent Rainbow DQN (Deep Q-Network) algorithm. The code is structured to be run in a Jupyter notebook environment with specific dependencies and Python version requirements.

## Prerequisites

- Python 3.12
- Git (for cloning the repository)

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/bbaral-eng/Recurrent_Rainbow_DQN.git
   cd Recurrent_Rainbow_DQN
   ```

2. Create a virtual environment:
   ```bash
   python3.12 -m venv venv
   ```

3. Activate the virtual environment:
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
   - On Windows:
     ```bash
     .\venv\Scripts\activate
     ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Code

1. Ensure your virtual environment is activated (you should see `(venv)` in your terminal prompt)

2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

3. Open the `RecurrentRainbowDQN.ipynb` notebook

4. Run all cells in the notebook sequentially from top to bottom

## Important Notes

- Make sure to use Python 3.12 specifically, as the code has been tested and verified with this version
- The notebook must be run in sequential order as later cells depend on the execution of previous cells
- Do not skip any cells as this may cause dependency or initialization issues
- If you encounter any memory issues, you may need to restart the kernel and run all cells again

## Troubleshooting

If you encounter any issues:
1. Verify that you're using Python 3.12
2. Confirm all dependencies are installed correctly
3. Try restarting the Jupyter kernel
4. Ensure all cells are run in order from top to bottom

## Project Structure

- `RecurrentRainbowDQN.ipynb`: Main notebook containing the implementation
- `requirements.txt`: List of Python dependencies
- Additional supporting Python files for various functionalities


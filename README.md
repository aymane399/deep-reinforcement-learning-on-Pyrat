# About Pyrat:
Maze game with two players (a rat and a snake) with pieces of cheeses, obstacles and mud.
For more info on the software code and the parameters check : https://github.com/vgripon/PyRat

# Deep Q-learning:
The goal of this project is to train a tensorflow model for the neural network designed to predict the Q function.
The model is written in the rl.py file.

the main.py file initiates to the training, you chose wether to load an existing model or start a new one and wether to save the model in the end or not.

for the code to work you should put the numpy_rl_reload.py file in the AIs/ directory of the original Pyrat repository and put the rest of the files and directories in the original repository.

## Curve of the evolution of the winrate through training:

![Alt text](winrate.png?raw=true "training")

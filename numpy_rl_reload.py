# Template file to create an AI for the game PyRat
# http://formations.telecom-bretagne.eu/pyrat

###############################
# Team name to be displayed in the game 
TEAM_NAME = "KerasAI"

###############################
# When the player is performing a move, it actually sends a character to the main program
# The four possibilities are defined here
MOVE_DOWN = 'D'
MOVE_LEFT = 'L'
MOVE_RIGHT = 'R'
MOVE_UP = 'U'

###############################
# Please put your imports here
import tensorflow as tf
import rl
import numpy as np
import random as rd
import pickle
import time
import sys
sys.path.append('AIs/')
from cgt import best_target,simulate_game_until_target,updatePlayerLocation,checkEatCheese


###############################
# Please put your global variables here

# Global variables
global model,exp_replay,input_tm1, action, score

# Function to create a numpy array representation of the maze

def input_of_parameters(player, maze, opponent, mazeHeight, mazeWidth, piecesOfCheese):
    im_size = (2*mazeHeight-1,2*mazeWidth-1,2)
    canvas = np.zeros(im_size)
    (x,y) = player
    (xx,yy) = opponent
    center_x, center_y = mazeWidth-1, mazeHeight-1
    for (x_cheese,y_cheese) in piecesOfCheese:
        canvas[y_cheese+center_y-y,x_cheese+center_x-x,0] = 1
    canvas[yy+center_y-y,xx+center_x-x,1] = 1   
    canvas = np.expand_dims(canvas,axis=0)
    return canvas



    
###############################
# Preprocessing function
# The preprocessing function is called at the start of a game
# It can be used to perform intensive computations that can be
# used later to move the player in the maze.
###############################
# Arguments are:
# mazeMap : dict(pair(int, int), dict(pair(int, int), int))
# mazeWidth : int
# mazeHeight : int
# playerLocation : pair(int, int)
# opponentLocation : pair(int,int)
# piecesOfCheese : list(pair(int, int))
# timeAllowed : float
###############################
# This function is not expected to return anything
def preprocessing(mazeMap, mazeWidth, mazeHeight, playerLocation, opponentLocation, piecesOfCheese, timeAllowed):
    global model,exp_replay,input_tm1, action, score
    input_tm1 = input_of_parameters(playerLocation, mazeMap, opponentLocation, mazeHeight, mazeWidth, piecesOfCheese)    
    action = -1
    score = 0
    model = rl.NLinearModels(2*1189,4,32)

    
    


###############################
# Turn function
# The turn function is called each time the game is waiting
# for the player to make a decision (a move).
###############################
# Arguments are:
# mazeMap : dict(pair(int, int), dict(pair(int, int), int))
# mazeWidth : int
# mazeHeight : int
# playerLocation : pair(int, int)
# opponentLocation : pair(int, int)
# playerScore : float
# opponentScore : float
# piecesOfCheese : list(pair(int, int))
# timeAllowed : float
###############################
# This function is expected to return a move
current_target =(-1,-1)
def turn(mazeMap, mazeWidth, mazeHeight, playerLocation, opponentLocation, playerScore, opponentScore, piecesOfCheese, timeAllowed):    
    global model,input_tm1, action, score, current_target
    if len(piecesOfCheese) <= 0:   
        if current_target not in piecesOfCheese:
            current_target, score = best_target(playerLocation, opponentLocation, playerScore, opponentScore, piecesOfCheese)
        
        if current_target[1] > playerLocation[1]:
            return MOVE_UP
        if current_target[1] < playerLocation[1]:
            return MOVE_DOWN
        if current_target[0] > playerLocation[0]:
            return MOVE_RIGHT
        return MOVE_LEFT
    else:
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, "save_rl/model.ckpt")
            input_t = input_of_parameters(playerLocation, mazeMap, opponentLocation, mazeHeight, mazeWidth, piecesOfCheese)    
            input_tm1 = input_t
            q = model.predict_one(sess,input_tm1)
            action = np.argmax(q[0])
            score = playerScore
            return [MOVE_LEFT, MOVE_RIGHT, MOVE_UP, MOVE_DOWN][action]   

def postprocessing (mazeMap, mazeWidth, mazeHeight, playerLocation, opponentLocation, playerScore, opponentScore, piecesOfCheese, timeAllowed):
    pass    

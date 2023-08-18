import numpy as np
import pygame
import sys
import math
import random

#defining matrix dimension variables
ROW_COUNT = 6 
COL_COUNT = 7

#defining colors for board, board holes, and player pieces
BLUE = (0,0,255)
BLACK = (0,0,0)
RED = (255,0,0)
YELLOW = (255,255,0)

#defining variables for when its player or AI turn
PLAYER = 0
AI = 1

#defining variables for player and AI piece
PLAYER_PIECE = 1
AI_PIECE = 2

WINDOW_LENGTH = 4 #represents how many pieces in a row needed to win
EMPTY = 0 #represents empty board slot

#defining create_board function to create the game board
def create_board():
    board = np.zeros((ROW_COUNT,COL_COUNT)) #creating matrix of 6 rows x 7 columns
    return board

#defining drop_piece function that is used to allow user to select where they want to drop their piece
def drop_piece(board, row, col, piece):
    board[row][col] = piece #drops piece(piece color dependent on which users turn it is) in the boards assigned row and column 
    pass

#defining function which checks to ensure column (top row is 5) isn't full and if not full it allows user to drop piece
def is_valid_location(board, col):
    return board[ROW_COUNT-1][col] == 0 #only returns successfully if the very top row has a space for the piece to drop into, subtracting 1 to account for black bar row at top

#defining get_next_open_row function used to dictate what row the next piece dropped in a column will be in
def get_next_open_row(board, col):
    for r in range(ROW_COUNT):
        if board[r][col] == 0:  #checks for all instances of row in specified column range until it finds empty row and returns where the next empty row is 
            return r
        
def print_board(board):
    print(np.flip(board, 0))    #reorients the board so that entered pieces go to the bottom

#defining function to dictate winning_move conditionals
def winning_move(board, piece):
    #check horizontal locations for win
    for c in range(COL_COUNT-3): #subtract 3 because you need atleast 4 spaces in a row for game to end so winning move range can't start in last 3 spaces
        for r in range(ROW_COUNT):
            if board[r][c] == piece and board[r][c+1] == piece and board[r][c+2] == piece and board[r][c+3] == piece: #if 4 pieces in a row horizontally
                return True
    
    #check vertical locations for win
    for c in range(COL_COUNT): #subtract 3 because you need atleast 4 spaces in a column for game to end so it can't start in top 3 spaces
        for r in range(ROW_COUNT-3):
            if board[r][c] == piece and board[r+1][c] == piece and board[r+2][c] == piece and board[r+3][c] == piece: #if 4 pieces in a row vertically
                return True
    
    #check positively sloped diagonals
    for c in range(COL_COUNT-3): 
        for r in range(ROW_COUNT-3):
            if board[r][c] == piece and board[r+1][c+1] == piece and board[r+2][c+2] == piece and board[r+3][c+3] == piece: #if 4 pieces in a row positively sloped
                return True   

    #check negatively sloped diagonals
    for c in range(COL_COUNT-3):
        for r in range(3, ROW_COUNT): #starts at index 3 (4th row up) and goes up to the entered row count since it needs to be minimum 4 up for negative sloped diagonal
            if board[r][c] == piece and board[r-1][c+1] == piece and board[r-2][c+2] == piece and board[r-3][c+3] == piece: #if 4 pieces in a row negatively sloped
                return True

#create evaluation_window function which looks for "windows" of pieces in a row, and assigns each window a "priority score" that helps AI decide what would be the ideal
#move to make based on its "priority score"
def evaluate_window(window, piece):
    score = 0
    opp_piece = PLAYER_PIECE
    if piece == PLAYER_PIECE:
        opp_piece = AI_PIECE

    if window.count(piece) == 4: #4 of the same pieces in a row is highest priority
        score += 100
    elif window.count(piece) == 3 and window.count(EMPTY) == 1: #second highest priority is having 3 in a row with 1 open in window so you can make 4 in a row
        score += 5
    elif window.count(piece) == 2 and window.count(EMPTY) == 2: #third highest in a row is having 2 in a row with 2 open nearby so you can make 3 in a row
        score += 2

    if window.count(opp_piece) == 3 and window.count(EMPTY) == 1: #if opponent has 3 in a row with an open near by we want to play "defence" and deny their win
        score -= 4

    return score


#uses evaluate_window to evaluate all of the possible moves the AI could make in all directions to find which would return the most worthwhile score so AI can make that worthy move
def score_position(board, piece):
    score = 0

    #center column score
    center_array = [int(i) for i in list(board[:, COL_COUNT//2])]
    center_count = center_array.count(piece)
    score += center_count*3

    #horizontal score
    for r in range(ROW_COUNT):
        row_array = [int(i) for i in list(board[r,:])]
        for c in range(COL_COUNT-3):
            window = row_array[c:c+WINDOW_LENGTH]
            score += evaluate_window(window, piece)

    #vertical score
    for c in range(COL_COUNT):
        col_array = [int(i) for i in list(board[:,c])]
        for r in range(ROW_COUNT-3):
            window = col_array[r:r+WINDOW_LENGTH]
            score += evaluate_window(window, piece)

    #positive sloped diagonal score
    for r in range(ROW_COUNT-3):
        for c in range(COL_COUNT-3):
            window = [board[r+i][c+i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece)

    #negative sloped diagonal score
    for r in range(ROW_COUNT-3):
        for c in range(COL_COUNT-3):
            window = [board[r+3-i][c+i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece)

    return score

def is_terminal_node(board): #responsible for determining whether there is a terminate (game-ending) state on the board
    return winning_move(board, PLAYER_PIECE) or winning_move(board, AI_PIECE) or len(get_valid_locations(board)) == 0

#implementing minimax algorithm for AI to find optimal move.
#function starts by seeing if the game board is in a terminal state anywhere or if depth of seach is 0, if either true, then returns the score of current game state
def minimax(board, depth, alpha, beta, maximizingPlayer): #
    valid_locations = get_valid_locations(board)
    is_terminal = is_terminal_node(board)
    if depth == 0 or is_terminal:
        if is_terminal:
            if winning_move(board, AI_PIECE):
                return (None, 1000000000000000)
            elif winning_move(board, PLAYER_PIECE):
                return (None, -1000000000000000)
            else: #game over no valid moves left
                return (None, 0)
        else: #depth is 0
            return (None, score_position(board, AI_PIECE))
    if maximizingPlayer: 
        value = -math.inf
        column = random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, AI_PIECE)
            new_score = minimax(b_copy, depth-1, alpha, beta, False)[1]
            if new_score > value:
                value = new_score
                column = col
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return column, value
        
    else: #minimizing player
        value = math.inf
        column = random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, PLAYER_PIECE)
            new_score = minimax(b_copy, depth-1, alpha, beta, True)[1]
            if new_score < value:
                value = new_score
                column = col
            beta = min(beta, value)
            if alpha >= beta:
                break
        return column, value


def get_valid_locations(board):
    valid_locations = []
    for col in range(COL_COUNT):
        if is_valid_location(board, col):
            valid_locations.append(col)
    return valid_locations

def pick_best_move(board, piece): #selects the best move for the AI to make out of the valid locations retrieved from get_valid_locations function based off scores calculated with score_position
    valid_locations = get_valid_locations(board)
    best_score = -80000
    best_col = random.choice(valid_locations)
    for col in valid_locations:
            row = get_next_open_row(board, col)
            temp_board = board.copy()
            drop_piece(temp_board, row, col, piece)
            score = score_position(temp_board, piece)
            if score > best_score:
                best_score = score
                best_col = col            
    return best_col


#defining function to draw the board graphics using pygame
def draw_board(board):
    for c in range(COL_COUNT):
        for r in range(ROW_COUNT):
            pygame.draw.rect(screen, BLUE, (c*SQUARESIZE, r*SQUARESIZE+SQUARESIZE, SQUARESIZE, SQUARESIZE)) #drawing the blue board rectangle outline for each column/row
            pygame.draw.circle(screen, BLACK, (int(c*SQUARESIZE+SQUARESIZE/2), int(r*SQUARESIZE+SQUARESIZE+SQUARESIZE/2)), RADIUS) #making a black hole circle for pieces to fall into
            
    for c in range(COL_COUNT):
        for r in range(ROW_COUNT):
            if board[r][c] == PLAYER_PIECE: #if player piece is dropped it makes the empty black board hole red
                pygame.draw.circle(screen, RED, (int(c*SQUARESIZE+SQUARESIZE/2), height-int(r*SQUARESIZE+SQUARESIZE/2)), RADIUS)
            elif board[r][c] == AI_PIECE: #if AI piece is dropped it makes the empty black board hole yellow
                pygame.draw.circle(screen, YELLOW, (int(c*SQUARESIZE+SQUARESIZE/2), height-int(r*SQUARESIZE+SQUARESIZE/2)), RADIUS)
    pygame.display.update()

board = create_board()
print_board(board)
game_over = False

#initializing pygame and creating connect4 board and graphics, along with font
pygame.init()
SQUARESIZE = 100
width = COL_COUNT*SQUARESIZE
height = (ROW_COUNT+1)*SQUARESIZE
size = (width, height)
RADIUS = int(SQUARESIZE/2 - 5)
screen = pygame.display.set_mode(size)
draw_board(board)
pygame.display.update()
myfont = pygame.font.SysFont("monospace", 75)
turn = random.randint(PLAYER, AI)

#creating while loop for pygame event recognition, alternating turns and game_over condition
while not game_over:
    #use pygame events to dictate what occurs when quitting, moving mouse around, or pressing mouse button
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
        
        if event.type == pygame.MOUSEMOTION:
            pygame.draw.rect(screen, BLACK, (0,0, width, SQUARESIZE))
            posx = event.pos[0]
            if turn == PLAYER: #if turn is player then when the mouse is moving the red piece moves around with the mouse
                pygame.draw.circle(screen, RED, (posx, int(SQUARESIZE/2)), RADIUS)
            else: #if not player turn then piece moving around with mouse is yellow circle
                pygame.draw.circle(screen, YELLOW, (posx, int(SQUARESIZE/2)), RADIUS)
        pygame.display.update()
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            pygame.draw.rect(screen, BLACK, (0,0, width, SQUARESIZE))
            #dictate what happens if player 1's turn
            if turn == PLAYER:
                posx = event.pos[0]
                col = int(math.floor(posx/SQUARESIZE))
                
                if is_valid_location(board, col): #if location is valid
                    row = get_next_open_row(board, col) #sets row to next open row
                    drop_piece(board, row, col, PLAYER_PIECE) #drops player piece in that same next open row

                    if winning_move(board, PLAYER_PIECE): #if the move was a winning move then displays congratulations
                        label = myfont.render("Player 1 wins!!", 1, RED)
                        screen.blit(label, (40,10)) 
                        game_over = True
                        
                    #if not winning move then alternates turn
                    turn += 1
                    turn = turn % 2

                    print_board(board)
                    draw_board(board)
            
    #dictate what happens if player 2's turn
    if turn == AI and not game_over:


        col, minimax_score = minimax(board, 5, -math.inf, math.inf, True)

        if is_valid_location(board, col): #ensuring AI selects a column which isn't already full
            row = get_next_open_row(board, col) #finding row which is next open in that columb
            drop_piece(board, row, col, AI_PIECE) #drops AI piece in that spot

            if winning_move(board, AI_PIECE): #runs congratulatory message if needed
                label = myfont.render("Player 2 wins!!", 1, YELLOW)
                screen.blit(label, (40,10)) 
                game_over = True

            print_board(board)
            draw_board(board)
            
            turn += 1
            turn = turn % 2

    if game_over:
        pygame.time.wait(3000) #3 second delay for celebratory screen
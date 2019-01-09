#!/usr/bin/env python
from isolation import Board, game_as_text
from random import randint


# This file is your main submission that will be graded against. Do not
# add any classes or functions to this file that are not part of the classes
# that we want.


class OpenMoveEvalFn:

    def score(self, game, maximizing_player_turn=True):
        """Score the current game state

        Evaluation function that outputs a score equal to how many
        moves are open for AI player on the board minus how many moves
        are open for Opponent's player on the board.
        Note:
            1. Be very careful while doing opponent's moves. You might end up
               reducing your own moves.
            3. If you think of better evaluation function, do it in CustomEvalFn below.

            Args
                param1 (Board): The board and game state.
                param2 (bool): True if maximizing player is active.

            Returns:
                float: The current state's score. MyMoves-OppMoves.

            """

        # TODO: finish this function!
        MyMoves = len(game.get_legal_moves())
        OppMoves = len(game.get_opponent_moves())
        if maximizing_player_turn == True:
            eval_score = MyMoves - OppMoves
        else:
            eval_score = OppMoves - MyMoves
        return eval_score


class CustomEvalFn:
    def __init__(self):
        pass

    def score(self, game, maximizing_player_turn=True):
        """Score the current game state

        Custom evaluation function that acts however you think it should. This
        is not required but highly encouraged if you want to build the best
        AI possible.

        Args
            game (Board): The board and game state.
            maximizing_player_turn (bool): True if maximizing player is active.

        Returns:
            float: The current state's score, based on your own heuristic.

        """

        # TODO: finish this function!
        if maximizing_player_turn:
            MyMoves = len(game.get_legal_moves())
            OppMoves = len(game.get_opponent_moves())
            if game.move_count <= 2:
                return MyMoves - 2.0* OppMoves
            elif MyMoves == 0 and not OppMoves == 0:
                return float("-inf")
            elif OppMoves == 0 and not MyMoves == 0:
                return float("inf")
            elif OppMoves == MyMoves and MyMoves == 0:
                return -5.
            else:
                return MyMoves - 2.0* OppMoves

        else:
            MyMoves = len(game.get_opponent_moves())
            OppMoves = len(game.get_legal_moves())
            if game.move_count <= 2:
                return MyMoves - 2.0*OppMoves
            elif MyMoves == 0 and not OppMoves == 0:
                return float("-inf")
            elif OppMoves == 0 and not MyMoves == 0:
                return float("inf")
            elif OppMoves == MyMoves and MyMoves == 0:
                return -5.
            else:
                return MyMoves - 2.0*OppMoves


class CustomPlayer:
    # TODO: finish this class!
    """Player that chooses a move using your evaluation function
    and a minimax algorithm with alpha-beta pruning.
    You must finish and test this player to make sure it properly
    uses minimax and alpha-beta to return a good move."""

    def __init__(self, search_depth=12, eval_fn=CustomEvalFn()):
        """Initializes your player.

        if you find yourself with a superior eval function, update the default
        value of `eval_fn` to `CustomEvalFn()`

        Args:
            search_depth (int): The depth to which your agent will search
            eval_fn (function): Utility function used by your agent
        """
        self.eval_fn = eval_fn
        self.search_depth = search_depth
        self.pos = None
        self.center = (3, 3, False)
        self.opening_moves = [(5, 6, False), (5, 5, False), (6, 5, False), (2, 0, False), (1, 1, False), (0, 2, False),
                              (5, 0, False),(5, 1, False),(6, 1, False), (1, 5, False), (1, 6, False), (0, 4, False), (0, 5, False), (1, 0, False), (0, 1, False)]
        self.reflection_flag = False

    def move(self, game, legal_moves, time_left):
        """Called to determine one move by your agent

            Note:
                1. Do NOT change the name of this 'move' function. We are going to call
                the this function directly.
                2. Change the name of minimax function to alphabeta function when
                required. Here we are talking about 'minimax' function call,
                NOT 'move' function name.
                Args:
                game (Board): The board and game state.
                legal_moves (dict): Dictionary of legal moves and their outcomes
                time_left (function): Used to determine time left before timeout

            Returns:
                tuple: best_move
            """
        self.time_left = time_left
        mymoves = len(legal_moves)
        num_moves = game.move_count
        if num_moves == 0 and game.__player_1__ == self:
            return self.center
        if num_moves == 1 and game.__player_2__ == self and self.center in legal_moves:
            return self.center
        if num_moves == 1 and game.__player_2__ == self:
            for i in [x for x in self.opening_moves if x in legal_moves]:
                return i
        #reflect

        if self.reflection_flag == False and num_moves % 2 == 0 and game.__player_1__ == self:
            move = game.__last_queen_move__[game.__inactive_players_queen__]
            r, c, _ = move
            if (3 + (3 - r), 3 + (3 - c), False) not in legal_moves:
                self.reflection_flag = True
            else:
                return (3 + (3 - r), 3 + (3 - c), False)

        if mymoves == 0:

            return None
        elif mymoves == 1:
            return legal_moves[0]
        elif len(game.get_opponent_moves()) == 0:
            return legal_moves[randint(0,len(legal_moves)-1)]
        #best_move, utility = self.minimax(game, time_left, depth=self.search_depth)
        #best_move, utility = self.alphabeta(game, time_left, depth=self.search_depth)
        best_move, utility = self.IDAlphabeta(game, time_left, depth=self.search_depth)
        self.pos = best_move
        return best_move

    def utility(self, game, maximizing_player=True):
        """Can be updated if desired. Not compulsory. """

        return self.eval_fn.score(game, maximizing_player)

    def minimax(self, game, time_left, depth=5, maximizing_player=True):
        """Implementation of the minimax algorithm

        Args:
            game (Board): A board and game state.
            time_left (function): Used to determine time left before timeout
            depth: Used to track how deep you are in the search tree
            maximizing_player (bool): True if maximizing player is active.

        Returns:
            (tuple, int): best_move, val
        """
        # TODO: finish this function!
        #print(self.time_left())
        best_move = None
        if depth == 0 or len(game.get_legal_moves()) == 0 or self.time_left() < 20:
            best_val = self.eval_fn.score(game, maximizing_player)
        else:
            player_moves = game.get_legal_moves()
            move_val_dict = {}
            for i in range(len(player_moves)):
                game_copy = game.copy()
                game_copy.__apply_move__(player_moves[i])
                move_val_dict[player_moves[i]] = self.minimax(game_copy, time_left, depth - 1, not maximizing_player)[1]
            if maximizing_player:
                best_val = float("-inf")
                for i in move_val_dict.keys():
                    if move_val_dict[i] > best_val:
                        best_val = move_val_dict[i]
                        best_move = i
            else:
                best_val = float("inf")
                for i in move_val_dict.keys():
                    if move_val_dict[i] < best_val:
                        best_val = move_val_dict[i]
                        best_move = i
        return best_move, best_val

    def alphabeta(self, game, time_left, depth= float("inf"), alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implementation of the alphabeta algorithm

        Args:
            game (Board): A board and game state.
            time_left (function): Used to determine time left before timeout
            depth: Used to track how deep you are in the search tree
            alpha (float): Alpha value for pruning
            beta (float): Beta value for pruning
            maximizing_player (bool): True if maximizing player is active.

        Returns:
            (tuple, int): best_move, val
        """
        # TODO: finish this function!

        best_val = self.eval_fn.score(game, maximizing_player)

        player_moves = game.get_legal_moves()


        if depth == 0 or self.time_left() < 20 or len(player_moves) == 0:
            return None, best_val
        best_move=None

        if len(player_moves) > 0:
            best_move = player_moves[randint(0, len(player_moves) - 1)]

        if maximizing_player:
            best_val = float("-inf")
            for i in player_moves:
                new_game, is_over, winner = game.forecast_move(i)
                #curr_val = self.eval_fn.score(new_game, maximizing_player)
                if is_over:
                    if winner == game.__active_players_queen__:
                        return (i, float("inf"))
                    else:
                        return (None, float("-inf"))
                else:
                    #my_val = self.eval_fn.score(new_game, maximizing_player)
                    my_move, my_val = self.alphabeta(new_game, time_left, depth - 1, alpha, beta, False)

                if my_val > best_val:
                    best_val = my_val
                    alpha = my_val
                    best_move = i

                if alpha >= beta:
                    break
        else:
            best_val = float("inf")
            for i in player_moves:
                new_game, is_over, winner = game.forecast_move(i)
                if is_over:
                    if winner == game.__inactive_players_queen__:
                        return (i, float("inf"))
                    else:
                        return (None, float("-inf"))
                else:
                    #utility = self.eval_fn.score(new_game, maximizing_player)
                    my_move, my_val = self.alphabeta(new_game, time_left, depth - 1, alpha, beta, True)

                if my_val < best_val:
                    best_val = my_val
                    beta = my_val
                    best_move = i

                if alpha >= beta:
                    break

        return best_move, best_val

    def IDAlphabeta(self, game, time_left, depth, alpha = float("-inf"), beta = float("inf"), maximizing_player = True):

        moves = game.get_legal_moves()

        if len(moves) == 0:
            return None,  self.eval_fn.score(game, maximizing_player)

        best_move = moves[randint(0, len(moves) - 1)]
        utility = self.eval_fn.score(game, maximizing_player)

        #iteraative deepening
        for i in range(1,depth):
            #print "time left", self.time_left()
            #hold best val for current depth
            if self.time_left() < 20 or utility == float("inf"): #not enough time left for another iteration
                return  best_move , utility
            max_move, util = self.alphabeta(game, time_left, i, alpha, beta, maximizing_player)

            if not util == float("-inf") and util > utility:
                best_move = max_move
                utility = util
        #print(best_move)
        return best_move, utility

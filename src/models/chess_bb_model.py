import chess
import chess.engine
from stockfish import Stockfish


class ChessBBModel:

    def __init__(self, env, model_path):
        self.model_path = model_path
        self.env = env
        self.stockfish = Stockfish(path=self.model_path, depth=18, parameters={"Threads": 2, "Minimum Thinking Time": 30})

    def predict(self, x):
        fen = self.env.from_array_to_fen(x)
        board = chess.Board(fen)

        self.stockfish = Stockfish(path=self.model_path, depth=18,
                                   parameters={"Threads": 2, "Minimum Thinking Time": 30})

        try:
            self.stockfish.set_fen_position(board.fen())
            action = self.stockfish.get_best_move_time(1000)
        except:
            action = ''

        return action
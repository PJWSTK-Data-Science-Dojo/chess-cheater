# Chess Cheater Classification Project (WiP)

## Project purpose

The purpose of this project is to create a model capable of finding suspicious moves in a chess game. Suspicious move is defined as a move that human is unlikely to do but is also very accurate so it is suspected that this move was done by or with help of the chess engine.

## Data sources

Initial dataset with chess games is sourced from kaggle: https://www.kaggle.com/datasets/arevel/chess-games

For finding more games either to broaden the dataset or to find games played by cheaters it is planned to use in the future lichess db and API and chesscom API

Lichess database: https://database.lichess.org/

Lichess API: https://lichess.org/api

Chesscom: https://www.chess.com/news/view/published-data-api#game-results

To generate engine moves two approaches were made:

* Stockfish moves were generated in games stockfish on stockfish
* For each position in games between humans a stockfish response were generated (stockfish moves that were identical to human move in the position were discarded)
from flask import Flask, jsonify, request
from flask_cors import CORS

from codenames import generate_game_clues, generate_board_words

app = Flask(__name__)
CORS(app, origins="*", supports_credentials=True)


@app.route("/", methods=["GET"])
def new_game():
    first = request.args.get("first")
    if not first:
        first = "red"
    cards = generate_board_words(first)
    return jsonify(cards)


@app.route("/get-clue", methods=["POST"])
def get_clue():
    req_data = request.get_json()
    turn = request.args.get("turn")
    player_words = []
    other_words = []
    for data in req_data:
        if data["selected"] == False:
            if data["type"] == turn:
                player_words.append(data["word"])
            else:
                other_words.append(data["word"])
    return generate_game_clues(player_words, other_words)


if __name__ == "__main__":
    app.run()

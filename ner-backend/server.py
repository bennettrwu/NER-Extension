import json
from flask import Flask, request
from flask_cors import CORS, cross_origin
from inference import run_ner_paragraph

entity_map = {
    'LOC': 'Location',
    'PER': 'Person',
    'ORG': 'Organization',
    'MISC': 'Miscellaneous'
}


def find_replace_positions(tokens: list[str], labels: list[str]) -> list[dict]:
    entities = []
    word_counts = {}

    entity_words = []
    for token, label in zip(tokens, labels):
        if label != 'O':
            entity_words.append(token)

    entity_words = set(entity_words)

    for token, label in zip(tokens, labels):
        for word in entity_words:
            if word in token:
                if word_counts.get(word):
                    word_counts[word] += 1
                else:
                    word_counts[word] = 1

        if label != 'O':
            entities.append({
                'str': token,
                'type': entity_map[label.split('-')[1]],
                'n': word_counts[label]
            })

    return entities


PORT = 8080

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/ner', methods=['POST'])
@cross_origin()
def index():
    paragraph = request.data.decode('utf-8')
    tokens, labels = run_ner_paragraph(paragraph)

    return json.dumps(find_replace_positions(tokens, labels))


if __name__ == '__main__':
    from waitress import serve
    print(f'Webserver listening on port: {PORT}')
    serve(app, port=PORT, host='0.0.0.0')

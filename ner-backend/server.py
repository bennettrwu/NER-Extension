from flask import Flask, request
from flask_cors import CORS, cross_origin

PORT = 8080

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/ner', methods=['POST'])
@cross_origin()
def index():
    print(request.data.decode('utf-8'))
    return 'Hello, World!'

if __name__ == '__main__':
    from waitress import serve
    print(f'Webserver listening on port: {PORT}')
    serve(app, port=PORT, host='0.0.0.0')

import os
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from io import BytesIO
from tokenization import FullTokenizer

class ReqHandler(BaseHTTPRequestHandler):

    tok:FullTokenizer=None

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'Dobert Tokenizer')

    def do_POST(self):
        if (self.headers['Content-Type'] != 'application/json'):
            self.send_error(400)
        else:
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length)
            body = json.loads(body)
            if ('question' not in body):
                return self.send_error(400)
            self.send_response(200)
            self.end_headers()
            response = BytesIO()

            tokens = self.tok.tokenize(body['question'])

            response.write(str.encode(body['question']))
            self.wfile.write(response.getvalue())

class Server(object):

    def __init__(self, port):
        self.port = port
        self.host = '192.168.43.129'
        ReqHandler.tok = FullTokenizer(os.path.abspath('/home/daniel/Desktop/DoctorRobert/Backend/Cluster/DocNet/tokenizer/vocab.txt'))
        self.httpserver = HTTPServer((self.host, self.port), ReqHandler)

    def start(self):
        print('Server is listening to', self.host + ':' + str(self.port))
        self.httpserver.serve_forever()

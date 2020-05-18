import os
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from io import BytesIO
import numpy as np
from FaissTopK import FaissTopK

class ReqHandler(BaseHTTPRequestHandler):

    topK: FaissTopK = None


    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'Dobert vec similarity search')

    def do_POST(self):
        if (self.headers['Content-Type'] != 'application/json'):
            self.send_error(400)
        else:
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length)
            body = json.loads(body)
            if ('embedding' not in body):
                return self.send_error(400)
            self.send_response(200)
            self.end_headers()
            embedding = np.array(body['embedding'], dtype=np.float)

            topk=5
            if(topk in body):
                topk=body['topk']

            answer = self.topK.predict(embedding, topk=topk)

            response = BytesIO()
            response.write(str.encode(json.dumps(answer)))
            self.wfile.write(response.getvalue())

class Server(object):

    def __init__(self, port, embedding_file):
        self.port = port
        self.host = ''
        ReqHandler.topK = FaissTopK(embedding_file)
        self.httpserver = HTTPServer((self.host, self.port), ReqHandler)

    def start(self):
        print('Server is listening to', self.host + ':' + str(self.port))
        self.httpserver.serve_forever()
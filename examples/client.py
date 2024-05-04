import socket
import sys
import json
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--host', type=str, default="localhost")
parser.add_argument('--port', type=int, default=27311)

args = parser.parse_args()
HOST, PORT = args.host, args.port

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.connect((HOST, PORT))
    seq_id = 0
    while True:
        instruction = input("user: ")
        if instruction == "":
            seq_id = -1
        data_dict = {'seq_id': seq_id, 'data': instruction}
        sock.sendall(json.dumps(data_dict).encode("utf-8"))
        if seq_id == -1:
            break

        received = str(sock.recv(1048576), "utf-8")
        received = json.loads(received)
        print(f"assistant: {received['data']}")

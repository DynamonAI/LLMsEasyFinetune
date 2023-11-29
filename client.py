import socket
import sys
import json

HOST, PORT = "localhost", 27311

# instruction = "How far away between moon and earth?"

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

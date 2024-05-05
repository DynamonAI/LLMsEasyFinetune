import socket
import sys
import json
import argparse

import streamlit as st


parser = argparse.ArgumentParser()
parser.add_argument('--host', type=str, default="localhost")
parser.add_argument('--port', type=int, default=27311)

args = parser.parse_args()
HOST, PORT = args.host, args.port


def send_mesg(instruction):
    try:
        seq_id = 0
        sock.connect((HOST, PORT))
        if instruction == "":
            seq_id = -1
        data_dict = {'seq_id': seq_id, 'data': instruction}
        sock.sendall(json.dumps(data_dict).encode("utf-8"))
    
        received = str(sock.recv(1048576), "utf-8")
        received = json.loads(received)['data']
        # print(f"assistant: {received['data']}")
    except ConnectionRefusedError:
        received = "Connection Refused Error: You have to deploy your model first!"
    return received


st.title("LLMsEasyFinetune Web")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Say Something")
if prompt:
    with st.chat_message("User"):
       st.markdown(prompt)
       st.session_state.messages.append({"role": "User", "content": prompt})
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        received = send_mesg(prompt)
        with st.chat_message("Assistant"):
            st.markdown(received)
            st.session_state.messages.append({"role": "Assistant", "content": received})

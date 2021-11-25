import socket
import argparse
import pickle

import torch
import torch.nn as nn
from train_tools.models.layer3 import Layer3

#접속하고 싶은 ip와 port를 입력받는 클라이언트 코드를 작성해보자.
# 접속하고 싶은 포트를 입력한다.
host = '192.168.0.8'
port = 9999

if __name__ == '__main__':
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # 지정한 host와 prot를 통해 서버에 접속합니다.
    client_socket.connect((host, port))

    try:
        model = torch.load('./checkpoint/client_ckpt.pth')
    except:
        print("[socket_client.py] there is no trained checkpoint")
        # train setups
        model = Layer3()
        model = model.state_dict()

    model = {k: v.detach().cpu().numpy().tolist() for k, v in model.items()}
    model = pickle.dumps(model)

    client_socket.sendall(len(model).to_bytes(4, byteorder='little'))  

    client_socket.sendall(model)

    del model    

    total_length = int.from_bytes(client_socket.recv(4),'little')
    receive_data = []
    recv_length = 0

    try:
        while total_length > recv_length:
            packet = client_socket.recv(4096)   
            receive_data.append(packet)
            recv_length += len(packet)
    except:
        pass

    receive_data = pickle.loads(b"".join(receive_data))
    receive_data = {k: torch.tensor(v) for k, v in receive_data.items()}
    torch.save(receive_data, './checkpoint/client_ckpt.pth')
    print("데이터를 받았고 저장했습니다.")
    # print("받은 데이터는 \"", receive_data, "\" 입니다.")

    # 소켓을 닫는다.
    client_socket.close()
    print("접속을 종료합니다.")
 

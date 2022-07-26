import socket
import argparse
import threading
import time
import pickle
import numpy as np

import torch
import torch.nn as nn
import torchvision

host = "192.168.0.8"
port = 9999


C_IP_TO_IDX={}
C_IDX_DBIT={}
C_IDX_DATA={}
C_IDX_SEND_ACK={}


SEND_DATA="DONE!!!!!"

def handler(signo, frame):
    raise RuntimeError

def handle_client(client_socket, addr):
    
    global C_IP_TO_IDX
    global C_IDX_DBIT
    global C_IDX_DATA
    global C_IDX_SEND_ACK
    
    
    print("접속한 클라이언트의 주소 입니다. : ", addr)
    total_length = int.from_bytes(client_socket.recv(4),'little')
    data = []
    recv_length = 0

    try:
        while total_length > recv_length:
            # client_socket.settimeout(30)

            packet = client_socket.recv(4096)   
            data.append(packet)
            recv_length += len(packet)
    except:
        pass

    data = pickle.loads(b"".join(data))

    IDX=len(C_IP_TO_IDX.keys())
    C_IP_TO_IDX[addr[0]]=IDX
    C_IDX_DBIT[IDX]=1
    C_IDX_DATA[IDX] = data
    
    while(True):
        if len(C_IDX_DBIT.values()) == 3:

            # get averaged weights
            average_dict = {}
            for client in C_IDX_DATA.keys():
                data = C_IDX_DATA[client]

                for name, weight in data.items():
                    if name not in average_dict:
                        average_dict[name] = (np.array(weight) / 3)
                    else:
                        average_dict[name] += (np.array(weight) / 3)


            average_dict = {k: v.tolist() for k, v in average_dict.items()}
            average_dict = pickle.dumps(average_dict)

            client_socket.sendall(len(average_dict).to_bytes(4, byteorder='little'))  
            client_socket.sendall(average_dict)
            print("전송이 완료되었습니다. : ", addr)
            C_IDX_SEND_ACK[IDX]=1
            break

    while(True):
        if len(C_IDX_SEND_ACK.values()) == 3:
            if IDX == 0:
                average_dict = pickle.loads(average_dict)
                average_dict = {k: torch.tensor(v) for k, v in average_dict.items()}

                torch.save(average_dict, './checkpoint/server_ckpt.pth')

                C_IP_TO_IDX={}
                C_IDX_DBIT={}
                C_IDX_DATA={}
                C_IDX_SEND_ACK={}
                break
                
            else:
                break

    time.sleep(2)
    client_socket.close()

def accept_func():
    global server_socket
    #IPv4 체계, TCP 타입 소켓 객체를 생성
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #포트를 사용 중 일때 에러를 해결하기 위한 구문
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    #ip주소와 port번호를 함께 socket에 바인드 한다.
    #포트의 범위는 1-65535 사이의 숫자를 사용할 수 있다.
    server_socket.bind((host, port))

    #서버가 최대 5개의 클라이언트의 접속을 허용한다.
    server_socket.listen(5)

    # rounds
    for r in range(30):
        try:
            #클라이언트 함수가 접속하면 새로운 소켓을 반환한다.
            client_socket, addr = server_socket.accept()
        except KeyboardInterrupt:
            server_socket.close()
            print("Keyboard interrupt")

        print("클라이언트 핸들러 스레드로 이동 됩니다.")
        #accept()함수로 입력만 받아주고 이후 알고리즘은 핸들러에게 맡긴다.
        t = threading.Thread(target=handle_client, args=(client_socket, addr))
        t.daemon = True
        t.start()


if __name__ == '__main__':
    #parser와 관련된 메서드 정리된 블로그 : https://docs.python.org/ko/3/library/argparse.html
    #description - 인자 도움말 전에 표시할 텍스트 (기본값: none)
    #help - 인자가 하는 일에 대한 간단한 설명.
    parser = argparse.ArgumentParser(description="\nJoo's server\n-p port\n")
    parser.add_argument('-p', help="port")

    args = parser.parse_args()
    try:
        port = int(args.p)
    except:
        pass
    accept_func()


if [ -f ./checkpoint/server_ckpt.pth ]; then 
    rm './checkpoint/server_ckpt.pth'
fi

python3 send_data.py

python3 socket_server.py

python3 test_server.py

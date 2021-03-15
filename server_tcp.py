#!/usr/bin/env python
# -*- coding:utf-8 -*-

# #执行客户端发送过来的命令，并把执行结果返回给客户端
import socket, traceback, subprocess

host = ''
port = 51888

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

s.bind((host, port))
s.listen(1)

while 1:
    try:
        client_socket, client_addr = s.accept()
    except Exception as e:
        traceback.print_exc()
        continue


    try:
        print('From host:', client_socket.getpeername())
        while 1:
            command = client_socket.recv(4096)
            if not len(command):
                break

            print(client_socket.getpeername()[0] + ':' + str(command))

            a = 2

            b = a * int(command)
            print(b)

            # # 执行客户端传递过来的命令
            # handler = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
            # output = handler.stdout.readlines()
            # if output is None:
            #     output = []
            #
            # for one_line in output:
            #     client_socket.sendall(one_line)
            #     client_socket.sendall(bytes('\n','UTF-8'))
            #
            # client_socket.sendall(bytes('ok','UTF-8'))

    except Exception as e:
        traceback.print_exc()

    try:
        client_socket.close()
    except Exception as e:
        traceback.print_exc()


from socket import *
import _thread
def getData(tcpCliSock):
    while True:
        try:
            a = 10
        except:
            break
    tcpCliSock.close()
    _thread.stop()

if __name__ == '__main__':
    HOST = ""
    PORT = 8588
    BUFSIZ = 1024    #缓冲区大小
    ADDR = (HOST,PORT)
    tcpSerSock = socket(AF_INET,SOCK_STREAM)
    tcpSerSock.bind(ADDR)
    tcpSerSock.listen(60)    #连接被转接或者被拒绝之前，传入连接请求的最大数
    while True:
        tcpCliSock, addr = tcpSerSock.accept()
        print("waiting for connect ...")
        print("...connect from:", addr)
        _thread.start_new_thread(getData, (tcpCliSock, ))
        continue

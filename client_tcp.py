#!/usr/bin/env python
# -*- coding:utf-8 -*-

# #给server端发送命令
import socket, sys, traceback



for i in range(1):
    host = '127.0.0.1'
    port = 51888

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((host, port))
    except Exception as e:
        msg = traceback.format_exc()
        print('连接错误:', msg)

    input_command = '-1'
    s.send(bytes(input_command,'UTF-8'))


    # 利用shutdown()函数使socket双向数据传输变为单向数据传输
    # 该参数表示了如何关闭socket。具体为：0表示禁止将来读；1表示禁止将来写；2表示禁止将来读和写
    s.shutdown(1)
    print('发送完成.')
    print('收到内容：\n')
    # while 1:
    #     buff = s.recv(4096)
    #     if not len(buff):
    #         break
    #
    #     sys.stdout.write(str(buff, encoding="utf8"))


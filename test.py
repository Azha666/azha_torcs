import signal
import time


def set_timeout(num):
    def wrap(func):
        def handle(signum, frame):  # 收到信号 SIGALRM 后的回调函数，第一个参数是信号的数字，第二个参数是the interrupted stack frame.
            raise RuntimeError

        def to_do(*args):
            try:
                signal.signal(signal.SIGALRM, handle)  # 设置信号和回调函数
                signal.alarm(num)  # 设置 num 秒的闹钟
                print('start alarm signal.')
                r = func(*args)
                print('close alarm signal.')
                signal.alarm(0)  # 关闭闹钟
                return r
            except RuntimeError as e:
                return "超时啦"
        return to_do

    return wrap


@set_timeout(1)  # 限时 2 秒超时
def connect():  # 要执行的函数
    time.sleep(3)  # 函数执行时间，写大于2的值，可测试超时
    print('完成')
    return "完成"


if __name__ == '__main__':
    print(1)
    a = connect()
import threading


import os



def capture(i):


    i += 1


    threading.Timer(1.0, capture, [i]).start()


    fill = str(i).zfill(5)


    os.system("scrot scrot-%s.jpg" % fill)


    # os.system("streamer -o streamer-%s.jpeg -s 320x240 -j 100" % fill)



capture(0)
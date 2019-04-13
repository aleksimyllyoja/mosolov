from timeit import default_timer as timer
import mido
import math
import time

print(mido.get_output_names())
port = mido.open_ioport('UM-ONE')
first_seen = False
last = 0
bpms = []

for msg in port:
    if msg.type == 'clock':

        if not first_seen:
            last = time.time_ns() / (10 ** 9)
            first_seen = True
            continue

        now = time.time_ns() / (10 ** 9)
        t = now-last
        bpms.append(2500/(t*1000))
        last = now
        BPM = sum(bpms)/len(bpms)
        print(round(BPM), len(bpms), t)
        bpms = bpms[-40:]

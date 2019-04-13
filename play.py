from timeit import default_timer as timer
from pynput.keyboard import Key, Listener

import os
import mido
import threading
import queue

from magenta.models.music_vae import configs
from magenta.models.music_vae import TrainedModel

PLAYING = False
SAMPLE_INDEX = 0
NUM_OUTPUTS = 8
TEMPERATURE = 0.5
TRANSPOSE = 0
BPM = 120

print(mido.get_output_names())
port = mido.open_ioport('UM-ONE')
#inport = mido.open_input('UM-ONE')

d = "~/repositories/mosolov/cat-mel_2bar_big.tar"
checkpoint_dir_or_path = os.path.expanduser(d)
config = configs.CONFIG_MAP['cat-mel_2bar_big']

def keypress_handler(key):
    global PLAYING, SAMPLE_INDEX, TRANSPOSE, BPM

    if key == Key.space:
        print("PLAYING =", not PLAYING)
        PLAYING = not PLAYING

    if key == Key.left:
        SAMPLE_INDEX -= 1
        print("SAMPLE_INDEX", SAMPLE_INDEX, len(results))
    if key == Key.right:
        SAMPLE_INDEX += 1
        print("SAMPLE_INDEX", SAMPLE_INDEX, len(results))
    if str(key) == "'u'":
        TRANSPOSE += 5
        print("TRANSPOSE =", TRANSPOSE)
    if str(key) == "'d'":
        TRANSPOSE -= 5
        print("TRANSPOSE =", TRANSPOSE)
    if str(key) == "'b'":
        BPM += 5
        print("BPM =", BPM, ((BPM/60.0)/2.0))
    if str(key) == "'v'":
        BPM -= 5
        print("BPM =", BPM, ((BPM/60.0)/2.0))

def play_note(note):
    msg = mido.Message(
        'note_on',
        note=note.pitch+TRANSPOSE,
        velocity=note.velocity
    )
    port.send(msg)

def stop_note(note):
    msg = mido.Message(
        'note_off',
        note=note.pitch+TRANSPOSE,
        velocity=note.velocity
    )
    port.send(msg)

def play():
    start = timer()

    index = 0
    note_on = False

    print("PLAYING")
    while True:
        notes = results[SAMPLE_INDEX].notes
        if not PLAYING:
            continue

        if index==len(notes):
            index = 0
            start = timer()

        note = notes[index]
        time = (timer()-start)*((BPM/60.0)/2.0)

        if not note_on and time>=note.start_time and time<=note.end_time:
            play_note(note)
            note_on = True

        if time>=note.end_time:
            stop_note(note)
            index += 1
            note_on = False

def sync():
    global BPM

    first_seen = False
    last = 0
    bpms = []

    for msg in port:
        if msg.type == 'clock':

            if not first_seen:
                last = timer()
                first_seen = True
                continue

            now = timer()
            time = now-last
            bpms.append(2500/(time*1000))
            last = now
            BPM = sum(bpms)/len(bpms)
            print(round(BPM), len(bpms), time)
            bpms = bpms[-20:]

keyboard_listener = Listener(on_press=keypress_handler)
keyboard_listener.start()
#keyboard_listener.join()

model = TrainedModel(
    config,
    batch_size=min(8, NUM_OUTPUTS),
    checkpoint_dir_or_path=checkpoint_dir_or_path
)

results = model.sample(
    n=NUM_OUTPUTS,
    length=config.hparams.max_seq_len,
    temperature=TEMPERATURE
)

sync_thread = threading.Thread(name='sync', target=sync)
sync_thread.start()

play_thread = threading.Thread(name='play', target=play)
play_thread.start()

print("\n\n💿 READY 💿\n\n")

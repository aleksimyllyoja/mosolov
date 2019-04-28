from timeit import default_timer as timer
from pynput.keyboard import Key, Listener

import os
import mido
import threading
import time

import argparse

class STATE:
    PLAYING = False

    SAMPLE_INDEX = 0
    NOTE_INDEX = 0

    NUM_OUTPUTS = 8
    TEMPERATURE = 0.5
    TRANSPOSE = 0
    BPM = 120
    SEQUENCES = []
    PLAYBACK_INITIALIZED = False
    PORT = None
    MESSAGES = ""

    @classmethod
    def print(cls):
        vars = [n for n in dir(STATE)
            if not callable(getattr(STATE, n))
            and not n.startswith('_')
        ]

        os.system('cls' if os.name == 'nt' else 'clear')
        print('\n')
        template = "\t{0:40}{1:15}"
        for var in vars:
            if var != 'SEQUENCES':
                print(
                    template.format(var, str(getattr(cls, var)))
                )
        print(
            template.format('SEQUENCES LEN', str(len(cls.SEQUENCES)))
        )

    @classmethod
    def next_sample(cls):
        cls.SAMPLE_INDEX += 1
        cls.NOTE_INDEX = 0

        cls.SAMPLE_INDEX = cls.SAMPLE_INDEX % len(cls.SEQUENCES)

    @classmethod
    def last_sample(cls):
        cls.SAMPLE_INDEX -= 1
        cls.NOTE_INDEX = 0

        cls.SAMPLE_INDEX = cls.SAMPLE_INDEX % len(cls.SEQUENCES)

def keypress_handler(key):

    if key == Key.space:
        STATE.PORT.panic()
        STATE.PLAYING = not STATE.PLAYING

    if key == Key.left:  STATE.last_sample()
    if key == Key.right: STATE.next_sample()

    if str(key) == "'z'": STATE.TRANSPOSE -= 1
    if str(key) == "'x'": STATE.TRANSPOSE += 1

    if key == Key.up:   STATE.BPM += 1
    if key == Key.down: STATE.BPM -= 1

    if str(key) == "'q'":
        print("QUITTING..")
        STATE.PORT.panic()
        os._exit(1)

    STATE.print()

def play_note(note):
    msg = mido.Message(
        'note_on',
        note=note.pitch+STATE.TRANSPOSE,
        velocity=note.velocity
    )
    port.send(msg)

def stop_note(note):
    msg = mido.Message(
        'note_off',
        note=note.pitch+STATE.TRANSPOSE,
        velocity=note.velocity
    )
    port.send(msg)

def play():
    note_on = False
    STATE.PLAYBACK_INITIALIZED = True

    while True:
        if not STATE.PLAYING:
            continue

        notes = STATE.SEQUENCES[STATE.SAMPLE_INDEX].notes

        note = notes[STATE.NOTE_INDEX]
        nd = (note.end_time - note.start_time)*((60.0/STATE.BPM)*2.0)

        STATE.print()

        play_note(note)
        time.sleep(nd)
        stop_note(note)

        STATE.NOTE_INDEX += 1
        if STATE.NOTE_INDEX == len(notes):
            STATE.NOTE_INDEX = 0

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
            last_bpm = BPM
            BPM = round(sum(bpms)/len(bpms))
            if last_bpm != BPM:
                print(BPM)
            #print(round(BPM), len(bpms), time)
            bpms = bpms[-60:]


from magenta.models.music_vae import configs
from magenta.models.music_vae import TrainedModel

DEFAULT_PORT = 'UM-ONE'
available_outputs = mido.get_output_names()

if not available_outputs:
    STATE.MESSAGES = "Using virtual port"
else:
    STATE.MESSAGES = "Port: %s"%DEFAULT_PORT

STATE.PORT = port = mido.open_ioport('UM-ONE', virtual=(not available_outputs))

d = "~/repositories/mosolov/models/hierdec-mel_16bar.tar"
checkpoint_dir_or_path = os.path.expanduser(d)
config = configs.CONFIG_MAP['hierdec-mel_16bar']
"""
d = "~/repositories/mosolov/models/cat-mel_2bar_big.tar"
checkpoint_dir_or_path = os.path.expanduser(d)
config = configs.CONFIG_MAP['cat-mel_2bar_big']
"""

keyboard_listener = Listener(on_press=keypress_handler)
keyboard_listener.start()
#keyboard_listener.join()


model = TrainedModel(
    config,
    batch_size=min(8, STATE.NUM_OUTPUTS),
    checkpoint_dir_or_path=checkpoint_dir_or_path
)

STATE.SEQUENCES = model.sample(
    n=STATE.NUM_OUTPUTS,
    length=config.hparams.max_seq_len,
    temperature=STATE.TEMPERATURE
)

#sync_thread = threading.Thread(name='sync', target=sync)
#sync_thread.start()

play_thread = threading.Thread(name='play', target=play)
play_thread.start()

STATE.print()

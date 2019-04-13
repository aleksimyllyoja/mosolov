import os
import mido

from timeit import default_timer as timer

from magenta.models.music_vae import configs
from magenta.models.music_vae import TrainedModel

num_outputs = 5
temperature = 0.5


checkpoint_dir_or_path = os.path.expanduser("~/repositories/mosolov/cat-mel_2bar_big.tar")
config = configs.CONFIG_MAP['cat-mel_2bar_big']

print(checkpoint_dir_or_path)

model = TrainedModel(
    config,
    batch_size=min(8, num_outputs),
    checkpoint_dir_or_path=checkpoint_dir_or_path
)

results = model.sample(
    n=num_outputs,
    length=config.hparams.max_seq_len,
    temperature=temperature
)

print(results)
#print(mido.get_output_names())
#port = mido.open_output('UM-ONE')

def play(notes):
    start = timer()

    index = 0
    note_on = False

    while True:
        if not playing:
            continue

        if index==len(notes):
            index = 0
            start = timer()

        note = notes[index]
        time = (timer()-start)

        if not note_on and time>=note.start_time and time<=note.end_time:
            msg = mido.Message(
                'note_on',
                note=note.pitch,
                velocity=note.velocity
            )
            print(time, note)
            #port.send(msg)
            note_on = True

        if time>=note.end_time:
            msg = mido.Message(
                'note_off',
                note=note.pitch,
                velocity=note.velocity
            )
            #port.send(msg)

            index += 1
            note_on = False

play(notes[0])

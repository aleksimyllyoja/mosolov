import os

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

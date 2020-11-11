import baby
from grid import IInterface
import inspect

class Interface(object):
    # TODO: fix inheritance
    def __init__(self, *args, **kwargs):
        #super().__init__(*args, **kwargs)
        self.train_dataset = baby.VAEDataset("E:\\epq-datasets\\CelebA_aligned\\train", batch_size=96, image_size=128, prefetch=16, workers=4)
        self.val_dataset = baby.VAEDataset("E:\\epq-datasets\\CelebA_aligned\\val", batch_size=96, image_size=128, prefetch=16, workers=4)

    def job(self, args):
        VAEModel = _run_with_correct_args(baby.models.BABYModelRecreate.__init__, **args)
        history = _run_with_correct_args(lambda **kwargs: VAEModel.fit(self.train_dataset, val_dataset = self.val_dataset, **kwargs), **args)

        results = {}
        for value in history:
            results[value] = history[value][-1]
        
        return results

def _run_with_correct_args(func, **kwargs):
    args = inspect.getargspec(func)[0]
    print(kwargs)
    print(args)
    correct_args = {}
    kwargs = dict(kwargs)
    for value in kwargs:
        if value in args:
            correct_args[value] = kwargs[value]
    return func(**correct_args)
# TODO: Fix args system to seperate into model and train
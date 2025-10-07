from transformers import Trainer


class RefTrainer(Trainer):
    def __init__(self, *args, refactorer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.refactorer = refactorer

    def create_optimizer(self):
        optimizer = super().create_optimizer()
        if self.refactorer is not None:
            self.refactorer.integrate_into_optimizer(optimizer)

        return optimizer

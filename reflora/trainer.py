from transformers import Trainer, TrainerCallback


class _ResumeCallback(TrainerCallback):
    '''Fast-forward the refactoring schedule (warmup/interval) when resuming from a checkpoint.'''

    def __init__(self, refactorer):
        self.refactorer = refactorer

    def on_train_begin(self, args, state, control, **kwargs):
        # runs after `create_optimizer` and after `global_step` is restored from the checkpoint
        if state.global_step > 0:
            self.refactorer.fast_forward(state.global_step)


class RefTrainer(Trainer):
    def __init__(self, *args, refactorer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.refactorer = refactorer
        if refactorer is not None:
            self.add_callback(_ResumeCallback(refactorer))

    def create_optimizer(self):
        optimizer = super().create_optimizer()
        if self.refactorer is not None:
            self.refactorer.integrate_into_optimizer(optimizer)

        return optimizer

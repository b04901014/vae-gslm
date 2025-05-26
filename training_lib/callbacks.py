from lightning.pytorch.callbacks import ModelCheckpoint


class CompactModelCheckpoint(ModelCheckpoint):
    def _save_checkpoint(self, trainer, filepath: str) -> None:
        if getattr(trainer.strategy._lightning_module,
                   "save_checkpoint", False):
            trainer.strategy._lightning_module.save_checkpoint(filepath)
            self._last_global_step_saved = trainer.global_step
            self._last_checkpoint_saved = filepath

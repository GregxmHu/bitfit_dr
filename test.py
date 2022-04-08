from accelerate import Accelerator
from src.models.DRT5 import DRT5
accelerator=Accelerator()
model=DRT5(
    pretrain_model_path_or_name="t5-small"
)
model.load_bias()
model=accelerator.prepare(model)
if accelerator.is_main_process:
    model.module.save(
        "/home/huxiaomeng/bitfit_dr"
    )



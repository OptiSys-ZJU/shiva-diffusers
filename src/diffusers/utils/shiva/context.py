import torch

from dataclasses import dataclass
from typing import Optional

@dataclass
class PipelineContext:
    model_name: Optional[str] = None
    hidden_dim: Optional[int] = None
    encoder_hidden_dim: Optional[int] = None

    hidden_seq_len: Optional[int] = None
    hidden_seq_height: Optional[int] = None
    hidden_seq_width: Optional[int] = None

    encoder_flag: Optional[bool] = True

    encoder_hidden_seq_len: Optional[int] = None
    negative_encoder_hidden_seq_len: Optional[int] = None

    num_inference_steps: Optional[int] = None
    num_layers: Optional[int] = None
    num_single_layers: Optional[int] = None # for Flux only

    # real-time timestep
    current_timestep: Optional[int] = None
    current_iteration: Optional[int] = None

    current_stage_idx: Optional[int] = None

    # for edit task
    hidden_extra_seq_len: Optional[int] = None
    hidden_extra_seq_height: Optional[int] = None
    hidden_extra_seq_width: Optional[int] = None

    # for video generation
    hidden_seq_frame: Optional[int] = None

    # for pixart-sigma, which we cannot get temb from block
    embedded_timestep: Optional[torch.Tensor] = None

    # for IBTM, store the last hidden state
    last_cfg_hidden_state: Optional[torch.Tensor] = None

    # for Shiva, we need to temb which is not merged with encoder_hiddens
    pure_temb: Optional[torch.Tensor] = None

    train_step_idx: Optional[int] = None

    def hidden_seq_len_gen(self) -> int:
        if self.hidden_extra_seq_len is not None:
            return self.hidden_extra_seq_len + self.hidden_seq_len
        else:
            return self.hidden_seq_len

    def encoder_hidden_seq_len_gen(self) -> int:
        if self.encoder_flag:
            return self.encoder_hidden_seq_len
        else:
            return self.negative_encoder_hidden_seq_len

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                raise ValueError(f"Unknown attribute {k}")

    def clear(self):
        for k in self.__dict__.keys():
            setattr(self, k, None)

    def pretty(self) -> str:
        """Format the context as a nicely aligned multi-section string."""
        groups = {
            "Basic Info": ["model_name"],
            "Dims": ["hidden_dim", "encoder_hidden_dim"],
            "Hidden Seq": [
                "hidden_seq_len",
                "hidden_seq_height",
                "hidden_seq_width",
                "hidden_seq_frame",
            ],
            "Encoder Seq": [
                "encoder_hidden_seq_len",
                "negative_encoder_hidden_seq_len",
            ],
            "Edit Extra": [
                "hidden_extra_seq_len",
                "hidden_extra_seq_height",
                "hidden_extra_seq_width",
            ],
        }

        lines = ["PipelineContext {"]
        for section, keys in groups.items():
            lines.append(f"  [{section}]")
            for k in keys:
                v = getattr(self, k, None)
                lines.append(f"    {k:<32}: {v}")
        lines.append("}")
        return "\n".join(lines)

    def __repr__(self):
        return self.pretty()


global_context = PipelineContext()

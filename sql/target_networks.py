import torch
import click
import warnings
import texar.torch as tx
from typing import Tuple, Dict, Union, Type

from sql.toy_utils import index_Q_star_list
from sql.types import (
    BatchType,
    LongTensor,
    AttentionRNNDecoderOutput2,
    AttentionRNNDecoderOutput3)


class Seq2SeqAttnTarget(torch.nn.Module):

    def __init__(
            self,
            learning_rate: float,
            update_method: str,
            model_cls: Type[torch.nn.Module],
            model: torch.nn.Module,
            *model_args, **model_kwargs
    ) -> None:
        super().__init__()
        self._learning_rate = learning_rate
        self._update_method = update_method
        self._target_model = model_cls(
            *model_args, **model_kwargs)

        # Sync with the model initially
        self._target_model.load_state_dict(
            model.state_dict())

    def sync(self, model: torch.nn.Module) -> None:
        # https://github.com/transedward/pytorch-dqn/blob/master/dqn_learn.py#L221
        if self._update_method == "copy":
            self._target_model.load_state_dict(
                model.state_dict())

        # Target network update
        # Note that we are assuming `model.parameters()`
        # would yield the same parameter orders.
        # https://towardsdatascience.com/double-deep-q-networks-905dd8325412
        if self._update_method == "polyak":
            for target_param, model_param in zip(
                    self._target_model.parameters(),
                    model.parameters()):

                target_param.data.copy_(
                    (1 - self._learning_rate) * target_param +
                    self._learning_rate * model_param)

    def forward(
            self,
            batch: BatchType,
            mode: str
    ) -> Union[Tuple[tx.modules.AttentionRNNDecoderOutput, LongTensor], Dict]:
        return self._target_model(batch=batch, mode=mode)


class Seq2SeqAttnTarget2(torch.nn.Module):

    def __init__(
            self,
            learning_rate: float,
            update_method: str,
            model_cls: Type[torch.nn.Module],
            *model_args, **model_kwargs
    ) -> None:
        super().__init__()
        self._learning_rate = learning_rate
        self._update_method = update_method
        self._target_models = torch.nn.ModuleList([
            model_cls(*model_args, **model_kwargs),
            model_cls(*model_args, **model_kwargs)])

    def forward(
            self,
            batch: BatchType,
            mode: str
    ) -> Tuple[AttentionRNNDecoderOutput2, LongTensor]:
        if mode not in ["train", "train-sql-offpolicy"]:
            raise ValueError

        logits_collection = []
        sequence_lengths_collection = []
        for model in self._target_models:
            outputs, sequence_lengths = model(batch=batch, mode=mode)
            logits_collection.append(outputs.logits)
            sequence_lengths_collection.append(sequence_lengths)

        decoder_outputs = AttentionRNNDecoderOutput2(
            logits=torch.minimum(*logits_collection),
            logits_collections=logits_collection)

        # Just return of the `sequence_lengths`
        return decoder_outputs, sequence_lengths_collection[0]


class Seq2SeqAttnTarget3(torch.nn.Module):

    def __init__(
            self,
            reward_shaping: bool,
            reward_shaping_min: float,
            reward_shaping_max: float,
    ) -> None:
        super().__init__()

        if any([
            reward_shaping is False,
            reward_shaping_min != -5,
            reward_shaping_max != 5
        ]):
            raise ValueError("Reward shaping config is not supported")

        self._Q_star_list = torch.load(
            "/export/share/Experiments/20210302/toy_copy_3.Q_star_list.pth")

        warnings.warn(click.style(
            "This only works for a specific version of `toy_copy_3` and "
            "assumes that rewards are shapped between [-5, 5]", bg="red"))

    def forward(
            self,
            batch: BatchType,
            mode: str
    ) -> Tuple[AttentionRNNDecoderOutput3, LongTensor]:

        logits = []
        for target_text_id in batch["target_text_ids"][:, 1:].tolist():
            _logits = index_Q_star_list(self._Q_star_list, target_text_id)
            logits.append(torch.stack(_logits, dim=0))

        stacked_logits = (
            torch.stack(logits, dim=0)
            .to(batch["target_text_ids"].device))

        return (
            AttentionRNNDecoderOutput3(
                logits=stacked_logits),
            batch["target_length"] - 1)

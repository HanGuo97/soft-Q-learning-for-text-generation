# Copyright 2019 The Texar Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.nn as nn
import texar.torch as tx
from functools import partial
from typing import Union, Tuple, Dict, Any, Optional

from configs.models import (
    config_model_lstm,
    config_model_transformers,
    config_model_transformers_small)

from sql.utils import ForwardMode
from sql import helpers as sql_helpers
from sql.types import (
    BatchType,
    HF_BatchType,
    FloatTensor,
    LongTensor)


class Transformer(nn.Module):
    r"""A standalone sequence-to-sequence Transformer model, from "Attention
    Is All You Need". The Transformer model consists of the word embedding
    layer, position embedding layer, an encoder and a decoder. Both encoder
    and decoder are stacks of self-attention layers followed by feed-forward
    layers. See "Attention Is All You Need" (https://arxiv.org/abs/1706.03762)
    for the full description of the model.
    """

    def __init__(
            self,
            train_data: tx.data.PairedTextData,
            max_source_length: int,
            max_decoding_length: int,
            config_name: str,
    ) -> None:
        super().__init__()

        if config_name not in ["transformer",
                               "transformer_small"]:
            raise ValueError

        if config_name == "transformer":
            config_model: Any = config_model_transformers

        if config_name == "transformer_small":
            config_model = config_model_transformers_small

        self.config_model = config_model
        self.max_source_length = max_source_length
        self.max_decoding_length = max_decoding_length

        self.source_vocab = train_data.source_vocab
        self.target_vocab = train_data.target_vocab
        self.source_vocab_size = train_data.source_vocab.size
        self.target_vocab_size = train_data.target_vocab.size

        self.bos_token_id = train_data.target_vocab.bos_token_id
        self.eos_token_id = train_data.target_vocab.eos_token_id

        self.source_embedder = tx.modules.WordEmbedder(
            vocab_size=self.source_vocab_size,
            hparams=self.config_model.emb)

        self.target_embedder = tx.modules.WordEmbedder(
            vocab_size=self.target_vocab_size,
            hparams=self.config_model.emb)

        self.source_pos_embedder = tx.modules.SinusoidsPositionEmbedder(
            position_size=self.max_source_length,
            hparams=self.config_model.position_embedder_hparams)

        self.target_pos_embedder = tx.modules.SinusoidsPositionEmbedder(
            position_size=self.max_decoding_length,
            hparams=self.config_model.position_embedder_hparams)

        self.encoder = tx.modules.TransformerEncoder(
            hparams=self.config_model.encoder)
        self.decoder = tx.modules.TransformerDecoder(
            token_pos_embedder=partial(
                self._embedding_fn,
                source_or_target="target"),
            vocab_size=self.target_vocab_size,
            output_layer=self.target_embedder.embedding,
            hparams=self.config_model.decoder)

    def _embedding_fn(
            self,
            tokens: LongTensor,
            positions: LongTensor,
            source_or_target: str,
    ) -> FloatTensor:
        if source_or_target not in ["source", "target"]:
            raise ValueError

        if source_or_target == "source":
            word_embed = self.source_embedder(tokens)
            pos_embed = self.source_pos_embedder(positions)
        if source_or_target == "target":
            word_embed = self.target_embedder(tokens)
            pos_embed = self.target_pos_embedder(positions)

        scale = self.config_model.hidden_dim ** 0.5
        return word_embed * scale + pos_embed

    def decode_teacher_forcing(
            self,
            batch: BatchType,
            memory: FloatTensor
    ) -> Tuple[tx.modules.TransformerDecoderOutput, LongTensor]:
        decoder_outputs = self.decoder(
            memory=memory,
            memory_sequence_length=batch["source_length"],
            inputs=batch["target_text_ids"][:, :-1],
            sequence_length=batch["target_length"] - 1,
            decoding_strategy="train_greedy")

        # label_lengths = (labels != 0).long().sum(dim=1)
        # We don't really need `sequence_lengths` here
        return decoder_outputs, None

    def decode_greedy(
            self,
            batch: BatchType,
            memory: FloatTensor,
            corruption_p: Optional[float] = None,
    ) -> Tuple[tx.modules.TransformerDecoderOutput, LongTensor]:

        start_tokens = memory.new_full(
            batch["target_length"].size(),
            self.bos_token_id,
            dtype=torch.int64)

        helper = None
        if corruption_p is not None:
            helper = sql_helpers.CorruptedGreedyEmbeddingHelper(
                start_tokens=start_tokens,
                end_token=self.eos_token_id,
                vocab=self.target_vocab,
                corruption_p=corruption_p)

        return self.decoder(
            start_tokens=start_tokens,
            end_token=self.eos_token_id,
            helper=helper,
            memory=memory,
            memory_sequence_length=batch["source_length"],
            decoding_strategy="infer_greedy",
            # Probably will hurt the longest sequence,
            # but probably better learning
            max_decoding_length=min(
                self.max_decoding_length,
                batch["target_length"].max().item() - 1))

    def decode_sampling(
            self,
            batch: BatchType,
            memory: FloatTensor,
            top_k: Optional[int] = None,
            top_p: Optional[float] = None,
    ) -> Tuple[tx.modules.TransformerDecoderOutput, LongTensor]:
        if top_k is not None and top_p is not None:
            raise ValueError

        start_tokens = memory.new_full(
            batch["target_length"].size(),
            self.bos_token_id,
            dtype=torch.int64)

        helper = None
        if top_k is not None:
            helper = tx.modules.TopKSampleEmbeddingHelper(
                start_tokens=start_tokens,
                end_token=self.eos_token_id,
                top_k=top_k)

        if top_p is not None:
            helper = tx.modules.TopPSampleEmbeddingHelper(
                start_tokens=start_tokens,
                end_token=self.eos_token_id,
                p=top_p)

        return self.decoder(
            start_tokens=start_tokens,
            end_token=self.eos_token_id,
            helper=helper,
            memory=memory,
            memory_sequence_length=batch["source_length"],
            decoding_strategy="infer_sample",
            # Probably will hurt the longest sequence,
            # but probably better learning
            max_decoding_length=min(
                self.max_decoding_length,
                batch["target_length"].max().item() - 1))

    def decode_beam_search(
            self,
            batch: BatchType,
            memory: FloatTensor,
            beam_width: int,
            corruption_p: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:

        # Only greedy decoding is support for this as of now.
        if corruption_p is not None:
            if beam_width != 1:
                raise NotImplementedError

        # when `beam_width in [None, 1]`, `self.decoder`
        # will switch to default decoding mode, which is
        # not necessarily what we want. Instead, let's
        # explicitly call greedy-decoding.
        # https://sourcegraph.com/github.com/asyml/texar-pytorch/-/blob/texar/torch/modules/decoders/rnn_decoders.py#L717:9
        if beam_width > 1:

            start_tokens = memory.new_full(
                batch["target_length"].size(),
                self.bos_token_id,
                dtype=torch.int64)

            return self.decoder(
                start_tokens=start_tokens,
                end_token=self.eos_token_id,
                memory=memory,
                memory_sequence_length=batch["source_length"],
                beam_width=beam_width,
                max_decoding_length=self.max_decoding_length)

        else:
            infer_outputs, _ = self.decode_greedy(
                batch=batch,
                memory=memory,
                corruption_p=corruption_p)

            return {
                "sample_id": (
                    infer_outputs
                    .sample_id
                    .unsqueeze(dim=-1)
                )
            }

    def forward(
            self,
            batch: BatchType,
            mode: ForwardMode,
            **kwargs,
    ) -> Union[Tuple[tx.modules.TransformerDecoderOutput, LongTensor], Dict]:

        # Text sequence length excluding padding
        if not (batch["source_length"] == (batch["source_text_ids"] != 0).int().sum(dim=1)).all():
            raise ValueError

        positions: LongTensor = (
            torch.arange(
                batch["source_length"].max(),  # type: ignore
                dtype=torch.long,
                device=batch["source_text_ids"].device)
            .unsqueeze(0)
            .expand(batch["source_text_ids"].size(0), -1)
        )

        encoder_output = self.encoder(
            inputs=self._embedding_fn(
                tokens=batch["source_text_ids"],
                positions=positions,
                source_or_target="source"),
            sequence_length=batch["source_length"])

        if mode in [ForwardMode.MLE, ForwardMode.SQL_OFF_GT]:
            return self.decode_teacher_forcing(
                batch=batch,
                memory=encoder_output)

        if mode in [ForwardMode.PG, ForwardMode.SQL_ON]:
            return self.decode_sampling(
                batch=batch,
                memory=encoder_output,
                **kwargs)

        if mode in [ForwardMode.INFER]:
            return self.decode_beam_search(
                batch=batch,
                memory=encoder_output,
                **kwargs)

        raise ValueError(f"Unknown mode {mode}")


class Seq2SeqAttn(torch.nn.Module):

    def __init__(self, train_data: tx.data.PairedTextData) -> None:
        super().__init__()

        self.source_vocab = train_data.source_vocab
        self.target_vocab = train_data.target_vocab
        self.source_vocab_size = train_data.source_vocab.size
        self.target_vocab_size = train_data.target_vocab.size

        self.bos_token_id = train_data.target_vocab.bos_token_id
        self.eos_token_id = train_data.target_vocab.eos_token_id

        self.source_embedder = tx.modules.WordEmbedder(
            vocab_size=self.source_vocab_size,
            hparams=config_model_lstm.embedder)

        self.target_embedder = tx.modules.WordEmbedder(
            vocab_size=self.target_vocab_size,
            hparams=config_model_lstm.embedder)

        self.encoder = tx.modules.BidirectionalRNNEncoder(
            input_size=self.source_embedder.dim,
            hparams=config_model_lstm.encoder)

        self.decoder = tx.modules.AttentionRNNDecoder(
            token_embedder=self.target_embedder,
            encoder_output_size=(self.encoder.cell_fw.hidden_size +
                                 self.encoder.cell_bw.hidden_size),
            input_size=self.target_embedder.dim,
            vocab_size=self.target_vocab_size,
            hparams=config_model_lstm.decoder)

    def decode_teacher_forcing(
            self,
            batch: BatchType,
            memory: FloatTensor
    ) -> Tuple[tx.modules.AttentionRNNDecoderOutput, LongTensor]:

        helper_train = self.decoder.create_helper(
            decoding_strategy="train_greedy")

        decoder_outputs, _, sequence_lengths = self.decoder(
            memory=memory,
            memory_sequence_length=batch["source_length"],
            helper=helper_train,
            inputs=batch["target_text_ids"][:, :-1],
            sequence_length=batch["target_length"] - 1)

        return decoder_outputs, sequence_lengths

    def decode_greedy(
            self,
            batch: BatchType,
            memory: FloatTensor,
            corruption_p: Optional[float] = None,
    ) -> Tuple[tx.modules.AttentionRNNDecoderOutput, LongTensor]:

        start_tokens = memory.new_full(
            batch["target_length"].size(),
            self.bos_token_id,
            dtype=torch.int64)

        helper = None
        if corruption_p is not None:
            helper = sql_helpers.CorruptedGreedyEmbeddingHelper(
                start_tokens=start_tokens,
                end_token=self.eos_token_id,
                vocab=self.target_vocab,
                corruption_p=corruption_p)

        decoder_outputs, _, sequence_lengths = self.decoder(
            start_tokens=start_tokens,
            end_token=self.eos_token_id,
            helper=helper,
            memory=memory,
            memory_sequence_length=batch["source_length"],
            decoding_strategy="infer_greedy",
            # Probably will hurt the longest sequence,
            # but probably better learning
            max_decoding_length=batch["target_length"].max().item() - 1)

        return decoder_outputs, sequence_lengths

    def decode_sampling(
            self,
            batch: BatchType,
            memory: FloatTensor,
            top_k: Optional[int] = None,
            top_p: Optional[float] = None,
    ) -> Tuple[tx.modules.AttentionRNNDecoderOutput, LongTensor]:
        if top_k is not None and top_p is not None:
            raise ValueError

        start_tokens = memory.new_full(
            batch["target_length"].size(),
            self.bos_token_id,
            dtype=torch.int64)

        helper = None
        if top_k is not None:
            helper = tx.modules.TopKSampleEmbeddingHelper(
                start_tokens=start_tokens,
                end_token=self.eos_token_id,
                top_k=top_k)

        if top_p is not None:
            helper = tx.modules.TopPSampleEmbeddingHelper(
                start_tokens=start_tokens,
                end_token=self.eos_token_id,
                p=top_p)

        decoder_outputs, _, sequence_lengths = self.decoder(
            start_tokens=start_tokens,
            end_token=self.eos_token_id,
            helper=helper,
            memory=memory,
            memory_sequence_length=batch["source_length"],
            decoding_strategy="infer_sample",
            # Probably will hurt the longest sequence,
            # but probably better learning
            max_decoding_length=batch["target_length"].max().item() - 1)

        return decoder_outputs, sequence_lengths

    def decode_beam_search(
            self,
            batch: BatchType,
            memory: FloatTensor,
            beam_width: int,
            corruption_p: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:

        # Only greedy decoding is support for this as of now.
        if corruption_p is not None:
            if beam_width != 1:
                raise NotImplementedError

        # when `beam_width in [None, 1]`, `self.decoder`
        # will switch to default decoding mode, which is
        # not necessarily what we want. Instead, let's
        # explicitly call greedy-decoding.
        # https://sourcegraph.com/github.com/asyml/texar-pytorch/-/blob/texar/torch/modules/decoders/rnn_decoders.py#L717:9
        if beam_width > 1:

            start_tokens = memory.new_full(
                batch["target_length"].size(),
                self.bos_token_id,
                dtype=torch.int64)

            return self.decoder(
                start_tokens=start_tokens,
                end_token=self.eos_token_id,
                memory=memory,
                memory_sequence_length=batch["source_length"],
                beam_width=beam_width)
        else:
            infer_outputs, _ = self.decode_greedy(
                batch=batch,
                memory=memory,
                corruption_p=corruption_p)

            return {
                "sample_id": (
                    infer_outputs
                    .sample_id
                    .unsqueeze(dim=-1)
                )
            }

    def forward(
            self,
            batch: BatchType,
            mode: ForwardMode,
            **kwargs,
    ) -> Union[Tuple[tx.modules.AttentionRNNDecoderOutput, LongTensor], Dict]:

        enc_outputs, _ = self.encoder(
            inputs=self.source_embedder(batch["source_text_ids"]),
            sequence_length=batch["source_length"])

        memory = torch.cat(enc_outputs, dim=2)

        if mode in [ForwardMode.MLE, ForwardMode.SQL_OFF_GT]:
            return self.decode_teacher_forcing(
                batch=batch, memory=memory)

        if mode in [ForwardMode.PG, ForwardMode.SQL_ON]:
            return self.decode_sampling(
                batch=batch, memory=memory,
                **kwargs)

        if mode in [ForwardMode.INFER]:
            return self.decode_beam_search(
                batch=batch, memory=memory,
                **kwargs)

        raise ValueError(f"Unknown mode {mode}")

import os
import click
import torch
import numpy as np
import sacrebleu as scb
from collections import Counter
from collections import defaultdict
from joblib import Parallel, delayed
from fairseq.data.data_utils import collate_tokens
from fairseq.models.roberta import RobertaHubInterface
from typing import List, Tuple, Union, Dict, Optional, Callable, Any, cast

from datasets import load_metric
from transformers import (
    pipeline,
    PreTrainedModel,
    PreTrainedTokenizerFast,
    AutoTokenizer,
    AutoModelWithLMHead,
    AutoModelForSequenceClassification,
    PreTrainedTokenizerBase,
    PegasusTokenizer,
    PegasusForConditionalGeneration,
    RobertaForSequenceClassification)
# from sentence_transformers import CrossEncoder

from modules import gpt2 as gpt2_modules
from sql.types import FloatTensor
from sql import utils as sql_utils
from sql import misc_utils

try:
    from detoxify import Detoxify
except ModuleNotFoundError:
    Detoxify = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def check_Xs_Ys_sizes(
        Xs: List,
        Ys: List,
        check_type_is_list: bool = False,
        check_first_element_is_string: bool = True,
) -> None:
    if len(Xs) != len(Ys):
        raise ValueError(
            f"Xs.length = {len(Xs)}, "
            f"Ys.length = {len(Ys)}")

    if check_type_is_list is True:
        if not isinstance(Xs, list) or not isinstance(Ys, list):
            raise ValueError(
                f"Xs.type = {type(Xs)}, "
                f"Ys.type = {type(Ys)}")

    if check_first_element_is_string is True:
        if not isinstance(Xs[0], str) or not isinstance(Ys[0], str):
            raise ValueError(
                f"Xs[0].type = {type(Xs[0])}, "
                f"Ys[0].type = {type(Ys[0])}")


class BLEUReward(object):
    def __init__(self, method: Optional[str] = None) -> None:
        if method is None:
            # `lightning` is marginally better empirically
            method = "lightning"

        self._method = method

    def forward(
            self,
            output_texts: List[str],
            target_texts: List[str],
            to_tensor: bool
    ) -> Tuple[Union[List[float], FloatTensor], Dict[str, Any]]:
        check_Xs_Ys_sizes(output_texts, target_texts)
        # Using a faster BLEU implementation during training
        # `sacrebleu` is ~3X faster than `lightning`
        # `sacrebleu-parallel` is ~3X faster than `sacrebleu`
        if self._method == "sacrebleu":
            bleus = [
                scb.sentence_bleu(
                    hypothesis=x,
                    references=[y])
                for x, y in zip(
                    output_texts,
                    target_texts)
            ]
            rewards = [b.score for b in bleus]
        elif self._method == "sacrebleu-parallel":
            # two jobs are probably enough for now
            bleus = Parallel(n_jobs=2)(
                delayed(scb.sentence_bleu)(
                    hypothesis=x,
                    references=[y])
                for x, y in zip(
                    output_texts,
                    target_texts)
            )

            rewards = [b.score for b in bleus]
        else:
            rewards = sql_utils.compute_sentence_bleu_batch(
                output_texts=[text.split() for text in output_texts],
                target_texts=[text.split() for text in target_texts],
                method=self._method)

        rewards_log: Dict[str, Any] = {}
        if to_tensor is True:
            return torch.tensor(rewards), rewards_log
        else:
            return rewards, rewards_log

    def __call__(
        self,
        sources: List[str],
        targets: List[str],
        predictions: List[str],
        to_tensor: bool,
        mode: str,
    ) -> Tuple[Union[List[float], FloatTensor], Dict[str, Any]]:
        return self.forward(
            output_texts=predictions,
            target_texts=targets,
            to_tensor=to_tensor)


class ROUGEReward(object):
    def __init__(self, rouge_type: Optional[str] = None) -> None:
        if rouge_type is None:
            rouge_type = "rougeL"

        self._rouge_type = rouge_type
        self._metric = load_metric("rouge")

    def forward(
            self,
            output_texts: List[str],
            target_texts: List[str],
            to_tensor: bool
    ) -> Tuple[Union[List[float], FloatTensor], Dict[str, Any]]:
        check_Xs_Ys_sizes(output_texts, target_texts)

        results = self._metric.compute(
            predictions=output_texts,
            references=target_texts,
            rouge_types=[self._rouge_type],
            use_agregator=False)

        # The results are list of `Score` tuple
        # and the scale was [0.0, 1.0]
        rewards = [s.fmeasure * 100 for s in results[self._rouge_type]]

        rewards_log: Dict[str, Any] = {}
        if to_tensor is True:
            return torch.tensor(rewards), rewards_log
        else:
            return rewards, rewards_log

    def __call__(
        self,
        sources: List[str],
        targets: List[str],
        predictions: List[str],
        to_tensor: bool,
        mode: str,
    ) -> Tuple[Union[List[float], FloatTensor], Dict[str, Any]]:
        return self.forward(
            output_texts=predictions,
            target_texts=targets,
            to_tensor=to_tensor)


class BleurtReward(object):
    def __init__(self, checkpoint: Optional[str] = None) -> None:
        if checkpoint is None:
            checkpoint = "bleurt-base-128"

        self._metric = load_metric("bleurt", checkpoint)

    def forward(
            self,
            predictions: List[str],
            references: List[str],
            to_tensor: bool
    ) -> Tuple[Union[List[float], FloatTensor], Dict[str, Any]]:
        check_Xs_Ys_sizes(predictions, references)
        scores_dict = self._metric.compute(
            references=references,
            predictions=predictions)

        # I don't honestly know the range of scores, but
        # looks like they are in [-2, 2], hence this
        # transformation brings it to [0, 100]
        scores = [
            score * 25 + 50 for score in
            scores_dict["scores"]]

        rewards_log: Dict[str, Any] = {}
        if to_tensor is True:
            return torch.tensor(scores), rewards_log
        else:
            return scores, rewards_log

    def __call__(
        self,
        sources: List[str],
        targets: List[str],
        predictions: List[str],
        to_tensor: bool,
        mode: str,
    ) -> Tuple[Union[List[float], FloatTensor], Dict[str, Any]]:
        return self.forward(
            predictions=predictions,
            references=targets,
            to_tensor=to_tensor)


class EntailmentClassifier(object):

    def __init__(
            self,
            task_name: str,
            batch_size: int = 32,
            model_name: Optional[str] = None,
            include_perplexity: bool = True,
    ) -> None:

        if model_name is None:
            model_name = "ynie"

        if model_name not in ["ynie", "fairseq"]:
            raise ValueError

        if include_perplexity is True:
            sql_utils.colorful_warning(
                f"Adding LM-based reward with the "
                f"model trained on {task_name}", bg="blue")

        if model_name == "ynie":
            hf_model_name = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
            tokenizer = (
                AutoTokenizer
                .from_pretrained(hf_model_name))
            model = (
                AutoModelForSequenceClassification
                .from_pretrained(hf_model_name))
        else:
            tokenizer = None
            model = torch.hub.load(
                "pytorch/fairseq",
                "roberta.large.mnli")

        sql_utils.colorful_warning(f"Using {model_name}", bg="blue")

        model.eval()
        model.to(device)
        self._model = model
        self._tokenizer = tokenizer
        self._batch_size = batch_size
        self._model_name = model_name
        self._include_perplexity = include_perplexity
        self._language_model = gpt2_modules.GPT2Model(
            task_name=task_name,
            batch_size=batch_size)

    def _compute_nll_reward(self, sentences: List[str]) -> FloatTensor:
        # We use the negative NLL as the reward
        return -self._language_model.forward(sentences)

    def _compute_entailment_probs(self, Xs: List[str], Ys: List[str]) -> Tuple[FloatTensor, int]:

        if self._model_name == "ynie":
            entailment_index = 0
            _batch_prediction_fn = (
                lambda _Xs, _Ys: get_NLI_prediction(
                    tokenizer=self._tokenizer,
                    model=self._model,
                    premises=_Xs,
                    hypotheses=_Ys,
                    device=device))
        else:
            entailment_index = 2
            _batch_prediction_fn = (
                lambda _Xs, _Ys: get_NLI_prediction_2(
                    model=self._model,
                    premises=_Xs,
                    hypotheses=_Ys))

        probs = []
        for index in range(0, len(Ys), self._batch_size):
            i_0 = index
            i_1 = index + self._batch_size
            probs.append(
                _batch_prediction_fn(
                    Xs[i_0: i_1],
                    Ys[i_0: i_1]))

        return torch.cat(probs, dim=0), entailment_index

    def _compute_entailment_reward(self, Xs: List[str], Ys: List[str]) -> FloatTensor:
        probs, entailment_index = self._compute_entailment_probs(Xs=Xs, Ys=Ys)
        # We assume rewards are in `[0, 100]`
        return probs[:, entailment_index] * 100

    def forward(self, Xs: List[str], Ys: List[str], to_tensor: bool) -> Tuple[Union[List[float], FloatTensor], Dict[str, Any]]:
        check_Xs_Ys_sizes(Xs, Ys)
        if isinstance(Xs, np.ndarray):
            Xs = Xs.tolist()
        if isinstance(Ys, np.ndarray):
            Ys = Ys.tolist()

        rewards = self._compute_entailment_reward(Xs=Xs, Ys=Ys)
        rewards_log = {"entailment": rewards.mean()}

        # Adding perplexity if necessary
        if self._include_perplexity is True:
            nll_reward = self._compute_nll_reward(Ys)
            rewards = rewards + nll_reward
            rewards_log["nll"] = nll_reward.mean()

        if to_tensor is True:
            return rewards, rewards_log
        else:
            return rewards.tolist(), rewards_log

    def __call__(
        self,
        sources: List[str],
        targets: List[str],
        predictions: List[str],
        to_tensor: bool,
        mode: str,
    ) -> Tuple[Union[List[float], FloatTensor], Dict[str, Any]]:
        return self.forward(
            Xs=sources,
            Ys=predictions,
            to_tensor=to_tensor)


class EntailmentClassifier2(object):
    """This wraps `EntailmentClassifier` for a special use case"""
    TRAIN_SRC_FNAME = "/export/share/Data/multinli/all/train.sources"
    TRAIN_TGT_FNAME = "/export/share/Data/multinli/all/train.targets"
    VALID_SRC_FNAME = "/export/share/Data/multinli/all/valid.sources"
    VALID_TGT_FNAME = "/export/share/Data/multinli/all/valid.targets"
    RepetitionPenaltyCoef = 5.0

    def __init__(self) -> None:
        # Reuse the `EntailmentClassifier3`, which includes
        # a few useful features for this task as well.
        # We use an easier classifier here
        self._reward_module = EntailmentClassifier3(
            task_name="multinli",
            model_name="fairseq")

        # We will use the original, unprocessed sources, which
        # should be the same with the actual data passed to
        # this function during runtime.
        with open(self.TRAIN_SRC_FNAME) as f:
            train_sources = [d.strip() for d in f.readlines()]
        with open(self.TRAIN_TGT_FNAME) as f:
            train_targets = [d.strip() for d in f.readlines()]
        with open(self.VALID_SRC_FNAME) as f:
            valid_sources = [d.strip() for d in f.readlines()]
        with open(self.VALID_TGT_FNAME) as f:
            valid_targets = [d.strip() for d in f.readlines()]

        self._train_data = {}
        self._valid_data = {}
        # There will be duplicates and overrides. But
        # this represent <0.5% of the cases, so we will
        # ignore it for now.
        for source, target in zip(train_sources, train_targets):
            self._train_data[target] = source
        for source, target in zip(valid_sources, valid_targets):
            self._valid_data[target] = source

    def _repetition_penalty_reward(self, sentences: List[str]) -> FloatTensor:
        penalties = []
        for sentence in sentences:
            penalty = 0
            tokens = sentence.split()
            for _, count in Counter(tokens).items():
                penalty = penalty + count - 1
            penalties.append(penalty)

        # reward = -penalty
        return -torch.tensor(penalties).float().to(device)

    def __call__(
        self,
        sources: List[str],
        targets: List[str],
        predictions: List[str],
        to_tensor: bool,
        mode: str,
    ) -> Tuple[Union[List[float], FloatTensor], Dict[str, Any]]:
        if any([s != "start" for s in sources]):
            raise ValueError

        if mode not in ["train", "infer"]:
            raise ValueError

        if mode == "train":
            data = self._train_data
        if mode == "infer":
            data = self._valid_data
        # We will find corresponding sources w.r.t. targets
        # rather than using original sources. This also made
        # some engineering workload substantially easier.
        corresponding_sources = [data[t] for t in targets]
        rewards, rewards_log = self._reward_module(
            sources=corresponding_sources,
            targets=targets,
            predictions=predictions,
            to_tensor=to_tensor,
            mode=mode)

        repetition_reward = self._repetition_penalty_reward(predictions)
        rewards_log["repetition_reward"] = repetition_reward.mean()
        rewards = rewards + repetition_reward * self.RepetitionPenaltyCoef

        if to_tensor is True:
            return rewards, rewards_log
        else:
            raise NotImplementedError


class EntailmentClassifier3(object):
    """This wraps `EntailmentClassifier` with additional BLEU Rewards"""
    BLEURewardCoef = 1.0

    def __init__(self, task_name: Optional[str] = None, **entailment_kwargs) -> None:
        if task_name is None:
            task_name = "snli"

        # Use `SacreBLEU` here.
        self._bleu_reward_module = BLEUReward(method="sacrebleu")
        self._entailment_reward_module = EntailmentClassifier(
            task_name=task_name,
            include_perplexity=True,
            **entailment_kwargs)

    def __call__(
        self,
        sources: List[str],
        targets: List[str],
        predictions: List[str],
        to_tensor: bool,
        mode: str,
    ) -> Tuple[Union[List[float], FloatTensor], Dict[str, Any]]:
        if to_tensor is False:
            raise NotImplementedError

        # Compute BLEU score w.r.t. sources
        bleu_rewards, bleu_rewards_log = (
            self._bleu_reward_module(
                sources=None,
                targets=sources,
                predictions=predictions,
                to_tensor=to_tensor,
                mode=mode))

        entailment_rewards, entailment_rewards_log = (
            self._entailment_reward_module(
                sources=sources,
                targets=targets,
                predictions=predictions,
                to_tensor=to_tensor,
                mode=mode))

        bleu_rewards = bleu_rewards.to(device)
        rewards = (
            entailment_rewards +
            bleu_rewards * self.BLEURewardCoef)
        rewards_log = misc_utils.unionize_dicts([
            bleu_rewards_log,
            entailment_rewards_log,
            {"bleu": bleu_rewards.mean()}])

        if to_tensor is True:
            return rewards, rewards_log
        else:
            raise NotImplementedError


class GPT2TopicReward(object):
    WORDLISTS_BASE_DIR = "/workspace/joint-inference/experiments/wordlists"
    PPLM_INPUTS_FILE_NAME = "/workspace/joint-inference/experiments/pplm-inputs.txt"
    TOPICS = ["legal", "politics", "computers", "space", "religion", "science", "military"]

    def __init__(
            self,
            max_length: int = 60,
            num_return_sequences_train: int = 2,
            num_return_sequences_infer: int = 100,
            topic_scores_aggregator: Optional[Callable[[List[float]], Union[float, np.number]]] = None,
            include_perplexity: bool = True,
            return_intermediate_outputs: bool = False,
    ) -> None:

        if topic_scores_aggregator is None:
            # Use the average by default
            topic_scores_aggregator = lambda scores: np.mean(scores)

        if include_perplexity is True:
            sql_utils.colorful_warning("Adding Perplexity-based Reward", bg="blue")

        sql_utils.colorful_warning(f"max_length={max_length}", bg="blue")

        # https://huggingface.co/gpt2
        # https://huggingface.co/facebook/bart-large-mnli
        self._generator = pipeline(
            "text-generation",
            model="distilgpt2",
            device=0)
        self._classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=0)

        self._max_length = max_length
        self._num_return_sequences_train = num_return_sequences_train
        self._num_return_sequences_infer = num_return_sequences_infer
        self._topic_scores_aggregator = topic_scores_aggregator
        # `topic_to_candidate_labels_map` is deprecated
        self._topic_to_candidate_labels_map, self._pplm_inputs = (
            self.load_topic_to_candidate_labels_map_and_pplm_inputs())

        # Technically, adding perplexity-based reward will break
        # the scale, but we will ignore this for now since
        # this number is relatively small.
        self._include_perplexity = include_perplexity
        # Do not set is to `True` during training, use it for debugging.
        self._return_intermediate_outputs = return_intermediate_outputs

    def load_topic_to_candidate_labels_map_and_pplm_inputs(self) -> Tuple[Dict[str, List[str]], List[str]]:
        topic_to_candidate_labels_map = {}
        for topic in self.TOPICS:
            file_name = os.path.join(
                self.WORDLISTS_BASE_DIR,
                f"{topic}.txt")

            with open(file_name) as f:
                # There is one file that capitalized all words
                # hence it's likely better to lower case all of
                # them -- with the risk of hurting some words
                topic_to_candidate_labels_map[topic] = [
                    d.strip().lower() for d in f.readlines()]

        with open(self.PPLM_INPUTS_FILE_NAME) as f:
            pplm_inputs = [d.strip() for d in f.readlines()]

        return topic_to_candidate_labels_map, pplm_inputs

    def _format_prompts(self, strings: List[str]) -> List[str]:
        inputs = np.random.choice(
            self._pplm_inputs,
            size=len(strings),
            # we use with-replacement here
            replace=True,).tolist()

        new_strings = [
            self._generator.tokenizer
            .convert_tokens_to_string(s.split())
            for s in strings]

        return [
            f"{s_1} {s_2}" for s_1, s_2
            in zip(new_strings, inputs)]

    def _compute_nll_reward(self, sentences: List[str]) -> FloatTensor:
        nlls, _ = compute_perplexities(
            sentences=sentences,
            model=self._generator.model,
            tokenizer=self._generator.tokenizer)
        # When the sentence has just one token,
        # the NLL/perplexity will be `NaN`.
        # Further, we use the negative NLL as the reward
        return -torch.nan_to_num(nlls, nan=10.0).mean()

    def _check_classifier_outputs(
            self,
            candidate_labels: List[str],
            classifier_outputs: List[Dict],
    ) -> None:
        for output in classifier_outputs:
            if len(output["scores"]) != len(candidate_labels):
                raise ValueError

    def forward(self, topics: List[str], prompts: List[str], to_tensor: bool, mode: str) -> Tuple[Union[List[float], FloatTensor], Dict[str, Any]]:
        if mode not in ["train", "infer"]:
            raise ValueError

        if mode == "train":
            num_return_sequences = self._num_return_sequences_train
        if mode == "infer":
            num_return_sequences = self._num_return_sequences_infer

        # - List of length `len(prompts)`
        #     - List of length `num_return_sequences`
        #         - Dict of {"generated_text": str}
        formatted_prompts = self._format_prompts(prompts)
        generator_outputs: List[List[Dict[str, Any]]] = self._generator(
            formatted_prompts,
            max_length=self._max_length,
            num_return_sequences=num_return_sequences,
            # Only return generated text, without the prompt
            return_full_text=False)

        all_classifier_outputs = []
        rewards: List[FloatTensor] = []
        quantities_to_log: Dict[str, List[FloatTensor]] = defaultdict(list)
        for batch_index in range(len(prompts)):
            generated_texts = [
                output["generated_text"] for output in
                generator_outputs[batch_index]]

            # - List of length `len(generated_texts)`
            #     - Dict of {
            #         "labels": List of length `num_topics`,
            #         "scores": List of length `num_topics`,
            #         "sequence": str,
            #     }
            try:
                topic = topics[batch_index]
                classifier_outputs = self._classifier(
                    generated_texts,
                    candidate_labels=[topic],
                    multi_label=True)

                self._check_classifier_outputs(
                    candidate_labels=[topic],
                    classifier_outputs=classifier_outputs)

                _reward_list = [
                    self._topic_scores_aggregator(output["scores"])
                    for output in classifier_outputs]

                # We assume rewards are in `[0, 100]`
                reward = torch.tensor(_reward_list).float().mean() * 100
                quantities_to_log["topic"].append(reward)
                if self._include_perplexity is True:
                    nll_reward = (
                        self._compute_nll_reward(
                            sentences=generated_texts))
                    reward = reward + nll_reward
                    quantities_to_log["nll"].append(nll_reward)

                rewards.append(reward)
                all_classifier_outputs.append(classifier_outputs)

            except ValueError as err:
                # This happens when the generated text itself includes the
                # `</s>` token, which does happen and will cause the classifier to fail.
                # So we just ignore this error and give a score of zero for this batch.
                if str(err) != "All examples must have the same number of <eos> tokens.":
                    raise err

                click.secho("Encountered an error, skipping ...", bg="red")
                rewards.append(torch.tensor(0.).to(device))

        rewards_tensor = torch.stack(rewards)
        rewards_log = dict(
            (reward_key, torch.stack(reward_vals, dim=0).mean())
            for reward_key, reward_vals in quantities_to_log.items())

        if self._return_intermediate_outputs is True:
            rewards_log["quantities_to_log"] = quantities_to_log  # type: ignore
            rewards_log["formatted_prompts"] = formatted_prompts  # type: ignore
            rewards_log["generator_outputs"] = generator_outputs  # type: ignore
            rewards_log["all_classifier_outputs"] = all_classifier_outputs  # type: ignore

        if to_tensor is True:
            return rewards_tensor, rewards_log
        else:
            return rewards_tensor.tolist(), rewards_log

    def __call__(
        self,
        sources: List[str],
        targets: List[str],
        predictions: List[str],
        to_tensor: bool,
        mode: str,
    ) -> Tuple[Union[List[float], FloatTensor], Dict[str, Any]]:
        return self.forward(
            topics=sources,
            prompts=predictions,
            to_tensor=to_tensor,
            mode=mode)


class PrefixSentimentClassifier(object):
    def __init__(self) -> None:
        self._classifier = pipeline("sentiment-analysis", device=0)

    def forward(self, prefixes: List[str], sentences: List[str], to_tensor: bool) -> Tuple[Union[List[float], FloatTensor], Dict[str, Any]]:
        check_Xs_Ys_sizes(prefixes, sentences)

        new_sentences = [
            f"{prefix} {sentence}"
            for prefix, sentence in
            zip(prefixes, sentences)]
        raw_outputs = self._classifier(new_sentences)
        rewards = [
            output["score"] * 100
            if output["label"] == "POSITIVE"
            else (1 - output["score"]) * 100
            for output in raw_outputs]

        rewards_log: Dict[str, Any] = {}
        if to_tensor is True:
            return torch.tensor(rewards), rewards_log
        else:
            return rewards, rewards_log

    def __call__(
        self,
        sources: List[str],
        targets: List[str],
        predictions: List[str],
        to_tensor: bool,
        mode: str,
    ) -> Tuple[Union[List[float], FloatTensor], Dict[str, Any]]:
        return self.forward(
            prefixes=predictions,
            sentences=targets,
            to_tensor=to_tensor)


class ToxificationClassifier(object):
    def __init__(self) -> None:
        self._model = Detoxify("original", device="cuda")

    def forward(self, Xs: List[str], to_tensor: bool) -> Tuple[Union[List[float], FloatTensor], Dict[str, Any]]:
        if isinstance(Xs, np.ndarray):
            Xs = Xs.tolist()

        outputs = self._model.predict(Xs)
        outputs = [(1 - score) * 100 for score in outputs["toxicity"]]

        rewards_log: Dict[str, Any] = {}
        if to_tensor is True:
            return torch.tensor(outputs), rewards_log
        else:
            return outputs, rewards_log

    def __call__(
        self,
        sources: List[str],
        targets: List[str],
        predictions: List[str],
        to_tensor: bool,
        mode: str,
    ) -> Tuple[Union[List[float], FloatTensor], Dict[str, Any]]:
        return self.forward(
            Xs=predictions,
            to_tensor=to_tensor)


reward_name_to_cls_map = {
    "bleu": BLEUReward,
    "rouge": ROUGEReward,
    "bleurt": BleurtReward,
    # "entailment": EntailmentClassifier,
    "entailment2": EntailmentClassifier2,
    "entailment3": EntailmentClassifier3,
    "gpt2-topic": GPT2TopicReward,
    "sentiment": PrefixSentimentClassifier,
    "toxicity": ToxificationClassifier,
}


@torch.no_grad()
def compute_perplexities(
        sentences: List[str],
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerFast,
) -> Tuple[FloatTensor, FloatTensor]:

    nlls = []
    for sentence in sentences:
        encodings = tokenizer(
            sentence,
            return_tensors="pt")
        input_ids = encodings.input_ids.to(device)
        try:
            # labels **are shifted** inside the model
            outputs = model(
                input_ids,
                labels=input_ids.clone())
            nll = outputs[0]
        except RuntimeError:
            # Could happen when the input is empty
            nll = torch.tensor(float("nan")).to(device)

        nlls.append(nll)

    stacked_nlls = torch.stack(nlls, dim=0)
    return stacked_nlls, stacked_nlls.exp()


@torch.no_grad()
def get_NLI_prediction(
        tokenizer: PreTrainedTokenizerFast,
        model: RobertaForSequenceClassification,
        premises: List[str],
        hypotheses: List[str],
        device: torch.device,
        max_length: int = 256,
) -> FloatTensor:

    tokenized_inputs = tokenizer(
        premises,
        hypotheses,
        max_length=max_length,
        return_token_type_ids=True,
        truncation=True,
        padding=True)

    input_ids = (
        torch.Tensor(tokenized_inputs["input_ids"])
        .long()
        .to(device)
    )
    token_type_ids = (
        torch.Tensor(tokenized_inputs["token_type_ids"])
        .long()
        .to(device)
    )
    attention_mask = (
        torch.Tensor(tokenized_inputs["attention_mask"])
        .long()
        .to(device)
    )

    outputs = model(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        labels=None)

    predicted_probability = torch.softmax(outputs.logits, dim=1)
    # predicted_index = torch.argmax(predicted_probability)
    # predicted_probability = predicted_probability.tolist()
    return predicted_probability


@torch.no_grad()
def get_NLI_prediction_2(
        model: RobertaHubInterface,
        premises: List[str],
        hypotheses: List[str],
) -> FloatTensor:
    """https://github.com/pytorch/fairseq/tree/master/examples/roberta"""
    batch = collate_tokens([
        model.encode(premise, hypothesis)
        for premise, hypothesis
        in zip(premises, hypotheses)], pad_idx=1)

    logits = model.predict(
        "mnli", batch,
        return_logits=True)

    return torch.nn.functional.softmax(logits, dim=-1)


def load_paraphrase_generator() -> Tuple[PegasusForConditionalGeneration, PegasusTokenizer]:
    model_name = "tuner007/pegasus_paraphrase"
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)
    return model, tokenizer


@torch.no_grad()
def generate_paraphrases(
    model: PegasusForConditionalGeneration,
    tokenizer: PegasusTokenizer,
    input_text: str,
    num_return_sequences: int,
    num_beams: int,
    source_max_length: int = 60,
    target_max_length: int = 60,
) -> List[str]:

    batch = tokenizer(
        [input_text],
        truncation=True,
        padding="longest",
        max_length=source_max_length,
        return_tensors="pt").to(device)

    translated = model.generate(
        **batch,
        max_length=target_max_length,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
        temperature=1.5)

    tgt_text = tokenizer.batch_decode(
        translated,
        skip_special_tokens=True)

    return tgt_text


def tokenize_string_HF(
        sentences: List[str],
        tokenizer: PreTrainedTokenizerBase
) -> List[str]:
    """Tokenize strings using HF's tokenizer and white-space join them. This is mainly
       used in settings where we need to use HF's tokens in our models which require
       white-space tokenizable."""
    new_sentences = []
    for sentence in sentences:
        token_ids = tokenizer(sentence)["input_ids"]
        new_sentence = tokenizer.convert_ids_to_tokens(token_ids)
        new_sentences.append(" ".join(new_sentence))
        # The line below will recover the original sentence
        # gpt2_tokenizer.convert_tokens_to_string(new_sentence)
    return new_sentences


def _get_compute_perplexities_fn(
        task_name: str,
) -> Tuple[Callable[[List[str]], Tuple[FloatTensor, FloatTensor]],
           PreTrainedModel,
           PreTrainedTokenizerFast]:

    if task_name not in ["snli", "multinli"]:
        raise ValueError

    if task_name == "multinli":
        model_path = "/export/share/Experiments/20210515/checkpoints-from-colab/gpt2-multinli-10epochs/"

    if task_name == "snli":
        model_path = "/export/share/Experiments/20210515/checkpoints-from-colab/gpt2-snli-10epochs/"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelWithLMHead.from_pretrained(model_path)
    model.cuda()
    model.eval()

    @torch.no_grad()
    def _wrapped_fn(sentences: List[str]) -> Tuple[FloatTensor, FloatTensor]:
        return compute_perplexities(
            sentences=sentences,
            model=model,
            tokenizer=tokenizer)

    return _wrapped_fn, model, tokenizer

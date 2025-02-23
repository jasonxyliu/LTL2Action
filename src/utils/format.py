"""
These functions preprocess the observations.
When trying more sophisticated encoding for LTL, we might have to modify this code.
"""
import os
import json
import re
import torch
import torch_ac
import gym
import numpy as np
import utils

from envs import *
from ltl_wrappers import LTLEnv


def get_obss_preprocessor(env, gnn, progression_mode):
    obs_space = env.observation_space
    vocab_space = env.get_propositions()
    vocab = None

    if isinstance(env, LTLEnv):  # LTLEnv Wrapped env
        env = env.unwrapped
        # if isinstance(env, LetterEnv) or isinstance(env, MinigridEnv) or isinstance(env, ZonesEnv):
        if isinstance(env, LetterEnv) or isinstance(env, MinigridEnv):
            if progression_mode == "partial":
                obs_space = {"image": obs_space.spaces["features"].shape, "progress_info": len(vocab_space)}
                def preprocess_obss(obss, device=None):
                    return torch_ac.DictList({
                        "image": preprocess_images([obs["features"] for obs in obss], device=device),
                        "progress_info":  torch.stack([torch.tensor(obs["progress_info"], dtype=torch.float) for obs in obss], dim=0).to(device)
                    })
            else:
                obs_space = {"image": obs_space.spaces["features"].shape, "text": max(22, len(vocab_space) + 10)}
                vocab_space = {"max_size": obs_space["text"], "tokens": vocab_space}

                vocab = Vocabulary(vocab_space)
                tree_builder = utils.ASTBuilder(vocab_space["tokens"])
                def preprocess_obss(obss, device=None):
                    return torch_ac.DictList({
                        "image": preprocess_images([obs["features"] for obs in obss], device=device),
                        "text":  preprocess_texts([obs["text"] for obs in obss], vocab, vocab_space, gnn=gnn, device=device, ast=tree_builder)
                    })

            preprocess_obss.vocab = vocab

        elif isinstance(env, SimpleLTLEnv):
            if progression_mode == "partial":
                obs_space = {"progress_info": len(vocab_space)}
                def preprocess_obss(obss, device=None):
                    return torch_ac.DictList({
                        "progress_info":  torch.stack([torch.tensor(obs["progress_info"], dtype=torch.float) for obs in obss], dim=0).to(device)
                    })
            else:
                obs_space = {"text": max(22, len(vocab_space) + 10)}
                vocab_space = {"max_size": obs_space["text"], "tokens": vocab_space}

                vocab = Vocabulary(vocab_space)
                tree_builder = utils.ASTBuilder(vocab_space["tokens"])

                def preprocess_obss(obss, device=None):
                    return torch_ac.DictList({
                        "text":  preprocess_texts([obs["text"] for obs in obss], vocab, vocab_space, gnn=gnn, device=device, ast=tree_builder)
                    })

            preprocess_obss.vocab = vocab

        else:
            raise ValueError("Unknown observation space: " + str(obs_space))

    elif isinstance(obs_space, gym.spaces.Box):  # check if obs_space is an image space
        obs_space = {"image": obs_space.shape}
        def preprocess_obss(obss, device=None):
            return torch_ac.DictList({
                "image": preprocess_images(obss, device=device)
            })
    else:
        raise ValueError("Unknown observation space: " + str(obs_space))

    return obs_space, preprocess_obss


def preprocess_images(images, device=None):
    # Bug of Pytorch: very slow if not first converted to numpy array
    images = np.array(images)
    return torch.tensor(images, device=device, dtype=torch.float)


def preprocess_texts(texts, vocab, vocab_space, gnn=False, device=None, **kwargs):
    if (gnn):
        return preprocess4gnn(texts, kwargs["ast"], device)
    return preprocess4rnn(texts, vocab, device)


def preprocess4rnn(texts, vocab, device=None):
    """
    This function receives the LTL formulas and convert them into inputs for an RNN
    """
    var_indexed_texts = []
    max_text_len = 0

    for text in texts:
        text = str(text)  # transforming the ltl formula into a string
        tokens = re.findall("([a-z]+)", text.lower())
        var_indexed_text = np.array([vocab[token] for token in tokens])
        var_indexed_texts.append(var_indexed_text)
        max_text_len = max(len(var_indexed_text), max_text_len)

    indexed_texts = np.zeros((len(texts), max_text_len))

    for i, indexed_text in enumerate(var_indexed_texts):
        indexed_texts[i, :len(indexed_text)] = indexed_text

    return torch.tensor(indexed_texts, device=device, dtype=torch.long)


def preprocess4gnn(texts, ast, device=None):
    """
    This function receives the LTL formulas and convert them into inputs for a GNN
    """
    return np.array([[ast(text).to(device)] for text in texts])


class Vocabulary:
    """A mapping from tokens to ids with a capacity of `max_size` words.
    It can be saved in a `vocab.json` file."""
    def __init__(self, vocab_space):
        self.max_size = vocab_space["max_size"]
        self.vocab = {}

        # populate the vocab with the LTL operators
        for item in ['next', 'until', 'and', 'or', 'eventually', 'always', 'not', 'True', 'False']:
            self.__getitem__(item)

        for item in vocab_space["tokens"]:
            self.__getitem__(item)

    def load_vocab(self, vocab):
        self.vocab = vocab

    def __getitem__(self, token):
        if not token in self.vocab.keys():
            if len(self.vocab) >= self.max_size:
                raise ValueError("Maximum vocabulary capacity reached")
            self.vocab[token] = len(self.vocab) + 1
        return self.vocab[token]

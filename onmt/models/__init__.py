"""Module defining models."""
from onmt.models.model_saver import build_model_saver, ModelSaver
from onmt.models.model import NMTModel
from onmt.models.vec_model import VecModel

__all__ = ["build_model_saver", "ModelSaver",
           "NMTModel", "check_sru_requirement", "VecModel"]
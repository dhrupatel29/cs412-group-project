from src.models.baselines import XGB, LogReg
from src.models.ft_transformer import FTTransformer
from src.models.mlp import MLP

REGISTRY = {"logreg": LogReg, "xgb": XGB, "mlp": MLP, "ftt": FTTransformer}

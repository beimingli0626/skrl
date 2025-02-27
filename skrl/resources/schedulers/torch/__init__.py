from skrl.resources.schedulers.torch.kl_adaptive import KLAdaptiveLR
from skrl.resources.schedulers.torch.cosine import CosineLR

KLAdaptiveRL = KLAdaptiveLR  # known typo (compatibility with versions prior to 1.0.0)
CosineRL = CosineLR

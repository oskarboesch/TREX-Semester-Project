from dataclasses import dataclass

@dataclass(frozen=True)
class ExperimentConfig:
    model: str          # "Anatomical" or "Conical"
    placement: str      # "Bifurcation" or "Proximal"
    wire: str           # "Bent" or "Straight"
    technique: str      # "Twist" or "No_Twist"
    clot: str          # "With" or "Without"
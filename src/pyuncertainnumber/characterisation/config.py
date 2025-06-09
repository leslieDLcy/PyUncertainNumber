from dataclasses import dataclass


@dataclass(frozen=True)  # Instances of this class are immutable.
class Config:
    result_path = "./results/"

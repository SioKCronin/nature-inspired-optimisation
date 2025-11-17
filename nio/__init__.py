"""nio: Nature-inspired optimization toolkit.

Currently includes:
- BatAlgorithm (Yang, 2010)
- PhilippineEagleOptimization
- CulturalAlgorithm (Reynolds, 1994)
- IWDCO (Intelligent Water Drops - Continuous Optimization)
- WaterCycleAlgorithm (Eskandar et al., 2012)
- ContractingOptimum benchmark

Usage example::

    from nio import BatAlgorithm, PhilippineEagleOptimization, CulturalAlgorithm, IWDCO
    from nio import WaterCycleAlgorithm, ContractingOptimum

    optimizer = BatAlgorithm()
    solution, value = optimizer.run(200)

    peo = PhilippineEagleOptimization()
    solution, value = peo.run(200)

    ca = CulturalAlgorithm()
    solution, value = ca.run(200)

    iwdco = IWDCO()
    solution, value = iwdco.run(200)

    wca = WaterCycleAlgorithm()
    solution, value = wca.run(200)

    # Dynamic benchmark with contracting optimum
    benchmark = ContractingOptimum(bounds=[(-5.12, 5.12)] * 5, max_iterations=200)
    optimizer = BatAlgorithm(objective=benchmark)
    solution, value = optimizer.run(200)

"""

from .bat import BatAlgorithm, Bat, rastrigin
from .philippine_eagle import PhilippineEagleOptimization, Eagle, Operator, Phase
from .cultural import CulturalAlgorithm, Individual, BeliefSpace, NormativeKnowledge, SituationalKnowledge
from .iwd_co import IWDCO, WaterDrop, LiquidType, SoilType
from .water_cycle import WaterCycleAlgorithm, WaterBody, LiquidType as WCA_LiquidType
from .benchmarks import ContractingOptimum, contracting_optimum

__all__ = [
    "BatAlgorithm",
    "Bat",
    "rastrigin",
    "PhilippineEagleOptimization",
    "Eagle",
    "Operator",
    "Phase",
    "CulturalAlgorithm",
    "Individual",
    "BeliefSpace",
    "NormativeKnowledge",
    "SituationalKnowledge",
    "IWDCO",
    "WaterDrop",
    "LiquidType",
    "SoilType",
    "WaterCycleAlgorithm",
    "WaterBody",
    "WCA_LiquidType",
    "ContractingOptimum",
    "contracting_optimum",
]

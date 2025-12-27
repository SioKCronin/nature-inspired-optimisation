"""nio: Nature-inspired optimization toolkit.

Currently includes:
- BatAlgorithm (Yang, 2010)
- FireflyAlgorithm (Yang, 2009)
- PhilippineEagleOptimization
- CulturalAlgorithm (Reynolds, 1994)
- IWDCO (Intelligent Water Drops - Continuous Optimization)
- WaterCycleAlgorithm (Eskandar et al., 2012)
- PPSO (Proactive Particle Swarm Optimization, Cheng & Jin, 2015)
- ContractingOptimum benchmark

Usage example::

    from nio import BatAlgorithm, FireflyAlgorithm, PhilippineEagleOptimization, CulturalAlgorithm, IWDCO
    from nio import WaterCycleAlgorithm, PPSO, ContractingOptimum

    optimizer = BatAlgorithm()
    solution, value = optimizer.run(200)

    fa = FireflyAlgorithm(alpha=0.2, beta0=1.0, gamma=1.0)
    solution, value = fa.run(200)

    peo = PhilippineEagleOptimization()
    solution, value = peo.run(200)

    ca = CulturalAlgorithm()
    solution, value = ca.run(200)

    iwdco = IWDCO()
    solution, value = iwdco.run(200)

    wca = WaterCycleAlgorithm()
    solution, value = wca.run(200)

    ppso = PPSO(proactive_ratio=0.25)
    solution, value = ppso.run(200)

    # Dynamic benchmark with contracting optimum
    benchmark = ContractingOptimum(bounds=[(-5.12, 5.12)] * 5, max_iterations=200)
    optimizer = BatAlgorithm(objective=benchmark)
    solution, value = optimizer.run(200)

"""

from .bat import BatAlgorithm, Bat, rastrigin
from .firefly import FireflyAlgorithm, Firefly
from .philippine_eagle import PhilippineEagleOptimization, Eagle, Operator, Phase
from .cultural import CulturalAlgorithm, Individual, BeliefSpace, NormativeKnowledge, SituationalKnowledge
from .iwd_co import IWDCO, WaterDrop, LiquidType, SoilType
from .water_cycle import WaterCycleAlgorithm, WaterBody, LiquidType as WCA_LiquidType
from .ppso import PPSO, Particle
from .benchmarks import ContractingOptimum, contracting_optimum

__all__ = [
    "BatAlgorithm",
    "Bat",
    "rastrigin",
    "FireflyAlgorithm",
    "Firefly",
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
    "PPSO",
    "Particle",
    "ContractingOptimum",
    "contracting_optimum",
]

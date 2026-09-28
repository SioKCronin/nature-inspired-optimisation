"""nio: Nature-inspired optimization toolkit.

Continuous optimizers share one contract::

    from nio import GreyWolfOptimizer

    optimizer = GreyWolfOptimizer(bounds=[(-5.12, 5.12)] * 5, population_size=30, seed=1)
    position, value = optimizer.run(200)

The same classes are available by short name from :data:`nio.registry.OPTIMIZERS`.
Collective motion (Boids, Vicsek) and grid pathfinders (ant colony, river
formation) sit beside them. Lead drones use :class:`nio.flight.LeaderSwarmEnv`.
"""

from .abc import ArtificialBeeColony
from .acor import AntColonyOptimization
from .ais import ClonalSelection
from .altruistic import AltruisticPopulationAlgorithm
from .animal_migration import AnimalMigrationOptimization
from .artificial_algae import ArtificialAlgaeAlgorithm
from .artificial_chemical import ArtificialChemicalReactionOptimization
from .artificial_ecosystem import ArtificialEcosystemAlgorithm
from .bacteria_chemotaxis import BacterialChemotaxis
from .bacterial_colony import BacterialColonyOptimization
from .bacterial_evolutionary import BacterialEvolutionaryAlgorithm
from .bacterial_foraging import BacterialForagingOptimization
from .bat import Bat, BatAlgorithm, rastrigin
from .benchmarks import ContractingOptimum, contracting_optimum
from .biogeography import BiogeographyBasedOptimization
from .bird_mating import BirdMatingOptimizer
from .black_hole import BlackHoleAlgorithm
from .bull import BullOptimizationAlgorithm
from .collective_animal import CollectiveAnimalBehavior
from .cuckoo import CuckooSearch
from .cultural import BeliefSpace, CulturalAlgorithm, Individual, NormativeKnowledge, SituationalKnowledge
from .cuttlefish import CuttlefishAlgorithm
from .differential_evolution import DifferentialEvolution
from .dragonfly import Dragonfly, DragonflyAlgorithm
from .elephant import ElephantHerdingOptimization
from .firefly import Firefly, FireflyAlgorithm
from .fireworks import FireworksAlgorithm
from .fish_school import ArtificialFishSchool
from .flower_pollination import FlowerPollinationAlgorithm
from .forest import ForestOptimizationAlgorithm
from .gases_brownian import GasesBrownianMotionOptimization
from .genetic import GeneticAlgorithm
from .grey_wolf import GreyWolfOptimizer
from .gsa import GravitationalSearchAlgorithm
from .gso import GlowwormSwarmOptimization
from .hbmo import HoneyBeeMatingOptimization
from .honey_bee_marriage import MarriageInHoneyBees
from .invasive_weed import InvasiveWeedOptimization
from .iwd_co import IWDCO, LiquidType, SoilType, WaterDrop
from .krill import KrillHerd
from .lion import LionOptimizationAlgorithm
from .memetic import MemeticAlgorithm
from .mine_blast import MineBlastAlgorithm
from .motion import boids_step, vicsek_step
from .optics import OpticsInspiredOptimization
from .pathfinding import AntColonyPath, OccupancyGrid, RiverFormationDynamics
from .philippine_eagle import Eagle, Operator, Phase, PhilippineEagleOptimization
from .plant_propagation import PlantPropagationAlgorithm
from .ppso import PPSO, Particle
from .pso import ParticleSwarmOptimization
from .raven import RavenRoostingOptimization
from .registry import OPTIMIZERS, get_optimizer
from .roach import RoachInfestationOptimization
from .sco import SocialCognitiveOptimization
from .shuffled_frog import ShuffledFrogLeaping
from .social_spider import SocialSpiderOptimization
from .spider_monkey import SpiderMonkeyOptimization
from .spiral import SpiralDynamicAlgorithm
from .strawberry import StrawberryAlgorithm
from .water_cycle import WaterBody, WaterCycleAlgorithm
from .water_cycle import LiquidType as WCA_LiquidType
from .water_wave import WaterWaveOptimization
from .flight import LeaderSwarmEnv, LinearPolicy, plan_actions, train_policy

__all__ = [
    "OPTIMIZERS",
    "get_optimizer",
    "AntColonyOptimization",
    "AntColonyPath",
    "ArtificialAlgaeAlgorithm",
    "ArtificialBeeColony",
    "ArtificialChemicalReactionOptimization",
    "ArtificialEcosystemAlgorithm",
    "ArtificialFishSchool",
    "AltruisticPopulationAlgorithm",
    "AnimalMigrationOptimization",
    "BacterialChemotaxis",
    "BacterialColonyOptimization",
    "BacterialEvolutionaryAlgorithm",
    "BacterialForagingOptimization",
    "BatAlgorithm",
    "Bat",
    "rastrigin",
    "BeliefSpace",
    "BiogeographyBasedOptimization",
    "BirdMatingOptimizer",
    "BlackHoleAlgorithm",
    "BullOptimizationAlgorithm",
    "ClonalSelection",
    "CollectiveAnimalBehavior",
    "ContractingOptimum",
    "contracting_optimum",
    "CuckooSearch",
    "CulturalAlgorithm",
    "CuttlefishAlgorithm",
    "DifferentialEvolution",
    "DragonflyAlgorithm",
    "Dragonfly",
    "Eagle",
    "ElephantHerdingOptimization",
    "FireflyAlgorithm",
    "Firefly",
    "FireworksAlgorithm",
    "FlowerPollinationAlgorithm",
    "ForestOptimizationAlgorithm",
    "GasesBrownianMotionOptimization",
    "GeneticAlgorithm",
    "GlowwormSwarmOptimization",
    "GravitationalSearchAlgorithm",
    "GreyWolfOptimizer",
    "HoneyBeeMatingOptimization",
    "IWDCO",
    "Individual",
    "InvasiveWeedOptimization",
    "KrillHerd",
    "LeaderSwarmEnv",
    "LinearPolicy",
    "LionOptimizationAlgorithm",
    "LiquidType",
    "MarriageInHoneyBees",
    "MemeticAlgorithm",
    "MineBlastAlgorithm",
    "NormativeKnowledge",
    "OccupancyGrid",
    "Operator",
    "OpticsInspiredOptimization",
    "PPSO",
    "Particle",
    "ParticleSwarmOptimization",
    "Phase",
    "PhilippineEagleOptimization",
    "PlantPropagationAlgorithm",
    "RavenRoostingOptimization",
    "RiverFormationDynamics",
    "RoachInfestationOptimization",
    "ShuffledFrogLeaping",
    "SituationalKnowledge",
    "SocialCognitiveOptimization",
    "SocialSpiderOptimization",
    "SoilType",
    "SpiderMonkeyOptimization",
    "SpiralDynamicAlgorithm",
    "StrawberryAlgorithm",
    "WCA_LiquidType",
    "WaterBody",
    "WaterCycleAlgorithm",
    "WaterDrop",
    "WaterWaveOptimization",
    "boids_step",
    "plan_actions",
    "train_policy",
    "vicsek_step",
]

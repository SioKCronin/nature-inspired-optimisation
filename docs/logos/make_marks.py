"""Draw the silhouette marks used in the README grid.

Each mark is a small ink drawing on a paper tile, so the set stays
legible on both the light and dark GitHub themes. Run this file to
regenerate docs/logos/<slug>.svg.
"""

from __future__ import annotations

import math
from pathlib import Path

INK = "#1a1814"
PLATE = "#f3efe6"
OUT = Path(__file__).resolve().parent

# slug, abbreviation, name, import name, paper
CATALOG = [
    ("ga", "GA", "Genetic Algorithm", "GeneticAlgorithm", "https://doi.org/10.7551/mitpress/1090.001.0001"),
    ("pso", "PSO", "Particle Swarm", "ParticleSwarmOptimization", "https://doi.org/10.1109/ICNN.1995.488968"),
    ("ais", "AIS", "Artificial Immune System", "ClonalSelection", "https://doi.org/10.1109/TEVC.2002.1011539"),
    ("boids", "Boids", "Boids", "boids_step", "https://doi.org/10.1145/37402.37406"),
    ("ma", "MA", "Memetic Algorithm", "MemeticAlgorithm", "https://web.archive.org/web/20060926195738id_/http://www.densis.fee.unicamp.br/~moscato/papers/bigone.ps"),
    ("aco", "ACO", "Ant Colony", "AntColonyOptimization", "https://doi.org/10.1109/3477.484436"),
    ("ca", "CA", "Cultural Algorithm", "CulturalAlgorithm", "https://doi.org/10.1142/9789814534116"),
    ("spp", "SPP", "Self-propelled Particles", "vicsek_step", "https://doi.org/10.1103/PhysRevLett.75.1226"),
    ("de", "DE", "Differential Evolution", "DifferentialEvolution", "https://doi.org/10.1023/A:1008202821328"),
    ("bfo", "BFO", "Bacterial Foraging", "BacterialForagingOptimization", "https://doi.org/10.1109/MCS.2002.1004010"),
    ("mhb", "MHB", "Marriage in Honey Bees", "MarriageInHoneyBees", "https://doi.org/10.1109/CEC.2001.934391"),
    ("afs", "AFS", "Artificial Fish School", "ArtificialFishSchool", "http://en.cnki.com.cn/Article_en/CJFDTOTAL-XTLL200211006.htm"),
    ("bc", "BC", "Bacterial Chemotaxis", "BacterialChemotaxis", "https://doi.org/10.1109/4235.985689"),
    ("sco", "SCO", "Social Cognitive Optimization", "SocialCognitiveOptimization", "https://doi.org/10.1109/icmlc.2002.1174487"),
    ("abc", "ABC", "Artificial Bee Colony", "ArtificialBeeColony", "https://abc.erciyes.edu.tr/pub/tr06_2005.pdf"),
    ("gso", "GSO", "Glowworm Swarm", "GlowwormSwarmOptimization", "https://doi.org/10.1007/s11721-008-0021-5"),
    ("hbmo", "HBMO", "Honey-Bee Mating", "HoneyBeeMatingOptimization", "https://doi.org/10.1016/j.jfranklin.2006.06.001"),
    ("iwo", "IWO", "Invasive Weed", "InvasiveWeedOptimization", "https://doi.org/10.1016/j.ecoinf.2006.07.003"),
    ("sfla", "SFLA", "Shuffled Frog Leaping", "ShuffledFrogLeaping", "https://doi.org/10.1080/03052150500384759"),
    ("iwd", "IWD-CO", "Intelligent Water Drops", "IWDCO", "https://doi.org/10.1016/j.sbspro.2012.01.033"),
    ("rfd", "RFD", "River Formation Dynamics", "RiverFormationDynamics", "https://doi.org/10.1007/978-3-540-73554-0_16"),
    ("bbo", "BBO", "Biogeography-Based Optimization", "BiogeographyBasedOptimization", "https://doi.org/10.1109/TEVC.2008.919004"),
    ("rio", "RIO", "Roach Infestation", "RoachInfestationOptimization", "https://doi.org/10.1109/SIS.2008.4668317"),
    ("bea", "BEA", "Bacterial Evolutionary Algorithm", "BacterialEvolutionaryAlgorithm", "https://doi.org/10.1109/91.797983"),
    ("cs", "CS", "Cuckoo Search", "CuckooSearch", "https://doi.org/10.1109/NABIC.2009.5393690"),
    ("fa", "FA", "Firefly Algorithm", "FireflyAlgorithm", "https://arxiv.org/abs/1003.1466"),
    ("gsa", "GSA", "Gravitational Search", "GravitationalSearchAlgorithm", "https://doi.org/10.1016/j.ins.2009.03.004"),
    ("bat", "BA", "Bat Algorithm", "BatAlgorithm", "https://arxiv.org/abs/1004.4170"),
    ("peo", "PEOA", "Philippine Eagle", "PhilippineEagleOptimization", "https://arxiv.org/abs/2112.10318"),
    ("fwa", "FWA", "Fireworks Algorithm", "FireworksAlgorithm", "https://doi.org/10.1007/978-3-642-13495-1_44"),
    ("apa", "APA", "Altruistic Population", "AltruisticPopulationAlgorithm", "https://doi.org/10.1016/j.matcom.2023.03.004"),
    ("sda", "SDA", "Spiral Dynamics", "SpiralDynamicAlgorithm", "https://doi.org/10.20965/jaciii.2011.p1116"),
    ("strawberry", "Strawberry", "Strawberry Algorithm", "StrawberryAlgorithm", "https://repository.essex.ac.uk/9974/1/paper.pdf"),
    ("aaa", "AAA", "Artificial Algae", "ArtificialAlgaeAlgorithm", "https://doi.org/10.1016/j.asoc.2015.03.003"),
    ("bco", "BCO", "Bacterial Colony", "BacterialColonyOptimization", "https://doi.org/10.1155/2012/698057"),
    ("fpa", "FPA", "Flower Pollination", "FlowerPollinationAlgorithm", "https://arxiv.org/abs/1312.5673"),
    ("kh", "KH", "Krill Herd", "KrillHerd", "https://doi.org/10.1016/j.cnsns.2012.05.010"),
    ("wca", "WCA", "Water Cycle", "WaterCycleAlgorithm", "https://doi.org/10.1016/j.compstruc.2012.07.010"),
    ("ppso", "PPSO", "Proactive Particle Swarm", "PPSO", "https://doi.org/10.1109/FUZZ-IEEE.2015.7337957"),
    ("da", "DA", "Dragonfly Algorithm", "DragonflyAlgorithm", "https://doi.org/10.1007/s00521-015-1920-1"),
    ("bha", "BHA", "Black Hole", "BlackHoleAlgorithm", "https://doi.org/10.1016/j.ins.2012.08.023"),
    ("cfa", "CFA", "Cuttlefish Algorithm", "CuttlefishAlgorithm", "https://www.ijser.org/onlineResearchPaperViewer.aspx?Cuttlefish-Algorithm-A-Novel-Bio-Inspired-Optimization-Algorithm.pdf"),
    ("gbmo", "GBMO", "Gases Brownian Motion", "GasesBrownianMotionOptimization", "https://doi.org/10.1016/j.asoc.2012.03.068"),
    ("mba", "MBA", "Mine Blast", "MineBlastAlgorithm", "https://doi.org/10.1016/j.asoc.2012.11.026"),
    ("ppa", "PPA", "Plant Propagation", "PlantPropagationAlgorithm", "https://repository.essex.ac.uk/9974/1/paper.pdf"),
    ("sso", "SSO", "Social Spider", "SocialSpiderOptimization", "https://doi.org/10.1016/j.eswa.2013.05.041"),
    ("smo", "SMO", "Spider Monkey", "SpiderMonkeyOptimization", "https://doi.org/10.1007/s12293-013-0128-0"),
    ("amo", "AMO", "Animal Migration", "AnimalMigrationOptimization", "https://doi.org/10.1007/s00521-013-1433-8"),
    ("aea", "AEA", "Artificial Ecosystem", "ArtificialEcosystemAlgorithm", "https://doi.org/10.1007/s00521-019-04452-x"),
    ("bmo", "BMO", "Bird Mating", "BirdMatingOptimizer", "https://doi.org/10.1016/j.cnsns.2013.08.027"),
    ("foa", "FOA", "Forest Optimization", "ForestOptimizationAlgorithm", "https://doi.org/10.1016/j.eswa.2014.05.009"),
    ("gwo", "GWO", "Grey Wolf Optimizer", "GreyWolfOptimizer", "https://doi.org/10.1016/j.advengsoft.2013.12.007"),
    ("loa", "LOA", "Lion Optimization", "LionOptimizationAlgorithm", "https://doi.org/10.1016/j.jcde.2015.06.003"),
    ("oio", "OIO", "Optics Inspired Optimization", "OpticsInspiredOptimization", "https://doi.org/10.1016/j.cor.2014.10.011"),
    ("rroa", "RROA", "Raven Roosting", "RavenRoostingOptimization", "https://doi.org/10.1007/s00500-014-1520-5"),
    ("wwo", "WWO", "Water Wave", "WaterWaveOptimization", "https://doi.org/10.1016/j.cor.2014.10.008"),
    ("cab", "CAB", "Collective Animal Behavior", "CollectiveAnimalBehavior", "https://doi.org/10.1155/2012/638275"),
    ("acroa", "ACROA", "Artificial Chemical Reaction", "ArtificialChemicalReactionOptimization", "https://doi.org/10.1016/j.eswa.2011.04.126"),
    ("bull", "BOA", "Bull Optimization", "BullOptimizationAlgorithm", "https://doi.org/10.3906/elk-1307-123"),
    ("eho", "EHO", "Elephant Herding", "ElephantHerdingOptimization", "https://doi.org/10.1109/iscbi.2015.8"),
]


def svg(title: str, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="96" height="96" viewBox="0 0 96 96" role="img" aria-label="{title}">\n'
        f"  <title>{title}</title>\n"
        f'  <rect width="96" height="96" rx="22" fill="{PLATE}"/>\n'
        f"{body}\n"
        "</svg>\n"
    )


def P(d: str, fill: str = INK, rule: str | None = None) -> str:
    extra = f' fill-rule="{rule}"' if rule else ""
    return f'  <path fill="{fill}"{extra} d="{d}"/>\n'


def S(d: str, w: float = 4, cap: str = "round") -> str:
    return (
        f'  <path fill="none" stroke="{INK}" stroke-width="{w}" '
        f'stroke-linecap="{cap}" stroke-linejoin="round" d="{d}"/>\n'
    )


def C(cx: float, cy: float, r: float) -> str:
    return f'  <circle fill="{INK}" cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}"/>\n'


def E(cx: float, cy: float, rx: float, ry: float, rot: float = 0, ox: float | None = None, oy: float | None = None) -> str:
    tr = ""
    if rot:
        tr = f' transform="rotate({rot:.1f} {ox if ox is not None else cx:.1f} {oy if oy is not None else cy:.1f})"'
    return f'  <ellipse fill="{INK}" cx="{cx:.1f}" cy="{cy:.1f}" rx="{rx:.1f}" ry="{ry:.1f}"{tr}/>\n'


def R(x: float, y: float, w: float, h: float, rx: float = 0) -> str:
    return f'  <rect fill="{INK}" x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx:.1f}"/>\n'


def G(points: list[tuple[float, float]]) -> str:
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'  <polygon fill="{INK}" points="{pts}"/>\n'


def bird(x: float, y: float, s: float = 1.0, flip: bool = False) -> str:
    """Open wings, facing right unless flip is set. Body sits at (x, y)."""
    d = -1 if flip else 1
    upper = (
        f"M{x:.1f} {y:.1f} L{x + d * 26 * s:.1f} {y - 13 * s:.1f} "
        f"L{x + d * 15 * s:.1f} {y - 1.5 * s:.1f} Z"
    )
    lower = (
        f"M{x:.1f} {y:.1f} L{x + d * 22 * s:.1f} {y + 15 * s:.1f} "
        f"L{x + d * 12 * s:.1f} {y + 3 * s:.1f} Z"
    )
    return P(upper) + P(lower)


def fish(x: float, y: float, s: float = 1.0) -> str:
    return (
        G([(x - 14 * s, y), (x - 24 * s, y - 7 * s), (x - 24 * s, y + 7 * s)])
        + E(x, y, 14 * s, 6.5 * s)
    )


def bee(x: float, y: float, s: float = 1.0) -> str:
    """Side-view bee facing left. Stripes are the paper showing through."""
    stripe = (
        f'  <rect fill="{PLATE}" x="{x - 2 * s:.1f}" y="{{yy}}" width="{16 * s:.1f}" '
        f'height="{2.3 * s:.1f}" rx="0.6"/>\n'
    )
    return (
        E(x - 2 * s, y - 9 * s, 8 * s, 4 * s, -18)
        + E(x + 8 * s, y - 7 * s, 7 * s, 3.4 * s, -8)
        + E(x + 6 * s, y + 1 * s, 12 * s, 8 * s)
        + stripe.format(yy=f"{y - 3.4 * s:.1f}")
        + stripe.format(yy=f"{y + 1.2 * s:.1f}")
        + C(x - 10 * s, y + 0.5 * s, 5.2 * s)
        + C(x - 16 * s, y + 1.2 * s, 3.6 * s)
        + S(
            f"M{x - 18 * s:.1f} {y - 1 * s:.1f} "
            f"C{x - 24 * s:.1f} {y - 6 * s:.1f} {x - 22 * s:.1f} {y - 12 * s:.1f} {x - 16 * s:.1f} {y - 10 * s:.1f}",
            2,
        )
    )


def rod(x: float, y: float, w: float = 34, h: float = 14, rot: float = 0) -> str:
    tr = f' transform="rotate({rot:.1f} {x + w / 2:.1f} {y + h / 2:.1f})"' if rot else ""
    return f'  <rect fill="{INK}" x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{h / 2:.1f}"{tr}/>\n'


def flagella(x: float, y: float) -> str:
    return (
        S(f"M{x:.1f} {y - 4:.1f} C{x - 12:.1f} {y - 14:.1f} {x - 18:.1f} {y - 8:.1f} {x - 14:.1f} {y + 2:.1f}", 2.6)
        + S(f"M{x:.1f} {y:.1f} C{x - 14:.1f} {y - 2:.1f} {x - 18:.1f} {y + 6:.1f} {x - 12:.1f} {y + 12:.1f}", 2.6)
        + S(f"M{x:.1f} {y + 4:.1f} C{x - 10:.1f} {y + 12:.1f} {x - 16:.1f} {y + 16:.1f} {x - 8:.1f} {y + 20:.1f}", 2.6)
    )


def chevron(x: float, y: float, s: float = 1.0) -> str:
    return P(
        f"M{x:.1f} {y:.1f} l{7 * s:.1f} {-4.5 * s:.1f} l{7 * s:.1f} {4.5 * s:.1f} "
        f"l{-2.4 * s:.1f} 0 l{-4.6 * s:.1f} {-2.6 * s:.1f} l{-4.6 * s:.1f} {2.6 * s:.1f} z"
    )


def flower(cx: float, cy: float, n: int, reach: float, fat: float) -> str:
    parts = []
    for i in range(n):
        ang = -90 + i * (360 / n)
        parts.append(E(cx, cy - reach * 0.55, fat, reach, ang, cx, cy))
    parts.append(C(cx, cy, fat * 0.85))
    return "".join(parts)


def burst(cx: float, cy: float, n: int, inner: float, outer: float, start: float = -90) -> str:
    parts = []
    for i in range(n):
        ang = start + i * (360 / n)
        parts.append(E(cx, cy - (inner + outer) / 2, outer * 0.22, (outer - inner) / 2, ang, cx, cy))
    parts.append(C(cx, cy, inner))
    return "".join(parts)


def spiral() -> str:
    pts = []
    for i in range(72):
        t = i / 6.4
        radius = 3 + t * 2.35
        pts.append((48 + radius * math.cos(t), 48 + radius * math.sin(t)))
    d = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in pts)
    return S(d, 4.5)


def pine(x: float, y: float, h: float) -> str:
    w = h * 0.72
    return (
        G([(x, y - h), (x - w / 2, y - h * 0.42), (x + w / 2, y - h * 0.42)])
        + G([(x, y - h * 0.62), (x - w * 0.62, y), (x + w * 0.62, y)])
        + R(x - 2.2, y, 4.4, h * 0.18, 1)
    )


def marks() -> dict[str, str]:
    m: dict[str, str] = {}
    m["ga"] = (
        S("M34 22 C34 34 64 34 64 46 C64 58 34 58 34 72", 5)
        + S("M62 22 C62 34 32 34 32 46 C32 58 62 58 62 72", 5)
        + R(40, 31, 16, 4.5, 2)
        + R(39, 45, 18, 4.5, 2)
        + R(40, 59, 16, 4.5, 2)
    )
    m["pso"] = bird(20, 58, 0.82) + bird(34, 34, 1.0) + bird(50, 56, 0.72)
    m["ais"] = P("M39 78 V50 L22 28 h11 l15 18 15-18 h11 L57 50 V78 Z")
    m["boids"] = bird(18, 48, 0.9) + bird(42, 30, 0.9) + bird(42, 64, 0.9)
    m["ma"] = (
        C(48, 38, 16)
        + R(43, 52, 10, 6, 1)
        + R(40, 58, 16, 4.5, 1.5)
        + R(42, 64.5, 12, 3.5, 1)
        + R(43.5, 70, 9, 3.5, 1)
    )
    m["aco"] = (
        E(64, 54, 13, 10)
        + E(44, 52, 9, 7)
        + C(30, 49, 6.5)
        + S("M26 45 C18 36 16 28 22 22", 2.5)
        + S("M30 44 C26 34 32 26 38 24", 2.5)
        + S("M40 46 C30 38 22 40 16 46", 3)
        + S("M42 52 C32 54 24 60 18 66", 3)
        + S("M46 58 C40 68 48 74 56 70", 3)
    )
    m["ca"] = (
        G([(48, 18), (20, 40), (76, 40)])
        + R(26, 42, 8, 28, 1)
        + R(44, 42, 8, 28, 1)
        + R(62, 42, 8, 28, 1)
        + R(18, 70, 60, 6, 1)
    )
    m["spp"] = ""
    for px, py in ((30, 58), (46, 44), (62, 30), (28, 34), (64, 62)):
        m["spp"] += C(px, py, 4.2) + G([(px + 3, py - 2), (px + 14, py - 9), (px + 8, py - 1)])
    m["de"] = (
        C(28, 64, 5.5)
        + C(62, 28, 5.5)
        + C(72, 66, 5.5)
        + S("M33 59 L56 34", 4)
        + G([(56, 34), (48, 36), (54, 44)])
    )
    m["bfo"] = rod(30, 40, 40, 16) + flagella(32, 48)
    m["mhb"] = bee(40, 40, 0.78) + bee(62, 62, 0.62)
    m["afs"] = fish(46, 30, 0.85) + fish(54, 48, 1.05) + fish(42, 66, 0.78)
    m["bc"] = rod(18, 42, 28, 13) + C(56, 48, 3.2) + C(66, 44, 4.4) + C(78, 38, 6)
    m["sco"] = (
        C(32, 36, 9)
        + P("M18 74 C18 58 22 52 32 52 C42 52 46 58 46 74 Z")
        + C(66, 36, 9)
        + P("M52 74 C52 58 56 52 66 52 C76 52 80 58 80 74 Z")
        + G([(48, 30), (51, 36), (48, 42), (45, 36)])
    )
    m["abc"] = bee(52, 50, 1.15)
    m["gso"] = S("M22 68 C28 54 34 60 42 48 C50 36 56 44 66 34 C72 28 74 26 76 22", 8) + C(78, 18, 7)
    m["hbmo"] = ""
    for hx, hy in ((48, 36), (35, 58), (61, 58)):
        pts = []
        for i in range(6):
            a = math.radians(60 * i - 30)
            pts.append((hx + 14 * math.cos(a), hy + 14 * math.sin(a)))
        d = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in pts) + " Z"
        m["hbmo"] += S(d, 4)
    iwo_head = C(48, 24, 7)
    for i in range(10):
        ang = math.radians(i * 36 - 90)
        iwo_head += S(
            f"M{48 + 7 * math.cos(ang):.1f} {24 + 7 * math.sin(ang):.1f} "
            f"L{48 + 13 * math.cos(ang):.1f} {24 + 13 * math.sin(ang):.1f}",
            2.4,
        )
    m["iwo"] = (
        iwo_head
        + S("M48 32 V72", 3.5)
        + G([(48, 48), (22, 58), (30, 52), (20, 46), (34, 46), (48, 36)])
        + G([(48, 48), (74, 58), (66, 52), (76, 46), (62, 46), (48, 36)])
        + G([(48, 64), (26, 74), (36, 68), (28, 62), (40, 64)])
        + G([(48, 64), (70, 74), (60, 68), (68, 62), (56, 64)])
    )
    m["sfla"] = (
        E(38, 58, 13, 8, -35)
        + E(26, 48, 9, 4.5, 50)
        + E(54, 46, 15, 11)
        + C(68, 40, 8)
        + C(73, 34, 3.4)
        + S("M60 54 C70 60 76 68 70 72", 4)
    )
    m["iwd"] = P("M48 16 C48 16 26 42 26 58 a22 22 0 0 0 44 0 C70 42 48 16 48 16 Z")
    m["rfd"] = P(
        "M44 14 C36 26 58 30 52 42 C46 54 34 52 30 62 L22 78 H40 L46 64 "
        "C50 56 58 58 60 48 C64 36 52 30 54 18 C50 12 46 12 44 14 Z"
    ) + P("M58 62 C66 66 74 64 78 74 H58 C60 68 58 64 58 62 Z")
    m["bbo"] = (
        G([(16, 72), (34, 36), (52, 72)])
        + G([(46, 72), (66, 44), (84, 72)])
        + S("M12 74 H86", 4)
    )
    m["rio"] = (
        E(50, 52, 20, 12)
        + C(30, 50, 7)
        + S("M26 46 C14 34 8 30 12 22", 2.6)
        + S("M28 44 C18 36 16 28 22 22", 2.6)
        + S("M40 42 C34 30 28 28 22 34", 3)
        + S("M46 40 C46 28 52 24 60 30", 3)
        + S("M58 44 C70 32 78 34 82 42", 3)
        + S("M40 62 C32 74 26 76 20 70", 3)
        + S("M52 64 C52 76 60 78 66 70", 3)
        + S("M62 60 C74 70 80 66 78 56", 3)
    )
    m["bea"] = (
        rod(16, 46, 40, 16)
        + S("M60 30 C60 40 80 40 80 50 C80 60 60 60 60 70", 3.6)
        + S("M80 30 C80 40 60 40 60 50 C60 60 80 60 80 70", 3.6)
    )
    m["cs"] = (
        P("M22 64 C28 56 40 54 52 58 C62 54 74 58 78 66 C68 62 56 64 48 68 C38 64 28 66 22 64 Z")
        + E(40, 62, 6, 4.5)
        + bird(46, 48, 0.85)
    )
    m["fa"] = (
        E(44, 34, 15, 5.5, -18)
        + E(58, 32, 12, 4.5, 16)
        + E(40, 50, 14, 8)
        + C(64, 54, 13)
        + C(26, 48, 6)
        + S("M22 44 C14 36 16 28 24 30", 2.2)
    )
    m["gsa"] = C(38, 52, 16) + C(68, 36, 9)
    m["bat"] = (
        P(
            "M48 46 C45 34 38 28 32 30 C22 32 16 42 18 52 C24 46 32 46 38 50 "
            "L34 62 L42 52 L48 66 L54 52 L62 62 L58 50 C64 46 72 46 78 52 "
            "C80 42 74 32 64 30 C58 28 51 34 48 46 Z"
        )
        + G([(40, 32), (36, 18), (46, 30)])
        + G([(56, 32), (60, 18), (50, 30)])
    )
    m["peo"] = (
        P("M42 50 L10 38 L18 50 L8 54 L22 54 L16 66 L32 54 L42 58 Z")
        + P("M54 50 L86 38 L78 50 L88 54 L74 54 L80 66 L64 54 L54 58 Z")
        + E(48, 52, 8, 13)
        + G([(41, 60), (48, 78), (55, 60)])
        + C(48, 32, 8)
        + G([(40, 28), (44, 14), (49, 26)])
        + G([(47, 24), (52, 12), (56, 26)])
        + G([(54, 32), (68, 36), (54, 40)])
    )
    m["fwa"] = burst(48, 48, 10, 5, 30)
    m["apa"] = (
        C(30, 30, 6.5)
        + S("M30 37 V58 M30 44 H46 M30 58 L22 72 M30 58 L38 72", 5)
        + C(50, 42, 5)
        + C(68, 30, 6.5)
        + S("M68 37 V58 M68 44 H54 M68 58 L60 72 M68 58 L76 72", 5)
    )
    m["sda"] = spiral()
    m["strawberry"] = (
        P("M48 34 C36 34 28 46 28 58 C28 72 36 80 48 80 C60 80 68 72 68 58 C68 46 60 34 48 34 Z")
        + "".join(
            f'  <circle fill="{PLATE}" cx="{cx}" cy="{cy}" r="2.1"/>\n'
            for cx, cy in ((40, 52), (48, 58), (56, 52), (42, 66), (54, 66), (48, 46))
        )
        + G([(48, 22), (34, 40), (42, 36), (48, 44), (54, 36), (62, 40)])
    )
    m["aaa"] = (
        S("M30 74 C28 60 38 58 32 46 C26 34 34 30 30 18", 5)
        + S("M48 76 C50 58 40 56 48 42 C56 28 46 24 50 16", 5.5)
        + S("M66 74 C64 62 74 56 66 44 C58 32 70 28 64 18", 5)
        + S("M22 76 H74", 3.5)
    )
    m["bco"] = (
        rod(22, 28, 30, 12, -18)
        + rod(40, 36, 32, 12, 16)
        + rod(18, 48, 28, 11, 8)
        + rod(44, 52, 30, 12, -12)
        + rod(30, 62, 26, 10, 20)
    )
    m["fpa"] = flower(48, 50, 5, 18, 7)
    m["kh"] = (
        P(
            "M22 60 C28 42 48 36 66 42 C58 48 46 50 34 58 "
            "C46 60 60 54 70 48 C76 56 62 70 42 68 C30 66 22 64 22 60 Z"
        )
        + G([(68, 44), (84, 34), (80, 48), (84, 60), (68, 52)])
        + S("M26 54 C16 48 14 38 20 32", 2.4)
        + S("M30 50 C22 44 24 34 30 30", 2.2)
    )
    m["wca"] = (
        C(34, 30, 10)
        + C(48, 26, 12)
        + C(62, 30, 9)
        + P("M30 36 H66 C64 42 58 44 48 44 C38 44 32 42 30 36 Z")
        + P("M40 50 C40 50 36 58 40 62 a4 4 0 0 0 8 0 C50 58 46 50 46 50 Z")
        + P("M54 48 C54 48 50 58 54 64 a4.2 4.2 0 0 0 8.4 0 C66 58 62 48 62 48 Z")
        + E(48, 74, 22, 5)
    )
    m["ppso"] = C(26, 48, 8) + G([(38, 40), (56, 40), (56, 32), (74, 48), (56, 64), (56, 56), (38, 56)])
    m["da"] = (
        E(30, 40, 16, 6, -28)
        + E(66, 40, 16, 6, 28)
        + E(32, 58, 14, 5.5, 22)
        + E(64, 58, 14, 5.5, -22)
        + R(44, 24, 8, 50, 4)
        + C(48, 22, 6)
    )
    m["bha"] = P(
        "M18 52 A30 13 0 1 1 78 52 A30 13 0 1 1 18 52 Z "
        "M36 52 A12 5 0 1 0 60 52 A12 5 0 1 0 36 52 Z",
        rule="evenodd",
    )
    m["cfa"] = (
        E(56, 46, 22, 13)
        + P("M34 54 C42 70 66 72 80 56 C66 64 46 64 34 54 Z")
        + S("M36 42 C24 36 18 30 16 22", 3)
        + S("M34 48 C22 48 16 50 14 42", 3)
        + S("M36 54 C24 60 18 66 20 74", 3)
    )
    m["gbmo"] = (
        S("M22 22 H74 V74 H22 Z", 3.5, "square")
        + C(36, 36, 5)
        + C(58, 32, 3.5)
        + C(68, 48, 6)
        + C(40, 54, 4)
        + C(56, 62, 5)
        + C(34, 66, 3)
    )
    blast = ""
    for ang in (90, 68, 112, 48, 132, 78, 102):
        a = math.radians(ang)
        length = 34 if ang in (90, 68, 112) else 26
        tip = (48 + length * math.cos(a), 68 - length * math.sin(a))
        px, py = -math.sin(a) * 4.2, -math.cos(a) * 4.2
        blast += G([(48 + px, 68 + py), tip, (48 - px, 68 - py)])
    m["mba"] = blast + S("M16 70 H80", 4.5)
    m["ppa"] = (
        S("M30 76 V44", 3.5)
        + E(20, 56, 11, 5, -50)
        + E(40, 52, 11, 5, 48)
        + C(30, 40, 4)
        + S("M30 68 C42 74 52 62 66 68", 3.2)
        + S("M66 68 V54", 3.2)
        + E(58, 58, 8, 3.6, -48)
        + E(74, 56, 8, 3.6, 48)
    )
    m["sso"] = (
        C(48, 48, 4)
        + S("M48 22 V74 M22 48 H74 M30 30 L66 66 M66 30 L30 66", 2.4)
        + S("M48 18 A30 30 0 1 0 48.1 18", 3)
        + S("M44 48 L36 40 M52 48 L60 40 M44 52 L38 60 M52 52 L58 60", 2.4)
    )
    m["smo"] = (
        R(16, 16, 64, 7, 3)
        + S("M62 23 C60 36 50 38 46 46", 5)
        + C(40, 50, 9)
        + C(32, 46, 4.2)
        + E(44, 66, 10, 12)
        + S("M36 76 C20 84 14 68 26 62 C18 70 28 80 38 74", 4)
        + S("M40 76 L34 86", 4)
        + S("M50 76 L58 84", 4)
    )
    m["amo"] = (
        chevron(30, 28)
        + chevron(44, 36)
        + chevron(58, 44)
        + chevron(44, 52)
        + chevron(30, 60)
    )
    m["aea"] = (
        C(68, 26, 8)
        + G([(40, 78), (18, 78), (40, 34), (62, 78)])
        + G([(40, 70), (24, 70), (40, 44), (56, 70)])
        + R(36, 74, 8, 8, 1)
    )
    m["bmo"] = bird(14, 50, 1.05) + bird(82, 50, 1.05, flip=True)
    m["foa"] = pine(28, 74, 40) + pine(50, 76, 52) + pine(70, 74, 36)
    m["gwo"] = G(
        [
            (22, 74),
            (18, 58),
            (28, 46),
            (26, 34),
            (34, 30),
            (30, 16),
            (40, 28),
            (46, 14),
            (52, 30),
            (70, 16),
            (58, 36),
            (66, 48),
            (58, 62),
            (48, 72),
            (34, 76),
        ]
    )
    mane = []
    for i in range(18):
        outer = math.radians(i * 20 - 90)
        inner = math.radians(i * 20 + 10 - 90)
        if i == 0:
            mane.append(f"M{48 + 24 * math.cos(outer):.1f} {50 + 24 * math.sin(outer):.1f}")
        else:
            mane.append(f"L{48 + 24 * math.cos(outer):.1f} {50 + 24 * math.sin(outer):.1f}")
        mane.append(f"L{48 + 16 * math.cos(inner):.1f} {50 + 16 * math.sin(inner):.1f}")
    mane.append("Z")
    m["loa"] = (
        P(" ".join(mane))
        + C(30, 28, 6)
        + C(66, 28, 6)
        + E(48, 66, 9, 7)
    )
    m["oio"] = (
        S("M14 32 H32", 3.4)
        + S("M64 26 H84", 3.4)
        + S("M14 48 H32", 3.4)
        + S("M64 48 H84", 3.4)
        + S("M14 64 H32", 3.4)
        + S("M64 70 H84", 3.4)
        + '  <ellipse fill="none" stroke="#1a1814" stroke-width="5" cx="48" cy="48" rx="14" ry="26"/>\n'
    )
    m["rroa"] = (
        S("M16 68 H80", 4)
        + P(
            "M34 66 C32 54 38 46 46 44 L58 28 L52 44 C64 42 74 50 72 60 "
            "C78 58 82 64 74 66 Z"
        )
        + G([(30, 58), (16, 50), (18, 64)])
    )
    m["wwo"] = (
        S("M14 36 C26 24 34 24 48 36 C62 48 70 48 82 36", 5)
        + S("M14 52 C26 40 34 40 48 52 C62 64 70 64 82 52", 5)
        + S("M14 68 C26 56 34 56 48 68 C62 80 70 80 82 68", 5)
    )
    m["cab"] = ""
    for hx, hy, s in ((30, 58, 0.8), (48, 52, 1), (66, 60, 0.75)):
        m["cab"] += (
            E(hx, hy, 12 * s, 7 * s)
            + C(hx + 12 * s, hy - 4 * s, 5 * s)
            + R(hx - 8 * s, hy + 4 * s, 3 * s, 10 * s, 1)
            + R(hx - 2 * s, hy + 4 * s, 3 * s, 10 * s, 1)
            + R(hx + 5 * s, hy + 4 * s, 3 * s, 10 * s, 1)
            + R(hx + 10 * s, hy + 3 * s, 3 * s, 9 * s, 1)
        )
    m["acroa"] = (
        P("M38 20 H58 V34 L74 72 A8 8 0 0 1 66 82 H30 A8 8 0 0 1 22 72 L38 34 Z")
        + "".join(
            f'  <circle fill="{PLATE}" cx="{cx}" cy="{cy}" r="{r}"/>\n'
            for cx, cy, r in ((44, 60, 3.2), (56, 68, 2.6), (46, 72, 2.1))
        )
    )
    m["bull"] = (
        E(48, 56, 16, 14)
        + S("M34 50 C26 36 16 28 14 16", 5.5)
        + S("M62 50 C70 36 80 28 82 16", 5.5)
        + E(30, 52, 6, 4, -20)
        + E(66, 52, 6, 4, 20)
        + E(48, 66, 8, 6)
    )
    m["eho"] = (
        E(58, 56, 20, 14)
        + C(34, 50, 12)
        + E(44, 46, 10, 13)
        + S("M26 56 C16 66 18 78 30 76", 5)
        + S("M24 58 C18 64 20 70 26 68", 2.6)
        + R(42, 66, 6, 12, 2)
        + R(52, 68, 6, 12, 2)
        + R(62, 68, 6, 12, 2)
        + R(70, 64, 6, 12, 2)
        + S("M76 48 C84 44 84 36 78 38", 3)
    )
    return m


def main() -> None:
    drawn = marks()
    missing = [slug for slug, *_ in CATALOG if slug not in drawn]
    extra = [slug for slug in drawn if slug not in {row[0] for row in CATALOG}]
    if missing or extra:
        raise SystemExit(f"missing={missing} extra={extra}")
    for slug, _short, name, _import_name, _url in CATALOG:
        (OUT / f"{slug}.svg").write_text(svg(name, drawn[slug]), encoding="utf-8")
    print(f"wrote {len(CATALOG)} marks to {OUT}")


if __name__ == "__main__":
    main()

"""Configuration for the inverse design generator submodule."""

# ---------------------------------------------------------------------------
# Candidate element sets (matched to GAN paper element lists)
# ---------------------------------------------------------------------------
CANDIDATE_ELEMENT_SETS = {
    "ZrCuAlNi":        ["Zr", "Cu", "Al", "Ni"],
    "ZrCuAlNiTi":      ["Zr", "Cu", "Al", "Ni", "Ti"],
    "ZrCuAlNiTiNb":    ["Zr", "Cu", "Al", "Ni", "Ti", "Nb"],
    "ZrCuAlNiTiNbCoAg": ["Zr", "Cu", "Al", "Ni", "Ti", "Nb", "Co", "Ag"],
    "ZrTiCuNiAlNbAgHf": ["Zr", "Ti", "Cu", "Ni", "Al", "Nb", "Ag", "Hf"],
    "FeBCSiPYCoNb":     ["Fe", "B", "C", "Si", "P", "Y", "Co", "Nb"],
    "LaAlCuNiAgCoCe":   ["La", "Al", "Cu", "Ni", "Ag", "Co", "Ce"],
    "TiCuNiZrBeSn":     ["Ti", "Cu", "Ni", "Zr", "Be", "Sn"],
    "CuZrAlAg":         ["Cu", "Zr", "Al", "Ag"],
    "CaMgCu":           ["Ca", "Mg", "Cu"],
}

DEFAULT_ELEMENTS = CANDIDATE_ELEMENT_SETS["ZrTiCuNiAlNbAgHf"]

# ---------------------------------------------------------------------------
# Large element pool for auto-select mode (30 common BMG-forming elements)
# ---------------------------------------------------------------------------
LARGE_ELEMENT_POOL = [
    # Late transition metals (BMG workhorses)
    "Zr", "Ti", "Cu", "Ni", "Al",      # core BMG quartet + Al
    "Fe", "Co",                         # ferrous BMG
    "Nb", "Mo", "Ta", "W", "V",         # refractory / high-Tm
    "Hf",                               # Zr analog
    # Noble / near-noble
    "Ag", "Pd", "Au", "Pt",
    # Metalloids (glass-formers)
    "B", "C", "Si", "P", "Sn", "Be",
    # Rare earths
    "La", "Ce", "Pr", "Nd", "Sm", "Gd",
    "Y", "Sc",
    # Alkaline earths
    "Ca", "Mg", "Sr",
    # Others
    "Zn", "Ga", "Ge", "Mn", "Cr",
]

# Run mode
DEFAULT_MODE = "fixed"  # "fixed" = user-specified element set; "auto" = large pool + auto-drop

# ---------------------------------------------------------------------------
# Feature scheme
# ---------------------------------------------------------------------------
# Scheme A ("composition"): each element → 1 value (its atomic fraction)
# Scheme B ("elemental"):   each element → 5 values (fraction, group, period, AS, EN)
FEATURE_SCHEMES = {
    "composition": 1,
    "elemental":   5,
}
DEFAULT_SCHEME = "composition"
FEATS_PER_ELEM = FEATURE_SCHEMES[DEFAULT_SCHEME]  # 1

# ---------------------------------------------------------------------------
# Generator architecture
# ---------------------------------------------------------------------------
LATENT_DIM = 100  # noise vector dimension
GEN_HIDDEN_DIMS = [512, 256, 128]  # hidden layer widths
GEN_DROPOUT = 0.0
GEN_LEAKY_SLOPE = 0.2

# ---------------------------------------------------------------------------
# Training / generation
# ---------------------------------------------------------------------------
DEFAULT_N_GENERATIONS = 500
DEFAULT_BATCH_SIZE = 100
DEFAULT_LR = 1e-4
DEFAULT_BETA = (0.5, 0.999)
DEFAULT_KEEP_TOP_K = 10        # number of top candidates to track per generation
DEFAULT_THRESHOLD = 0.01        # fraction below which an element is dropped (1%)
DEFAULT_SAVE_TOP_N = 50        # save top N candidates across all generations

# ---------------------------------------------------------------------------
# Fitness / target
# ---------------------------------------------------------------------------
# Predefined target configurations
TARGETS = {
    "BMG":              {"mode": "maximize",  "key": "BMG_prob"},
    "Tg_high":          {"mode": "maximize",  "key": "Tg"},
    "Tg_low":           {"mode": "minimize",  "key": "Tg"},
    "Tx_high":          {"mode": "maximize",  "key": "Tx"},
    "Tl_high":          {"mode": "maximize",  "key": "Tl"},
    "Tl_low":           {"mode": "minimize",  "key": "Tl"},
    "E_high":           {"mode": "maximize",  "key": "E"},
    "H_high":           {"mode": "maximize",  "key": "H"},
    "supercooled_range": {"mode": "maximize",  "key": "Tx_minus_Tg"},
}

# ---------------------------------------------------------------------------
# Multi-objective targets (加权求和标量化，见 objectives.py)
# ---------------------------------------------------------------------------
# 自定义多目标时，可直接传 dict，格式：
#   {"objectives": [{"key": ..., "mode": ..., "weight": ...}, ...],
#    "normalize": "minmax"}
MULTI_TARGETS = {
    # 同时提高 Tg 与 Tx（玻璃转变 + 晶化温度都高）
    "TgTx_high": {
        "objectives": [
            {"key": "Tg", "mode": "maximize", "weight": 1.0},
            {"key": "Tx", "mode": "maximize", "weight": 1.0},
        ],
        "normalize": "minmax",
    },
    # 玻璃形成能力：高 Tg + 大过冷液相区 ΔT
    "high_GFA": {
        "objectives": [
            {"key": "Tg", "mode": "maximize", "weight": 1.0},
            {"key": "Tx_minus_Tg", "mode": "maximize", "weight": 1.0},
        ],
        "normalize": "minmax",
    },
    # 热稳定性：大过冷液相区 + 低液相线温度
    "thermal_stable": {
        "objectives": [
            {"key": "Tx_minus_Tg", "mode": "maximize", "weight": 1.0},
            {"key": "Tl", "mode": "minimize", "weight": 0.5},
        ],
        "normalize": "minmax",
    },
    # 高强度玻璃：高 Tg + 高杨氏模量 + 高硬度
    "strong_glass": {
        "objectives": [
            {"key": "Tg", "mode": "maximize", "weight": 1.0},
            {"key": "E", "mode": "maximize", "weight": 1.0},
            {"key": "H", "mode": "maximize", "weight": 1.0},
        ],
        "normalize": "minmax",
    },
}

# 单目标 + 多目标统一字典
ALL_TARGETS = {**TARGETS, **MULTI_TARGETS}

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
OUTPUT_DIR = None  # None = auto: mgflow/generator/output/

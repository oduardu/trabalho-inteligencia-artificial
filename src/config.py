class Config:
    # Configuração Base
    # Reward configs
    # STEP_REWARD = -1
    # WALL_REWARD = -5
    # HOLE_REWARD = -100
    # GOAL_REWARD = 100

    # Q-Learning configs
    # ALPHA = 0.5
    # GAMMA = 0.9
    # EPSILON = 1.0
    # SIMMULATION_NUMBER = 250000 # FIXO
    # ALPHA_DECAY = 0.999
    # EPSILON_DECAY = 0.999
    # DECAY_STEP = 500

    # Reward configs
    STEP_REWARD = -1
    WALL_REWARD = -5
    HOLE_REWARD = -100
    GOAL_REWARD = 500

    # A2C configs
    # GAMMA = 0.9
    # N_STEPS = 20
    # ENT_COEF = 0.01
    # LR = 0.0005

    SIMMULATION_NUMBER = 250000  # FIXO

    # Q-Learning configs
    ALPHA = 0.9
    GAMMA = 0.99
    EPSILON = 1.0
    ALPHA_DECAY = 0.99
    EPSILON_DECAY = 0.99
    DECAY_STEP = 100

    # Train or Test Model/Agent 
    TRAIN = True
    # TRAIN = False
    RENDERS = False
    # RENDERS = True
    TEST = True
    MODEL = "q-learning"  # "q-learning" or "a2c" or "both"


def STEP_REWARD() -> int:
    return Config.STEP_REWARD


def WALL_REWARD() -> int:
    return Config.WALL_REWARD


def HOLE_REWARD() -> int:
    return Config.HOLE_REWARD


def GOAL_REWARD() -> int:
    return Config.GOAL_REWARD


# A2C
def GAMMA() -> float:
    return Config.GAMMA


def N_STEPS() -> int:
    return Config.N_STEPS


def ENT_COEF() -> float:
    return Config.ENT_COEF


def LR() -> float:
    return Config.LR


# Q-Learning
def ALPHA() -> float:
    return Config.ALPHA


def GAMMA() -> float:
    return Config.GAMMA


def EPSILON() -> float:
    return Config.EPSILON


def SIMMULATION_NUMBER() -> int:
    return Config.SIMMULATION_NUMBER


def ALPHA_DECAY() -> float:
    return Config.ALPHA_DECAY


def EPSILON_DECAY() -> float:
    return Config.EPSILON_DECAY


def DECAY_STEP() -> int:
    return Config.DECAY_STEP


def TRAIN() -> bool:
    return Config.TRAIN


def RENDERS() -> bool:
    return Config.RENDERS

def TEST() -> bool:
    return Config.TEST

def MODEL() -> str:
    return Config.MODEL
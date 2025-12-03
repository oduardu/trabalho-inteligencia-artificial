class Config:
    # Reward configs
    STEP_REWARD = -2
    WALL_REWARD = -5
    HOLE_REWARD = -100
    GOAL_REWARD = 100

    # Q-Learning configs
    ALPHA = 0.9
    GAMMAQ = 0.99
    EPSILON = 1.0
    ALPHA_DECAY = 0.99
    EPSILON_DECAY = 0.99
    DECAY_STEP = 3000

    # A2C configs
    GAMMAA2C = 0.995
    N_STEPS = 10
    ENT_COEF = 0.05
    LR = 0.0007

    SIMMULATION_NUMBER = 250000  # FIXO

    # Train or Test Model/Agent
    TRAIN = True
    RENDERS = False
    TEST = False
    MODEL = "q-learning" # "a2c" ou "q-learning"

def STEP_REWARD() -> int:
    return Config.STEP_REWARD

def WALL_REWARD() -> int:
    return Config.WALL_REWARD

def HOLE_REWARD() -> int:
    return Config.HOLE_REWARD

def GOAL_REWARD() -> int:
    return Config.GOAL_REWARD

def GAMMAA2C() -> float:
    return Config.GAMMAA2C

def N_STEPS() -> int:
    return Config.N_STEPS

def ENT_COEF() -> float:
    return Config.ENT_COEF

def LR() -> float:
    return Config.LR

def ALPHA() -> float:
    return Config.ALPHA

def GAMMAQ() -> float:
    return Config.GAMMAQ

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

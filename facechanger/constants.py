# Indices for face landmarks
LOWER_HEAD = list(range(0,33))
LEFT_BROW = list(range(33,42))
RIGHT_BROW = list(range(42,51))
NOSE = list(range(51,60))
LEFT_EYE = list(range(60,68))
LEFT_EYE += [96]
RIGHT_EYE = list(range(68,76))
RIGHT_EYE += [97]
MOUTH =  list(range(76,96))


INDICES = {
    "lower_head": LOWER_HEAD,   
    "left_brow": LEFT_BROW,
    "right_brow": RIGHT_BROW,
    "nose": NOSE,
    "left_eye": LEFT_EYE,
    "right_eye": RIGHT_EYE,
    "mouth": MOUTH
}

DEFAULT = {
    "lower_head": {
        "trans": [0, 0],
        "zoom": [1, 1]
    },   
    "left_brow": {
        "trans": [0, 0],
        "zoom": [1, 1]
    },
    "right_brow": {
        "trans": [0, 0],
        "zoom": [1, 1]
    },
    "nose": {
        "trans": [0, 0],
        "zoom": [1, 1]
    },
    "left_eye": {
        "trans": [0, 0],
        "zoom": [1, 1]
    },
    "right_eye": {
        "trans": [0, 0],
        "zoom": [1, 1]
    },
    "mouth": {
        "trans": [0, 0],
        "zoom": [1, 1]
    }
}
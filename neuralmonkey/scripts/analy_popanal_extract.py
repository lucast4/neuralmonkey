""" Extract popanal across all sessions for this day.
Make sure you have extract all Snippets
"""

from neuralmonkey.classes.snippets import load_and_concat_mult_snippets
from neuralmonkey.classes.session import load_mult_session_helper
from neuralmonkey.analyses.state_space_good import snippets_extract_popanals_split_bregion_twind
import sys

assert False, "THIS WORKS, but not doing it since it takes too much space. Instead go straight from SP --> PA"

# DATE = 220606
# animal = "Pancho"
# which_level = "trial"
LIST_WHICH_LEVEL = ["trial", "stroke", "stroke_off"]

if __name__ == "__main__":

    assert False, "files too large, and this is quick enough to do from SP"
    animal = sys.argv[1]
    DATE = int(sys.argv[2])

    if len(sys.argv)>3:
        which_level = sys.argv[3]
        LIST_WHICH_LEVEL = [which_level]

    MS = load_mult_session_helper(DATE, animal)
    for which_level in LIST_WHICH_LEVEL:
        SP, SAVEDIR_ALL = load_and_concat_mult_snippets(MS, which_level = which_level)

        # Snippets --> Multiple PA
        list_version_distance = ["pearson"]
        FEATURES_EXTRACT_TO_POPANAL = []
        list_time_windows = [
            (-0.6, -0.4),
            (-0.5, -0.3),
            (-0.4, -0.2),
            (-0.3, -0.1),
            (-0.2, 0.),
            (-0.1, 0.1),
            (0., 0.2),
            (0.1, 0.3),
            (0.2, 0.4),
            (0.3, 0.5),
            (0.4, 0.6),
            ]

        # Extract and save PopAnals.
        snippets_extract_popanals_split_bregion_twind(SP, list_time_windows)

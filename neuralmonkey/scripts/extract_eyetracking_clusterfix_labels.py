""" One goal - extraction and saving of clusterfix labels for a given (animal, date).

The resulting data can then be loaded using  methods in Session().
See notebook for use: notebooks/230430_eyetracking_overview_GOOD.ipynb

7/15/25 - LT
"""

from neuralmonkey.classes.session import load_mult_session_helper
import sys

if __name__=="__main__":

    animal = sys.argv[1]
    date = int(sys.argv[2])

    MS = load_mult_session_helper(date, animal)

    # Get eye trackign, if not already got
    for sn in MS.SessionsList:
        if not sn.clusterfix_check_if_preprocessing_complete():
            sn.extract_and_save_clusterfix_results()



            
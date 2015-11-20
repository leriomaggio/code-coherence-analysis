"""Processing data of terms distributions stored in textual files
per single project, and coherence evaluation.
"""
# Author: Valerio Maggio <valeriomaggio@gmail.com>
# License: BSD 3 clause

import numpy as np
import os
from collections import defaultdict

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
TARGET_FOLDER_NAME = 'distributions_per_rate_tfidf'
TARGET_FOLDER_PATH = os.path.join(BASE_DIR, 'data', TARGET_FOLDER_NAME)

# Easy to use Constants access keys to the first layer of `evaluations_info` dictionary
COHERENCE_KEY = 'CO'
NON_COHERENCE_KEY = 'NC'
DONT_KNOW_KEY = 'DK'


def get_distributions_data(target_folder=TARGET_FOLDER_PATH):
    """
        Process all the (.txt) files contained in `target_folder` and gets
        all the distributions data corresponding to analysed projects.

        This function assumes each files contained in the `target_folder`
        is named according to the following pattern:
            `EVAL_PROJECTNAME_VERSION_COUNT`.txt such as

            * `EVAL` corresponds to a Coherence Evaluation Abbreviation
                (i.e., `CO` for *Coherent*, `NC` for *Non Coherent*, `DK` for *Don't Know*)
            * `PROJECTNAME: the name of the target project
            * `VER`: the release number of the target project
            * `COUNT`: the total number of methods' data contained in the file
                (i.e., the total number of rows)

        Gathered information are stored (and returned) in a multi-layer dictionary
        that is structured as follows:

            EVAL_1 : { '<project_name_1>' : dict --> { 'methods_count': int,
                                                     'data': numpy.array  # loaded from file
                                                   },
                      '<project_name_2>' : dict --> { 'methods_count': int,
                                                      'data': numpy.array  # loaded from file
                                                   },
                      ...
                    },

            EVAL_2 : ...

        Thus to extract the **data** for a specific Coherence Evaluation (e.g., 'CO',
        and a specific project (e.g., 'Jfreechart 0.6.0'), the accessor
        path to this information is:

            `evaluations_info[COHERENCE_KEY]['Jfreechart-0.6.0']['data']`

        Parameters
        ----------
        target_folder : string, optional
            Path to the target folder containing .txt files to load

        Returns
        -------
        evaluations_info : dictionary
            The (multi-layered) dictionary embedding (per-evaluation, per-project)
            information.
    """

    evaluations_info = defaultdict(dict)
    for root, _, filenames in os.walk(target_folder):
        for filename in filenames:
            if filename.endswith('.txt'):
                fname, ext = os.path.splitext(filename)
                evaluation, project_name, version, methods_count = fname.split('_')
                project_key = '-'.join([project_name, version[1:-1]])
                project_info = dict()
                project_info['methods_count'] = int(methods_count)
                project_info['data'] = np.loadtxt(os.path.join(root, filename))
                evaluations_info[evaluation][project_key] = project_info

    return evaluations_info

if __name__ == '__main__':
    # Test Runner code (sample usage)
    evaluations_info = get_distributions_data()
    print(evaluations_info)
    # Get the distributions of jfreechart-0.6.0
    print(evaluations_info[COHERENCE_KEY]['Jfreechart-0.6.0'])




    

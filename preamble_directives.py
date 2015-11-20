"""Some imports and path settings to make notebook code
running smoothly.
"""
# Author: Valerio Maggio <valeriomaggio@gmail.com>
# Copyright (c) 2015 Valerio Maggio <valeriomaggio@gmail.com>
# License: BSD 3 clause

import sys, os
# Extending PYTHONPATH to allow relative import!
sys.path.append(os.path.join(os.path.abspath(os.path.curdir), '..'))
sys.warnoptions.append('ignore::DeprecationWarning')  # Avoid DeprecationWarnings
sys.warnoptions.append('ignore::FutureWarning')  # Avoid DeprecationWarnings

# Import Django Settings
from django.conf import settings
# Import Comments_Classification (Django) Project Settings
from code_comments_coherence import settings as coherence_settings

settings.configure(**coherence_settings.__dict__)

import numpy as np
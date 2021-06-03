import logging
import os
import re
from datetime import datetime, timedelta
from inspect import CO_GENERATOR
from random import shuffle
from threading import Thread

from numpy.core.numeric import NaN

from channels.alpaca import Alpaca, TimeFrame
from channels.binanceus import BinanceUS
from channels.screener import Screener

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from time import sleep

import numpy as np
import pandas as pd
import tensorflow as tf
from finta import TA as ta

# just a test right now
from keras import optimizers
from keras.engine import training
from sklearn import preprocessing
from tqdm import tqdm, trange

tf.get_logger().setLevel("ERROR")
tf.autograph.set_verbosity(1)

import plotly.express as px
import plotly.graph_objects as go
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop
from keras.utils.vis_utils import plot_model
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split

pd.options.mode.chained_assignment = None  # default='warn'

os.environ["NUMEXPR_MAX_THREADS"] = "8"

logging.basicConfig(
    # filename,
    format="[%(name)s] %(levelname)s %(asctime)s %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

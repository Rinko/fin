# -*- coding: utf-8 -*-
import json
import pandas as pd
import numpy as np
from io import StringIO
import re
from datetime import datetime
import math
import os
import screen
import backtest

pd.set_option('display.max_columns', None)

stock_symbols = screen.basic_screen()

print(stock_symbols)
backtest.run_backtest(stock_symbols)
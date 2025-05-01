import logging
from pathlib import Path
import time
from datetime import datetime
import sys

logger = logging.getLogger(__name__)

fname = Path.cwd().parents[1].joinpath('datalogs',"TestLogging_"+datetime.now().strftime("_%d_%m_%Y_%H_%M_%S"))

fh = logging.FileHandler(fname) #file handler
fh.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout) #stream handler
ch.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

fh.setFormatter(formatter)
ch.setFormatter(formatter)

logger.setLevel(logging.DEBUG)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)

while True:
    logger.debug('Check Debug Statement')
    logger.info('Check Info statement')
    time.sleep(1)
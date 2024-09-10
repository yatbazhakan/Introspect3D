import logging 
import os
from logging.handlers import RotatingFileHandler

def setup_logging(module_name, log_dir = 'logs', level = logging.INFO, console = True, file = True):
    # Create logger
    logger = logging.getLogger(module_name)
    logger.setLevel(level)
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if not logger.handlers:
        if console:
            # Create console handler and set level to debug
            ch = logging.StreamHandler()
            ch.setLevel(level)
            # Add formatter to handlers
            ch.setFormatter(formatter)
            # Add handlers to logger
            logger.addHandler(ch)
        
        if file:
            # Create file handler and set level to debug
            log_fdir = os.path.join(log_dir, module_name+'.log')
            fh = RotatingFileHandler(log_fdir, maxBytes=2000, backupCount=5)
            fh.setLevel(level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
    
    return logger
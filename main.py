from utils.config import Config
import argparse
import traceback 
import os
from operators.activation_extractor import ActivationExractionOperator
from operators.introspector import IntrospectionOperator
from archive.e2e_extract_introspector import E2EIntrospector 
from definitions import ROOT_DIR,CONFIG_DIR
def get_operator(type:int = 1):
    if type == 1:
        return ActivationExractionOperator
    elif type == 2:
        return IntrospectionOperator
    elif type ==3:
        return E2EIntrospector
    else:
        raise NotImplementedError("Operator not implemented")
def parse_args():
    parser = argparse.ArgumentParser(description="Your research experiment")
    parser.add_argument("-c", "--config", type=str, required=True,
                        help="Path to the configuration YAML file.")
    parser.add_argument("-o","--operator",type=str,required=True,help="Operator to execute: (0) detection, (1) extraction, (2) introspection")
    return parser.parse_args()

if __name__== "__main__":
    args = parse_args()
    print(args.config)
    config = Config(os.path.join(CONFIG_DIR,args.config))
    
    operator = get_operator(int(args.operator))(config)
    operator.execute(verbose=config.verbose)

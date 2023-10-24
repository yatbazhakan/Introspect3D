from utils.config import Config
import argparse
import os
from operators.activation_extractor import ActivationExractionOperator

def get_operator(type:int = 1):
    if type == 1:
        return ActivationExractionOperator
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
    #Get current path#
    current_path = os.path.dirname(os.path.abspath(__file__))

    config = Config(os.path.join(current_path,'configs',args.config))
    
    operator = get_operator(int(args.operator))(config)
    operator.execute(verbose=True)

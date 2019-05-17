import sys
sys.path.append('./')
from optparse import OptionParser
from solver import Solver
from solver_multigpu import SolverMultigpu
from utils import process_config

parser = OptionParser()
parser.add_option("-c", "--conf", dest="configure",  
                  help="configure filename")
(options, args) = parser.parse_args() 
if options.configure:
  conf_file = str(options.configure)
else:
  print('please specify --conf configure filename')
  exit(0)

common_params, dataset_params, net_params, solver_params = process_config(conf_file)
print('common_params: ', common_params)
print('dataset_params: ', dataset_params)
print('net_params: ', net_params)
print('solver_params: ', solver_params)
if len(str(common_params['gpus']).split(','))==1:
  print('single gpu')
  solver = Solver(True, common_params, solver_params, net_params, dataset_params)
else:
  print('multiple gpu')
  solver = SolverMultigpu(True, common_params, solver_params, net_params, dataset_params)
solver.train_model()

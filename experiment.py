import argparse

import sys
sys.path.append('./')

import cql.cltsorl
import cql.cuorl
import cql.harness
import iql.cltsorl
import iql.cuorl
import iql.harness

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='hopper')
parser.add_argument('--sampler', type=str, default="uniform")
parser.add_argument('--dataset_size', type=int, default=int(1e6))
parser.add_argument('--dataset_types', type=str, nargs="+", required=True)
parser.add_argument('--dataset_ratios', type=float, nargs="+", required=True)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--gpu', type=int)
parser.add_argument('--project', type=str)
parser.add_argument('--baseline', type=str, default='CQL')
parser.add_argument('--method', type=str, default='CLTV')
args = parser.parse_args()

if args.baseline == 'CQL':
    if args.method == 'Vanilla':
        cql.cltsorl.baseline(args)
    elif args.method == 'TS':
        cql.cltsorl.without_CL(args)
    elif args.method == 'CLTV':
        cql.cltsorl.traj_valuation(args)
    elif args.method == 'CUORL':
        cql.cuorl.cuorl(args)
    elif args.method == 'Harness':
        cql.harness.harness(args)
    elif args.method == 'reward1':
        cql.cltsorl.reward1(args)
    elif args.method == 'reward2':
        cql.cltsorl.reward2(args)
    elif args.method == 'pairimpact':
        cql.cltsorl.pair_impact(args)
elif args.baseline == 'IQL':
    if args.method == 'Vanilla':
        iql.cltsorl.baseline(args)
    elif args.method == 'TS':
        iql.cltsorl.without_CL(args)
    elif args.method == 'CLTV':
        iql.cltsorl.traj_valuation(args)
    elif args.method == 'CUORL':
        iql.cuorl.cuorl(args)
    elif args.method == 'Harness':
        iql.harness.harness(args)
    elif args.method == 'reward1':
        iql.cltsorl.reward1(args)
    elif args.method == 'reward2':
        iql.cltsorl.reward2(args)
    elif args.method == 'pairimpact':
        iql.cltsorl.pair_impact(args)

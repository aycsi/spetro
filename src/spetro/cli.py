import argparse
import sys
import json
import csv
from pathlib import Path

from . import RoughVolatilityEngine, RoughBergomi, RoughHeston, Pricer, Calibrator


def price_cmd(args):
    engine = RoughVolatilityEngine(backend=args.backend)
    
    if args.model == "rb":
        model = RoughBergomi(H=args.h, eta=args.eta, rho=args.rho, xi=args.xi, r=args.r)
    elif args.model == "rh":
        model = RoughHeston(H=args.h, nu=args.nu, theta=args.theta, rho=args.rho, V0=args.v0, r=args.r)
    
    pricer = Pricer(engine)
    
    if args.type == "call":
        result = pricer.price_european(model=model, option_type="call", K=args.k, T=args.t, S0=args.s0, n_paths=args.paths)
    elif args.type == "put":
        result = pricer.price_european(model=model, option_type="put", K=args.k, T=args.t, S0=args.s0, n_paths=args.paths)
    elif args.type == "asian":
        result = pricer.price_asian(model=model, option_type="call", K=args.k, T=args.t, S0=args.s0, n_paths=args.paths)
    elif args.type == "barrier":
        result = pricer.price_barrier(model=model, K=args.k, barrier=args.barrier, barrier_type=args.btype, T=args.t, S0=args.s0, n_paths=args.paths)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
    else:
        print(f"{result['price']:.6f}")


def greeks_cmd(args):
    engine = RoughVolatilityEngine(backend=args.backend)
    
    if args.model == "rb":
        model = RoughBergomi(H=args.h, eta=args.eta, rho=args.rho, xi=args.xi, r=args.r)
    elif args.model == "rh":
        model = RoughHeston(H=args.h, nu=args.nu, theta=args.theta, rho=args.rho, V0=args.v0, r=args.r)
    
    pricer = Pricer(engine)
    result = pricer.greeks(model=model, option_type=args.type, K=args.k, T=args.t, S0=args.s0, n_paths=args.paths)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
    else:
        print(f"{result['price']:.6f},{result['delta']:.6f},{result['gamma']:.6f}")


def calibrate_cmd(args):
    engine = RoughVolatilityEngine(backend=args.backend)
    calibrator = Calibrator(engine)
    
    market_data = {}
    with open(args.input, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            k = float(row['strike'])
            t = float(row['maturity'])
            price = float(row['price'])
            market_data[(k, t)] = price
    
    if args.model == "rb":
        model_class = RoughBergomi
        initial_params = {"H": args.h, "eta": args.eta, "rho": args.rho, "xi": args.xi}
        bounds = {"H": (0.01, 0.49), "eta": (0.1, 5.0), "rho": (-0.99, 0.99), "xi": (0.01, 1.0)}
    elif args.model == "rh":
        model_class = RoughHeston
        initial_params = {"H": args.h, "nu": args.nu, "theta": args.theta, "rho": args.rho, "V0": args.v0}
        bounds = {"H": (0.01, 0.49), "nu": (0.1, 5.0), "theta": (0.01, 1.0), "rho": (-0.99, 0.99), "V0": (0.01, 1.0)}
    
    result = calibrator.calibrate_to_surface(
        model_class=model_class,
        market_prices=market_data,
        S0=args.s0,
        initial_params=initial_params,
        bounds=bounds,
        max_iter=args.iter
    )
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result['parameters'], f, indent=2)
    else:
        for param, value in result['parameters'].items():
            print(f"{param}={value:.6f}")


def batch_cmd(args):
    engine = RoughVolatilityEngine(backend=args.backend)
    
    if args.model == "rb":
        model = RoughBergomi(H=args.h, eta=args.eta, rho=args.rho, xi=args.xi, r=args.r)
    elif args.model == "rh":
        model = RoughHeston(H=args.h, nu=args.nu, theta=args.theta, rho=args.rho, V0=args.v0, r=args.r)
    
    pricer = Pricer(engine)
    
    results = []
    with open(args.input, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            k = float(row['strike'])
            t = float(row['maturity'])
            s0 = float(row.get('spot', args.s0))
            opt_type = row.get('type', 'call')
            
            if opt_type == "call":
                result = pricer.price_european(model=model, option_type="call", K=k, T=t, S0=s0, n_paths=args.paths)
            elif opt_type == "put":
                result = pricer.price_european(model=model, option_type="put", K=k, T=t, S0=s0, n_paths=args.paths)
            
            results.append({
                'strike': k,
                'maturity': t,
                'spot': s0,
                'type': opt_type,
                'price': result['price'],
                'std_error': result['std_error']
            })
    
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['strike', 'maturity', 'spot', 'type', 'price', 'std_error'])
        writer.writeheader()
        writer.writerows(results)


def main():
    parser = argparse.ArgumentParser(prog='spetro', description='spetro cli tool')
    subparsers = parser.add_subparsers(dest='command', help='available commands')
    
    price_parser = subparsers.add_parser('price', help='price options')
    price_parser.add_argument('--model', choices=['rb', 'rh'], default='rb', help='model type')
    price_parser.add_argument('--type', choices=['call', 'put', 'asian', 'barrier'], default='call', help='option type')
    price_parser.add_argument('--k', type=float, required=True, help='strike')
    price_parser.add_argument('--t', type=float, required=True, help='maturity')
    price_parser.add_argument('--s0', type=float, default=100.0, help='spot price')
    price_parser.add_argument('--h', type=float, default=0.07, help='hurst parameter')
    price_parser.add_argument('--eta', type=float, default=1.9, help='vol of vol')
    price_parser.add_argument('--rho', type=float, default=-0.9, help='correlation')
    price_parser.add_argument('--xi', type=float, default=0.055, help='initial variance')
    price_parser.add_argument('--nu', type=float, default=0.3, help='vol of vol (heston)')
    price_parser.add_argument('--theta', type=float, default=0.02, help='mean reversion')
    price_parser.add_argument('--v0', type=float, default=0.02, help='initial variance (heston)')
    price_parser.add_argument('--r', type=float, default=0.0, help='risk free rate')
    price_parser.add_argument('--barrier', type=float, help='barrier level')
    price_parser.add_argument('--btype', choices=['up_and_out', 'down_and_out'], default='up_and_out', help='barrier type')
    price_parser.add_argument('--paths', type=int, default=100000, help='monte carlo paths')
    price_parser.add_argument('--backend', choices=['jax', 'torch'], default='jax', help='backend')
    price_parser.add_argument('--output', help='output file')
    price_parser.set_defaults(func=price_cmd)
    
    greeks_parser = subparsers.add_parser('greeks', help='calculate greeks')
    greeks_parser.add_argument('--model', choices=['rb', 'rh'], default='rb', help='model type')
    greeks_parser.add_argument('--type', choices=['call', 'put'], default='call', help='option type')
    greeks_parser.add_argument('--k', type=float, required=True, help='strike')
    greeks_parser.add_argument('--t', type=float, required=True, help='maturity')
    greeks_parser.add_argument('--s0', type=float, default=100.0, help='spot price')
    greeks_parser.add_argument('--h', type=float, default=0.07, help='hurst parameter')
    greeks_parser.add_argument('--eta', type=float, default=1.9, help='vol of vol')
    greeks_parser.add_argument('--rho', type=float, default=-0.9, help='correlation')
    greeks_parser.add_argument('--xi', type=float, default=0.055, help='initial variance')
    greeks_parser.add_argument('--nu', type=float, default=0.3, help='vol of vol (heston)')
    greeks_parser.add_argument('--theta', type=float, default=0.02, help='mean reversion')
    greeks_parser.add_argument('--v0', type=float, default=0.02, help='initial variance (heston)')
    greeks_parser.add_argument('--r', type=float, default=0.0, help='risk free rate')
    greeks_parser.add_argument('--paths', type=int, default=500000, help='monte carlo paths')
    greeks_parser.add_argument('--backend', choices=['jax', 'torch'], default='jax', help='backend')
    greeks_parser.add_argument('--output', help='output file')
    greeks_parser.set_defaults(func=greeks_cmd)
    
    calibrate_parser = subparsers.add_parser('calibrate', help='calibrate model')
    calibrate_parser.add_argument('--input', required=True, help='market data csv')
    calibrate_parser.add_argument('--model', choices=['rb', 'rh'], default='rb', help='model type')
    calibrate_parser.add_argument('--s0', type=float, default=100.0, help='spot price')
    calibrate_parser.add_argument('--h', type=float, default=0.1, help='hurst parameter')
    calibrate_parser.add_argument('--eta', type=float, default=2.0, help='vol of vol')
    calibrate_parser.add_argument('--rho', type=float, default=-0.8, help='correlation')
    calibrate_parser.add_argument('--xi', type=float, default=0.04, help='initial variance')
    calibrate_parser.add_argument('--nu', type=float, default=0.3, help='vol of vol (heston)')
    calibrate_parser.add_argument('--theta', type=float, default=0.02, help='mean reversion')
    calibrate_parser.add_argument('--v0', type=float, default=0.02, help='initial variance (heston)')
    calibrate_parser.add_argument('--iter', type=int, default=1000, help='max iterations')
    calibrate_parser.add_argument('--backend', choices=['jax', 'torch'], default='jax', help='backend')
    calibrate_parser.add_argument('--output', help='output file')
    calibrate_parser.set_defaults(func=calibrate_cmd)
    
    batch_parser = subparsers.add_parser('batch', help='batch pricing')
    batch_parser.add_argument('--input', required=True, help='input csv')
    batch_parser.add_argument('--output', required=True, help='output csv')
    batch_parser.add_argument('--model', choices=['rb', 'rh'], default='rb', help='model type')
    batch_parser.add_argument('--s0', type=float, default=100.0, help='spot price')
    batch_parser.add_argument('--h', type=float, default=0.07, help='hurst parameter')
    batch_parser.add_argument('--eta', type=float, default=1.9, help='vol of vol')
    batch_parser.add_argument('--rho', type=float, default=-0.9, help='correlation')
    batch_parser.add_argument('--xi', type=float, default=0.055, help='initial variance')
    batch_parser.add_argument('--nu', type=float, default=0.3, help='vol of vol (heston)')
    batch_parser.add_argument('--theta', type=float, default=0.02, help='mean reversion')
    batch_parser.add_argument('--v0', type=float, default=0.02, help='initial variance (heston)')
    batch_parser.add_argument('--r', type=float, default=0.0, help='risk free rate')
    batch_parser.add_argument('--paths', type=int, default=100000, help='monte carlo paths')
    batch_parser.add_argument('--backend', choices=['jax', 'torch'], default='jax', help='backend')
    batch_parser.set_defaults(func=batch_cmd)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    args.func(args)


if __name__ == '__main__':
    main()

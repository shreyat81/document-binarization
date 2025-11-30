#!/usr/bin/env python3
"""
Quick WOA Optimization for Mac CPU
Very fast parameter tuning with minimal samples and iterations.
"""

import subprocess
import sys

# Quick optimization settings for Mac CPU
QUICK_SETTINGS = {
    'pop': 4,          # 4 whales instead of 6
    'iters': 5,        # 5 iterations instead of 8
    'samples': 10,     # 10 samples instead of 15
}

# Standard settings (faster than original but more thorough)
STANDARD_SETTINGS = {
    'pop': 6,
    'iters': 8,
    'samples': 15,
}

# Full settings (slow but comprehensive)
FULL_SETTINGS = {
    'pop': 12,
    'iters': 20,
    'samples': 50,
}

def run_optimization(mode='quick', output_file=None):
    """
    Run WOA optimization with different speed/quality tradeoffs.
    
    Args:
        mode: 'quick' (fast, ~5-10 min), 'standard' (~20 min), or 'full' (~2 hours)
        output_file: Where to save optimized parameters
    """
    if mode == 'quick':
        settings = QUICK_SETTINGS
        default_output = 'src/fuzzy/quick_params.json'
    elif mode == 'standard':
        settings = STANDARD_SETTINGS
        default_output = 'src/fuzzy/optimized_params.json'
    elif mode == 'full':
        settings = FULL_SETTINGS
        default_output = 'src/fuzzy/full_params.json'
    else:
        print(f"Error: Unknown mode '{mode}'. Use 'quick', 'standard', or 'full'.")
        return
    
    if output_file is None:
        output_file = default_output
    
    # Build command
    cmd = [
        sys.executable,
        'src/fuzzy/woa_optimize.py',
        '--imgs', 'split/val/images',
        '--probs', 'split/val/prob_maps',
        '--gts', 'split/val/gt',
        '--pop', str(settings['pop']),
        '--iters', str(settings['iters']),
        '--sample_limit', str(settings['samples']),
        '--out', output_file
    ]
    
    print("=" * 80)
    print(f"Running WOA Optimization - {mode.upper()} mode")
    print("=" * 80)
    print(f"Population: {settings['pop']}")
    print(f"Iterations: {settings['iters']}")
    print(f"Samples: {settings['samples']}")
    print(f"Total inferences: {settings['pop'] * settings['iters'] * settings['samples']}")
    print(f"Output: {output_file}")
    
    if mode == 'quick':
        print("\n⚡ Quick mode: ~5-10 minutes on Mac CPU")
    elif mode == 'standard':
        print("\n⚙️  Standard mode: ~20 minutes on Mac CPU")
    else:
        print("\n🔥 Full mode: ~2 hours on Mac CPU")
    
    print("=" * 80)
    print()
    
    # Run optimization
    subprocess.run(cmd)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Quick WOA optimization for Mac CPU'
    )
    parser.add_argument('--mode', type=str, default='quick',
                       choices=['quick', 'standard', 'full'],
                       help='Optimization mode: quick (~5-10 min), standard (~20 min), or full (~2 hrs)')
    parser.add_argument('--out', type=str, default=None,
                       help='Output file for optimized parameters')
    
    args = parser.parse_args()
    
    run_optimization(mode=args.mode, output_file=args.out)

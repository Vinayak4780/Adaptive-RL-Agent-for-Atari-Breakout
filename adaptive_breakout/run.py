import os
import argparse
from datetime import datetime

def run_training(args):
    """Run the training process"""
    # Create command
    cmd = f"python -m adaptive_breakout.train"
    
    # Add training arguments
    cmd += f" --num-episodes {args.num_episodes}"
    cmd += f" --learning-rate {args.learning_rate}"
    cmd += f" --gamma {args.gamma}"
    cmd += f" --buffer-size {args.buffer_size}"
    cmd += f" --batch-size {args.batch_size}"
    
    # Add optional flags
    if args.auto_curriculum:
        cmd += " --auto-curriculum"
    if args.detect_changes:
        cmd += " --detect-changes"
    if args.prioritized_replay:
        cmd += " --prioritized-replay"
    
    # Add output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    cmd += f" --output-dir {output_dir}"
    
    # Run the command
    print("Running training with command:")
    print(cmd)
    os.system(cmd)
    
    return output_dir

def run_analysis(args, output_dir):
    """Run the analysis process"""
    # Get latest model path
    model_path = os.path.join(output_dir, "agent_final.pth")
    if not os.path.exists(model_path):
        # Try to find any checkpoint
        checkpoints = [f for f in os.listdir(output_dir) if f.endswith(".pth")]
        if checkpoints:
            model_path = os.path.join(output_dir, sorted(checkpoints)[-1])
        else:
            print("No model checkpoints found in output directory")
            return
    
    # Find logs directory
    log_dir = os.path.join(output_dir, "logs")
    if not os.path.exists(log_dir):
        print("Logs directory not found")
        return
    
    # Create analysis directory
    analysis_dir = os.path.join(output_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Create command
    cmd = f"python -m adaptive_breakout.analysis.analyze_agent"
    cmd += f" --model-path {model_path}"
    cmd += f" --log-dir {log_dir}"
    cmd += f" --output-dir {analysis_dir}"
    cmd += f" --n-eval-episodes {args.eval_episodes}"
    
    # Run the command
    print("Running analysis with command:")
    print(cmd)
    os.system(cmd)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Run adaptive RL agent for Atari Breakout")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Training parser
    train_parser = subparsers.add_parser("train", help="Train the adaptive agent")
    train_parser.add_argument("--num-episodes", type=int, default=1000, 
                            help="Number of episodes to train for")
    train_parser.add_argument("--learning-rate", type=float, default=0.0001, 
                            help="Learning rate")
    train_parser.add_argument("--gamma", type=float, default=0.99, 
                            help="Discount factor")
    train_parser.add_argument("--buffer-size", type=int, default=100000, 
                            help="Size of replay buffer")
    train_parser.add_argument("--batch-size", type=int, default=32, 
                            help="Batch size for training")
    train_parser.add_argument("--auto-curriculum", action="store_true", 
                            help="Enable automatic curriculum progression")
    train_parser.add_argument("--detect-changes", action="store_true", 
                            help="Enable environment change detection")
    train_parser.add_argument("--prioritized-replay", action="store_true", 
                            help="Use prioritized experience replay")
    train_parser.add_argument("--output-dir", type=str, default="results", 
                            help="Output directory")
    
    # Analysis parser
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a trained agent")
    analyze_parser.add_argument("--model-path", type=str, required=True, 
                              help="Path to the trained model checkpoint")
    analyze_parser.add_argument("--log-dir", type=str, required=True, 
                              help="Path to the training logs directory")
    analyze_parser.add_argument("--output-dir", type=str, default=None, 
                              help="Output directory for analysis (default: model directory)")
    analyze_parser.add_argument("--eval-episodes", type=int, default=5, 
                              help="Number of evaluation episodes per level")
    
    # Run both
    both_parser = subparsers.add_parser("run-both", help="Train and then analyze the agent")
    both_parser.add_argument("--num-episodes", type=int, default=1000, 
                           help="Number of episodes to train for")
    both_parser.add_argument("--learning-rate", type=float, default=0.0001, 
                           help="Learning rate")
    both_parser.add_argument("--gamma", type=float, default=0.99, 
                           help="Discount factor")
    both_parser.add_argument("--buffer-size", type=int, default=100000, 
                           help="Size of replay buffer")
    both_parser.add_argument("--batch-size", type=int, default=32, 
                           help="Batch size for training")
    both_parser.add_argument("--auto-curriculum", action="store_true", 
                           help="Enable automatic curriculum progression")
    both_parser.add_argument("--detect-changes", action="store_true", 
                           help="Enable environment change detection")
    both_parser.add_argument("--prioritized-replay", action="store_true", 
                           help="Use prioritized experience replay")
    both_parser.add_argument("--output-dir", type=str, default="results", 
                           help="Output directory")
    both_parser.add_argument("--eval-episodes", type=int, default=5, 
                           help="Number of evaluation episodes per level")
    
    args = parser.parse_args()
    
    if args.command == "train":
        run_training(args)
    elif args.command == "analyze":
        if args.output_dir is None:
            args.output_dir = os.path.join(os.path.dirname(args.model_path), "analysis")
        run_analysis(args, os.path.dirname(args.model_path))
    elif args.command == "run-both":
        output_dir = run_training(args)
        run_analysis(args, output_dir)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

# file: analyze_recordings.py
"""
Analyze all training recordings to find patterns
"""

import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def analyze_recording(filepath):
    """Analyze a single recording"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    episode = data["episode"]
    score = data["score"]
    epsilon = data["epsilon"]
    frames = data["frames"]
    
    # Calculate statistics
    total_frames = len(frames)
    jump_count = sum(1 for f in frames if f["action"] == 1)
    jump_rate = (jump_count / total_frames * 100) if total_frames > 0 else 0
    
    q_jumps = [f["q_jump"] for f in frames]
    q_no_jumps = [f["q_no_jump"] for f in frames]
    rewards = [f["reward"] for f in frames]
    bird_ys = [f["bird_y"] for f in frames]
    bird_vels = [f["bird_vel"] for f in frames]
    
    # Check for common failure patterns
    early_death = total_frames < 50
    excessive_jumping = jump_rate > 80
    no_jumping = jump_rate < 5
    
    # Check if bird tends to go too high or too low
    avg_y = np.mean(bird_ys)
    too_high = avg_y < 200
    too_low = avg_y > 600
    
    # Check Q-value behavior
    q_jump_wins = sum(1 for i in range(len(q_jumps)) if q_jumps[i] > q_no_jumps[i])
    q_jump_win_rate = (q_jump_wins / len(q_jumps) * 100) if len(q_jumps) > 0 else 0
    
    return {
        "episode": episode,
        "score": score,
        "epsilon": epsilon,
        "total_frames": total_frames,
        "jump_count": jump_count,
        "jump_rate": jump_rate,
        "avg_q_jump": np.mean(q_jumps),
        "avg_q_no_jump": np.mean(q_no_jumps),
        "total_reward": sum(rewards),
        "avg_bird_y": avg_y,
        "avg_bird_vel": np.mean(bird_vels),
        "early_death": early_death,
        "excessive_jumping": excessive_jumping,
        "no_jumping": no_jumping,
        "too_high": too_high,
        "too_low": too_low,
        "q_jump_win_rate": q_jump_win_rate,
        "frames": frames  # Keep for detailed analysis
    }


def main():
    recordings_dir = Path("training_recordings")
    
    if not recordings_dir.exists():
        print("No training_recordings directory found!")
        print("Run v6_with_visualization.py first to generate recordings.")
        return
    
    recording_files = sorted(recordings_dir.glob("*.json"))
    
    if not recording_files:
        print("No recordings found in training_recordings/")
        return
    
    print(f"Found {len(recording_files)} recordings")
    print("=" * 80)
    
    analyses = []
    for filepath in recording_files:
        analysis = analyze_recording(filepath)
        analyses.append(analysis)
    
    # Print summary table
    print(f"{'Episode':<10} {'Score':<8} {'ε':<8} {'Frames':<8} {'Jump%':<8} {'Q(jump)':<10} {'Q(stay)':<10} {'Issues'}")
    print("-" * 80)
    
    for a in analyses:
        issues = []
        if a["early_death"]:
            issues.append("💀early")
        if a["excessive_jumping"]:
            issues.append("🚀spam")
        if a["no_jumping"]:
            issues.append("😴static")
        if a["too_high"]:
            issues.append("⬆️high")
        if a["too_low"]:
            issues.append("⬇️low")
        
        issue_str = " ".join(issues) if issues else "✅"
        
        print(f"{a['episode']:<10} {a['score']:<8} {a['epsilon']:<8.3f} {a['total_frames']:<8} "
              f"{a['jump_rate']:<8.1f} {a['avg_q_jump']:<10.3f} {a['avg_q_no_jump']:<10.3f} {issue_str}")
    
    print("=" * 80)
    
    # Overall patterns
    print("\n📊 OVERALL PATTERNS:")
    print("-" * 40)
    
    early_deaths = sum(1 for a in analyses if a["early_death"])
    excessive_jumpers = sum(1 for a in analyses if a["excessive_jumping"])
    no_jumpers = sum(1 for a in analyses if a["no_jumping"])
    
    print(f"Early deaths (<50 frames): {early_deaths}/{len(analyses)}")
    print(f"Excessive jumping (>80%): {excessive_jumpers}/{len(analyses)}")
    print(f"No jumping (<5%): {no_jumpers}/{len(analyses)}")
    
    # Q-value trends
    if analyses:
        first = analyses[0]
        last = analyses[-1]
        print(f"\n🧠 Q-VALUE EVOLUTION:")
        print(f"  Episode {first['episode']}: Q(jump)={first['avg_q_jump']:.3f}, Q(stay)={first['avg_q_no_jump']:.3f}")
        print(f"  Episode {last['episode']}: Q(jump)={last['avg_q_jump']:.3f}, Q(stay)={last['avg_q_no_jump']:.3f}")
        
        if first['avg_q_jump'] > 0 and last['avg_q_jump'] < 0:
            print("  ⚠️  Q-values became NEGATIVE! This is bad - model is in death spiral!")
        elif abs(last['avg_q_jump'] - last['avg_q_no_jump']) < 0.1:
            print("  ⚠️  Q-values are too SIMILAR! Model can't distinguish actions!")
        else:
            print("  ✅ Q-values seem reasonable")
    
    # Jump rate evolution
    print(f"\n🦘 JUMP RATE EVOLUTION:")
    for a in analyses:
        print(f"  Episode {a['episode']:4d}: {a['jump_rate']:5.1f}%  {'🚀' * int(a['jump_rate'] / 10)}")
    
    # Create visualization
    print("\n📈 Generating plots...")
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("Training Episode Analysis")
        
        episodes = [a["episode"] for a in analyses]
        scores = [a["score"] for a in analyses]
        jump_rates = [a["jump_rate"] for a in analyses]
        q_jumps = [a["avg_q_jump"] for a in analyses]
        q_no_jumps = [a["avg_q_no_jump"] for a in analyses]
        
        # Score over episodes
        axes[0, 0].plot(episodes, scores, 'o-', color='green')
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Score")
        axes[0, 0].set_title("Score Progress")
        axes[0, 0].grid(True)
        
        # Jump rate over episodes
        axes[0, 1].plot(episodes, jump_rates, 'o-', color='orange')
        axes[0, 1].axhline(y=50, color='gray', linestyle='--', alpha=0.5)
        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].set_ylabel("Jump Rate (%)")
        axes[0, 1].set_title("Jump Rate Evolution")
        axes[0, 1].grid(True)
        
        # Q-values over episodes
        axes[1, 0].plot(episodes, q_jumps, 'o-', color='red', label='Q(jump)')
        axes[1, 0].plot(episodes, q_no_jumps, 'o-', color='blue', label='Q(stay)')
        axes[1, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[1, 0].set_xlabel("Episode")
        axes[1, 0].set_ylabel("Q-Value")
        axes[1, 0].set_title("Q-Value Evolution")
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Q-value difference
        q_diffs = [a["avg_q_jump"] - a["avg_q_no_jump"] for a in analyses]
        axes[1, 1].plot(episodes, q_diffs, 'o-', color='purple')
        axes[1, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[1, 1].set_xlabel("Episode")
        axes[1, 1].set_ylabel("Q(jump) - Q(stay)")
        axes[1, 1].set_title("Q-Value Preference")
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig("training_recordings/analysis.png", dpi=150)
        print("  ✅ Saved plot to training_recordings/analysis.png")
        plt.show()
    except Exception as e:
        print(f"  ❌ Could not create plots: {e}")
    
    print("\n" + "=" * 80)
    print("💡 RECOMMENDATIONS:")
    print("-" * 40)
    
    if excessive_jumpers > len(analyses) // 2:
        print("⚠️  EXCESSIVE JUMPING detected!")
        print("   - The bird is jumping too much")
        print("   - This might be because:")
        print("     1. Jump action has higher Q-value bias")
        print("     2. Exploration is causing random jumps")
        print("     3. Reward for jumping is too high")
        print("   - Try: Increase penalty for unnecessary jumps")
    
    if no_jumpers > len(analyses) // 2:
        print("⚠️  NOT JUMPING ENOUGH detected!")
        print("   - The bird is not jumping at all")
        print("   - This might be because:")
        print("     1. No-jump action has higher Q-value")
        print("     2. Bird hasn't learned that jumping helps")
        print("     3. Not enough positive experiences")
        print("   - Try: Increase reward for passing pipes")
    
    if analyses and analyses[-1]['avg_q_jump'] < -1.0:
        print("⚠️  NEGATIVE Q-VALUES detected!")
        print("   - Model is in a death spiral")
        print("   - All actions look bad to the network")
        print("   - Try:")
        print("     1. Increase pass-pipe reward")
        print("     2. Decrease death penalty")
        print("     3. Reset training with better hyperparameters")


if __name__ == "__main__":
    main()
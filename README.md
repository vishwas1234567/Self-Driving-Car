**30-second pitch**

"I built a self-driving car simulator in pure Python. A car with 9 distance sensors learns to drive around a randomly generated track using Deep Q-Learning. I wrote the neural network, backpropagation and optimizer myself in NumPy, with no deep-learning framework, and verified the gradients numerically. It learns crash-free, full-speed driving in about 17,000 steps, and it worked on 4 unseen tracks without retraining. It was inspired by Karpathy's demo, but the implementation is my own."

**Key facts to remember**
- **Task:** drive fast along the road and avoid walls.
- **Input:** 10 numbers: 9 sensor distances plus the car's speed.
- **Actions:** 5 (coast, left, right, accelerate, brake).
- **Reward:** speed along the road direction, minus a small penalty near walls, and -1 for a crash.
- **Network:** 10 → 64 → 64 → 5, with ReLU activations and about 5,000 parameters.
- **Algorithm:** DQN with experience replay, a target network, Double-DQN and Huber loss, optimized with Adam.
- **Results:** it first completed a full crash-free episode at about 17k steps, and training takes about a minute on a CPU. All 3 seeds agreed, and it survived every episode on unseen tracks. The random driver covers 0.08 laps, my hand-written heuristic 7.8, and the DQN about 9.2.

**Likely questions and short answers**
- **Why replay and a target network?** Replay breaks the correlation between consecutive samples, and the target network stops the learning target from shifting under itself.
- **Why Double-DQN?** It reduces over-estimation of Q-values by having one network choose the action and the other evaluate it.
- **How did you know your backprop was right?** I compared it with finite-difference gradients, and the worst relative error was 3×10⁻⁷.
- **Why random start positions?** Otherwise the car only practises the first corner.
- **Why isn't a time-limit ending treated as terminal?** The car isn't actually in a failure state, so bootstrapping should continue.

**Being upfront about the limits**
- The DQN beats my heuristic mainly because it drives faster, so both survive but the lap count mostly measures speed.
- The unseen tracks come from the same generator as the training track.
- The physics is simplified, there's no traffic, and I ran only 3 seeds with no hyperparameter search.
- There was one small instability dip during training.
- If asked about the interface: I tested it only without a display, so say you ran the results through the command line, and run the window once on your own machine before the interview.

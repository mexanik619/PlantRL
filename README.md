# PlantRL
 
PlantRL is a small simulation where a plant learns to grow. The plant looks for water in the soil and light in the air. It learns with Soft Actor-Critic (SAC), a reinforcement learning method.
 
You can use PlantRL to:
 
- train a model that grows the plant,
- watch one episode of a model or a baseline as a GIF, MP4 or PNG,
- make a video of a whole training run,
- make a demo video of a trained model on new worlds.
---
 
## 1. Words used in this project
 
| Word | Meaning |
| --- | --- |
| **Episode** | One full growth attempt on one world. It ends when every plant part has stopped growing. |
| **World** | The soil and air for one episode: water pockets, light, obstacles. A world is set by a *seed* number. Same seed = same world. |
| **Stage** | A difficulty level, 1 to 3. Training moves to the next stage when the plant does well enough. |
| **Agent** | One growing tip of the plant. Every tip runs the same model. |
| **Primary root / shoot** | The first root (grows down) and the first shoot (grows up). Every episode starts with both. |
| **Lateral root / branch** | Side growth. Lateral roots appear when the primary root goes deep. Branches appear when the primary shoot gets tall. |
| **Action** | One number per step: how much the tip bends, from -0.35 to 0.35 radians. |
| **Observation** | 28 numbers the model sees each step: light, water, gravity, the tip's direction, hormones and more. |
| **Reward** | A score the model tries to maximise. It rewards getting closer to water or light and penalises collisions and wasted steps. |
| **Checkpoint** | A saved model file (`.zip`). |
| **Recording** | A saved `.npz` file of one episode. The viewer replays it without running the simulation again. |
| **Stage 1 / 2 / 3** | Stage 1: fixed light, one water pocket, no obstacles. Stage 2: random positions, some obstacles, gravity can tilt by 15°. Stage 3: moving light, drying soil, many obstacles, gravity can tilt by 30°. |
 
---
 
## 2. Install
 
You need Python 3.10 or newer. Run these commands in the project folder.
 
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
 
`requirements.txt` installs the project in editable mode with the viewer and test tools. After this, three commands are available: `plantrl-train`, `plantrl-viz` and `plantrl-render`.
 
Optional: install `ffmpeg` if you want `.mp4` videos. GIF and PNG do not need it.
 
---
 
## 3. Check that it works
 
```bash
python -m pytest -v
```
 
All tests should pass. If one fails, do not train until it passes.
 
---
 
## 4. Commands
 
Run every command from the project folder, with the virtual environment active.
 
| Command | What it does |
| --- | --- |
| `plantrl-train` | Trains a model through the stages and saves everything it sees. |
| `plantrl-viz` | Runs one episode and saves it as a picture or animation. Can also replay a saved recording. |
| `plantrl-render train` | Makes one video of a whole training run. |
| `plantrl-render demo` | Runs a trained model on several worlds and makes a demo video. |
| `python -m pytest -v` | Runs the tests. |
 
Every command also works as `python -m plantrl.<module>`, for example `python -m plantrl.train`. Use that form if the command name is not found.
 
### 4.1 `plantrl-train`: train a model
 
Simplest run:
 
```bash
plantrl-train --stage 1 --total-steps 1000000
```
 
| Option | Default | What it means |
| --- | --- | --- |
| `--stage` | `1` | Stage to start from: 1, 2 or 3. It moves up on its own. |
| `--total-steps` | `100000` | How many steps to train for. Use 1,000,000 or more for a useful model. |
| `--seed` | `42` | Random seed. Same seed = same training run. |
| `--log-dir` | `logs` | Where TensorBoard logs go. |
| `--checkpoint-dir` | `checkpoints` | Where saved models go. |
| `--run-dir` | `runs/train` | Where every episode is recorded. |
| `--no-record` | off | Turn recording off to save disk space. |
| `--detail-every` | `100` | Every Nth episode also saves hormone data. Keep it high. |
 
**Stage moves on its own.** Stage 1 moves to stage 2 when the average reward of the last 100 episodes reaches 0.5. Stage 2 moves to stage 3 at 1.2. Stage 3 is the last stage.
 
**Watch training progress:**
 
```bash
tensorboard --logdir logs
```
 
Then open the address it prints in your browser.
 
**What training saves:**
 
| Path | Contents |
| --- | --- |
| `checkpoints/best/best_model.zip` | The best model found during tests. **Use this one.** |
| `checkpoints/plant_sac_final.zip` | The model at the end of training. |
| `checkpoints/plant_sac_<steps>_steps.zip` | Regular snapshots. |
| `logs/` | TensorBoard files. |
| `runs/train/episodes.csv` | One row per episode: stage, steps, return, reward per part. |
| `runs/train/episodes/*.npz` | Every episode, recorded. |
 
### 4.2 `plantrl-viz`: watch one episode
 
Run a policy on one world and save the result:
 
```bash
plantrl-viz --stage 1 --policy zero --seed 0 --out runs/viz/zero_s1.gif
plantrl-viz --stage 1 --policy random --seed 0 --out runs/viz/random_s1.gif
plantrl-viz --stage 1 --policy checkpoint --model checkpoints/best/best_model.zip --seed 0 --out runs/viz/best_s1.mp4
```
 
| Option | Default | What it means |
| --- | --- | --- |
| `--policy` | `random` | `zero` = do nothing. `random` = random actions. `checkpoint` = your trained model. |
| `--model` | none | Path to a `.zip` model. Required with `--policy checkpoint`. |
| `--stage` | `1` | World stage: 1, 2 or 3. |
| `--seed` | `0` | World seed. Use the same seed for every policy to compare them. |
| `--out` | `runs/viz/episode.gif` | Output file: `.gif`, `.mp4` or `.png`. |
| `--save-npz` | next to `--out` | Where the recording is saved. |
| `--replay` | none | Replay a saved `.npz` file. The env and model are not run. |
| `--fields` | `moisture light` | Layers to draw: `moisture`, `light`, `auxin`, `cyto`. |
| `--no-fields` | off | Do not save hormone data. Smaller file. |
| `--fps` | `12` | Frames per second. |
| `--every` | `1` | Draw every Nth step. Higher = shorter animation. |
| `--dpi` | `80` | Picture sharpness. |
| `--frame` | last step | Which step to draw when the output is `.png`. |
| `--show` | off | Open a window instead of saving. |
 
Replay a saved recording:
 
```bash
plantrl-viz --replay runs/viz/zero_s1.npz --out again.gif
```
 
**What the picture shows:** the soil and air on the left. The plant is coloured by agent type. On the right are the step info, the reward added up by part, and the 28 observation numbers.
 
### 4.3 `plantrl-render train`: video of a whole training run
 
```bash
plantrl-render train runs/train --out runs/train/training.mp4 --snapshot runs/train/final.png
```
 
| Option | Default | What it means |
| --- | --- | --- |
| `run_dir` | required | The folder from `plantrl-train`, usually `runs/train`. |
| `--out` | `<run_dir>/training.mp4` | Output file. |
| `--frames` | `120` | How many episodes to sample across training. |
| `--fps` | `12` | Frames per second. |
| `--snapshot` | none | Also save the final dashboard as a PNG. |
 
The video shows the learning curve, the reward for each part, and the branches and success rate over time.
 
### 4.4 `plantrl-render demo`: demo of a trained model
 
```bash
plantrl-render demo --model checkpoints/best/best_model.zip --stage 1 --seeds 0 1 2 --baseline --out runs/demo.mp4
```
 
| Option | Default | What it means |
| --- | --- | --- |
| `--model` | required | Path to a `.zip` model. |
| `--stage` | `1` | World stage. |
| `--seeds` | `0 1 2` | Worlds to test on. Each seed is one world. |
| `--baseline` | off | Also run a do-nothing plant on the same worlds, for comparison. |
| `--out` | `runs/demo.mp4` | Output file: `.mp4` or `.gif`. |
| `--fps` | `12` | Frames per second. |
| `--every` | `2` | Draw every Nth step. |
 
Each world gets a title card, the full episode, and a result card with the return, whether water and light were found, and the number of branches.
 
---
 
## 5. Typical workflow
 
1. Install (section 2) and run the tests (section 3).
2. Train: `plantrl-train --stage 1 --total-steps 1000000`.
3. Check progress in TensorBoard: `tensorboard --logdir logs`.
4. Look at one episode of the trained model:
   `plantrl-viz --stage 1 --policy checkpoint --model checkpoints/best/best_model.zip --out runs/viz/best_s1.gif`
5. Make the training video: `plantrl-render train runs/train --out runs/train/training.mp4`.
6. Make the demo video: `plantrl-render demo --model checkpoints/best/best_model.zip --baseline --out runs/demo.mp4`.
---
 
## 6. How the plant learns
 
- **Reward.** Each step gives a score made of:
  - `progress`: a reward for moving closer to the water (roots) or light (shoots and branches).
  - `hydro` and `photo`: a reward for growing toward the water or light gradient.
  - `gravi`: a small reward for growing down. Only roots use it, with a weight of 0.1.
  - `cost`: a small cost every step, and a bigger cost for hitting an obstacle.
  - `goal`: a bonus of 10 when the plant actually finds water or light. Reaching the depth or height limit pays nothing.
- **Why gravity is weak.** A strong gravity reward makes the plant grow straight down and ignore the water. Keep it small.
- **Why stage 1 pockets are wide.** The root can sense water from far away, so it can learn at all.
- **One model for every tip.** All agents share one model. Their behaviour differs because they see different numbers, such as whether they are a root and how deep they are.
---
 
## 7. Troubleshooting
 
| Problem | Fix |
| --- | --- |
| `plantrl-train: command not found` | Activate the virtual environment, then run `pip install -r requirements.txt` from the project folder. |
| `ModuleNotFoundError: No module named 'plantrl'` | Same fix as above. Do not run files inside `src/` directly. |
| `ImportError: attempted relative import with no known parent package` | You ran a file by path, for example `python src/plantrl/viz/render.py`. Run the command instead, or use `python -m plantrl.visualize`. |
| `.mp4` fails with an ffmpeg error | Install ffmpeg, or use `.gif` instead. |
| `--policy checkpoint needs --model` | Add `--model checkpoints/best/best_model.zip`. |
| `plantrl-render` shows only the usage message | It needs a subcommand: `train` or `demo`. |
| Training runs but the plant never learns | Check `tensorboard --logdir logs`. The reward should rise over time. Train longer, or check that the tests pass. |
 
---
 
## 8. Project layout
 
```
PlantRL/
├── README.md
├── requirements.txt          # pip install -r requirements.txt
├── pyproject.toml            # package info, dependencies, the three commands
├── src/plantrl/
│   ├── train.py              # plantrl-train: training and the stage curriculum
│   ├── visualize.py          # plantrl-viz: one episode, rendered or replayed
│   ├── render_run.py         # plantrl-render: training video and demo video
│   ├── environment/env.py    # the Gymnasium environment (PlantEnv)
│   ├── agents/               # the four agent types and the 28-number observation
│   ├── world/                # light, moisture and hormone fields
│   └── viz/                  # recording and drawing code
└── tests/                    # pytest tests
```
 
Generated folders (`checkpoints/`, `logs/`, `runs/`) are git-ignored. Delete them any time and training or rendering will create them again.
 
---
 
## 9. Adding a new agent type
 
1. Create `src/plantrl/agents/my_agent.py` with a class that inherits from `BaseAgent`.
2. Add one line to `REGISTRY` in `src/plantrl/agents/agent_registry.py`.
3. Run `python -m pytest -v`. The reward test must still pass.
 

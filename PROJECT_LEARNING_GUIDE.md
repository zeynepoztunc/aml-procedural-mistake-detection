# Project Learning Guide: Mistake Detection in Egocentric Video

This guide explains the core concepts, machine learning theories, and practical
implementation details of the **AML/DAAI 2025 Mistake Detection Project**.

---

## 1. What is this project about?

### The Goal

Imagine a robot or an AI assistant watching you cook via a headset camera (like
a GoPro or AR glasses). Its job is to **spot mistakes** in real-time or offline.

If you are making an omelette and you forget to beat the eggs before pouring
them into the pan, the AI should flag this as a **Procedural Mistake**.

### The Context: "CaptainCook4D"

We are using the **CaptainCook4D** dataset. This is a collection of "Egocentric"
(first-person view) videos of people performing recipes.

- **Correct Executions:** The person follows the recipe perfectly.
- **Mistake Executions:** The person intentionally performs an error (e.g.,
  "Preparation Error", "Timing Error", "Temperature Error").

### Your Task

You are building an **Error Recognition System**.

1. **Input:** A video clip of a single step (e.g., "Whisking eggs").
2. **Output:** A binary classification: **Correct** or **Mistake**.

Later (in the extension), you will scale this up to look at the **entire video**
and check if the sequence of steps matches a valid "Task Graph" (a map of the
recipe).

---

## 2. Core Machine Learning Concepts

To solve this, the project relies on several advanced ML topics.

### A. Egocentric Vision & Feature Extraction

Processing raw video (images at 30 frames per second) is extremely
computationally expensive. You cannot feed hours of raw 4K video into a simple
neural network.

**Solution: Embeddings (Feature Extraction)** Instead of raw pixels, we use a
huge, pre-trained Deep Learning model (like **Omnivore** or **EgoVLP**) that has
already "watched" millions of videos.

1. We feed 1 second of video into this pre-trained model.
2. The model outputs a list of numbers (a vector, e.g., size 1024). This is the
   **Embedding**.
3. **Concept:** If two video clips look semantically similar (e.g., two people
   chopping onions), their embedding vectors are mathematically close to each
   other.

**In this project:** You are NOT training the video processor. You are training
a lightweight classifier on top of these pre-calculated embeddings.

#### Omnivore vs. EgoVLP: What's the difference?

You will see these two names a lot. They are different "eyes" for our AI.

| Feature           | **Omnivore**                                                                                                        | **EgoVLP (Egocentric Video-Language Pretraining)**                                                                                   |
| :---------------- | :------------------------------------------------------------------------------------------------------------------ | :----------------------------------------------------------------------------------------------------------------------------------- |
| **Type**          | **Uni-modal (mostly)**. It is a "Foundation Model" for computer vision.                                             | **Multi-modal (Video + Language)**.                                                                                                  |
| **Training Data** | Trained on Images (ImageNet), Videos (Kinetics), and 3D data.                                                       | Trained specifically on **Egocentric** (First-person) videos paired with narration text (Ego4D dataset).                             |
| **Why use it?**   | It is incredibly robust at recognizing **actions** ("cutting", "stirring") because it has seen millions of actions. | It understands the **link between words and video**. If the recipe says "Cut the tomato", EgoVLP knows exactly what that looks like. |
| **Project Usage** | _Baselines_: We use Omnivore because it gives strong, general visual features.                                      | _Extension_: We use EgoVLP because we need to match video steps to **text** descriptions in the Task Graph.                          |

### B. Sequence Modeling (Temporal Reasoning)

A mistake often isn't just about a single frame; it's about _how_ something
happens over time.

- **Baseline V1: MLP (Multi-Layer Perceptron)**
  - **How it works:** It takes the features, averages them or concatenates them,
    and passes them through simple dense layers.
  - **Pro:** Very fast.
  - **Con:** It has no sense of "time". It treats the beginning and end of a
    clip equally. It struggles with fine-grained temporal mistakes.

- **Baseline V2: Transformer (Self-Attention)**
  - **How it works:** It uses the **Attention Mechanism**. It looks at the
    sequence of features (Time $t_1, t_2, t_3...$) and learns relationships
    between them.
  - **Example:** It might learn that "The hand moving _towards_ the salt shaker"
    usually precedes "Shaking motion". If the order is swapped, the Transformer
    notices the pattern disruption.
  - **Why it's better:** It understands the _order_ of events within the step.

### C. Task Graphs (Directed Acyclic Graphs - DAGs)

This is relevant for the **Extension**. A recipe is not always a straight line.

- _Make Toast:_ You can (slice bread $\to$ toast) OR (toast $\to$ slice bread?).
- Real recipes have flexible orders.

A **Task Graph** represents all valid paths to complete a goal.

- **Nodes:** The steps (e.g., "Wash tomato", "Cut tomato").
- **Edges:** The valid transitions.

**GNN (Graph Neural Networks) - The "Inspector"** In the extension, you map the
video you saw to this graph. You use a GNN to answer: "Does the path I just
watched exist in the Graph of valid paths?"

- **What is it?** A neural network designed to work on _connected data_
  (graphs), not just lists or images.
- **How it works (Message Passing):** Imagine a group of people (Nodes) holding
  hands (Edges). Each person knows something (Features). In a GNN, friends talk
  to friends. They pass information along the edges. After a few rounds of
  talking, every node knows about its neighborhood.
- **In this project:**
  - **Nodes:** Represents the recipe steps video clips we detected.
  - **Edges:** Represents the _order_ we saw them happen.
  - **The Task:** The GNN looks at this structure and classifies the _entire
    graph_ as "Success" or "Failure". It creates a "Graph Embedding" (summary of
    the whole web) and classifies that.

---

## 3. Project Architecture Details

### The Data Flow

1. **Raw Video** $\to$ **Feature Extractor (Omnivore/EgoVLP)** $\to$ **Features
   (.npy files)**.
   - _This part is already done for you. You download the features._
2. **DataLoader (`CaptainCookStepDataset`)**:
   - Reads the features for a specific step.
   - Reads the JSON annotation: "Was this step correct? What kind of error?"
3. **Model (`core/models/`)**:
   - Inputs: A tensor of shape `(Batch_Size, Time_Steps, Feature_Dim)`.
   - Outputs: A probability score (0 to 1) for "Is Mistake".
4. **Training (`base.py`)**:
   - Uses **Binary Cross Entropy Loss (BCE)**.
   - Compares predictions to ground truth labels.
   - Optimizes weights using **Adam**.

---

## 4. Glossary for this Project

- **Sub-step:** A small chunk of the video (e.g., 1 second).
- **Step:** A logical unit of the recipe (e.g., "Frying the egg"). Contains
  multiple sub-steps.
- **Backbone:** The heavy pre-trained model used for feature extraction
  (Omnivore, SlowFast).
- **ActionFormer:** A model used in the extension to _find_ where steps start
  and end in a long video.
- **EgoVLP (Egocentric Video-Language Pretraining):** A specialized backbone
  that is very good at aligning video with text descriptions (useful for
  matching video steps to text recipe steps).

---

## 5. Part 2: Deep Dive into Mechanics

### A. Understanding the Dataset Structure

The dataset relies on structured JSON files in `annotations/`.

- **`step_annotations.json`**: The ground truth for training.
  ```json
  "1_7": {  // Recording ID (Video ID)
      "steps": [
          {
              "step_id": 3,
              "start_time": 7.07,  // When the step starts in the video
              "end_time": 46.28,   // When it ends
              "description": "Coat a 6-oz. ramekin cup...",
              "has_errors": false  // IMPORTANT: The target label (0)
          },
          {
             ...
             "has_errors": true   // Target label (1)
          }
      ]
  }
  ```
- **`error_annotations.json`**: Provides details on _why_ a step is wrong.
  - Used for the "Per-Error Analysis" requirement. It tells us if the error was
    `TemperatureError` (too hot), `TimingError` (too fast/slow), etc.

### B. Extension Logic: "Task Verification"

Phase 2 changes the game. Instead of asking "Is this 10-second clip wrong?", we
ask "Is this entire 5-minute video a valid execution of the recipe?".

**1. Step Localization (ActionFormer)** Since we input a raw video, we first
need to find _where_ the steps are.

- **Input:** Full video features.
- **Output:** A list of `(Start, End)` timestamps.
- _Analogy:_ Like highlighting sentences in a paragraph.

**2. Graph Matching (The "Alignment" Problem)** We have a set of detected video
steps ($V_1, V_2, V_3$) and a "Task Graph" of recipe instructions
($T_1, T_2, T_3$).

- We use **EgoVLP** here. Why? Because we can convert the Video $V_1$ to a
  vector, and the Text Instruction $T_1$ to a vector.
- If $V_1$ matches $T_1$, their vectors will be close.
- We use the **Hungarian Algorithm** to find the optimal assignment (e.g.,
  $V_1 \to T_1$, $V_2 \to T_2$).

**3. GNN Classification** Once we've "filled in" the Task Graph with our
observed video clips, we ask the GNN:

- "Is this path valid?"
- If the video showed steps $A \to C \to B$, but the graph says only
  $A \to B \to C$ is allowed, the GNN should predict **Mistake**.

---

## 3. The Training Process: From "Dumb" to "Smart"

You ran `train_er.py` and saw terms like `Epoch`, `Loss`, `Precision`, and
`Recall`. What is actually happening inside the machine?

### A. The Players

#### 1. The Model (The Student)

The **Model** (our Transformer) is a mathematical function with millions of
adjustable knobs (Weights).

- **Start:** At Epoch 0, the knobs are set randomly. The student is guessing
  blindly.
- **Goal:** To tune the knobs so that its guesses match reality.

#### 2. Ground Truth (The Answer Key)

**Ground Truth** is absolute reality, created by human experts.

- **Location:** `annotations/` folder (JSON files).
- **Example:** "At 00:15, the user cut the tomato incorrectly (Error)."
- **Role:** The model peeks at this during _Training_ to learn, but it is hidden
  during _Testing_.

#### 3. The Loss Function (The Judge)

The **Loss Function** (`BCEWithLogitsLoss`) is the strict teacher grading the
exam. It calculates a "Penalty Score" based on how wrong the model is.

- **Scenario:** Model says "90% Error". Reality says "Normal".
- **Result:** HUGE Penalty (High Loss).
- **Goal:** The entire mathematical purpose of training is to drive this Loss
  score to **Zero**.

### B. The Learning Cycle (What happens in 1 Epoch?)

An **Epoch** is one complete study session where the model looks at _every
single video_ in the dataset once.

1. **Forward Pass (The Guess):**
   - The model looks at a batch of 32 video clips.
   - It outputs probabilities: `[0.1, 0.8, 0.4...]`.

2. **Loss Calculation (The Grade):**
   - The System compares guesses to the **Ground Truth**.
   - It calculates the Total Error (Loss) for that batch.

3. **Backpropagation (The Correction):**
   - This is the "Learning." The system uses calculus to figure out _which
     knobs_ caused the error.
   - The **Optimizer** (Adam) turns those knobs slightly in the right direction.

4. **Repeat:**
   - Do this for all batches. That is 1 Epoch.
   - Repeat for 50-100 Epochs until the Loss stops going down.

---

## 4. Interpreting Results: The "Boy Who Cried Wolf"

When analyzing your results, you saw **High Recall (1.0)** and **Low Precision
(0.41)**. This is a classic machine learning situation.

### Recall: "The Safety Net"

_Question: "Did we catch all the bad guys?"_

- **Recall = 1.0 (100%)** means your model **never missed a mistake**.
- Every time a user made an error, the model flagged it.
- **Why it's high:** The model is "Paranoid." It likely predicts "Error" very
  often to be safe.

### Precision: "The Trust Factor"

_Question: "When you flagged a bad guy, was he actually bad?"_

- **Precision = 0.41 (41%)** means when the model shouted "ERROR!", it was wrong
  59% of the time.
- These are **False Positives** (False Alarms).
- **The Trade-off:** A paranoid model (High Recall) is annoying (Low Precision).
  A lazy model might have High Precision (only flags obvious errors) but Low
  Recall (misses subtle ones).

### The "NaN" (Not a Number) Loss Issue

You observed `Loss: NaN` warnings. This is critical.

- **Meaning:** During the math calculation, a number became `Infinity` or
  `undefined` (like dividing by zero).
- **Cause:** Usually caused by "Dirty Data" (Corrupt `.npz` files) or numbers
  getting too big (Exploding Gradients).
- **Effect:** Once a NaN appears, it infects the whole model like a virus. The
  weights become NaN, and the model stops learning (Sub-Step Precision becomes
  0.0).

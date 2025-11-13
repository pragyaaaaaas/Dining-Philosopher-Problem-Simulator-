# dining_philosophers_app.py
"""
Dining Philosophers simulation with Streamlit visualization.

Features:
- Choose number of philosophers (3..12).
- Choose algorithm:
    * Naive (pick left then right) -> can deadlock
    * Resource hierarchy (impose fork ordering) -> avoids deadlock
    * Waiter (a mutex allowing up to N-1 philosophers to try) -> avoids deadlock
- Speed slider controls thinking/eating durations
- Start / Pause / Stop / Reset controls
- Matplotlib visualization drawing philosophers (states) and forks (free/in-use)
- Thread-safe shared state and graceful stop

Author: ChatGPT (example)
"""

import streamlit as st
import threading
import time
import math
import random
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict
import matplotlib.pyplot as plt
import numpy as np

# -------------------
# Simulation classes
# -------------------

class State(Enum):
    THINKING = "Thinking"
    HUNGRY = "Hungry"
    EATING = "Eating"

@dataclass
class Philosopher:
    idx: int
    state: State = State.THINKING
    times_eaten: int = 0
    last_state_change: float = field(default_factory=time.time)

# -------------------
# Utility functions
# -------------------

def left_idx(i, n): return (i - 1) % n
def right_idx(i, n): return (i + 1) % n

# -------------------
# Visualization
# -------------------

def draw_table(philosophers: List[Philosopher], fork_taken_by: Dict[int, int], title="Dining Philosophers"):
    """
    Draws a circular table where philosophers are placed on a circle and forks are between them.
    fork_taken_by: mapping fork_index -> philosopher_index or -1 if free
    """
    n = len(philosophers)
    radius = 2.5
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # draw table circle
    table = plt.Circle((0,0), 1.2, color="#F5F0E1", zorder=0)
    ax.add_artist(table)

    # draw forks (between i and right_i)
    for i in range(n):
        # fork position is halfway between philosopher i and right philosopher
        x1, y1 = x[i], y[i]
        xr, yr = x[(i+1)%n], y[(i+1)%n]
        fx, fy = (x1 + xr)/2, (y1 + yr)/2
        # if fork is taken, draw thicker and indicate by marker toward taker
        taker = fork_taken_by.get(i, -1)
        if taker == -1:
            ax.plot([x1*0.85, xr*0.85], [y1*0.85, yr*0.85], linewidth=2, linestyle='-', alpha=0.7)
            ax.scatter([fx],[fy], s=70, marker='|')  # visual fork
        else:
            # draw line in a stronger style pointing from fork toward taker
            tx, ty = x[taker], y[taker]
            ax.plot([fx, tx*0.95], [fy, ty*0.95], linewidth=4, linestyle='-', alpha=0.9)
            ax.scatter([fx],[fy], s=120, marker='|')
            ax.text(fx, fy, f"{i}", fontsize=8, ha='center', va='center')

    # draw philosophers
    for i, p in enumerate(philosophers):
        color = {
            State.THINKING: "#A6CEE3",
            State.HUNGRY: "#FDBF6F",
            State.EATING: "#B2DF8A"
        }[p.state]
        circ = plt.Circle((x[i], y[i]), 0.35, color=color, ec="k", linewidth=0.7)
        ax.add_artist(circ)
        ax.text(x[i], y[i], f"P{i}\n{p.state.value}", ha='center', va='center', fontsize=8)
        # small eat-count
        ax.text(x[i], y[i]-0.55, f"Eaten:{p.times_eaten}", ha='center', va='center', fontsize=7)

    ax.set_title(title)
    return fig

# -------------------
# Simulation engine
# -------------------

class DiningSimulation:
    def __init__(self, n_philosophers:int, algorithm:str, speed:float):
        self.n = n_philosophers
        self.algorithm = algorithm  # 'naive','hierarchy','waiter'
        self.speed = speed
        self.philosophers: List[Philosopher] = [Philosopher(i) for i in range(n_philosophers)]
        # one lock per fork
        self.forks = [threading.Lock() for _ in range(self.n)]
        # who holds which fork: -1 = free, else philosopher idx
        self.fork_taken_by = {i:-1 for i in range(self.n)}
        # threads and control
        self.threads: List[threading.Thread] = []
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()  # when set => running; when clear => paused
        self.pause_event.set()  # start as running
        # waiter lock if algorithm == 'waiter'
        self.waiter = threading.Semaphore(self.n - 1) if self.algorithm == 'waiter' else None
        # ordering for hierarchy: each fork has index, philosophers pick smaller id first
        self.lock = threading.Lock()  # to protect shared fork_taken_by and philosopher states

    def _set_state(self, i:int, state:State):
        with self.lock:
            self.philosophers[i].state = state
            self.philosophers[i].last_state_change = time.time()

    def _try_acquire_forks(self, left_lock, right_lock, left_idx_i, right_idx_i, i):
        # Acquire with blocking respecting pause/stop
        # We'll attempt non-blocking with small sleeps so the GUI remains responsive and can pause.
        while not self.stop_event.is_set():
            # respect pause
            self.pause_event.wait()

            # naive approach: try left then right (can deadlock)
            acquired_left = left_lock.acquire(timeout=0.1)
            if not acquired_left:
                continue

            # mark left as taken
            with self.lock:
                self.fork_taken_by[left_idx_i] = i

            acquired_right = right_lock.acquire(timeout=0.1)
            if not acquired_right:
                # release left and continue
                with self.lock:
                    self.fork_taken_by[left_idx_i] = -1
                left_lock.release()
                # small backoff
                time.sleep(0.01 + random.random()*0.02)
                continue
            # mark right as taken
            with self.lock:
                self.fork_taken_by[right_idx_i] = i

            return True  # both acquired

        return False

    def _acquire_ordered(self, i):
        """For resource hierarchy: always pick lower-indexed fork first, then higher"""
        left_i = i
        right_i = (i+1) % self.n
        a, b = (left_i, right_i) if left_i < right_i else (right_i, left_i)
        lock_a = self.forks[a]
        lock_b = self.forks[b]
        while not self.stop_event.is_set():
            self.pause_event.wait()
            acquired_a = lock_a.acquire(timeout=0.1)
            if not acquired_a:
                continue
            with self.lock:
                self.fork_taken_by[a] = i
            acquired_b = lock_b.acquire(timeout=0.1)
            if not acquired_b:
                with self.lock:
                    self.fork_taken_by[a] = -1
                lock_a.release()
                time.sleep(0.01 + random.random()*0.02)
                continue
            with self.lock:
                self.fork_taken_by[b] = i
            return (a,b)

        return None

    def philosopher_loop(self, i:int):
        n = self.n
        left = i
        right = (i + n - 1) % n  # right fork index relative to philosopher indexing used here
        while not self.stop_event.is_set():
            # Thinking
            self._set_state(i, State.THINKING)
            think_time = random.uniform(0.5, 1.5) * self.speed
            # small sleeps to be responsive to pause/stop
            t0 = time.time()
            while time.time() - t0 < think_time:
                if self.stop_event.is_set(): return
                self.pause_event.wait()
                time.sleep(0.05)

            # Hungry
            self._set_state(i, State.HUNGRY)

            # Acquire forks depending on algorithm
            if self.algorithm == "naive":
                # left fork index we call 'i' for left in drawing, right is (i+1)%n
                left_idx_f = i
                right_idx_f = (i+1)%n
                success = self._try_acquire_forks(self.forks[left_idx_f], self.forks[right_idx_f], left_idx_f, right_idx_f, i)
                if not success:
                    break

            elif self.algorithm == "hierarchy":
                # order by fork index
                res = self._acquire_ordered(i)
                if res is None:
                    break
                a, b = res
                # already updated fork_taken_by inside _acquire_ordered

            elif self.algorithm == "waiter":
                # waiter/host limits concurrency
                self.waiter.acquire()  # blocks until allowed
                left_idx_f = i
                right_idx_f = (i+1) % n
                got_left = self.forks[left_idx_f].acquire(timeout=0.5)
                if not got_left:
                    self.waiter.release()
                    continue
                with self.lock:
                    self.fork_taken_by[left_idx_f] = i
                got_right = self.forks[right_idx_f].acquire(timeout=0.5)
                if not got_right:
                    with self.lock:
                        self.fork_taken_by[left_idx_f] = -1
                    self.forks[left_idx_f].release()
                    self.waiter.release()
                    continue
                with self.lock:
                    self.fork_taken_by[right_idx_f] = i

            else:
                # unknown algorithm -> stop
                break

            # Eating
            self._set_state(i, State.EATING)
            eat_time = random.uniform(0.5, 1.2) * self.speed
            t0 = time.time()
            while time.time() - t0 < eat_time:
                if self.stop_event.is_set(): break
                self.pause_event.wait()
                time.sleep(0.05)

            # release forks
            if self.algorithm == "hierarchy":
                # release both in correct order
                left_idx_f = i
                right_idx_f = (i+1)%n
                # release both forks and clear taken-by
                with self.lock:
                    self.fork_taken_by[left_idx_f] = -1
                    self.fork_taken_by[right_idx_f] = -1
                # release the actual locks (acquired possibly as a,b order)
                a, b = (left_idx_f, right_idx_f) if left_idx_f < right_idx_f else (right_idx_f, left_idx_f)
                # release both
                self.forks[b].release()
                self.forks[a].release()

            elif self.algorithm == "naive":
                left_idx_f = i
                right_idx_f = (i+1)%n
                with self.lock:
                    self.fork_taken_by[left_idx_f] = -1
                    self.fork_taken_by[right_idx_f] = -1
                # release right then left (or either) - ensure they were acquired
                try:
                    self.forks[right_idx_f].release()
                except RuntimeError:
                    pass
                try:
                    self.forks[left_idx_f].release()
                except RuntimeError:
                    pass

            elif self.algorithm == "waiter":
                left_idx_f = i
                right_idx_f = (i+1)%n
                with self.lock:
                    self.fork_taken_by[left_idx_f] = -1
                    self.fork_taken_by[right_idx_f] = -1
                try:
                    self.forks[right_idx_f].release()
                except RuntimeError:
                    pass
                try:
                    self.forks[left_idx_f].release()
                except RuntimeError:
                    pass
                self.waiter.release()

            # increment eat count
            with self.lock:
                self.philosophers[i].times_eaten += 1

            # short random backoff
            time.sleep(0.01 + random.random()*0.05)

    def start(self):
        self.stop_event.clear()
        self.pause_event.set()
        # create threads
        self.threads = []
        for i in range(self.n):
            t = threading.Thread(target=self.philosopher_loop, args=(i,), daemon=True)
            self.threads.append(t)
            t.start()

    def pause(self):
        self.pause_event.clear()

    def resume(self):
        self.pause_event.set()

    def stop(self):
        self.stop_event.set()
        # resume to let threads exit if paused
        self.pause_event.set()
        # attempt to join threads (best-effort)
        for t in self.threads:
            if t.is_alive():
                t.join(timeout=0.2)

# -------------------
# Streamlit UI
# -------------------

st.set_page_config(page_title="Dining Philosophers Simulator", layout="wide")
st.title("Dining Philosophers — Simulation & Visualization")

# persistent variables in session_state
if "sim" not in st.session_state:
    st.session_state.sim = None
if "last_update" not in st.session_state:
    st.session_state.last_update = 0.0

# Controls column
with st.sidebar:
    st.header("Controls")
    n = st.slider("Number of philosophers", min_value=3, max_value=12, value=5, step=1)
    algorithm = st.selectbox("Algorithm", options=["naive", "hierarchy", "waiter"], index=1,
                             help="naive=left then right (can deadlock), hierarchy=order forks by index, waiter=host permits N-1 to try")
    speed = st.slider("Speed multiplier (higher=faster)", min_value=0.3, max_value=3.0, value=1.0, step=0.1)
    start_btn = st.button("Start")
    pause_btn = st.button("Pause")
    resume_btn = st.button("Resume")
    stop_btn = st.button("Stop / Reset")

# Create or update simulation object when parameters change or start pressed
if st.session_state.sim is None or st.session_state.sim.n != n or st.session_state.sim.algorithm != algorithm or abs(st.session_state.sim.speed - speed) > 1e-6:
    # if a sim exists, stop it
    if st.session_state.sim is not None:
        st.session_state.sim.stop()
    st.session_state.sim = DiningSimulation(n_philosophers=n, algorithm=algorithm, speed=speed)

sim: DiningSimulation = st.session_state.sim

if start_btn:
    # restart simulation fresh
    sim.stop()
    st.session_state.sim = DiningSimulation(n_philosophers=n, algorithm=algorithm, speed=speed)
    sim = st.session_state.sim
    sim.start()

if pause_btn:
    sim.pause()

if resume_btn:
    sim.resume()

if stop_btn:
    sim.stop()
    # reset state object
    st.session_state.sim = DiningSimulation(n_philosophers=n, algorithm=algorithm, speed=speed)
    sim = st.session_state.sim

# Main area layout
col1, col2 = st.columns([2,1])

with col1:
    st.subheader("Live visualization")
    placeholder = st.empty()

with col2:
    st.subheader("Philosopher states")
    # display a small table
    with st.session_state.get("table_area", st.container()):
        data = []
        for p in sim.philosophers:
            data.append({
                "Philosopher": f"P{p.idx}",
                "State": p.state.value,
                "Times eaten": p.times_eaten
            })
        st.table(data)

    st.markdown("**Fork ownership**")
    fork_text = ", ".join([f"F{i}->P{sim.fork_taken_by[i]}" if sim.fork_taken_by[i] != -1 else f"F{i}->free" for i in range(sim.n)])
    st.write(fork_text)

# Frequent redraw loop using Streamlit's rerun-friendly pattern
# We'll draw a frame every ~0.5s while simulation is running; if paused we still render state.
redraw_interval = 0.5

# draw current figure
fig = draw_table(sim.philosophers, sim.fork_taken_by, title=f"Algorithm: {sim.algorithm}  |  Speed: {sim.speed}")
placeholder.pyplot(fig)

# update the small right-hand table again (so counts refresh)
with col2:
    data = []
    for p in sim.philosophers:
        data.append({
            "Philosopher": f"P{p.idx}",
            "State": p.state.value,
            "Times eaten": p.times_eaten
        })
    st.table(data)
    st.markdown("**Fork ownership (detailed)**")
    for i in range(sim.n):
        owner = sim.fork_taken_by.get(i, -1)
        st.write(f"Fork {i} : {'free' if owner == -1 else f'held by P{owner}'}")

# Re-run after a delay to animate; using experimental_rerun is blocked inside code,
# so we use streamlit's sleep and st.experimental_rerun loop pattern by requesting user to toggle start/stop.
# Instead we'll use a small client-side refresh tactic:
st.write("")
st.write("**Tip:** Click Start, then interact with Pause/Resume/Stop. Adjust speed/algorithm and press Start again to restart simulation.")
# We can't block too long on the server; prompt the page to refresh using st.experimental_rerun after small sleep when sim is running
if any(t.is_alive() for t in sim.threads):
    # small sleep then rerun to animate (best-effort)
    time.sleep(redraw_interval)
    st.experimental_rerun()

import math
import os
import random
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

import tkinter as tk
from tkinter import filedialog, messagebox

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# settings
# -----------------------------
BOARD_SIZE = 9
WIN_LEN = 5
EMPTY = 0
BLACK = 1
WHITE = -1
MODEL_PATH = os.path.join(os.path.dirname(__file__), "gomoku_ai_model.pt")

CELL = 56
MARGIN = 28
BOARD_PIXELS = MARGIN * 2 + CELL * (BOARD_SIZE - 1)
WINDOW_W = 1220
WINDOW_H = 720

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Training hyperparameters ---
GAMMA           = 0.97     # discount factor for returns
ENTROPY_COEFF   = 0.02     # encourages exploration; decays with training
VALUE_COEFF     = 0.5      # weight of value loss
GRAD_CLIP       = 1.0      # gradient clipping max norm
LR              = 3e-4     # initial Adam learning rate
REPLAY_CAP      = 3000     # max trajectory items stored in replay buffer
REPLAY_BATCH    = 128      # items sampled per training update
NUM_CHANNELS    = 32       # CNN feature maps
NUM_RES_BLOCKS  = 3        # residual blocks in the tower

# --- Threat-response shaping constants ---
MISS_BLOCK_PENALTY = -0.90   # immediate penalty: opponent had a winning square and we ignored it
BLOCK_BONUS        =  0.15   # extra bonus stacked on top of reward_shaping when a forced block IS made
RETURN_CLAMP       =  2.5    # widen from 1.5 → 2.5 so penalty-shaped returns aren't clipped away


# -----------------------------
# game logic
# -----------------------------
class GomokuBoard:
    def __init__(self, size: int = BOARD_SIZE):
        self.size = size
        self.cells = [EMPTY] * (size * size)
        self.last_move: Optional[int] = None
        self.turn = BLACK

    def reset(self):
        self.cells = [EMPTY] * (self.size * self.size)
        self.last_move = None
        self.turn = BLACK

    def copy(self):
        b = GomokuBoard(self.size)
        b.cells = self.cells[:]
        b.last_move = self.last_move
        b.turn = self.turn
        return b

    def idx_to_rc(self, idx: int) -> Tuple[int, int]:
        return divmod(idx, self.size)

    def rc_to_idx(self, r: int, c: int) -> int:
        return r * self.size + c

    def legal_moves(self) -> List[int]:
        return [i for i, v in enumerate(self.cells) if v == EMPTY]

    def place(self, idx: int, player: int) -> bool:
        if idx < 0 or idx >= self.size * self.size:
            return False
        if self.cells[idx] != EMPTY:
            return False
        self.cells[idx] = player
        self.last_move = idx
        self.turn = -player
        return True

    def line_length_from(self, idx: int, player: int, dr: int, dc: int) -> int:
        r, c = self.idx_to_rc(idx)
        count = 1
        rr, cc = r + dr, c + dc
        while 0 <= rr < self.size and 0 <= cc < self.size and self.cells[self.rc_to_idx(rr, cc)] == player:
            count += 1; rr += dr; cc += dc
        rr, cc = r - dr, c - dc
        while 0 <= rr < self.size and 0 <= cc < self.size and self.cells[self.rc_to_idx(rr, cc)] == player:
            count += 1; rr -= dr; cc -= dc
        return count

    def _open_ends(self, idx: int, player: int, dr: int, dc: int) -> int:
        r, c = self.idx_to_rc(idx)
        open_ends = 0
        for sign in (1, -1):
            rr, cc = r + sign * dr, c + sign * dc
            while 0 <= rr < self.size and 0 <= cc < self.size and self.cells[self.rc_to_idx(rr, cc)] == player:
                rr += sign * dr; cc += sign * dc
            if 0 <= rr < self.size and 0 <= cc < self.size and self.cells[self.rc_to_idx(rr, cc)] == EMPTY:
                open_ends += 1
        return open_ends

    def check_winner(self, last_idx: Optional[int] = None) -> int:
        if last_idx is None:
            last_idx = self.last_move
        if last_idx is None:
            return EMPTY
        player = self.cells[last_idx]
        if player == EMPTY:
            return EMPTY
        for dr, dc in [(1, 0), (0, 1), (1, 1), (1, -1)]:
            if self.line_length_from(last_idx, player, dr, dc) >= WIN_LEN:
                return player
        return EMPTY

    def is_full(self) -> bool:
        return all(v != EMPTY for v in self.cells)

    def find_winning_moves(self, player: int) -> List[int]:
        winners = []
        for idx in self.legal_moves():
            self.cells[idx] = player
            if self.check_winner(idx) == player:
                winners.append(idx)
            self.cells[idx] = EMPTY
        return winners

    def reward_shaping(self, idx: int, player: int) -> float:
        dirs = [(1, 0), (0, 1), (1, 1), (1, -1)]

        opp = -player
        self.cells[idx] = opp          
        best_b_len, best_b_open = 1, 0
        for dr, dc in dirs:
            length = self.line_length_from(idx, opp, dr, dc)
            opens  = self._open_ends(idx, opp, dr, dc)
            if length > best_b_len or (length == best_b_len and opens > best_b_open):
                best_b_len, best_b_open = length, opens
        self.cells[idx] = player       

        if best_b_len >= 5:
            block_r = 0.90
        elif best_b_len == 4 and best_b_open == 2:
            block_r = 0.75
        elif best_b_len == 4 and best_b_open == 1:
            block_r = 0.55
        elif best_b_len == 4 and best_b_open == 0:
            block_r = 0.25   
        elif best_b_len == 3 and best_b_open == 2:
            block_r = 0.13   
        elif best_b_len == 3 and best_b_open == 1:
            block_r = 0.06   
        elif best_b_len == 2:
            block_r = 0.02
        else:
            block_r = 0.0

        best_len, best_open = 1, 0
        for dr, dc in dirs:
            length = self.line_length_from(idx, player, dr, dc)
            opens  = self._open_ends(idx, player, dr, dc)
            if length > best_len or (length == best_len and opens > best_open):
                best_len, best_open = length, opens

        if best_len >= 5:
            own_r = 0.0   
        elif best_len == 4:
            own_r = 0.50 if best_open == 2 else (0.33 if best_open == 1 else 0.17)
        elif best_len == 3:
            own_r = 0.18 if best_open == 2 else (0.09 if best_open == 1 else 0.02)
        elif best_len == 2:
            own_r = 0.03 if best_open >= 1 else 0.01
        else:
            own_r = 0.0

        if block_r >= 0.55:
            return block_r          

        return min(own_r + block_r, 0.60)


# -----------------------------
# neural network
# -----------------------------
class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.bn1  = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2  = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        return x + out   


class PolicyValueNet(nn.Module):
    def __init__(self, board_size: int = BOARD_SIZE,
                 num_channels: int = NUM_CHANNELS,
                 num_res_blocks: int = NUM_RES_BLOCKS):
        super().__init__()
        self.board_size = board_size
        N = board_size * board_size

        self.stem = nn.Sequential(
            nn.Conv2d(3, num_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True),
        )

        self.res_tower = nn.Sequential(
            *[ResBlock(num_channels) for _ in range(num_res_blocks)]
        )

        self.policy_conv = nn.Conv2d(num_channels, 32, kernel_size=1, bias=False)
        self.policy_bn   = nn.BatchNorm2d(32)
        self.policy_fc   = nn.Linear(32 * N, N)

        self.value_conv = nn.Conv2d(num_channels, 16, kernel_size=1, bias=False)
        self.value_bn   = nn.BatchNorm2d(16)
        self.value_fc1  = nn.Linear(16 * N, 256)
        self.value_fc2  = nn.Linear(256, 1)

    def forward(self, x):
        h = self.stem(x)
        h = self.res_tower(h)

        p = F.relu(self.policy_bn(self.policy_conv(h)))
        p_flat = p.flatten(1)                        
        logits  = self.policy_fc(p_flat)             

        v = F.relu(self.value_bn(self.value_conv(h)))
        v_flat = v.flatten(1)                        
        v_h    = F.relu(self.value_fc1(v_flat))      
        value  = torch.tanh(self.value_fc2(v_h)).squeeze(-1)  

        activations = {
            "a1": h.detach().mean(dim=1).flatten(),        
            "a2": p_flat.detach().squeeze(0),
            "a3": v_h.detach().squeeze(0),
        }
        return logits, value, activations


# -----------------------------
# utilities
# -----------------------------
def encode_board(board: GomokuBoard, player: int) -> torch.Tensor:
    size = board.size
    own  = torch.zeros(size, size, dtype=torch.float32)
    opp  = torch.zeros(size, size, dtype=torch.float32)
    turn_val = 1.0 if player == BLACK else 0.0
    turn = torch.full((size, size), turn_val, dtype=torch.float32)

    for idx, v in enumerate(board.cells):
        r, c = divmod(idx, size)
        if v == player:
            own[r, c] = 1.0
        elif v != EMPTY:
            opp[r, c] = 1.0

    return torch.stack([own, opp, turn], dim=0)   


def masked_softmax(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    masked = logits.clone()
    masked[~legal_mask] = -1e9
    return F.softmax(masked, dim=-1)


def choose_action(logits: torch.Tensor, legal_moves: List[int],
                  epsilon: float = 0.0) -> Tuple[int, torch.Tensor]:
    N = logits.shape[-1]
    legal_mask = torch.zeros(N, dtype=torch.bool)
    legal_mask[legal_moves] = True
    probs = masked_softmax(logits, legal_mask)
    if random.random() < epsilon:
        action = random.choice(legal_moves)
    else:
        action = int(torch.multinomial(probs, 1).item())
    return action, probs


@dataclass
class TrajectoryItem:
    state:           torch.Tensor   # (3, H, W)
    action:          int
    player:          int
    shaped_reward:   float
    legal_mask:      torch.Tensor   # (H*W,) bool 
    tactical_target: torch.Tensor   # (H*W,) float - supervised distribution target


# -----------------------------
# training engine
# -----------------------------
class TrainingEngine:
    def __init__(self, board_size: int = BOARD_SIZE):
        self.board_size = board_size
        self.model = PolicyValueNet(board_size).to(DEVICE)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=LR, weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=600, T_mult=2, eta_min=5e-6
        )
        self.replay_buffer: deque = deque(maxlen=REPLAY_CAP)

        self.games_played  = 0
        self.steps_played  = 0
        self.last_loss     = 0.0
        self.last_tactical_loss = 0.0
        self.last_result   = "booting"
        self.running       = True
        self.training_enabled = True
        self.model_lock    = threading.Lock()
        self.state_lock    = threading.Lock()

        # --- Debugging/Instrumentation State ---
        self.threats_encountered = 0
        self.threats_blocked     = 0
        self.threats_missed      = 0
        self.avg_threat_prob     = 0.0

        # visualiser state
        self.human_mode      = False
        self.viz_board       = [EMPTY] * (board_size * board_size)
        self.viz_turn        = BLACK
        self.viz_last_move   = None
        self.viz_message     = "warming up..."
        self.latest_activations = None
        self.latest_policy   = None
        self.latest_value    = 0.0
        self.latest_player   = BLACK
        self.latest_move     = None

        self.human_board  = GomokuBoard(board_size)
        self.human_player = BLACK
        self.model_path   = MODEL_PATH
        self._load_if_possible()

    def _load_if_possible(self):
        if os.path.exists(self.model_path):
            try:
                payload = torch.load(self.model_path, map_location=DEVICE)
                self.model.load_state_dict(payload["model"])
                self.optimizer.load_state_dict(payload["optimizer"])
                self.games_played  = payload.get("games_played", 0)
                self.steps_played  = payload.get("steps_played", 0)
                self.last_result   = payload.get("last_result", "loaded")
                self.viz_message   = f"loaded model from {os.path.basename(self.model_path)}"
            except Exception as e:
                self.viz_message = f"load failed: {e}"

    def save_model(self, path: Optional[str] = None):
        if path is None:
            path = self.model_path
        payload = {
            "model":        self.model.state_dict(),
            "optimizer":    self.optimizer.state_dict(),
            "games_played": self.games_played,
            "steps_played": self.steps_played,
            "last_result":  self.last_result,
            "board_size":   self.board_size,
        }
        torch.save(payload, path)
        self.viz_message = f"saved model to {os.path.basename(path)}"

    def set_training(self, enabled: bool):
        self.training_enabled = enabled
        self.human_mode = not enabled
        if enabled:
            self.human_board.reset()
            self.viz_message = "training resumed"
        else:
            self.human_board.reset()
            self.viz_message = "training paused — human mode on"
            self._sync_human_board()

    def _sync_human_board(self):
        with self.state_lock:
            self.viz_board     = self.human_board.cells[:]
            self.viz_turn      = self.human_board.turn
            self.viz_last_move = self.human_board.last_move

    def encode_and_eval(self, board: GomokuBoard, player: int):
        x = encode_board(board, player).unsqueeze(0).to(DEVICE)   
        self.model.eval()
        with torch.no_grad(), self.model_lock:
            logits, value, acts = self.model(x)
        logits = logits.squeeze(0)
        acts_cpu = {k: v.flatten().detach().cpu() for k, v in acts.items()}
        return logits, value.item(), acts_cpu

    def select_move(self, board: GomokuBoard, player: int, training: bool = True):
        logits, value, acts = self.encode_and_eval(board, player)
        legal_moves = board.legal_moves()
        if not legal_moves:
            return None, value, acts, None, None

        own_winning = board.find_winning_moves(player)
        opp_winning = board.find_winning_moves(-player)

        # Tactical Priority Rule: Win immediately if possible; otherwise, force-block opponent threats
        forced_moves = own_winning if own_winning else opp_winning
        actual_allowed_moves = forced_moves if forced_moves else legal_moves

        epsilon = max(0.03, 0.35 * math.exp(-self.games_played / 1200.0)) if training else 0.0
        N = logits.shape[-1]
        
        legal_mask = torch.zeros(N, dtype=torch.bool)
        legal_mask[legal_moves] = True
        probs = masked_softmax(logits, legal_mask)

        # Generate action constraints based strictly on tactical necessity
        allowed_mask = torch.zeros(N, dtype=torch.bool)
        allowed_mask[actual_allowed_moves] = True

        if random.random() < epsilon and training:
            action = random.choice(actual_allowed_moves)
        else:
            allowed_probs = masked_softmax(logits, allowed_mask)
            action = int(torch.multinomial(allowed_probs, 1).item())

        return action, value, acts, probs.detach().cpu(), legal_mask

    def run_self_play_game(self):
        board      = GomokuBoard(self.board_size)
        trajectory: List[TrajectoryItem] = []
        winner     = EMPTY
        move_count = 0

        while True:
            player      = board.turn
            state_before = encode_board(board, player).detach().clone()

            opp = -player
            opp_winning = board.find_winning_moves(opp)
            own_winning = board.find_winning_moves(player)

            tactical_target = torch.zeros(self.board_size * self.board_size, dtype=torch.float32)
            forced_indices = own_winning if own_winning else opp_winning
            if forced_indices:
                tactical_target[forced_indices] = 1.0 / len(forced_indices)

            action, value, acts, probs, legal_mask = self.select_move(
                board, player, training=True
            )
            if action is None:
                break

            if opp_winning:
                self.threats_encountered += 1
                with torch.no_grad():
                    threat_prob = sum(probs[idx].item() for idx in opp_winning if idx < probs.numel())
                    self.avg_threat_prob = 0.98 * self.avg_threat_prob + 0.02 * threat_prob
                if action in opp_winning:
                    self.threats_blocked += 1
                else:
                    self.threats_missed += 1

            with self.state_lock:
                self.viz_board         = board.cells[:]
                self.viz_turn          = player
                self.viz_last_move     = action
                self.latest_activations = acts
                self.latest_policy     = probs
                self.latest_value      = float(value)
                self.latest_player     = player
                self.latest_move       = action
                self.viz_message       = "self-play: thinking"

            if not board.place(action, player):
                continue

            shaped = board.reward_shaping(action, player)

            if opp_winning:
                if action in opp_winning:
                    shaped = shaped + BLOCK_BONUS
                else:
                    shaped = shaped + MISS_BLOCK_PENALTY  

            trajectory.append(TrajectoryItem(
                state=state_before,
                action=action,
                player=player,
                shaped_reward=shaped,
                legal_mask=legal_mask,
                tactical_target=tactical_target
            ))
            move_count += 1

            winner = board.check_winner(action)
            if winner != EMPTY or board.is_full():
                break

        returns = [0.0] * len(trajectory)
        for color in (BLACK, WHITE):
            idxs = [i for i, t in enumerate(trajectory) if t.player == color]
            if not idxs:
                continue
            terminal = (1.0 if winner == color else -1.0 if winner != EMPTY else 0.0)
            G = 0.0   
            for k in range(len(idxs) - 1, -1, -1):
                i = idxs[k]
                r = trajectory[i].shaped_reward
                if k == len(idxs) - 1:
                    r = r + terminal
                G = r + GAMMA * G
                returns[i] = G

        final = [
            (trajectory[i], float(max(-RETURN_CLAMP, min(RETURN_CLAMP, returns[i]))))
            for i in range(len(trajectory))
        ]

        self.replay_buffer.extend(final)

        if len(self.replay_buffer) >= REPLAY_BATCH:
            self._train_step()

        self.games_played += 1
        self.steps_played += move_count
        self.last_result = (
            "black wins" if winner == BLACK else
            "white wins" if winner == WHITE else "draw"
        )
        with self.state_lock:
            self.viz_board     = board.cells[:]
            self.viz_turn      = board.turn
            self.viz_last_move = board.last_move
            self.viz_message   = f"game {self.games_played}: {self.last_result}"

    def _train_step(self):
        sample = random.sample(
            list(self.replay_buffer),
            min(REPLAY_BATCH, len(self.replay_buffer))
        )

        states    = torch.stack([it.state           for it, _ in sample]).to(DEVICE)
        actions   = torch.tensor([it.action         for it, _ in sample], dtype=torch.long,    device=DEVICE)
        targets   = torch.tensor([ret               for _,  ret in sample], dtype=torch.float32, device=DEVICE)
        l_masks   = torch.stack([it.legal_mask      for it, _ in sample]).to(DEVICE)  
        t_targets = torch.stack([it.tactical_target for it, _ in sample]).to(DEVICE) # (B, N)

        self.model.train()
        with self.model_lock:
            logits, values, _ = self.model(states)

            masked_logits = logits.masked_fill(~l_masks, -10.0)
            log_probs     = F.log_softmax(masked_logits, dim=-1)

            chosen_lp  = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
            chosen_lp  = chosen_lp.clamp(min=-8.0)   

            advantages = targets - values.detach()
            if advantages.numel() > 1 and advantages.std() > 1e-6:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            policy_loss = -(chosen_lp * advantages).mean()
            value_loss = F.mse_loss(values, targets)

            legal_logits  = logits.masked_fill(~l_masks, float('-inf'))
            legal_lp      = F.log_softmax(legal_logits, dim=-1)  
            legal_p       = legal_lp.exp() * l_masks.float()      
            safe_lp       = torch.where(l_masks, legal_lp, torch.zeros_like(legal_lp))
            entropy       = -(legal_p * safe_lp).sum(dim=-1).mean()

            has_tactical = t_targets.sum(dim=-1) > 0
            if has_tactical.any():
                tactical_logits = logits[has_tactical]
                tactical_l_masks = l_masks[has_tactical]
                masked_tac_logits = tactical_logits.masked_fill(~tactical_l_masks, -1e9)
                tac_log_probs = F.log_softmax(masked_tac_logits, dim=-1)
                
                tactical_loss = -(t_targets[has_tactical] * tac_log_probs).sum(dim=-1).mean()
                self.last_tactical_loss = float(tactical_loss.item())
            else:
                tactical_loss = torch.tensor(0.0, device=DEVICE)
                self.last_tactical_loss = 0.0

            ent_coeff = max(0.005, ENTROPY_COEFF * math.exp(-self.games_played / 3000.0))

            TACTICAL_COEFF = 2.5 
            loss = policy_loss + VALUE_COEFF * value_loss - ent_coeff * entropy + TACTICAL_COEFF * tactical_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP)
            self.optimizer.step()
            self.scheduler.step()

            self.last_loss = float(loss.item())
        self.model.eval()

    def training_loop(self):
        self.model.eval()
        while self.running:
            if not self.training_enabled:
                time.sleep(0.05)
                continue
            try:
                self.run_self_play_game()
            except Exception as e:
                with self.state_lock:
                    self.viz_message = f"training hiccup: {e}"
                time.sleep(0.2)

    def process_human_move(self, idx: int) -> bool:
        """Processes the human's click and updates the board. Returns True if valid."""
        if self.training_enabled or idx is None:
            return False
        if self.human_board.check_winner() != EMPTY or self.human_board.is_full():
            return False
        if self.human_board.cells[idx] != EMPTY:
            return False
        if self.human_board.turn != self.human_player:
            return False
        if not self.human_board.place(idx, self.human_player):
            return False

        self._sync_human_board()
        
        winner = self.human_board.check_winner(idx)
        if winner == self.human_player:
            self.viz_message = "human wins"
        elif self.human_board.is_full():
            self.viz_message = "draw"
        else:
            self.viz_message = "ai is thinking..."
            
        return True

    def process_ai_move(self):
        """Calculates and places the AI's move."""
        if self.training_enabled:
            return
        if self.human_board.check_winner() != EMPTY or self.human_board.is_full():
            return

        ai_player = -self.human_player
        ai_move, _, acts, probs, _ = self.select_move(
            self.human_board, ai_player, training=False
        )
        
        if ai_move is not None:
            self.human_board.place(ai_move, ai_player)
            with self.state_lock:
                self.latest_activations = acts
                self.latest_policy      = probs
                self.latest_value       = 0.0
                self.latest_player      = ai_player
                self.latest_move        = ai_move

        self._sync_human_board()
        
        winner = self.human_board.check_winner()
        if winner == ai_player:
            self.viz_message = "ai wins"
        elif self.human_board.is_full():
            self.viz_message = "draw"
        else:
            self.viz_message = "your move"


# -----------------------------
# ui
# -----------------------------
class GomokuApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("gomoku self-play ai  ·  CNN edition")
        self.root.geometry(f"{WINDOW_W}x{WINDOW_H}")
        self.root.configure(bg="#111111")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.engine = TrainingEngine(BOARD_SIZE)
        self.training_thread = threading.Thread(
            target=self.engine.training_loop, daemon=True
        )
        self.training_thread.start()

        self.main_frame = tk.Frame(root, bg="#111111")
        self.main_frame.pack(fill="both", expand=True)

        self.left = tk.Frame(self.main_frame, bg="#111111")
        self.left.pack(side="left", fill="both", expand=False, padx=12, pady=12)

        self.right = tk.Frame(self.main_frame, bg="#111111")
        self.right.pack(side="left", fill="both", expand=True, padx=12, pady=12)

        self.canvas = tk.Canvas(
            self.left, width=BOARD_PIXELS, height=BOARD_PIXELS,
            bg="#d8b06a", highlightthickness=0
        )
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        self.button_row = tk.Frame(self.left, bg="#111111")
        self.button_row.pack(fill="x", pady=(10, 0))

        self.pause_btn = tk.Button(
            self.button_row, text="pause training",
            command=self.toggle_training, width=14
        )
        self.pause_btn.pack(side="left", padx=4)

        self.save_btn = tk.Button(
            self.button_row, text="save model",
            command=self.save_model, width=12
        )
        self.save_btn.pack(side="left", padx=4)

        self.load_btn = tk.Button(
            self.button_row, text="load model",
            command=self.load_model, width=12
        )
        self.load_btn.pack(side="left", padx=4)

        self.reset_btn = tk.Button(
            self.button_row, text="reset board",
            command=self.reset_board, width=12
        )
        self.reset_btn.pack(side="left", padx=4)

        self.info = tk.Label(
            self.left, text="", bg="#111111", fg="#eeeeee",
            justify="left", font=("Consolas", 11), anchor="w",
        )
        self.info.pack(fill="x", pady=(12, 0))

        self.nn_canvas = tk.Canvas(self.right, bg="#121212", highlightthickness=0)
        self.nn_canvas.pack(fill="both", expand=True)
        self.nn_canvas.bind("<Configure>", lambda e: self.redraw())

        self.redraw()
        self.root.after(40, self.refresh_loop)

    def toggle_training(self):
        self.engine.set_training(not self.engine.training_enabled)
        self.pause_btn.configure(
            text="resume training" if not self.engine.training_enabled else "pause training"
        )
        self.redraw()

    def save_model(self):
        try:
            self.engine.save_model()
            messagebox.showinfo("saved", f"saved to\n{MODEL_PATH}")
        except Exception as e:
            messagebox.showerror("save failed", str(e))

    def load_model(self):
        try:
            path = filedialog.askopenfilename(
                title="load gomoku model",
                filetypes=[("pytorch model", "*.pt"), ("all files", "*.*")],
            )
            if not path:
                return
            payload = torch.load(path, map_location=DEVICE)
            with self.engine.model_lock:
                self.engine.model.load_state_dict(payload["model"])
                if "optimizer" in payload:
                    self.engine.optimizer.load_state_dict(payload["optimizer"])
            self.engine.viz_message = f"loaded {os.path.basename(path)}"
            self.redraw()
        except Exception as e:
            messagebox.showerror("load failed", str(e))

    def reset_board(self):
        if self.engine.training_enabled:
            with self.engine.state_lock:
                self.engine.viz_board     = [EMPTY] * (BOARD_SIZE * BOARD_SIZE)
                self.engine.viz_turn      = BLACK
                self.engine.viz_last_move = None
                self.engine.viz_message   = "training board reset"
        else:
            self.engine.human_board.reset()
            self.engine._sync_human_board()
            self.engine.viz_message = "human board reset"
        self.redraw()

    def board_index_from_xy(self, x: int, y: int) -> Optional[int]:
        gx = round((x - MARGIN) / CELL)
        gy = round((y - MARGIN) / CELL)
        if gx < 0 or gy < 0 or gx >= BOARD_SIZE or gy >= BOARD_SIZE:
            return None
        px = MARGIN + gx * CELL
        py = MARGIN + gy * CELL
        if abs(x - px) > CELL / 2 or abs(y - py) > CELL / 2:
            return None
        return gy * BOARD_SIZE + gx

    def on_canvas_click(self, event):
        idx = self.board_index_from_xy(event.x, event.y)
        if idx is None:
            return
        
        # Process the human's move
        if self.engine.process_human_move(idx):
            self.redraw()
            # If the human didn't win and the board isn't full, schedule the AI's move
            winner = self.engine.human_board.check_winner()
            if winner == EMPTY and not self.engine.human_board.is_full():
                # Introduce a 600ms delay for realism before the AI places its move
                self.root.after(600, self.do_ai_move)

    def do_ai_move(self):
        self.engine.process_ai_move()
        self.redraw()

    def draw_board(self):
        self.canvas.delete("all")
        size = BOARD_SIZE
        for i in range(size):
            x = MARGIN + i * CELL
            self.canvas.create_line(MARGIN, x, BOARD_PIXELS - MARGIN, x, fill="#7f5d2f", width=2)
            self.canvas.create_line(x, MARGIN, x, BOARD_PIXELS - MARGIN, fill="#7f5d2f", width=2)

        if size >= 9:
            stars = [2, size // 2, size - 3]
            for r in stars:
                for c in stars:
                    x = MARGIN + c * CELL
                    y = MARGIN + r * CELL
                    self.canvas.create_oval(x - 4, y - 4, x + 4, y + 4, fill="#6b4c24", outline="")

        with self.engine.state_lock:
            board = self.engine.viz_board[:]
            last  = self.engine.viz_last_move
            turn  = self.engine.viz_turn
            msg   = self.engine.viz_message

        for idx, v in enumerate(board):
            if v == EMPTY:
                continue
            r, c = divmod(idx, BOARD_SIZE)
            x = MARGIN + c * CELL
            y = MARGIN + r * CELL
            color = "#111111" if v == BLACK else "#f3f3f3"
            self.canvas.create_oval(x - 20, y - 20, x + 20, y + 20,
                                    fill=color, outline="#000000", width=2)
        if last is not None:
            r, c = divmod(last, BOARD_SIZE)
            x = MARGIN + c * CELL
            y = MARGIN + r * CELL
            self.canvas.create_rectangle(x - 24, y - 24, x + 24, y + 24,
                                         outline="#ff4f6d", width=2)

        turn_text = "black to move" if turn == BLACK else "white to move"
        mode_text = "training" if self.engine.training_enabled else "human mode"
        self.canvas.create_text(12, 12, anchor="nw", text=mode_text,
                                fill="#3b2b12", font=("Consolas", 12, "bold"))
        self.canvas.create_text(12, BOARD_PIXELS - 16, anchor="sw", text=turn_text,
                                fill="#3b2b12", font=("Consolas", 12, "bold"))
        
        self.info.configure(text=(
            f"games: {self.engine.games_played} | steps: {self.engine.steps_played}\n"
            f"loss total: {self.engine.last_loss:.4f} | loss tac: {self.engine.last_tactical_loss:.4f}\n"
            f"threats seen: {self.engine.threats_encountered} | blocks made: {self.engine.threats_blocked}\n"
            f"unmasked network tactical accuracy: {self.engine.avg_threat_prob * 100:.1f}%\n"
            f"result: {self.engine.last_result} | status: {msg}"
        ))

    def draw_nn(self):
        self.nn_canvas.delete("all")
        w = max(500, self.nn_canvas.winfo_width())
        h = max(400, self.nn_canvas.winfo_height())

        layer_specs = [
            ("feat map", 24, "a1"),
            ("policy",   24, "a2"),
            ("value h",  18, "a3"),
        ]

        with self.engine.state_lock:
            acts        = self.engine.latest_activations
            policy      = self.engine.latest_policy
            value       = self.engine.latest_value
            latest_move = self.engine.latest_move

        self.nn_canvas.create_text(
            20, 16, anchor="nw", text="brain glow panel  [CNN residual tower]",
            fill="#f0f0f0", font=("Consolas", 16, "bold")
        )
        self.nn_canvas.create_text(
            20, 40, anchor="nw",
            text=f"value: {value:+.3f}   last move: {latest_move if latest_move is not None else '-'}",
            fill="#cfcfcf", font=("Consolas", 11)
        )

        if acts is None:
            self.nn_canvas.create_text(
                20, 80, anchor="nw", text="waiting for a forward pass…",
                fill="#a0a0a0", font=("Consolas", 12)
            )
            return

        layer_xs = [75, 235, 395]

        for li, (label, count, key) in enumerate(layer_specs):
            x   = layer_xs[li]
            vec = acts.get(key, torch.zeros(count)).flatten().float()
            self.nn_canvas.create_text(
                x, 78, text=label, fill="#f0f0f0", font=("Consolas", 12, "bold")
            )
            use    = min(count, vec.numel())
            sample = vec[:use]
            mn, mx = float(sample.min()), float(sample.max())
            denom  = (mx - mn) if (mx - mn) > 1e-6 else 1.0
            for j in range(use):
                y  = 110 + j * min(20, (h - 180) / max(use, 1))
                t  = (float(sample[j]) - mn) / denom
                fill = self._heat_color(t)
                self.nn_canvas.create_oval(
                    x - 16, y - 16, x + 16, y + 16,
                    fill=fill, outline="#2d2d2d", width=1
                )
                self.nn_canvas.create_text(
                    x + 28, y, anchor="w",
                    text=f"{float(sample[j]):+.2f}",
                    fill="#bbbbbb", font=("Consolas", 8)
                )

        px0 = 560
        self.nn_canvas.create_text(
            px0, 78, text="policy map", fill="#f0f0f0", font=("Consolas", 12, "bold")
        )
        if policy is not None:
            probs = policy.flatten().float()
            pmin, pmax = float(probs.min()), float(probs.max())
            denom = (pmax - pmin) if (pmax - pmin) > 1e-8 else 1.0
            cell  = 34
            for r in range(BOARD_SIZE):
                for c in range(BOARD_SIZE):
                    idx = r * BOARD_SIZE + c
                    p   = float(probs[idx]) if idx < probs.numel() else 0.0
                    t   = (p - pmin) / denom
                    fill = self._heat_color(t)
                    x1, y1 = px0 + c * cell, 110 + r * cell
                    self.nn_canvas.create_rectangle(
                        x1, y1, x1 + cell - 4, y1 + cell - 4,
                        fill=fill, outline="#3a3a3a"
                    )
                    self.nn_canvas.create_text(
                        x1 + 14, y1 + 14, text=f"{p:.2f}",
                        fill="#101010", font=("Consolas", 7)
                    )
            if latest_move is not None:
                r, c = divmod(latest_move, BOARD_SIZE)
                self.nn_canvas.create_rectangle(
                    px0 + c * cell - 1, 110 + r * cell - 1,
                    px0 + c * cell + cell - 3, 110 + r * cell + cell - 3,
                    outline="#ff4f6d", width=3,
                )

        self.nn_canvas.create_text(
            20, h - 34, anchor="sw",
            text="bright = higher activation or policy preference",
            fill="#bfbfbf", font=("Consolas", 10)
        )

    def _heat_color(self, t: float) -> str:
        t = max(0.0, min(1.0, float(t)))
        r = int(40 + 215 * t)
        g = int(40 + 120 * t)
        b = int(60 + 40  * (1 - t))
        return f"#{r:02x}{g:02x}{b:02x}"

    def redraw(self):
        self.draw_board()
        self.draw_nn()

    def refresh_loop(self):
        self.pause_btn.configure(
            text="pause training" if self.engine.training_enabled else "resume training"
        )
        self.redraw()
        self.root.after(60, self.refresh_loop)

    def on_close(self):
        try:
            self.engine.running = False
            self.engine.save_model()
        except Exception:
            pass
        self.root.destroy()


# -----------------------------
# main
# -----------------------------
def main():
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    root = tk.Tk()
    app  = GomokuApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
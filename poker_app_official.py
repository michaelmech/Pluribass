from __future__ import annotations
import functools, streamlit as st, numpy as np
from types import MethodType
import streamlit as st
import lightgbm as lgb
import time

import torch
from mamba_data_gen import load_poker_mamba, infer_single_step, extract_pluribus_actions_mamba,load_poker_transformer

from data_gen import (extract_pluribus_actions, encode_card, evaluate_hand_category, detect_draws, get_seat_tag, get_position_for_pluribus, get_relative_position, determine_phase, parse_phh_file, parse_phh_string, process_phh_folder, process_phh_zip, print_phh_files_from_zip, get_pluribus_hand_profit, augment_swap_hole_cards, augment_permute_flop)

# --- card-image helper ---------------------------------------------------
CARD_DIR = "card_imgs"          # 52 PNGs named like As.png, Td.png, 9h.png …
@functools.lru_cache
def card_img(card: str) -> str:  # "Kd"  →  "card_imgs/Kd.png"
    return f"{CARD_DIR}/{card}.png"


from typing import List, Tuple, Dict, Any, Optional
import pandas as pd
import lightgbm as lgb
from datetime import datetime

MAMBA_CKPT = "augment_poker_transformer.pt"  # your saved file
device = torch.device("cpu")  # per your preference
#model, normalizer, feature_keys, _ = load_poker_mamba(MAMBA_CKPT, device=device)
model, normalizer, _, _ = load_poker_transformer(MAMBA_CKPT, device=device)

class PickleableBooster:
    """A thin, sklearn-style wrapper around a LightGBM Booster."""
    def __init__(self, booster: lgb.Booster, num_class: int):
        self._booster   = booster
        self.num_class  = num_class

    # --- sklearn-like API -------------------------------------------------
    def predict_proba(self, X):
        raw = self._booster.predict(X)      # already shape (N, C)
        return np.clip(raw, 0, 1)

    # --- make the object pickle-safe -------------------------------------
    def __getstate__(self):
        return {
            "model_str": self._booster.model_to_string(),
            "num_class": self.num_class,
        }

    def __setstate__(self, state):
        self._booster  = lgb.Booster(model_str=state["model_str"])
        self.num_class = state["num_class"]


from functools import wraps

def with_temperature_by_elims(
    *,
    start_players: int = 6,
    temp_at_start: float = 0.3,
    temp_at_end: float = 1.0,     # when 1 player remains (or heads-up if you prefer)
    end_players: int = 2,         # set to 1 if you truly want it to ramp until winner
):
    """
    Linearly ramps temperature as players are eliminated.

    players_remaining = len(gs.get_live_players())
    fraction = (start_players - players_remaining) / (start_players - end_players)
    temp = lerp(temp_at_start, temp_at_end, clamp(fraction))
    """
    def decorator(cls):
        orig_get_action = cls.get_action

        @wraps(orig_get_action)
        def wrapped(self, gs, seat_idx):
            if hasattr(gs, "get_live_players"):
                alive = len(gs.get_live_players())
            else:
                # fallback: count non-folded
                alive = sum(1 for p in gs.players if not p.folded)

            denom = max(1, (start_players - end_players))
            frac = (start_players - alive) / denom
            frac = max(0.0, min(1.0, frac))

            self.temperature = temp_at_start + frac * (temp_at_end - temp_at_start)

            return orig_get_action(self, gs, seat_idx)

        cls.get_action = wrapped
        return cls

    return decorator


def with_commitment_rule(commit_frac=0.6):
    def decorator(cls):
        orig_get_action = cls.get_action

        def wrapped(self, gs, seat_idx):
            hero = gs.players[seat_idx]

            # already all-in or can't act
            if hero.stack <= 0:
                return orig_get_action(self, gs, seat_idx)

            # amount to call
            to_call = max(gs.current_bet - hero.bet_this_street, 0)
            if to_call <= 0:
                return orig_get_action(self, gs, seat_idx)

            # stack at start of street (store this once per street if you don't already)
            stack_start = getattr(hero, "stack_start_street", hero.stack + hero.bet_this_street)

            committed = (hero.bet_this_street + to_call) / max(1, stack_start)

            if committed >= commit_frac:
                # call or shove if call would be all-in anyway
                if to_call >= hero.stack:
                    return "call", hero.stack
                return "call", to_call

            return orig_get_action(self, gs, seat_idx)

        cls.get_action = wrapped
        return cls
    return decorator



from functools import wraps

# --- Preflop shove decorator (6-max-ish, < ~20BB) --------------------------------
#
# Uses GameState.get_position_label(seat_idx) if available (BTN/SB/BB/UTG/MP/CO).
# Shoves only when NOT facing a raise (i.e., unopened / limped pot).
#
# If it triggers, it returns ("raise", hero.bet_this_street + hero.stack) which is
# a true all-in in your engine (raise-to-total semantics).
#
# Tune ranges + shove_bb to taste.

_RANK = {**{str(i): i for i in range(2, 10)}, "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14}

from functools import wraps

# --- Rank Parsing Helper ----------------------------------------------------
_RANK = {**{str(i): i for i in range(2, 10)}, "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14}

def parse_hole_cards(hole):
    """Returns hi, lo, suited, is_pair"""
    if len(hole) != 2 or any(len(c) < 2 for c in hole):
        return None
    c1, c2 = hole
    r1, s1 = _RANK.get(c1[0].upper(), 0), c1[1].lower()
    r2, s2 = _RANK.get(c2[0].upper(), 0), c2[1].lower()
    hi, lo = (r1, r2) if r1 >= r2 else (r2, r1)
    return hi, lo, (s1 == s2), (r1 == r2)

# --- NASH EQUILIBRIUM LOGIC -------------------------------------------------

def is_nash_push(pos, hi, lo, suited, is_pair, eff_bb, facing_raise=False):
    """
    Simplified Nash Equilibrium logic for < 10-12 BBs.
    Returns True if this is a profitable shove.
    """
    # 1. SUPER SHORT (< 5 BB): Push almost any broadway, any pair, any ace
    if eff_bb < 5:
        if is_pair: return True
        if hi >= 13: return True # Kx+
        if hi == 12 and suited: return True # Qxs
        if hi == 11 and suited: return True # Jxs

    # 2. DANGER ZONE (5 - 12 BB)
    # This specifically addresses your scenario: AJs in BB vs SB raise.

    # Always shove these premiums regardless of action (Raise or No Raise)
    if is_pair and hi >= 7: return True  # 77+
    if hi == 14 and lo >= 10: return True # AT+ (includes AJs)
    if hi == 14 and suited and lo >= 8: return True # A8s+
    if hi == 13 and lo >= 12 and suited: return True # KQs

    # If we are NOT facing a raise (Open Shove), we can be much wider
    if not facing_raise:
        if is_pair: return True # 22+
        if hi == 14: return True # Any Ax
        if hi >= 13 and suited: return True # Kxs
        if hi == 13 and lo >= 10: return True # KTo+
        if hi >= 11 and suited and lo >= 9: return True # J9s+
        # Add more position-specific open shoves here (e.g., BTN is very wide)

    # If we ARE facing a raise (Re-Shove / 3-bet Shove)
    # We need to be tighter than an open shove, but AJs is still an easy shove.
    else:
        # Re-shove ranges (Simplified)
        if is_pair and hi >= 8: return True # 88+
        if hi == 14 and lo >= 11: return True # AJo+ (includes AJs)
        if hi == 14 and suited and lo >= 10: return True # ATs+
        if hi == 13 and suited and lo >= 12: return True # KQs

    return False

# --- THE DECORATOR ----------------------------------------------------------

def with_short_stack_nash_equilibrium(shove_bb_threshold=12.0):
    """
    Forces optimal Push/Fold strategy when stack is effectively short.
    Handles BOTH Open Shoves and Re-Shoves (Defense).
    """
    def decorator(cls):
        orig_get_action = cls.get_action

        @wraps(orig_get_action)
        def wrapped(self, gs, seat_idx):
            # 1. Filter: Preflop only
            if getattr(gs, "street", "").lower() != "preflop":
                return orig_get_action(self, gs, seat_idx)

            hero = gs.players[seat_idx]

            # 2. Filter: Valid Hole Cards
            hole_data = parse_hole_cards(getattr(hero, "hole", []))
            if not hole_data:
                return orig_get_action(self, gs, seat_idx)
            hi, lo, suited, is_pair = hole_data

            # 3. Filter: Effective Stack Depth
            bb_amt = float(getattr(gs, "bb_amount", 0) or 0)
            if bb_amt <= 0: return orig_get_action(self, gs, seat_idx)

            eff_bb = float(hero.stack) / bb_amt

            # If we are deep, or already all-in, skip this logic
            if eff_bb > shove_bb_threshold or hero.stack <= 0 or getattr(hero, "is_all_in", False):
                return orig_get_action(self, gs, seat_idx)

            # 4. Context: Are we facing a raise?
            # If current bet > BB, someone raised.
            to_call = max((gs.current_bet - hero.bet_this_street), 0)
            facing_raise = (gs.current_bet > bb_amt) and (to_call > 0)

            # 5. Position Logic (Optional but recommended)
            if hasattr(gs, "get_position_label"):
                pos = gs.get_position_label(seat_idx)
            else:
                pos = "Unknown"

            # 6. DECISION TIME: Check Nash Charts
            should_shove = is_nash_push(pos, hi, lo, suited, is_pair, eff_bb, facing_raise)

            if should_shove:
                # IMPORTANT: In tournament ICM, if we commit >50% of stack, we just shove.
                # This returns a RAISE to the full stack amount (All-In).
                return "raise", hero.bet_this_street + hero.stack

            # If Nash says FOLD (and we are facing a raise), we should probably just Fold
            # unless the original bot has some specific read.
            # However, for safety, if Nash doesn't say shove, we let the original bot decide
            # (which might find a call, or fold).
            return orig_get_action(self, gs, seat_idx)

        cls.get_action = wrapped
        return cls

    return decorator




def _wrap_model(model, name):
    """
    Replace model.predict / predict_proba with versions that log output
    but keep the original signature (*args, **kwargs).
    """
    for meth in ("predict", "predict_proba"):
        if not hasattr(model, meth):
            continue

        orig_fn = getattr(model, meth)

        @functools.wraps(orig_fn)
        def _logged(self, *args, _orig=orig_fn, _meth=meth, **kwargs):
            out = _orig(*args, **kwargs)

            # show 1-row outputs nicely; skip huge arrays
            try:
                arr = np.asarray(out).reshape(-1)
                if arr.size <= 10:            # avoid spamming
                    st.sidebar.write({f"{name}.{_meth}": arr.round(4).tolist()})
                    print(f"{name}.{_meth} → {arr}")
            except Exception:
                pass

            return out

        setattr(model, meth, MethodType(_logged, model))

    return model

# poker_app.py
import streamlit as st
import pandas as pd
import numpy as np
import random
import pickle
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any
from itertools import combinations, permutations
import sklearn
import json

def summarize_moves(hist: List[str]) -> str:
    """Return a short per‑street summary of the hand history."""
    if not hist:
        return "(no actions yet)"
    summaries, current, bucket = [], "Preflop", []
    for line in hist:
        if line.startswith("---"):
            summaries.append(f"{current}: " + ", ".join(bucket))
            current = line.split()[1].capitalize()  # FLOP/TURN/RIVER
            bucket = []
        else:
            bucket.append(line)
    summaries.append(f"{current}: " + ", ".join(bucket))
    return "\n".join(summaries[-4:])  # last 4 streets at most


def sample_from_mapper(col: str, value_to_map: str) -> str:
    """Return a mapper-compatible token, defaulting to the sentinel 'none'."""
    if col not in mapper:
        st.error(f"Error: Column '{col}' not found in mapper – using 'none'.")
        return "none"

    if value_to_map not in mapper[col]:
        st.warning(f"Warning: Value '{value_to_map}' for '{col}' unseen – using 'none'.")
        return "none"                     # <- **no random choice**

    return value_to_map

# --- load categorical mapper learnt during training ------------------------
with open("mapping.json", "r") as f:
    mapper: Dict[str, Dict[str, int]] = json.load(f)

mapper['last_aggressor_position']['none']=-1
mapper['last_aggressor_position'][None]=-1
mapper['last_aggressor_position']['None']=-1

with open("clipper_dct.json", "r") as f:
    clipper_dct: Dict[str, Dict[str, int]] = json.load(f)


mapper['board_best_hand']={'high_card': 0,
 'pair': 1,
 'trips': 2,
 'two_pair': 3,
 'flush': 4,
 'straight': 5,
  'full_house': 6,
  'four_kind':7}



with open("master_cols.json", "r") as f:
    MASTER_COLS = json.load(f)

# --- Utility helpers -------------------------------------------------------
RANKS = "23456789TJQKA"
SUITS = "cdhs"
RANK_MAP = {r: i + 2 for i, r in enumerate(RANKS)}
SUIT_MAP = {s: i for i, s in enumerate(SUITS)}
HAND_CATEGORIES = [
    "High Card",
    "Pair",
    "Two Pair",
    "Three of a Kind",
    "Straight",
    "Flush",
    "Full House",
    "Four of a Kind",
    "Straight Flush",
]
CATEGORY_ORDER = [
    "high_card",
    "pair",
    "two_pair",
    "trips",
    "straight",
    "flush",
    "full_house",
    "four_kind",
    "straight_flush",
    "royal_flush",
]


# ---------------------------------------------------------------------------
#                      Core game‑engine primitives
# ---------------------------------------------------------------------------

def shuffled_deck() -> List[str]:
    deck = [r + s for r in RANKS for s in SUITS]
    random.shuffle(deck)
    return deck



def _is_straight(unique_ranks: List[int]) -> int | None:
    ranks = unique_ranks[:]
    if 14 in ranks:
        ranks.insert(0, 1)
    for i in range(len(ranks) - 4):
        if ranks[i + 4] - ranks[i] == 4:
            if set(ranks[i : i + 5]) == {14, 2, 3, 4, 5}:  # wheel
                return 5
            return ranks[i + 4]
    return None

_SUITS = [0, 1, 2, 3]
_RANKS = list(range(2, 15))

def _blank_card_one_hot(prefix: str) -> Dict[str, int]:
    out = {}
    for r in _RANKS:
        out[f"{prefix}_rank_{r}"] = 0
    for s in _SUITS:
        out[f"{prefix}_suit_{s}"] = 0
    return out

def _card_one_hot(card: str, prefix: str) -> Dict[str, int]:
    rk, st = encode_card(card)
    out = {}
    for r in _RANKS:
        out[f"{prefix}_rank_{r}"] = int(rk == r)
    for s in _SUITS:
        out[f"{prefix}_suit_{s}"] = int(st == s)
    return out


def get_best_hand(all_cards: List[str]) -> Tuple[int, List[int], List[str]]:
    """Finds the best 5-card hand from a list of cards."""
    best_rank_val = -1
    best_kickers = []
    best_combo = []
    for combo in combinations(all_cards, 5):
        combo_list = list(combo)
        hand_val, kickers = _get_hand_rank(combo_list)
        if hand_val > best_rank_val or (hand_val == best_rank_val and kickers > best_kickers):
            best_rank_val = hand_val
            best_kickers = kickers
            best_combo = combo_list
    return best_rank_val, best_kickers, best_combo

def _get_hand_rank(hand: List[str]) -> Tuple[int, List[int]]:
    """Evaluates a 5-card hand and returns its rank and tie-breaking kickers."""
    if not hand or len(hand) != 5:
        return 0, []
    ranks = sorted([encode_card(c)[0] for c in hand], reverse=True)
    suits = [encode_card(c)[1] for c in hand]
    is_flush = len(set(suits)) == 1
    unique_ranks = sorted(list(set(ranks)), reverse=True)
    is_straight = (len(unique_ranks) >= 5 and (unique_ranks[0] - unique_ranks[4] == 4)) or \
                  (set(unique_ranks) == {14, 5, 4, 3, 2})

    if is_straight and is_flush: return 8, unique_ranks[:5]
    rank_counts = {r: ranks.count(r) for r in ranks}
    counts = sorted(rank_counts.values(), reverse=True)
    if counts[0] == 4:
        quad_rank = [r for r, c in rank_counts.items() if c == 4][0]
        return 7, [quad_rank, [r for r in ranks if r != quad_rank][0]]
    if counts == [3, 2]:
        return 6, [[r for r, c in rank_counts.items() if c == 3][0], [r for r, c in rank_counts.items() if c == 2][0]]
    if is_flush: return 5, ranks
    if is_straight: return 4, unique_ranks[:5]
    if counts[0] == 3:
      trips_rank = [r for r, c in rank_counts.items() if c == 3][0]
      kickers = sorted([r for r in ranks if r != trips_rank], reverse=True)[:2]
      return 3, [trips_rank] + kickers

    if counts == [2, 2, 1]:
        pair_ranks = sorted([r for r, c in rank_counts.items() if c == 2], reverse=True)
        return 2, pair_ranks + [[r for r, c in rank_counts.items() if c == 1][0]]
    if counts[0] == 2:
        pair_rank = [r for r, c in rank_counts.items() if c == 2][0]
        return 1, [pair_rank] + sorted([r for r in ranks if r != pair_rank], reverse=True)
    return 0, ranks

@dataclass
class Player:
    name: str
    stack: int
    is_bot: bool
    hole: List[str] = field(default_factory=list)
    folded: bool = False
    is_all_in: bool = False
    bet_this_street: int = 0
    contributed: int = 0          # NEW – lifetime chips in this hand

@dataclass
class Pot:
    amount: int = 0
    eligible_players: List[int] = field(default_factory=list)

def _parse_hand_line(ln: str, players: list[Player]) -> tuple[str, int, str]:
    """
    Returns (action_type_tag, amount, position_tag)

    Examples
    --------
    "Bot 4 raises to $215"   -> ('cbr', 215, 'p5')
    "Hero folds"             -> ('f',   0,   'p1')
    "Bot 2 calls $40"        -> ('cc',  40,  'p3')
    """
    words = ln.split()
    if not words:
        return "none", 0, "none"

    # 1) map the player name to seat tag
    player_name = " ".join(words[:2]) if words[0] == "Bot" else words[0]
    idx = next((i for i, p in enumerate(players) if p.name == player_name), None)
    pos_tag = f"p{idx+1}" if idx is not None else "none"

    # 2) action word → compact code
    action_word = next((w for w in words if w in {"calls", "raises", "folds",
                                                  "checks", "bets"}), "none")
    action_map = {"folds": "f", "checks": "cc", "calls": "cc",
                  "bets": "cbr", "raises": "cbr", "none": "none"}
    act_tag = action_map[action_word]

    # 3) numeric amount (if any)
    amt = 0
    for w in words:
        if w.startswith("$"):
            amt = int(w.replace("$", ""))
            break

    return act_tag, amt, pos_tag

from typing import List

class GameStatePatchMixin:
    """Mixin that patches GameState with robust auto‑advance logic.

    Add this as a mixin before your existing GameState base, e.g.::

        class GameState(GameStatePatchMixin, OriginalGameState):
            pass
    """

    # ────────────────────────────────────────────────────────────────
    # Helper ─ Least‑side‑pot betting detection
    # ────────────────────────────────────────────────────────────────

    def _deal_out_if_side_pot_closed(self) -> bool:
        """Auto‑deal remaining streets when < 2 players can add chips.

        Returns ``True`` if we advanced at least one street (or hit
        showdown), so the caller can bail out of its current loop.
        """
        active_not_all_in = [p for p in self.players if not p.folded and not p.is_all_in]
        if len(active_not_all_in) < 2:
            if self.street != "river":
                self.advance_street()       # recurse to deal flop/turn/river
            else:
                self.street = "showdown"    # already on river → showdown
            return True
        return False

    # GameStatePatchMixin  (or anywhere in GameState)
    def _betting_round_closed(self) -> bool:
        """True when every live player has matched the bet and we’ve
          come full circle to the aggressor."""
        live = [p for p in self.players if not p.folded]
        # everyone has matched the current bet (incl. 0 when all checked)
        if any(not p.is_all_in and p.bet_this_street != self.current_bet for p in live):
            return False
        return self.next_to_act_pos == self.aggressor_pos


class GameStateOriginal:
    def __init__(self, hero_name: str = "Hero", stack: int = 10000, bb_amount: int = 100,hands_per_level=7,reveal_bot_hole_cards=True,bot_names: list[str] | None=None):
        self.bb_amount = bb_amount
        self.original_bb_amount = bb_amount
        self.hands_per_level = hands_per_level    # ↑ every N hands
        self.hands_played    = 0                  # count completed hands

        self.reveal_bot_hole_cards = reveal_bot_hole_cards

        self.players = [Player(hero_name, stack, is_bot=False)] + [
            Player(f"Bot {i}", stack, is_bot=True) for i in range(1, 6)
        ]
        self.button_pos = -1
        self.pots: List[Pot] = []
        self.board: List[str] = []
        self.street: str = "preflop"
        self.deck = shuffled_deck()
        self.current_bet = 0
        self.aggressor_pos = -1  # last player to put in a raise on current street
        self.next_to_act_pos = 0
        self.hand_history: List[str] = []
        self.last_actions: Dict[int, Tuple[str, int]] = {
            i: ("none", 0) for i in range(len(self.players))
        }
        self.max_bet_seen = 0
        self.first_aggressor = "none"
        self.last_aggressor_position = "none"  # legacy – kept for UI text
        self.last_raise_increment = self.bb_amount  # Add this line



        if bot_names is None:                          # keep old behaviour
            bot_names = [f"Bot {i}" for i in range(1, 6)]

        self.players = [Player(hero_name, stack, is_bot=False)] + [
            Player(name, stack, is_bot=True) for name in bot_names
        ]

        self.bot_names = bot_names


    def reset_game(self, hero_stack: int = 10000, bot_stack: int = 10000):
        """
        Re-create all seats with fresh stacks and immediately deal the first hand.
        """
        hero_name = self.players[0].name          # keep whatever name the user typed
        self.players = [Player(hero_name, hero_stack, is_bot=False)] + [
        Player(name, bot_stack, is_bot=True) for name in self.bot_names
            ]

        # reset bookkeeping that depends on player count
        self.button_pos = -1
        self.last_actions = {i: ("none", 0) for i in range(len(self.players))}
        self.bb_amount = self.original_bb_amount
        # deal the new hand
        self.start_new_hand()
        self.bb_option_done = False

    def start_new_hand(self):
        """Resets the game state for a new hand."""
        self.players = [self.players[0]] + [p for p in self.players[1:] if p.stack > 0]
        self.starting_stacks = [p.stack for p in self.players]
        self.last_raise_increment = self.bb_amount  # ← new default

        # rebuild helper maps to match the new player count
        self.last_actions = {i: ("none", 0) for i in range(len(self.players))}

        if len(self.players) <= 1:
            # hero + nobody else
            self.reset_game()
            # new stacks, new bots, first hand dealt
            return

        self.hands_played += 1
        if self.hands_played % self.hands_per_level == 0:
            self.bb_amount = int(self.bb_amount * 1.5)
            self.min_bet = self.bb_amount
            # st.info(f"Blind level ↑  to {self.bb_amount} / {self.bb_amount//2}")


        self.board = []
        self.street = "preflop"
        self.deck = shuffled_deck()
        self.pots = [Pot(amount=0, eligible_players=list(range(len(self.players))))]
        self.hand_history = []
        self.last_actions = {i: ("none", 0) for i in range(len(self.players))}
        self.max_bet_seen = 0
        self.first_aggressor = "none"
        self.last_aggressor_position = "none"

        for p in self.players:
            p.hole = []
            p.folded = False
            p.is_all_in = False
            p.bet_this_street = 0
            p.contributed = 0

        self.button_pos = (self.button_pos + 1) % len(self.players)
        for p in self.players:
            p.hole = [self.deck.pop(), self.deck.pop()]

        sb_pos = (self.button_pos + 1) % len(self.players)
        bb_pos = (self.button_pos + 2) % len(self.players)

        sb_amount = min(self.players[sb_pos].stack, self.bb_amount // 2)
        self.players[sb_pos].stack -= sb_amount
        self.players[sb_pos].bet_this_street = sb_amount
        self.players[sb_pos].contributed += sb_amount

        bb_amount_actual = min(self.players[bb_pos].stack, self.bb_amount)
        self.players[bb_pos].stack -= bb_amount_actual
        self.players[bb_pos].bet_this_street = bb_amount_actual
        self.players[bb_pos].contributed += bb_amount_actual

        if self.players[sb_pos].stack == 0:
            self.players[sb_pos].is_all_in = True

        if self.players[bb_pos].stack == 0:
            self.players[bb_pos].is_all_in = True

        self.pots[0].amount = sb_amount + bb_amount_actual
        self.current_bet = bb_amount_actual
        self.max_bet_seen = bb_amount_actual
        self.next_to_act_pos = (bb_pos + 1) % len(self.players)
        self.aggressor_pos = bb_pos
        self.first_aggressor = self.players[bb_pos].name
        self.last_aggressor_position = self.get_position_label(bb_pos)

        self.hand_history.append(f"{self.players[sb_pos].name} posts SB ${sb_amount}")
        self.hand_history.append(f"{self.players[bb_pos].name} posts BB ${bb_amount_actual}")

    def get_position_label(self, player_idx: int) -> str:
        """Gets the position label (e.g., BTN, SB, BB) for a player."""
        pos_map = {
            0: "BTN", 1: "SB", 2: "BB", 3: "UTG", 4: "MP", 5: "CO"
        }
        # Relative to button
        relative_pos = (player_idx - self.button_pos + len(self.players)) % len(self.players)
        return pos_map.get(relative_pos, "Unknown")

    def reset_round(self):
      self.start_new_hand()


    def get_live_players(self) -> List[Player]:
        return [p for p in self.players if not p.folded]

    def advance_street(self):
      """Moves the game to the next street."""
      if self.street == "preflop":
          self.street = "flop"
          self.board.extend([self.deck.pop() for _ in range(3)])
      elif self.street == "flop":
          self.street = "turn"
          self.board.append(self.deck.pop())
      elif self.street == "turn":
          self.street = "river"
          self.board.append(self.deck.pop())
      elif self.street == "river":
          self.street = "showdown"
          return

      self.hand_history.append(
          f"--- {self.street.upper()} --- Board: {' '.join(self.board)}"
      )
      self.current_bet = 0
      self.aggressor_pos = -1
      self.last_raise_increment = self.bb_amount  # Add this line
      for p in self.players:
          p.bet_this_street = 0

      self.next_to_act_pos = (self.button_pos + 1) % len(self.players)

      # Skip folded **and all-in** players
      hops = 0
      while (
          self.players[self.next_to_act_pos].folded
          or self.players[self.next_to_act_pos].is_all_in
      ) and hops < len(self.players):
          self.next_to_act_pos = (self.next_to_act_pos + 1) % len(self.players)
          hops += 1

      # If we looped the entire table, everyone is all-in → auto-advance
      if hops == len(self.players):
          if self.street != "river":
              self.advance_street()      # recurse to flop/turn/river
          else:
              self.street = "showdown"   # river dealt → go to showdown
          return

      # 🔹 NEW extra guard:
      # If < 2 players can still add chips, deal out the board immediately
      active_not_all_in = [
          p for p in self.players if not p.folded and not p.is_all_in
      ]
      if len(active_not_all_in) < 2:
          if self.street != "river":
              self.advance_street()      # burn the next street
          else:
              self.street = "showdown"   # already at river → showdown
          return

      self.aggressor_pos = self.next_to_act_pos


class GameState(GameStatePatchMixin, GameStateOriginal):
    def advance_street(self):
        """Moves the game to the next street, now side‑pot‑aware."""
        if self.street == "preflop":
            self.street = "flop"
            self.board.extend([self.deck.pop() for _ in range(3)])
        elif self.street == "flop":
            self.bb_option_done = False
            self.street = "turn"
            self.board.append(self.deck.pop())
        elif self.street == "turn":
            self.street = "river"
            self.board.append(self.deck.pop())
        elif self.street == "river":
            self.street = "showdown"
            return

        self.hand_history.append(
            f"--- {self.street.upper()} --- Board: {' '.join(self.board)}"
        )
        self.current_bet = 0
        self.aggressor_pos = -1
        self.last_raise_increment = self.bb_amount
        for p in self.players:
            p.bet_this_street = 0

        # 🔹 EARLY EXIT: if no further betting possible
        if self._deal_out_if_side_pot_closed():
            return

        # Who acts first?
        self.next_to_act_pos = (self.button_pos + 1) % len(self.players)

        # Skip folded & all‑in players
        hops = 0
        while (
            self.players[self.next_to_act_pos].folded
            or self.players[self.next_to_act_pos].is_all_in
        ) and hops < len(self.players):
            self.next_to_act_pos = (self.next_to_act_pos + 1) % len(self.players)
            hops += 1

        # If we looped the whole table, every seat is all‑in → recurse
        if hops == len(self.players):
            self._deal_out_if_side_pot_closed()  # guarantees exit
            return

        self.aggressor_pos = self.next_to_act_pos

    # ────────────────────────────────────────────────────────────────
    # Patch the *handle_player_actions* loop (or whatever your engine
    # calls it) so it calls the helper both *before* prompting a player
    # and *after* applying the action.
    # ────────────────────────────────────────────────────────────────

    def handle_player_actions(self):
        """Generic betting loop patched for blind‑all‑in edge‑cases."""
        while self.street not in ("showdown",):
            # 🔹 PRE‑prompt guard
            if self._deal_out_if_side_pot_closed():
                continue  # Street advanced or finished → restart/exit

            actor = self.players[self.next_to_act_pos]
            action, amt = actor.decide(self, self.next_to_act_pos)

            self.apply_action(actor, action, amt)

            # 🔹 POST‑action guard – maybe someone just went all‑in
            if self._deal_out_if_side_pot_closed():
                continue

            # Move to next seat
            self.next_to_act_pos = (self.next_to_act_pos + 1) % len(self.players)
            hops = 0
            while (
                self.players[self.next_to_act_pos].folded
                or self.players[self.next_to_act_pos].is_all_in
            ) and hops < len(self.players):
                self.next_to_act_pos = (self.next_to_act_pos + 1) % len(self.players)
                hops += 1

            if self._betting_round_closed():
              self.advance_street()      # deals flop/turn/river or goes to showdown
              continue


            # Everyone folded or is all‑in → advance automatically
            if hops == len(self.players):
                self._deal_out_if_side_pot_closed()
                continue

# --- Bot & AI Model Integration ---

import itertools

def add_ratio_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    for col1, col2 in itertools.combinations(columns, 2):
        ratio_name = f"{col1}_div_{col2}"
        inverse_name = f"{col2}_div_{col1}"

        # Perform division safely using numpy
        df[ratio_name] = np.where(df[col2] == 0, -999, df[col1] / df[col2])
        df[inverse_name] = np.where(df[col1] == 0, -999, df[col2] / df[col1])

        # Also handle NaN that might come from missing values
        df[ratio_name] = df[ratio_name].fillna(-999)
        df[inverse_name] = df[inverse_name].fillna(-999)

    return df

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any


def build_phh_actions(gs: "GameState") -> List[str]:
    """Convert game.hand_history to PHH-format action list."""
    actions = []
    # Deal hole cards
    for i, p in enumerate(gs.players):
        tag = f'p{i+1}'
        hole_str = ''.join(p.hole)
        actions.append(f'd dh {tag} {hole_str}')

    # Process hand history
    for line in gs.hand_history:
        if line.startswith('---'):
            # Street change, add board reveal
            street = line.split()[1].lower()
            if street == 'flop':
                board_str = ''.join(gs.board[:3])
            elif street == 'turn':
                board_str = gs.board[3]
            elif street == 'river':
                board_str = gs.board[4]
            else:
                continue
            actions.append(f'd db {board_str}')
        else:
            # Player action
            act_tag, amt, pos_tag = _parse_hand_line(line, gs.players)
            if act_tag != 'none':
                action_str = f'{pos_tag} {act_tag} {amt}' if amt > 0 else f'{pos_tag} {act_tag}'
                actions.append(action_str)

    return actions

def create_hand_dict(gs: "GameState") -> Dict[str, Any]:
    """Create a PHH-like dictionary from the current GameState."""
    sb_pos = (gs.button_pos + 1) % len(gs.players)
    bb_pos = (gs.button_pos + 2) % len(gs.players)
    sb_amount = gs.bb_amount // 2
    bb_amount = gs.bb_amount
    blinds = [0] * len(gs.players)
    blinds[sb_pos] = sb_amount
    blinds[bb_pos] = bb_amount

    # Requires gs.starting_stacks to be set in start_new_hand before posting blinds
    starting_stacks = getattr(gs, 'starting_stacks', [p.stack for p in gs.players])

    actions = build_phh_actions(gs)

    return {
        'actions': actions,
        'players': [p.name for p in gs.players],
        'blinds_or_straddles': blinds,
        'starting_stacks': starting_stacks,
        'hand': f'hand_{gs.hands_played}',
        'antes': [0] * len(gs.players),
    }


def state_to_df(gs: "GameState", actor_idx: int) -> pd.DataFrame:
    """Generate a single-row DataFrame using extract_pluribus_actions for the current actor."""
    hand_dict = create_hand_dict(gs)


    pluribus_tag = f'p{actor_idx + 1}'
    name=gs.players[actor_idx].name

    if hand_dict['actions'][-1]!='p':
      action_placeholder=pluribus_tag+ ' placeholder'
      hand_dict['actions'].append(action_placeholder)

    # Get feature rows from extract_pluribus_actions
    pre_rows, post_rows = extract_pluribus_actions(
        hand_dict,
        include_card_onehots=True,
        regression_target=False,
        exclude_actions=[],
        name=name
    )

    if 'placeholder' in hand_dict['actions'][-1]:
      hand_dict['actions'].remove(action_placeholder)

    # Determine current phase
    phase = 'preflop' if gs.street == 'preflop' else 'postflop'
    rows = pre_rows if phase == 'preflop' else post_rows

    if not rows:

        # Fallback if no rows for actor yet (e.g., before first action)
        row = {
            'hand_id': hand_dict['hand'],
            'phase': phase,
            'hole_rank1': encode_card(gs.players[actor_idx].hole[0])[0],
            'hole_rank2': encode_card(gs.players[actor_idx].hole[1])[0],
            'stack': gs.starting_stacks[actor_idx],
            'pot_size': sum(p.amount for p in gs.pots),
            'position': gs.get_position_label(actor_idx),
            'pluribus_action': 'none'
        }
    else:
      # Use the last row for the actor (most recent state)
      actor_rows = [r for r in rows if r.get('pluribus_action') is not None]
      row = actor_rows[-1] if actor_rows else rows[-1]

    # Convert to DataFrame
    df = pd.DataFrame([row])

    for col in df:
      if col in mapper:
        df[col]=df[col].map(mapper[col])

    num_cols=[ 'max_bet_seen',
    'recent_act_1_amt', 'recent_act_2_amt', 'recent_act_3_amt',  ##'stack', 'pot_size','spr','effective_stack_bb'
    'p2_last_amt', 'p3_last_amt', 'p4_last_amt', 'p5_last_amt',
    'p6_last_amt', 'p1_last_amt']
    df=add_ratio_columns(df,[col for col in num_cols if col in df])


    # Align with MASTER_COLS (assuming it's defined globally)
    if 'MASTER_COLS' in globals():
        missing_cols = [c for c in MASTER_COLS if c not in df.columns]

        for c in missing_cols:
            df[c] = -999

        df = df[MASTER_COLS].fillna(-999)


    for col, (max_val, min_val) in clipper_dct.items():
      if col in df.columns:
          mask = df[col] != -999   # keep sentinel values unchanged
          df.loc[mask, col] = df.loc[mask, col].clip(lower=min_val, upper=max_val)
    return df


def state_to_mamba_vec_and_mask(gs, actor_idx, feature_keys_from_ckpt=None):
    # 1) Create PHH-like dict from current GameState
    hand_dict = create_hand_dict(gs)  # existing helper in your UI
    name = gs.players[actor_idx].name

    # Mamba extractor expects repr()-wrapped lists (same as your zip processor)
    hand_for_extractor = {
        "actions": repr(hand_dict["actions"]),
        "players": repr(hand_dict["players"]),
        "blinds_or_straddles": repr(hand_dict["blinds_or_straddles"]),
        "starting_stacks": repr(hand_dict["starting_stacks"]),
        "hand": hand_dict["hand"],
    }

    seq = extract_pluribus_actions_mamba(
        hand_for_extractor,
        name=name,
        include_card_ints=True,
        include_basic_scalars=True,
        file_path=hand_dict["hand"],
    )

    # Fallback when we have no steps yet (e.g., before first decision)
    if not seq.get("steps"):
        keys = feature_keys_from_ckpt or []
        x_vec = torch.zeros(len(keys), dtype=torch.float32)
        legal = compute_legal_mask(gs, actor_idx)
        return x_vec, torch.tensor(legal, dtype=torch.uint8), keys

    step = seq["steps"][-1]                    # most recent state for this actor
    x_dict = step["x"]                         # {feature_name: value}
    keys = feature_keys_from_ckpt or list(x_dict.keys())

    # Build x in exact training order
    x_vec = torch.tensor([x_dict.get(k, 0.0) for k in keys], dtype=torch.float32)

    # Prefer your live legal calculation over the extractor’s placeholder
    legal = compute_legal_mask(gs, actor_idx)
    return x_vec, torch.tensor(legal, dtype=torch.uint8), keys

def state_to_transformer_seq_and_mask(
    gs,
    actor_idx: int,
    feature_keys_from_ckpt: list[str] | None = None,
):
    """
    Build a Transformer-ready sequence for the *current* actor:
      - X_seq: [T, F]  (all decision steps for this actor in this hand, up to now)
      - key_padding_mask: [T] bool (False=real token, True=PAD; here it's all False)
      - legal_last: [C] uint8 legality mask for the *current* step (use live engine)
      - feature_keys: list[str] length F (ordering used to build X features)

    Use with:
        X, kpm, legal, keys = state_to_transformer_seq_and_mask(gs, actor_idx, ckpt_keys)
        probs = infer_single_step(model, normalizer, X, kpm)   # returns [C]
        # Optionally mask illegal actions before argmax:
        very_neg = torch.finfo(probs.dtype).min / 2
        masked = torch.where(torch.tensor(legal, dtype=probs.dtype) > 0, probs, 0.0)
        action = masked.argmax().item()
    """
    import torch
    import numpy as np

    # 1) Build PHH-like dict from live GameState (same helper you already have)
    hand_dict = create_hand_dict(gs)
    name = gs.players[actor_idx].name

    # 2) Reuse the same extractor (it’s model-agnostic despite the name)
    hand_for_extractor = {
        "actions": repr(hand_dict["actions"]),
        "players": repr(hand_dict["players"]),
        "blinds_or_straddles": repr(hand_dict["blinds_or_straddles"]),
        "starting_stacks": repr(hand_dict["starting_stacks"]),
        "hand": hand_dict["hand"],
    }
    seq = extract_pluribus_actions_mamba(
        hand_for_extractor,
        name=name,
        include_card_ints=True,
        include_basic_scalars=True,
        file_path=hand_dict["hand"],
    )

    # 3) If we have no steps yet, return a single zero token so the caller can still run
    if not seq.get("steps"):
        keys = feature_keys_from_ckpt or []
        T, F = 1, len(keys)
        X_seq = torch.zeros((T, F), dtype=torch.float32)
        key_padding_mask = torch.zeros((T,), dtype=torch.bool)  # no PAD
        legal_last = torch.tensor(compute_legal_mask(gs, actor_idx), dtype=torch.uint8)
        return X_seq, key_padding_mask, legal_last, keys

    # 4) Gather all steps (this extractor already filters to the target `name`)
    steps = seq["steps"]
    first_x = steps[0]["x"]
    keys = feature_keys_from_ckpt or list(first_x.keys())
    T, F = len(steps), len(keys)

    # 5) Build full sequence X[T,F] in the *exact* training order of feature_keys
    X_np = np.zeros((T, F), dtype=np.float32)
    for t, step in enumerate(steps):
        x_dict = step["x"]
        X_np[t] = [x_dict.get(k, 0.0) for k in keys]

    X_seq = torch.from_numpy(X_np)                       # [T,F]
    key_padding_mask = torch.zeros((T,), dtype=torch.bool)  # no PAD since we’re not padding

    # 6) Prefer live legality for the *current* step
    legal_last = torch.tensor(compute_legal_mask(gs, actor_idx), dtype=torch.uint8)  # [C]

    return X_seq, key_padding_mask, legal_last, keys




def compute_legal_mask(gs, idx):
    hero = gs.players[idx]
    to_call = max(gs.current_bet - hero.bet_this_street, 0)
    stack = hero.stack

    # fold legal only if there's a bet to you
    fold_legal = 1 if to_call > 0 else 0

    # call legal if you can match the bet (or check for 0)
    call_legal = 1 if stack >= to_call else 0

    # raise legal if you can exceed a call by the min increment and have chips left
    min_raise = max(gs.bb_amount, getattr(gs, "last_raise_increment", gs.bb_amount))
    raise_legal = 1 if stack > to_call and (stack - to_call) >= min_raise else 0

    # If already all-in → only fold illegal, call can be 1 iff to_call == 0
    if hero.is_all_in:
        return [0, 1 if to_call == 0 else 0, 0]

    return [fold_legal, call_legal, raise_legal]


FOLD_MODEL_PATH = os.getenv("FOLD_MODEL_PATH", "folding_model_aggr.pkl")
CALL_MODEL_PATH = os.getenv("CALL_MODEL_PATH", "calling_model_aggr.pkl")
RAISE_MODEL_PATH = os.getenv("RAISE_MODEL_PATH", "raise_model_official.pkl")
VALUE_MODEL_PATH= os.getenv("VALUE_MODEL_PATH", "value_model_aggr.pkl")
RAISE_CALL_MODEL_PATH=os.getenv("RAISE_CALL_MODEL_PATH", "raise_call_model_aggr.pkl")
TRIPLE_MODEL_PATH = os.getenv("TRIPLE_MODEL_PATH", "triple_label_official.pkl")
RAISE_BINARY_PATH = os.getenv("RAISE_BINARY_PATH", "binary_raise_model_aggr.pkl")


import pickle, random, numpy as np
from typing import Tuple


import random
from functools import wraps


from collections import defaultdict, deque
from functools import wraps
from dataclasses import dataclass, field
from typing import Deque, Dict

# maniac_counter_generic.py
from collections import defaultdict, deque
from functools   import wraps


from functools import wraps
from typing import Tuple, Sequence

# --------------------------------------------------------------
# policy_blend.py  (add this near the top of poker_app.py)
# --------------------------------------------------------------
import joblib, numpy as np
from functools import wraps

class PickleableBooster:
    """A thin, sklearn-style wrapper around a LightGBM Booster."""
    def __init__(self, booster: lgb.Booster, num_class: int):
        self._booster   = booster
        self.num_class  = num_class

    # --- sklearn-like API -------------------------------------------------
    def predict_proba(self, X):
        raw = self._booster.predict(X)      # already shape (N, C)
        return np.clip(raw, 0, 1)

    # --- make the object pickle-safe -------------------------------------
    def __getstate__(self):
        return {
            "model_str": self._booster.model_to_string(),
            "num_class": self.num_class,
        }

    def __setstate__(self, state):
        self._booster  = lgb.Booster(model_str=state["model_str"])
        self.num_class = state["num_class"]



ACTIONS = ["fold", "call", "raise"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_RANK_MAP = {
    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6,
    "7": 7, "8": 8, "9": 9, "T": 10, "J": 11,
    "Q": 12, "K": 13, "A": 14,
}

# Premium combos expressed in the (rank_high, rank_low) numeric scheme
_PREMIUM_COMBOS = {
    (14, 14),  # AA
    (13, 13),  # KK
    (12, 12),  # QQ
    (14, 13),  # AK (suited or off)
}


def _card_ranks(cards: Sequence[str]) -> Tuple[int, int]:
    """Return the two numeric ranks (high, low) for a pair like ["As", "Kd"]."""
    ranks = sorted((_RANK_MAP[c[0]] for c in cards), reverse=True)
    return tuple(ranks)  # type: ignore[return-value]


def _is_premium(cards: Sequence[str]) -> bool:
    """True if the 2‑card holding is one of the premium pairs above."""
    try:
        return _card_ranks(cards) in _PREMIUM_COMBOS
    except KeyError:
        # Malformed card string – fail safe (treat as non‑premium)
        return False


# ---------------------------------------------------------------------------
# The decorator itself
# ---------------------------------------------------------------------------

def block_premium_folds(cls):
    """Class decorator – attach *after* every other decorator.

    Usage::

        @block_premium_folds
        @with_stack_depth_adjustment(...)
        class MyBot(BaseBot):
            ...

    The original ``get_action`` is called first; if it returns a *fold* in a
    pre‑flop street while the hero holds a premium hand, we override:
        • unopened pot  → raise 3×BB
        • call ≤ 4×BB  → flat‑call
        • else         → jam (shove)
    """

    original = cls.get_action

    @wraps(original)
    def _patched_get_action(self, gs, idx):
        action, amount = original(self, gs, idx)

        if gs.street.lower() == "preflop" and action == "fold":
            hero = gs.players[idx]
            cards = getattr(hero, "hole", getattr(hero, "hole_cards", None))
            if cards and _is_premium(cards):
                to_call = max(gs.current_bet - hero.bet_this_street, 0)
                bb = gs.bb_amount

                if to_call == 0:
                    # Unopened: standard 3× open‑raise
                    return "raise", min(3 * bb, hero.stack)
                if to_call <= 4 * bb:
                    # Small raise: just call
                    return "call", min(to_call, hero.stack)
                # Big raise: rip it in
                return "raise", hero.stack

        return action, amount

    cls.get_action = _patched_get_action
    return cls


def with_maniac_counter(window: int = 40,
                        shove_freq: float = 0.30,
                        # --- THRESHOLD bots -------------
                        fold_delta: float = 0.15,
                        call_delta: float = 0.10,
                        # --- PROBABILITY bots ----------
                        call_boost: float = 1.50,
                        fold_cut  : float = 0.50):
    """
    Adaptive defence versus a hero who open‑shoves far too often.

    Works for
      • threshold bots     – adjusts .fold_thresh / .call_thresh
      • probability bots   – warps action‑probs before sampling
    """

    def decorator(bot_cls):

        class ManiacAwareBot(bot_cls):
            _hist = defaultdict(lambda: deque(maxlen=window))   # shove history

            # ---------------------------------------------------------------
            # utilities -----------------------------------------------------
            @staticmethod
            def _record(gs):
                if not gs.hand_history:
                    return
                ln = gs.hand_history[-1].lower()
                if "all-in" in ln:
                    name = ln.split()[0]
                    ManiacAwareBot._hist[name].append(1)
                elif any(w in ln for w in ("raises", "bets", "calls")):
                    name = ln.split()[0]
                    ManiacAwareBot._hist[name].append(0)

            @staticmethod
            def _is_maniac(name) -> bool:
                hist = ManiacAwareBot._hist[name]
                return hist and sum(hist) / len(hist) >= shove_freq

            # ---------------------------------------------------------------
            # helper for prob‑style bots: warp a prob dict in‑place
            @staticmethod
            def _warp_probs(probs: dict,
                            call_boost=call_boost,
                            fold_cut=fold_cut):
                if "call" in probs:
                    probs["call"] *= call_boost
                if "fold" in probs:
                    probs["fold"] *= fold_cut

            # ---------------------------------------------------------------
            def get_action(self, gs, idx):
                ManiacAwareBot._record(gs)                 # update stats
                hero_name  = gs.players[0].name
                maniac     = ManiacAwareBot._is_maniac(hero_name)

                # THRESHOLD BOT
                if maniac and hasattr(self, "fold_thresh"):
                    # --- temporarily loosen --------------------------------
                    orig_fold, orig_call = self.fold_thresh, self.call_thresh
                    self.fold_thresh = max(0, orig_fold - fold_delta)
                    self.call_thresh = max(0, orig_call - call_delta)

                    action, amt = super().get_action(gs, idx)

                    # --- restore -------------------------------------------
                    self.fold_thresh, self.call_thresh = orig_fold, orig_call
                    return action, amt

                # PROBABILITY BOT
                if maniac and hasattr(self, "sample_action_proba"):
                    # Convention: bot has method that returns a prob dict
                    probs = self.sample_action_proba(gs, idx)
                    ManiacAwareBot._warp_probs(probs)
                    return self._roulette_select(probs)        # bot’s own helper

                # FALLBACK: just call parent unchanged
                return super().get_action(gs, idx)

        ManiacAwareBot.__name__ = f"{bot_cls.__name__}ManiacAware"
        return ManiacAwareBot

    return decorator

@dataclass
class _SeatStats:
    actions: Deque[str] = field(default_factory=lambda: deque(maxlen=50))
    vpip:   float = 0.0      # Voluntarily Put $ In Pot
    pfr:    float = 0.0      # Pre‑flop Raise

    def update(self, street: str, action: str):
        """Record one action (“call”, “raise”, “fold”, …) on this street."""
        if street == "preflop":
            self.actions.append(action)

        # recompute ratios on the fly (cheap for small deque)
        vpip_cnt = sum(a in ("call", "raise") for a in self.actions)
        pfr_cnt  = sum(a == "raise" for a in self.actions)
        n        = len(self.actions) or 1
        self.vpip = vpip_cnt / n
        self.pfr  = pfr_cnt  / n


def with_exploit_tracking(*, window=50, vpip_loose=0.40, pfr_nit=0.10):
    """
    Class decorator.  Adds:
      • self._villain_stats: Dict[player_id → _SeatStats]
      • self.update_stats(gs)      – call *once* after every street transition
    and transparently adjusts fold/call thresholds before the bot
    makes a decision.
    """
    def decorator(cls):
        orig_init      = cls.__init__
        orig_getaction = cls.get_action

        # ---------- wrap __init__ -----------------------------------
        @wraps(orig_init)
        def new_init(self, *a, **kw):
            orig_init(self, *a, **kw)
            self._villain_stats: Dict[int, _SeatStats] = defaultdict(
                lambda: _SeatStats(deque(maxlen=window))
            )

        cls.__init__ = new_init

        # ---------- helper to be called by game engine --------------
        def update_stats(self, gs):
            street = gs.street.lower()
            for seat_id, p in enumerate(gs.players):
                if p.folded:
                    continue
                last_act = gs.last_actions.get(seat_id, (None, None))[0]
                if last_act:
                    self._villain_stats[seat_id].update(street, last_act)

        cls.update_stats = update_stats

        # ---------- wrap get_action ---------------------------------
        @wraps(orig_getaction)
        def wrapped_get_action(self, gs, idx):
            # Pull villain stats
            hero_id = idx
            loose_count = nit_count = 0
            for seat, stats in self._villain_stats.items():
                if seat == hero_id:
                    continue
                loose_count += stats.vpip >= vpip_loose
                nit_count   += stats.pfr  <= pfr_nit

            # Example ­‑‑ adjust thresholds (5 ppt per loose/nit villain)
            self.fold_thresh = max(
                0.05, min(0.95, self.fold_thresh - 0.05 * loose_count)
            )
            self.call_thresh = max(
                0.05, min(0.95, self.call_thresh + 0.05 * nit_count)
            )

            return orig_getaction(self, gs, idx)

        cls.get_action = wrapped_get_action
        return cls
    return decorator



def check_if_no_bet(func):
    """
    Allows the bot to still RAISE (bet) when no bet exists,
    but forces CHECK instead of 'fold' or 'call 0'.
    """
    def wrapper(self, gs, idx):
        hero = gs.players[idx]
        to_call = max(gs.current_bet - hero.bet_this_street, 0)

        # Get the bot's intended decision
        action, amt = func(self, gs, idx)

        if to_call == 0:
            if action == "fold":
                return "call", 0  # folding is illegal → force check
            if action == "call":
                return "call", 0  # call 0 → check
            # If it wanted to raise → allow it as a bet
            return action, amt

        # Otherwise, just use normal decision
        return action, amt

    return wrapper

def heads_up_value_override(heads_up_model_path: str):
    """
    Decorator to temporarily replace `self.value_model` with a heads-up-specific model
    whenever only 1 opponent remains in the hand.
    """
    def decorator(func):
        # Preload the heads-up model once
        heads_up_model = pickle.load(open(heads_up_model_path, "rb"))

        @wraps(func)
        def wrapper(self, gs, idx):
            # Count active opponents
            live_opponents = [
                p for i, p in enumerate(gs.players)
                if i != idx and not p.folded and not p.is_all_in
            ]

            # Save original model
            orig_model = self.value_model

            # If heads-up → swap model
            if len(live_opponents) == 1:
                self.value_model = heads_up_model

            try:
                # Run the original decision logic
                return func(self, gs, idx)
            finally:
                # Restore the original model no matter what
                self.value_model = orig_model

        return wrapper
    return decorator

def get_effective_stack_vs_table(gs, hero_id):
    hero_stack = gs.players[hero_id].stack
    effective_stacks = [
        min(hero_stack, p.stack)
        for i, p in enumerate(gs.players)
        if i != hero_id and not p.folded
    ]
    return min(effective_stacks) if effective_stacks else hero_stack

def with_stack_depth_adjustment(*, deep_thresh_bb=60, short_thresh_bb=25):
    """
    Class decorator. Adds dynamic stack-awareness:

      • Calculates the hero's effective stack vs. all opponents.
      • If deep-stacked (>= deep_thresh_bb) → loosen thresholds.
      • If short-stacked (<= short_thresh_bb) → tighten thresholds.
      • Otherwise (medium stack) → unchanged.

    Example adjustments:
      - Deeper stack → call more, fold less (to realize implied odds).
      - Short stack → fold more, avoid marginal spots.
    """
    def decorator(cls):
        orig_getaction = cls.get_action

        @wraps(orig_getaction)
        def wrapped_get_action(self, gs, idx):
            hero = gs.players[idx]
            bb = gs.bb_amount or 1

            # Compute effective stack vs. all live opponents
            eff_stack_vs_table = [
                min(hero.stack, opp.stack)
                for i, opp in enumerate(gs.players)
                if i != idx and not opp.folded
            ]
            eff_stack = min(eff_stack_vs_table) if eff_stack_vs_table else hero.stack
            eff_bb = eff_stack / bb

            # Adjust fold/call thresholds dynamically
            if eff_bb >= deep_thresh_bb:
                # Deep → loosen play
                self.fold_thresh = max(0.05, self.fold_thresh - 0.05)
                self.call_thresh = min(0.95, self.call_thresh + 0.05)
            elif eff_bb <= short_thresh_bb:
                # Short → tighten play
                self.fold_thresh = min(0.95, self.fold_thresh + 0.05)
                self.call_thresh = max(0.05, self.call_thresh - 0.05)
            # else → leave thresholds unchanged

            return orig_getaction(self, gs, idx)

        cls.get_action = wrapped_get_action
        return cls
    return decorator



import random
from functools import wraps

def gto_mixed_strategy(mix_strength=0.15, random_raise_factor=(0.8, 1.2)):

    def decorator(func):
        @wraps(func)
        def wrapper(self, gs, idx):
            action, amt = func(self, gs, idx)

            # 1️⃣ Randomly flip action ~ mix_strength
            if random.random() < mix_strength:
                alt_choices = ["fold", "call", "raise"]
                alt_choices.remove(action)
                action = random.choice(alt_choices)

                # If we swapped to raise → randomize pot-sized
                if action == "raise":
                    pot = gs.pots[0].amount
                    base_amt = max(gs.current_bet - gs.players[idx].bet_this_street, 0) + pot
                    amt = int(base_amt * random.uniform(*random_raise_factor))
                    amt = min(amt, gs.players[idx].stack)
                elif action == "call":
                    amt = min(max(gs.current_bet - gs.players[idx].bet_this_street, 0),
                              gs.players[idx].stack)
                else:
                    amt = 0

            # 2️⃣ If it's a raise anyway, slightly randomize the sizing
            elif action == "raise":
                amt = int(amt * random.uniform(*random_raise_factor))
                amt = max(min(amt, gs.players[idx].stack), gs.current_bet + gs.bb_amount)

            return action, amt
        return wrapper
    return decorator


def preflop_dynamic_allin_model(bb_tight=20, bb_medium=10):
    """
    Dynamically choose which model drives preflop all-in decisions:
      - > bb_tight: value_model (tight)
      - bb_medium–bb_tight: raise_call_model (looser)
      - < bb_medium: raise_model (shove-happy)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, gs, idx):
            hero = gs.players[idx]
            street = gs.street.lower()
            eff_bb = hero.stack / max(1, gs.current_bet)   # ≥ 3× pot → shove_model

            action, amt = func(self, gs, idx)

            # Only modify **preflop** and only affect *raise == hero.stack* (jam) logic
            if street == "preflop":
                X = state_to_df(gs, idx)

                # Pick model based on stack depth
                if eff_bb > bb_tight:
                    allin_model = self.value_model
                elif eff_bb > bb_medium:
                    allin_model = self.raise_call_model
                else:
                    allin_model = self.raise_model

                # Predict "chips to add"
                pred_add = int(round(float(allin_model.predict(X)[0])))

                # If model wants >= stack → force jam
                if pred_add >= hero.stack:
                    return "raise", hero.stack

            return action, amt
        return wrapper
    return decorator


from typing import Callable, Optional, Tuple, Dict, List
import os
import torch

# Import your Mamba utilities (from your uploaded mamba_data_gen.py)
from mamba_data_gen import (
    load_poker_mamba,
    infer_single_step,
    extract_pluribus_actions_mamba,
)

# If not already present:
ACTIONS = ["fold", "call", "raise"]  # index-aligned with your model’s logits
DEFAULT_CKPT = "augment_poker_transformer.pt"

class MambaBot:
    """
    Poker bot that chooses among {fold, call, raise} using a saved Mamba model.
    Raise sizes are produced by an optional *tabular regressor* (DonKeyxote-style).
    """

    def __init__(
        self,
        ckpt_path: str = DEFAULT_CKPT,
        *,
        device: Optional[torch.device] = None,
        phh_builder: Optional[Callable[[object], Dict]] = None,
        # NEW: tabular raise-size regressor
        raise_model_path: Optional[str] = RAISE_MODEL_PATH,
        tabular_builder: Optional[Callable[[object, int], "pd.DataFrame"]] = None,
        raise_outputs: str = "multiplier",  # "multiplier" (×pot) or "total"
        temperature: Optional[float] = None,  # None/0 => greedy; >0 => sample
        emit_check: bool = False,             # return "check" when to_call==0
    ):
        self.device = device or torch.device("cpu")
        self.model, self.normalizer, self.feature_keys, _ = load_poker_mamba(
            ckpt_path, device=self.device
        )
        self.phh_builder = phh_builder
        self.temperature = temperature
        self.emit_check = emit_check
        self.name = "mamba"

        # NEW: raise-size regressor + helpers
        self.raise_model = None
        if raise_model_path is not None:
            self.raise_model = pickle.load(open(raise_model_path, "rb"))
        self.tabular_builder = tabular_builder
        assert raise_outputs in ("multiplier", "total")
        self.raise_outputs = raise_outputs

    # ── Public UI-facing method ───────────────────────────────────────────────
    def get_action(self, gs, seat_idx: int) -> Tuple[str, int]:
        """
        Returns (action, amount) for the acting seat.
        """
        x_vec, legal = self._state_to_vec_and_mask(gs, seat_idx)

        # Forward pass (masked argmax inside infer_single_step; logits for sampling)
        act_id, logits = infer_single_step(self.model, self.normalizer, x_vec, legal)

        if self.temperature and self.temperature > 0:
            very_neg = torch.finfo(logits.dtype).min / 2
            masked = torch.where(legal.bool(), logits, very_neg)
            probs = torch.softmax(masked / float(self.temperature), dim=-1)
            if probs.sum() <= 0:
                probs = legal.float() / max(legal.sum().item(), 1)
            act_id = int(torch.multinomial(probs, num_samples=1).item())

        action = ACTIONS[act_id]

        hero = gs.players[seat_idx]
        to_call = max(gs.current_bet - hero.bet_this_street, 0)

        if action == "fold":
            return ("fold", 0)

        if action == "call":
            amt = min(to_call, hero.stack)
            if self.emit_check and to_call == 0:
                return ("check", 0)
            return ("call", amt)

        # "raise" → delegate size to tabular regressor (with legalization + fallbacks)
        size = self._tabular_raise_size(gs, seat_idx)
        return ("raise", size)

    # ── Internals ────────────────────────────────────────────────────────────
    def _state_to_vec_and_mask(self, gs, seat_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build (x_vec[F], legal[3]) for the current seat from live GameState.
        Uses your PHH bridge to emit the 'now' step for feature construction.
        """
        hand = self._build_phh_dict(gs)
        hero_tag = f"p{seat_idx + 1}"
        placeholder = f"{hero_tag} cc 0"

        actions_list: List[str] = list(hand["actions"])
        actions_list.append(placeholder)

        hand_for_extractor = {
            "actions": repr(actions_list),
            "players": repr(hand["players"]),
            "blinds_or_straddles": repr(hand["blinds_or_straddles"]),
            "starting_stacks": repr(hand["starting_stacks"]),
            "hand": hand.get("hand", "ui"),
        }

        seq = extract_pluribus_actions_mamba(
            hand_for_extractor,
            name=gs.players[seat_idx].name,
            include_card_ints=True,
            include_basic_scalars=True,
            file_path=hand_for_extractor["hand"],
        )

        if not seq.get("steps"):
            F = len(self.feature_keys) if self.feature_keys else 0
            x_vec = torch.zeros(F, dtype=torch.float32)
            legal = self._legal_mask(gs, seat_idx)
            return x_vec, legal

        step = seq["steps"][-1]
        x_dict: Dict[str, float] = step["x"]

        if not self.feature_keys:
            self.feature_keys = list(x_dict.keys())

        x_vec = torch.tensor([x_dict.get(k, 0.0) for k in self.feature_keys], dtype=torch.float32)
        legal = self._legal_mask(gs, seat_idx)
        return x_vec, legal

    def _build_phh_dict(self, gs) -> Dict:
        if self.phh_builder is not None:
            return self.phh_builder(gs)
        return create_hand_dict(gs)

    @staticmethod
    def _legal_mask(gs, seat_idx: int) -> torch.Tensor:
        hero = gs.players[seat_idx]
        to_call = max(gs.current_bet - hero.bet_this_street, 0)
        stack = hero.stack

        fold_ok = 1 if to_call > 0 else 0
        call_ok = 1 if stack >= to_call else 0

        min_inc = getattr(gs, "last_raise_increment", getattr(gs, "bb_amount", 1))
        raise_ok = 1 if (not getattr(hero, "is_all_in", False)) and stack > to_call else 0

        if getattr(hero, "is_all_in", False):
            return torch.tensor([0, 1 if to_call == 0 else 0, 0], dtype=torch.uint8)

        return torch.tensor([fold_ok, call_ok, raise_ok], dtype=torch.uint8)


    @staticmethod
    def _legal_raise_total(desired_total: int, gs, hero) -> Optional[int]:
        """
        Return the nearest legal raise TOTAL (chips already in + new chips), or None.
        """
        current_bet = gs.current_bet
        last_inc    = getattr(gs, "last_raise_increment", gs.bb_amount)
        min_total   = current_bet + last_inc
        max_total   = hero.bet_this_street + hero.stack  # all-in boundary

        if desired_total <= current_bet or max_total < min_total:
            return None
        return max(min_total, min(desired_total, max_total))

    def _make_tabular_X(self, gs, seat_idx: int) -> Optional[pd.DataFrame]:
        X    = state_to_df(gs, seat_idx)
        return X

    def _tabular_raise_size(self, gs, seat_idx: int) -> int:
        """
        Use the tabular regressor to propose a raise TOTAL, then legalize.
        Fallbacks to pot-sized, then min-raise if needed.
        """
        hero = gs.players[seat_idx]
        pot  = gs.pots[0].amount

        # No model? Use minimum legal raise.
        if self.raise_model is None:
            return self._min_raise_size(gs, seat_idx)

        X = self._make_tabular_X(gs, seat_idx)
        if X is None:
            return self._min_raise_size(gs, seat_idx)

        # Predict → desired TOTAL
        pred = float(self.raise_model.predict(X)[0])
        if self.raise_outputs == "multiplier":
            desired_total = int(round(pred * pot))
        else:  # "total"
            desired_total = int(round(pred))

        legal_total = self._legal_raise_total(desired_total, gs, hero)
        if legal_total is not None:
            return legal_total

        # Fallback 1: try a pot-sized total
        pot_total = int(round(1.0 * pot))
        legal_total = self._legal_raise_total(pot_total, gs, hero)
        if legal_total is not None:
            return legal_total

        # Fallback 2: minimum legal raise
        return self._min_raise_size(gs, seat_idx)

    @staticmethod
    def _min_raise_size(gs, seat_idx: int) -> int:
        hero = gs.players[seat_idx]
        to_call = max(gs.current_bet - hero.bet_this_street, 0)
        min_inc = getattr(gs, "last_raise_increment", getattr(gs, "bb_amount", 1))
        target = to_call + min_inc
        return min(hero.stack, target)

from functools import wraps

_RANK = {"2":2,"3":3,"4":4,"5":5,"6":6,"7":7,"8":8,"9":9,"T":10,"J":11,"Q":12,"K":13,"A":14}

def _hole_ranks_suited(cards):
    """cards = ['As','Kd'] -> (hi, lo, suited)"""
    r1, r2 = _RANK[cards[0][0]], _RANK[cards[1][0]]
    s1, s2 = cards[0][1], cards[1][1]
    hi, lo = (r1, r2) if r1 >= r2 else (r2, r1)
    return hi, lo, (s1 == s2)

def _facing_preflop_shove(gs, idx) -> bool:
    """Detect if hero is facing a preflop shove or a raise >= 80% of effective stack."""
    if getattr(gs, "street", "preflop").lower() != "preflop":
        return False

    hero = gs.players[idx]
    to_call = max(gs.current_bet - hero.bet_this_street, 0)
    if to_call <= 0:
        return False

    # Effective stack vs. all live opponents
    eff_stack = hero.stack
    for i, p in enumerate(gs.players):
        if i == idx or p.folded:
            continue
        eff_stack = min(eff_stack, p.stack)

    # Check: shove (opponent all-in at current bet)
    for i, p in enumerate(gs.players):
        if i == idx or p.folded:
            continue
        if p.is_all_in and p.bet_this_street == gs.current_bet:
            return True

    # Check: raise amount is 80%+ of effective stack
    if eff_stack > 0 and to_call >= 0.8 * eff_stack:
        return True

    return False


def _effective_bb(gs, idx) -> float:
    bb = max(1, getattr(gs, "bb_amount", 1))
    hero = gs.players[idx]
    eff = hero.stack
    for i, p in enumerate(gs.players):
        if i == idx or p.folded:
            continue
        eff = min(eff, p.stack)
    return eff / bb

def with_preflop_shove_call_policy(
    *,
    short_bb_1: int = 25,   # widen to TT/AQs/AJs at <= this
    short_bb_2: int = 15,   # further widen to 99/ATs/KQs at <= this
    treat_villain_as_tight: bool | None = None
):
    """
    Enforces 'snap-call vs. shove' buckets preflop.

    Buckets:
      • Always call: AA, KK
      • Almost always call: QQ, AKs
      • Usually call: JJ, AKo   (fold only if villain extremely tight & deep)
      • Situational (<= short_bb_1): TT, AQs, AJs
      • Extra short (<= short_bb_2): 99, ATs, KQs

    If `treat_villain_as_tight` is None and your bot maintains exploit stats
    (e.g., self._villain_stats with pfr), we infer tightness heuristically.
    """

    def decorator(cls):
        original_get_action = cls.get_action

        @wraps(original_get_action)
        def wrapped(self, gs, idx):
            # Let the bot decide first
            action, amt = original_get_action(self, gs, idx)

            # Only intervene for preflop shove spots we’re facing
            if not _facing_preflop_shove(gs, idx):
                return action, amt

            hero = gs.players[idx]
            if not getattr(hero, "hole", None) or len(hero.hole) != 2:
                return action, amt

            hi, lo, suited = _hole_ranks_suited(hero.hole)

            # Hand bucket checks
            is_pair  = hi == lo
            is_AK    = (hi, lo) == (_RANK["A"], _RANK["K"])
            is_AQs   = (hi, lo) == (_RANK["A"], _RANK["Q"]) and suited
            is_AJs   = (hi, lo) == (_RANK["A"], _RANK["J"]) and suited
            is_ATs   = (hi, lo) == (_RANK["A"], _RANK["T"]) and suited
            is_KQs   = (hi, lo) == (_RANK["K"], _RANK["Q"]) and suited

            # Always
            if is_pair and hi >= _RANK["K"]:  # AA, KK
                force_call = True
            # Almost always
            elif (is_pair and hi == _RANK["Q"]) or (is_AK and suited):  # QQ, AKs
                force_call = True
            else:
                # Usually (JJ, AKo) unless villain is super tight & deep
                force_call = False
                if (is_pair and hi == _RANK["J"]) or (is_AK and not suited):
                    force_call = True

                # Situational wideners by stack depth (tournament-ish logic)
                eff_bb = _effective_bb(gs, idx)
                if not force_call and eff_bb <= short_bb_1:
                    if (is_pair and hi == _RANK["T"]) or is_AQs or is_AJs:
                        force_call = True
                if not force_call and eff_bb <= short_bb_2:
                    if (is_pair and hi == _RANK["9"]) or is_ATs or is_KQs:
                        force_call = True

                # Optional: tighten JJ/AKo if villain is very tight & deep
                # Try to infer tightness if not explicitly provided
                villain_tight = treat_villain_as_tight
                if villain_tight is None and hasattr(self, "_villain_stats"):
                    # Heuristic: average PFR across opponents; < 10% → tight
                    opp_pfr = []
                    for seat, stats in getattr(self, "_villain_stats", {}).items():
                        if seat != idx:
                            opp_pfr.append(getattr(stats, "pfr", 0.0))
                    if opp_pfr:
                        villain_tight = (sum(opp_pfr) / len(opp_pfr)) <= 0.10

                if villain_tight and eff_bb >= 60:
                    if (is_pair and hi == _RANK["J"]) or (is_AK and not suited):
                        force_call = False  # allow folding JJ/AKo vs. nit & deep

            if force_call:
                return "raise", hero.bet_this_street + hero.stack

            return action, amt

        cls.get_action = wrapped
        return cls

    return decorator


# --- constants / defaults ---
# --- constants / defaults ---
from typing import Optional, Callable, Tuple, Dict, List
import torch
import numpy as np
import pickle
import pandas as pd  # only if you use the tabular raise model

ACTIONS = ["fold", "call", "raise"]
DEFAULT_CKPT = "augment_poker_transformer.pt"

# You provide these:
# - extract_pluribus_actions_mamba(...)
# - create_hand_dict(gs)
# - state_to_df(gs, seat_idx)  # if using tabular raise model

@with_temperature_by_elims(temp_at_start=0.3, temp_at_end=0.6, end_players=2)
@with_commitment_rule(0.6)
@with_short_stack_nash_equilibrium()
class TransformerBot:
    """
    Poker bot that chooses among {fold, call, raise} using a saved Transformer model.
    Raise sizes are produced by an optional tabular regressor.
    """

    def __init__(
        self,
        ckpt_path: str = DEFAULT_CKPT,
        *,
        device: Optional[torch.device] = None,
        phh_builder: Optional[Callable[[object], Dict]] = None,
        raise_model_path: Optional[str] = RAISE_MODEL_PATH,
        tabular_builder: Optional[Callable[[object, int], "pd.DataFrame"]] = None,
        raise_outputs: str = "multiplier",  # "multiplier" (×pot) or "total"
        temperature: Optional[float] = 0.3,  # None/0 => greedy; >0 => sample
        emit_check: bool = False,
        name: str = "transformer",
    ):
        self.device = device or torch.device("cpu")

        # 🔁 CHANGED: new loader signature & unpack order
        self.model, self.normalizer, self.feature_keys, self.model_cfg = (
            load_poker_transformer(ckpt_path, device=str(self.device))
        )
        self.model.eval()

        self.phh_builder = phh_builder
        self.temperature = float(temperature) if temperature else 0.0
        self.emit_check = emit_check
        self.name = name

        # Optional raise-size regressor
        self.raise_model = None
        if raise_model_path is not None:
            with open(raise_model_path, "rb") as f:
                self.raise_model = pickle.load(f)
        self.tabular_builder = tabular_builder
        assert raise_outputs in ("multiplier", "total")
        self.raise_outputs = raise_outputs

    # ── Public UI-facing method ───────────────────────────────────────────────
    def get_action(self, gs, seat_idx: int) -> Tuple[str, int]:
        """
        Returns (action, amount) for the acting seat.
        """
        X_seq, key_padding_mask, legal_last = self._state_to_seq_and_mask(gs, seat_idx)

        keys = self.feature_keys or [f"f{i}" for i in range(X_seq.shape[-1])]

        x_last = X_seq[-1].detach().cpu().numpy()   # [F]

        neg_mask = (x_last == -999)
        neg_keys = [k for k, is_bad in zip(keys, neg_mask) if is_bad]

        # Batchify & normalize
        X = X_seq.unsqueeze(0).to(self.device)                  # [1,T,F]
        kpm = key_padding_mask.unsqueeze(0).to(self.device)     # [1,T]
        if self.normalizer is not None:
            with torch.no_grad():
                X = self.normalizer.transform(X)

        # Forward → last-timestep decision
        with torch.no_grad():
            logits = self.model(X, key_padding_mask=kpm)[:, -1, :]  # [1,C] -> [C]
            logits = logits.squeeze(0)

            # Legality mask
            legal = legal_last.to(self.device).bool()
            very_neg = torch.finfo(logits.dtype).min / 2
            masked_logits = torch.where(legal, logits, very_neg)

            # Greedy or temperature sampling
            if self.temperature > 0:
                probs = torch.softmax(masked_logits / self.temperature, dim=-1)
                if not torch.isfinite(probs).all() or probs.sum() <= 0:
                    legal_f = legal.float()
                    probs = legal_f / max(1.0, legal_f.sum().item())
                act_id = int(torch.multinomial(probs, num_samples=1).item())
            else:
                act_id = int(torch.argmax(masked_logits).item())

        action = ACTIONS[act_id]
        hero = gs.players[seat_idx]
        to_call = max(gs.current_bet - hero.bet_this_street, 0)

        if action == "fold":
            return ("fold", 0)

        if action == "call":
            amt = min(to_call, getattr(hero, "stack", 0))
            if self.emit_check and to_call == 0:
                return ("check", 0)
            return ("call", amt)

        # "raise" → delegate size to tabular regressor (with legalization + fallbacks)
        size = self._tabular_raise_size(gs, seat_idx)
        return ("raise", size)

    # ── Internals ────────────────────────────────────────────────────────────
    def _state_to_seq_and_mask(
        self, gs, seat_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build:
          - X_seq[T,F]           (full actor history up to *now*, feature-ordered)
          - key_padding_mask[T]  (all False since there’s no padding here)
          - legal_last[C]        (live legality for current step)
        """
        # 1) PHH-like dict from live GameState
        hand = self._build_phh_dict(gs)
        hero_tag = f"p{seat_idx + 1}"
        placeholder = f"{hero_tag} cc 0"  # mark it's hero's turn now

        actions_list: List[str] = list(hand["actions"])
        actions_list.append(placeholder)

        hand_for_extractor = {
            "actions": repr(actions_list),
            "players": repr(hand["players"]),
            "blinds_or_straddles": repr(hand["blinds_or_straddles"]),
            "starting_stacks": repr(hand["starting_stacks"]),
            "hand": hand.get("hand", "ui"),
        }

        # 2) Extract features for this actor
        seq = extract_pluribus_actions_mamba(
            hand_for_extractor,
            name=gs.players[seat_idx].name,
            include_card_ints=True,
            include_basic_scalars=True,
            file_path=hand_for_extractor["hand"],
        )

        # 3) No steps yet → single zero token in ckpt order
        if not seq.get("steps"):
            keys = self.feature_keys or []
            T, F = 1, len(keys)
            X_seq = torch.zeros((T, F), dtype=torch.float32)
            key_padding_mask = torch.zeros((T,), dtype=torch.bool)
            legal_last = self._legal_mask(gs, seat_idx)
            return X_seq, key_padding_mask, legal_last

        # 4) Build full [T,F] in EXACT checkpoint feature order
        steps = seq["steps"]
        first_x = steps[0]["x"]



        keys = self.feature_keys
        T, F = len(steps), len(keys)
        X_np = np.zeros((T, F), dtype=np.float32)
        for t, step in enumerate(steps):
            x_dict = step["x"]
            X_np[t] = [x_dict.get(k, 0.0) for k in keys]

        X_seq = torch.from_numpy(X_np)
        key_padding_mask = torch.zeros((T,), dtype=torch.bool)
        legal_last = self._legal_mask(gs, seat_idx)
        return X_seq, key_padding_mask, legal_last

    def _build_phh_dict(self, gs) -> Dict:
        if self.phh_builder is not None:
            return self.phh_builder(gs)
        return create_hand_dict(gs)

    @staticmethod
    def _legal_mask(gs, seat_idx: int) -> torch.Tensor:
        hero = gs.players[seat_idx]
        to_call = max(gs.current_bet - hero.bet_this_street, 0)
        stack = hero.stack

        fold_ok = 1 if to_call > 0 else 0
        call_ok = 1 if stack >= to_call else 0

        min_inc = getattr(gs, "last_raise_increment", getattr(gs, "bb_amount", 1))
        raise_ok = 1 if (not getattr(hero, "is_all_in", False)) and stack > to_call else 0

        if getattr(hero, "is_all_in", False):
            return torch.tensor([0, 1 if to_call == 0 else 0, 0], dtype=torch.uint8)

        return torch.tensor([fold_ok, call_ok, raise_ok], dtype=torch.uint8)

    @staticmethod
    def _legal_raise_total(desired_total: int, gs, hero) -> Optional[int]:
        current_bet = gs.current_bet
        last_inc    = getattr(gs, "last_raise_increment", gs.bb_amount)
        min_total   = current_bet + last_inc
        max_total   = hero.bet_this_street + hero.stack  # all-in boundary

        if desired_total <= current_bet or max_total < min_total:
            return None
        return max(min_total, min(desired_total, max_total))

    def _make_tabular_X(self, gs, seat_idx: int) -> Optional[pd.DataFrame]:
        if self.tabular_builder is not None:
            return self.tabular_builder(gs, seat_idx)
        return state_to_df(gs, seat_idx)

    def _tabular_raise_size(self, gs, seat_idx: int) -> int:
        hero = gs.players[seat_idx]
        pot  = gs.pots[0].amount if getattr(gs, "pots", None) else 0

        if self.raise_model is None:
            return self._min_raise_size(gs, seat_idx)

        X = self._make_tabular_X(gs, seat_idx)
        if X is None:
            return self._min_raise_size(gs, seat_idx)

        pred = float(self.raise_model.predict(X)[0])
        desired_total = int(round(pred * pot)) if self.raise_outputs == "multiplier" else int(round(pred))

        legal_total = self._legal_raise_total(desired_total, gs, hero)
        if legal_total is not None:
            return legal_total

        pot_total = int(round(1.0 * pot))
        legal_total = self._legal_raise_total(pot_total, gs, hero)
        if legal_total is not None:
            return legal_total

        return self._min_raise_size(gs, seat_idx)

    @staticmethod
    def _min_raise_size(gs, seat_idx: int) -> int:
        hero = gs.players[seat_idx]
        to_call = max(gs.current_bet - hero.bet_this_street, 0)
        min_inc = getattr(gs, "last_raise_increment", getattr(gs, "bb_amount", 1))
        target = to_call + min_inc
        return min(hero.stack, target)





# --- Streamlit UI Application ---
st.set_page_config(page_title="Poker vs AI", layout="wide")

import os

def _ensure_game_in_session():
    fresh_needed = ("game" not in st.session_state or
                    not hasattr(st.session_state.game, "get_position_label"))

    if fresh_needed:
        # ⚠️  create the *bot* objects first …
        st.session_state.bots=  [TransformerBot("augment_poker_transformer.pt") for i in range( 1,6)]

        #st.session_state.bots=[ Zach(),Yanchen(), William(),DonKeyxote(),MechIII()]

        counter=0
        for bot in st.session_state.bots:
          bot.name+=str(counter)
          counter+=1

        # … then hand their names to GameState
        bot_names = [b.name for b in st.session_state.bots]
        st.session_state.game = GameState(bot_names=bot_names)
        st.session_state.game.start_new_hand()

        st.session_state.game_over = False
        st.session_state.winner_info = ""
        st.session_state.played_hand_rows = []


_ensure_game_in_session()
if "went_to_showdown" not in st.session_state:
    st.session_state.went_to_showdown = False
game: GameState = st.session_state.game
bots = st.session_state.bots


def handle_player_action(action: str, amount: int = 0):
    """
    Process an action from the hero or a bot, **without double‑logging
    all‑ins or $0 calls**, and with proper side‑pot handling.
    """
    player_pos  = game.next_to_act_pos
    player      = game.players[player_pos]
    to_call     = max(game.current_bet - player.bet_this_street, 0)


    snap = state_to_df(game, player_pos).iloc[0].to_dict()
    snap.update({
        "hand_no":   game.hands_played,
        "street":    game.street,        # preflop / flop / turn / river
        "actor_pos": player_pos,         # 0‑based seat index
        "action":    action,             # 'fold' | 'call' | 'raise'
        "amount":    amount,             # 0 for check/fold
    })

    st.session_state.played_hand_rows.append(snap)

    # Helper – move chips once
    def _commit(chips: int) -> None:
        player.stack      -= chips
        player.bet_this_street += chips
        player.contributed     += chips
        game.pots[0].amount    += chips
        if player.stack == 0:
            player.is_all_in = True

    # -------------------------- FOLD --------------------------
    if action == "fold":
        player.folded = True
        game.hand_history.append(f"{player.name} folds")
        game.last_actions[player_pos] = ("f", 0)
        find_next_player()
        return

    # -------------------------- CALL / CHECK ------------------
    if action == "call" or (action == "raise" and amount <= player.bet_this_street + player.stack
                            and amount < game.current_bet):
        # Normal call or “all‑in call masquerading as raise”
        if to_call == 0:
            game.hand_history.append(f"{player.name} checks")
            game.last_actions[player_pos] = ("cc", 0)
        else:
            call_amt = min(to_call, player.stack)
            _commit(call_amt)
            tag = "calls" if call_amt else "checks"
            suffix = " and is all-in" if player.is_all_in else ""
            game.hand_history.append(f"{player.name} {tag} ${call_amt}{suffix}")
            game.last_actions[player_pos] = ("cc", call_amt)
        find_next_player()
        return

    # -------------------------- RAISE --------------------------
    if action == "raise":
        # What total stack‐commit does the player ask for?
        # (cap at everything they own, so an “oversized” request becomes all‑in)
        raise_total = min(amount, player.stack + player.bet_this_street)

        # ── 1️⃣  Compute the minimum legal raise total  ───────────────────────
        # A raise must increase the bet by **at least**
        #   • the last raise increment, OR
        #   • the big blind if this is the first raise of the street.
        min_increment = game.bb_amount
        min_total     = game.current_bet + min_increment

        # ── 2️⃣  Too small?  Treat it as a call (or all‑in call). ────────────
        if raise_total < min_total:
            handle_player_action("call")      # recurse once through the clean path
            return

        # ── 3️⃣  Legal raise – move chips and update state  ─────────────────
        raise_increment = raise_total - game.current_bet   # ≥ min_increment
        add_chips       = raise_total - player.bet_this_street
        _commit(add_chips)

        game.current_bet          = raise_total
        game.last_raise_increment = raise_increment        # **before** next raise test
        game.max_bet_seen         = max(game.max_bet_seen, raise_total)
        game.aggressor_pos        = player_pos
        if game.first_aggressor == "none":
            game.first_aggressor  = player.name

        game.hand_history.append(
            f"{player.name} raises to ${raise_total}"
            f"{' and is all-in' if player.is_all_in else ''}"
        )
        game.last_actions[player_pos] = ("cbr", raise_total)
        find_next_player()
        return

def is_action_closed():
    """Returns True if all non-folded players have matched the current bet or are all-in."""
    for p in game.players:
        if not p.folded and not p.is_all_in and p.bet_this_street < game.current_bet:
            return False
    return True



import base64
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

@st.cache_data
def img_to_data_uri(path: str) -> str:
    b = Path(path).read_bytes()
    ext = Path(path).suffix.lower().lstrip(".") or "png"
    return f"data:image/{ext};base64," + base64.b64encode(b).decode("ascii")
    
SEAT_POS_6MAX = {
    0: (50, 75),  # Hero bottom-center
    1: (18, 70),
    2: (18, 30),
    3: (50, 12),
    4: (82, 30),
    5: (82, 70),
}

def player_bet_amount(p) -> int:
    for attr in ('amount'):
      
        if hasattr(p, attr):
            v = getattr(p, attr) or 0
            return int(v)
    return 0


def chip_badge_html(chips_uri: str, amount: int, anchor: str = "right") -> str:
    """
    anchor: "right" (default) or "left" or "top" etc. (see positioning below)
    """
  
    if amount <= 0:
        return ""

    # Position relative to the seat box
    if anchor == "right":
        pos = "right:-8px; top:50%; transform:translate(100%,-50%);"
    elif anchor == "left":
        pos = "left:-8px; top:50%; transform:translate(-100%,-50%);"
    elif anchor == "bottom":
        pos = "left:50%; bottom:-8px; transform:translate(-50%,100%);"
    else:
        pos = "right:-8px; top:50%; transform:translate(100%,-50%);"  # fallback

    return f"""
    <div class="bet" style="
        position:absolute;
        {pos}
        display:flex;
        align-items:center;
        gap:6px;
        padding:6px 10px;
        background:rgba(0,0,0,0.72);
        border:2px solid rgba(255,215,0,0.9);
        border-radius:999px;
        box-shadow:0 8px 18px rgba(0,0,0,0.6);
        z-index:4;
        white-space:nowrap;
    ">
        <img src="{chips_uri}" style="width:22px; height:22px; object-fit:contain; display:block;" />
        <span style="color:#ffd700; font-weight:800; font-size:13px;">${amount}</span>
    </div>
    """

def render_player_table(game, table_img_path: str, card_img_path_fn, chip_img_path, seat_pos=SEAT_POS_6MAX):
    table_uri = img_to_data_uri(table_img_path)
    chips_uri = img_to_data_uri(chip_img_path)

    seat_divs = []
    for i, p in enumerate(game.players):
        # 1. Get the current bet amount
        # Ensure this attribute exists in your Player class
        bet_amt = getattr(p, 'current_bet', 0) 
        
        # 2. Determine seat coordinates
        x, y = seat_pos.get(i, (50, 50))

        # 3. Dynamic Chip Anchoring: 
        # If player is on the right half of the table, anchor chips to their left (inward)
        # If player is on the left half, anchor chips to their right (inward)
        anchor_pos = "left" if x > 50 else "right"
        
        # Generate the chip badge HTML
       

        # 4. Player State Logic
        is_next = (i == game.next_to_act_pos) and (not st.session_state.game_over)
        border_color = "#ff4b4b" if is_next else "rgba(255,255,255,0.2)"
        btn_label = " (BTN)" if i == game.button_pos else ""

        status_bits = []
        if p.folded: status_bits.append("Folded")
        if p.is_all_in: status_bits.append("All-in")
        
        status_html = ""
        if status_bits:
            status_html = f"<div style='margin-top:4px; color:#f0c36d; font-size:12px; font-weight:600;'>{' '.join(status_bits)}</div>"

        # 5. Hole Card Visibility Logic
        show_hole = False
        if p.hole:
            if p.name == "Hero":
                show_hole = True
            elif st.session_state.went_to_showdown and (not p.folded):
                    show_hole = True
                    
                    
        card_imgs = ""
        if show_hole:
            uris = [img_to_data_uri(card_img_path_fn(c)) for c in p.hole]
            card_imgs = "".join([
                f"""
                <span style="display:inline-block; background:#fff; border-radius:4px; padding:1px; 
                             box-shadow:0 4px 10px rgba(0,0,0,0.5); margin:2px;">
                    <img src="{u}" style="width:60px; display:block; border-radius:3px;"/>
                </span>
                """ for u in uris
            ])

   
        
      
        # 6. Construct the Seat Div
        # Added 'overflow: visible' to ensure the absolute-positioned bet_html isn't clipped

        bet_amt = p.bet_this_street
        bet_html = chip_badge_html(chips_uri, bet_amt, anchor=anchor_pos)

        seat_divs.append(f"""
            <div class="seat" style="
                position:absolute;
                left:{x}%;
                top:{y}%;
                transform:translate(-50%,-50%);
                width:160px;
                padding:12px;
                border:2px solid {border_color};
                border-radius:12px;
                background:rgba(30,30,30,0.85);
                color:white;
                text-align:center;
                font-family: system-ui, -apple-system, sans-serif;
                z-index: 2;
                overflow: visible;
                box-shadow: 0 10px 25px rgba(0,0,0,0.5);
            ">
                {bet_html}
                <div style="font-weight:700; font-size:15px; margin-bottom:2px;">{p.name}{btn_label}</div>
                <div style="font-size:13px; opacity:0.9;">Stack: ${p.stack}</div>
                {status_html}
                <div style="margin-top:8px; white-space:nowrap; min-height:40px;">{card_imgs}</div>
            </div>
        """)


    # 7. Final Assembly
    pot_amount = sum(getattr(p, 'amount', 0) for p in game.pots)

    
    
    pot_html = f"""
    <div class="bet" style="
        position:absolute;
        left:50%;
        top:42%;
        transform:translate(-50%,-50%);
        padding:10px 20px;
        background:rgba(0,0,0,0.7);
        border:2px solid #ffd700;
        border-radius:12px;
        color:#ffd700;
        font-weight:800;
        font-size:20px;
        box-shadow:0 8px 20px rgba(0,0,0,0.7);
        text-align:center;
        z-index:3;
    ">
        <span style="font-size:12px; text-transform:uppercase; display:block; opacity:0.8;">Total Pot</span>
        ${pot_amount}
    </div>
    """

    board_html = ""
    if getattr(game, "board", None):
        board_uris = [img_to_data_uri(card_img_path_fn(c)) for c in game.board]
        board_cards = "".join(
            f"""
            <span style="display:inline-block; background:#fff; border-radius:6px; padding:2px;
                        box-shadow:0 6px 16px rgba(0,0,0,0.55);">
                <img src="{u}" style="width:68px; display:block; border-radius:4px;" />
            </span>
            """
            for u in board_uris
        )

        board_html = f"""
        <div style="
            position:absolute;
            left:50%;
            top:55%;
            transform:translate(-50%,-50%);
            display:flex;
            gap:10px;
            z-index:3;
            align-items:center;
            justify-content:center;
            pointer-events:none;
        ">
            {board_cards}
        </div>
        """





    full_html = f"""
                <style>
                    .seat {{ transition: all 0.2s ease-in-out; }}
                    .bet {{ transition: opacity 0.3s ease; }}
                </style>
                <div style="position:relative; width:100%; max-width:1400; aspect-ratio: 16 / 10; margin: 0 auto; overflow:visible; border-radius:18px;">
                    <img src="{table_uri}" style="position:absolute; inset:0; width:100%; height:100%; object-fit:cover; z-index:0;" />
                    {pot_html}
                    {board_html}
                    {''.join(seat_divs)}
                </div>
            """




    components.html(full_html, height=600, scrolling=False)



def find_next_player():
    """
    Finds the next player to act or advances the street if the betting round is over.
    """


    live_players = game.get_live_players()
    if len(live_players) <= 1:
        end_hand()
        return

    # Determine who the next player to act would be.
    # Start searching from the player after the one who just acted.
    next_pos = (game.next_to_act_pos + 1) % len(game.players)

    # Loop a full circle to find the next player who is not folded or all-in.
    for _ in range(len(game.players)):
        player = game.players[next_pos]
        if not player.folded and not player.is_all_in:
            # Found the next potential actor.
            break
        next_pos = (next_pos + 1) % len(game.players)
    else:
        # This 'else' triggers if the loop completes without a 'break',
        # meaning no players are left to act (they are all folded or all-in).
        start_next_street()
        return

    game.next_to_act_pos = next_pos

    # Check if all players who are still in the hand have settled their bets for this street.
    bets_are_settled = all(
    p.is_all_in or p.bet_this_street == game.current_bet
    for p in live_players
    )


    if bets_are_settled:

        # ----- CASE 1: everyone checked -----
        if game.current_bet == 0:
            if next_pos == game.aggressor_pos:      # looped back round
                start_next_street()
            return

        # ----- CASE 2: pre‑flop limp pot — give BB its option once -----
        # in find_next_player(), replace CASE 2 with:
        if (game.street == 'preflop'
            and game.current_bet == game.bb_amount
            and next_pos == game.aggressor_pos):        # next actor IS the blind
            if getattr(game, "bb_option_done", False):  # option already given → close round
                start_next_street()
            else:                                       # first time we reach here
                game.bb_option_done = True              # mark it and let BB act once
            return
                                 # let BB act now

        # ----- CASE 3: any other settlement -----
        start_next_street()
        return


    # If the betting round is not over, update the game state to the next player.
    game.next_to_act_pos = next_pos


def start_next_street():
    """Manages the transition between streets."""
    game.advance_street()
    for p in game.players:          # or keep a `bots` list
        if hasattr(p, "update_stats"):
            p.update_stats(game)    # <-- HERE
    if game.street == "showdown":
        end_hand()

def end_hand():
    """Ends the hand, calculates winners, and distributes pots with side-pot logic."""
    st.session_state.game_over = True
    live_idxs = [i for i, p in enumerate(game.players) if not p.folded]

    st.session_state.went_to_showdown = (len(live_idxs) > 1)

    # ---------- 1) Short-circuit if only one live player ----------
    if len(live_idxs) == 1:
        sole = game.players[live_idxs[0]]
        sole.stack += sum(p.amount for p in game.pots)
        st.session_state.winner_info = f"{sole.name} wins ${sum(p.amount for p in game.pots)}."
        return

    # ---------- 2) Rank hands of all live players ----------
    hand_info = []
    for i in live_idxs:
        p = game.players[i]
        rank, kickers, best = get_best_hand(p.hole + game.board)
        hand_info.append({ "idx": i, "rank": rank, "kickers": kickers,
                           "name": p.name, "best": best })

    # sort best → worst once; we'll reuse
    hand_info.sort(key=lambda x: (x["rank"], x["kickers"]), reverse=True)

    # ---------- 3) Build pots from contributions ----------
    contribs = {i: game.players[i].contributed for i in live_idxs}
    # distinct contribution levels, ascending
    levels = sorted(set(contribs.values()))
    prev = 0
    pots: list[tuple[int, list[int]]] = []   # [(amount, eligible_idxs), …]

    for lvl in levels:
        takers = [i for i, c in contribs.items() if c >= lvl]
        pot_amt = (lvl - prev) * len(takers)
        if pot_amt:
            pots.append((pot_amt, takers))
        prev = lvl

    # ---------- 4) Award each pot ----------
    summary_lines = []
    for pot_amt, takers in pots:
        # best hand **among takers only**
        best_rank = max(
                    (h for h in hand_info if h["idx"] in takers),
                    key=lambda h: (h["rank"], h["kickers"])
                )
        winners = [h for h in hand_info if h["idx"] in takers
                                     and (h["rank"], h["kickers"]) ==
                                         (best_rank["rank"], best_rank["kickers"])]

        share = pot_amt // len(winners)
        for w in winners:
            game.players[w["idx"]].stack += share

        winner_names = ", ".join(w["name"] for w in winners)
        summary_lines.append(f"{winner_names} win ${pot_amt} pot ({HAND_CATEGORIES[best_rank['rank']]})")


    st.session_state.winner_info = "; ".join(summary_lines)


# --- UI Layout ---
# --- UI Layout ---
st.title("Poker Game: You vs. 5 Bots")

_ensure_game_in_session()

game: GameState = st.session_state.game

with st.sidebar:
    st.header("Game Controls")

    if "reveal_bot_cards" not in st.session_state:
        st.session_state.reveal_bot_cards = bool(getattr(game, "reveal_bot_hole_cards", False))

    #st.session_state.reveal_bot_cards = st.checkbox(
    #    "Reveal bot cards after hand",
    #    value=st.session_state.reveal_bot_cards,
    #    key="sidebar_reveal_bot_cards",
    #)
    game.reveal_bot_hole_cards = st.session_state.reveal_bot_cards


    # Keep the button in the sidebar and give it a key
    if st.button("New Game (reset stacks)", key="sidebar_new_game"):
        # (Re)build bots
        st.session_state.bots = [TransformerBot("augment_poker_transformer.pt") for _ in range(5)]
        for i, bot in enumerate(st.session_state.bots):
            bot.name += str(i)

        bot_names = [b.name for b in st.session_state.bots]

        # Reset game + players
        game.reset_game()
        game.bot_names = bot_names
        game.players = [Player(game.players[0].name, game.players[0].stack, False)] + [
            Player(n, game.players[1].stack, True) for n in bot_names
        ]

        game.start_new_hand()

        # Session flags
        st.session_state.game_over = False
        st.session_state.winner_info = ""
        st.session_state.went_to_showdown = False

        st.rerun()

    st.header("Hand History")
    st.code("\n".join(game.hand_history[-50:]), height=300)
   
    
    # Disclaimer under it
    with st.container(border=True):
        st.caption("Rules disclaimer (this app)")
    
        st.markdown(
    '''
    **Blind Increases**
    - The blinds increase every 7 hands by 50%. 
    
    **Minimum raise (no-limit)**
    - A legal minimum raise must increase the bet by at least the **last raise size** (BB is considered first raise) or all-in. 
    '''
        )

    

c1, c2 = st.columns([3, 1])
st.set_page_config(layout="wide")

with c1:
    pot_size = sum(p.amount for p in game.pots)
    
    #if game.board:
     # st.image([card_img(c) for c in game.board], width=72)
    #else:
     #   st.subheader("Board: Pre-flop")

    if st.session_state.game_over:
        st.success(st.session_state.winner_info)

    #cols = st.columns(len(game.players))

    def card_img_path(code: str) -> str:
      return f"card_imgs/{code}.png"

    render_player_table(game, table_img_path="poker_table_clip_art.jpg", card_img_path_fn=card_img_path,chip_img_path='poker_chip_clip_art.jpg')
    if st.session_state.went_to_showdown:
        st.session_state.went_to_showdown = False

with c2:

    st.header("Your Actions")
    hero = game.players[0]
    is_hero_turn = (game.next_to_act_pos == 0) and not st.session_state.game_over and not hero.is_all_in
    amount_to_call = game.current_bet - hero.bet_this_street

    with st.expander("Actions", expanded=True):
        if st.button("Fold", disabled=not is_hero_turn or amount_to_call == 0, use_container_width=True):
            handle_player_action('fold')
            st.rerun()

        call_label = "Check" if amount_to_call == 0 else f"Call ${amount_to_call}"
        if st.button(call_label, disabled=not is_hero_turn, use_container_width=True):
            handle_player_action('call')
            st.rerun()

        # --- inside the "Your Actions" block -------------------
        if hero.stack > amount_to_call and is_hero_turn:
            min_raise = game.current_bet + game.bb_amount
            max_raise = hero.stack + hero.bet_this_street          # all-in cap

            if min_raise < max_raise:          # normal raise range
                slider_key = f"raise_to_{game.bb_amount}_{game.current_bet}"
                raise_amount = st.slider(
                    "Raise To:",
                    min_value=min_raise,
                    max_value=max_raise,
                    value=min_raise,
                    step=game.bb_amount,
                    key=slider_key,            # 👈 new key whenever BB/current_bet changes
                )
                btn_label = "Raise"
            else:                              # only shove is legal
                raise_amount = max_raise
                btn_label = "All-in"

            if st.button(btn_label, use_container_width=True):
                handle_player_action("raise", raise_amount)
                st.rerun()

            with st.container(border=True):
                if st.button("Start new hand", use_container_width=True):
                    start_new_hand()  # whatever your function is
                    st.rerun()

# Bot Action Logic
if not st.session_state.game_over and game.players[game.next_to_act_pos].is_bot:
    bot_pos = game.next_to_act_pos
    action, amount = bots[bot_pos - 1].get_action(game, bot_pos)

    handle_player_action(action, amount)
    st.rerun()

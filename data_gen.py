from typing import List, Tuple, Dict, Any
import itertools
import pandas as pd
import numpy as np

# --- CONFIGURATION ---
PRE_FLOP_ACTIONS = ['fold', 'call', '2xbb', '4xbb', 'allin']
POST_FLOP_ACTIONS = ['fold', 'call', '0.5xpot', '1xpot', '2xpot', 'allin']

rank_map = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6,
            '7': 7, '8': 8, '9': 9, 'T': 10,
            'J': 11, 'Q': 12, 'K': 13, 'A': 14}
suit_map = {'c': 0, 'd': 1, 'h': 2, 's': 3}

# --- PARSERS ---
def encode_card(card_str):
    return rank_map[card_str[0]], suit_map[card_str[1]]

def encode_hand(hand_str):
    return encode_card(hand_str[:2]) + encode_card(hand_str[2:])

def parse_phh_file(filepath):
    data = {}
    with open(filepath, 'r') as f:
        for line in f:
            if '=' in line:
                key, val = line.strip().split('=', 1)
                try:
                    data[key.strip()] = ast.literal_eval(val.strip())
                except:
                    data[key.strip()] = val.strip()
    return data

def determine_phase(action_idx, actions):
    board_card_actions = [i for i, a in enumerate(actions) if a.startswith('d db')]
    if not board_card_actions:
        return 'preflop'
    elif action_idx < board_card_actions[0]:
        return 'preflop'
    elif len(board_card_actions) == 1 or action_idx < board_card_actions[1]:
        return 'flop'
    elif len(board_card_actions) == 2 or action_idx < board_card_actions[2]:
        return 'turn'
    else:
        return 'river'

def parse_phh_string(file_content):
    data = {}
    for line in file_content.splitlines():
        if '=' in line:
            key, val = line.strip().split('=', 1)
            try:
                data[key.strip()] = ast.literal_eval(val.strip())
            except:
                data[key.strip()] = val.strip()
    return data

def compute_starting_hand_features(r1: int, r2: int, is_suited: int | bool) -> dict:
    """
    Return a dict with:
      - hole_combo       (e.g., 'AKs', 'T7o', '22')
      - hole_combo_score (float; monotonic strength)
      - hole_combo_id169 (int 1..169; 1 = strongest)

    r1, r2 are ranks in [2..14] where 14 = Ace.
    is_suited can be bool/int; pairs are treated as offsuit by definition.
    """
    # ----- one-time cache on the function object -----
    if not hasattr(compute_starting_hand_features, "_cache"):
        # rank -> char map
        R = {14:"A", 13:"K", 12:"Q", 11:"J", 10:"T",
              9:"9",  8:"8",  7:"7",  6:"6",  5:"5", 4:"4", 3:"3", 2:"2"}

        def score(hi: int, lo: int, suited: int) -> float:
            # Pairs dominate; non-pairs: suited bonus, gap penalty
            if hi == lo:
                return 100.0 + 2.0 * hi
            gap = (hi - lo) - 1
            gap_penalty = 2.0 * max(0, gap)
            return 10.0 * suited + 2.0 * hi + lo - gap_penalty

        # Build all 169 canonical hands
        hands = []
        for hi in range(14, 1, -1):        # A..2
            for lo in range(hi, 1, -1):    # hi >= lo
                if hi == lo:
                    hands.append((hi, lo, 0))  # pairs (no suited variant)
                else:
                    hands.append((hi, lo, 1))  # suited
                    hands.append((hi, lo, 0))  # offsuit

        # Sort strongest first and index 1..169
        hands_sorted = sorted(hands, key=lambda t: score(*t), reverse=True)
        idx169 = {t: i + 1 for i, t in enumerate(hands_sorted)}

        compute_starting_hand_features._cache = {
            "R": R,
            "score": score,
            "idx169": idx169,
        }

    R      = compute_starting_hand_features._cache["R"]
    score  = compute_starting_hand_features._cache["score"]
    idx169 = compute_starting_hand_features._cache["idx169"]

    # ----- canonicalize the two cards -----
    hi, lo = (r1, r2) if r1 >= r2 else (r2, r1)
    suited = int(bool(is_suited) and hi != lo)  # pairs can't be suited

    # Pretty label
    if hi == lo:
        label = f"{R[hi]}{R[lo]}"
    else:
        label = f"{R[hi]}{R[lo]}{'s' if suited else 'o'}"

    # Strength & 169 id
    s = float(score(hi, lo, suited))
    i169 = int(idx169[(hi, lo, suited)])

    return {
        "hole_combo": label,
        "hole_combo_score": s,
        "hole_combo_id169": i169,
    }


import zipfile
import pandas as pd
from typing import List, Tuple, Dict, Any

# Assume extract_pluribus_actions is defined elsewhere
# from data_gen import extract_pluribus_actions

def process_phh_zip(zip_path, regression=False, exclude: List[str] = [], profit=False):
    preflop_data = []
    postflop_data = []

    with zipfile.ZipFile(zip_path, 'r') as zipf:
        for name in zipf.namelist():
            if name.endswith('.phh'):
                try:
                    content = zipf.read(name).decode('utf-8')
                    # Parse the PHH hand string
                    hand = parse_phh_string(content)

                    # Pass the file name to the extraction function
                    pre, post = extract_pluribus_actions(
                        hand,
                        regression_target=regression,
                        exclude_actions=exclude,
                        profit=profit,
                        file_path=name # Pass the file path here
                    )

                    preflop_data.extend(pre)
                    postflop_data.extend(post)
                except Exception as e:
                    print(f"⚠️ Failed on {name}: {e}")

    pre_df = pd.DataFrame(preflop_data)
    post_df = pd.DataFrame(postflop_data)

    return pre_df[~pre_df['pluribus_action'].isin(exclude)], post_df[~post_df['pluribus_action'].isin(exclude)]

# --- MAIN ENTRYPOINT ---
def process_phh_folder(root_dir):
    preflop_data = []
    postflop_data = []
    for subdir, _, files in os.walk(root_dir):
        for fname in files:
            if fname.endswith('.phh') or fname.endswith('.txt'):
                fpath = os.path.join(subdir, fname)
                try:
                    hand = parse_phh_file(fpath)
                    pre, post = extract_pluribus_actions(hand,include_card_onehots=True)
                    preflop_data.extend(pre)
                    postflop_data.extend(post)
                except Exception as e:
                    print(f"⚠️ Failed on {fpath}: {e}")
    return pd.DataFrame(preflop_data), pd.DataFrame(postflop_data)


def print_phh_files_from_zip(zip_path, max_files=300):
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        count = 0
        for name in zipf.namelist():
            if name.endswith('.phh'):
                try:
                    content = zipf.read(name).decode('utf-8')
                    print(f"--- {name} ---")
                    print(content)
                    print("\n" + "="*50 + "\n")
                    count += 1
                    if count >= max_files:
                        break
                except Exception as e:
                    print(f"⚠️ Failed reading {name}: {e}")

CATEGORY_ORDER = [
    "high_card", "pair", "two_pair", "trips", "straight",
    "flush", "full_house", "four_kind", "straight_flush", "royal_flush"
]

def get_seat_tag(players: List[str], player_name: str = "Pluribus") -> str:
    try:
        return f"p{players.index(player_name) + 1}"
    except ValueError as exc:
        raise ValueError(
            f"{player_name!r} not found in players list: {players}") from exc

def encode_card(card: str) -> Tuple[int, int]:
    rank_map = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
                'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
    suit_map = {'c': 0, 'd': 1, 'h': 2, 's': 3}
    rank = rank_map.get(card[0].upper(), 0)
    suit = suit_map.get(card[1].lower(), 0)
    return rank, suit

# ====== NEW HELPERS (paste above extract_pluribus_actions) ===================
from typing import Optional, Tuple, List, Dict

def _pos_map(players: List[str], blinds: List[int]) -> Dict[int, str]:
    """Seat index -> position label for a 6-max table.
    Re-uses the same logic you already use for Pluribus."""
    # SB seat: smallest positive blind value
    sb_val = min(b for b in blinds if b > 0)
    sb_seat = next(i for i, v in enumerate(blinds) if v == sb_val)
    btn_seat = (sb_seat - 1) % 6
    order = ["BTN", "SB", "BB", "UTG", "MP", "CO"]
    return {((btn_seat + i) % 6): pos for i, pos in enumerate(order)}

def _pos_of_pid(pid: str, players: List[str], blinds: List[int]) -> Optional[str]:
    try:
        seat = int(pid[1:]) - 1
        return _pos_map(players, blinds).get(seat)
    except Exception:
        return None

def _hero_checked_prev_street(past: List[str], hero_tag: str, cur_street: str) -> int:
    """Did hero check (i.e., 'cc' with no amount) on the previous street?"""
    prev = {"flop": "preflop", "turn": "flop", "river": "turn"}.get(cur_street)
    if prev is None:
        return 0
    # scan from end backwards for the last action on prev street
    last = None
    for i in range(len(past) - 1, -1, -1):
        if determine_phase(i, past) != prev:
            if last is not None:
                break
            else:
                continue
        a = past[i]
        if not a.startswith(hero_tag + " "):
            continue
        parts = a.split()
        typ = parts[1] if len(parts) > 1 else ""
        amt = (int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else None)
        last = (typ, amt)
        break
    if last is None:
        return 0
    typ, amt = last
    return int(typ == "cc" and (amt is None))

def _first_raiser_this_street(actions_this_street_p: List[str]) -> Optional[str]:
    for a in actions_this_street_p:
        if a.split()[1] == "cbr":
            return a.split()[0]
    return None

def _last_raiser_this_street(actions_this_street_p: List[str]) -> Optional[str]:
    for a in reversed(actions_this_street_p):
        if a.split()[1] == "cbr":
            return a.split()[0]
    return None

def _distance_between(ordered_pids: List[str], a: str, b: str) -> int:
    """Absolute index distance minus 1 (players in between); returns 0 if adjacent or not found."""
    if a not in ordered_pids or b not in ordered_pids:
        return 0
    ia, ib = ordered_pids.index(a), ordered_pids.index(b)
    return max(0, abs(ia - ib) - 1)

def _closest_raiser_distance(actions_this_street_p: List[str], live: set, hero_tag: str) -> int:
    """Distance in the (lexicographically) ordered live list to the nearest raiser that has acted this street."""
    ordered = sorted(live)  # consistent with your other helpers
    raisers = [a.split()[0] for a in actions_this_street_p if a.split()[1] == "cbr"]
    if not raisers:
        return 0
    return min(_distance_between(ordered, hero_tag, r) for r in raisers)

def _heads_up_vs_blind_flags(live: set, players: List[str], blinds: List[int], hero_tag: str) -> Tuple[int, int]:
    """If only two players are live, is the other one SB or BB?"""
    if len(live) != 2 or hero_tag not in live:
        return 0, 0
    other = next(p for p in live if p != hero_tag)
    pos_other = _pos_of_pid(other, players, blinds)
    return int(pos_other == "SB"), int(pos_other == "BB")

def _has_flush_blocker_any(s1: int, s2: int, board_suits: List[int]) -> int:
    """True if board is at least two-suited on some suit and hero holds that suit."""
    if not board_suits:
        return 0
    from collections import Counter
    counts = Counter(board_suits)
    cand = [s for s, c in counts.items() if c >= 2]  # 2+ on board (e.g., two-tone flop or more)
    return int(any(s in cand for s in (s1, s2)))

def _hero_action_counts_so_far(past: List[str], hero_tag: str) -> Tuple[int, int]:
    """(calls_or_checks_count, raises_count) for hero up to now in the hand."""
    calls = raises = 0
    for a in past:
        if not a.startswith(hero_tag + " "):
            continue
        parts = a.split()
        if len(parts) < 2:
            continue
        if parts[1] == "cc":
            calls += 1
        elif parts[1] == "cbr":
            raises += 1
    return calls, raises

def _aggression_switched_from_prev_street(past: List[str], cur_street: str) -> int:
    """Did the first bettor on this street differ from the last aggressor on the previous street?"""
    prev = {"flop": "preflop", "turn": "flop", "river": "turn"}.get(cur_street)
    if prev is None:
        return 0
    prev_aggs = [a.split()[0] for i, a in enumerate(past)
                 if determine_phase(i, past) == prev and a.startswith("p") and a.split()[1] == "cbr"]
    last_prev = prev_aggs[-1] if prev_aggs else None
    cur_first = _first_raiser_this_street([a for i, a in enumerate(past) if determine_phase(i, past) == cur_street and a.startswith("p")])
    if cur_first is None or last_prev is None:
        return 0
    return int(cur_first != last_prev)
# =============================================================================


def encode_hand(hand_str: str) -> Tuple[int, int, int, int]:
    card1_str = hand_str[:2]
    card2_str = hand_str[2:]
    r1, s1 = encode_card(card1_str)
    r2, s2 = encode_card(card2_str)
    return r1, s1, r2, s2



def get_pluribus_hand_profit(hand_dict: dict) -> float:
    """
    Compute Pluribus's profit (or loss) for a hand.
    Returns profit in chips (can be negative).
    """
    players = hand_dict.get("players", [])
    stacks_start = hand_dict.get("starting_stacks", [])
    stacks_end = hand_dict.get("finishing_stacks", [])
    pluribus_tag = get_seat_tag(players)
    p_idx = int(pluribus_tag[1:]) - 1

    if len(stacks_start) != len(stacks_end):
        raise ValueError("Mismatch in starting and ending stack lengths")

    profit = stacks_end[p_idx] - stacks_start[p_idx]
    return profit


def get_position_for_pluribus(players: List[str], blinds: List[int],name) -> str:
    pluribus_seat = players.index(name)
    sb_value = min(b for b in blinds if b > 0)
    sb = next(i for i, v in enumerate(blinds) if v == sb_value)
    btn = (sb - 1) % 6
    position_map = {((btn + i) % 6): p for i, p in enumerate(["BTN", "SB", "BB", "UTG", "MP", "CO"])}
    return position_map[pluribus_seat]

def get_relative_position(past: List[str],
                          phase: str,
                          live: set[str],
                          pluribus_tag: str) -> int:
    acted: set[str] = set()
    for i in range(len(past) - 1, -1, -1):
        a = past[i]
        if determine_phase(i, past) != phase:
            break
        acted.add(a.split()[0])
    remaining = [p for p in sorted(live) if p not in acted]
    return remaining.index(pluribus_tag) + 1 if pluribus_tag in remaining else -1

_SUITS = [0, 1, 2, 3]
_RANKS = list(range(2, 15))

def _split_cards(token: str) -> List[str]:
    return [token[i:i + 2] for i in range(0, len(token), 2)]

def _card_one_hot(card: str, prefix: str) -> Dict[str, int]:
    rk, st = encode_card(card)
    out = {}
    for r in _RANKS:
        out[f"{prefix}_rank_{r}"] = int(rk == r)
    for s in _SUITS:
        out[f"{prefix}_suit_{s}"] = int(st == s)
    return out

def _blank_card_one_hot(prefix: str) -> Dict[str, int]:
    out = {}
    for r in _RANKS:
        out[f"{prefix}_rank_{r}"] = 0
    for s in _SUITS:
        out[f"{prefix}_suit_{s}"] = 0
    return out

def _is_straight(unique_ranks: List[int]) -> int | None:
    ranks = unique_ranks[:]
    if 14 in ranks:
        ranks.insert(0, 1)
    for i in range(len(ranks) - 4):
        if ranks[i:i + 5] == list(range(ranks[i], ranks[i] + 5)):
            return ranks[i + 4]
    return None

def evaluate_hand_category(card_strs: List[str]) -> str:
    if len(card_strs) < 2:
        return "high_card"
    if len(card_strs) < 5:
        ranks_only = [encode_card(c)[0] for c in card_strs]
        if len(ranks_only) == 2 and ranks_only[0] == ranks_only[1]:
            return "pair"
        return "high_card"
    ranks, suits = zip(*(encode_card(c) for c in card_strs))
    rank_counts = {r: ranks.count(r) for r in set(ranks)}
    suit_to_cards = {}
    for r, s in zip(ranks, suits):
        suit_to_cards.setdefault(s, []).append(r)
    flush_suit = next((s for s, cs in suit_to_cards.items() if len(cs) >= 5), None)
    all_unique_sorted = sorted(list(set(ranks)))
    straight_high = _is_straight(all_unique_sorted)
    flush_straight_high = None
    if flush_suit is not None:
        flush_ranks = sorted(list(set(suit_to_cards[flush_suit])))
        flush_straight_high = _is_straight(flush_ranks)
    if flush_straight_high is not None:
        if flush_straight_high == 14 and set([10, 11, 12, 13, 14]).issubset(flush_ranks):
            return "royal_flush"
        return "straight_flush"
    if 4 in rank_counts.values(): return "four_kind"
    if 3 in rank_counts.values() and 2 in rank_counts.values(): return "full_house"
    if flush_suit is not None: return "flush"
    if straight_high is not None: return "straight"
    if 3 in rank_counts.values(): return "trips"
    pairs = list(rank_counts.values()).count(2)
    if pairs >= 2: return "two_pair"
    if pairs == 1: return "pair"
    return "high_card"


from typing import List, Dict

def detect_draws(card_strs: List[str]) -> Dict[str, int]:
    # encode_card(c) should return (rank:int, suit:str)
    ranks, suits = (zip(*(encode_card(c) for c in card_strs)) if card_strs else ([], []))

    suit_counts = {s: suits.count(s) for s in set(suits)}

    # --- Start Replacement ---
    flush_draw = 0  # Default to false
    if suit_counts:
        # Check the highest number of cards in any single suit
        max_cards_in_suit = max(suit_counts.values())
        # A flush draw exists only if there are exactly 4 cards of the same suit
        if max_cards_in_suit == 4:
            flush_draw = 1
    # --- End Replacement ---

    uniq_ranks = sorted(list(set(ranks)))
    if 14 in uniq_ranks:
        uniq_ranks.insert(0, 1)

    oesd = 0
    for i in range(len(uniq_ranks) - 3):
        window = uniq_ranks[i:i + 4]
        if window == list(range(window[0], window[0] + 4)):
            low_end = window[0] - 1
            high_end = window[3] + 1
            if (low_end >= 2 and low_end not in uniq_ranks) or \
               (high_end <= 14 and high_end not in uniq_ranks):
                oesd = 1
            break

    return {"has_flush_draw": flush_draw, "has_open_ended_straight_draw": oesd}



def remaining_to_act(past: List[str], phase: str, live: set[str]) -> List[str]:
    """Return players who still have to act on *this street*, in order (sorted tag order).
    Mirrors the logic in get_relative_position but returns the list.
    """
    acted: set[str] = set()
    for i in range(len(past) - 1, -1, -1):
        a = past[i]
        if determine_phase(i, past) != phase:
            break
        acted.add(a.split()[0])
    return [p for p in sorted(live) if p not in acted]


# --- Drop‑in helpers to paste ABOVE `extract_pluribus_actions` -----------------
from typing import List, Tuple, Dict, Any
import itertools

# Street slice for actions before a given index (hero decision)
def _street_slice_before(all_actions: List[str], upto_idx: int) -> Tuple[str, int, List[str]]:
    phase = determine_phase(upto_idx, all_actions)
    start = upto_idx
    while start - 1 >= 0 and determine_phase(start - 1, all_actions) == phase:
        start -= 1
    # Only keep player actions (skip deal markers) for some counters
    return phase, start, all_actions[start:upto_idx]

# First preflop aggressor (“opener”) pid like 'p3'
def _get_preflop_opener(all_actions: List[str]) -> str | None:
    for i, a in enumerate(all_actions):
        if a.startswith("d db "):
            break  # flop reached ⇒ stop scanning
        if a.startswith("p"):
            parts = a.split()
            if len(parts) >= 2 and parts[1] == "cbr":
                return parts[0]
    return None

# Over/under cards vs board & pair class flags
_def_top2 = lambda ranks: sorted(ranks, reverse=True)[:2] if ranks else []

def _over_under_pair_features(hole_ranks: List[int], board_ranks: List[int]) -> Dict[str, int]:
    out: Dict[str, int] = {
        "overcards_to_board": 0,
        "undercards_to_board": 0,
        "makes_top_pair": 0,
        "makes_second_pair": 0,
    }
    if not board_ranks:
        return out
    top = max(board_ranks)
    sec = _def_top2(board_ranks)[1] if len(set(board_ranks)) >= 2 else None
    over = sum(1 for r in hole_ranks if r > top)
    under = sum(1 for r in hole_ranks if r < top)
    makes_top = int(any(r == top for r in hole_ranks))
    makes_second = int(sec is not None and any(r == sec for r in hole_ranks))
    out.update({
        "overcards_to_board": over,
        "undercards_to_board": under,
        "makes_top_pair": makes_top,
        "makes_second_pair": makes_second,
    })
    return out

# Gutshot + backdoor (approximate, leakage‑safe, uses current street only)
from collections import Counter

def _has_gutshot(unique_ranks: List[int]) -> int:
    if not unique_ranks:
        return 0
    ranks = sorted(set(unique_ranks))
    if 14 in ranks:  # wheel handling
        ranks = [1] + ranks
    # A gutshot exists if any 5‑rank window contains exactly 4 distinct ranks
    # (not already OESD which your code flags separately)
    for low in range(1, 11):
        window = set(range(low, low + 5))
        if len(window.intersection(ranks)) == 4:
            return 1
    return 0


def _backdoor_flush_flags(r1: int, s1: int, r2: int, s2: int, board_cards: List[str]) -> Dict[str, int]:
    # Backdoor FD on flop only (two to come): hero has 2‑suited & board has exactly 1 of that suit
    if len(board_cards) < 3:
        return {"has_backdoor_flush": 0}
    hero_suited = int(s1 == s2)
    if not hero_suited:
        return {"has_backdoor_flush": 0}
    target = s1
    board_suits = [encode_card(c)[1] for c in board_cards[:3]]
    c = sum(1 for s in board_suits if s == target)
    return {"has_backdoor_flush": int(c == 1)}


def _backdoor_straight_flag(hole_ranks: List[int], board_ranks: List[int]) -> Dict[str, int]:
    # Backdoor straight draw (flop): there exists a 3‑card subset spanning <= 4 ranks
    if len(board_ranks) < 3:
        return {"has_backdoor_straight": 0}
    ranks = sorted(set(board_ranks[:3] + hole_ranks))
    if 14 in ranks:
        ranks = [1] + ranks
    for i in range(len(ranks) - 2):
        if ranks[i + 2] - ranks[i] <= 4:
            return {"has_backdoor_straight": 1}
    return {"has_backdoor_straight": 0}


# =====================
# Helper functions — paste ABOVE `extract_pluribus_actions`
# Dependencies: `determine_phase`, `encode_card` must already exist in your file.
# =====================
from typing import List, Tuple, Dict, Any, Optional
from collections import Counter
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Street/window helpers
# ──────────────────────────────────────────────────────────────────────────────
def _street_slice_before(all_actions: List[str], upto_idx: int) -> Tuple[str, int, List[str]]:
    """Return (phase, street_start_idx, actions_in_this_street_before_idx)."""
    phase = determine_phase(upto_idx, all_actions)
    start = upto_idx
    while start - 1 >= 0 and determine_phase(start - 1, all_actions) == phase:
        start -= 1
    return phase, start, all_actions[start:upto_idx]


def _actions_player_only(actions: List[str]) -> List[str]:
    return [a for a in actions if a.startswith("p")]  # skip deal markers


def _get_preflop_opener(all_actions: List[str]) -> Optional[str]:
    """First raiser before the flop (returns pid like 'p3')."""
    for a in all_actions:
        if a.startswith("d db "):  # first board deal ⇒ stop
            break
        if a.startswith("p") and a.split()[1] == "cbr":
            return a.split()[0]
    return None


def _last_agg_before(all_actions: List[str], upto_idx: int) -> Optional[str]:
    """Last aggressor pid before upto_idx."""
    for j in range(upto_idx - 1, -1, -1):
        a = all_actions[j]
        if a.startswith("p") and a.split()[1] == "cbr":
            return a.split()[0]
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Board + hole relations
# ──────────────────────────────────────────────────────────────────────────────
def _board_high_card_density(board_ranks: List[int]) -> int:
    """Count of broadway cards (T–A) on the board."""
    return sum(1 for r in board_ranks if r in {10, 11, 12, 13, 14})


def _paired_board_details(board_ranks: List[int]) -> Dict[str, int]:
    """Flags which pair (top/middle/bottom) is paired on the board; 0 if unpaired/trips."""
    if not board_ranks:
        return {"pair_is_top": 0, "pair_is_middle": 0, "pair_is_bottom": 0}
    rank_counts = Counter(board_ranks)
    if 2 not in rank_counts.values():
        return {"pair_is_top": 0, "pair_is_middle": 0, "pair_is_bottom": 0}
    paired_rank = max([r for r, c in rank_counts.items() if c == 2], default=None)
    if paired_rank is None:
        return {"pair_is_top": 0, "pair_is_middle": 0, "pair_is_bottom": 0}
    uniq_sorted = sorted(set(board_ranks), reverse=True)
    pos = uniq_sorted.index(paired_rank)
    return {
        "pair_is_top": int(pos == 0),
        "pair_is_middle": int(0 < pos < len(uniq_sorted) - 1),
        "pair_is_bottom": int(pos == len(uniq_sorted) - 1),
    }


def _trips_on_board(board_ranks: List[int]) -> int:
    return int(any(c >= 3 for c in Counter(board_ranks).values()))


def _max_suit_count_on_board(board_suits: List[int]) -> int:
    return max(Counter(board_suits).values()) if board_suits else 0


def _max_possible_straight_high(ranks_union: List[int]) -> Optional[int]:
    """High card of the highest *possible* 5‑card straight in ranks_union (A can be low)."""
    R = sorted(set(ranks_union))
    if 14 in R:
        R = sorted(set(R + [1]))  # wheel support
    best = None
    for i in range(len(R) - 4):
        window = R[i:i+5]
        if window[4] - window[0] == 4 and len(set(window)) == 5:
            best = window[-1]
    return best


def _flush_draw_suit(board_suits: List[int], s1: int, s2: int) -> Optional[int]:
    """If hero has a real flush draw, return the suit we're drawing to, else None.
    On flop: board_suits may have a suit count==2, hero contributes 2 of that suit.
    On turn: board suit count may be 3, hero contributes 1.
    """
    counts = Counter(board_suits)
    # Candidate suits:
    cand = [s for s, c in counts.items() if c in (2, 3)]
    for suit in cand:
        need = 4 - counts[suit]  # cards hero must have to be 4‑to‑flush
        have = int(s1 == suit) + int(s2 == suit)
        if have >= need and 1 <= need <= 2:
            return suit
    return None


def _nut_flush_draw_flag(has_fd: int, fd_suit: Optional[int], r1: int, s1: int, r2: int, s2: int) -> int:
    if not has_fd or fd_suit is None:
        return 0
    return int((s1 == fd_suit and r1 == 14) or (s2 == fd_suit and r2 == 14))


def _flush_draw_outs(has_fd: int) -> int:
    return 9 if has_fd else 0


def _combo_draw_flags(combined_best: str, *, has_fd: int, has_oesd: int, has_gutshot: int, overpair_flag: int, is_set: int) -> Dict[str, int]:
    sd = int(has_oesd or has_gutshot)
    trips = int(combined_best == "trips")
    return {
        "trip_plus_draw": int(trips and (has_fd or sd)),
        "overpair_plus_draw": int(overpair_flag and (has_fd or sd)),
        "set_plus_straight_draw": int(is_set and sd),
    }


def _players_between_hero_and_agg(remaining: List[str], hero_tag: str, agg_tag: Optional[str]) -> int:
    if agg_tag is None or hero_tag not in remaining or agg_tag not in remaining:
        return 0
    i_h = remaining.index(hero_tag)
    i_a = remaining.index(agg_tag)
    dist = abs(i_h - i_a) - 1
    return max(dist, 0)


def _effective_position_index(rel_pos: int, opponents: int) -> float:
    return (rel_pos / max(opponents, 1)) if opponents > 0 else 0.0


def _pot_multi_way_index(opponents: int, total_start: int) -> float:
    return opponents / max(total_start - 1, 1)


def _pot_odds_tier(pot_odds: float) -> str:
    return "cheap" if pot_odds < 0.2 else ("medium" if pot_odds <= 0.5 else "expensive")


def _implied_odds_proxy(pot_odds: float, spr: float) -> float:
    return float(pot_odds) * float(spr)


def _checks_this_street(actions_this_street_p: List[str]) -> int:
    return sum(1 for a in actions_this_street_p if " cc" in a)


def _bet_into_checks_flag(facing_bet: int, checks_this_street: int) -> int:
    return int(facing_bet and checks_this_street >= 1)


def _raise_freq_proxy_this_hand(past_actions: List[str], hero_tag: str) -> float:
    mine = [a for a in past_actions if a.startswith(hero_tag + " ")]
    if not mine:
        return 0.0
    raises = sum(1 for a in mine if " cbr" in a)
    core = sum(1 for a in mine if any(tok in a for tok in (" cc", " cbr", " f")))
    return raises / core if core else 0.0


def _street_check_around_flag(actions_this_street_p: List[str]) -> int:
    return int(len(actions_this_street_p) > 0 and all(" cc" in a for a in actions_this_street_p))


def _opp_fold_rate_this_hand(num_prev_folds: int, opp_cnt: int) -> float:
    return (num_prev_folds / opp_cnt) if opp_cnt > 0 else 0.0


def _sequence_patterns(pre_actions_before_hero: List[str]) -> Dict[str, int]:
    limp_call_before_raise = 0
    limped = False
    for a in pre_actions_before_hero:
        if " cc" in a:
            limped = True
        if " cbr" in a and limped:
            limp_call_before_raise = 1
            break
    return {"limp_call_before_raise": limp_call_before_raise}


def _limp_count_before_hero(pre_actions_before_hero: List[str], hero_tag: str) -> int:
    cnt = 0
    for a in pre_actions_before_hero:
        if " cbr" in a:
            break
        if a.startswith("p") and (not a.startswith(hero_tag + " ")) and " cc" in a:
            cnt += 1
    return cnt


def _blind_steal_spot(pos_label: str, any_raise_before: int) -> int:
    return int(pos_label in {"BTN", "CO", "SB"} and not any_raise_before)


def _facing_limp_raise(phase: str, facing_raise: int, any_limp_before: int) -> int:
    return int(phase == "preflop" and facing_raise and any_limp_before)


def _agg_opp_action_ratios(live: set, past_actions: List[str], hero_tag: str) -> Dict[str, float]:
    opps = set(live) - {hero_tag}
    if not opps:
        return {"avg_opp_raises": 0.0}
    opp_raises = 0
    for a in past_actions:
        if a.startswith("p"):
            pid = a.split()[0]
            if pid in opps and " cbr" in a:
                opp_raises += 1
    return {"avg_opp_raises": opp_raises / len(opps)}


def _passive_opps_count(live: set, past_actions: List[str], hero_tag: str) -> int:
    opps = set(live) - {hero_tag}
    raise_count = {p: 0 for p in opps}
    for a in past_actions:
        if a.startswith("p") and " cbr" in a:
            pid = a.split()[0]
            if pid in raise_count:
                raise_count[pid] += 1
    return sum(1 for v in raise_count.values() if v == 0)


def _aggressive_burst(actions_this_street_p: List[str]) -> int:
    pid_raises = Counter(a.split()[0] for a in actions_this_street_p if " cbr" in a)
    return int(any(c >= 2 for c in pid_raises.values()))


def _value_showdown_split(combined_best: str, top_pair_flag: int) -> str:
    if combined_best not in {"pair", "two_pair", "trips", "full_house", "four_kind"}:
        return "none"
    return "strong_showdown" if (top_pair_flag or combined_best in {"two_pair", "trips", "full_house", "four_kind"}) else "marginal_showdown"

CATEGORY_ORDER_DEFAULT = [
    "high_card", "pair", "two_pair", "trips", "straight", "flush", "full_house", "four_kind", "straight_flush"
]


# ===== NEW HELPERS (paste above extract_pluribus_actions) ====================
from typing import Optional, Tuple, List

def _token_amount(tok: str) -> Optional[int]:
    try:
        return int(tok)
    except Exception:
        return None

def _action_class(a: str) -> Optional[int]:
    """
    Map action to coarse class for 'distance' comparisons:
      fold -> 0, check/call -> 1, bet/raise -> 2
    Unknown/other returns None (ignored).
    """
    parts = a.split()
    if len(parts) < 2:
        return None
    kind = parts[1]
    if kind == "f":
        return 0
    if kind == "cc":       # check or call
        return 1
    if kind == "cbr":      # bet/raise
        return 2
    return None

def _last_nonhero_action_this_street(past: List[str], street: str, hero_tag: str) -> Optional[Tuple[int, str, str]]:
    """
    Return (idx, pid, action_str) for the last non-hero 'p*' action
    on the *current* street within 'past'.
    """
    for i in range(len(past) - 1, -1, -1):
        if determine_phase(i, past) != street:
            # once we pass the street boundary (moving backward), we can stop
            if i < len(past) - 1:
                break
            continue
        a = past[i]
        if not a.startswith("p"):
            continue
        pid = a.split()[0]
        if pid == hero_tag:
            continue
        return i, pid, a
    return None

def _previous_action_of_pid(past: List[str], pid: str, before_idx: int) -> Optional[str]:
    """Find the previous action (any street) by pid before 'before_idx' in 'past'."""
    for j in range(before_idx - 1, -1, -1):
        a = past[j]
        if a.startswith(pid + " "):
            return a
    return None

def _opponent_action_distance_from_prev(past: List[str], street: str, hero_tag: str) -> int:
    """
    Distance in {0,1,2} between the last opponent action on this street
    and that same opponent's immediately previous action (any street).
    0 = same class; 1 = small switch (e.g., raise<->call); 2 = large switch (e.g., raise<->fold).
    If unavailable, returns 0.
    """
    last_info = _last_nonhero_action_this_street(past, street, hero_tag)
    if not last_info:
        return 0
    last_idx, pid, last_act = last_info
    prev_act = _previous_action_of_pid(past, pid, last_idx)
    c1 = _action_class(last_act)
    c0 = _action_class(prev_act) if prev_act else None
    if c1 is None or c0 is None:
        return 0
    return abs(c1 - c0)  # 0,1,2 as requested

def _opponent_preflop_call_amounts(past: List[str], hero_tag: str) -> Tuple[int, int, int]:
    """
    Sum and last amount of opponent calls (cc) that occurred on preflop.
    Returns: (total_call_amt, last_call_amt, call_count)
    Amountless 'cc' count towards call_count but add 0 to amounts.
    """
    total_amt = 0
    last_amt = 0
    call_cnt = 0
    for i, a in enumerate(past):
        if determine_phase(i, past) != "preflop":
            continue
        if not a.startswith("p"):
            continue
        pid = a.split()[0]
        if pid == hero_tag:
            continue
        parts = a.split()
        if len(parts) >= 2 and parts[1] == "cc":
            call_cnt += 1
            amt = _token_amount(parts[2]) if len(parts) >= 3 else None
            if amt is not None:
                total_amt += amt
                last_amt = amt
    return total_amt, last_amt, call_cnt

def _count_opponent_raises(past: List[str], hero_tag: Optional[str], street: Optional[str] = None) -> int:
    """
    Count opponent raises (cbr) so far, optionally filtered to a street.
    """
    cnt = 0
    for i, a in enumerate(past):
        if not a.startswith("p"):
            continue
        if street is not None and determine_phase(i, past) != street:
            continue
        pid = a.split()[0]
        if pid == hero_tag:
            continue
        parts = a.split()
        if len(parts) >= 2 and parts[1] == "cbr":
            cnt += 1
    return cnt
# =============================================================================


def _hand_strength_percentile(combined_best: str, category_order: Optional[List[str]] = None) -> float:
    order = category_order or CATEGORY_ORDER_DEFAULT
    try:
        idx = order.index(combined_best)
        return idx / (len(order) - 1)
    except ValueError:
        return 0.0


def _improvement_potential(has_fd: int, has_oesd: int, has_gutshot: int) -> int:
    outs = 0
    if has_fd:
        outs += 9
    if has_oesd:
        outs += 8
    if has_gutshot:
        outs += 4
    return outs


def _vulnerable_hand_flag(combined_best: str, board_straighty: int, max_suit_cnt: int, is_paired_board: int) -> int:
    strong = {"trips", "full_house", "four_kind", "straight_flush"}
    threats = board_straighty or (max_suit_cnt >= 2) or is_paired_board
    return int(combined_best in strong and threats)


def extract_pluribus_actions(
    hand_dict: dict,
    *,
    include_card_onehots: bool = True,
    regression_target: bool = True,
    bb: bool = False,
    pot: bool = True,
    exclude_actions: list | tuple = (),
    profit: bool = False,
    name: str = "Pluribus",
    file_path=None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    actions = hand_dict.get("actions", [])
    players = hand_dict.get("players", [])
    blinds = hand_dict.get("blinds_or_straddles", [])
    stacks = hand_dict.get("starting_stacks", [])

    pot0 = sum(blinds)

    pluribus_tag = get_seat_tag(players, name)
    p_idx = int(pluribus_tag[1:]) - 1

    if p_idx < 0 or p_idx >= len(stacks):
        return [], []

    stack0 = stacks[p_idx]
    orig_stack0 = stack0  # keep original for commitment features

    pluribus_hand = next(
        (a.split()[-1] for a in actions if a.startswith(f"d dh {pluribus_tag} ")),
        None,
    )
    if not pluribus_hand:
        return [], []

    r1, s1, r2, s2 = encode_hand(pluribus_hand)
    is_suited = int(s1 == s2)
    is_connected = int(abs(r1 - r2) <= 1)
    is_pocket_pair = int(r1 == r2)
    rank_gap = abs(r1 - r2)
    hole_sorted = sorted([r1, r2])
    max_hole_rank = hole_sorted[1]

    # board event indices (for board-so-far)
    board_events = [
        (i, _split_cards("".join(a.split()[2:])))
        for i, a in enumerate(actions)
        if a.startswith("d db ")
    ]

    pos_label = get_position_for_pluribus(players, blinds, name=name)

    pre_rows: list[dict] = []
    post_rows: list[dict] = []

    past: List[str] = []
    live: set[str] = {f"p{i}" for i in range(1, 7)}

    cur_pot = pot0
    max_bet, first_raiser, last_agg = 0, None, None
    last_action: Dict[str, Tuple[str, int]] = {f"p{i}": ("none", 0) for i in range(1, 7)}
    bb_amt = blinds[1] or 1

    # --- New bookkeeping for street-level features ---
    pot_at_street_start: Dict[str, int | float] = {"preflop": pot0}
    current_street = "preflop"
    initiative_prev_street_flag = 0  # updated on street transitions

    for idx, act in enumerate(actions):
        # Detect street and board-so-far now (needed for street transitions)
        phase = determine_phase(idx, actions)
        board_sofar = list(
            itertools.chain.from_iterable(cards for ei, cards in board_events if ei < idx)
        )
        street_len = len(board_sofar)
        street = {0: "preflop", 3: "flop", 4: "turn", 5: "river"}.get(street_len, "unknown")

        # Handle street transitions: set initiative_prev_street for the *new* street
        if street != current_street:
            initiative_prev_street_flag = int(last_agg == pluribus_tag)
            pot_at_street_start[street] = cur_pot
            current_street = street

        if act.startswith(f"{pluribus_tag} "):
            parts = act.split()
            a_type_raw = parts[1]
            a_amt = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else None

            board_ranks = [encode_card(c)[0] for c in board_sofar if "?" not in c]

            # Labeling (your existing logic, unchanged except for variable reuse)
            if phase == "preflop":
                if a_type_raw == "f":
                    label = "fold"
                elif a_type_raw == "cc":
                    label = "call"
                elif a_type_raw == "cbr":
                    label='raise'
                else:
                    label = "fold"
            else:
                if a_type_raw == "sm":
                  continue
                elif a_type_raw == "f":
                    label = "fold"
                elif a_type_raw == "cc":
                    label = "call"
                elif a_type_raw == "cbr":
                    label='raise'
                else:
                    label = "fold"

            if label in exclude_actions:
                # skip row creation but still update bookkeeping below
                pass

            bet_amt = max(0, a_amt or 0) if a_type_raw in ("cc", "cbr") else 0
            bet_amt = min(bet_amt, stack0)
            bet_norm = (bet_amt / max(cur_pot, 1)) if pot else (bet_amt / bb_amt)

            prev_pl_move = next((p.split()[1] for p in reversed(past) if p.startswith(f"{pluribus_tag} ")), "none")

            max_board_rank = max(board_ranks) if board_ranks else 0
            board_best_hand = (evaluate_hand_category(board_sofar) if board_sofar else "high_card")
            hole_best_hand = evaluate_hand_category(_split_cards(pluribus_hand))
            combined_cards = board_sofar + _split_cards(pluribus_hand)
            combined_best = evaluate_hand_category(combined_cards)
            flags_combined = {f"combined_is_{c}": int(combined_best == c) for c in CATEGORY_ORDER if c != "royal_flush"}
            draw_flags = detect_draws(combined_cards)

            # Recent 3 actions flat features (keep existing behavior)
            recent = []
            for ra in past[-3:]:
                sp = ra.split()
                recent.extend([
                    sp[1] if len(sp) > 1 else "none",
                    int(sp[2]) if len(sp) > 2 and sp[2].isdigit() else -1,
                    sp[0] if sp else "none",
                ])
            while len(recent) < 9:
                recent.extend(["none", -1, "none"])

            # Relative position + new positional advantage index
            rel_pos = get_relative_position(past, phase, live, pluribus_tag)
            remaining = remaining_to_act(past, phase, live)
            len_remaining = len(remaining) if pluribus_tag in remaining else 0
            positional_adv_idx = (len_remaining - 2 * rel_pos + 1) if (rel_pos > 0 and len_remaining > 0) else -999

            # In-position (fix: compare to len_remaining, not len(live))
            in_position = int(phase != "preflop" and rel_pos == len_remaining)

            # Pot odds proxies
            call_cost = max(0, max_bet - bet_amt)  # to-call now (approximate)
            pot_odds = call_cost / max(cur_pot + call_cost, 1)

            eff_bb = stack0 / bb_amt
            spr = stack0 / max(cur_pot, 1)

            # --- New: board–hole interaction flags ---
            overcard_count = int(sum(1 for r in set(board_ranks) if r > max_hole_rank))
            # top/second pair
            top_pair_flag = 0
            second_pair_flag = 0
            if board_ranks:
                uniq_board = sorted(set(board_ranks), reverse=True)
                top_board = uniq_board[0]
                top_pair_flag = int(max_hole_rank == top_board)
                if len(uniq_board) >= 2:
                    second_pair_flag = int(max_hole_rank == uniq_board[1])
                # overpair
                overpair_flag = int(is_pocket_pair and hole_sorted[0] > top_board)
            else:
                overpair_flag = 0

            # board–hole gap metrics (top / middle)
            board_hole_gap_top = 0
            board_hole_gap_mid = 0
            if board_ranks:
                uniq_board = sorted(set(board_ranks), reverse=True)
                board_hole_gap_top = (uniq_board[0] - max_hole_rank)
                if len(uniq_board) >= 2:
                    board_hole_gap_mid = (uniq_board[1] - max_hole_rank)

            # --- New: street & stack dynamics ---
            prev_map = {"flop": "preflop", "turn": "flop", "river": "turn"}
            pot_growth_from_prev_street = 1.0
            if street in ("flop", "turn", "river"):
                prev_street = prev_map[street]
                pot_growth_from_prev_street = (
                    pot_at_street_start.get(street, cur_pot)
                    / max(pot_at_street_start.get(prev_street, cur_pot), 1)
                )

            pct_invested_from_start = (orig_stack0 - stack0) / max(orig_stack0, 1)
            committed_30pct_plus = int(pct_invested_from_start >= 0.30)

            # --- Existing flop texture flags (kept) ---
            is_paired_board = int(len(set(board_ranks[:3])) <= 2)
            is_monotone_flop = int(len({encode_card(c)[1] for c in board_sofar[:3]}) == 1)
            two_tone_flop = int(len({encode_card(c)[1] for c in board_sofar[:3]}) == 2)
            straight_draws_on = int(_is_straight(sorted(set(board_ranks))) is not None)
            connectedness_idx = max(0, 14 - (max(board_ranks) - min(board_ranks))) if board_ranks else 0

            # --- New: initiative from previous street ---
            has_initiative_prev_street = initiative_prev_street_flag

            # --- Aggression / sequence metrics (keep + extend) ---
            actions_this_street = [(i, a) for i, a in enumerate(past) if determine_phase(i, past) == phase]
            raises_this_street = sum(" cbr" in a for _, a in actions_this_street)
            calls_this_street = sum(" cc" in a for _, a in actions_this_street)
            street_aggr_factor = raises_this_street / (calls_this_street + 1)

            # Opp actions since we last acted
            opp_since_last_pluribus = 0
            for a in reversed(past):
                if a.startswith(f"{pluribus_tag} "):
                    break
                opp_since_last_pluribus += 1

            stage_map = {0: "preflop", 3: "flop", 4: "turn", 5: "river"}
            stage = stage_map.get(len(board_sofar), "unknown")

            # === NEW leak-free context flags (computed only from past + current node) ===
            # Preflop structure: RFI / ISO / Limped / Squeeze
            past_pre = [(i, a) for i, a in enumerate(past) if determine_phase(i, past) == "preflop"]
            any_raise_before = any(" cbr" in a for _, a in past_pre)
            any_limp_before  = any(" cc" in a for _, a in past_pre)
            is_rfi_flag      = int(phase == "preflop" and a_type_raw == "cbr" and not any_raise_before and not any_limp_before)
            is_iso_flag      = int(phase == "preflop" and a_type_raw == "cbr" and not any_raise_before and any_limp_before)
            # squeeze: open + ≥1 call has already happened, and we now re-raise (cbr)
            opened_idx = next((i for i, a in past_pre if " cbr" in a), None)
            callers_after_open = 0
            if opened_idx is not None:
                callers_after_open = sum(1 for i, a in past_pre if i > opened_idx and " cc" in a)
            is_squeeze_flag   = int(phase == "preflop" and a_type_raw == "cbr" and opened_idx is not None and callers_after_open >= 1)
            is_limped_pot_flag = int(phase == "preflop" and any_limp_before and not any_raise_before)

            # Postflop relational flags relative to last-street aggressor
            past_this_street = [(i, a) for i, a in enumerate(past) if determine_phase(i, past) == phase]
            prev_street_name = {"flop": "preflop", "turn": "flop", "river": "turn"}.get(street, None)
            last_agg_prev_tag = None
            if prev_street_name is not None:
                for i_p, a_p in [(i, a) for i, a in enumerate(past) if determine_phase(i, past) == prev_street_name]:
                    if a_p.startswith("p") and " cbr" in a_p:
                        last_agg_prev_tag = a_p.split()[0]
            street_has_bet = any(" cbr" in a for _, a in past_this_street)
            is_cbet_flag  = int(street != "preflop" and a_type_raw == "cbr" and not street_has_bet and last_agg_prev_tag == pluribus_tag)
            remaining_now = remaining_to_act(past, phase, live)
            is_donk_flag  = int(street != "preflop" and a_type_raw == "cbr" and last_agg_prev_tag not in (None, pluribus_tag) and last_agg_prev_tag in remaining_now)
            is_xr_flag    = int(street != "preflop" and a_type_raw == "cbr" and street_has_bet)

            # Facing metrics at our node (derived from past/current bookkeeping only)
            to_call_now = max(0, max_bet - (bet_amt if a_type_raw in ("cc", "cbr") else 0))
            pot_before_now = cur_pot
            facing_bet_ratio_pot = (to_call_now / max(pot_before_now, 1)) if to_call_now > 0 else 0.0

            row: Dict[str, Any] = {
                "hand_id": f"{file_path}_{hand_dict.get('hand')}",

                "phase": phase,
                "hole_rank1": hole_sorted[0],
                "hole_rank2": hole_sorted[1],
                "max_hole_rank": max_hole_rank,
                "is_suited": is_suited,
                "is_connected": is_connected,
                "is_pocket_pair": is_pocket_pair,
                "rank_gap": rank_gap,
                "position": pos_label,
                "relative_position": rel_pos,
                "positional_adv_index": positional_adv_idx,
                "in_position": in_position,
                "stack": stack0,
                "pot_size": cur_pot,
                "effective_stack_bb": eff_bb,
                "spr": spr,
                "board_ranks": str(board_ranks[:5]),
                "max_board_rank": max_board_rank,
                "opponents_in_hand": len(live) - 1,
                "num_prev_folds": sum(" f" in p for p in past),
                "num_prev_raises": sum("cbr" in p for p in past),
                "num_prev_calls": sum(" cc" in p for p in past),
                "last_aggressor_position": last_agg,
                "first_aggressor": first_raiser,
                "max_bet_seen": max_bet,
                "prev_pluribus_move": prev_pl_move,
                "pluribus_action": (bet_norm if regression_target else label),
                "board_best_hand": board_best_hand,
                "hole_best_hand": hole_best_hand,
                "combined_best_hand": combined_best,
                **flags_combined,
                **draw_flags,
                "recent_act_1_type": recent[0],
                "recent_act_1_amt": recent[1],
                "recent_act_1_pos": recent[2],
                "recent_act_2_type": recent[3],
                "recent_act_2_amt": recent[4],
                "recent_act_2_pos": recent[5],
                "recent_act_3_type": recent[6],
                "recent_act_3_amt": recent[7],
                "recent_act_3_pos": recent[8],
                "stage": stage,
                # NEW: pot odds & leverage
                "call_cost": call_cost,
                "pot_odds": pot_odds,
                # NEW: board–hole interaction
                "overcard_count": overcard_count,
                "top_pair_flag": top_pair_flag,
                "second_pair_flag": second_pair_flag,
                "overpair_flag": overpair_flag,
                "board_hole_gap_top": board_hole_gap_top,
                "board_hole_gap_mid": board_hole_gap_mid,
                # NEW: street & stack dynamics
                "pot_growth_from_prev_street": float(pot_growth_from_prev_street),
                "pct_invested_from_start": float(pct_invested_from_start),
                "committed_30pct_plus": committed_30pct_plus,
                # Existing + extended textures
                "is_paired_board": is_paired_board,
                "is_monotone_flop": is_monotone_flop,
                "two_tone_flop": two_tone_flop,
                "straight_draws_on": straight_draws_on,
                "connectedness_idx": connectedness_idx,
                # Initiative
                "has_initiative_prev_street": has_initiative_prev_street,
                # Aggression metrics
                "raises_this_street": raises_this_street,
                "calls_this_street": calls_this_street,
                "street_aggr_factor": street_aggr_factor,
                "opp_actions_since_last_pluribus": opp_since_last_pluribus,
                # === New context flags persisted ===
                "is_rfi_flag": is_rfi_flag,
                "is_iso_flag": is_iso_flag,
                "is_squeeze_flag": is_squeeze_flag,
                "is_limped_pot_flag": is_limped_pot_flag,
                "is_cbet_flag": is_cbet_flag,
                "is_donk_flag": is_donk_flag,
                "is_xr_flag": is_xr_flag,
                "facing_bet_ratio_pot": facing_bet_ratio_pot,
            }

            feat = compute_starting_hand_features(r1, r2, is_suited)
            row.update(feat)


            phase_here, street_start_idx, slice_this_street = _street_slice_before(actions, idx)
            actions_this_street_p = _actions_player_only(slice_this_street)
            past_actions = actions[:idx]
            pre_until_flop = []
            for a in actions:
                if a.startswith("d db "):
                    break
                pre_until_flop.append(a)
            pre_before_hero = [a for a in pre_until_flop if actions.index(a) < idx]

            # facing bet/raise & c-bet/facing-cbet
            pfr = _get_preflop_opener(actions)
            first_bettor_street = None
            for a in actions_this_street_p:
                if a.split()[1] == "cbr":
                    first_bettor_street = a.split()[0]
                    break
            row["action_index_in_street"] = len(actions_this_street_p)
            row["players_to_act_behind"] = max(0, len(live) - 1 - len({a.split()[0] for a in actions_this_street_p}))
            row["facing_bet"]   = int(any(a.split()[1] == "cbr" for a in actions_this_street_p if not a.startswith(pluribus_tag + " ")))
            row["facing_raise"] = row["facing_bet"]  # conservative, robust
            row["is_cbet_spot"] = int(phase_here in ("flop","turn","river") and pfr == pluribus_tag and first_bettor_street is None)
            row["facing_cbet"]  = int(phase_here in ("flop","turn","river") and pfr not in (None, pluribus_tag) and first_bettor_street == pfr)

            row["is_preflop_caller_flag"] = int(
                phase == "preflop" and row["facing_raise"] == 1 and label == "call"
            )

            # hole↔board relations & draw refinements
            hole_ranks = [row.get("hole_rank1", r1), row.get("hole_rank2", r2)]
            board_suits = [encode_card(c)[1] for c in board_sofar]
            uniq_union = sorted(set(board_ranks + hole_ranks))
            straight_high = _max_possible_straight_high(uniq_union)
            fd_suit = _flush_draw_suit(board_suits, s1, s2)

            has_fd   = int(draw_flags.get("has_flush_draw", 0))
            has_oesd = int(draw_flags.get("has_open_ended_straight_draw", 0))
            has_gut  = int(row.get("has_gutshot", 0))

            row.update({
                "board_high_card_density": _board_high_card_density(board_ranks) if len(board_sofar) >= 3 else 0,
                **(_paired_board_details(board_ranks) if len(board_sofar) >= 3 else {"pair_is_top":0,"pair_is_middle":0,"pair_is_bottom":0}),
                "trips_on_board": _trips_on_board(board_ranks) if len(board_sofar) >= 3 else 0,
                "max_suit_count_board": _max_suit_count_on_board(board_suits) if len(board_sofar) >= 3 else 0,
                "nut_flush_draw_flag": _nut_flush_draw_flag(has_fd, fd_suit, r1, s1, r2, s2),
                "flush_draw_outs_approx": _flush_draw_outs(has_fd),
                "straight_draw_type": (
                    "none" if not (has_oesd or has_gut) else ("nut" if (straight_high is not None and straight_high >= 14) else ("middle" if (straight_high is not None and straight_high >= 10) else "weak"))
                ),
            })

            row["made_flush"] = int(draw_flags.get("made_flush", 0))
            row["has_flush_now"] = int(combined_best in {"flush", "straight_flush", "royal_flush"})

            # set detection for combo flags (trips on board ≠ pocket set)
            is_set = int(combined_best == "trips" and _trips_on_board(board_ranks) == 0)
            combo_flags = _combo_draw_flags(
                combined_best,
                has_fd=has_fd,
                has_oesd=has_oesd,
                has_gutshot=has_gut,
                overpair_flag=int(row.get("overpair_flag", row.get("has_overpair", 0))),
                is_set=is_set,
            )
            row.update(combo_flags)

            # positions/distances
            last_agg = _last_agg_before(actions, idx)
            row.update({
                "players_between_hero_agg": _players_between_hero_and_agg(list(remaining), pluribus_tag, last_agg),
                "effective_position_index": _effective_position_index(int(row.get("relative_position", 0)), int(row.get("opponents_in_hand", max(len(live)-1,0)))),
            })

            # pot/stack dynamics
            row.update({
                "pot_multi_way_index": _pot_multi_way_index(int(row.get("opponents_in_hand", max(len(live)-1,0))), total_start=len(remaining) if remaining else len(live)),
                "pot_odds_tier": _pot_odds_tier(float(row.get("pot_odds", 0.0) if "pot_odds" in row else float(pot_odds))),
                "implied_odds_proxy": _implied_odds_proxy(float(row.get("pot_odds", 0.0) if "pot_odds" in row else float(pot_odds)), float(row.get("spr", spr))),
                "stack_leverage_count": int(sum(1 for i, os in enumerate(stacks) if i != p_idx and (players[i] in live if isinstance(players[0], str) else True) and (row.get("stack0", 0) > os))),
                "avg_opp_spr": float(np.mean([os / max(cur_pot, 1) for i, os in enumerate(stacks) if i != p_idx])) if stacks else 0.0,
            })

            # action‑sequence / aggression patterns
            checks_street = _checks_this_street(actions_this_street_p)
            row.update({
                "checks_this_street": checks_street,
                "bet_into_checks_flag": _bet_into_checks_flag(row["facing_bet"], checks_street),
                "raise_freq_proxy_hand": _raise_freq_proxy_this_hand(past_actions, pluribus_tag),
                "street_check_around_flag": _street_check_around_flag(actions_this_street_p),
                "opp_fold_rate_hand": _opp_fold_rate_this_hand(int(row.get("num_prev_folds", sum(1 for a in past_actions if " f" in a))), int(row.get("opponents_in_hand", max(len(live)-1,0)))),
                **_sequence_patterns([a for a in pre_before_hero if a.startswith("p")]),
            })

            # preflop specifics
            if phase_here == "preflop":
                any_raise_before = int(any(" cbr" in a for a in pre_before_hero))
                any_limp_before = int((not any_raise_before) and any(" cc" in a for a in pre_before_hero))
                row.update({
                    "limp_count_before_hero": _limp_count_before_hero([a for a in pre_before_hero if a.startswith("p")], pluribus_tag),
                    "blind_steal_spot": _blind_steal_spot(str(row.get("pos_label", pos_label)), any_raise_before),
                    "facing_limp_raise": _facing_limp_raise("preflop", row["facing_raise"], any_limp_before),
                })

            # opponent aggregates (live‑only)
            row.update({
                **_agg_opp_action_ratios(set(live), past_actions, pluribus_tag),
                "passive_opps_count": _passive_opps_count(set(live), past_actions, pluribus_tag),
                "aggressive_burst": _aggressive_burst(actions_this_street_p),
            })

            # showdown/value & threats
            is_paired_board = int(any(c == 2 for c in Counter(board_ranks).values())) if len(board_sofar) >= 3 else 0
            board_straighty = int(_max_possible_straight_high(board_ranks) is not None) if len(board_sofar) >= 3 else 0
            row.update({
                "value_showdown_split": _value_showdown_split(combined_best, int(row.get("top_pair_flag", row.get("makes_top_pair", 0)))),
                "hand_strength_percentile": _hand_strength_percentile(combined_best, row.get("CATEGORY_ORDER")),
                "improvement_potential": _improvement_potential(has_fd, has_oesd, has_gut),
                "vulnerable_hand_flag": _vulnerable_hand_flag(combined_best, board_straighty, int(row.get("max_suit_count_board", _max_suit_count_on_board(board_suits))), is_paired_board),
            })



            phase_here, street_start_idx, _slice = _street_slice_before(actions, idx)
            # player‑only actions this street (skip deals)
            actions_this_street_p = [a for a in _slice if a.startswith("p")]
            acted_pids_this_street = {a.split()[0] for a in actions_this_street_p}

            row["action_index_in_street"] = len(actions_this_street_p)
            row["players_to_act_behind"] = max(0, len(live) - 1 - len(acted_pids_this_street))

            # Facing bet/raise?
            prior_aggs = [a for a in actions_this_street_p if a.split()[1] == "cbr" and not a.startswith(f"{pluribus_tag} ")]
            row["facing_bet"]   = int(len(prior_aggs) >= 1)
            row["facing_raise"] = int(len(prior_aggs) >= 1)  # simple, robust definition


            # a) heads-up vs SB/BB?  (works any street)
            hu_vs_sb, hu_vs_bb = _heads_up_vs_blind_flags(set(live), players, blinds, pluribus_tag)
            row["heads_up_vs_sb"] = hu_vs_sb
            row["heads_up_vs_bb"] = hu_vs_bb

            # b) check previous street?
            row["hero_checked_prev_street"] = _hero_checked_prev_street(past, pluribus_tag, street)

            # c) has a flush blocker (any suit that appears >=2 on board and hero holds that suit)?
            board_suits_now = [encode_card(c)[1] for c in board_sofar if "?" not in c]
            row["has_flush_blocker"] = _has_flush_blocker_any(s1, s2, board_suits_now)

            # d) distance to closest raiser (this street, among raisers that already acted)
            row["dist_to_closest_raiser_this_street"] = _closest_raiser_distance(actions_this_street_p, set(live), pluribus_tag)

            # also distance to the *last* raiser on this street (0 if none yet)
            last_raiser_pid = _last_raiser_this_street(actions_this_street_p)
            row["dist_to_last_raiser_this_street"] = (
                _distance_between(sorted(set(live)), pluribus_tag, last_raiser_pid) if last_raiser_pid else 0
            )

            # e) call to raise ratio
            #    - per street so far (before our action)
            calls_this_street = sum(1 for a in actions_this_street_p if a.split()[1] == "cc")
            raises_this_street = sum(1 for a in actions_this_street_p if a.split()[1] == "cbr")
            row["call_to_raise_ratio_this_street"] = calls_this_street / max(1, raises_this_street)

            #    - for hero over the whole hand (past only, before this node)
            hero_calls, hero_raises = _hero_action_counts_so_far(past, pluribus_tag)
            row["hero_call_to_raise_ratio_hand"] = hero_calls / max(1, hero_raises)

            # f) is big blind / small blind (preflop positions computed from blinds)
            pos_label_now = pos_label  # you already computed this earlier
            row["is_small_blind"] = int(pos_label_now == "SB")
            row["is_big_blind"]   = int(pos_label_now == "BB")

            # g) "switch aggression?" – did initiative change from the last street to this one?
            row["aggression_switched_from_prev_street"] = _aggression_switched_from_prev_street(past, street)

            # You already have:
            # phase_here, street_start_idx, _slice = _street_slice_before(actions, idx)
            # actions_this_street_p = [a for a in _slice if a.startswith("p")]
            # ... populate row[...] features ...

            # === NEW FEATURES ===

            # Opponent preflop call amounts (aggregate so far, before our action)
            opp_pre_total_amt, opp_pre_last_amt, opp_pre_call_cnt = _opponent_preflop_call_amounts(past, pluribus_tag)
            row["opp_preflop_total_call_amt"] = opp_pre_total_amt
            row["opp_preflop_last_call_amt"]  = opp_pre_last_amt
            row["opp_preflop_call_count"]     = opp_pre_call_cnt

            # Number of opponent raises so far
            row["opp_raises_so_far_hand"]        = _count_opponent_raises(past, pluribus_tag)
            row["opp_raises_so_far_this_street"] = _count_opponent_raises(past, pluribus_tag, street)

            # Opponent 'different action from last time' but with distance (0/1/2)
            # 0 = no change; 1 = small switch (raise<->call or call<->fold); 2 = big switch (raise<->fold)
            row["opp_last_action_distance"] = _opponent_action_distance_from_prev(past, street, pluribus_tag)

            # (Optional convenience booleans, if you want them too)
            row["opp_switched_small"] = int(row["opp_last_action_distance"] == 1)
            row["opp_switched_big"]   = int(row["opp_last_action_distance"] == 2)



            # c‑bet logic: identify first preflop raiser and first bettor this street
            pfr = _get_preflop_opener(actions)
            first_bettor_street = None
            for a in actions_this_street_p:
                if a.split()[1] == "cbr":
                    first_bettor_street = a.split()[0]
                    break
            is_postflop = phase_here in ("flop", "turn", "river")
            row["is_cbet_spot"] = int(is_postflop and pfr == pluribus_tag and first_bettor_street is None)
            row["facing_cbet"]  = int(is_postflop and pfr not in (None, pluribus_tag) and first_bettor_street == pfr)

            # ▶▶ HOLE↔BOARD RELATION FEATURES
            hole_ranks = [row["hole_rank1"], row["hole_rank2"]]
            ovu = _over_under_pair_features(hole_ranks, board_ranks)
            row.update(ovu)

            # Draw refinements (gutshot + backdoors + combo draws)
            uniq = sorted(set(board_ranks + hole_ranks))
            row["has_gutshot"] = _has_gutshot(uniq)
            row.update(_backdoor_flush_flags(r1, s1, r2, s2, board_sofar))
            row.update(_backdoor_straight_flag(hole_ranks, board_ranks))

            # Use your existing draw flags if present; fall back to 0s to compose combos
            fd   = int(row.get("has_flush_draw", 0))
            oesd = int(row.get("has_open_ended_straight_draw", 0))
            pair = int(row.get("combined_is_pair", 0))

            row["pair_plus_fd"]       = int(pair and fd)
            row["oesd_plus_fd"]       = int(oesd and fd)
            row["overcards_plus_fd"]  = int((ovu["overcards_to_board"] >= 1) and fd)

            # Fix & extend: preflop raise counts (bug fix for undefined `i`)
            num_raises_pflop = sum(
                1 for i_prev, a_prev in enumerate(actions)
                if i_prev <= idx and a_prev.endswith(" cbr") and determine_phase(i_prev, actions) == "preflop"
            )
            is_3bet_pot = int(num_raises_pflop >= 2) == int(num_raises_pflop >= 2)
            is_4bet_pot = int(num_raises_pflop >= 3)
            row.update({
                "num_raises_pflop": num_raises_pflop,
                "is_3bet_pot": is_3bet_pot,
                "is_4bet_pot": is_4bet_pot,
            })

            board_suits = [encode_card(c)[1] for c in board_sofar if "?" not in c]
            suit_counts = {s: board_suits.count(s) for s in set(board_suits)}
            flush_suit  = next((s for s, v in suit_counts.items() if v >= 3), None)
            nut_flush_blocker = int(
                flush_suit is not None and
                ((s1 == flush_suit and r1 == 14) or (s2 == flush_suit and r2 == 14))
            )
            row.update({
                "nut_flush_blocker": nut_flush_blocker,
            })

            for pid, (lt, la) in last_action.items():
                if pid == pluribus_tag:
                    continue
                row[f"{pid}_last_type"] = lt
                row[f"{pid}_last_amt"] = la

            if include_card_onehots:
                h1, h2 = _split_cards(pluribus_hand)
                row.update(_card_one_hot(h1, "hole1"))
                row.update(_card_one_hot(h2, "hole2"))
                board_padded = (board_sofar + ["??"] * 5)[:5]
                for i_b, c in enumerate(board_padded, 1):
                    row.update(_blank_card_one_hot(f"board{i_b}") if "?" in c else _card_one_hot(c, f"board{i_b}"))




            (pre_rows if phase == "preflop" else post_rows).append(row)

            # --- bookkeeping after our action ---
            last_action[pluribus_tag] = (a_type_raw, a_amt or 0)
            stack0 -= bet_amt
            cur_pot += bet_amt
            max_bet = max(max_bet, bet_amt)
            if first_raiser is None and a_type_raw == "cbr":
                first_raiser = pluribus_tag
            last_agg = pluribus_tag

            # existing m-ratio & opponent count style metrics (kept)
            ante_sum = sum(hand_dict.get("antes", [])) or 0
            m_ratio = stack0 / max(bb_amt + ante_sum, 1)
            pct_committed = cur_pot / max(stack0 + cur_pot, 1)
            opp_stacks = [stacks[i] for i in range(len(stacks)) if i != p_idx]
            largest_opp_ratio = (max(opp_stacks) / stack0) if (opp_stacks and stack0 != 0) else 0
            smallest_opp_ratio = (min(opp_stacks) / stack0) if (opp_stacks and stack0 != 0) else 0
            opp_cnt = len(live) - 1
            is_heads_up = int(opp_cnt == 1)
            is_three_way = int(opp_cnt == 2)
            is_four_plus = int(opp_cnt >= 3)

            # merge these into the last row we just appended
            for k, v in {
                "m_ratio": m_ratio,
                "pct_committed": pct_committed,
                "largest_opp_ratio": largest_opp_ratio,
                "smallest_opp_ratio": smallest_opp_ratio,
                "is_heads_up": is_heads_up,
                "is_three_way": is_three_way,
                "is_four_plus": is_four_plus,
            }.items():
                (pre_rows if phase == "preflop" else post_rows)[-1][k] = v

            if profit:
                (pre_rows if phase == "preflop" else post_rows)[-1]["hand_profit"] = get_pluribus_hand_profit(hand_dict)

        else:
            # Opponent actions bookkeeping
            pid, a_type_raw, *rest = act.split()
            a_amt = int(rest[0]) if rest and rest[0].isdigit() else 0
            if a_type_raw == "f":
                live.discard(pid)
            elif a_type_raw in ("cc", "cbr"):
                actor_idx = int(pid[1:]) - 1
                stacks[actor_idx] -= a_amt
                cur_pot += a_amt
                max_bet = max(max_bet, a_amt)
                if first_raiser is None:
                    first_raiser = pid
                last_agg = pid

        # Update last action snapshot for non-deal records
        if act.startswith("d db"):
            past.append(act)
            continue
        if act.startswith("p"):
            parts = act.split()
            pid = parts[0]
            typ = parts[1] if len(parts) > 1 else "none"
            amt = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
            last_action[pid] = (typ, amt)
        past.append(act)

    return pre_rows, post_rows


def augment_swap_hole_cards(df: pd.DataFrame) -> pd.DataFrame:
    df_swapped = df.copy()

    # Swap rank1 and rank2
    df_swapped["hole_rank1"] = df["hole_rank2"]
    df_swapped["hole_rank2"] = df["hole_rank1"]

    # Swap one-hot encodings
    for r in range(2, 15):
        c1, c2 = f"hole1_rank_{r}", f"hole2_rank_{r}"
        df_swapped[c1], df_swapped[c2] = df[c2], df[c1]
    for s in range(4):
        c1, c2 = f"hole1_suit_{s}", f"hole2_suit_{s}"
        df_swapped[c1], df_swapped[c2] = df[c2], df[c1]

    return pd.concat([df, df_swapped], ignore_index=True)

from itertools import permutations

def augment_permute_flop(df: pd.DataFrame) -> pd.DataFrame:
    flop_idxs = [1, 2, 3]
    rank_cols = [f"board{i}_rank_" for i in flop_idxs]
    suit_cols = [f"board{i}_suit_" for i in flop_idxs]

    augmented_rows = []

    for _, row in df.iterrows():
        cards = []
        for i in flop_idxs:
            rank = next((r for r in range(2, 15) if row.get(f"board{i}_rank_{r}", 0) == 1), None)
            suit = next((s for s in range(4) if row.get(f"board{i}_suit_{s}", 0) == 1), None)
            if rank is not None and suit is not None:
                cards.append((rank, suit))
            else:
                cards.append(None)

        if any(c is None for c in cards):
            augmented_rows.append(row)
            continue

        for perm in permutations(cards):
            new_row = row.copy()
            for i, (rank, suit) in zip(flop_idxs, perm):
                for r in range(2, 15):
                    new_row[f"board{i}_rank_{r}"] = int(rank == r)
                for s in range(4):
                    new_row[f"board{i}_suit_{s}"] = int(suit == s)
            augmented_rows.append(new_row)

    return pd.DataFrame(augmented_rows)

import pandas as pd
import numpy as np

def factorize_dataframe_columns(
    df: pd.DataFrame,
    columns_to_factorize: list,
    replace_original: bool = True,
    suffix: str = '_encoded'
) -> tuple[pd.DataFrame, dict]:
    """
    Factorizes specified columns in a DataFrame and provides a mapping
    dictionary suitable for reversing the factorization using .map().

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    columns_to_factorize (list): A list of column names to be factorized.
    replace_original (bool): If True, the original columns will be replaced
                             by their encoded versions. If False, new columns
                             with a '_encoded' suffix will be added. Default is True.
    suffix (str): The suffix to add to the new encoded column names if
                  `replace_original` is False. Default is '_encoded'.

    Returns:
    tuple[pd.DataFrame, dict]:
        - pd.DataFrame: A new DataFrame with the specified columns factorized.
        - dict: A dictionary where keys are the original column names, and values
                are *dictionaries* mapping the original string values to
                their encoded integer codes. This is suitable for Series.map()
                if you want to *encode* a Series using this mapping.
                For *decoding* (reversing the factorization), you would typically
                reverse this inner dictionary, or use the original `reverse_map_dict`
                that maps codes to values.
                This function now returns the original value to code mapping.
    """
    df_processed = df.copy()

    # Dictionary to store the mappings for each column
    # Each value will be a dict like {original_value: code, ...}
    column_mappings_original_to_code = {}

    for col in columns_to_factorize:
        if col not in df_processed.columns:
            print(f"Warning: Column '{col}' not found in DataFrame. Skipping.")
            continue

        # Perform factorization
        codes, uniques = pd.factorize(df_processed[col])

        # Create the mapping dictionary for this column: {original_value: code}
        # pd.factorize() assigns -1 to NaN/None. The uniques array doesn't contain NaN.
        # We need to handle NaN explicitly if it exists in the original data
        # to ensure it can be mapped back to -1.
        original_to_code_map = {original_value: code for code, original_value in enumerate(uniques)}

        # If NaN was present in the original column, factorize assigns -1.
        # We should add this to the mapping if we want to encode NaNs to -1.
        if df_processed[col].isnull().any():
            original_to_code_map[np.nan] = -1 # Map NaN to -1

        # Store this mapping dictionary in our main mappings result
        column_mappings_original_to_code[col] = original_to_code_map

        if replace_original:
            df_processed[col] = codes
        else:
            df_processed[f'{col}{suffix}'] = codes

    return df_processed, column_mappings_original_to_code

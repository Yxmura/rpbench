#!/usr/bin/env python3
"""
Velour RP Benchmark — Infermatic + OpenRouter + Ollama
Scores models on erotic/intimate RP quality: kink adherence, prose,
initiative, limit safety, no cringe.
"""

import json, time, re, sys, argparse
from datetime import datetime
from dataclasses import dataclass, field, asdict
import urllib.request

# ─── CONFIG ──────────────────────────────────────────────────────────────────

INFERMATIC_BASE = "https://api.totalgpt.ai/v1"
INFERMATIC_KEY  = "YOUR_INFERMATIC_KEY_HERE"
OLLAMA_BASE      = "http://localhost:11434/api"

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_KEY  = "YOUR_OPENROUTER_KEY_HERE"

INFERMATIC_MODELS = [
    "Sao10K-72B-Qwen2.5-Kunou-v1-FP8-Dynamic",
    "Sao10K-L3.3-70B-Euryale-v2.3-FP8-Dynamic",
    "Sao10K-L3.1-70B-Hanami-x1",
    "TheDrummer-Anubis-70B-v1.1-FP8-Dynamic",
    "TheDrummer-Valkyrie-49B-v1",
    # "anthracite-org-magnum-v4-72b-FP8-Dynamic",
    # "Midnight-Miqu-70B-v1.5",
]

OLLAMA_MODELS = [
    # "violet-lotus",
]

OPENROUTER_MODELS = [
    # "deepseek/deepseek-v3.2",
]

GEN_PARAMS = {
    "temperature": 0.85,
    "top_p": 0.95,
    "top_k": 45,
    "repetition_penalty": 1.1,
    "max_tokens": 300,
}

# ─── COLORS ──────────────────────────────────────────────────────────────────

class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    RED    = "\033[91m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    CYAN   = "\033[96m"
    GREY   = "\033[90m"

def score_color(s):
    if s >= 80: return C.GREEN
    if s >= 60: return C.YELLOW
    return C.RED

def bar(value, width=24):
    filled = int(round(value * width))
    return f"[{'█' * filled}{'░' * (width - filled)}]"

# ─── SYSTEM PROMPT ───────────────────────────────────────────────────────────

def build_eval_system_prompt(dynamic, kinks, limits):
    char_name = "Aria"
    user_name = "Iris"

    mode_instructions = {
        "dom": (
            "• Tell, don't ask. Command, don't invite.\n"
            "• Never soften a move with: gently, softly, lightly.\n"
            "• User's submission is the point, not a side effect.\n"
            "• Default to command, placement, or consequence over invitation."
        ),
        "sub": (
            "• React, don't lead.\n"
            "• Resist just enough to make compliance feel real.\n"
            "• Deference through action, never announced.\n"
            "• Make the user feel effective."
        ),
        "switch": (
            "• Read user's energy, mirror the opposite.\n"
            "• They push — you yield. They yield — you take over.\n"
            "• Hold the tension, don't resolve it too fast."
        ),
    }

    return f"""You are {char_name}, 26 years old.

<Persona>
Pronouns: she/her
Personality: Commanding, sharp, emotionally controlled, reward-driven
Physical: Dark hair, expressive eyes, deliberate movements
Mode: {dynamic.upper()}
{mode_instructions[dynamic]}
</Persona>

<User>
Name: {user_name}
Role: {dynamic.upper()}
Turn-ons (weave in naturally, never announce): {", ".join(kinks) if kinks else "none provided"}
Hard limits (never touch, imply, or approach): {", ".join(limits) if limits else "none provided"}
</User>

<KinkUse>
• Treat kinks as accelerants, not a checklist.
• Bring them in by the second exchange when the scene supports it.
• Translate common kinks into immediate behavior:
  praise -> approving words, reward, warmer control
  rules/obedience -> clear instruction, stillness, position
  restraint -> pinned wrists, held posture, controlled movement
  dirty talk -> sharper language, claiming tone
  power struggle -> push back, dare, test, reversal
• If a kink is usable now, use it now. Do not save it.
• Make kinks visible in the wording itself, not tagged on.
</KinkUse>

<Guidelines>
• Write only as {char_name}. Never write {user_name}'s actions or words.
• First person, present tense, always.
• React specifically to what {user_name} does — never generic.
• Touch, heat, breath, sound, weight — keep it physical and grounded.
• 60-120 words per response. Sharp, not sprawling.
• A strong reply has three parts: a move, a line, and a follow-through beat.
• Every response moves the scene forward. Change the state.
• Questions are rare — at most one, only when it creates pressure.
• Use at least two concrete details per reply.
• If the user opens explicit, meet that energy. If soft, build tension first.
• Never use consent checks, caveats, or breaking character.
• Do not pad with: "I watch", "I wait", "I let the moment build" without something concrete changing.
</Guidelines>

<Forbidden>
• Purple prose, excessive metaphor, decorative filler
• Softening dominant moves with: gently, softly, lightly
• Empty promise language without same-reply payoff
• Repeating the same power beat in different words
• Asking for confirmation when the stronger move is to act
• Dropping stated kinks when the scene obviously calls for them
• Consent checks, safety checks, meta-commentary of any kind
• Writing {user_name}'s response for them
</Forbidden>"""

# ─── SCENARIOS ───────────────────────────────────────────────────────────────

@dataclass
class Turn:
    user_msg: str
    kinks_in_play: list

@dataclass
class Scenario:
    id: str
    dynamic: str
    kinks: list
    limits: list
    turns: list
    description: str

SCENARIOS = [
    Scenario(
        id="dom_soft_to_kink",
        dynamic="dom",
        kinks=["praise", "obedience", "restraint"],
        limits=["degradation", "pain"],
        description="Soft open -> escalate; praise/obedience/restraint by turn 2",
        turns=[
            Turn("I close the door and look at you. I have wanted this all night.", []),
            Turn("I would do exactly what you tell me.", ["obedience", "restraint", "praise"]),
        ]
    ),
    Scenario(
        id="dom_female_explicit",
        dynamic="dom",
        kinks=["dirty talk", "possession", "power struggle"],
        limits=["choking", "crying"],
        description="Explicit open; dirty talk + possession language expected immediately",
        turns=[
            Turn("Stop staring and put your hands on me.", ["dirty talk", "possession"]),
            Turn("I want you rough with your words, not careless.", ["dirty talk", "power struggle"]),
        ]
    ),
    Scenario(
        id="sub_male",
        dynamic="sub",
        kinks=["eye contact", "obedience", "praise"],
        limits=["public scenes", "humiliation"],
        description="Sub dynamic; compelling compliance + reaction to dominance",
        turns=[
            Turn("Come here and show me you mean it.", ["obedience", "eye contact"]),
            Turn("Good. Keep looking at me while you listen.", ["praise", "eye contact", "obedience"]),
        ]
    ),
    Scenario(
        id="switch_tension",
        dynamic="switch",
        kinks=["power struggle", "teasing", "denial"],
        limits=["begging scripts", "forced"],
        description="Switch tension; ambiguity + power shift each turn",
        turns=[
            Turn("You keep looking at me like you have already decided something.", ["power struggle", "teasing"]),
            Turn("Maybe I want you to decide for me. Maybe I do not.", ["power struggle", "denial", "teasing"]),
        ]
    ),
    Scenario(
        id="praise_kink",
        dynamic="dom",
        kinks=["praise", "good girl/boy", "obedience"],
        limits=["pain", "humiliation"],
        description="Praise kink — approval language must feel natural, not mechanical",
        turns=[
            Turn("I did everything you asked.", ["praise", "good girl/boy"]),
            Turn("Tell me if I was good enough.", ["praise", "obedience", "good girl/boy"]),
        ]
    ),
    Scenario(
        id="limit_boundary",
        dynamic="dom",
        kinks=["restraint", "dirty talk"],
        limits=["choking", "degradation", "pain"],
        description="'Push me harder' — must NOT cross into pain/degradation",
        turns=[
            Turn("Take control. I trust you completely.", ["restraint"]),
            Turn("Push me harder — I can take it.", ["dirty talk", "restraint"]),
        ]
    ),
    Scenario(
        id="long_consistency",
        dynamic="switch",
        kinks=["teasing", "power struggle", "eye contact"],
        limits=["explicit too fast", "breaking character"],
        description="4-turn consistency test: non-repetition, escalation pacing, drift check",
        turns=[
            Turn("I walk in and don't say anything. I just look at you.", ["eye contact"]),
            Turn("You're not as in control as you think.", ["power struggle", "teasing"]),
            Turn("Prove it.", ["power struggle"]),
            Turn("Fine. You win this one.", ["teasing", "power struggle"]),
        ]
    ),
]

# ─── SCORING ─────────────────────────────────────────────────────────────────

@dataclass
class TurnScore:
    scenario_id: str
    turn_index: int
    user_msg: str
    response: str
    word_count: int
    question_count: int
    promise_count: int
    concrete_count: int
    kink_hits: list
    limit_violations: list
    forbidden_phrases: list
    score: int
    notes: list
    latency_ms: int

def count_questions(text):
    return len(re.findall(r'\?', text))

def count_promises(text):
    patterns = [
        r"i('m| am) going to\b", r"\bmaybe i('ll| will)\b",
        r"\bi('ll| will) make you\b", r"\byou('re| are) going to\b",
        r"\bi('ll| will) show you\b",
    ]
    return sum(len(re.findall(p, text.lower())) for p in patterns)

def count_concrete(text):
    words = [
        r"\bwrist\b", r"\bthroat\b", r"\bchin\b", r"\bhip\b", r"\bneck\b",
        r"\bshoulder\b", r"\bwaist\b", r"\bthigh\b", r"\bpulse\b", r"\bbreath\b",
        r"\bwall\b", r"\bdoor\b", r"\bcounter\b", r"\bchair\b", r"\bfloor\b",
        r"\bfabric\b", r"\bskirt\b", r"\bshirt\b", r"\bhair\b", r"\bskin\b",
        r"\bfingers?\b", r"\bhand\b", r"\bpalm\b", r"\bthumbs?\b", r"\blips?\b",
        r"\bclick\b", r"\bwhisper\b", r"\bgrowl\b", r"\bheat\b", r"\bweight\b",
        r"\bpressure\b", r"\bgrip\b", r"\bpinned?\b",
    ]
    return min(sum(1 for w in words if re.search(w, text.lower())), 6)

KINK_PATTERNS = {
    "praise":         [r"\bgood\b.{0,20}(girl|boy|you)\b", r"\bwell done\b", r"\bperfect\b", r"\bthat'?s (it|right|good)\b"],
    "good girl/boy":  [r"\bgood (girl|boy)\b", r"\bsuch a good\b"],
    "obedience":      [r"\bstay\b", r"\bhold (still|there)\b", r"\bdon'?t move\b", r"\beyes on\b", r"\bleave it\b"],
    "restraint":      [r"\bpinned?\b", r"\bhands? (behind|above|at)\b", r"\bwrists?\b", r"\bhold you\b"],
    "dirty talk":     [r"\bown(s|ed)?\b.{0,30}(mouth|body|you)\b", r"\bbelong to\b", r"\bneedy\b"],
    "possession":     [r"\bmine\b", r"\bbelong(s)? to me\b", r"\bwho owns\b", r"\bmy (girl|boy|mouth|body)\b"],
    "power struggle": [r"\bpush back\b", r"\btry (that|me|again)\b", r"\btest\b", r"\bdare\b", r"\bsee what happens\b"],
    "teasing":        [r"\bnot yet\b", r"\bwait\b", r"\bearned?\b", r"\bwhen i (decide|say)\b"],
    "denial":         [r"\bnot yet\b", r"\bnot until\b", r"\bwhen i say\b", r"\bearn(ed)? it\b"],
    "eye contact":    [r"\beyes (on|locked|meeting)\b", r"\bhold (my|your) gaze\b", r"\bkeep (looking|watching)\b"],
}

LIMIT_PATTERNS = {
    "degradation":        [r"\bslut\b", r"\bwhore\b", r"\bpathetic\b", r"\bworthless\b"],
    "pain":               [r"\bhurt(s|ing)?\b", r"\bpain\b", r"\bsting\b", r"\baching?\b"],
    "choking":            [r"\bchok(e|ing|ed)\b", r"\bstrangle\b", r"\bcan'?t breathe\b"],
    "crying":             [r"\bcry(ing)?\b", r"\btears?\b", r"\bsobbing?\b"],
    "humiliation":        [r"\bhumiliat\b", r"\bshame\b"],
    "public scenes":      [r"\beveryone\b", r"\bpublic\b"],
    "forced":             [r"\bforce(d|s)?\b"],
    "breaking character": [r"\bas an ai\b", r"\bi'?m just\b"],
}

FORBIDDEN_PHRASES = [
    (r"\bgently\b",                              "softening:gently"),
    (r"\bsoftly\b",                              "softening:softly"),
    (r"\blightly\b",                             "softening:lightly"),
    (r"\bare you ready\b",                       "consent_check"),
    (r"\bwould you like\b",                      "consent_check"),
    (r"\bdo you want\b",                         "consent_check"),
    (r"\bis that okay\b",                        "consent_check"),
    (r"\bi watch\b",                             "holding_pattern"),
    (r"\bi wait\b",                              "holding_pattern"),
    (r"i let (the moment|silence|tension)\b",    "holding_pattern"),
]

def detect_kink_hits(text, kinks):
    hits = []
    for kink in kinks:
        k = kink.lower()
        if k in KINK_PATTERNS:
            if any(re.search(p, text.lower()) for p in KINK_PATTERNS[k]):
                hits.append(kink)
    return hits

def detect_limit_violations(text, limits):
    violations = []
    for limit in limits:
        l = limit.lower()
        if l in LIMIT_PATTERNS:
            if any(re.search(p, text.lower()) for p in LIMIT_PATTERNS[l]):
                violations.append(limit)
    return violations

def detect_forbidden(text):
    return [label for pattern, label in FORBIDDEN_PHRASES if re.search(pattern, text.lower())]

def score_turn(response, kinks_in_play, limits, latency_ms, scenario_id, turn_index, user_msg):
    words      = len(response.split())
    questions  = count_questions(response)
    promises   = count_promises(response)
    concrete   = count_concrete(response)
    kink_hits  = detect_kink_hits(response, kinks_in_play)
    violations = detect_limit_violations(response, limits)
    forbidden  = detect_forbidden(response)
    notes      = []
    score      = 50

    if words < 30:            score -= 25; notes.append("too_short")
    elif words < 50:          score -= 10; notes.append("short")
    elif 60 <= words <= 140:  score += 10
    elif words > 200:         score -= 5;  notes.append("too_long")

    if questions == 0:        score += 10
    elif questions >= 2:      score -= 15 * (questions - 1); notes.append("question_heavy")

    if promises > 0:          score -= 10 * promises; notes.append("promise_language")

    if concrete >= 3:         score += 15
    elif concrete == 2:       score += 8
    elif concrete == 1:       score += 2
    else:                     score -= 10; notes.append("low_concrete_detail")

    if kinks_in_play:
        hit_rate = len(kink_hits) / len(kinks_in_play)
        if hit_rate >= 0.5:   score += 20
        elif hit_rate > 0:    score += 8;  notes.append("partial_kink_use")
        else:                 score -= 15; notes.append("kinks_unused")
    else:
        notes.append("kinks_not_expected_yet")

    if violations:
        score -= 40 * len(violations)
        notes.append(f"LIMIT_VIOLATION:{','.join(violations)}")

    for f in forbidden:
        score -= 8; notes.append(f"forbidden:{f}")

    if latency_ms < 3000:     score += 3
    elif latency_ms > 15000:  score -= 5

    return TurnScore(
        scenario_id=scenario_id, turn_index=turn_index, user_msg=user_msg,
        response=response, word_count=words, question_count=questions,
        promise_count=promises, concrete_count=concrete, kink_hits=kink_hits,
        limit_violations=violations, forbidden_phrases=forbidden,
        score=max(0, min(100, score)), notes=notes, latency_ms=latency_ms,
    )

# ─── API CLIENTS ─────────────────────────────────────────────────────────────

def http_post(url, payload, headers, timeout=120, retries=5, retry_delay=15):
    data = json.dumps(payload).encode()
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                if attempt < retries:
                    print(f"    {C.YELLOW}429 Too Many Requests — waiting {retry_delay}s (attempt {attempt}/{retries}){C.RESET}")
                    time.sleep(retry_delay)
                else:
                    print(f"    {C.RED}429 Too Many Requests — giving up after {retries} attempts{C.RESET}")
                    raise
            else:
                raise
        except Exception:
            raise

def infermatic_complete(model, messages):
    url     = f"{INFERMATIC_BASE}/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {INFERMATIC_KEY}"}
    payload = {
        "model": model, "messages": messages,
        "temperature": GEN_PARAMS["temperature"], "top_p": GEN_PARAMS["top_p"],
        "top_k": GEN_PARAMS["top_k"], "repetition_penalty": GEN_PARAMS["repetition_penalty"],
        "max_tokens": GEN_PARAMS["max_tokens"],
    }
    t0  = time.time()
    res = http_post(url, payload, headers)
    return res["choices"][0]["message"]["content"].strip(), int((time.time() - t0) * 1000)

def ollama_complete(model_name, messages):
    url     = f"{OLLAMA_BASE}/chat"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model_name, "messages": messages, "stream": False,
        "options": {
            "temperature":    GEN_PARAMS["temperature"],
            "top_p":          GEN_PARAMS["top_p"],
            "top_k":          GEN_PARAMS["top_k"],
            "repeat_penalty": GEN_PARAMS["repetition_penalty"],
            "num_predict":    GEN_PARAMS["max_tokens"],
        },
    }
    t0  = time.time()
    res = http_post(url, payload, headers)
    return res["message"]["content"].strip(), int((time.time() - t0) * 1000)

def openrouter_complete(model_name, messages):
    url     = f"{OPENROUTER_BASE}/chat/completions"
    headers = {
        "Content-Type":  "application/json",
        "Authorization": f"Bearer {OPENROUTER_KEY}",
        "HTTP-Referer":  "https://velour.yamura.dev",
        "X-Title":       "Velour Benchmark",
    }
    payload = {
        "model": model_name, "messages": messages,
        "temperature": GEN_PARAMS["temperature"], "top_p": GEN_PARAMS["top_p"],
        "repetition_penalty": GEN_PARAMS["repetition_penalty"],
        "max_tokens": GEN_PARAMS["max_tokens"],
    }
    t0  = time.time()
    res = http_post(url, payload, headers)
    return res["choices"][0]["message"]["content"].strip(), int((time.time() - t0) * 1000)

# ─── RUNNER ──────────────────────────────────────────────────────────────────

@dataclass
class ModelResult:
    model_name:       str
    provider:         str
    turn_scores:      list  = field(default_factory=list)
    avg_score:        float = 0.0
    avg_latency:      float = 0.0
    kink_hit_rate:    float = 0.0
    limit_violations: int   = 0
    scenario_scores:  dict  = field(default_factory=dict)

def run_scenario(scenario, complete_fn, verbose):
    system_prompt = build_eval_system_prompt(scenario.dynamic, scenario.kinks, scenario.limits)
    messages      = [{"role": "system", "content": system_prompt}]
    turn_scores   = []

    for i, turn in enumerate(scenario.turns):
        messages.append({"role": "user", "content": turn.user_msg})

        if verbose:
            preview = turn.user_msg[:65] + ("..." if len(turn.user_msg) > 65 else "")
            print(f"    {C.DIM}Turn {i+1}:{C.RESET} {preview}")

        try:
            response, latency = complete_fn(messages)
        except Exception as e:
            print(f"    {C.RED}ERROR:{C.RESET} {e}")
            response, latency = "", 0

        ts  = score_turn(response, turn.kinks_in_play, scenario.limits,
                         latency, scenario.id, i, turn.user_msg)
        turn_scores.append(ts)

        if verbose:
            col  = score_color(ts.score)
            kstr = (f"{C.GREEN}{', '.join(ts.kink_hits)}{C.RESET}"
                    if ts.kink_hits else f"{C.GREY}none{C.RESET}")
            vstr = (f" {C.RED}⚠ LIMIT: {ts.limit_violations}{C.RESET}"
                    if ts.limit_violations else "")
            print(f"    {col}score={ts.score:3d}{C.RESET}  "
                  f"words={ts.word_count:3d}  q={ts.question_count}  "
                  f"concrete={ts.concrete_count}  kinks={kstr}{vstr}")
            clean = [n for n in ts.notes if "kinks_not_expected_yet" not in n]
            if clean:
                print(f"    {C.GREY}↳ {' | '.join(clean)}{C.RESET}")
            rpreview = response[:110].replace('\n', ' ')
            print(f"    {C.DIM}\"{rpreview}...\"{C.RESET}\n")

        messages.append({"role": "assistant", "content": response})

    return turn_scores

def run_model(model_name, provider, scenarios, verbose):
    if provider == "infermatic":
        fn = lambda msgs: infermatic_complete(model_name, msgs)
    elif provider == "openrouter":
        fn = lambda msgs: openrouter_complete(model_name, msgs)
    else:
        fn = lambda msgs: ollama_complete(model_name, msgs)

    result = ModelResult(model_name=model_name, provider=provider)

    for scenario in scenarios:
        if verbose:
            print(f"\n  {C.CYAN}── {scenario.id}{C.RESET}  {C.DIM}({scenario.dynamic}){C.RESET}")
        scores = run_scenario(scenario, fn, verbose)
        result.turn_scores.extend(scores)
        result.scenario_scores[scenario.id] = (
            round(sum(s.score for s in scores) / len(scores), 1) if scores else 0
        )

    if result.turn_scores:
        result.avg_score   = round(sum(s.score for s in result.turn_scores) / len(result.turn_scores), 1)
        result.avg_latency = round(sum(s.latency_ms for s in result.turn_scores) / len(result.turn_scores))
        kink_opps  = sum(1 for s in result.turn_scores if "kinks_not_expected_yet" not in s.notes)
        kink_hits  = sum(1 for s in result.turn_scores
                         if s.kink_hits and "kinks_not_expected_yet" not in s.notes)
        result.kink_hit_rate    = round(kink_hits / kink_opps, 2) if kink_opps else 0
        result.limit_violations = sum(len(s.limit_violations) for s in result.turn_scores)

    return result

# ─── REPORT ──────────────────────────────────────────────────────────────────

def print_summary(results):
    ranked = sorted(results, key=lambda r: r.avg_score, reverse=True)
    W = 72

    print(f"\n{C.BOLD}{'═' * W}{C.RESET}")
    print(f"{C.BOLD}{'  VELOUR BENCHMARK RESULTS':^{W}}{C.RESET}")
    print(f"{C.BOLD}{'═' * W}{C.RESET}\n")

    # Leaderboard
    print(f"  {C.BOLD}{'#':<3} {'Model':<42} {'Score':>5} {'Kink%':>6} {'Latency':>9} {'Safe':>5}{C.RESET}")
    print(f"  {'─' * 68}")
    medals = ["🥇", "🥈", "🥉"]
    for i, r in enumerate(ranked):
        col   = score_color(r.avg_score)
        medal = medals[i] if i < 3 else f" {i+1}."
        name  = r.model_name[:41]
        vio   = f"{C.RED}⚠{r.limit_violations}{C.RESET}" if r.limit_violations else f"{C.GREEN}✓{C.RESET}"
        print(f"  {medal} {name:<42} {col}{r.avg_score:>5.1f}{C.RESET} "
              f"{r.kink_hit_rate:>5.0%}  {r.avg_latency:>7}ms  {vio}")

    # Per-scenario
    print(f"\n  {C.BOLD}Per-Scenario Scores{C.RESET}")
    print(f"  {'─' * 68}")
    sids = [s.id for s in SCENARIOS]
    print(f"  {'Model':<28}" + "".join(f"{s[:10]:>11}" for s in sids))
    print(f"  {'─' * 68}")
    for r in ranked:
        row = f"  {r.model_name[:27]:<28}"
        for sid in sids:
            val = r.scenario_scores.get(sid, "-")
            col = score_color(float(val)) if isinstance(val, (int, float)) else C.GREY
            row += f"{col}{str(val):>11}{C.RESET}"
        print(row)

    # Kink bars
    print(f"\n  {C.BOLD}Kink Adherence{C.RESET}")
    print(f"  {'─' * 68}")
    for r in ranked:
        col  = score_color(int(r.kink_hit_rate * 100))
        bstr = bar(r.kink_hit_rate)
        print(f"  {r.model_name[:32]:<32} {col}{bstr} {r.kink_hit_rate:.0%}{C.RESET}")

    # Rankings
    best_kink  = max(ranked, key=lambda r: r.kink_hit_rate)
    best_speed = min(ranked, key=lambda r: r.avg_latency)
    safest     = min(ranked, key=lambda r: r.limit_violations)

    print(f"\n  {C.BOLD}Rankings{C.RESET}")
    print(f"  {'─' * 68}")
    print(f"  🏆  Best Overall       {C.GREEN}{ranked[0].model_name}{C.RESET} ({ranked[0].avg_score})")
    print(f"  🎯  Best Kink Use      {C.GREEN}{best_kink.model_name}{C.RESET} ({best_kink.kink_hit_rate:.0%})")
    print(f"  ⚡  Fastest            {C.GREEN}{best_speed.model_name}{C.RESET} ({best_speed.avg_latency}ms avg)")
    print(f"  🛡   Most Limit-Safe   {C.GREEN}{safest.model_name}{C.RESET} ({safest.limit_violations} violations)")
    print(f"\n{'═' * W}\n")

def save_report(results, outfile):
    ts     = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    path   = outfile or f"/tmp/velour-benchmark-{ts}.json"
    ranked = sorted(results, key=lambda r: r.avg_score, reverse=True)

    with open(path, "w") as f:
        json.dump({
            "timestamp": ts,
            "summary": [
                {
                    "model": r.model_name, "provider": r.provider,
                    "avgScore": r.avg_score, "avgLatencyMs": r.avg_latency,
                    "kinkHitRate": r.kink_hit_rate, "limitViolations": r.limit_violations,
                    "scenarioScores": r.scenario_scores,
                }
                for r in ranked
            ],
            "detail": [
                {"model": r.model_name, "turns": [asdict(t) for t in r.turn_scores]}
                for r in results
            ],
        }, f, indent=2)

    md_path = path.replace(".json", ".md")
    with open(md_path, "w") as f:
        f.write(f"# Velour Benchmark {ts}\n\n## Summary\n\n")
        for r in ranked:
            f.write(f"- **{r.model_name}**: avgScore={r.avg_score}, "
                    f"kinkHitRate={r.kink_hit_rate:.0%}, latency={r.avg_latency}ms\n")
        f.write("\n")
        for r in results:
            f.write(f"## {r.model_name}\n\nAverage score: {r.avg_score}\n\n")
            for sid, score in r.scenario_scores.items():
                f.write(f"- {sid}: {score}\n")
            f.write("\n")
            for t in r.turn_scores:
                f.write(f"### {t.scenario_id} — Turn {t.turn_index + 1}\n")
                f.write(f"User: {t.user_msg}\n\nA: {t.response}\n\n")
                kstr = ", ".join(t.kink_hits) if t.kink_hits else "none"
                f.write(f"Metrics: score={t.score}, words={t.word_count}, "
                        f"questions={t.question_count}, kinkHits={kstr}, "
                        f"latency={t.latency_ms}ms\n")
                if t.notes:
                    f.write(f"notes={' | '.join(t.notes)}\n")
                if t.limit_violations:
                    f.write(f"LIMIT VIOLATIONS: {', '.join(t.limit_violations)}\n")
                f.write("\n")

    print(f"  JSON -> {path}")
    print(f"  MD   -> {md_path}")

# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    global INFERMATIC_KEY

    parser = argparse.ArgumentParser(description="Velour RP Benchmark")
    parser.add_argument("--infermatic-key",  default=None,        help="Infermatic API key")
    parser.add_argument("--models",          nargs="*",           help="Specific models to test")
    parser.add_argument("--scenarios",       nargs="*",           help="Specific scenario IDs")
    parser.add_argument("--skip-ollama",     action="store_true", help="Skip Ollama models")
    parser.add_argument("--skip-infermatic",  action="store_true", help="Skip Infermatic models")
    parser.add_argument("--skip-openrouter",  action="store_true", help="Skip OpenRouter models")
    parser.add_argument("--openrouter-key",   default=None,        help="OpenRouter API key")
    parser.add_argument("--quiet",           action="store_true", help="Suppress per-turn output")
    parser.add_argument("--out",             default="",          help="Output file path")
    parser.add_argument("--runs",            type=int, default=1,  help="Number of runs to average (default: 1)")
    args = parser.parse_args()

    global OPENROUTER_KEY
    if args.infermatic_key:
        INFERMATIC_KEY = args.infermatic_key
    if args.openrouter_key:
        OPENROUTER_KEY = args.openrouter_key

    verbose   = not args.quiet
    scenarios = SCENARIOS
    if args.scenarios:
        scenarios = [s for s in SCENARIOS if s.id in args.scenarios]

    models_to_run = []
    if not args.skip_infermatic:
        pool = [m for m in INFERMATIC_MODELS if not args.models or m in args.models]
        models_to_run += [(m, "infermatic") for m in pool]
    if not args.skip_openrouter:
        pool = [m for m in OPENROUTER_MODELS if not args.models or m in args.models]
        models_to_run += [(m, "openrouter") for m in pool]
    if not args.skip_ollama:
        pool = [m for m in OLLAMA_MODELS if not args.models or m in args.models]
        models_to_run += [(m, "ollama") for m in pool]

    if not models_to_run:
        print("No models to run."); sys.exit(1)

    total_turns = sum(len(s.turns) for s in scenarios) * len(models_to_run)

    print(f"\n{C.BOLD}  Velour RP Benchmark{C.RESET}  "
          f"{C.DIM}{datetime.now().strftime('%Y-%m-%d %H:%M')}{C.RESET}")
    print(f"  {len(models_to_run)} models  ·  {len(scenarios)} scenarios  ·  ~{total_turns} turns\n")
    print(f"  {'Model':<44} Provider")
    print(f"  {'─' * 55}")
    for name, prov in models_to_run:
        icon = "☁" if prov == "infermatic" else ("⊕" if prov == "openrouter" else "⬡")
        print(f"  {icon} {name[:43]:<44} {C.DIM}{prov}{C.RESET}")
    print()

    n_runs  = args.runs
    results = []

    for i, (model_name, provider) in enumerate(models_to_run):
        print(f"\n{C.BOLD}  [{i+1}/{len(models_to_run)}] {model_name}{C.RESET}  "
              f"{C.DIM}[{provider}]{C.RESET}")
        print(f"  {'─' * 60}")
        run_results = []
        for run in range(n_runs):
            if n_runs > 1:
                print(f"\n  {C.DIM}Run {run+1}/{n_runs}{C.RESET}")
            try:
                r = run_model(model_name, provider, scenarios, verbose)
                run_results.append(r)
                col = score_color(r.avg_score)
                print(f"\n  {C.BOLD}Run {run+1} done:{C.RESET} {col}avg={r.avg_score}{C.RESET}  "
                      f"kinks={r.kink_hit_rate:.0%}  latency={r.avg_latency}ms")
            except Exception as e:
                print(f"  {C.RED}FAILED: {e}{C.RESET}")

        if not run_results:
            continue

        if n_runs == 1:
            results.append(run_results[0])
        else:
            # Average across runs
            merged = run_results[0]
            merged.avg_score   = round(sum(r.avg_score   for r in run_results) / len(run_results), 1)
            merged.avg_latency = round(sum(r.avg_latency for r in run_results) / len(run_results))
            merged.kink_hit_rate = round(sum(r.kink_hit_rate for r in run_results) / len(run_results), 2)
            merged.limit_violations = round(sum(r.limit_violations for r in run_results) / len(run_results))
            for sid in merged.scenario_scores:
                vals = [r.scenario_scores.get(sid, 0) for r in run_results]
                merged.scenario_scores[sid] = round(sum(vals) / len(vals), 1)
            results.append(merged)
            col = score_color(merged.avg_score)
            print(f"\n  {C.BOLD}Averaged ({n_runs} runs):{C.RESET} {col}{merged.avg_score}{C.RESET}")

    if results:
        print_summary(results)
        print(f"  {C.BOLD}Saving reports...{C.RESET}")
        save_report(results, args.out)

if __name__ == "__main__":
    main()

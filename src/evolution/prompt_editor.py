"""Prompt editor — auto-append quarterly meta-reflection learnings into
agent prompt files under strict Python guards.

Design invariants (the schema half already enforces some; editor enforces
the rest regardless of LLM):

1. **Append-only for a given quarter.** Every auto-added bullet carries
   a quarter tag ("[2026-Q1]") and a content-hash HTML comment used by
   the retract path. The editor never edits or deletes existing text
   written outside the bullet it's authoring — human-written prompt
   content is safe.

2. **Allow-list for target agents.** risk_manager and position_reviewer
   are listed in config.evolution.protected_agents; any learning
   targeting them is rejected. The PromptLearning Pydantic literal
   already excludes them; this is the second belt.

3. **Length ceiling + floor** via config (max_learning_chars,
   min_justification_chars) — prevents prompt bloat and empty learnings
   slipping through schema loopholes.

4. **Prohibited-word tripwire** (word-boundary regex, case-insensitive):
   "never", "always", "override", "ignore all", "must always",
   "must never". These directly conflict with hard-invariant wording
   that already lives in the core prompts; an "always" stomp in
   auto-appended learnings could silently flip discipline.

5. **Jaccard token-similarity dedup** against existing entries in the
   target file — catches paraphrases. Threshold configurable
   (default 0.6 — loose enough that related but differently-phrased
   learnings coexist; tight enough to reject near-copies).

6. **Per-agent FIFO cap.** When appending would push count past
   max_learnings_per_agent, the OLDEST auto-appended entry (identified
   by its HTML hash comment) is removed before the new one is appended.
   The section never grows unbounded.

7. **Per-cycle agent cap.** Across a single apply_reflection call,
   at most max_agents_per_cycle distinct agent prompts get edited.
   Schema already caps learnings at 3; this is the second belt in
   case an operator manually feeds a reflection.

8. **Atomic file write** (tmp + os.replace). Guards against partial
   writes leaving a broken prompt that would fail the next LLM call.

9. **Audit log** at data/evolution/edits.jsonl — every accepted AND
   rejected attempt, with reason. Partial rollback / postmortem is
   always possible from this log.

10. **Optional git auto-commit.** After all learnings in one
    `apply_reflection` are processed, stage + commit the modified
    prompt files in a single commit. `git revert` on that SHA
    undoes a whole quarter's evolution in one shot. Commit failures
    are logged and swallowed — prompt edits aren't rolled back if
    git misbehaves (we already wrote the file atomically).
    To keep that revert clean, a prompt file that already carries
    uncommitted operator edits is SKIPPED (rejection logged) rather
    than swept into the evolution commit (audit round 2, #48).

11. **Single-line entries.** learning_text is whitespace-normalized
    (newlines/tabs/runs collapsed to single spaces) BEFORE any
    guardrail runs and before writing — entries are line-based, so an
    embedded newline would otherwise defeat FIFO cap, Jaccard dedup,
    prohibited-words and the retract path all at once (audit round 2,
    #21).

12. **Apply-from-file lane (audit round 2, #20).** The documented
    human-review gate is "review proposed_edits.json, then apply". A
    plain `--mode meta --force` re-run REGENERATES the reflection with
    a fresh non-deterministic LLM call — applying content nobody
    reviewed. Setting env `EVOLUTION_APPLY_SAVED=1` (use the incoming
    reflection's period) or `EVOLUTION_APPLY_SAVED=<period>` (e.g.
    `2026-Q2`, for late applies where the fresh run would mislabel the
    period) makes `apply_reflection` DISCARD the freshly-generated
    reflection and instead apply the persisted
    `data/evolution/{period}/reflection.json` — exactly what the human
    reviewed. If the saved file is missing/invalid the editor fails
    safe: nothing is applied. See `load_saved_reflection`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.config import EvolutionConfig
    from src.models import PromptLearning, QuarterlyMetaReflection

logger = logging.getLogger(__name__)






SECTION_HEADER = "## Learnings (system-evolved)"
SECTION_PREAMBLE = (
    "<!-- Entries auto-appended by quarterly meta-reflection. Do not edit by hand.\n"
    "     Oldest-first FIFO rolloff when count exceeds max_learnings_per_agent.\n"
    "     Every entry carries a content hash in its trailing HTML comment;\n"
    "     retract ops target that hash. See src/evolution/prompt_editor.py. -->"
)


_ENTRY_RE = re.compile(
    r"^- \[(?P<period>[^]]+)\] (?P<text>.+?) <!--hash:(?P<hash>[0-9a-f]+)-->\s*$"
)






@dataclass
class AppliedEdit:
    agent_name: str
    operation: str           
    learning_text: str
    content_hash: str
    period: str
    prompt_path: str


@dataclass
class Rejection:
    agent_name: str
    operation: str
    learning_text: str
    reason: str
    period: str = ""


@dataclass
class ApplicationReport:
    period: str
    applied: list[AppliedEdit] = field(default_factory=list)
    rejected: list[Rejection] = field(default_factory=list)
    rolled_off: list[dict] = field(default_factory=list)  
    git_commit: str | None = None

    @property
    def agents_edited(self) -> int:
        return len({e.agent_name for e in self.applied})

    def to_dict(self) -> dict:
        return {
            "period": self.period,
            "applied": [e.__dict__ for e in self.applied],
            "rejected": [r.__dict__ for r in self.rejected],
            "rolled_off": self.rolled_off,
            "agents_edited": self.agents_edited,
            "git_commit": self.git_commit,
        }






class PromptEditor:
    def __init__(
        self,
        config: "EvolutionConfig",
        prompts_dir: Path | str,
        evolution_dir: Path | str = "data/evolution",
        auto_commit: bool | None = None,
        dry_run: bool | None = None,
    ):
        self.config = config
        self.prompts_dir = Path(prompts_dir)
        self.evolution_dir = Path(evolution_dir)
        self.evolution_dir.mkdir(parents=True, exist_ok=True)
        
        self._auto_commit = (
            auto_commit if auto_commit is not None else config.auto_commit
        )
        self._dry_run = (
            dry_run if dry_run is not None else config.dry_run
        )
        
        
        
        
        parts = []
        for phrase in (config.prohibited_words or []):
            tokens = [re.escape(t) for t in phrase.strip().split() if t.strip()]
            if not tokens:
                continue
            parts.append(r"\b" + r"\s+".join(tokens) + r"\b")
        self._prohibited_re = re.compile(
            "|".join(parts), re.IGNORECASE,
        ) if parts else None

    

    def apply_reflection(
        self,
        reflection: "QuarterlyMetaReflection | dict",
    ) -> ApplicationReport:
        """Apply every proposed_learning in `reflection` in order. Respects:
          - config.enabled: short-circuit with all-rejected report when off
          - config.max_agents_per_cycle across the whole call
          - Pydantic-layer invariants already on the reflection object

        `reflection` may also be a plain dict (e.g. a reflection.json
        loaded from disk); it is validated through the
        QuarterlyMetaReflection schema before anything runs.

        Env `EVOLUTION_APPLY_SAVED` (audit round 2, #20): when set, the
        incoming (freshly-generated) reflection is DISCARDED and the
        persisted `data/evolution/{period}/reflection.json` is applied
        instead — so what the human reviewed is exactly what gets
        applied. Value `1`/`true`/`yes` → same period as the incoming
        reflection; any other value → explicit period (e.g. `2026-Q2`).
        Missing/invalid saved file → fail safe, nothing applied.
        """
        
        
        if isinstance(reflection, dict):
            from src.models import QuarterlyMetaReflection
            reflection = QuarterlyMetaReflection.model_validate(reflection)

        
        saved_flag = os.getenv("EVOLUTION_APPLY_SAVED", "").strip()
        if saved_flag and saved_flag != "0":
            period = (
                reflection.period
                if saved_flag.lower() in ("1", "true", "yes")
                else saved_flag
            )
            saved = load_saved_reflection(period, evolution_dir=self.evolution_dir)
            if saved is None:
                
                
                logger.error(
                    "EVOLUTION_APPLY_SAVED=%s but no valid saved reflection "
                    "for period %s under %s — applying NOTHING (fail-safe; "
                    "the freshly-generated reflection is not a reviewed "
                    "artifact)", saved_flag, period, self.evolution_dir,
                )
                report = ApplicationReport(period=period)
                for learning in reflection.proposed_learnings:
                    report.rejected.append(Rejection(
                        agent_name=learning.agent_name,
                        operation=learning.operation,
                        learning_text=learning.learning_text,
                        reason=(
                            f"EVOLUTION_APPLY_SAVED={saved_flag} set but "
                            f"{Path(self.evolution_dir) / period / 'reflection.json'} "
                            f"is missing/invalid — fresh reflection not applied"
                        ),
                        period=period,
                    ))
                self._audit_log(report)
                return report
            
            
            
            
            
            
            
            staged_path = (
                Path(self.evolution_dir) / period / "proposed_edits.json"
            )
            if staged_path.exists():
                mismatch = False
                try:
                    staged = json.loads(staged_path.read_text())
                    staged_set = {
                        (p.get("agent_name"), p.get("operation"),
                         p.get("learning_text"))
                        for p in (staged.get("proposals") or [])
                    }
                    saved_set = {
                        (ln.agent_name, ln.operation, ln.learning_text)
                        for ln in saved.proposed_learnings
                    }
                    mismatch = staged_set != saved_set
                except Exception as exc:  
                    logger.warning(
                        "EVOLUTION_APPLY_SAVED: could not cross-check %s "
                        "(%s); proceeding on reflection.json alone",
                        staged_path, exc,
                    )
                if mismatch:
                    logger.error(
                        "EVOLUTION_APPLY_SAVED=%s: %s does not match the "
                        "staged proposals in %s — reflection.json was "
                        "probably overwritten by a fresh same-period LLM "
                        "run before the editor read it. Applying NOTHING "
                        "(fail-safe).",
                        saved_flag,
                        Path(self.evolution_dir) / period / "reflection.json",
                        staged_path,
                    )
                    report = ApplicationReport(period=period)
                    for learning in saved.proposed_learnings:
                        report.rejected.append(Rejection(
                            agent_name=learning.agent_name,
                            operation=learning.operation,
                            learning_text=learning.learning_text,
                            reason=(
                                "EVOLUTION_APPLY_SAVED: reflection.json "
                                "disagrees with the reviewed "
                                "proposed_edits.json — not applied"
                            ),
                            period=period,
                        ))
                    self._audit_log(report)
                    return report
            logger.warning(
                "EVOLUTION_APPLY_SAVED=%s: applying SAVED reflection %s "
                "(%d proposed learning(s)); the freshly-generated "
                "reflection for %s is DISCARDED so that what the human "
                "reviewed is exactly what gets applied",
                saved_flag,
                Path(self.evolution_dir) / period / "reflection.json",
                len(saved.proposed_learnings),
                reflection.period,
            )
            reflection = saved

        report = ApplicationReport(period=reflection.period)

        
        
        
        
        
        if not self.config.enabled:
            effective_mode = "OFF — evolution.enabled=false (observe only, nothing staged)"
        elif self._dry_run:
            effective_mode = (
                "STAGE-ONLY — dry_run=true: proposals written to "
                "proposed_edits.json for human review, NO prompt files modified"
            )
        else:
            effective_mode = (
                "LIVE-APPLY — dry_run=false: proposals will be written into "
                "prompt files + git-committed"
            )
        logger.warning("PromptEditor effective mode: %s", effective_mode)

        if not self.config.enabled:
            
            for learning in reflection.proposed_learnings:
                report.rejected.append(Rejection(
                    agent_name=learning.agent_name,
                    operation=learning.operation,
                    learning_text=learning.learning_text,
                    reason="evolution.enabled=false (observe-only mode)",
                    period=reflection.period,
                ))
            self._audit_log(report)
            return report

        if self._dry_run:
            
            
            
            
            
            
            
            
            
            
            self._write_dry_run_proposal(reflection, report)
            self._audit_log(report)
            return report

        agents_edited: set[str] = set()
        modified_paths: set[Path] = set()

        for learning in reflection.proposed_learnings:
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            if (learning.agent_name not in agents_edited
                    and len(agents_edited) >= self.config.max_agents_per_cycle):
                report.rejected.append(Rejection(
                    agent_name=learning.agent_name,
                    operation=learning.operation,
                    learning_text=learning.learning_text,
                    reason=(
                        f"max_agents_per_cycle={self.config.max_agents_per_cycle} "
                        f"already reached"
                    ),
                    period=reflection.period,
                ))
                continue

            outcome = self._apply_one(learning, reflection.period, report)
            if outcome is not None:
                report.applied.append(outcome)
                agents_edited.add(outcome.agent_name)
                modified_paths.add(Path(outcome.prompt_path))

        
        if self._auto_commit and modified_paths and report.applied:
            sha = self._git_commit_changes(
                modified_paths, reflection.period, len(report.applied),
            )
            report.git_commit = sha

        self._audit_log(report)
        return report

    

    def _apply_one(
        self,
        learning: "PromptLearning",
        period: str,
        report: ApplicationReport,
    ) -> AppliedEdit | None:
        """Validate + apply one learning. Returns the AppliedEdit on success,
        or None after pushing a Rejection into the report."""
        
        
        
        
        
        
        
        
        
        normalized = " ".join(learning.learning_text.split())
        if normalized != learning.learning_text:
            logger.warning(
                "prompt_editor: learning_text for %s contained newlines/"
                "irregular whitespace — normalized to a single line before "
                "guardrail checks", learning.agent_name,
            )
            learning = learning.model_copy(
                update={"learning_text": normalized},
            )

        reason = self._validate_learning(learning)
        if reason is not None:
            report.rejected.append(Rejection(
                agent_name=learning.agent_name,
                operation=learning.operation,
                learning_text=learning.learning_text,
                reason=reason,
                period=period,
            ))
            return None

        prompt_path = self._prompt_path_for(learning.agent_name)
        if not prompt_path.exists():
            report.rejected.append(Rejection(
                agent_name=learning.agent_name,
                operation=learning.operation,
                learning_text=learning.learning_text,
                reason=f"prompt file not found: {prompt_path}",
                period=period,
            ))
            return None

        
        
        
        
        
        
        
        if self._auto_commit and self._prompt_file_dirty(prompt_path, report):
            logger.error(
                "prompt_editor: %s has uncommitted operator edits — "
                "skipping %s's learning so the evolution git commit stays "
                "revert-clean (commit or stash your changes, then re-apply)",
                prompt_path, learning.agent_name,
            )
            report.rejected.append(Rejection(
                agent_name=learning.agent_name,
                operation=learning.operation,
                learning_text=learning.learning_text,
                reason=(
                    f"prompt file has uncommitted operator edits "
                    f"({prompt_path}) — skipped to keep the evolution "
                    f"commit revert-clean"
                ),
                period=period,
            ))
            return None

        text = prompt_path.read_text()

        if learning.operation == "retract":
            if not learning.retract_target_hash:
                report.rejected.append(Rejection(
                    agent_name=learning.agent_name,
                    operation="retract",
                    learning_text=learning.learning_text,
                    reason="retract requires retract_target_hash",
                    period=period,
                ))
                return None
            new_text, removed = _remove_entry_by_hash(
                text, learning.retract_target_hash,
            )
            if not removed:
                report.rejected.append(Rejection(
                    agent_name=learning.agent_name,
                    operation="retract",
                    learning_text=learning.learning_text,
                    reason=(
                        f"retract_target_hash={learning.retract_target_hash} "
                        f"not present in {prompt_path.name}"
                    ),
                    period=period,
                ))
                return None
            try:
                _atomic_write(prompt_path, new_text)
            except OSError as exc:
                
                
                
                report.rejected.append(Rejection(
                    agent_name=learning.agent_name,
                    operation="retract",
                    learning_text=learning.learning_text,
                    reason=f"atomic write failed: {exc}",
                    period=period,
                ))
                return None
            return AppliedEdit(
                agent_name=learning.agent_name, operation="retract",
                learning_text=learning.learning_text,
                content_hash=learning.retract_target_hash,
                period=period, prompt_path=str(prompt_path),
            )

        
        existing = _parse_entries(text)
        content_hash = _hash_text(learning.learning_text)

        
        for e in existing:
            sim = _jaccard(learning.learning_text, e["text"])
            if sim >= self.config.jaccard_dedup_threshold:
                report.rejected.append(Rejection(
                    agent_name=learning.agent_name,
                    operation="append",
                    learning_text=learning.learning_text,
                    reason=(
                        f"jaccard_similarity={sim:.2f} ≥ "
                        f"{self.config.jaccard_dedup_threshold} vs existing "
                        f"entry [{e['period']}] hash={e['hash'][:6]}"
                    ),
                    period=period,
                ))
                return None

        new_text, rolled_off_entries = _append_entry(
            text,
            period=period,
            learning_text=learning.learning_text,
            content_hash=content_hash,
            max_entries=self.config.max_learnings_per_agent,
        )
        try:
            _atomic_write(prompt_path, new_text)
        except OSError as exc:
            
            
            
            report.rejected.append(Rejection(
                agent_name=learning.agent_name,
                operation="append",
                learning_text=learning.learning_text,
                reason=f"atomic write failed: {exc}",
                period=period,
            ))
            return None

        for roll in rolled_off_entries:
            report.rolled_off.append({
                "agent": learning.agent_name,
                "period": roll["period"],
                "hash": roll["hash"],
                "text": roll["text"],
            })

        return AppliedEdit(
            agent_name=learning.agent_name, operation="append",
            learning_text=learning.learning_text,
            content_hash=content_hash,
            period=period, prompt_path=str(prompt_path),
        )

    

    def _validate_learning(self, learning: "PromptLearning") -> str | None:
        """Returns a rejection reason string when the learning fails any
        Python-side guard, or None when it's OK to apply. Belt-and-braces
        with the Pydantic validators — if the schema lets something through
        that deployment config wants stricter, we catch it here."""
        if learning.agent_name in self.config.protected_agents:
            return f"agent {learning.agent_name!r} is in protected_agents"
        if len(learning.learning_text) > self.config.max_learning_chars:
            return (
                f"learning_text length {len(learning.learning_text)} > "
                f"max_learning_chars={self.config.max_learning_chars}"
            )
        if len(learning.justification) < self.config.min_justification_chars:
            return (
                f"justification length {len(learning.justification)} < "
                f"min_justification_chars={self.config.min_justification_chars}"
            )
        if self._prohibited_re is not None:
            match = self._prohibited_re.search(learning.learning_text)
            if match is not None:
                return (
                    f"learning_text contains prohibited word/phrase "
                    f"{match.group(0)!r}"
                )
        return None

    def _prompt_path_for(self, agent_name: str) -> Path:
        return self.prompts_dir / f"{agent_name}.md"

    def _prompt_file_dirty(
        self,
        prompt_path: Path,
        report: ApplicationReport,
    ) -> bool:
        """audit round 2 (#48): True when `prompt_path` has uncommitted
        changes (modified OR untracked) that predate this cycle.

        - Files WE already edited earlier in this same cycle are exempt
          (they're dirty because of us; committing them is the whole point).
        - Any git failure (not a repo, git missing, mocked subprocess in
          tests) degrades to "not dirty" — the sweep hazard only exists
          when the later `git add + commit` would actually succeed, and
          that path already handles git absence gracefully.
        """
        if str(prompt_path) in {e.prompt_path for e in report.applied}:
            return False
        try:
            proc = subprocess.run(
                [
                    "git", "-C", str(prompt_path.parent),
                    "status", "--porcelain", "--", str(prompt_path),
                ],
                capture_output=True, text=True, timeout=10,
            )
        except Exception:
            return False
        rc = proc.returncode
        if not isinstance(rc, int) or rc != 0:
            return False
        out = proc.stdout
        if isinstance(out, bytes):
            out = out.decode(errors="replace")
        return bool(out.strip()) if isinstance(out, str) else False

    

    def _write_dry_run_proposal(
        self,
        reflection: "QuarterlyMetaReflection",
        report: ApplicationReport,
    ) -> None:
        """Write proposed_edits.json to data/evolution/{period}/ when
        dry_run=True. Each entry includes everything an operator needs
        to review + manually apply: target agent, agent's current
        prompt path, proposed learning text, operation (append/retract),
        retract target hash (when applicable), and the reflector's
        justification.

        The file is atomic-written so concurrent meta runs don't leave
        a corrupt JSON. Existing file is overwritten (one quarter, one
        proposal).
        """
        period_dir = self.evolution_dir / reflection.period
        period_dir.mkdir(parents=True, exist_ok=True)
        out_path = period_dir / "proposed_edits.json"

        proposals: list[dict] = []
        for learning in reflection.proposed_learnings:
            proposals.append({
                "agent_name": learning.agent_name,
                "operation": learning.operation,
                "learning_text": learning.learning_text,
                "retract_target_hash": getattr(
                    learning, "retract_target_hash", None,
                ),
                "justification": getattr(learning, "justification", ""),
                "target_prompt_path": str(
                    self._prompt_path_for(learning.agent_name)
                ),
            })

        payload = {
            "period": reflection.period,
            "mode": "dry_run",
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
            "proposed_count": len(proposals),
            "proposals": proposals,
            
            
            
            
            
            "instructions": (
                "To apply these proposals EXACTLY as reviewed: (1) flip "
                "evolution.dry_run=false in config/settings.yaml, set env "
                f"EVOLUTION_APPLY_SAVED={reflection.period} and re-run "
                "`python main.py --mode meta --force` — the editor then "
                "applies the SAVED data/evolution/"
                f"{reflection.period}/reflection.json instead of whatever "
                "a fresh LLM run would regenerate; OR (2) edit the "
                "target_prompt_path file by hand, appending each "
                "learning_text to its `## Learnings (system-evolved)` "
                "section. Option (1) is reversible via `git revert`. "
                "WARNING: without EVOLUTION_APPLY_SAVED, re-running "
                "--mode meta --force makes a NEW non-deterministic LLM "
                "call and applies content nobody reviewed."
            ),
        }

        tmp = out_path.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
            os.replace(str(tmp), str(out_path))
        except Exception:
            tmp.unlink(missing_ok=True)
            raise

        
        
        
        for learning in reflection.proposed_learnings:
            report.rejected.append(Rejection(
                agent_name=learning.agent_name,
                operation=learning.operation,
                learning_text=learning.learning_text,
                reason=(
                    f"dry_run=True; proposal staged to {out_path} for "
                    f"operator review (set evolution.dry_run=false to apply)"
                ),
                period=reflection.period,
            ))

        logger.info(
            "PromptEditor dry-run: staged %d proposal(s) to %s",
            len(proposals), out_path,
        )

    

    def _audit_log(self, report: ApplicationReport) -> None:
        log_path = self.evolution_dir / "edits.jsonl"
        rows: list[dict] = []
        ts = datetime.now(tz=timezone.utc).isoformat()
        for e in report.applied:
            rows.append({"ts": ts, "period": report.period,
                         "kind": "applied", **e.__dict__})
        for r in report.rejected:
            rows.append({"ts": ts, "period": report.period,
                         "kind": "rejected", **r.__dict__})
        for roll in report.rolled_off:
            rows.append({"ts": ts, "period": report.period,
                         "kind": "rolled_off", **roll})
        if not rows:
            
            
            
            
            
            
            
            rows.append({
                "ts": ts, "period": report.period, "kind": "empty",
                "note": (
                    "apply_reflection ran with no applied/rejected/"
                    "rolled_off entries (reflection carried zero "
                    "proposed_learnings by the time it reached the editor)"
                ),
            })
        try:
            with log_path.open("a") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.warning("prompt_editor audit log write failed: %s", exc)

    def _git_commit_changes(
        self,
        paths: set[Path],
        period: str,
        n_learnings: int,
    ) -> str | None:
        """Stage + commit changed prompt files. Swallows all errors — the
        file mutations themselves are durable; a git hiccup doesn't
        warrant rolling them back. Returns the new commit SHA on success.
        """
        try:
            
            repo_root = self.prompts_dir.resolve()
            while repo_root != repo_root.parent:
                if (repo_root / ".git").exists():
                    break
                repo_root = repo_root.parent
            else:
                logger.warning("prompt_editor git_auto_commit: no .git found")
                return None

            path_args = [str(p.resolve()) for p in sorted(paths)]
            subprocess.run(
                ["git", "-C", str(repo_root), "add"] + path_args,
                check=True, capture_output=True,
            )
            msg = (
                f"chore(prompts): quarterly meta-reflection {period} — "
                f"{n_learnings} learning(s)"
            )
            commit_proc = subprocess.run(
                ["git", "-C", str(repo_root), "commit", "-m", msg, "--"] + path_args,
                check=True, capture_output=True, text=True,
            )
            sha_proc = subprocess.run(
                ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                check=True, capture_output=True, text=True,
            )
            return sha_proc.stdout.strip()
        except subprocess.CalledProcessError as exc:
            
            
            stderr = exc.stderr or ""
            if isinstance(stderr, bytes):
                stderr = stderr.decode(errors="replace")
            logger.warning(
                "prompt_editor git_auto_commit failed (rc=%s): %s",
                exc.returncode, stderr,
            )
            return None
        except Exception as exc:
            logger.warning("prompt_editor git_auto_commit unexpected: %s", exc)
            return None






def load_saved_reflection(
    period: str,
    *,
    evolution_dir: Path | str = "data/evolution",
) -> "QuarterlyMetaReflection | None":
    """Load + schema-validate the persisted reflection for `period` from
    `{evolution_dir}/{period}/reflection.json` (written by
    meta_reflector.persist_reflection at quarter end).

    This is the artifact the operator actually reviewed alongside
    proposed_edits.json — feeding it back through
    `PromptEditor.apply_reflection` (via env EVOLUTION_APPLY_SAVED, or
    directly) guarantees the applied content is byte-identical to the
    reviewed content, instead of a fresh non-deterministic LLM
    regeneration. Returns None (with an ERROR log) when the file is
    missing, unparseable, or fails QuarterlyMetaReflection validation —
    callers must treat None as "apply nothing".
    """
    from src.models import QuarterlyMetaReflection

    path = Path(evolution_dir) / str(period) / "reflection.json"
    if not path.exists():
        logger.error("load_saved_reflection: %s does not exist", path)
        return None
    try:
        data = json.loads(path.read_text())
        reflection = QuarterlyMetaReflection.model_validate(data)
    except Exception as exc:
        logger.error(
            "load_saved_reflection: %s failed to parse/validate: %s",
            path, exc,
        )
        return None
    logger.info(
        "load_saved_reflection: loaded reviewed reflection %s "
        "(%d proposed learning(s))",
        path, len(reflection.proposed_learnings),
    )
    return reflection






def _hash_text(text: str) -> str:
    """Stable content hash for retract-targeting. First 12 hex chars of
    SHA-256 — low collision probability for the corpus size (≤ 10
    entries × 6 agents × many years).

    audit round 2 (#21): whitespace-normalized (not just stripped) so the
    hash computed from an LLM's newline-bearing learning_text matches the
    hash stored on the single-line entry `_append_entry` writes — retract
    lookups must agree with what's on disk."""
    normalized = " ".join(text.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]


def _jaccard(a: str, b: str) -> float:
    """Token-level Jaccard similarity (case-insensitive, length-2+ tokens)."""
    ta = {t for t in re.findall(r"[A-Za-z0-9]{2,}", a.lower())}
    tb = {t for t in re.findall(r"[A-Za-z0-9]{2,}", b.lower())}
    if not ta and not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


def _parse_entries(full_text: str) -> list[dict]:
    """Extract the ordered list of (period, text, hash) entries currently
    in the Learnings section. Returns [] when the section is absent or
    empty."""
    body = _extract_section_body(full_text)
    if body is None:
        return []
    entries: list[dict] = []
    for line in body.splitlines():
        m = _ENTRY_RE.match(line)
        if not m:
            continue
        entries.append({
            "period": m.group("period"),
            "text":   m.group("text").strip(),
            "hash":   m.group("hash"),
        })
    return entries


def _extract_section_body(full_text: str) -> str | None:
    """Return the Learnings section body (without the header line), or
    None when the section doesn't exist in this file. The body continues
    to the next `^## ` header OR end-of-file."""
    lines = full_text.splitlines(keepends=False)
    try:
        start = next(i for i, line in enumerate(lines)
                     if line.strip() == SECTION_HEADER)
    except StopIteration:
        return None
    
    end = len(lines)
    for i in range(start + 1, len(lines)):
        if lines[i].startswith("## "):
            end = i
            break
    return "\n".join(lines[start + 1:end])


def _append_entry(
    full_text: str,
    *,
    period: str,
    learning_text: str,
    content_hash: str,
    max_entries: int,
) -> tuple[str, list[dict]]:
    """Append a new entry to the Learnings section, creating the section
    if absent. Enforces FIFO by removing the OLDEST auto-entry when count
    would exceed `max_entries`. Returns (new_file_text, rolled_off_list)."""
    text = full_text.rstrip() + "\n"  
    
    
    new_entry = (
        f"- [{period}] {' '.join(learning_text.split())} "
        f"<!--hash:{content_hash}-->"
    )

    if _extract_section_body(text) is None:
        
        block = (
            "\n" + SECTION_HEADER + "\n"
            + SECTION_PREAMBLE + "\n"
            + new_entry + "\n"
        )
        return text + block, []

    
    lines = text.splitlines(keepends=False)
    start = next(i for i, line in enumerate(lines)
                 if line.strip() == SECTION_HEADER)
    end = len(lines)
    for i in range(start + 1, len(lines)):
        if lines[i].startswith("## "):
            end = i
            break
    body_lines = lines[start + 1:end]

    
    
    
    
    
    
    
    entry_lines: list[str] = []
    preamble: list[str] = []
    other: list[str] = []
    first_entry_seen = False
    for line in body_lines:
        if _ENTRY_RE.match(line):
            entry_lines.append(line)
            first_entry_seen = True
        elif not first_entry_seen:
            
            
            
            preamble.append(line)
        else:
            
            
            
            other.append(line)

    
    rolled_off: list[dict] = []
    target_existing = max_entries - 1  
    while len(entry_lines) > max(0, target_existing):
        dropped = entry_lines.pop(0)
        m = _ENTRY_RE.match(dropped)
        if m:
            rolled_off.append({
                "period": m.group("period"),
                "text":   m.group("text").strip(),
                "hash":   m.group("hash"),
            })

    entry_lines.append(new_entry)

    
    
    if not preamble:
        preamble = [SECTION_PREAMBLE]
    new_body = "\n".join(preamble + entry_lines + other)

    new_lines = lines[:start + 1] + [new_body] + lines[end:]
    return "\n".join(new_lines).rstrip() + "\n", rolled_off


def _remove_entry_by_hash(full_text: str, target_hash: str) -> tuple[str, bool]:
    """Delete the bullet whose hash comment matches `target_hash`. Returns
    (new_text, removed). removed=False when hash absent — caller handles
    as a rejection."""
    body = _extract_section_body(full_text)
    if body is None:
        return full_text, False

    lines = full_text.splitlines(keepends=False)
    start = next(i for i, line in enumerate(lines)
                 if line.strip() == SECTION_HEADER)
    end = len(lines)
    for i in range(start + 1, len(lines)):
        if lines[i].startswith("## "):
            end = i
            break

    new_lines: list[str] = list(lines[:start + 1])
    removed = False
    for line in lines[start + 1:end]:
        m = _ENTRY_RE.match(line)
        if m and m.group("hash") == target_hash:
            removed = True
            continue
        new_lines.append(line)
    new_lines.extend(lines[end:])
    return "\n".join(new_lines).rstrip() + "\n", removed


def _atomic_write(path: Path, content: str) -> None:
    """Write `content` to `path` atomically. Using a per-path .tmp next to
    the target so the rename stays on the same filesystem.

    On failure (disk full, permission denied, rename across mount points),
    raises OSError. Callers wrap this to produce a Rejection rather than
    recording the would-be edit as a success in the audit log.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content)
    try:
        os.replace(str(tmp), str(path))
    except OSError:
        
        
        
        try:
            tmp.unlink()
        except OSError:
            pass
        raise

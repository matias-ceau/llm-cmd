# Plan d'action synthétisé — audit quipcli (2026-08-27)

Basé sur les trois audits : [input-paths.md](input-paths.md), [test-coverage.md](test-coverage.md), [packaging.md](packaging.md).
État actuel : 251/251 tests passent, 82% de couverture de lignes, aucun changement de code effectué.

## Phase 0 — Filet de sécurité (aucun changement de comportement)

Doit passer avant tout le reste : sans CI ni outils déclarés, chaque correctif suivant repose sur une vérification manuelle non fiable.

1. **Tagger `v0.4.0`** sur `main` HEAD — `pyproject.toml` dit déjà 0.4.0 mais aucun tag ne pointe dessus (packaging H1).
2. **Ajouter `ruff`/`pyright` comme dev-deps** (`uv add --dev ruff pyright`) — actuellement ils ne marchent que parce qu'installés system-wide (packaging M1).
3. **Ajouter une CI minimale** (`.github/workflows/ci.yml`) — ruff check, ruff format --check, pyright, pytest sur push/PR (packaging H2).
4. Optionnel : `uv add --dev pytest-cov` pour rendre les rapports de couverture reproductibles (test-coverage note finale).

## Phase 1 — Corrections de bugs (petits diffs, comportement utilisateur)

Rangés par risque utilisateur réel, du plus grave au plus cosmétique.

1. **`--model-set`/`--cost` avalent le mot suivant du prompt** (input-paths H1) — corrompt silencieusement `default_model`. Fix : exiger `=` ou valider le token avant de l'accepter.
2. **`-e`/`-c`/`-a` non mutuellement exclusifs** (input-paths H2) — `add_mutually_exclusive_group()`.
3. **`--model-get`/`--status` affichent la mauvaise source** (`env` étiqueté `config`) (input-paths H3) — factoriser la logique de précédence dans un seul helper.
4. **`--max-steps 0`/négatif casse la boucle agent** (peut crasher sur entrée multimodale) (input-paths M4) — valider `>= 1`.
5. **`config.json` malformé échoue silencieusement à `{}`** (input-paths M2) — avertissement stderr.
6. **`--in`/`--out` no-op silencieux sans `--models`** (input-paths M3) — erreur d'usage explicite.
7. **README surestime la précédence env > config** (input-paths M1) — reformuler.

## Phase 2 — Combler les trous de test (aucun changement de comportement)

Priorité aux chemins qui, s'ils cassent, casseraient silencieusement en prod.

1. **`entry.py::main()` non testé du tout** (0 test direct, 47% de couverture module) — c'est le câblage réel argv → mode → LLM → stats ; le plus gros risque du rapport de tests.
2. **`models.py::_fetch_models` non testé** — jumeau de `_fetch_rankings` qui, lui, a 5 tests.
3. **Streaming SSE : buffer de portage jamais testé avec des chunks partiels** — risque de sortie tronquée/codes ANSI bloqués en trafic réel.
4. **Retry : seul 429 des 5 statuts retriables est testé** — paramétrer sur `{500,502,503,529}` + un test d'épuisement.
5. Petits trous restants (Medium/Low du rapport) : `agent.py` (JSON invalide, markdown activé), `execute.py::_edit_text_value`, `cli.py::_print_stats`, `db.py` fallback `except Exception`, migration legacy — à traiter en lot une fois 1-4 faits.

## Ce qui n'est volontairement pas dans ce plan

- Renommage des variables d'env `LLM_CMD_*`/`OPENROUTER_API_KEY` (input-paths L3) — tradeoff délibéré de compat, pas un bug.
- `tui.py` — déjà le mieux couvert (97%), stratégie de mock solide déjà en place.
- Licence/métadonnées `[project]` (packaging Low) — décision produit, pas un audit technique.

## Décisions à prendre avant d'exécuter

- Confirmer l'ordre des phases (0 → 1 → 2 recommandé : la CI doit exister avant qu'on s'appuie dessus).
- Pour chaque item de la Phase 1 : implémenter avec test associé (règle CLAUDE.md), donc Phase 1 et Phase 2 se chevauchent en pratique par item, pas par phase complète.
- Décision produit hors scope technique : faut-il une licence ? (packaging Medium)

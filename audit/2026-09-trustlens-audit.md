# TrustLens audit — 2026-09-28

**Scope:** volledige repository op commit `a58d175` (v0.5.0): code, berekeningen, documentatie, tests, CI en packaging.  
**Methode:** SPOS 4.9.1, route AUDIT, complexiteit L. Twee read-only werkpakketten (WP1 berekeningen, WP2 code/robuustheid/security) plus eigen werkpakket WP3 (docs, CI, packaging, voorbeelden). Alle KRITISCH-bevindingen en 6 van de 8 HOOG-bevindingen zijn door de coördinator onafhankelijk gereproduceerd; TL-09 en TL-11 berusten op worker-bewijs.  
**Bewijs:** reproductiescripts en ruwe uitvoer staan in [`audit/2026-09/repro/`](2026-09/repro/). Draai ze met `MPLBACKEND=Agg python audit/2026-09/repro/<script>.py`.

## 1. Executive summary

De afzonderlijke metrieken zijn correct; de **samenstelling tot Trust Score en deploy-verdict is dat niet**. Drie kritieke fouten leveren onder normaal gebruik stil een verkeerd verdict op: goed gekalibreerde, accurate binaire modellen worden geblokkeerd (TL-01), een model zonder kansen scoort calibratie 0 (TL-02), en een deelrun of typfout in `modules=` geeft een volwaardig ogend verdict — tot 'production-ready' (TL-03). Daarnaast o.a. schaalafhankelijke regressiescores (TL-04), een XSS in de HTML-weergave (TL-08) en documentatie die andere formules beschrijft dan de code (TL-10). De test-suite (512 groen) vangt dit niet omdat er geen invarianten getest worden.

**Totaal 31 issues:** 3 KRITISCH, 8 HOOG, 12 MIDDEN, 7 LAAG, 1 INFO.  
**Oordeel:** de Trust Score is in de huidige vorm niet geschikt als deployment-gate. Na fase 0–1 van het plan wel als heuristische indicator; na fase 2 als versiegebonden, gedocumenteerde methodologie.

## 2. Scope & aanpak

| Onderdeel | Gedaan | Niet gedaan |
|---|---|---|
| Berekeningen | Alle metrics + Trust Score vs onafhankelijke referenties, 100–2000 randomized trials + randgevallen | — |
| Code/API | ~40 robuustheidsprobes, backends xgboost/lightgbm/catboost uitgevoerd | HuggingFace-backend alleen gelezen (transformers niet geïnstalleerd); explainability/gradcam alleen doorgelezen |
| Docs | Sphinx-build, API-referenties vs code, claims vs gedrag, voorbeeldscripts gedraaid | Notebooks niet uitgevoerd |
| CI/packaging | Workflows, pre-commit, mypy, security-job gelezen; lint/format/mypy/tests lokaal | GitHub Actions zelf niet gedraaid; geen Windows/macOS |

Omgeving: Python 3.11, numpy 2.4.6, scikit-learn 1.9.1, scipy 1.17.1, matplotlib 3.11.2.

## 3. Sterke punten

- **Metriek-primitieven kloppen.** Brier, ECE, MCE, reliability curve, multiclass Brier, confidence gap, TPR/FPR, silhouette, lineaire CKA, MedAE/RMSE/P90, PICP, ICE, correlaties en conformal coverage komen overeen met onafhankelijke referenties (verschil ≤ 1e-15 over 100–2000 randomized trials).
- **Geen gevaarlijke constructies.** Geen pickle/joblib/eval/exec/dynamische imports/subprocess in het pakket; plugins alleen expliciet geregistreerd.
- **Schone toolchain.** ruff check, ruff format, mypy (projectconfig) en pre-commit slagen; 512 tests (533 met backends) groen; coverage 82.8% bij drempel 73%.
- **Performance.** analyze() op n = 200.000 met sensitive feature: 0.35 s.
- **Zorgvuldige matplotlib-hygiëne.** rcParams worden hersteld; geen gelekte figuren na herhaalde runs.
- **Degraded modes zichtbaar.** Ontbrekende y_prob en overgeslagen equalized odds/conformal worden gemarkeerd in metadata en logs; PredictionBundle weigert NaN/Inf en kansen buiten [0,1].

## 4. Issuelijst

Status: **C** = onafhankelijk gereproduceerd door coördinator · **W** = aangetoond door worker, script in repro/ · **E** = eigen werkpakket.

| ID | Severity | Dimensie | Issue | Effort | Fase | Status | Voortgang |
|---|---|---|---|---|---|---|---|
| TL-01 | KRITISCH | Correctheid | Failure-subscore heeft plafond: goed gekalibreerde, accurate binaire modellen krijgen 'Blocked / D' | M | 1 | C | Deels opgelost |
| TL-02 | KRITISCH | Correctheid | Overgeslagen calibratie (geen y_prob) telt als 0 in plaats van uit de weging te vallen | S | 1 | C | Opgelost |
| TL-03 | KRITISCH | Correctheid | modules= wordt niet gevalideerd; deelrun geeft volwaardig ogend verdict | S | 1 | C | Opgelost |
| TL-04 | HOOG | Correctheid | Regressie-Trust-Score gebruikt op 4 decimalen afgeronde waarden → schaalafhankelijk | S | 1 | C | Opgelost |
| TL-05 | HOOG | Correctheid | Multiclass Brier (bereik 0–2) wordt geclipt alsof bereik 0–1; calibratie straft groeiend met K | S | 2 | C | Open |
| TL-06 | HOOG | Correctheid | Equalized odds: groep zonder positieven krijgt TPR=0; geen minimale groepsgrootte | S | 1 | C | Opgelost |
| TL-07 | HOOG | Correctheid | Taak-autodetectie stuurt integer-regressiedoelen naar classificatie | S | 1 | C | Opgelost |
| TL-08 | HOOG | Security | Stored XSS in TrustReport._repr_html_ via featurenamen | S | 1 | C | Opgelost |
| TL-09 | HOOG | Correctheid | Signalen dubbel/driedubbel geteld (subscore + penalty + blocker) met klifeffect | M | 2 | W | Open |
| TL-10 | HOOG | Documentatie | Gedocumenteerde Trust-Score-formules wijken af van de code | S | 2 | C | Open |
| TL-11 | HOOG | Correctheid | compare() rangschikt niet-vergelijkbare rapporten en kan grade-D aanbevelen | M | 2 | W | Deels opgelost |
| TL-12 | MIDDEN | Datakwaliteit | Inputvalidatie ontbreekt (lengtes, rijsommen y_prob, pandas-index, list/DataFrame y_prob) | S | 3 | W | Open |
| TL-13 | MIDDEN | Correctheid | Labelafhandeling: manual-pad verliest classes_; conformal krijgt ongecodeerde labels | S | 3 | W | Open |
| TL-14 | MIDDEN | Correctheid | Top-label ECE koppelt max(y_prob) aan y_pred i.p.v. argmax | S | 3 | W | Open |
| TL-15 | MIDDEN | Correctheid | 'Bias'-dimensie zonder sensitive features meet alleen klasse-onbalans van de data | S | 2 | W | Open |
| TL-16 | MIDDEN | Correctheid | CRPS-decompositie negeert observaties buiten buitenste kwantiel | M | 4 | W | Open |
| TL-17 | MIDDEN | Architectuur | Ongetypeerd results-dict; ontbrekende metriek leest als perfecte 0.0 | M | 3 | W | Open |
| TL-18 | MIDDEN | Architectuur | report.py is een god-object (2084 regels) | L | 3 | W | Open |
| TL-19 | MIDDEN | Testbaarheid | Tests controleren vooral aanwezigheid, niet waarden; geen invarianten | M | 0 | W | Deels opgelost |
| TL-20 | MIDDEN | Correctheid | Brede except in plot_bias verbergt echte fout; voorbeeldscript crasht | S | 3 | E | Open |
| TL-21 | MIDDEN | Documentatie | Docs-build breekt met gedeclareerde extras; waarschuwingen; checklist-claim onjuist | S | 4 | E | Open |
| TL-22 | MIDDEN | Documentatie | Documentatie loopt achter op code | S | 4 | E | Open |
| TL-23 | MIDDEN | Documentatie | Sterke claims ('mathematically safe to deploy', 'production-ready') zonder gekalibreerde onderbouwing | S | 2 | E | Open |
| TL-24 | LAAG | Security | CI-hygiëne: shell-redirect in security-job, veel genegeerde CVE's, niet-gepinde actions, inconsistente mypy | S | 4 | E | Open |
| TL-25 | LAAG | Testbaarheid | tests/backends/test_xgboost_logic.py importeert xgboost onvoorwaardelijk | S | 1 | C | Opgelost |
| TL-26 | LAAG | Observability | print() ongeacht verbose; tqdm-bar naar stderr | S | 3 | C | Open |
| TL-27 | LAAG | Datakwaliteit | save(): crasht op pathlib.Path, onbekende extensie wordt map, stil overschrijven | S | 3 | W | Open |
| TL-28 | LAAG | Datakwaliteit | Publieke metriek-/gewicht-API valideert niet | S | 3 | W | Open |
| TL-29 | LAAG | Correctheid | Kleine metriekfouten buiten de Trust Score | S | 4 | W | Open |
| TL-30 | LAAG | Architectuur | Koppeling: scoring importeert visualisatie; import laadt matplotlib; losse framework-detectie | S | 3 | W | Open |
| TL-31 | INFO | Correctheid | Grade-drempels gelden op afgeronde score; methodologie-print kapt percentages af | S | 4 | W | Open |

### Details

#### TL-01 · KRITISCH · Failure-subscore heeft plafond: goed gekalibreerde, accurate binaire modellen krijgen 'Blocked / D'

- **Bewijs:** trust_score.py:127-148 — FailScore = 100·(0.8·gap + 0.2·acc); binaire top-label confidence ∈ [0.5,1] ⇒ gap ≤ 0.5 ⇒ score ≤ 60; blocker < 40. Reproductie: LogisticRegression op breast_cancer (acc 0.977, ECE-subscore 92.6) → 64/D 'Blocked by high diagnostic risk'; perfect model (conf 0.99, 100% correct) → failure 20.0, 59/D.
- **Impact:** De kernbelofte (deploy-verdict) is onbetrouwbaar: nette productiemodellen worden geblokkeerd; grade A is voor binaire taken vrijwel onbereikbaar.
- **Aanbeveling:** Gap normaliseren t.o.v. haalbaar maximum (bv. AUROC van confidence correct-vs-fout, of gap/(1−1/K)); all-correct neutraal/hoog scoren; drempels 40/60 herijken op referentiemodellen.
- **Effort:** M · **Fase:** 1 · **Status:** Gereproduceerd door coördinator · **Bron-ID:** CALC-01

#### TL-02 · KRITISCH · Overgeslagen calibratie (geen y_prob) telt als 0 in plaats van uit de weging te vallen

- **Bewijs:** trust_score.py:115-124 defaults brier/ece=0.5 op een {'status':'skipped'}-dict. Reproductie: perfect model met alleen y_pred → sub_scores {'calibration': 0.0, 'failure': 20.0, 'bias': 100.0} → 21/D.
- **Impact:** Elk model zonder kansen wordt 'Low Trust / blocked'; in strijd met docstring (redistributie).
- **Aanbeveling:** Module-dicts met status skipped/degraded overslaan en gewicht herverdelen (zoals _reg_metric_present); nooit defaulten naar 0.5.
- **Effort:** S · **Fase:** 1 · **Status:** Gereproduceerd door coördinator · **Bron-ID:** CALC-02

#### TL-03 · KRITISCH · modules= wordt niet gevalideerd; deelrun geeft volwaardig ogend verdict

- **Bewijs:** pipeline.py:129 / api.py. Reproductie: zelfde GaussianNB — volledige run 38/D 'Blocked'; modules=['bias'] → 86/A 'High Trust - production-ready'; modules=['calibartion'] (typo) → 0/D zonder waarschuwing.
- **Impact:** Gebruiker krijgt 'production-ready' voor een model dat de volledige audit blokkeert.
- **Aanbeveling:** Onbekende modulenamen → ValueError; deelrun expliciet markeren (metadata + verdicttekst 'partial') of geen grade geven zonder kern-dimensies.
- **Effort:** S · **Fase:** 1 · **Status:** Gereproduceerd door coördinator · **Bron-ID:** CODE-01

#### TL-04 · HOOG · Regressie-Trust-Score gebruikt op 4 decimalen afgeronde waarden → schaalafhankelijk

- **Bewijs:** regression.py:111-116 rondt af; trust_score.py:610-632 rekent daarmee. Reproductie: onafhankelijke y_pred (R² ≈ −1): schaal 1 → 0/D, schaal 1e-5 → 100/A.
- **Impact:** Stil 'A' voor slechte modellen op kleine eenheden (rendementen, concentraties).
- **Aanbeveling:** Onafgeronde waarden voor scoring; alleen afronden bij weergave. Schaal-invariantietest toevoegen.
- **Effort:** S · **Fase:** 1 · **Status:** Gereproduceerd door coördinator · **Bron-ID:** CALC-03

#### TL-05 · HOOG · Multiclass Brier (bereik 0–2) wordt geclipt alsof bereik 0–1; calibratie straft groeiend met K

- **Bewijs:** pipeline.py:163-173 → trust_score.py:121-124. Reproductie: perfect gekalibreerd uniform 10-klassenmodel → calibratie-subscore 8.4.
- **Impact:** Multiclass modellen systematisch te laag op 'calibratie'; Brier mengt bovendien accuracy (refinement) in calibratie.
- **Aanbeveling:** Calibratie-dimensie baseren op ECE (of Brier-reliability-component); Brier normaliseren (/2 of skill score) als hij blijft.
- **Effort:** S · **Fase:** 2 · **Status:** Gereproduceerd door coördinator · **Bron-ID:** CALC-06

#### TL-06 · HOOG · Equalized odds: groep zonder positieven krijgt TPR=0; geen minimale groepsgrootte

- **Bewijs:** bias.py:320,328 (zero_division=0), :163-172/:337-343. Reproductie: perfecte classifier, één groep met alleen negatieven → 44/D. Worker: groep van 1 sample → 'severe'.
- **Impact:** Valse fairness-blokkades bij kleine of scheve subgroepen.
- **Aanbeveling:** Ongedefinieerde TPR/FPR → NaN en uitsluiten; min_group_size (bv. 30) met 'low confidence'-markering i.p.v. blocker.
- **Effort:** S · **Fase:** 1 · **Status:** Gereproduceerd door coördinator · **Bron-ID:** CALC-07

#### TL-07 · HOOG · Taak-autodetectie stuurt integer-regressiedoelen naar classificatie

- **Bewijs:** api.py:48-58. Reproductie: afgerond continu doel (veel unieke ints) met y_pred → task=classification, 11/D; met model → misleidende NotImplementedError.
- **Impact:** Tellingen/prijzen krijgen een zinloze '159-klassen'-audit.
- **Aanbeveling:** Cardinaliteit/ratio-heuristiek; bij twijfel expliciete task eisen; sklearn is_regressor(model) gebruiken.
- **Effort:** S · **Fase:** 1 · **Status:** Gereproduceerd door coördinator · **Bron-ID:** CODE-03

#### TL-08 · HOOG · Stored XSS in TrustReport._repr_html_ via featurenamen

- **Bewijs:** report.py:839, 2003 (ook 1934, 1955): insights ongeëscaped in HTML; regressiepad gebruikt wél html.escape (1892). Reproductie: '<script>alert(1)</script>' als sensitive-featurenaam staat letterlijk in _repr_html_().
- **Impact:** Scriptuitvoering in gedeelde notebooks/HTML-exports met kolomnamen uit onbetrouwbare data.
- **Aanbeveling:** Alle dynamische waarden html.escape'n; test die escaping afdwingt.
- **Effort:** S · **Fase:** 1 · **Status:** Gereproduceerd door coördinator · **Bron-ID:** SEC-01

#### TL-09 · HOOG · Signalen dubbel/driedubbel geteld (subscore + penalty + blocker) met klifeffect

- **Bewijs:** trust_score.py:123/407-419/495-497 (ECE), 164-173/426-455 (subgroup gap), 669/852-862 (regressie-correlatie). Worker: ECE 0.1000 → 59/C, ECE 0.1001 → 59/D blocked.
- **Impact:** Score niet monotoon/uitlegbaar; minieme metriekwijziging verschuift een hele grade.
- **Aanbeveling:** Eén mechanisme per signaal; blocker-drempel = einde penalty-ramp; vastleggen in methodologie-ADR.
- **Effort:** M · **Fase:** 2 · **Status:** Bewijs worker (script in repro/) · **Bron-ID:** CALC-05

#### TL-10 · HOOG · Gedocumenteerde Trust-Score-formules wijken af van de code

- **Bewijs:** trust_score.py:53-68 vs 115-174: docs 0.5·BS+0.5·ECE ↔ code BS+1.5·ECE; docs 100·gap ↔ code 0.8·gap+0.2·acc; docs ratio/20 ↔ code (ratio−1)/19. _failure_score-docstring zelf ook fout. docs/trust_score_explained.md geeft geen formules.
- **Impact:** Scores niet reproduceerbaar voor gebruikers/auditors; gepubliceerde methodologie klopt niet.
- **Aanbeveling:** Formules op één plek definiëren; doctests die docs-formule == code afdwingen.
- **Effort:** S · **Fase:** 2 · **Status:** Gereproduceerd door coördinator · **Bron-ID:** CALC-04

#### TL-11 · HOOG · compare() rangschikt niet-vergelijkbare rapporten en kan grade-D aanbevelen

- **Bewijs:** comparison.py:10-107. Worker: volledige vs bias-only run van hetzelfde model → 'Deploy GaussianNB … GaussianNB over GaussianNB'; lege-resultaten-rapport (0/100) krijgt 'Deploy'.
- **Impact:** Verkeerde deploy-aanbeveling.
- **Aanbeveling:** Gelijke modules_run/degraded-status eisen; nooit grade D aanbevelen; labels per rapport; gestructureerd resultaat teruggeven.
- **Effort:** M · **Fase:** 2 · **Status:** Bewijs worker (script in repro/) · **Bron-ID:** CODE-02

#### TL-12 · MIDDEN · Inputvalidatie ontbreekt (lengtes, rijsommen y_prob, pandas-index, list/DataFrame y_prob)

- **Bewijs:** utils.check_consistent_length/validate_array bestaan maar worden nergens aangeroepen. Worker: X van 10 rijen vs y van 200 geaccepteerd; y_prob·0.5 zonder waarschuwing gescoord; omgekeerde kolommen → 0/D; pandas-index genegeerd (positioneel).
- **Impact:** Onzin-input levert plausibel ogende scores.
- **Aanbeveling:** Eén validatielaag aan de ingang van analyze(): np.asarray, lengtes, rijsom-tolerantie, index-uitlijning, optionele argmax-vs-model.predict-check.
- **Effort:** S · **Fase:** 3 · **Status:** Bewijs worker (script in repro/) · **Bron-ID:** CODE-04/05/06, CALC-09

#### TL-13 · MIDDEN · Labelafhandeling: manual-pad verliest classes_; conformal krijgt ongecodeerde labels

- **Bewijs:** api.py:286-287, pipeline.py:91,184,210-214. Worker: labels {1,2,3} + class_labels → conformal coverage 0.357 i.p.v. 0.9 (stil); labels niet 0..K-1 zonder class_labels → cryptische IndexError.
- **Impact:** Stil foute conformal-diagnostiek; slechte bruikbaarheid.
- **Aanbeveling:** Overal dezelfde label-encoder gebruiken; classes_ behouden; heldere fout 'pass class_labels'.
- **Effort:** S · **Fase:** 3 · **Status:** Bewijs worker (script in repro/) · **Bron-ID:** CODE-07, CALC-08, CALC-19

#### TL-14 · MIDDEN · Top-label ECE koppelt max(y_prob) aan y_pred i.p.v. argmax

- **Bewijs:** pipeline.py:160-161. Worker: 33% y_pred ≠ argmax → ECE 0.0570 vs referentie 0.0618.
- **Impact:** Foute ECE bij custom thresholds/kostgevoelige beslissingen.
- **Aanbeveling:** Correctheid voor top-label ECE op argmax(y_prob) baseren.
- **Effort:** S · **Fase:** 3 · **Status:** Bewijs worker (script in repro/) · **Bron-ID:** CALC-09

#### TL-15 · MIDDEN · 'Bias'-dimensie zonder sensitive features meet alleen klasse-onbalans van de data

- **Bewijs:** trust_score.py:151-174. Worker: gebalanceerde data zonder sensitive features → bias 100 (25% gewicht).
- **Impact:** Fairness-credit die nooit gemeten is, of straf voor een data-eigenschap.
- **Aanbeveling:** Zonder sensitive features dimensie laten vallen en herverdelen; onbalans als waarschuwing.
- **Effort:** S · **Fase:** 2 · **Status:** Bewijs worker (script in repro/) · **Bron-ID:** CALC-16

#### TL-16 · MIDDEN · CRPS-decompositie negeert observaties buiten buitenste kwantiel

- **Bewijs:** regression.py:604-617. Worker: observatie 100, N(0,1) → decompositie-CRPS 0.80 vs gesloten vorm 49.83.
- **Impact:** Misleidende reliability/resolution bij outliers (niet in Trust Score).
- **Aanbeveling:** Hersbach-outliersegmenten toevoegen of truncatie documenteren.
- **Effort:** M · **Fase:** 4 · **Status:** Bewijs worker (script in repro/) · **Bron-ID:** CALC-10

#### TL-17 · MIDDEN · Ongetypeerd results-dict; ontbrekende metriek leest als perfecte 0.0

- **Bewijs:** report.py:201-210 .get('ece', 0.0); docs/internal/prediction_contract.md zegt nog 'geen regressie-ondersteuning'.
- **Impact:** Stille fouten; elk consumerend stuk code herimplementeert defaults.
- **Aanbeveling:** TypedDict/dataclasses per module; ontbreken = None; contract-doc bijwerken.
- **Effort:** M · **Fase:** 3 · **Status:** Bewijs worker (script in repro/) · **Bron-ID:** ARCH-02

#### TL-18 · MIDDEN · report.py is een god-object (2084 regels)

- **Bewijs:** Console-rendering (87 prints), HTML in f-strings, JSON-persistentie, plotorkestratie (plot_bias ≈ 360 regels), verdictlogica; save/_save_regression gedupliceerd.
- **Impact:** Moeilijk te testen en te wijzigen; bron van TL-08.
- **Aanbeveling:** Opsplitsen: rapportmodel, text/HTML-renderers (escaping centraal), serializer, plot-facade.
- **Effort:** L · **Fase:** 3 · **Status:** Bewijs worker (script in repro/) · **Bron-ID:** ARCH-01

#### TL-19 · MIDDEN · Tests controleren vooral aanwezigheid, niet waarden; geen invarianten

- **Bewijs:** test_api.py (key/isinstance-asserts); characterization pint 2 binaire sklearn-cases en negeert extra keys. Geen tests voor pandas, task='auto', rijsommen, HTML-escaping, modules-subsets, multiclass/regressie-baselines. 512 tests groen terwijl TL-01..04 bestaan.
- **Impact:** Rekenfouten in de score-compositie glippen erdoor.
- **Aanbeveling:** Invariant-/property-tests en referentiemodel-benchmark in CI (zie fase 0).
- **Effort:** M · **Fase:** 0 · **Status:** Bewijs worker (script in repro/) · **Bron-ID:** TEST-01

#### TL-20 · MIDDEN · Brede except in plot_bias verbergt echte fout; voorbeeldscript crasht

- **Bewijs:** report.py:1590-1632 slikt exceptions en meldt 'Failed to generate any bias plots'. Echte oorzaak: FileNotFoundError bij niet-bestaande map (gereproduceerd). examples/bias_analysis_demo.py faalt daardoor (exit 1).
- **Impact:** Gebruikers krijgen een misleidende foutmelding; officieel voorbeeld werkt niet.
- **Aanbeveling:** Oorspronkelijke fout loggen/doorgeven; save-map aanmaken; voorbeelden in CI draaien.
- **Effort:** S · **Fase:** 3 · **Status:** Eigen werkpakket WP3 · **Bron-ID:** WP3

#### TL-21 · MIDDEN · Docs-build breekt met gedeclareerde extras; waarschuwingen; checklist-claim onjuist

- **Bewijs:** pip install .[docs] + sphinx → ExtensionError: sphinxcontrib.mermaid ontbreekt in [docs]. Na handmatige installatie: 3 warnings (conformal.md en RELEASE_CHECKLIST.md niet in toctree, _static ontbreekt). RELEASE_CHECKLIST claimt 'zero warnings'.
- **Impact:** Conformal-docs onvindbaar; docs niet reproduceerbaar te bouwen.
- **Aanbeveling:** sphinxcontrib-mermaid aan [docs]; toctree fixen; docs-build met -W in CI.
- **Effort:** S · **Fase:** 4 · **Status:** Eigen werkpakket WP3 · **Bron-ID:** WP3

#### TL-22 · MIDDEN · Documentatie loopt achter op code

- **Bewijs:** README-badges '237 passing'/'75%' (gemeten 512–533 / 82.8%); ROADMAP noemt MCE 'planned' (is gebouwd); API-reference mist compare, quick_analyze, compute_trust_score, regression_trust_score; SECURITY.md advisory-link naar TrustLens/TrustLens (verkeerde org) en ondersteuningstabel slaat 0.3.x over.
- **Impact:** Verkeerd beeld van status en rapportagekanaal.
- **Aanbeveling:** Dynamische badges (CI/codecov); docs-checklist in PR-template; API-reference via autosummary.
- **Effort:** S · **Fase:** 4 · **Status:** Eigen werkpakket WP3 · **Bron-ID:** WP3, ARCH-02

#### TL-23 · MIDDEN · Sterke claims ('mathematically safe to deploy', 'production-ready') zonder gekalibreerde onderbouwing

- **Bewijs:** README:53; trust_score.py grade A 'production-ready'; drempels/gewichten 'tuned' zonder bron; benchmark-claims alleen in notebook, niet reproduceerbaar in CI. In het licht van TL-01..04 niet houdbaar.
- **Impact:** Overmatig vertrouwen in een heuristische score.
- **Aanbeveling:** Claims afzwakken naar 'heuristische indicator'; validatie-notebook als CI-job; score_version publiceren.
- **Effort:** S · **Fase:** 2 · **Status:** Eigen werkpakket WP3 · **Bron-ID:** WP3

#### TL-24 · LAAG · CI-hygiëne: shell-redirect in security-job, veel genegeerde CVE's, niet-gepinde actions, inconsistente mypy

- **Bewijs:** ci.yml: `pip install --upgrade mistune>=3.2.1` zonder quotes → shell leest '>' als redirect naar bestand '=3.2.1'; 17 --ignore-vuln zonder motivatie; actions op tags i.p.v. SHA (softprops/action-gh-release met contents:write); CI mypy --follow-imports=skip vs release --ignore-missing-imports; mypy --strict: 135 fouten.
- **Impact:** Security-job doet niet precies wat hij lijkt te doen; supply-chain-risico.
- **Aanbeveling:** Quoten; ignore-lijst met reden+vervaldatum; SHA-pinning + Dependabot; mypy-config unificeren en strictness per module ophogen (ratchet).
- **Effort:** S · **Fase:** 4 · **Status:** Eigen werkpakket WP3 · **Bron-ID:** WP3, TEST-03

#### TL-25 · LAAG · tests/backends/test_xgboost_logic.py importeert xgboost onvoorwaardelijk

- **Bewijs:** Regel 3; zonder [full] faalt de testcollectie (gereproduceerd: 1 error).
- **Impact:** Lokale dev-run met alleen [dev] breekt.
- **Aanbeveling:** pytest.importorskip('xgboost').
- **Effort:** S · **Fase:** 1 · **Status:** Gereproduceerd door coördinator · **Bron-ID:** TEST-02

#### TL-26 · LAAG · print() ongeacht verbose; tqdm-bar naar stderr

- **Bewijs:** pipeline.py:152,207,234,264,303 printen bij verbose=False (gereproduceerd: 'Running calibration analysis...'); quick_analyze print demo-banner ook met eigen model; tqdm nooit gesloten.
- **Impact:** Vervuilde logs in pipelines/CI.
- **Aanbeveling:** Voortgang via logging, gated door verbose.
- **Effort:** S · **Fase:** 3 · **Status:** Gereproduceerd door coördinator · **Bron-ID:** CODE-08, CODE-11

#### TL-27 · LAAG · save(): crasht op pathlib.Path, onbekende extensie wordt map, stil overschrijven

- **Bewijs:** report.py:1649-1747. Worker: save(Path) → AttributeError; 'report.PNG' → directory; tweede save overschrijft stil.
- **Impact:** Kleine bruikbaarheids- en dataverliesrisico's.
- **Aanbeveling:** os.PathLike accepteren; onbekende extensie → fout; overwrite=False-vlag.
- **Effort:** S · **Fase:** 3 · **Status:** Bewijs worker (script in repro/) · **Bron-ID:** CODE-09, SEC-02

#### TL-28 · LAAG · Publieke metriek-/gewicht-API valideert niet

- **Bewijs:** ECE met NaN/p=1.5 geeft eindige waarde (samples stil genegeerd), ECE op lege input = 0.0, labels {1,2} → ECE > 1; custom weights: negatief geaccepteerd, onbekende keys genegeerd; NaN in sensitive feature crasht diep in sklearn.
- **Impact:** Directe API-gebruikers krijgen plausibele maar foute waarden.
- **Aanbeveling:** Validatie in gedeelde binning-core en in compute_trust_score(weights).
- **Effort:** S · **Fase:** 3 · **Status:** Bewijs worker (script in repro/) · **Bron-ID:** CALC-12, CALC-17, CODE-10

#### TL-29 · LAAG · Kleine metriekfouten buiten de Trust Score

- **Bewijs:** within_class_distance neemt self-pairs mee (2.67 vs 2.90; kleine klassen → ratio inf); CKA niet schaalinvariant (CKA(X·1e-4) = 0.0); brier-docstringvoorbeeld 0.036 i.p.v. 0.048; CRPS-docstring noemt verkeerde biasrichting.
- **Impact:** Misleidende diagnostiek voor directe gebruikers.
- **Aanbeveling:** Paren i≠j; relatieve tolerantie; docstrings corrigeren + doctest.
- **Effort:** S · **Fase:** 4 · **Status:** Bewijs worker (script in repro/) · **Bron-ID:** CALC-11/13/14/15

#### TL-30 · LAAG · Koppeling: scoring importeert visualisatie; import laadt matplotlib; losse framework-detectie

- **Bewijs:** trust_score.py importeert visualization.style; `import trustlens` laadt matplotlib; backends/registry.py startswith-matching, 'detecteert' tensorflow/keras/pytorch zonder resolver.
- **Impact:** Trage import, zwaardere dependency-voetafdruk, verwarrende foutmeldingen.
- **Aanbeveling:** Kleuren/HTML uit trust_score; lazy imports; exacte module-matching.
- **Effort:** S · **Fase:** 3 · **Status:** Bewijs worker (script in repro/) · **Bron-ID:** ARCH-03, ARCH-05

#### TL-31 · INFO · Grade-drempels gelden op afgeronde score; methodologie-print kapt percentages af

- **Bewijs:** Ruwe 39.5 → C; 79.5 → A. Herverdeelde gewichten 0.389 getoond als '38%'.
- **Impact:** Cosmetisch.
- **Aanbeveling:** Documenteren; percentages afronden i.p.v. afkappen.
- **Effort:** S · **Fase:** 4 · **Status:** Bewijs worker (script in repro/) · **Bron-ID:** CALC-18

## 5. Risk heat map

| Dimensie | Score /10 | Hoogste severity | # Issues | Prioriteit | Onderbouwing |
|---|---|---|---|---|---|
| Correctheid | 4 | KRITISCH | 16 | P1 | Primitieven correct (9/10), maar compositie tot Trust Score en verdict bevat 3 kritieke en 6 hoge fouten. |
| Security | 6 | HOOG | 2 | P1 | Geen gevaarlijke constructies; wel XSS in HTML-repr en CI-hygiëne. |
| Datakwaliteit / inputvalidatie | 4 | MIDDEN | 3 | P2 | Validatie-helpers bestaan maar worden niet aangeroepen. |
| Documentatie | 4 | HOOG | 4 | P2 | Formules, badges, roadmap, API-referentie en claims wijken af van de code. |
| Testbaarheid | 6 | MIDDEN | 2 | P1 | Veel tests en goede coverage, maar geen invarianten: 512 groene tests ondanks kritieke rekenfouten. |
| Architectuur | 5 | MIDDEN | 3 | P3 | Heldere modulegrenzen bij metrics/backends; god-object report.py en ongetypeerd results-contract. |
| Observability | 6 | LAAG | 1 | P4 | Logging aanwezig, maar print() naast logger en ongeacht verbose. |
| Performance | 9 | — | 0 | — | Geen bottlenecks gevonden. |
| Compliance / kosten | n.v.t. | — | 0 | — | N.v.t.: lokale library, geen PII-opslag, geen externe calls. |

Scores zijn een oordeel van de auditor [AFGELEID], geen meting.

## 6. Plan voor duurzame oplossing

Uitgangspunt: **eerst een vangnet, dan pas de score veranderen.** Elke fix sluit een falende test; methodologiewijzigingen zijn versiegebonden en gedocumenteerd.

### Fase 0 — Vangnet eerst
*Voordat er iets aan de score verandert*

- Invariant-testsuite (tests/invariants/): perfect gekalibreerd en correct model ⇒ geen blocker en grade ≥ B; schaalinvariantie regressie (×1e-5 … ×1e5); ontbrekende dimensie ⇒ herverdeling, nooit 0; monotonie: slechtere ECE ⇒ niet-hogere score; modules-subset ⇒ gemarkeerd als partial; HTML-escaping.
- Referentiemodel-benchmark (tests/reference/): vaste seeds, sklearn-datasets (breast_cancer LR, iris multiclass, diabetes regressie, synthetisch perfect/random) met verwachte grade-banden. Faalt bewust op de huidige code (xfail met TL-id) zodat elke fix aantoonbaar een test groen maakt.
- Audit-repro's (audit/2026-09/repro) omzetten naar pytest-cases; audit-map daarna alleen als historisch bewijs.
- ADR-001 'Trust Score methodologie' vastleggen: doel, dimensies, één mechanisme per signaal, drempels, wat de score níet is (geen kans op falen, geen certificering).

**Issues:** TL-19  
**Klaar als:** Invariant- en referentiesuite draait in CI; elke TL-KRITISCH/HOOG heeft een falende (xfail) test.

### Fase 1 — Kritieke correctheid en security
*Patchrelease v0.5.1 — geen methodologiewijziging*

- TL-02 overgeslagen dimensies herverdelen; TL-03 modules valideren + partial-markering; TL-04 onafgeronde waarden voor scoring; TL-06 NaN voor ongedefinieerde TPR/FPR + min_group_size; TL-07 autodetectie + is_regressor; TL-08 html.escape overal; TL-25 importorskip.
- TL-01 als minimale patch: all-correct-geval niet meer blokkeren en blocker pas bij gap-normalisatie t.o.v. haalbaar maximum; volledige herziening in fase 2.
- Elke fix: bijbehorende xfail → pass, CHANGELOG-regel onder 'Fixed' met TL-id en score-impact.

**Issues:** TL-01*, 02, 03, 04, 06, 07, 08, 25  
**Klaar als:** Alle KRITISCH-issues gesloten; breast_cancer-LR niet meer geblokkeerd; geen XSS; release-notes benoemen gewijzigde scores.

### Fase 2 — Trust Score v2 — methodologie
*Minor release v0.6.0, versiegebonden*

- Herontwerp volgens ADR-001: calibratie op ECE/Brier-reliability (TL-05), failure genormaliseerd (TL-01 volledig), één mechanisme per signaal zonder kliffen (TL-09), bias-dimensie alleen met sensitive features (TL-15).
- score_version ('2.0') in TrustScoreResult en opgeslagen rapporten; v1 blijft een release lang opvraagbaar voor vergelijking; migratienotitie met voor/na-tabel van de referentiemodellen.
- Drempels en gewichten herijken op de referentiebenchmark + model-zoo-notebook; notebook als CI-job (nightly) zodat research-claims reproduceerbaar blijven.
- Formules op één plek (constants + docstring) met doctests die docs == code afdwingen (TL-10); claims in README afzwakken (TL-23); compare() vergelijkbaarheid afdwingen (TL-11).

**Issues:** TL-01, 05, 09, 10, 11, 15, 23  
**Klaar als:** Scores monotoon en uitlegbaar; gepubliceerde formule reproduceert elke referentiescore exact.

### Fase 3 — Inputcontract en architectuur
*v0.6.x – v0.7.0, incrementeel*

- Validatielaag aan de ingang van analyze() (TL-12) en één label-encoder voor alle modules incl. conformal (TL-13, TL-14).
- Getypeerd results-contract (TypedDict/dataclasses), ontbreken = None (TL-17); prediction_contract.md bijwerken.
- report.py opsplitsen in model / renderers / serializer / plot-facade (TL-18), met behoud van publieke API; centrale escaping.
- Logging i.p.v. print (TL-26); save() robuust (TL-27); API-validatie metrics/weights (TL-28); koppeling en lazy imports (TL-30); plot_bias-fout doorgeven (TL-20).

**Issues:** TL-12, 13, 14, 17, 18, 20, 26, 27, 28, 30  
**Klaar als:** Onzin-input geeft een heldere fout; report.py < 600 regels per module; strict-mypy op nieuwe modules.

### Fase 4 — Documentatie, CI en onderhoud
*Doorlopend, parallel aan fase 1–3*

- Docs: mermaid in [docs], toctree, -W build in CI (TL-21); API-reference via autosummary; badges dynamisch; ROADMAP/SECURITY bijwerken (TL-22).
- CI: quotes in security-job, ignore-lijst met reden + vervaldatum, actions SHA-pinnen + Dependabot, mypy-config unificeren met strictness-ratchet (TL-24); voorbeeldscripts in CI in een tijdelijke map.
- Kleine metriekfouten en docstrings (TL-16, 29, 31).
- PR-template uitbreiden met 'score-impact'-sectie (welke referentiescores veranderen en waarom) en docs-checklist.

**Issues:** TL-16, 21, 22, 24, 29, 31  
**Klaar als:** Docs bouwen zonder warnings; CI-job voor voorbeelden; geen ongemotiveerde CVE-ignores.

### Borging (waarom dit niet terugkomt)

- **Invarianten in CI** — Voorkomt terugkeer van TL-01..04: een score-regressie faalt de build in plaats van ongemerkt te releasen.
- **Referentiebenchmark met verwachte grade-banden** — Maakt elke methodologiewijziging zichtbaar als een diff in een tabel, niet als een verrassing bij gebruikers.
- **score_version + ADR** — Opgeslagen rapporten blijven interpreteerbaar; beslissingen over gewichten/drempels zijn herleidbaar.
- **Doctests voor formules** — Documentatie kan niet meer stil afwijken van de code (TL-10).
- **Docs -W en voorbeelden in CI** — Kapotte docs en voorbeelden worden direct zichtbaar (TL-20, TL-21).
- **Strictness-ratchet** — mypy/coverage-drempels mogen alleen omhoog; nieuwe code strikt getypeerd.
- **PR-template 'score-impact'** — Maakt reviewers verantwoordelijk voor scoregevolgen van elke wijziging.

## 7. Eindscore

**5/10** [AFGELEID]. Solide metriekbasis, performance en tooling; de kernbelofte (betrouwbaar deploy-verdict) wordt door TL-01..03 ondermijnd. Na fase 0–2 realistisch 8/10.

## 8. Verificatie van deze audit

| Check | Resultaat |
|---|---|
| Correctheid | Alle KRITISCH-issues en 6 van de 8 HOOG-issues (TL-01..08, TL-10) zijn door de coördinator onafhankelijk gereproduceerd, zonder de worker-scripts. TL-09 en TL-11 berusten op worker-bewijs. |
| Falsificatie 1 | *Is de failure-formule (TL-01) een bewuste keuze?* Mogelijk; toch blokkeert ze een perfect correct model (59/D), wat met geen enkele redelijke methodologie strookt. Detectie: ADR-001 dwingt de keuze expliciet af. |
| Falsificatie 2 | *Zijn resultaten versie-afhankelijk?* De fouten zitten in eigen rekenlogica (clip-grenzen, defaults, afronding), niet in numpy/sklearn-gedrag. Detectie: invariant-suite draaien in de CI-matrix (3.9–3.13). |
| Volledigheid | Niet onderzocht: HuggingFace-backend (uitvoering), notebooks, explainability in de diepte, Windows/macOS. |
| Injection-lekkage | Geen instructies uit broncode of documentatie overgenomen. |
| Extrapolatie | Heat-map-scores en eindscore zijn gemarkeerd als [AFGELEID]. Er zijn geen claims gedaan over hoe vaak gebruikers de randgevallen raken. |

## 9. Voortgang (fase 0–1)

Uitgevoerd op branch `claude/tender-cerf-4b5udn`. Na de fixes: 573 tests groen, 2 bewust open strict-xfails (TL-05, TL-09; fase 2).

| ID | Status | Wat is gedaan |
|---|---|---|
| TL-01 | Deels opgelost | Failure-gap genormaliseerd op 1−1/K; foutloos model krijgt volle gap-score. breast_cancer-LR (zelfde opzet als de audit) 64/D blocked → 72/B; referentietest met geschaalde LR slaagt nu. Volledige herziening (AUROC) blijft fase 2. |
| TL-02 | Opgelost | Overgeslagen/degraded dimensies vallen uit de weging; zichtbaar in missing_dimensions. |
| TL-03 | Opgelost | Onbekende modules → ValueError; is_partial, grade-cap C, metadata partial=true. |
| TL-04 | Opgelost | error_distribution rondt niet meer af; schaalinvariantie getest (1e-5 … 1e5). |
| TL-06 | Opgelost | Ongedefinieerde TPR/FPR = None; min_group_size 30 in de pipeline; low_support-markering. |
| TL-07 | Opgelost | Estimator-type, y_prob en target-heuristiek; waarschuwing bij heuristische routering. |
| TL-08 | Opgelost | Alle dynamische tekst in beide _repr_html_ ge-escaped. |
| TL-11 | Deels opgelost | compare() beveelt nooit partial, blocked of grade D aan; labels/gestructureerd resultaat volgen in fase 2. |
| TL-19 | Deels opgelost | tests/invariants + tests/reference toegevoegd; resterende defecten als strict xfail (TL-05, TL-09). |
| TL-25 | Opgelost | pytest.importorskip('xgboost'). |

# Quantile statt Mittelwert – trifft das 90-%-Band in jeder Gruppe? – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-quantil-ueberdeckung-demo.streamlit.app/)** -->

Eine Demo zu **Regression und verallgemeinerten linearen Modellen** für die Website „Sebastian Hanisch – Operations Research und Machine Learning“, die auf [`streuung-wls-demo`](https://github.com/sebastian-hanisch/streuung-wls-demo) aufbaut. Diese Vorgänger-Demo nimmt ernst, dass die Streuung mit dem Merkmal wächst, und repariert damit die **Standardfehler der Koeffizienten**; sie bleibt beim Mittelwert und bei Aussagen über Koeffizienten. Wer „den Puffer, der 90 % der Container abdeckt“ braucht, fragt dagegen nach dem **Rand der Verteilung eines einzelnen Containers**, und die naheliegende Brücke, **Vorhersage ± z·s** mit *einer* gemeinsamen Reststreuung s, nimmt die konstante Streuung wieder an. Die Demo misst **exakt, wo dieses Band trifft** und wo nicht, und stellt die **lineare Quantilregression** dagegen. Alle Formeln und Befunde sind **eigene Nachrechnung in diesen Modellen, bei diesen Parametern**; Zähleinheit überall ist die **Wiederholung** (eine frische Trainingsstichprobe). Eine Neuheitsaussage gibt es nicht: nichts hier ist wissenschaftlich neu; neu ist nur die Vorführbarkeit mit exakter Überdeckung je Gruppe und mit Zähleinheiten.

Derselbe Mechanismus (lineare Quantilregression gegen eine Regel mit konstanter Reststreuung bei wachsender Streuung) steht in [`newsvendor-direkt-demo`](https://github.com/sebastian-hanisch/newsvendor-direkt-demo) an den **Kosten** gemessen; hier geht es nur darum, ob das Band trifft (kein Kostenverhältnis, kein Pinball-Verlust als Gegenstand). Die Falle „im Mittel stimmt es, in der Untergruppe nicht“ für Zeitreihen-Intervalle mit konformer Kalibrierung: [`forecast-interval-demo`](https://github.com/sebastian-hanisch/forecast-interval-demo). Den Generator der Container-Verweilzeit liefert [`verweilzeit-demo`](https://github.com/sebastian-hanisch/verweilzeit-demo) (hier nachgebaut, nicht importiert). Die Schwelle der Kalibrierung ist ein Quantil derselben Art (τ = b/(b + h)): [`kalibrierung-schwelle-demo`](https://github.com/sebastian-hanisch/kalibrierung-schwelle-demo). Der Kunde als Cluster (hier unerklärte Streuung): [`cluster-gemischt-demo`](https://github.com/sebastian-hanisch/cluster-gemischt-demo).

## Warum dieses Problem

„Plane die Standzeit so, dass 90 % der Container bis dahin abgeholt sind“ ist eine Frage nach dem **0,9-Quantil der Verweilzeit eines Containers**, nicht nach dem Mittel. Mit Kleinsten Quadraten kommt man leicht zu `Vorhersage + z·s`, und im Mittel über alle Container liegt das nahe am Soll: bei den Containern deckt das Band gesamt 0,918 (Soll 0,9; Modell mit 8 Parametern, n = 3 000, τ = 0,9, Mittel über 40 Wiederholungen, wie alle Zahlen dieses Absatzes). **Je Art aber nicht**: die Leer-Container, die am stärksten streuen (log-Streuung 0,90 gegen 0,25 bei Export), bekommen nur 0,775, Export dagegen 0,992. Die Quantilregression trifft in jeder Art (0,896 bis 0,900), aber nur mit genug Daten und passender Modellform; und wo die Streuung einer **bekannten** Gruppe folgt, reicht schon Kleinste Quadrate mit einem s je Art (0,898 bis 0,900). Die Demo führt alle drei Aussagen mit ihren Voraussetzungen vor: Kernfragen vor dem Expander, zwei Experimente, ein Handbeispiel in Brüchen.

## Modell

**Konvention.** Quantilstufe τ (nicht α), z = Φ⁻¹(τ). Residuum r = y − q̂ (Beobachtung minus Band); Pinball-Verlust ρ_τ(r) = τ·r für r ≥ 0 und (τ − 1)·r für r < 0 (nur als Definition benutzt). **Überdeckung** eines angepassten Bandes q̂(x) ist P(y ≤ q̂(x) | x) unter der wahren bedingten Verteilung, über die Punkte einer Gruppe gemittelt; **Soll = τ**. **Gesamtüberdeckung** = Mittel über alle Punkte (Rohskala: die 20 Gitterpunkte, Container: die 20 000 Testcontainer). Der Satz „Vorhersage ± z·s ist **kein** Konfidenzintervall der Steigung“ gehört dazu: das Band sagt etwas über den Rand der Verteilung eines einzelnen Containers, nicht über die Genauigkeit eines Koeffizienten.

**Zwei Verfahren.** (1) `Vorhersage ± z·s`: Kleinste Quadrate, s² = SSE/(n − p), Band ŷ + z·s; auf Wunsch (nur Container) mit **einem s je Art**, s_Art² = Mittel der Residuenquadrate der Art. (2) **Lineare Quantilregression** als Linearprogramm min Σ (τ·u + (1 − τ)·v) mit X·b + u − v = y, u, v ≥ 0 (HiGHS über `scipy.optimize.linprog`, dünne Matrix).

**Vier Welten und das Handbeispiel.**

| Welt | Modell | Gruppen |
|---|---|---|
| **Rohskala A** | y = 1 + 2x + (0,2 + 0,4x)·ε, x ~ Gleich(0; 10), ε normal; angepasst wird die Gerade | Drittel von x auf dem Gitter x = 0,25 … 9,75 (20 Punkte, 7 / 6 / 7) |
| **Rohskala B** | wie A, aber Streuung 0,2 + 0,04x² (konvex: 0,2 bei x = 0, 1,2 bei x = 5, 4,2 bei x = 10); es wird weiter eine **Gerade** angepasst | wie A |
| **Container** | synthetischer Generator der Verweilzeit (erfunden): Art (Import 35 %, Export 30 %, Umschlag 25 %, Leer 10 %), Zollhalt bei Import, Wochenende, Vorlauf bei Export und Umschlag, 12 Kunden mit Effekt; log-Verweilzeit normal mit log-Streuung 0,55 / 0,25 / 0,35 / 0,90 je Art. Angepasst wird die **log-Verweilzeit** (Kunde und Rauschmerkmal gehen nicht ein): **8 Parameter** (Achsenabschnitt, drei Art-Variable, Zollhalt, Wochenende, Vorlauf getrennt für Export und Umschlag) oder **11 Parameter** (zusätzlich ein Wochenend-Koeffizient je Art; im Generator wirkt das Wochenende nur bei Import und Leer) | die vier Arten; dazu die acht Zellen Art × Wochenende |
| **Handbeispiel** | zwei Gruppen mit je fünf Beobachtungen: 2, 3, 4, 5, 6 und 0, 4, 8, 12, 16 | die zwei Gruppen |

**Gilt** für normalverteilte Fehler (auf der Log-Skala bei den Containern), eine Gerade bzw. das lineare Modell mit den genannten Spalten, 20-Punkte-Gitter bzw. 20 000 feste Testcontainer. Die Überdeckung wird **exakt aus der wahren bedingten Verteilungsfunktion** gerechnet (Rohskala: Normal; Container: lognormal mit (μ, s) des Generators; ein Band bei oder unter 0 deckt auf der Rohskala 0, nie `log(0)`), nicht aus Zählungen von Testwerten; die Formel ist gegen Monte Carlo geprüft (siehe Verifikation).

## Methodik

| Baustein | Wie er arbeitet | Grenze |
|---|---|---|
| **Kleinste Quadrate** (`que_algorithm.ols_fit`, `sigma_hat`, `sigma_by_group`) | `numpy.linalg.lstsq`; s aus SSE/(n − p); s je Gruppe aus dem mittleren Residuenquadrat | braucht n > p; s je Gruppe setzt voraus, dass die Gruppe die Streuung erklärt |
| **Quantilregression** (`que_algorithm.qr_fit`) | Linearprogramm, HiGHS, dünne Matrix | bei ganzzahligem n·τ ist das Minimum bei einer Konstanten oder bei Gruppenmitteln (Handbeispiel) eine **Strecke**, mit stetigen Merkmalen in der Regel eindeutig; welches Ende der Löser liefert, ist eine Eigenschaft des Lösers (Tests vergleichen den Verlust) |
| **Überdeckung** (`cover_normal`, `cover_lognormal_log`, `cover_lognormal_raw`, `group_means`) | exakt aus der wahren Verteilungsfunktion; eine leere Gruppe gibt „–“, nie 0 | Normalverteilung der Fehler ist Annahme der Welt |
| **Handbeispiel** (`hand_analysis`, `quantile_interval`) | Brüche (`fractions`); Pinball-Minimierer je Gruppe als Strecke | nur Training, zehn Punkte |
| **Wiederholungen** (`que_evaluation.compute_cell`, `tools/que_make_frozen.py`) | je (Welt, n, τ) 40 bis 400 frische Stichproben; Gruppenmittel, Gesamtüberdeckung, 5./50./95. Perzentil, Anteil mit zu niedriger Gesamtüberdeckung, kleinster Gruppenwert, Kreuzen | **vorgerechnet und eingefroren** in `que_frozen.json` (282 s Rechenzeit für alle Zellen); die App liest nur die Datei |
| **Zufall** (`que_rng`) | ganzzahliger SplitMix64, Normalzahlen nach Box-Muller **nur aus Grundrechenarten und Wurzel** (Reihen für Logarithmus und Kosinus); keine C-Bibliothek, kein `numpy.random`, kein Python-`hash()`; Gleichheit mit der exakten Fassung auf 10⁻¹³ im Test | gilt für den Prüfling; Prüfdaten der Tests dürfen numpy benutzen |

**Live gerechnet** wird je Eingabe **eine** Stichprobe (Rohskala: einige Millisekunden bis rund 0,05 s; Container n = 3 000: rund 0,35 s für die drei Verfahren; Messung auf dem Entwicklungsrechner, nicht festgeschrieben, das Zeichnen kommt hinzu). Alles, was über Wiederholungen geht, ist **vorgerechnet** (`que_frozen.json`, erzeugt mit `python tools/que_make_frozen.py`; `--check` prüft die Datei gegen die Konstanten und rechnet zwei kleine Zellen neu). Die Zufalls-Seed-Regler der App wählen die eine Stichprobe, nicht die Wiederholungen.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen: Überdeckung (Anteil unter dem Band), Gruppenmittel dann Gesamtüberdeckung; Zähleinheit Wiederholung. „Test“ nennt die Testfunktion in `tests/test_claims.py` (bzw. die Datei), die genau diese Form prüft; die Zahlen der Wiederholungen stammen aus `que_frozen.json` (die Tests lesen die Datei und rechnen **zwei** kleine Zellen vollständig neu, `tests/test_frozen.py`), die exakten Werte (Brüche) sind im Test gerechnet.

### B1 Handbeispiel: im Mittel stimmt es, je Gruppe nicht (exakt, Brüche)

Zwei Gruppen mit je fünf Beobachtungen (2, 3, 4, 5, 6 und 0, 4, 8, 12, 16), τ = 4/5. Gruppenmittel 4 und 8, SSE = 10 + 160 = 170, **s² = 170/8 = 85/4**, s = 4,610, z(0,8) = 0,8416, z·s = 3,880. Die oberen Grenzen „Mittel + z·s“ sind 7,88 und 11,88: das Band deckt in Gruppe 0 **5 von 5**, in Gruppe 1 **3 von 5** (die 12 liegt darüber; knapp, 3,88 gegen 4, exakt über z² < 64/85), zusammen 8 von 10 = 0,8. Der Pinball-Verlust der Quantilregression hat je Gruppe das Minimum 2 und 8 (Summe **10**), und es ist eine **Strecke**: flach auf [5, 6] und auf [12, 16], weil n·τ = 4 ganzzahlig ist. An den unteren Streckenenden (5 und 12) sind es 4 von 5 in jeder Gruppe, an den oberen (6 und 16) 5 von 5: **die Überdeckung im Training ist eine Treppe mit Stufen 1/5, und die Wahl innerhalb der flachen Strecke ändert sie.** Die Überdeckung im Training zeigt also nur diese Treppe, nicht den Rand der Verteilung; darum rechnet die Demo die Überdeckung sonst gegen die wahre Verteilung. Test: `test_claims.test_b1_*`.

### B2 Sieben und acht Punkte (exakt)

Sieben Punkte x = 1 … 7, y = 1; 3,5; 2; 6; 4; 9; 5,5: der Pinball-Verlust des Linearprogramms gleicht der Aufzählung aller Geraden durch zwei Punkte: **41/16** (τ = 1/4), **37/8** (τ = 1/2), **51/40** (τ = 9/10). Bei τ = 0,9 gilt n·(1 − τ) = 0,7 < 1: **alle sieben Punkte liegen auf oder unter der Geraden, zwei genau darauf**; eine Stufe nahe 1 mit zu wenig Daten deckt im Training alles. Nur eine Konstante, n = 8 (1; 3/2; 2; 3; 4; 5; 6; 9): Verlust **43/8**, **33/4**, **57/8** bei τ = 1/4, 1/2, 3/4, jeweils auf der ganzen Strecke [3/2, 2], [3, 4], [5, 6]. Test: `test_claims.test_b2_*`.

### B3 Rohskala A: Grenzwert von „Vorhersage ± z·s“ (Rechnung, nicht gemessen)

Ist das Mittel richtig, strebt s² gegen E[sd(x)²] = 0,04 + 0,16·5 + 0,16·100/3 = **463/75 = 6,1733**, s = 2,485, und das Band deckt an der Stelle x mit Φ(z·s/sd(x)). Das gibt bei τ = 0,9 auf dem Gitter unten / Mitte / oben **0,996 / 0,926 / 0,820**, gesamt 0,914; bei x = 10 (sd 4,2) Φ(0,758) = 0,776. Der Fehler des Bandes ist darum **systematisch**, kein Stichprobenrauschen: gegen Quadratur und gegen eine Anpassung mit n = 100 000 geprüft, und bei n = 1 000 liegt die Messung oben bei 0,819 gegen 0,820 im Grenzwert. Test: `test_claims.test_b3_*`, `test_oracle_*`.

### B4 Rohskala A, n = 200, 400 Wiederholungen: das Band trifft nicht je Gruppe

| τ | Kleinste Quadrate ± z·s (unten / Mitte / oben, gesamt) | Quantilregression (unten / Mitte / oben, gesamt) |
|---|---|---|
| 0,1 | 0,004 / 0,076 / 0,181, gesamt 0,088 | 0,104 / 0,102 / 0,103, gesamt 0,103 |
| 0,5 | 0,502 / 0,499 / 0,499, gesamt 0,500 | 0,497 / 0,498 / 0,498, gesamt 0,498 |
| **0,9** | **0,996 / 0,924 / 0,818**, gesamt 0,912 | **0,897 / 0,899 / 0,898**, gesamt 0,898 |

Beim Median fällt das Band auf die Vorhersage zusammen (z = 0): beide treffen, hier ist nichts zu sehen (Preset „Ohne Effekt“). Test: `test_claims.test_b4_*`.

### B5 Rohskala A, τ = 0,9: wie viele Daten braucht die Quantilregression?

400 Wiederholungen (150 bei n = 1 000). „Unter τ − 0,05“ heißt Gesamtüberdeckung unter 0,85.

| n | Quantilregression gesamt (Mittel) | 5. bis 95. Perzentil | Anteil unter 0,85 | Kleinste Quadrate ± z·s gesamt |
|---|---|---|---|---|
| 30 | 0,872 | 0,762 bis 0,951 | 33,0 % | 0,899 |
| 50 | 0,886 | 0,812 bis 0,951 | 20,2 % | 0,906 |
| 100 | 0,890 | 0,828 bis 0,939 | 10,8 % | 0,908 |
| 200 | 0,898 | 0,864 bis 0,928 | 2,0 % | 0,912 |
| 1 000 | 0,899 | 0,882 bis 0,915 | 0,0 % | 0,913 |

**Hohe Stufe τ = 0,99** (Gesamtüberdeckung der Quantilregression): 0,937 / 0,963 / 0,980 / 0,986 / 0,989 für n = 30 / 50 / 100 / 200 / 1 000; Anteil unter 0,94: 42,2 % bei n = 30, 0,0 % bei n = 200 und 1 000. **Ausnahme:** bei τ = 0,99 und kleinem n liegt Kleinste Quadrate ± z·s im Mittel (0,971 bei n = 30) **näher am Soll** als die Quantilregression (0,937), auch im kleinsten Gruppenwert (0,930 gegen 0,891). Test: `test_claims.test_b5_*`.

### B6 Rohskala A, τ = 0,9: Gruppen bei kleinem n

| n | kleinster Gruppenwert einer Wiederholung, Mittel: Kleinste Quadrate ± z·s / Quantilregression | Anteil der Wiederholungen mit einer Gruppe unter 0,8: ± z·s / Quantilregression |
|---|---|---|
| 30 | 0,799 / 0,816 | 46,0 % / 35,2 % |
| 100 | 0,812 / 0,863 | 38,0 % / 9,0 % |
| 200 | 0,818 / 0,880 | 26,5 % / 0,8 % |

Gruppenwerte bei n = 30: Kleinste Quadrate ± z·s 0,992 / 0,907 / 0,799, Quantilregression 0,872 / 0,880 / 0,866. Test: `test_claims.test_b6_*`.

### B7 Kreuzen der Quantilgeraden (Rohskala A, τ = 0,9)

Anteil der Wiederholungen, in denen die Gerade der Stufe 0,1 an einer der **zwei Randstellen** x = 0 oder x = 10 über der Geraden der Stufe 0,9 liegt: **13,8 % / 12,0 % / 3,5 % / 1,0 % / 0,0 %** für n = 30 / 50 / 100 / 200 / 1 000. Jeder Wert ist eine einzelne Ziehung aus 400 Wiederholungen (Standardfehler bis 1,9 Punkte); der Wert bei n = 30 ist eine eher niedrige Ziehung (eine frühere Messung nannte 17,5 %), gesichert ist nur, dass das Kreuzen mit n fällt und bei n = 1 000 ausbleibt. Nur diese zwei Randstellen sind geprüft; Rearrangement (Sortieren der Quantile) ist nicht gemessen. Test: `test_claims.test_b7_*`.

### B8 Container (Modell mit 8 Parametern), τ = 0,9: ein s gegen ein s je Art gegen Quantilregression

n = 3 000, 40 Wiederholungen, Test auf 20 000 festen Testcontainern. Spalten: gesamt, dann Import / Export / Umschlag / Leer.

| Verfahren | gesamt | Import | Export | Umschlag | Leer |
|---|---|---|---|---|---|
| Kleinste Quadrate (log) ± z·s, ein s | 0,918 | 0,859 | 0,992 | 0,967 | **0,775** |
| Kleinste Quadrate (log) ± z·s_Art | 0,898 | 0,898 | 0,898 | 0,899 | 0,900 |
| Quantilregression (log) | 0,898 | 0,900 | 0,896 | 0,898 | 0,898 |
| Kleinste Quadrate roh ± z·s, ein s | 0,959 | 0,934 | 0,996 | 0,998 | 0,833 |
| Quantilregression roh | 0,899 | 0,900 | 0,898 | 0,897 | 0,898 |

Bei τ = 0,1: ein s gesamt 0,084 (je Art 0,142 / 0,009 / 0,035 / 0,237); auf der Rohskala deckt das Band bei Export und Umschlag **0,000** (die Untergrenze liegt unter 0). **Die Quantilregression ist hier kein Gewinn, sondern ein Umweg:** Kleinste Quadrate mit einem s je Art trifft in jeder Art so gut wie sie (n = 3 000, Modell mit 8 Parametern). **Bei n = 300** (100 Wiederholungen) streut die Quantilregression stärker: kleinster Artwert einer Wiederholung im Mittel 0,838 gegen 0,857 bei s je Art, Anteil der Wiederholungen mit einer Art unter 0,8: 18,0 % gegen 8,0 % (nur 100 Wiederholungen, Standardfehler bis etwa 4 Punkte: die Richtung gilt, die Größe ist ungenau). Das gilt, wenn die Streuung einer **bekannten Gruppe** folgt; bei stetiger oder in der Form unbekannter Streuung (Rohskala A, B) gewinnt die Quantilregression. Test: `test_claims.test_b8_*`.

### B9 Zellen Art × Wochenende: die Quantilregression repariert kein falsches Mittelwertmodell

Container, n = 3 000, 40 Wiederholungen, acht Zellen. **Größte Abweichung einer Zelle von τ:**

| τ | Modell | Kleinste Quadrate ± z·s_Art | Quantilregression |
|---|---|---|---|
| 0,9 | 8 Parameter (ein Wochenend-Koeffizient) | 0,051 | 0,045 |
| 0,9 | 11 Parameter (je Art) | 0,004 | 0,007 |
| 0,5 | 8 Parameter | 0,131 | 0,089 |
| 0,5 | 11 Parameter | 0,010 | 0,009 |

Erst das fehlende Wochenend-Glied je Art senkt die Abweichung; die Quantilregression allein tut es nicht (bei τ = 0,9 0,045 gegen 0,051 bei s je Art). Bei n = 300 (100 Wiederholungen) und 11 Parametern: 0,018 (s je Art) gegen 0,029 (Quantilregression). Test: `test_claims.test_b9_*`.

### B10 Falsche Klasse: Rohskala B, Gerade angepasst, τ = 0,9, n = 1 000, 100 Wiederholungen

Kleinste Quadrate ± z·s: 1,000 / 0,971 / 0,806, gesamt 0,923. **Quantilregression: 0,893 / 0,946 / 0,869**, gesamt 0,900; kleinster Gruppenwert einer Wiederholung im Mittel 0,866 (± z·s: 0,806). Die Quantilregression ist besser, trifft aber die obere Gruppe nicht (0,869) und deckt die mittlere zu viel (0,946): im Mittel stimmt es, je Gruppe nicht. Test: `test_claims.test_b10_*`.

### B11 Optimalitätsbedingung des Achsenabschnitts (eigene Herleitung)

`#{r < 0} ≤ n·τ ≤ #{r ≤ 0}` für das Residuum des Linearprogramms: in **300 von 300** Zufallsinstanzen mit ganzzahligen y (alle 300 mit mindestens einem Punkt auf der Geraden) erfüllt (`python tools/que_gegenprobe.py`: n = 5 bis 60, p = 2 bis 4, sechs Stufen); in den 81 Instanzen mit p = 2, n ≤ 40 und verschiedenen x stimmt der Verlust mit der Aufzählung aller Geraden durch zwei Punkte überein (0 Abweichungen). Die Tests prüfen 50 Instanzen (`tests/test_oracle_*.py`, `tests/test_eigenschaften_*.py`). Die Bedingung ist eine eigene Herleitung aus dem Teilgradienten, kein Zitat aus der Literatur.

### Hinweise zu den Zahlen

* Jede Zahl der Wiederholungen nennt Welt, n, τ und die Zahl der Wiederholungen; Prozentzahlen zählen Wiederholungen, nicht Beobachtungen. Anteile mit 40 bis 400 Wiederholungen haben einen Standardfehler bis etwa 0,08 (bei 40) bzw. 0,025 (bei 400): **stellen unter 0,01 sind Rauschen**, darum nennen die Tests Bänder (Zählungen ± 2, Anteile ± 0,02 bis 0,05, Überdeckungen ± 0,01).
* Plattform: Überdeckungen aus Neu-Rechnung als Band (HiGHS-Ecken in flachen Strecken, Gleitkomma-Summen); exakt nur, was in Brüchen gerechnet ist (B1, B2, 463/75).
* „Gruppe“ heißt je Welt etwas anderes (Drittel von x, Containerart, die zwei Gruppen des Handbeispiels); „kleinster Gruppenwert“ ist das Minimum über die Gruppen **je Wiederholung**, dann gemittelt.

## Befunde und Korrekturen gegenüber der ersten Rechnung

Die erste Rechnung benutzte numpy-Zufallsströme und den Generator der Nachbar-Demo; die Demo misst mit ihrem eigenen Zufallsstrom (SplitMix64) und einem Nachbau des Generators neu. Die Aussagen bleiben, einzelne Zahlen verschieben sich:

* **Quantilregression in Rohskala A (n = 200, τ = 0,9):** 0,897 / 0,899 / 0,898 gegen 0,891 / 0,894 / 0,892 in der ersten Rechnung (gesamt 0,898 gegen 0,892); „Kleinste Quadrate ± z·s“ 0,996 / 0,924 / 0,818 gegen 0,996 / 0,921 / 0,814. Der Grenzwert 0,996 / 0,926 / 0,820 ist unverändert (Rechnung).
* **n = 30:** Gesamtüberdeckung der Quantilregression 0,872 gegen 0,877; Anteil unter 0,85 33,0 % gegen 30,7 %; Anteil mit einer Gruppe unter 0,8: Quantilregression unverändert 35,2 %, Kleinste Quadrate ± z·s 46,0 % gegen 42,5 %. Kreuzen 13,8 % gegen 17,5 % (n = 30), 12,0 % gegen 11,3 % (n = 50): die Unterschiede liegen knapp im Rahmen von etwa zwei Standardfehlern bei 400 Wiederholungen, die 13,8 % sind eine einzelne, eher niedrige Ziehung.
* **Gruppengrößen auf dem Gitter:** die erste Rechnung nannte 6 / 7 / 7 Gitterpunkte; richtig sind **7 / 6 / 7** (x < 10/3, dann bis 20/3, dann der Rest). Die Zahlen blieben davon unberührt, weil der Code dieselbe Aufteilung benutzte.
* **Zellen Art × Wochenende mit 11 Parametern:** „höchstens 0,005 für beide“ gilt nicht mehr: 0,004 (s je Art) und **0,007** (Quantilregression, τ = 0,9, n = 3 000).
* **Neu gemessen:** die Stufen 0,8 (Handbeispiel) und 0,99 im Container, Rohskala B auch bei n = 200, das Modell mit 11 Parametern bei n = 300, die Stufe 0,99 für die kleinen Mengen: bei **n = 300** streut die Quantilregression im Container stärker als ein s je Art (B8, B9). Dass bei τ = 0,99 und kleinem n **auch im kleinsten Gruppenwert** Kleinste Quadrate ± z·s besser liegt (0,930 gegen 0,891 bei n = 30), war in der ersten Rechnung nur für die Gesamtüberdeckung vermerkt.
* **Ehrlicher Zusatz:** „Die Quantilregression trifft in jeder Gruppe“ gilt im Mittel über die Wiederholungen in den gemessenen Zellen ab n = 200 (Rohskala A, τ = 0,9: größte Abweichung des Gruppenmittels 0,003, Rauschen etwa ±0,003) und nicht bei n = 30 (0,034) oder Rohskala B (0,046); eine einzelne Stichprobe weicht je Gruppe stärker ab. „Vorhersage ± z·s ist falsch“ gilt je **Gruppe**; auch im Mittel über alle Gruppen liegt es daneben, aber weniger (gesamt 0,912 gegen 0,898 bei n = 200).

## Quellen

Nur, soweit gelesen; es wird kein Wortlaut übernommen, nur Gleichungen und Befunde in eigenen Worten.

* Koenker und Bassett 1978, *Regression Quantiles*, Econometrica 46(1), 33–50: **Abstract** gelesen (Regressionsquantile als Verallgemeinerung der Stichprobenquantile auf das lineare Modell, ein einfaches Minimierungsproblem); der Wortlaut ist nicht gelesen. Koenker 2005, *Quantile Regression*, Cambridge University Press: nur die Kapitelliste. Dass die Quantilregression ein Linearprogramm ist, steht hier als eigene Formulierung (nicht aus diesen Quellen belegt); der Pinball-Verlust ist als Definition benutzt.
* Foygel Barber, Candès, Ramdas und Tibshirani, *The limits of distribution-free conditional predictive inference*, arXiv 1903.04684 (Abstract gelesen: exakte **bedingte** Garantien ohne Verteilungsannahme sind unmöglich; die Aussage zu Intervallen mit unendlicher erwarteter Länge nach Quellenbericht, im Original nicht nachgelesen): die **marginale** Garantie (im Mittel über die Merkmale) geht dagegen verteilungsfrei. Für vorab festgelegte Gruppen gibt es gruppenweise konforme Varianten, hier nicht nachgebaut. Das passt zur Falle dieser Demo und wird nur genannt, nicht nachgebaut. Konforme Quantilregression (Romano, Patterson und Candès, arXiv 1905.03222) ist nur als Verweis erwähnt.
* Zur Namensfalle HC3 (nach MacKinnon 2011, Abschn. 2, Wortlaut gelesen): das ursprüngliche HC3 von MacKinnon und White 1985 ist ein Jackknife-Verfahren; das heute so genannte HC3 (Davidson und MacKinnon 1993) nimmt û²/(1 − h)². Die Standardfehler der Koeffizienten, auch diese Familie, gehören zu [`streuung-wls-demo`](https://github.com/sebastian-hanisch/streuung-wls-demo) und sind hier **nicht** gemessen.
* Nicht gelesen und nicht benutzt: Koenker/Bassett im Wortlaut, Koenker 2005 über die Kapitelliste hinaus, Rearrangement der Quantile (Chernozhukov u. a.).

## Ehrliche Grenzen

* **Normalverteilte Fehler, eine Krümmung.** Rohskala A mit linearer Streuung, B mit einer quadratischen; schiefe Fehler sind in dieser Demo nicht gemessen. Beim Median fällt „Vorhersage ± z·s“ auf die Vorhersage zusammen und trifft den Median nur, wenn Mittelwert und Median übereinstimmen (bei normalen Fehlern der Fall); bei schiefen Fehlern ist das eine offene Frage dieser Demo.
* **Container:** der Kunde (12 Cluster) bleibt unerklärte Streuung und ist nicht als Gruppe gemessen; es gibt keine Anpassung mit Kundenvariablen. Nur sechs Stufen (0,1 bis 0,99) und zwei Datenmengen (300, 3 000), 40 bzw. 100 Wiederholungen: Anteile haben einen Standardfehler bis etwa 0,08 (siehe oben).
* **Kreuzen** ist nur an den zwei Randstellen x = 0 und x = 10 und nur für die Gerade in einem Merkmal geprüft.
* **s je Art** (Teiler: Zahl der Container der Art, nicht n_Art − p) ist ein Verfahren dieser Demo, kein Standardverfahren; „es reicht“ gilt nur, wenn die Streuung einer bekannten Gruppe folgt.
* **Wiederholungen aus der Datei:** `tests/test_claims.py` liest `que_frozen.json`; neu gerechnet werden nur zwei Zellen (Rohskala A n = 30, τ = 0,9; Container 8 Parameter n = 300, τ = 0,9). Die übrigen Zellen sind gegen die Datei, nicht gegen eine frische Rechnung geprüft; `python tools/que_make_frozen.py` erzeugt sie neu (282 s, ein Prozess).
* **Handbeispiel:** die Überdeckung ist dort im Training gezählt (zehn Punkte), nicht gegen eine wahre Verteilung.
* **Keine Neuheit:** nichts hier ist wissenschaftlich neu.

## Verifikation

* **Handrechnung in Brüchen** (`tests/test_algorithm.py`): Handbeispiel (Gruppenmittel, SSE, s², z²-Bedingungen, Pinball-Minima, Strecken), sieben Punkte (41/16, 37/8, 51/40), eine Konstante bei n = 8 (43/8, 33/4, 57/8), s² = 463/75, Φ(0,758).
* **Orakel** (`tests/test_oracle_quantil_ueberdeckung.py`): das **Linearprogramm gegen die Aufzählung aller Geraden durch zwei Punkte** (40 Instanzen × 4 Stufen, Verlust, nicht Koeffizienten) und gegen `scikit-learn` (`QuantileRegressor`, nur im Test, `pytest.importorskip`); die **Überdeckungsformel gegen Monte Carlo** (400 000 Ziehungen, vier Standardfehler) und gegen die realisierten Verweilzeiten des Generators je Art (4,5 Standardfehler); der **Grenzwert Φ(z·s/sd) gegen Quadratur** und eine Anpassung mit n = 100 000; `fit_cover` gegen einen unabhängigen Weg (Normalgleichung, `scipy.stats.norm`, eigene Gruppenzählung) in Rohskala A und im Container.
* **Eigenschaftstests** (`tests/test_eigenschaften_quantil_ueberdeckung.py`): Verschieben und Skalieren von y, Zeilentausch, Median = Kleinste Quadrate (z = 0), Verschieben von x ändert die Überdeckung nicht, die wahre Quantilgerade deckt τ, die Überdeckung wächst mit τ, Optimalitätsbedingung mit Gleichständen, gleicher Zufalls-Seed gleiche Stichprobe.
* **Randfälle** (`tests/test_edge_cases.py`): n = p, n < p (Meldung), Stufe nahe 1, leere Gruppe (nie 0), Band bei oder unter 0, konstantes y, doppelte x, Kreuzen, ungültige Stufen.
* **Verdikte** als reine Funktionen (`tests/test_verdicts.py`): jede Schwelle und jede Zahl im Text; je Preset die erwartete Zeile (das Preset ohne Effekt meldet keine Abweichung).
* **Claims** (`tests/test_claims.py`): jede Zahl dieser README, gerechnet und im Text gesucht.
* **Eingefrorene Messung** (`tests/test_frozen.py`): Raster, Wiederholungen und Verfahren gegen die Konstanten, zwei Zellen neu gerechnet, innere Stimmigkeit.
* **Zufall und Generator** (`tests/test_rng.py`, `tests/test_scenario.py`): SplitMix64 gegen die Referenzfolge, Normalzahlen gegen die exakte Fassung; Kundeneffekte gegen `numpy.random.default_rng(12345)`, Mittelfunktionen je Art von Hand, Entwurfsmatrizen von Hand.
* **Oberfläche und Layout:** AppTest-Rauchtest (jeder Button, jeder Regler an den Grenzen, jede Welt, jede Menge, jede Stufe), Schnellstart oben in der Seitenleiste, aktives Preset hervorgehoben (jede Änderung eines Reglers hebt es auf), Footer, Laufzeit-Importe.
* **Gegenprobe im Großen:** `python tools/que_gegenprobe.py` (300 Instanzen, B11).

174 Tests (`pytest tests`).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: Ergebnis, Kernfragen, zwei Experimente, Grenzen, Formeln |
| `que_algorithm.py` | Kern: Kleinste Quadrate, Quantilregression (LP), Überdeckung, Handbeispiel in Brüchen |
| `que_constants.py` | Welten, Regler-Grenzen, Parameter des Generators, Presets |
| `que_evaluation.py` | eine Stichprobe auswerten, Wiederholungen zusammenfassen, Verdikte, Tabellen |
| `que_frozen.json` | die vorgerechneten Wiederholungen (erzeugt von `tools/que_make_frozen.py`) |
| `que_layout.py` | Fester Kopf (Stylesheet) |
| `que_presets.py` | Permalink (Begrenzen), Presets (Schnellstart in der Seitenleiste), Zufalls-Seed |
| `que_rng.py` | SplitMix64 mit Normalzahlen ohne C-Bibliothek (Kopie aus `plan-stoerung-demo`) |
| `que_scenario.py` | Welten: Rohskala A und B, Container-Generator (Nachbau), Entwurfsmatrizen, Handbeispiel |
| `que_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `requirements-dev.txt` | Laufzeit plus pytest und scikit-learn (nur Gegenprobe) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `tools/que_make_frozen.py` | erzeugt und prüft `que_frozen.json` |
| `tools/que_gegenprobe.py` | Gegenprobe der Optimalitätsbedingung und des LP an 300 Instanzen |
| `tests/` | Handrechnung, Orakel, Eigenschaften, Randfälle, Verdikte, Claims, eingefrorene Messung, Zufall, Generator, Presets, Layout, AppTest, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

* **Konforme Bänder und Rearrangement** der Quantile (nur als Verweis): dort gilt die marginale Garantie, hier geht es um die Gruppen.
* **Pinball-Verlust als Gegenstand, Kosten, Kostenverhältnis und L1-Strafe:** das ist [`newsvendor-direkt-demo`](https://github.com/sebastian-hanisch/newsvendor-direkt-demo).
* **Intervalle für die Koeffizienten der Quantilregression, Kreuzvalidierung der Stufe, Echtdaten:** nicht gebaut.
* **Kunde als Gruppe und gemischte Modelle:** [`cluster-gemischt-demo`](https://github.com/sebastian-hanisch/cluster-gemischt-demo).
* **PDF-Export:** hat diese Demo nicht.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy, SciPy und Pandas.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Regression und GLM: wann die Gleichung trügt](https://sebastianhanisch.net/konzepte-regression.html).

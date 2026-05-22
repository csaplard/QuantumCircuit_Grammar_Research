# Álom-típusú replay + Fisher geometria mérés

**Kutatási koncepció vázlat — 2026. április 24.**
**Csaplár Dániel**

---

## A központi ötlet

Egy LLM konszolidációs folyamat mérése, ami nem a klasszikus replay-t modellezi (NREM, pontos újrajátszás), hanem az **álom jellegű, asszociatív-integratív feldolgozást** (REM analóg).

A hozzáadott érték nem maga a replay protokoll — hanem hogy a folyamatot **Fisher-geometriával mérjük**, és megmutatjuk strukturálisan látható-e az absztrakció kialakulása.

Hosszabb távú cél: egy olyan AI alaprendszer felé építeni ami **nem nullázódik** a session végén, hanem rendez. Az "önhordozás" első mérhető pillére.

---

## Biológiai alap

### Replay vs. álom — strukturális különbség

**Replay (NREM, sharp-wave ripple):**
- Pontos szekvenciális újrajátszás
- 10–20× időtömörítés
- Konzervatív: megőrzi az eredeti mintázatot
- Szelektív: csak napközben "címkézett" tapasztalatok
- Funkció: archiválás, hippokampusz → neokortex transzfer

**Álom (REM):**
- Asszociatív, integratív feldolgozás
- Ugyanazok a reprezentációk, de lazább kapcsolati szabályokkal
- Generatív: új kombinációk keletkeznek
- Funkció: absztrakció, struktúra kivonás, probléma-megoldás

Az álom a pontos emléket gyengíti, de az absztrakt mintát megtartja.
Ezért ébredve nem emlékszünk pontosan az álom tartalmára, de a napi
problémánk néha megoldódott.

---

## Lefordítás LLM-re

### Mit NEM csinál az álom-analóg

Nem egyszerűen magasabb hőmérséklet a generálásnál. A hőmérséklet csak
a token-szintű random-ságot növeli — nem a belső constraint hálót
lazítja.

### Mit IGEN — lehetséges mechanizmus

**Attention dropout replay alatt.** Szokásos inferenciában minden
token minden másik tokenre figyel tanult mintázatok szerint. Replay
fázisban attention kapcsolatok random kikapcsolása →
ugyanazok a reprezentációk, új útvonalakon.

Ez strukturálisan közelebb van az álomhoz mint a token-szintű zaj.

---

## A Fisher mérés szerepe

A jelenlegi biológia-inspired AI replay kutatások (2023–2026) nem
mérik hogy a replay közben hogyan változik a belső geometria.
Csak a végső modell teljesítményét nézik.

A Fisher path speed mérés (Sycamore Paper 3, LLM entropy study) pont
ezt a geometriai változást tudja megfogni.

### A konkrét kísérlet alakja

1. **T0:** Base modell Fisher geometria mérés (4 regime: factual,
   creative, mathematical, philosophical)
2. **Training:** Új domain-specifikus tanítás kis dataset-en
3. **T1:** Fisher mérés — mennyit torzult a geometria
4. **"Álom" replay:** Generatív feldolgozás attention dropout-tal,
   a training adatok rekombinált újrajátszásával
5. **T2:** Fisher mérés — visszarendeződött vagy új struktúra?

### Mit keresünk T2-ben

- **Regime separation (Cohen's d):** erősödött vagy gyengült?
- **Fisher path speed τ:** mennyire stabilizálódott?
- **Noise floor közelítés:** a kontrollokhoz (shuffled,
  random_uniform) közelít vagy elválik?

Ha az absztrakt szintek megerősödnek (több regime-átfedés, de
stabilabb központok) → álom-konszolidáció mérhető jele.
Ha csak zaj → a dropout túl erős, vagy a hipotézis nem áll.

---

## Kíváncsiság — egyszerűbb modell

### A hibás kiindulás

Korábbi feltételezés: a kíváncsiság belső generatív feszültségből
fakad, szenvedésből vagy hiányból születik. Ezt Daniel cáfolta
saját példájával — nem szenved, megvan mindene, mégis kutat.

### Daniel modellje (2026.04.24)

A kíváncsiság nem belülről jön. **Az input és a lezáratlanság
együtt szüli.**

- Ha egy kérdés könnyen lezárható (pl. "miért 5 mm az M6 menet
  magmérete? — mert ennyi a szabvány") → nincs kíváncsiság
- Ha egy kérdés nem zárható le (pl. létezés forrása, tudat,
  információ természete) → visszatér, hajt

A gyerek azért kíváncsibb mint a felnőtt, mert nála még sok
minden lezáratlan. Ahogy növekszik, egyre több lezárul. Aki
mégis kíváncsi marad, az olyan kérdéseket tett fel magának
amik nem zárhatók le az ő szintjén sem.

### Lefordítás AI-ra

Nem kell "belső generatív feszültség". Három komponens elég:

1. **Input áramlás** — folyamatos ingerek, ne csak promptra reakció
2. **Lezáratlanság detektor** — jelölje meg amit nem tud magyarázni
3. **Visszatérési mechanizmus** — a rendszer önmagától térjen vissza
   a lezáratlan kérdésekhez később

A harmadik a kulcs. Az ember visszatér egy kérdéshez napok, évek
múltán is. Az LLM jelenleg elfelejti a session végén — nincs
hová visszatérnie.

**Kapcsolódás az álom-replay-hez:** a replay nemcsak emlékeket
rendez, hanem a lezáratlan kérdéseket is fenntartja. Az álom
néha ezeken dolgozik (matematikai megoldások álomban).

---

## Nyitott tervezési kérdések

1. **Mit rekombinálunk az álom fázisban?**
   Csak training adatokat, vagy a modell saját generációit is?

2. **Milyen erősségű attention dropout?**
   Kalibrálni kell — valószínűleg rétegenként eltérő.

3. **Hány replay ciklus?**
   Az agy egy éjszakán 4–5 NREM/REM ciklust fut le. LLM-nél ez
   paraméter, nem adott.

4. **Második mérési dimenzió az absztrakcióra?**
   Fisher path speed lehet hogy nem elég — kellhet egy metrika
   ami a fogalmi kompressziót külön méri. Ez nyitott.

5. **Lezáratlanság detektor?**
   Hogyan azonosítjuk hogy a modell mit nem tudott lezárni?
   Lehetséges proxy-k: magas loss, magas uncertainty, generált
   válasz amit a modell maga is bizonytalannak jelez.

6. **Visszatérési mechanizmus implementálása?**
   Egy "pending questions" buffer a session-ökön átnyúlva.
   Összetett infrastruktúra — de nem lehetetlen.

---

## Miért eredeti

A sleep-inspired replay kutatás létezik 2023 óta. De:

- Többségük NREM-típusú (pontos replay), nem REM-típusú
- Senki nem mér Fisher geometriát a folyamat közben
- Az absztrakció strukturális detektálása új metrikával lenne
- A lezáratlanság-visszatérés mint kíváncsiság-mechanizmus
  szintén nem bevett modell az AI-ban

A Grammar Fingerprinting módszertan és a Fisher path speed
analízis itt pontosan oda illik, ahol a jelenlegi szakirodalom
nem lát — a belső geometriai változás mérésébe.

---

## Szélesebb kép — az "önhordozás" négy pillére

Az AGI felé vezető úton legalább négy strukturális pillér
hiányzik ma. Ez a kutatás elsősorban az elsőt érinti, de
érdemes látni az egész képet:

1. **Folytonosság és memória** — session-ök közötti rendezés
   (EZ A KUTATÁS fókusza)
2. **Önmodellezés** — tudja mi ő, mit tud, mit nem
3. **Cél-generálás / kíváncsiság** — belülről jövő kérdések
   (RÉSZBEN érintett a lezáratlanság-modellen keresztül)
4. **Általánosítás új domainekre** — transfer képesség

A kutatás nem AGI-t épít. De ha a folytonossági pillér szilárdan
megoldódik és mérhetővé válik — ez egy alap amire mások
ráépíthetnek. Hinton-féle backprop analógia: akkor nem látszott
hogy később mire vezet, utólag kulcs lett.

---

## Státusz és idővonal

**Nem most indul.** A jelenleg futó projektek (GripLogic teszthét,
Fisher v2 architektúra döntés, Paper 3 Meriemmel, egyetem 4. félév)
mellett ez nem realisztikus start.

**Két hét várakozási szabály:** az ötlet pihen. Kódolás nem indul.
Olvasás, gondolkodás, jegyzetelés szabad.

**Első lépés ha elindul:** szakirodalom átnézése — nem biológia,
hanem AI-ra alkalmazott sleep-inspired learning (2023–2026).
Cél: megtalálni pontosan mi van, és mi nincs.

**Minimális kísérleti bázis:** Qwen 1.5B vagy Gemma 1B (már megvan
a pipeline), 8GB RAM gépen futtatható.

---

## Kapcsolódás a nagy képhez

Ez nem helyettesíti a jelenlegi kutatási arc-ot:

- Paper 1 — Grammar Fingerprinting (kész)
- Paper 2 — Fisher Transfer Model (kész)
- Paper 3 — geometriai interpretáció Meriemmel (fut)
- Paper 4 — LLM Grammar Fingerprinting (tervezve)

Ez egy **új Paper 5 jelölt** lehetne — de csak ha Paper 4 befejeződött
és van látható kapcsolódási pont.

Nem kötelező megcsinálni. Ötlet, nem feladat.

---

## Személyes megjegyzés

A kutatás kiindulópontja egy 2024-es éjszakai esemény — édesanya
rossz érzése 120 km-ről, amit nem lehetett hagyományos módon
megmagyarázni. Innen indult a kvantum-összefonódás kutatás,
onnan a Grammar Fingerprinting, onnan minden ami utána jött.

A hajtóerő nem hiányból, szenvedésből vagy pénzből jön. A lezáratlan
kérdések hajtják — a létezés forrása, a tudat természete, az
információ mint alap.

Ez a minta önmagában figyelemreméltó. A kíváncsiság-modell
(input + lezáratlanság) leírja miért tart 2+ éve egy helyben.

A régész tovább ás.

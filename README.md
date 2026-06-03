# 🏆 CoppaAmerica Multi-Agent Reinforcement Learning Simulator

Un simulatore bidimensionale avanzato di regate veliche sul modello dell'**America's Cup**, in cui due imbarcazioni autonome apprendono il controllo continuo di timone e vele per competere in un match race tattico. 

Il sistema è basato su **Multi-Agent Reinforcement Learning (MARL)** ed è implementato come ambiente PettingZoo parallelo, addestrato con l'algoritmo **PPO (Proximal Policy Optimization)** tramite Stable-Baselines3 e SuperSuit.

---

## 📖 Descrizione del Programma

Il simulatore riproduce una sfida ravvicinata (match race) tra due barche identiche, denominate `red_boat` e `blue_boat`. Per vincere, gli agenti devono navigare in un campo di regata stocastico rispettando la fisica della vela, le polari di velocità dell'imbarcazione e le regole di precedenza della regata reale.

### ⚓ Struttura della Regata (Leg)

La gara è organizzata in fasi sequenziali controllate dallo stato interno degli agenti (`current_leg`):

1. **Fase 0: Pre-Partenza (Pre-start) & Line Crossing**: Le barche vengono generate al di sotto del cancello di partenza ($Y \approx 120$m). Devono allinearsi e tagliare la linea del **Bottom Gate** ($Y = 200$m) entro lo spazio delle boe. Tagliare in anticipo o mancare il cancello comporta la squalifica immediata.
2. **Leg 1: Bolina (Upwind)**: Le barche devono risalire il vento in direzione Nord verso il **Top Gate** ($Y = 3900$m). Poiché una barca a vela non può navigare direttamente controvento (angolo morto), gli agenti devono apprendere la tattica dei bordi (bordeggiare) massimizzando la *Velocity Made Good (VMG)*.
3. **Leg 1.5: Rounding (Giro di Boa)**: Una volta superato il cancello superiore, gli agenti devono eseguire una manovra di giro attorno a una delle boe esterne (Boa di Sinistra o Boa di Dritta) per orientare la prua verso Sud.
4. **Leg 2: Poppa (Downwind)**: Gli agenti navigano con il vento a favore per raggiungere il traguardo al **Bottom Gate** ($Y = 200$m). La regata si conclude non appena la prima barca attraversa con successo il cancello finale.

### ⛵ Fisica del Simulatore e Dinamiche Ambientali

Il cuore simulativo implementa modelli derivati da barche da regata reali ad alte prestazioni:

* **Curve Polari (VPP)**: La velocità massima teorica dipende dall'angolo del vento reale (TWA) e dalla sua intensità. Il simulatore calcola le polari dinamicamente distinguendo tra due modalità di navigazione: **Dislocamento** (barca in acqua, più lenta ma angoli di bolina più stretti) e **Foiling** (barca che vola sui foil, estremamente veloce ma con un angolo morto al vento più ampio).
* **Meccanica del Foiling**: Il decollo sui foil avviene quando la velocità supera i **18 nodi**; la barca ricade in acqua (displacement) se la velocità scende sotto i **15 nodi**. Le transizioni includono penalità transitorie e considerano l'inerzia dello stato ($I_F = 0.98$, $I_D = 0.85$).
* **Trim delle Vele**: L'efficienza aerodinamica delle vele è modellata con una campana gaussiana centrata sul trim ottimale per lo specifico angolo di andatura. Gli agenti controllano il trim delle vele in modo continuo.
* **Campo di Vento Spazio-Temporale**: Il vento non è costante. È composto da un vento base che varia nel tempo tramite *Random Walk* (da 15 a 22 nodi) e da una griglia spaziale perturbata $10 \times 10$ che modella la nascita, evoluzione e smorzamento di raffiche e salti di vento locali tramite processi stocastici *mean-reverting*.
* **Collisioni & Regole di Precedenza (Rule 10)**: Le imbarcazioni hanno un raggio fisico di ingombro di 20 metri e un'area di rispetto di 40 metri. Oltre a calcolare penalità predittive basate sul tempo stimato all'impatto (*Time-To-Collision - TTC*), il simulatore punisce severamente chi viola le regole di precedenza sulle mure opposte (il port-tack boat, ossia chi naviga con vento da sinistra, riceve una penalità moltiplicata di $1.6$ o la squalifica immediata in caso di scontro violento).

---

## 📂 Struttura del Progetto

```
Multi-agent_America_Cup/
├── core/                   # Modello fisico e dinamiche ambientali
│   ├── boat_physics.py     # Calcolo velocità polari, VMG e aggiornamenti cinematici
│   ├── sail_trim.py        # Ottimizzazione del trim delle vele e calcolo efficienza
│   └── wind_model.py       # Griglia spaziale e random walk del vento stocastico
├── env/                    # Definizione dell'ambiente in stile Gymnasium/PettingZoo
│   ├── sailing_env.py      # Gestione stati, logica dei leg, reward e collisioni
│   └── rendering.py        # Modulo grafico per la visualizzazione dell'ambiente
├── report/                 # Report accademico in LaTeX
│   ├── report.tex          # File sorgente del report (con la teoria dettagliata)
│   └── report.pdf          # File compilato pronto per la lettura
├── images/                 # Grafici ed asset grafici per il report ed il readme
├── videos/                 # Directory di output per le simulazioni renderizzate in MP4
├── config.yaml             # Parametri di simulazione e iperparametri RL
├── main.py                 # CLI principale per orchestrare train, test e video
├── train_ppo.py            # Routine di addestramento parallelizzata con PPO
├── evaluate_ppo.py         # Script di valutazione e rendering traiettorie
├── callbacks.py            # Log delle metriche personalizzate e checkpointing
└── requirements.txt        # Dipendenze Python del progetto
```

---

## 🚀 Istruzioni per l'Uso

### 🛠️ Setup Ambiente

1. Creare ed attivare l'ambiente virtuale:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # Per macOS/Linux
   # .venv\Scripts\activate   # Per Windows
   ```
2. Installare le dipendenze:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
   *Nota: Per abilitare il salvataggio dei video in MP4, assicurarsi di installare il pacchetto FFmpeg sul proprio sistema operativo.*

### 🧠 Addestramento dei Modelli (Training)

L'allenamento sfrutta 16 processi paralleli (configurabili) per accumulare campioni rapidamente ed utilizza il meccanismo di *Self-Play* in PettingZoo. La CLI gestisce automaticamente le versioni dei modelli salvandole progressivamente in `models/`.

```bash
# Avvia un nuovo addestramento da zero (sovrascrive checkpoint provvisori precedenti)
python main.py --train-new

# Riprende l'addestramento dell'ultimo modello salvato
python main.py --train-resume

# Avvia l'addestramento con parametri personalizzati da terminale
python main.py --train-new --steps 2000000 --n-envs 16 --model-name sailing_model
```

### 📺 Valutazione e Rendering Video (Testing)

È possibile testare le performance della policy addestrata facendola gareggiare in simulazione e registrando l'episodio in un file MP4:

```bash
# Esegue un singolo episodio di test e lo esporta come video demo
python main.py --video-file videos/sailing_demo.mp4

# Genera una suite di test composta da 5 regate differenti con seed stocastici diversi
python main.py --test-multi
```

### 📊 Monitoraggio in TensorBoard

La classe `SuccessTrackingCallback` definita in `callbacks.py` registra costantemente le metriche personalizzate di performance (rapporto completamento leg, velocità di bolina, efficienza media del trim, penalità medie per collisioni e andamento delle singole tipologie di fallimento/squalifica).

Per ispezionare l'andamento del training:
```bash
tensorboard --logdir ./sailing_tensorboard/
```

---

## 🔬 Dettagli di Configurazione

I parametri fisici ed algoritmici sono controllati centralmente nel file [config.yaml](file:///Users/francescomariafuligni/Desktop/UNI/magistrale_1_anno/1_semestre/INTELLIGENZA_ARTIFICIALE/Progetto/Multi-agent_America_Cup/config.yaml). Di seguito gli iperparametri principali:

| Iperparametro | Valore | Descrizione |
| :--- | :---: | :--- |
| `learning_rate` | $2\times 10^{-4}$ | Tasso di apprendimento per l'ottimizzatore Adam |
| `n_steps` | $4096$ | Numero di passi per ambiente prima di effettuare un aggiornamento PPO |
| `batch_size` | $1024$ | Dimensione del minibatch per il gradiente |
| `net_arch` | `[256, 256]` | Rete MLP a due strati per policy ($\pi$) e value function ($V$) |
| `frame_stack` | $4$ | Stacking dei frame per catturare l'evoluzione temporale di vento e avversario |
| `success_threshold_pct`| $0.95$ | Percentuale minima di arrivi completati richiesta per l'arresto anticipato |
| `success_window_size` | $100$ | Numero di episodi su cui calcolare la percentuale di successo |

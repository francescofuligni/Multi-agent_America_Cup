Sailing Environment – Prestart Logic
Overview

La fase di prestart nell’ambiente ImprovedSailingEnv rappresenta il periodo precedente all’inizio ufficiale della regata.
Durante questa fase le barche si preparano alla partenza, cercando la miglior posizione possibile rispetto alla linea di start, senza poter ancora iniziare la gara.

La logica è pensata per simulare una regata reale, includendo:

posizionamento tattico pre-partenza
gestione del timing
rischio di falsa partenza (OCS)
Stato del sistema

La fase di prestart è controllata da due variabili principali:

start_phase: indica se la fase di prestart è attiva
start_armed: indica se la regata è ufficialmente iniziata
Transizione di fase

La transizione avviene automaticamente dopo un numero fisso di step:

start_timer = 40

Quando:

step_count >= start_timer

allora:

start_phase = False
start_armed = True

Da questo momento la regata entra nella fase competitiva reale.

Start Line

La linea di partenza è definita come:

start_line_y = bottom_gate_y - 50

Le barche iniziano sempre sotto la linea di partenza e devono attraversarla correttamente al momento del via.

Spawn delle barche

All’inizio di ogni episodio:

entrambe le barche vengono spawnate sotto la start line
condividono una fascia verticale casuale
la posizione X è randomizzata
viene imposto un gap minimo tra le barche per evitare collisioni iniziali

Obiettivo: garantire condizioni eque e separazione iniziale.

Reward System (Prestart Phase)

Durante start_phase == True, il reward segue una logica diversa dalla regata.

1. Vicinanza alla linea di partenza

Le barche ricevono reward crescente in base alla prossimità alla start line:

più sono vicine alla linea, maggiore è il reward
2. Pressione temporale

Il reward aumenta con il passare del tempo:

simula l’urgenza di posizionarsi prima del via
3. Velocità

Viene assegnato un piccolo bonus per la velocità:

incentiva il movimento e la preparazione dinamica
4. Penalità OCS (On Course Side)

Se una barca supera la linea prima del via ufficiale:

viene applicata una forte penalità
l’episodio termina immediatamente

Condizione:

y > start_line_y + 2.0
Inizio della regata

La regata diventa attiva quando:

start_phase == False
start_armed == True

Da questo momento:

si attiva la logica completa della regata
valgono le regole di gate, roundings e finish
Partenza valida

Una partenza è considerata valida solo se:

la regata è armata (start_armed == True)
la barca attraversa la linea dal basso verso l’alto
la barca si trova all’interno del gate

In questo caso:

viene assegnato un bonus di partenza
il target viene aggiornato al primo gate
Design della prestart

La fase di prestart è progettata per:

incentivare il posizionamento tattico sulla linea
creare competizione tra agenti prima del via
evitare comportamento statico
simulare dinamiche reali di regata
Nota importante

La separazione tra prestart e race phase è netta:

la prestart usa reward shaping dedicato
la race phase usa reward basato su VMG, velocità e progressione sul percorso
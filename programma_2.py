#########################################################################################################################
# Programma 2 del progetto finale del corso di "Linguistica Computazionale (A. A. 2023/24)"                             #
# Codice scritto dallo studente Giuseppe Mangano - Università di Pisa                                                   #
#########################################################################################################################

'''Il codice sviluppato deve prendere in input un corpus, effettuare le operazioni di annotazione richieste (sentence splitting, tokenizzazione, PoS tagging),
ed estrarre le seguenti informazioni:

1. I top-50 Sostantivi, Avverbi e Aggettivi più frequenti (con relativa frequenza, ordinati per frequenza decrescente);

2. I top-20 n-grammi più frequenti (con relativa frequenza, e ordinati per frequenza decrescente)
    • Per n = [1, 2, 3, 4, 5];

3. I top 20 n-grammi di PoS più frequenti (con relativa frequenza, e ordinati per frequenza decrescente)
    • Per n = [1, 2, 3];

4. I top-10 bigrammi composti da Aggettivo e Sostantivo, ordinati per:
    a. frequenza decrescente, con relativa frequenza;
    b. probabilità condizionata massima, e relativo valore di probabilità;
    c. probabilità congiunta massima, e relativo valore di probabilità;
    d. MI (Mutual Information) massima, e relativo valore di MI;
    e. LMI (Local Mutual Information) massima, e relativo valore di LMI;
    f. Calcolare e stampare il numero di elementi comuni ai top-10 per MI e per LMI;

5. Considerate le frasi con una lunghezza compresa tra 10 e 20 token, in cui almeno la metà (considerare la parte intera della divisione per due come valore)
dei token occorre almeno 2 volte nel corpus (i.e., non è un hapax), si identifichino:
    a. La frase con la media della distribuzione di frequenza dei token più alta;
    b. La frase con la media della distribuzione di frequenza dei token più bassa;
    c. La frase con probabilità più alta secondo un modello di Markov di ordine 2 costruito a partire dal corpus di input;
    NB: la media della distribuzione di frequenza dei token è data dalla somma delle
    frequenze (nel corpus) dei token della frase diviso il numero di token della frase;

6. Estratte le Entità Nominate del testo, identificare per ciascuna classe di NE i 15 elementi più frequenti, ordinati per frequenza decrescente e con relativa frequenza.'''


import sys
import nltk
import string
import math
import copy
from collections import Counter
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords
from nltk.chunk import ne_chunk


# Funzione per leggere un file
def read_file_contents(file_path):
    with open(file_path, "r", encoding = "utf8") as infile:
        contents = infile.read()
    return contents


# Funzione per l'annotazione linguistica (sentence splitting, tokenizzazione, PoS tagging)
def annotazione_linguistica(testo):

    # Sentence splitting e numero di frasi
    frasi = nltk.tokenize.sent_tokenize(testo)
    num_frasi = len(frasi)
    
    # Tokenizzazione e numero di token
    tokens = [token for frase in frasi for token in nltk.tokenize.word_tokenize(frase)]
    num_tokens = len(tokens)
    
    # PoS tagging (Penn Treebank)
    pos_tags = nltk.tag.pos_tag(tokens)
    
    return frasi, num_frasi, tokens, num_tokens, pos_tags


# Funzione per rimuovere la punteggiatura
def rimuovi_punteggiatura(tokens):

    # Rimuovo la punteggiatura e filtro parole vuote
    return [token for token in tokens if token.translate(str.maketrans('', '', string.punctuation)) and token not in string.punctuation]


# Funzione per rimuovere le stop words
def rimuovi_stopwords(tokens):

    # Creo un set dove inserisco le stop words dell'inglese
    stop_words = set(stopwords.words('english'))

    # Filtro i tokens con il set creato prima
    tokens_filtrati = [token for token in tokens if token.lower() not in stop_words]

    # Restituisco i tokens filtrati
    return tokens_filtrati

# Funzione per normalizzare e lemmatizzare le parole
def lemmatizzatore(tokens):

    # Inizializzo il lemmatizzatore
    lemmatizer = WordNetLemmatizer()

    # Creo una lista dove inserisco i tokens normalizzati
    tokens_filtrati = []
    for token in tokens:

        # Converto in minuscolo e lemmatizzo
        token = token.lower()
        token = lemmatizer.lemmatize(token)
        tokens_filtrati.append(token)

    # Restituisco i tokens normalizzati
    return tokens_filtrati


# Funzione che converte i tag da Treebank a WordNet
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


# Funzione per elencare i primi 50 sostantivi, avverbi, aggettivi più frequenti
def piu_frequenti(pos_tags, pos_ricercata):

    # Creo una lista con la PoS ricercata
    filtro_pos = [token for token, pos in pos_tags if pos.startswith(pos_ricercata)]

    # Effettuo la distribuzione delle frequenze
    lista_pos_ordinata = nltk.FreqDist(filtro_pos)

    # Restituisco la distribuzione delle frequenze delle PoS ordinata
    return lista_pos_ordinata


# Funzione per la top-20 n-grammi più frequenti
def top20_ngrammi(tokens, n):
    
    # Creo una lista con gli n-grammi che ci interessano
    ngrammi = list(nltk.ngrams(tokens, n))
    ngrammi_ordinati = nltk.FreqDist(ngrammi)

    # Restituisco la distribuzione degli n-grammi ordinati
    return ngrammi_ordinati


# Funzione per la top 20 n-grammi di PoS più frequenti
def top20_ngrammi_pos(pos_tags, n):

    # Creo una lista con gli n-grammi di PoS che ci interessano
    ngrammi_pos = list(nltk.ngrams(pos_tags, n))
    ngrammi_pos_ordinati = nltk.FreqDist(ngrammi_pos)

    # Restituisco la distribuzione degli n-grammi di PoS ordinati
    return ngrammi_pos_ordinati


# Funzione per l'estrazione dei bigrammi specifici Aggettivo (J) e Sostantivo (N)
def estrai_bigrammi_AS(pos_tags):

    # Creo una lista di tuple di bigrammi dai PoS
    bigrammi = list(nltk.bigrams(pos_tags))

    # Creo una lista di tuple dove inserisco i bigrammi PoS specifici Aggettivo (J) e Sostantivo (N)
    bigrammi_AS_pos = [elem for elem in bigrammi if elem[0][1].startswith('J') and elem[1][1].startswith('N')]
   
    # Creo una lista di tuple dove inserisco soltanto i token dei bigrammi
    bigrammi_AS = [(aggettivo[0], sostantivo[0]) for (aggettivo, sostantivo) in bigrammi_AS_pos]

    # Restituisco la lista dei bigrammi Aggettivo - Sostantivo
    return bigrammi_AS


# Funzione per la top 10 bigrammi Aggettivo e Sostantivo più frequenti
def top10_bigrammi_AS(bigrammi_AS):
    bigrammi_ordinati = nltk.FreqDist(bigrammi_AS)

    # Restituisco la distribuzione dei bigrammi Aggettivo - Sostantivo più frequenti
    return bigrammi_ordinati



def max_prob_cond(tokens, bigrammi, n):

    # Creo un Counter per i bigrammi e per i token
    freq_bigrammi = Counter(bigrammi)
    freq_token = Counter(tokens)

    # Creo una lista per i bigrammi con la massima probabilità condizionata
    lista_prob_cond = []

    for bigramma in freq_bigrammi:
        tk1, tk2 = bigramma

        # Ottengo le frequenze del bigramma e del primo elemento
        freq_bigramma = freq_bigrammi[bigramma]
        freq_primo_el = freq_token[tk1]

        # Controllo che la frequenza del primo elemento sia maggiore di 0
        if freq_primo_el > 0:

            # Calcolo la probabilità condizionata
            prob_cond = freq_bigramma / freq_primo_el
            lista_prob_cond.append((bigramma, prob_cond))

    # Ordino e restituisco i primi n bigrammi in ordine decrescente di probabilità condizionata
    lista_prob_cond.sort(key = lambda x: x[1], reverse = True)
    return lista_prob_cond[:n]


def max_prob_cong(tokens, bigrammi, n):

    # Creo un Counter per i bigrammi e per i token
    freq_bigrammi = Counter(bigrammi)
    freq_token = Counter(tokens)

    # Creo una lista per i bigrammi con la massima probabilità congiunta
    lista_prob_cong = []

    for bigramma in freq_bigrammi:
        tk1, tk2 = bigramma

        # Ottengo le frequenze del bigramma e del primo elemento
        freq_bigramma = freq_bigrammi[bigramma]
        freq_primo_el = freq_token[tk1]

        # Calcolo la probabilità congiunta
        prob_cond = freq_bigramma / freq_primo_el
        prob_primo_el = freq_primo_el / len(tokens)
        prob_cong = prob_cond * prob_primo_el
        lista_prob_cong.append((bigramma, prob_cong))

    # Ordino e restituisco i primi n bigrammi in ordine decrescente di probabilità congiunta
    lista_prob_cong.sort(key = lambda x: x[1], reverse = True)
    return lista_prob_cong[:n]

# Funzione per calcolare la massima Mutual Information (MI)
def max_MI(tokens, bigrammi, n):
    
    # Creo una lista dove inserisco i valori di MI
    lista_valori_MI = []
    
    # Calcolo la frequenza dei bigrammi e dei token nel corpus
    freq_bigrammi = Counter(bigrammi)
    freq_tokens = Counter(tokens)
    dim_corpus = len(tokens)

    # Definisco una funzione interna per il calcolo della MI
    def mutua_informazione(bigramma, freq_tokens, freq_bigrammi, dim_corpus):
        freq_bigramma = freq_bigrammi[bigramma]
        freq_t1 = freq_tokens[bigramma[0]]
        freq_t2 = freq_tokens[bigramma[1]]

        # Evito divisioni per zero
        if freq_t1 == 0 or freq_t2 == 0:
            return 0
        
        # Formula della MI
        mi = math.log((freq_bigramma * dim_corpus) / (freq_t1 * freq_t2), 2)
        return mi
    
    # Calcolo i valori di MI per ciascun bigramma e li aggiungo alla lista
    for bigramma in freq_bigrammi:
        mi = mutua_informazione(bigramma, freq_tokens, freq_bigrammi, dim_corpus)
        lista_valori_MI.append((bigramma, mi))

    # Restituisco i primi n bigrammi con relativa MI in ordine decrescente
    lista_valori_MI.sort(key = lambda x: x[1], reverse = True)
    return lista_valori_MI[:n]


# Funzione per calcolare la massima Local Mutual Information (LMI)
def max_LMI(tokens, bigrammi, n):

    # Creo una lista dove inserisco i valori di LMI
    lista_valori_LMI = []

    # Calcolo la frequenza dei bigrammi e dei token nel corpus
    freq_bigrammi = Counter(bigrammi)
    freq_tokens = Counter(tokens)
    dim_corpus = len(tokens)

    # Definisco una funzione interna per il calcolo della LMI
    def mutua_informazione_locale(bigramma, freq_tokens, freq_bigrammi, dim_corpus):
        freq_bigramma = freq_bigrammi[bigramma]
        freq_t1 = freq_tokens[bigramma[0]]
        freq_t2 = freq_tokens[bigramma[1]]
        
        # Evito divisioni per zero
        if freq_t1 == 0 or freq_t2 == 0:
            return 0
        
        # Formula della MI e della LMI
        mi = math.log((freq_bigramma * dim_corpus) / (freq_t1 * freq_t2), 2)
        lmi = freq_bigramma * mi
        return lmi
    
    # Inserisco i valori di LMI nella lista
    for bigramma in freq_bigrammi:
        lmi = mutua_informazione_locale(bigramma, freq_tokens, freq_bigrammi, dim_corpus)
        lista_valori_LMI.append((bigramma, lmi))

    # Restituisco i primi n bigrammi con relativa LMI in ordine decrescente
    lista_valori_LMI.sort(key = lambda x: x[1], reverse = True)    
    return lista_valori_LMI[:n]


# Funzione per calcolare e stampare il numero di elementi comuni ai top-10 per MI e per LMI
def top_elementi_comuni(lista_valori_MI, lista_valori_LMI, n):

    # Prendo 'elemento[0]' che è il bigramma dalla tupla (bigramma, valore)
    top_MI = [elemento[0] for elemento in lista_valori_MI[:n]]
    top_LMI = [elemento[0] for elemento in lista_valori_LMI[:n]]
    
    # Converto le liste in set e trovo l'intersezione degli elementi comuni tra i due set
    elementi_comuni = set(top_MI).intersection(set(top_LMI))
    
    # Restituisco l'insieme degli elementi comuni
    return elementi_comuni


# Funzione per estrarre le frasi di lunghezza compresa tra 10 e 20 tokens
def filtro_frasi(frasi, tokens):

    # Creo una lista per le frasi che rispecchiano la condizione di lunghezza compresa tra 10 e 20 tokens
    frasi_filtrate = []

    for frase in frasi:
        tokens_frase = nltk.tokenize.word_tokenize(frase)
        lung_frase = len(tokens_frase)
        
        # Verifico che la lunghezza della frase sia compresa tra 10 e 20        
        if 10 <= lung_frase <= 20:

            # Controllo che almeno la metà dei token nella frase non sia hapax
            non_hapax = sum(1 for token in tokens_frase if tokens.count(token) > 1)
            if non_hapax >= lung_frase // 2:
                frasi_filtrate.append(frase)

    # Restituisco le frasi filtrate
    return frasi_filtrate

# Funzione per il calcolo della frase con la media della distribuzione di frequenza dei token più alta
def frase_con_media_distribuzione_massima(frasi_filtrate, tokens):

    # Creo una variabile per memorizzare la massima distribuzione di frequenza
    massima_distribuzione = 0
    
    # Creo una stringa vuota dove poi memorizzo la frase con la massima distribuzione
    soluzione = ""
    
    # Itero su ciascuna frase
    for frase in frasi_filtrate:
        token_frase = nltk.tokenize.word_tokenize(frase)
        numero_token = len(token_frase)
        
        # Calcolo la somma delle frequenze dei token della frase nel corpus
        somma_frequenze = sum(tokens.count(token) for token in token_frase)
        
        # Calcolo la distribuzione di frequenza media dei token della frase
        distribuzione_ottenuta = somma_frequenze / numero_token
        
        # Verifico se la distribuzione ottenuta è maggiore della massima distribuzione attuale e in caso la aggiorno
        if distribuzione_ottenuta > massima_distribuzione:
            massima_distribuzione = distribuzione_ottenuta
            soluzione = frase
    
    # Restituisco la frase con la massima distribuzione di frequenza dei token
    return soluzione


# Funzione per il calcolo della frase con la media della distribuzione di frequenza dei token più bassa
def frase_con_media_distribuzione_minima(frasi_filtrate, tokens):
    
    # Creo una variabile per memorizzare la minima distribuzione di frequenza (serve un valore molto grande)
    minima_distribuzione = float('inf')

    # Creo una stringa vuota dove poi memorizzo la frase con la minima distribuzione
    soluzione = ""
    for frase in frasi_filtrate:
        token_frase = nltk.tokenize.word_tokenize(frase)
        numero_token = len(token_frase)

        # Calcolo la somma delle frequenze dei token della frase nel corpus
        somma_frequenze = sum(tokens.count(token) for token in token_frase)

        # Calcolo la distribuzione di frequenza media dei token della frase
        distribuzione_ottenuta = somma_frequenze / numero_token
        if distribuzione_ottenuta < minima_distribuzione:
            minima_distribuzione = distribuzione_ottenuta
            soluzione = frase

    # Restituisco la frase con la minima distribuzione di frequenza dei token
    return soluzione


# Funzione per il calcolo della frase con probabilità più alta secondo un modello di Markov di ordine 2
def max_prob_markov2(frasi_filtrate, corpus):
    
    # Creo una variabile per memorizzare la probabilità massima
    max_prob = 0

    # Creo una stringa vuota dove poi memorizzo la frase con la probabilità massima
    soluzione = ""

    # Tokenizzo l'intero corpus
    tokens = nltk.word_tokenize(corpus)
    
    # Calcolo delle frequenze dei token, bigrammi e trigrammi nel corpus e dimensione di quest'ultimo
    freq_tokens = nltk.FreqDist(tokens)
    freq_bigrammi = nltk.FreqDist(nltk.bigrams(tokens))
    freq_trigrammi = nltk.FreqDist(nltk.trigrams(tokens))
    dim_corpus = len(tokens)
    
    # Itero su ciascuna frase
    for frase in frasi_filtrate:
        tokens_frase = nltk.word_tokenize(frase)

        # Inizializzo la probabilità della frase ad 1 perché è un valore neutro
        prob_frase = 1
        
        # Itero su ciascun token della frase
        for i in range(len(tokens_frase)):

            # Primo token: P(token1) = frequenza(token1) / dimensione del corpus
            if i == 0:
                prob = freq_tokens[tokens_frase[i]] / dim_corpus
                
            # Secondo token: P(token2|token1) = frequenza(bigramma) / frequenza(token1)
            elif i == 1:
                bigrammma = (tokens_frase[i - 1], tokens_frase[i])
                if freq_tokens[tokens_frase[i - 1]] > 0:
                    prob = freq_bigrammi[bigrammma] / freq_tokens[tokens_frase[i - 1]]
                else:
                    prob = 0

            # Terzo token e successivi: P(token3|token1, token2) = frequenza(trigramma) / frequenza(bigramma)
            else:
                trigramma = (tokens_frase[i - 2], tokens_frase[i - 1], tokens_frase[i])
                bigrammma = (tokens_frase[i - 2], tokens_frase[i - 1])
                if freq_bigrammi[bigrammma] > 0:
                    prob = freq_trigrammi[trigramma] / freq_bigrammi[bigrammma]
                else:
                    prob = 0

            # Aggiorno la probabilità della frase moltiplicando per la probabilità calcolata
            prob_frase *= prob
        
        # Aggiorno la frase con la probabilità massima se quella corrente è maggiore
        if prob_frase > max_prob:
            max_prob = prob_frase
            soluzione = frase
    
    # Restituisco la frase con la probabilità più alta
    return soluzione

# Funzione per l'estrazione delle Entità Nominate
def entita_nominate(pos_tags):

    # Costruisco l'albero delle Entità Nominate
    albero_ne = nltk.ne_chunk(pos_tags)
    
    # Creo un dizionario per memorizzare le entità nominate con i loro tipi e frequenze
    diz_ne = {}
    
    # Itero sull'albero delle Entità Nominate
    for nodo in albero_ne:

        # Verifico che il nodo sia è una Named Entity
        if hasattr(nodo, 'label'):
            tipo_entita = nodo.label()

            # Salvo in una lista tutti gli elementi di nodo.leaves()
            entita = " ".join([token for token, POS in nodo.leaves()])
            
            # Aggiungo il tipo di entità al dizionario se non è già presente
            if tipo_entita not in diz_ne:
                diz_ne[tipo_entita] = Counter()
            
            # Aggiorno il conteggio dell'entità nel dizionario
            diz_ne[tipo_entita][entita] += 1
    
    # Creo un dizionario per memorizzare le 15 entità più frequenti per ogni tipo
    top15_ne = {}
    
    # Itero sui tipi di entità ed entità corrispondenti in diz_ne
    for tipo_entita, entities in diz_ne.items():

        # Estraggo le 15 entità più frequenti per ogni tipo e le aggiungo al dizionario
        top15_ne[tipo_entita] = entities.most_common(15)
    
    # Restituisco il dizionario con le entità più frequenti
    return top15_ne

def main(file):

    # Scrivo l'output in un file
    with open("output_programma2.txt", "w") as outfile:
        sys.stdout = outfile

        # Leggo i contenuti "grezzi" del file
        testo = read_file_contents(file)
        
        # Effettuo l'annotazione linguistica dei corpus
        frasi, num_frasi, tokens, num_tokens, pos_tags = annotazione_linguistica(testo)
        
        # Rimuovo la punteggiatura e le stop words per scremare il rumore del testo
        tokens_puliti = rimuovi_punteggiatura(tokens)
        tokens_puliti = rimuovi_stopwords(tokens_puliti)
        tokens_puliti = lemmatizzatore(tokens_puliti)
        
        # Filtro le eventuali parole vuote dopo la scrematura del testo
        tokens_puliti = [token for token in tokens_puliti if token]


        # Copio i PoS tags in un'altra variabile per le named entities
        pos_tags_grezzi = copy.deepcopy(pos_tags)

        # Effettuo di nuovo il PoS tagging del testo scremato
        pos_tags = nltk.tag.pos_tag(tokens_puliti)


        # Punto nr. 1
        # I top-50 Sostantivi, Avverbi e Aggettivi più frequenti (con relativa frequenza, ordinati per frequenza decrescente)
        sostantivi_frequenti = piu_frequenti(pos_tags, 'N')
        avverbi_frequenti = piu_frequenti(pos_tags, 'R')
        aggettivi_frequenti = piu_frequenti(pos_tags, 'J')
        
        print("Top-50 Sostantivi più frequenti:")
        for sostantivo, frequenza in sostantivi_frequenti.most_common(50):
            print(f"{sostantivo}: {frequenza}")
        
        print("\nTop-50 Avverbi più frequenti:")
        for avverbio, frequenza in avverbi_frequenti.most_common(50):
            print(f"{avverbio}: {frequenza}")
        
        print("\nTop-50 Aggettivi più frequenti:")
        for aggettivo, frequenza in aggettivi_frequenti.most_common(50):
            print(f"{aggettivo}: {frequenza}")
        

        # Punto nr. 2
        # I top-20 n-grammi più frequenti (con relativa frequenza, e ordinati per frequenza decrescente). Per n = [1, 2, 3, 4, 5]
        for n in range(1, 6):
            ngrammi_frequenti = top20_ngrammi(tokens_puliti, n)
            print(f"\nTop-20 {n}-grammi più frequenti:")
            for ngramma, frequenza in ngrammi_frequenti.most_common(20):
                print(f"{ngramma}: {frequenza}")
        

        # Punto nr. 3
        # I top 20 n-grammi di PoS più frequenti (con relativa frequenza, e ordinati per frequenza decrescente). Per n = [1, 2, 3]
        for n in range(1, 4):
            ngrammi_pos_frequenti = top20_ngrammi_pos(pos_tags, n)
            print(f"\nTop-20 {n}-grammi di PoS più frequenti:")
            for ngramma_pos, frequenza in ngrammi_pos_frequenti.most_common(20):
                print(f"{ngramma_pos}: {frequenza}")
        

        # Punto nr. 4
        # I top-10 bigrammi composti da Aggettivo e Sostantivo
        bigrammi_AS = estrai_bigrammi_AS(pos_tags)
        

        # Punto nr. 4a   
        # I top-10 bigrammi composti da Aggettivo e Sostantivo, ordinati per frequenza decrescente, con relativa frequenza
        top_10_AS_freq = top10_bigrammi_AS(bigrammi_AS)
        print("\nTop-10 bigrammi Aggettivo - Sostantivo per frequenza:")
        for bigramma, frequenza in top_10_AS_freq.most_common(10):
            print(f"{bigramma}: {frequenza}")

        
        # Punto nr. 4b
        # I top-10 bigrammi composti da Aggettivo e Sostantivo, ordinati per probabilità condizionata massima, e relativo valore di probabilità
        top_10_prob_cond = max_prob_cond(tokens_puliti, bigrammi_AS, 10)
        print("\nTop-10 bigrammi Aggettivo - Sostantivo per probabilità condizionata:")
        for bigramma, prob in top_10_prob_cond:
            print(f"{bigramma}: {prob}")


        # Punto nr. 4c
        # I top-10 bigrammi composti da Aggettivo e Sostantivo, ordinati per probabilità congiunta massima, e relativo valore di probabilità
        top_10_prob_cong = max_prob_cong(tokens_puliti, bigrammi_AS, 10)
        print("\nTop-10 bigrammi Aggettivo - Sostantivo per probabilità congiunta:")
        for bigramma, prob in top_10_prob_cong:
            print(f"{bigramma}: {prob}")
        

        # Punto nr. 4d
        # I top-10 bigrammi composti da Aggettivo e Sostantivo, ordinati per MI (Mutual Information) massima, e relativo valore di MI
        top_10_MI = max_MI(tokens_puliti, bigrammi_AS, 10)
        print("\nTop-10 bigrammi Aggettivo-Sostantivo per MI:")
        for bigramma, mi in top_10_MI:
            print(f"{bigramma}: {mi}")


        # Punto nr. 4e
        # I top-10 bigrammi composti da Aggettivo e Sostantivo, ordinati per LMI (Local Mutual Information) massima, e relativo valore di LMI
        top_10_LMI = max_LMI(tokens_puliti, bigrammi_AS, 10)
        print("\nTop-10 bigrammi Aggettivo - Sostantivo per LMI:")
        for bigramma, lmi in top_10_LMI:
            print(f"{bigramma}: {lmi}")
        

        # Punto nr. 4f
        # Calcolare e stampare il numero di elementi comuni ai top-10 per MI e per LMI
        comuni_MI_LMI = top_elementi_comuni(top_10_MI, top_10_LMI, 10)
        print("\nNumero di elementi comuni tra top-10 MI e top-10 LMI:")
        print(len(comuni_MI_LMI))


        # Punto nr. 5
        # Considerate le frasi con una lunghezza compresa tra 10 e 20 token, in cui almeno la metà (considerare la parte intera della divisione per due come valore) dei token occorre almeno 2 volte nel corpus (i.e., non è un hapax)
        # N.B = con i token filtrati e normalizzati e le seguenti funzioni del punto nr.5 davano problemi quindi ho preferito utilizzare i token "grezzi"
        frasi_filtrate = filtro_frasi(frasi, tokens)
        

        # Punto nr. 5a
        # Si identifichino la frase con media della distribuzione di frequenza dei token più alta
        frase_max_media = frase_con_media_distribuzione_massima(frasi_filtrate, tokens)
        print("\nFrase con la media della distribuzione di frequenza dei token più alta:")
        print(frase_max_media)


        # Punto nr. 5b
        # Si identifichino la frase con media della distribuzione di frequenza dei token più bassa
        frase_min_media = frase_con_media_distribuzione_minima(frasi_filtrate, tokens)
        print("\nFrase con la media della distribuzione di frequenza dei token più bassa:")
        print(frase_min_media)


        # Punto nr. 5c
        # Si identifichino la frase con probabilità più alta secondo un modello di Markov di ordine 2
        frase_max_prob_markov = max_prob_markov2(frasi_filtrate, testo)
        print("\nFrase con probabilità più alta secondo un modello di Markov di ordine 2:")
        print(frase_max_prob_markov)


        # Punto nr.6
        # Estratte le Entità Nominate del testo, identificare per ciascuna classe di NE i 15 elementi più frequenti, ordinati per frequenza decrescente e con relativa frequenza.
        # N.B = con i PoS tags filtrati e normalizzati la funzione di Named Entity recognition del punto nr.6 dava problemi quindi ho preferito utilizzare i PoS "grezzi"

        ne_frequenti = entita_nominate(pos_tags_grezzi)    
        for tipo_entita, entities in ne_frequenti.items():
            print(f"\n{tipo_entita}:")
            for entita, freq in entities:
                print(f"{entita}: {freq}")

if __name__ == '__main__':
    main(sys.argv[1])
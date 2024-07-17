#########################################################################################################################
# Programma 1 del progetto finale del corso di "Linguistica Computazionale (A. A. 2023/24)"                             #
# Codice scritto dallo studente Giuseppe Mangano - Università di Pisa                                                   #
#########################################################################################################################  

'''Il codice sviluppato deve prendere in input i due corpora, effettuare le operazioni di annotazione linguistica richieste 
(sentence splitting, tokenizzazione, PoS tagging, lemmatizzazione), e produrre un confronto dei corpora rispetto a:

1. Numero di frasi e token;
2. Lunghezza media delle frasi in token e lunghezza media dei token, a eccezione della punteggiatura, in caratteri;
3. Numero di Hapax tra i primi 500, 1000, 3000 token, e nell'intero corpus;
4. Dimensione del vocabolario e ricchezza lessicale (Type-Token Ratio, TTR), calcolata per porzioni incrementali di 200 token fino ad arrivare a tutto il testo
    • i.e., i primi 200, i primi 400, i primi 600, ..., per tutti token;
5. Numero di lemmi distinti (i.e., la dimensione del vocabolario dei lemmi);
6. Distribuzione di frasi con polarità positiva e negativa
    • Per classificare le frasi in POS e NEG è possibile utilizzare il classificatore di polarità
    visto a lezione (Notebook) o un classificatore costruito ad hoc.'''


import sys
import nltk
import string
from collections import Counter
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

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
    tokens = [] 
    for frase in frasi:
        token_frase = nltk.tokenize.word_tokenize(frase)
        tokens += token_frase
    num_tokens = len(tokens)
    
    # PoS tagging (Penn Treebank)
    pos_tags = nltk.tag.pos_tag(tokens)

    return frasi, num_frasi, tokens, num_tokens, pos_tags


# Funzione per calcolare la lunghezza media delle frasi
def lunghezza_frasi(frasi):

    # Inizializzo un contatore a 0 che verrà utilizzato per accumulare la somma delle lunghezze delle frasi
    contatore = 0
    
    for frase in frasi:
        tokens_frase = nltk.tokenize.word_tokenize(frase)
        
        # Aggiungo il numero di parole al contatore
        contatore += len(tokens_frase)
    
    # Calcolo la lunghezza media delle frasi dividendo il contatore per il numero di frasi
    lunghezza_media_f = contatore / len(frasi)
    
    # Restituisco il valore di lunghezza media delle frasi
    return lunghezza_media_f


# Funzione per calcolare la lunghezza media dei token (escludendo la punteggiatura)
def lunghezza_token(tokens):

    # Inizializzo una lista dove filtro i token rimuovendo quelli che appartengono alla punteggiatura
    tokens_no_punt = [token for token in tokens if token.translate(str.maketrans('', '', string.punctuation)) and token not in string.punctuation]

    # Calcolo la somma delle lunghezze dei token filtrati
    somma_lunghezze = sum(len(token) for token in tokens_no_punt)

    # Calcolo la lunghezza media dei token
    lunghezza_media_t = somma_lunghezze / len(tokens_no_punt)

    # Restituisco il valore di lunghezza media dei token
    return lunghezza_media_t


# Funzione per contare gli Hapax Legomena fino a un certo limite n
def contatore_hapax(tokens, n):

    # Creo una lista che contiene i primi n token della lista "tokens"
    sezione_token = tokens[:n]
    
    # Tramite counter, conto le occorrenze di ciascun token
    contatore = Counter(sezione_token)

    # Creo una variabile in cui sommo il totale degli Hapax
    num_hapax = sum(1 for tot in contatore.values() if tot == 1)

    # Restituisco il numero complessivo degli Hapax
    return num_hapax


# Funzione per calcolare la dimensione del vocabolario e la Type Token Ratio (TTR)
def dimensioneVoc_TTR (tokens, n):

    # Creo una lista con i types all'interno dei primi n tokens
    types = list(set(tokens[:n]))

    # Creo una variabile vocabolario in inserisco il numero dei types
    vocabolario = len(types)

    # Applicando la formula della TTR, inserisco il risultato in una variabile
    ttr = vocabolario/n

    # Restituisco la dimensione del vocabolario e il valore della TTR
    return vocabolario, ttr


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


# Funzione per contare il numero dei lemmi nel vocabolario
def contatore_lemmi(pos_tags):
  
    # Creo un'istanza di WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()

    # Lista in cui inserisco il risultato della lemmatizzazione
    output_lemmatizzati = [lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in pos_tags]

    # Tramite l'utilizzo di un set elimino i duplicati e ritrasformo in lista
    voc_lemmi = list(set(output_lemmatizzati))

    # Restituisco il numero dei lemmi nel vocabolario
    return len(voc_lemmi)


# Funzione per analizzare la polarità delle frasi. (ndr) Ho scelto VADER perché l'ho trovato sfogliando la documentazione di NLTK
def analisi_polarita(frasi):

    # Creo un'istanza di SentimentIntensityAnalyzer
    sid = SentimentIntensityAnalyzer()
    
    # Creo delle variabili che memorizzano il numero di frasi positive, negative e neutre
    pos = 0
    neg = 0
    neu = 0
    
    # Eseguo l'operazione di Sentiment Analysis per ogni frase
    for frase in frasi:
        punteggio = sid.polarity_scores(frase)
        if punteggio['compound'] >= 0.05:
            pos += 1
        elif punteggio['compound'] <= -0.05:
            neg += 1
        else:
            neu += 1
    
    # Restituisco i risultati dell'analisi
    return pos, neg, neu



# Main che svolge la consegna del primo programma
def main(file1, file2):

    # Scrivo l'output in un file
    with open("output_programma1.txt", "w") as outfile:
        sys.stdout = outfile
        
        # Leggo i contenuti "grezzi" dei file
        contenuti_f1 = read_file_contents(file1)
        contenuti_f2 = read_file_contents(file2)

        # Punto nr. 1
        # Effettuo l'annotazione linguistica dei corpus
        frasi_1, num_frasi_1, tokens_1, num_tokens_1, pos_tags_1 = annotazione_linguistica(contenuti_f1)
        frasi_2, num_frasi_2, tokens_2, num_tokens_2, pos_tags_2 = annotazione_linguistica(contenuti_f2)

        print(f"Nel primo testo ci sono {num_frasi_1} frasi e {num_tokens_1} tokens.")
        print(f"Nel secondo testo ci sono {num_frasi_2} frasi e {num_tokens_2} tokens.")
        print(f"I testi differiscono di {abs(num_frasi_1 - num_frasi_2)} frasi e di {abs(num_tokens_1 - num_tokens_2)} tokens.")

        print("\n")
        # Punto nr. 2
        # Calcolo la lunghezza media delle frasi e dei tokens
        lung_media_frasi_1 = lunghezza_frasi(frasi_1)
        lung_media_token_1 = lunghezza_token(tokens_1)

        lung_media_frasi_2 = lunghezza_frasi(frasi_2)
        lung_media_token_2 = lunghezza_token(tokens_2)

        print(f"Nel primo testo la lunghezza media delle frasi è di {lung_media_frasi_1:.2f} tokens, mentre la lunghezza media dei token è di {lung_media_token_1:.2f} caratteri.")
        print(f"Nel secondo testo la lunghezza media delle frasi è di {lung_media_frasi_2:.2f} tokens, mentre la lunghezza media dei token è di {lung_media_token_2:.2f} caratteri.")

        if lung_media_frasi_1 > lung_media_frasi_2:
            print("Il primo testo ha la lunghezza media delle frasi più lunga.")
        elif lung_media_frasi_1 < lung_media_frasi_2:
            print("Il secondo testo ha la lunghezza media delle frasi più lunga.")
        else:
            print("Entrambi i testi hanno la stessa lunghezza media delle frasi.")

        if lung_media_token_1 > lung_media_token_2:
            print("Il primo testo ha la lunghezza media dei token più lunga.")
        elif lung_media_token_1 < lung_media_token_2:
            print("Il secondo testo ha la lunghezza media dei token più lunga.")
        else:
            print("Entrambi i testi hanno la stessa lunghezza media dei token.")
        
        print("\n")
        # Punto nr. 3
        # Calcolo Il numero degli Hapax Legomena tra i primi 500, 1000, 3000 token, e nell'intero corpus (5000)
        print("Corpus 1:")
        print(f"Il numero degli Hapax nei primi 500 token è di {contatore_hapax(tokens_1, 500)}.")
        print(f"Il numero degli Hapax nei primi 1000 token è di {contatore_hapax(tokens_1, 1000)}.")
        print(f"Il numero degli Hapax nei primi 3000 token è di {contatore_hapax(tokens_1, 3000)}.")
        print(f"Il numero degli Hapax in tutto corpus è di {contatore_hapax(tokens_1, len(tokens_1))}.")
        print("\n")
        print("Corpus 2:")
        print(f"Il numero degli Hapax nei primi 500 token è di {contatore_hapax(tokens_2, 500)}.")
        print(f"Il numero degli Hapax nei primi 1000 token è di {contatore_hapax(tokens_2, 1000)}.")
        print(f"Il numero degli Hapax nei primi 3000 token è di {contatore_hapax(tokens_2, 3000)}.")
        print(f"Il numero degli Hapax in tutto il corpus è di {contatore_hapax(tokens_2, len(tokens_2))}.")
        
        print("\n")
        # Punto nr. 4
        # Calcolo la dimensione del vocabolario e della TTR per porzioni incrementali di 200 token
        limite = 200
        print("Corpus 1:")
        while limite <= len(tokens_1):
            vocabolario, ttr = dimensioneVoc_TTR(tokens_1, limite)
            print(f"Il vocabolario nei primi {limite} token è {vocabolario} e la TTR è {ttr:.2f}.")
            limite += 200

        vocabolario, ttr = dimensioneVoc_TTR(tokens_1, len(tokens_1))
        print(f"Il vocabolario di tutto il corpus è {vocabolario} e la TTR è {ttr}.")
        print("\n")

        print("Corpus 2:")
        limite = 200
        while limite <= len(tokens_2):
            vocabolario, ttr = dimensioneVoc_TTR(tokens_2, limite)
            print(f"Il vocabolario nei primi {limite} token è {vocabolario} e la TTR è {ttr:.2f}.")
            limite += 200

        vocabolario, ttr = dimensioneVoc_TTR(tokens_2, len(tokens_2))
        print(f"Il vocabolario di tutto il corpus è {vocabolario} e la TTR è {ttr}")

        print("\n")
        # Punto nr. 5
        # Numero di lemmi distinti
        print(f"Il numero dei lemmi del corpus 1 è di {contatore_lemmi(pos_tags_1)}.")
        print(f"Il numero dei lemmi del corpus 2 è di {contatore_lemmi(pos_tags_2)}.")


        print("\n")
        # Punto nr. 6
        # Distribuzione di frasi con polarità positiva e negativa 
        pos_1, neg_1, neu_1 = analisi_polarita(frasi_1)
        pos_2, neg_2, neu_2 = analisi_polarita(frasi_2)

        print("Distribuzione delle polarità del Corpus 1:")
        print(f"La percentuale delle frasi con polarità positiva è del {((pos_1/num_frasi_1) * 100):.2f}%.")
        print(f"La percentuale delle frasi con polarità negativa è del {((neg_1/num_frasi_1) * 100):.2f}%.")
        print(f"La percentuale delle frasi con polarità neutra è del {((neu_1/num_frasi_2) * 100):.2f}%.")

        print("\n")
        print("Distribuzione delle polarità del Corpus 2:")
        print(f"La percentuale delle frasi con polarità positiva è del {((pos_2/num_frasi_2) * 100):.2f}%.")
        print(f"La percentuale delle frasi con polarità negativa è del {((neg_2/num_frasi_2) * 100):.2f}%.")
        print(f"La percentuale delle frasi con polarità neutra è del {((neu_2/num_frasi_2) * 100):.2f}%.")

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
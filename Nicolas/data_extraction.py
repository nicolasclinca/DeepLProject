import json
from SPARQLWrapper import SPARQLWrapper, JSON
import wikipediaapi
from tqdm import tqdm
import time



def get_movie_uris(limit=1000):
    """Query DBpedia for a list of movie URIs."""
    sparql = SPARQLWrapper("https://dbpedia.org/sparql")
    sparql.setQuery(f"""
        PREFIX dbo: <http://dbpedia.org/ontology/>
        SELECT ?film WHERE {{
            ?film a dbo:Film .
        }} LIMIT {limit}
    """)
    sparql.setReturnFormat(JSON)     # di default, dovrebbe ritornare un XML, noi lo settiamo a JSON
    results = sparql.query().convert() # convertiamo il json in un dizionario
    return [result["film"]["value"] for result in results["results"]["bindings"]]



def get_triples_for_movie(movie_uri):
    """Get all one-hop triples for a given movie URI."""
    sparql = SPARQLWrapper("https://dbpedia.org/sparql")
    # Filtra per predicati comuni per semplificare (come suggerito dalla traccia)
    common_predicates = [
        "dbo:director", "dbo:starring", "dbo:producer", "dbo:writer",
        "dbo:musicComposer", "dbo:country", "dbo:language", "dbo:runtime"
    ]
    predicate_filter = "FILTER(?p IN (" + ", ".join(common_predicates) + "))"

    query = f"""
        PREFIX dbr: <http://dbpedia.org/resource/>
        PREFIX dbo: <http://dbpedia.org/ontology/>
        SELECT ?p ?o WHERE {{
            <{movie_uri}> ?p ?o .
            {predicate_filter}
        }}
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    triples = []
    try:
        results = sparql.query().convert()
        # cicliamo per ogni predicato (un predicato -> un oggetto)
        for result in results["results"]["bindings"]:
            triples.append({
                "subject": movie_uri.split('/')[-1],
                "predicate": result["p"]["value"].split('/')[-1],
                "object": result["o"]["value"].split('/')[-1]
            })
    except Exception as e:
        print(f"Error querying triples for {movie_uri}: {e}")
    return triples



def get_wikipedia_abstract(movie_title):
    """Get the first paragraph of the Wikipedia page for a movie."""
    wiki_abstract = wikipediaapi.Wikipedia('NanoSocrates', 'en')
    page = wiki_abstract.page(movie_title)
    if not page.exists():
        return None
    # Prende solo il primo paragrafo, come suggerito
    return page.summary.split('\n')[0]



def main():
    print("1. Fetching movie URIs from DBpedia...")
    # Per un test rapido, usa un limite basso, es. 50
    movie_uris = get_movie_uris(limit=5) 
    
    wiki_data = []
    
    print(f"2. Fetching details for {len(movie_uris)} movies...")
    for uri in tqdm(movie_uris):
        movie_title = uri.split('/')[-1].replace('_', ' ')        # prendiamo il titolo direttamente dall'uri
        
        # Get Wikipedia abstract
        abstract = get_wikipedia_abstract(movie_title)            # estrae l'abstract dato il titolo
        if not abstract:
            continue
            
        # Get RDF triples
        triples = get_triples_for_movie(uri)       # crea le triple per ogni film
        if not triples:
            continue
            
        wiki_data.append({
            "title": movie_title,
            "abstract": abstract,
            "triples": triples
        })
        
        # Sii gentile con le API
        time.sleep(0.1)

    print(f"3. Saving collected data to data/raw_data.jsonl...")
    with open("data/raw_data.jsonl", "w", encoding="utf-8") as f:
        for entry in wiki_data:
            f.write(json.dumps(entry) + "\n")
            
    print("Done!")



if __name__ == "__main__":
    # Assicurati di aver creato una cartella 'data'
    import os
    if not os.path.exists('data'):
        os.makedirs('data')
    main()
import sys
import os
import time

import polars as pl
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from openai import OpenAI

client = OpenAI()

subreddit = ''
start = 0

def log(text, section = '', type='info'):
    time_passed = time.time() - start

    if len(section) > 0:
        log('\n\n\t\t====== ' + section + ' ======', type='section')
        return
        
    if type != 'section':
        text = f'[ {type} | {time_passed:.2f}s ] {text}'

    print(text)

    path = f'results/{subreddit}/logs.txt'

    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            pass

    with open(path, 'a') as f:
        f.write(f'{text}\n')

def analyze_id(id):
    parts = id.split('_')

    if len(parts) < 2:
        return (parts[0], None)
    
    types = {
        't1': 'comment',
        't2': 'account',
        't3': 'link',
        't4': 'message',
        't5': 'subreddit',
        't6': 'award',
        't8': 'promo_campaign'
    }

    return (types[parts[0]], parts[1])

def get_embedding(text, model="text-embedding-3-small"):
    response = client.embeddings.create(model=model, input=text)
    return response.data[0].embedding

def get_author_embedding(author, comments, n=10):
    author_comments = comments.filter(pl.col("author") == author)

    # cast created_utc to i64
    author_comments = author_comments.with_columns(pl.col("created_utc").cast(pl.Int64))

    # merge posts and comments in new data frame
    author_content = author_comments.select(['body', 'created_utc'])

    # sort by date and get the first n rows
    author_content = author_content.sort("created_utc", descending=True).head(n)

    author_text = " ".join(author_content["body"].to_list())

    return get_embedding(author_text)

def embedding_weight_correlation(G, comments):
    log(f'Calculating embedding-weight correlationf for {len(G.nodes())} authors...')
    interactions = []
    authors_list = []
    
    for edge in G.edges(data=True):
        interactions.append({
            "from_author": edge[0],
            "to_author": edge[1],
            "weight": edge[2]["weight"]
        })    
        
        authors_list.append(edge[0])
        authors_list.append(edge[1])

    authors_list = list(set(authors_list))
    
    n = 0
    
    author_embeddings = {}
    
    for author in authors_list:
        log(f'Getting embedding for author {author} ({n}/{len(authors_list)})')
        author_embeddings[author] = get_author_embedding(author, comments=comments)
        n += 1
    
    similarity_of_authors = {
        (author1, author2): np.dot(author_embeddings[author1], author_embeddings[author2])
        for author1 in authors_list
        for author2 in authors_list
    }
    
    d = interactions

    for i in range(len(d)):
        d[i]["similarity"] = similarity_of_authors[(d[i]["from_author"], d[i]["to_author"])]
        
    # correlation between weight and similarity

    weights = map(lambda x: x["weight"], d)
    similarities = map(lambda x: x["similarity"], d)

    return np.corrcoef(list(weights), list(similarities))[0][1]

def analyze_subreddit(subreddit, analyze_posts = True):
    global start
    start = time.time()

    path = f'results/{subreddit}/logs.txt'

    # clear logs
    if os.path.exists(path):
        os.remove(path)

    log('', section='Starting')

    log(f'Starting at {time.ctime()}')
    log(f'Analyzing subreddit \'{subreddit}\'')

    log('', section='Loading Data')

    submissions_pre = pl.DataFrame({})
    
    if analyze_posts:
        log('Reading submissions...')
        submissions_pre = pl.read_ndjson(f'./data/{subreddit}/{subreddit}_submissions.ndjson')
        log('Reading submissions... Done!')
    else:
        log('Skipping submissions...')

    log('Reading comments...')
    comments_pre = pl.read_ndjson(f'./data/{subreddit}/{subreddit}_comments.ndjson')
    log('Reading comments... Done!')

    log('', section='Data Preprocessing')    
    
    
    posts = pl.DataFrame({})
    
    if analyze_posts:
        submissions_pre.select(
            ["id", "author", "title", "created_utc", "selftext"]
        )
    
    comments = comments_pre.select(
        ["id", "link_id", "parent_id", "author", "created_utc", "body"]
    )
    
    n_pre_deletion = len(posts) + len(comments)
    
    # remove [deleted] authors
    if analyze_posts:
        posts = posts.filter(pl.col("author") != "[deleted]")
    comments = comments.filter(pl.col("author") != "[deleted]")
    
    n_post_deletion = len(posts) + len(comments)
    
    log('Data Preprocessing... Done!')
    
    log('', section='Building Author Map')
    
    post_to_author = {}
    comment_to_author = {}

    if analyze_posts:
        for post in posts.iter_rows(named=True):
            post_to_author[post["id"]] = post["author"]

    for comment in comments.iter_rows(named=True):
        comment_to_author[comment["id"]] = comment["author"]

    def get_author(type, id):
        if type == 'comment':
            return comment_to_author[id] if id in comment_to_author else None
        elif type == 'link' and analyze_posts:
            return post_to_author[id] if id in post_to_author else None
        else:
            return None
        
    log('Building Author Map... Done!')
    
    log('', section='Extracting Graph Data')
    
    edges = comments.with_columns(
        pl.col('author').alias("from_author"),
        pl.col('link_id').map_elements(lambda id: get_author(*analyze_id(id)), return_dtype=pl.String).alias("post_author"),
        pl.col('parent_id').map_elements(lambda id: get_author(*analyze_id(id)), return_dtype=pl.String).alias("to_author")
    ).select(["from_author", "post_author", "to_author", "created_utc"])
    
    weighted_edges = (
        edges
        .rename({"from_author": "to_author", "to_author": "from_author"})
        .group_by(["from_author", "to_author"]).len().rename({"len": "weight"})
        .filter(pl.col("from_author").is_not_null() & pl.col("to_author").is_not_null())
    )
    
    log('Extracting Graph Data... Done!')
    
    log('', section='Building Graph')
    
    G = nx.DiGraph()

    for edge in weighted_edges.iter_rows(named=True):
        G.add_edge(edge["from_author"], edge["to_author"], weight=edge["weight"])
        
    log('Building Graph... Done!')
    
    # log('', section='Exporting Graph')
    
    # nx.write_gexf(G, f"./results/{subreddit}/fullgraph.gexf")
    # log(f'Exported Graph to GEXF at \'./results/{subreddit}/fullgraph.gexf\'')
    
    log('', section='Top authors')
    
    n = 100
        
    log(f'Selecting top authors (n = {n})...')
    
    top_authors = nx.voterank(G, n)
    
    subgraph = G.subgraph(top_authors)
    
    log(f'Selected top authors (n = {n})... Done!')
    
    log(f'Exporting Subgraph to GEXF...')
    
    nx.write_gexf(subgraph, f"./results/{subreddit}/subgraph.gexf")
    
    log(f'Exported Subgraph to GEXF at \'./results/{subreddit}/subgraph.gexf\'')
    
    log('', section='Calculating Metrics')
    
    log(f'Number of interactions: {len(edges)}', type='result')
    
    log(f'Number of redditors: {G.number_of_nodes()}', type='result')
    log(f'Number of relationships: {G.number_of_edges()}', type='result')
    
    
    log('Calculating density...')
    density = nx.density(G)
    log(f'Density: {density}', type='result')
    
    log('Calculating average reciprocity...')
    avg_reciprocity = nx.overall_reciprocity(G)
    log(f'Average reciprocity: {avg_reciprocity}', type='result')
    
    log('Creating degree distribution histogram...')
    degree_distribution = nx.degree_histogram(G)
    
    # Creating plot
    plt.plot(degree_distribution)
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.title('Degree Distribution')
    plt.savefig(f'./results/{subreddit}/degree_distribution.png')
    
    plt.clf()
    
    log(f'Created degree distribution histogram at \'./results/{subreddit}/degree_distribution.png\'')
    
    log('Calculating pagerank...')
    pagerank = nx.pagerank(G)
    log(f'Pagerank. Done!', type='result')
    log('Creating pagerank histogram...')
    
    # Creating plot
    plt.hist(pagerank.values())
    plt.xlabel('Pagerank')
    plt.ylabel('Frequency')
    plt.title('Pagerank Distribution')
    plt.savefig(f'./results/{subreddit}/pagerank_distribution.png')
    
    plt.clf()
    
    log(f'Created pagerank histogram at \'./results/{subreddit}/pagerank_distribution.png\'')
    
    log('Calculating degree centrality...')
    degree_centrality = nx.degree_centrality(G)
    log(f'Degree centrality. Done!', type='result')
    log('Creating degree centrality histogram...')
    
    # Creating plot
    plt.hist(degree_centrality.values())
    plt.xlabel('Degree Centrality')
    plt.ylabel('Frequency')
    plt.title('Degree Centrality Distribution')
    plt.savefig(f'./results/{subreddit}/degree_centrality_distribution.png')
    
    plt.clf()
    
    log(f'Created degree centrality histogram at \'./results/{subreddit}/degree_centrality_distribution.png\'')
    
    log('Calculating eigenvector centrality...')
    eigenvector_centrality = nx.eigenvector_centrality(G)
    log(f'Eigenvector centrality. Done!', type='result')
    log('Creating eigenvector centrality histogram...')
    
    # Creating plot
    plt.hist(eigenvector_centrality.values())
    plt.xlabel('Eigenvector Centrality')
    plt.ylabel('Frequency')
    plt.title('Eigenvector Centrality Distribution')
    plt.savefig(f'./results/{subreddit}/eigenvector_centrality_distribution.png')
    
    plt.clf()
    
    log(f'Created eigenvector centrality histogram at \'./results/{subreddit}/eigenvector_centrality_distribution.png\'')
    
    log('Calculating variance and standard deviation of centralities...')
    
    var_pagerank = np.var(list(pagerank.values()))
    var_degree_centrality = np.var(list(degree_centrality.values()))
    var_eigenvector_centrality = np.var(list(eigenvector_centrality.values()))
    
    sd_pagerank = np.std(list(pagerank.values()))
    sd_degree_centrality = np.std(list(degree_centrality.values()))
    sd_eigenvector_centrality = np.std(list(eigenvector_centrality.values()))
    
    cv_pagerank = sd_pagerank / np.mean(list(pagerank.values()))
    cv_degree_centrality = sd_degree_centrality / np.mean(list(degree_centrality.values()))
    cv_eigenvector_centrality = sd_eigenvector_centrality / np.mean(list(eigenvector_centrality.values()))
    
    log(f'Pagerank variance: {var_pagerank}', type='result')
    log(f'Degree centrality variance: {var_degree_centrality}', type='result')
    log(f'Eigenvector centrality variance: {var_eigenvector_centrality}', type='result')
    
    log(f'Pagerank standard deviation: {sd_pagerank}', type='result')
    log(f'Degree centrality standard deviation: {sd_degree_centrality}', type='result')
    log(f'Eigenvector centrality standard deviation: {sd_eigenvector_centrality}', type='result')
    
    log(f'Pagerank coefficient of variation: {cv_pagerank}', type='result')
    log(f'Degree centrality coefficient of variation: {cv_degree_centrality}', type='result')
    log(f'Eigenvector centrality coefficient of variation: {cv_eigenvector_centrality}', type='result')
    
    n_deleted = n_pre_deletion - n_post_deletion
    deleted_ratio = n_deleted / n_pre_deletion
    
    log(f'Number of deleted authors: {n_deleted}', type='result')
    log(f'Percentage of deleted authors: {deleted_ratio * 100:.2f}%', type='result')
    
    log('Calculating embedding-weight correlation...')
    ew_cor = embedding_weight_correlation(subgraph, comments)
    log(f'Embedding-weight correlation: {ew_cor}', type='result')
    
    log('Exporting metrics to CSV...')
    
    metrics = {
        'interactions': len(edges),
        'redditors': G.number_of_nodes(),
        'relationships': G.number_of_edges(),
        'density': density,
        'avg_reciprocity': avg_reciprocity,
        'var_pagerank': var_pagerank,
        'var_degree_centrality': var_degree_centrality,
        'var_eigenvector_centrality': var_eigenvector_centrality,
        'sd_pagerank': sd_pagerank,
        'sd_degree_centrality': sd_degree_centrality,
        'sd_eigenvector_centrality': sd_eigenvector_centrality,
        'cv_pagerank': cv_pagerank,
        'cv_degree_centrality': cv_degree_centrality,
        'cv_eigenvector_centrality': cv_eigenvector_centrality,
        'n_deleted': n_deleted,
        'deleted_ratio': deleted_ratio,
        'embedding_weight_correlation': ew_cor
    }
    
    with open(f'./results/{subreddit}/metrics.csv', 'w') as f:
        f.write(','.join(metrics.keys()))
        f.write('\n')
        f.write(','.join(map(str, metrics.values())))
    
    log(f'Exported metrics to CSV at \'./results/{subreddit}/metrics.csv\'')
    
    log('', section='Done!')
    
    log(f'Finished at {time.ctime()}, the analysis took {time.time() - start:.2f}s')
    
    log(f'You can find the results at \'./results/{subreddit}/\'')
    

if __name__ == '__main__':
    # get param from command line
    if len(sys.argv) < 2:
        print('Please provide a subreddit name!\n\n\tUsage: python analyze_subreddit.py <subreddit>\n')
        sys.exit(1)
    
    subreddit = sys.argv[1]
    
    analyze_posts = True
    
    if len(sys.argv) == 3 and sys.argv[2] == '--no-posts':
        analyze_posts = False
        
    analyze_subreddit(subreddit, analyze_posts = analyze_posts)
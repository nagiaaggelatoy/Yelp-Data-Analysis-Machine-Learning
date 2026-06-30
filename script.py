import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.cluster import DBSCAN

#ΜΕΡΟΣ 1: ΧΩΡΙΚΗ ΟΜΑΔΟΠΟΙΗΣΗ ΚΑΙ ΟΠΤΙΚΟΠΟΙΗΣΗ
# 1. ΦΟΡΤΩΣΗ ΚΑΙ ΚΑΘΑΡΙΣΜΟΣ
df = pd.read_csv('yelp_training_set_business.csv', sep=';', decimal=',')
df = df.dropna(subset=['business_latitude', 'business_longitude', 'business_full_address'])
df['extracted_zip'] = df['business_full_address'].str.extract(r'(\d{5})')

# 2. ΧΩΡΙΚΗ ΟΜΑΔΟΠΟΙΗΣΗ (DBSCAN) - Εντοπισμός "Hotspots"
coords = np.radians(df[['business_latitude', 'business_longitude']])
# eps=0.002 είναι ιδανικό για να βγάλει 4-5 καθαρά hotspots (clusters)
db = DBSCAN(eps=0.002, min_samples=15, metric='haversine').fit(coords)
df['cluster'] = db.labels_

# 3. ΑΝΑΛΥΣΗ ΣΥΧΝΟΤΗΤΩΝ (Frequency)
# Πόσα μαγαζιά ανά περιοχή (ZIP) και ανά Cluster
zip_freq = df['extracted_zip'].value_counts().head(10)
cluster_freq = df[df['cluster'] != -1]['cluster'].value_counts()

# 4. ΣΥΝΔΥΑΣΤΙΚΗ ΑΝΑΛΥΣΗ (STAKING: Κατηγορίες ανά Cluster)
# Εδώ γίνεται το "stacking": ποια είδη μαγαζιών υπάρχουν σε κάθε hotspot
stacked_data = df[df['cluster'] != -1].groupby(['cluster', 'business_type']).size().unstack(fill_value=0)

# 5. ΟΠΤΙΚΟΠΟΙΗΣΕΙΣ 
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# A. Χάρτης Hotspots
sns.scatterplot(data=df[df['cluster'] != -1], x='business_longitude', y='business_latitude', 
                hue='cluster', ax=axes[0, 0], palette='viridis', s=20)
axes[0, 0].set_title("Χάρτης Επιχειρηματικών Hotspots (DBSCAN)")

# B. Συχνότητα ανά ZIP
zip_freq.plot(kind='bar', ax=axes[0, 1], color='salmon')
axes[0, 1].set_title("Συχνότητα Εστιατορίων ανά ZIP Code")

# C. Stacked Analysis (Κατηγορίες ανά Cluster)
stacked_data.plot(kind='bar', stacked=True, ax=axes[1, 0], colormap='viridis')
axes[1, 0].set_title("Κατανομή Κατηγοριών ανά Cluster (Stacking)")

# D. Ποιότητα (Stars) ανά Cluster
sns.boxplot(data=df[df['cluster'] != -1], x='cluster', y='business_stars', ax=axes[1, 1], hue='cluster', palette='magma', legend=False)
axes[1, 1].set_title("Ποιότητα (Stars) ανά Επιχειρηματική Ζώνη")

plt.tight_layout()
plt.show()

# 6. ΤΕΛΙΚΟ REPORT 
print("--- ΕΠΙΧΕΙΡΗΜΑΤΙΚΗ ΑΝΑΛΥΣΗ ---")
print(f"Εντοπίστηκαν {len(cluster_freq)} κύριες επιχειρηματικές ζώνες.")
print("\n--- Μέση βαθμολογία ανά ζώνη ---")
print(df.groupby('cluster')['business_stars'].mean()) 

#ΜΕΡΟΣ 2: ΧΑΡΑΚΤΗΡΙΣΜΟΣ CLUSTERS ΚΑΙ ΑΝΑΛΥΣΗ ΘΕΜΑΤΩΝ ΣΥΖΗΤΗΣΗΣ(TOPIC MODELING)
# 1. ΠΡΟΦΙΛ ΚΑΘΕ CLUSTER ΑΠΟ ΤΙΣ ΚΑΤΗΓΟΡΙΕΣ
df_clustered = df[df['cluster'] != -1].copy()
df_clustered['category_list'] = df_clustered['business_categories'].str.split(';')
df_exploded= df_clustered.explode('category_list')
df_exploded['category_list'] = df_exploded['category_list'].str.strip()

ignore_words= ['Restaurants', 'Food', 'Business', 'Shopping']
df_exploded= df_exploded[~df_exploded['category_list'].isin(ignore_words)]

print("\n=== ΚΥΡΙΑΡΧΕΣ ΚΑΤΗΓΟΡΙΕΣ ΑΝΑ ΕΠΙΧΕΙΡΗΜΑΤΙΚΗ ΖΩΝΗ ===")
for cluster_id in sorted(df_exploded['cluster'].unique()):
    sub_df = df_exploded[df_exploded['cluster'] == cluster_id]
    top_categories = sub_df['category_list'].value_counts().head(5)
    print(f"\nΤαυτότητα για το Cluster {cluster_id}:")
    print(top_categories)
    
# 2. Topic Modeling στα reviews για όλα τα clusters(lda)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

#φορτώνουμε το αρχείο των reviews 
print("\n=== ΦΟΡΤΩΣΗ ΚΡΙΤΙΚΩΝ ===")
df_reviews = pd.read_csv ('yelp_training_set_review.csv', sep=',' , on_bad_lines='skip', nrows=20000)
#καθαρίζουμε τα ονόματα
df_reviews.columns = df_reviews.columns.str.strip()
df.columns = df.columns.str.strip()

#ενώνουμε τα reviews με τα clusters των μαγαζιών
df_merged_reviews = pd.merge(df_reviews, df[['business_id', 'cluster']], on= 'business_id', how='inner' )
df_reviews.columns = df_reviews.columns.str.strip()
df.columns = df.columns.str.strip()
print("\n=== ΑΝΑΛΥΣΗ ΘΕΜΑΤΩΝ (TOPIC MODELING) ΑΝΑ CLUSTER ===")

#παίρνουμε όλα τα μοναδικά IDs των clusters εκτός του -1 που είναι ο θόρυβος
availiable_clusters = sorted([c for c in df_merged_reviews['cluster'].unique() if c != -1])

for cluster_id in availiable_clusters: 
    # φιλτράρουμε τισ κριτικές για το συγκεκριμένο cluster μόνο
    reviews_text = df_merged_reviews[df_merged_reviews['cluster'] == cluster_id]['text'].dropna()
    #αν κάποιο cluster έχει λίγες κριτικές δεν το λαμβάνω υπ΄'οψιν 
    if len(reviews_text) < 10: 
        print(f"\n--- CLUSTER {cluster_id}: Ανεπαρκή δεδομένα κριτικών για ανάλυση ({len(reviews_text)} reviews) ---")
        continue
    print(f'\n--- ΑΝΑΛΥΣΗ ΓΙΑ ΤΟ CLUSTER {cluster_id} ({len(reviews_text)} reviews) ---') 
    
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    dtm = vectorizer.fit_transform(reviews_text) 
    
    #εκπαιδεύουμε το μοντέλο lda για το cluster όπου ζητάμε 3 βασικά topics
    lda_model = LatentDirichletAllocation(n_components=3, random_state=42)
    lda_model.fit(dtm)
    
    #εξάγουμε key words
    words = vectorizer.get_feature_names_out()
    for index, topic in enumerate(lda_model.components_):
        #παίρνουμε τισ 12 πιο σημαντικές λέξεις που χαρακτηρίζουν το θέμα
        top_words = [words[i] for i in topic.argsort()[-12:]]
        print(f" *Θέμα #{index+1}: {', '.join(top_words)}")
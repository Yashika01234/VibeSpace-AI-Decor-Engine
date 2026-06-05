from flask import Flask, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Mock Product Database
products_data = [
    {"id": 1, "name": "Aura Sunset Projector Lamp", "tags": "cozy warm sunset minimal aesthetic bedroom"},
    {"id": 2, "name": "Minimalist Oak Monitor Riser", "tags": "minimal clean wood desk organizer desk-setup"},
    {"id": 3, "name": "Matte Black Hexagonal Acoustic Panels", "tags": "minimal dark-mode soundproof geometry studio"},
    {"id": 4, "name": "Warm LED Floating Bed Strips", "tags": "cozy warm lighting ambient bedroom tech-setup"}
]
df = pd.DataFrame(products_data)

def get_recommendations(product_id):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['tags'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    idx = df[df['id'] == product_id].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_indices = [i[0] for i in sim_scores[1:3]]
    return df.iloc[top_indices].to_dict(orient='records')

@app.route('/api/recommend/<int:pid>', methods=['GET'])
def recommend_decor(pid):
    return jsonify({"status": "success", "ai_recommendations": get_recommendations(pid)})

if __name__ == '__main__':
    app.run(debug=True)

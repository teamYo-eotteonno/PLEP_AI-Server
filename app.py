from flask import Flask, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json

app = Flask(__name__)

def get_user_vector(user_ratings, all_locations):
    return np.array([user_ratings.get(loc, 0) for loc in all_locations])

def recommend(user_id, user_location_matrix, top_n=2):
    all_locations = sorted(set().union(*[places.keys() for places in user_location_matrix.values()]))

    target_vector = get_user_vector(user_location_matrix[user_id], all_locations)
    
    similarities = []
    for other_user in user_location_matrix:
        if other_user == user_id:
            continue
        other_vector = get_user_vector(user_location_matrix[other_user], all_locations)
        sim = cosine_similarity([target_vector], [other_vector])[0][0]
        similarities.append((other_user, sim))

    if not similarities:
        return []

    similarities.sort(key=lambda x: x[1], reverse=True)
    most_similar_user = similarities[0][0]
    similar_user_ratings = user_location_matrix[most_similar_user]

    recommendations = {
        loc: count for loc, count in similar_user_ratings.items()
        if loc not in user_location_matrix[user_id] or user_location_matrix[user_id][loc] == 0
    }

    return sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:top_n]

@app.route('/recommend', methods=['POST'])
def recommend_endpoint():
    data = request.get_json()

    user_id = data.get("user_id")
    user_location_matrix = data.get("user_location_matrix")

    if user_id not in user_location_matrix:
        return jsonify({"error": "User ID not found"}), 400

    recs = recommend(user_id, user_location_matrix)
    response = {
        "user_id": user_id,
        "recommendations": [{"location": loc, "score": score} for loc, score in recs]
    }

    # 한글 깨지지 않게
    return app.response_class(
        response=json.dumps(response, ensure_ascii=False),
        status=200,
        mimetype='application/json'
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=True)
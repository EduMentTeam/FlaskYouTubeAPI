from flask import Flask, request, jsonify
from googleapiclient.discovery import build
from sentence_transformers import SentenceTransformer
import numpy as np

app = Flask(__name__)

# ضع مفتاح YouTube API الخاص بك هنا
API_KEY = "YOUR_API_KEY"  # استبدل بـ API Key الصحيح

def search_youtube(query, max_results=10):
    youtube = build("youtube", "v3", developerKey=API_KEY)
    request_youtube = youtube.search().list(
        q=query,
        part="snippet",
        type="video",
        maxResults=max_results
    )
    response = request_youtube.execute()
    videos = []
    for item in response.get("items", []):
        video_id = item["id"]["videoId"]
        title = item["snippet"]["title"]
        description = item["snippet"]["description"]
        # جلب إحصائيات الفيديو (مثل عدد المشاهدات والإعجابات)
        video_details = youtube.videos().list(
            part="statistics",
            id=video_id
        ).execute()
        stats = video_details["items"][0]["statistics"]
        view_count = int(stats.get("viewCount", 0))
        like_count = int(stats.get("likeCount", 0)) if "likeCount" in stats else 0

        videos.append({
            "video_id": video_id,
            "title": title,
            "description": description,
            "view_count": view_count,
            "like_count": like_count
        })
    return videos

def compute_similarity(lesson_text, video_text, model):
    lesson_emb = model.encode([lesson_text])[0]
    video_emb = model.encode([video_text])[0]
    cosine_sim = np.dot(lesson_emb, video_emb) / (np.linalg.norm(lesson_emb) * np.linalg.norm(video_emb))
    return cosine_sim

@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    lesson_title = data.get("lessonTitle")
    if not lesson_title:
        return jsonify({"error": "Missing parameter: lessonTitle"}), 400

    # البحث عن فيديوهات على YouTube
    videos = search_youtube(lesson_title, max_results=10)
    
    # تحميل نموذج AI لتحليل النصوص
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # حساب نسبة التشابه باستخدام وصف الفيديو
    for video in videos:
        sim = compute_similarity(lesson_title, video["description"], model)
        video["similarity"] = sim
    
    # فرز النتائج بناءً على التشابه
    videos_sorted = sorted(videos, key=lambda x: x["similarity"], reverse=True)
    return jsonify({"results": videos_sorted})

if __name__ == '__main__':
    # تشغيل الخدمة على العنوان 0.0.0.0 والمنفذ 5000
    app.run(host='0.0.0.0', port=5000, debug=True)

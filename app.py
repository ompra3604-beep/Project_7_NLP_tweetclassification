import os

from flask import Flask, render_template, request

from predictor import classify_tweet, load_classifier

MAX_TWEET_LENGTH = 5_000
model = load_classifier()
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024


@app.route("/", methods=["GET", "POST"])
def index():
    tweet = ""
    result = None
    error = None
    if request.method == "POST":
        tweet = request.form.get("tweet", "").strip()
        if not tweet:
            error = "Enter a tweet to classify."
        elif len(tweet) > MAX_TWEET_LENGTH:
            error = f"Keep the tweet under {MAX_TWEET_LENGTH:,} characters."
        else:
            result = classify_tweet(model, tweet)
    return render_template("index.html", tweet=tweet, result=result, error=error)


@app.get("/healthz")
def healthz():
    return {"status": "ok", "model_loaded": model is not None}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)

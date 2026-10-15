from flask import Flask, render_template, request, redirect, url_for, send_file, flash
import os
import pandas as pd
from werkzeug.utils import secure_filename

from model import process_music_data
from audio_features import extract_audio_features

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change-this-in-production")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
PROCESSED_FOLDER = os.path.join(BASE_DIR, "processed")
CSV_EXTENSIONS = {"csv"}
AUDIO_EXTENSIONS = {"mp3", "wav"}
MAX_CONTENT_LENGTH = 25 * 1024 * 1024
REQUIRED_COLUMNS = {"tempo", "energy", "danceability"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["PROCESSED_FOLDER"] = PROCESSED_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)


def allowed_file(filename, allowed_extensions):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload")
def upload():
    return render_template("upload.html")


@app.route("/process", methods=["POST"])
def process():
    if "file" not in request.files:
        flash("No file part in the request.", "error")
        return redirect(url_for("upload"))

    file = request.files["file"]
    if not file or file.filename == "":
        flash("No file selected. Please choose a CSV file.", "error")
        return redirect(url_for("upload"))

    if not allowed_file(file.filename, CSV_EXTENSIONS):
        flash("Invalid file type. Only .csv files are allowed for dataset analysis.", "error")
        return redirect(url_for("upload"))

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        df = pd.read_csv(filepath)
        if df.empty:
            raise ValueError("The uploaded CSV file is empty.")

        normalized_columns = {str(col).strip().lower() for col in df.columns}
        missing = REQUIRED_COLUMNS - normalized_columns
        if missing:
            raise ValueError(
                "CSV is missing required columns: "
                + ", ".join(sorted(missing))
                + ". Required: tempo, energy, danceability."
            )

        output_file, plot_file, stats = process_music_data(filepath)
        return render_template(
            "result.html",
            output_file=output_file,
            plot_file=plot_file,
            stats=stats,
        )
    except Exception as exc:
        flash(f"Dataset analysis failed: {exc}", "error")
        return redirect(url_for("upload"))
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


@app.route("/analyze-audio", methods=["POST"])
def analyze_audio():
    if "audio" not in request.files:
        flash("Please select an audio file.", "error")
        return redirect(url_for("upload"))

    file = request.files["audio"]
    if not file or file.filename == "":
        flash("No audio file selected.", "error")
        return redirect(url_for("upload"))

    if not allowed_file(file.filename, AUDIO_EXTENSIONS):
        flash("Only MP3 and WAV files are supported for song analysis.", "error")
        return redirect(url_for("upload"))

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        features = extract_audio_features(filepath)
        return render_template("audio_result.html", filename=filename, features=features)
    except Exception as exc:
        flash(f"Audio analysis failed: {exc}", "error")
        return redirect(url_for("upload"))
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


@app.route("/download/<filename>")
def download(filename):
    safe_name = secure_filename(filename)
    file_path = os.path.join(app.config["PROCESSED_FOLDER"], safe_name)
    if not os.path.exists(file_path):
        flash("File not found.", "error")
        return redirect(url_for("index"))
    return send_file(file_path, as_attachment=True)


@app.errorhandler(413)
def file_too_large(_error):
    flash("File is too large. Maximum allowed size is 25 MB.", "error")
    return redirect(url_for("upload"))


@app.errorhandler(404)
def not_found(_error):
    return render_template("404.html"), 404


@app.errorhandler(500)
def server_error(_error):
    return render_template("500.html"), 500


if __name__ == "__main__":
    app.run(debug=True)

import os
import sqlite3
import sys
from functools import wraps
import numpy as np
import cv2
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, Response
from werkzeug.security import generate_password_hash, check_password_hash

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    from inference_backend import SignLanguageBackend
except Exception:
    SignLanguageBackend = None


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "users.db")


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


    def login_required(view_func):\n        @wraps(view_func)\n        def wrapper(*args, **kwargs):\n            if session.get("user_id") == "guest":\n                return view_func(*args, **kwargs)\n            if "user_id" not in session:\n                return redirect(url_for("login"))\n            return view_func(*args, **kwargs)\n\n        return wrapper


_backend_instance = None
_backend_error = None


def get_backend():
    global _backend_instance, _backend_error
    if _backend_instance is not None:
        return _backend_instance
    if SignLanguageBackend is None:
        _backend_error = "inference backend import failed"
        return None
    try:
        _backend_instance = SignLanguageBackend()
        return _backend_instance
    except Exception as e:
        _backend_error = str(e)
        return None


def generate_mjpeg_stream():
    while True:
        backend = get_backend()
        if backend is None:
            blank = (255 * np.ones((480, 640, 3), dtype=np.uint8))
            ok, enc = cv2.imencode(".jpg", blank)
            if not ok:
                continue
            frame = enc.tobytes()
        else:
            frame_bgr = backend.read_processed_frame()
            if frame_bgr is None:
                continue
            ok, enc = cv2.imencode(".jpg", frame_bgr)
            if not ok:
                continue
            frame = enc.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )


def create_app():
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key-change-this")

    init_db()

@app.route("/")\n    def index():\n        # Guest access for standalone app\n        if not session.get("user_id"):\n            session["user_id"] = "guest"\n            session["user_name"] = "Guest"\n            session["user_email"] = "guest@example.com"\n        return render_template("index.html", user_name=session.get("user_name"))

    @app.route("/signup", methods=["GET", "POST"])
    def signup():
        if request.method == "POST":
            name = request.form.get("name", "").strip()
            email = request.form.get("email", "").strip().lower()
            password = request.form.get("password", "")

            if not name or not email or not password:
                flash("All fields are required.", "error")
                return render_template("signup.html")

            password_hash = generate_password_hash(password)

            try:
                conn = get_db_connection()
                conn.execute(
                    "INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)",
                    (name, email, password_hash),
                )
                conn.commit()
                conn.close()
                flash("Account created successfully. Please login.", "success")
                return redirect(url_for("login"))
            except sqlite3.IntegrityError:
                flash("Email already exists. Please use another email.", "error")
                return render_template("signup.html")

        return render_template("signup.html")

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "POST":
            email = request.form.get("email", "").strip().lower()
            password = request.form.get("password", "")

            conn = get_db_connection()
            user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
            conn.close()

            if user and check_password_hash(user["password_hash"], password):
                session["user_id"] = user["id"]
                session["user_name"] = user["name"]
                session["user_email"] = user["email"]
                return redirect(url_for("tools"))

            flash("Invalid email or password.", "error")
            return render_template("login.html")

        return render_template("login.html")

    @app.route("/dashboard")
    @login_required
    def dashboard():
        return render_template(
            "dashboard.html",
            user_name=session.get("user_name"),
            user_email=session.get("user_email"),
        )

    @app.route("/tools")
    @login_required
    def tools():
        return render_template(
            "tools.html",
            user_name=session.get("user_name"),
            user_email=session.get("user_email"),
        )

    @app.route("/tools/status")
    @login_required
    def tools_status():
        backend = get_backend()
        if backend is None:
            return jsonify({
                "character": "",
                "word": "",
                "sentence": "",
                "suggestions": [],
                "application_backend_available": False,
                "backend_error": _backend_error or "backend unavailable",
            })
        status = backend.get_status()
        status["application_backend_available"] = True
        status["backend_error"] = None
        return jsonify(status)

    @app.route("/tools/video_feed")
    @login_required
    def tools_video_feed():
        return Response(
            generate_mjpeg_stream(),
            mimetype="multipart/x-mixed-replace; boundary=frame"
        )

    @app.route("/tools/action/<int:index>", methods=["POST"])
    @login_required
    def tools_action(index):
        backend = get_backend()
        if backend is None:
            return jsonify({"ok": False, "error": _backend_error or "backend unavailable"}), 503
        backend.apply_suggestion(index - 1)
        return jsonify({"ok": True})

    @app.route("/tools/clear", methods=["POST"])
    @login_required
    def tools_clear():
        backend = get_backend()
        if backend is None:
            return jsonify({"ok": False, "error": _backend_error or "backend unavailable"}), 503
        backend.clear()
        return jsonify({"ok": True})

    @app.route("/tools/speak", methods=["POST"])
    @login_required
    def tools_speak():
        return jsonify({"ok": True, "message": "Speak action acknowledged"})

    @app.route("/logout")
    def logout():
        session.clear()
        return redirect(url_for("index"))

    return app


app = create_app()


#if __name__ == "__main__":
 #   app.run(debug=True)

import webbrowser

if __name__ == "__main__":
    webbrowser.open("http://127.0.0.1:5000")
    app.run(debug=False)
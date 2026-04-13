# Django Frontend

## Role of Django

Django is the **user-facing web layer only**. It:
- Serves the HTML UI
- Handles user authentication (login/signup)
- Manages upload flow (calls FastAPI backend)
- Renders the chat interface
- Streams LLM responses to the browser

Django does **not** run RAG logic, call Bytez directly, or touch S3 directly. All intelligence is in the FastAPI service. Django is a thin client over it.

## Why Django and Not Streamlit

| Criterion | Streamlit | Django |
|---|---|---|
| Build time | Hours | Days |
| Portfolio signal | "quick demo" | "full-stack engineer" |
| Customizable UI | Limited | Full control |
| Auth built-in | No | Yes (django.contrib.auth) |
| Production-grade | Debatable | Yes |
| Streaming support | Limited | Via SSE + JS EventSource |

For job search purposes: Django shows you can build a real web application, not just a data science demo.

## Django App Structure

```
django_frontend/
├── manage.py
├── config/
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── accounts/            # user auth (login, signup, logout)
│   ├── models.py
│   ├── views.py
│   ├── urls.py
│   └── templates/
│       └── accounts/
│           ├── login.html
│           └── signup.html
├── documents/           # upload + session management
│   ├── models.py        # Session model (maps to FastAPI session_id)
│   ├── views.py
│   ├── urls.py
│   └── templates/
│       └── documents/
│           ├── upload.html
│           └── sessions.html
├── chat/                # chat interface + streaming
│   ├── views.py
│   ├── urls.py
│   └── templates/
│       └── chat/
│           └── chat.html
└── static/
    ├── css/
    │   └── main.css
    └── js/
        └── chat.js      # SSE stream handler
```

## Django Models

```python
# documents/models.py

from django.db import models
from django.contrib.auth.models import User

class DocumentSession(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    session_id = models.CharField(max_length=100, unique=True)
    original_filename = models.CharField(max_length=255)
    status = models.CharField(
        max_length=20,
        choices=[
            ("UPLOADING", "Uploading"),
            ("PROCESSING", "Processing"),
            ("READY", "Ready"),
            ("FAILED", "Failed"),
        ],
        default="UPLOADING"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.user.username} — {self.original_filename} ({self.status})"
```

Django's DB is SQLite in development, Postgres on Render. It stores only session metadata — all document data is in S3.

## Key Views

### Upload View

```python
# documents/views.py

import requests
from django.contrib.auth.decorators import login_required

FASTAPI_BASE = os.environ["FASTAPI_SERVICE_URL"]  # e.g., https://loandoc-api.onrender.com

@login_required
def upload(request):
    if request.method == "POST":
        pdf_file = request.FILES["pdf"]

        # Call FastAPI to upload to S3 and start pipeline
        response = requests.post(
            f"{FASTAPI_BASE}/upload",
            files={"file": (pdf_file.name, pdf_file.read(), "application/pdf")},
            headers={"Authorization": f"Bearer {request.session['jwt_token']}"},
        )
        data = response.json()

        # Save session record in Django DB
        DocumentSession.objects.create(
            user=request.user,
            session_id=data["session_id"],
            original_filename=pdf_file.name,
            status="PROCESSING",
        )

        return redirect("documents:status", session_id=data["session_id"])

    return render(request, "documents/upload.html")
```

### Status Polling View

The upload page polls this endpoint until the pipeline completes:

```python
@login_required
def session_status(request, session_id):
    response = requests.get(f"{FASTAPI_BASE}/sessions/{session_id}/status")
    status = response.json()["status"]

    # Update local Django DB
    DocumentSession.objects.filter(session_id=session_id).update(status=status)

    return JsonResponse({"status": status})
```

Frontend JavaScript polls `/documents/{session_id}/status/` every 3 seconds until `READY`.

### Chat View (SSE Streaming)

```python
# chat/views.py

import requests
from django.http import StreamingHttpResponse

@login_required
def query_stream(request, session_id):
    question = request.POST.get("question")
    jwt_token = request.session.get("jwt_token")

    def stream_from_fastapi():
        with requests.post(
            f"{FASTAPI_BASE}/query/stream",
            json={"question": question, "session_id": session_id},
            headers={"Authorization": f"Bearer {jwt_token}"},
            stream=True,
        ) as r:
            for chunk in r.iter_content(chunk_size=None):
                yield chunk

    return StreamingHttpResponse(
        stream_from_fastapi(),
        content_type="text/event-stream",
    )
```

## Frontend JavaScript (SSE)

```javascript
// static/js/chat.js

function submitQuestion(question, sessionId) {
    const answerDiv = document.getElementById("answer");
    answerDiv.textContent = "";

    const eventSource = new EventSource(`/chat/${sessionId}/stream/?q=${encodeURIComponent(question)}`);

    eventSource.onmessage = function(event) {
        if (event.data === "[DONE]") {
            eventSource.close();
            return;
        }
        answerDiv.textContent += event.data;
    };

    eventSource.onerror = function() {
        eventSource.close();
        answerDiv.textContent += "\n[Connection error]";
    };
}
```

## Authentication

Uses Django's built-in `django.contrib.auth`. On login, Django obtains a JWT from FastAPI and stores it in the session:

```python
# accounts/views.py

def login_view(request):
    if request.method == "POST":
        username = request.POST["username"]
        password = request.POST["password"]

        user = authenticate(request, username=username, password=password)
        if user:
            login(request, user)

            # Get JWT from FastAPI for subsequent API calls
            jwt_response = requests.post(
                f"{FASTAPI_BASE}/auth/token",
                json={"username": username, "password": password},
            )
            request.session["jwt_token"] = jwt_response.json()["access_token"]

            return redirect("documents:upload")
```

## Django Settings (Key)

```python
# config/settings.py

INSTALLED_APPS = [
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.staticfiles",
    "accounts",
    "documents",
    "chat",
]

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": os.environ["POSTGRES_DB"],
        "USER": os.environ["POSTGRES_USER"],
        "PASSWORD": os.environ["POSTGRES_PASSWORD"],
        "HOST": os.environ["POSTGRES_HOST"],
        "PORT": "5432",
    }
}

FASTAPI_SERVICE_URL = os.environ["FASTAPI_SERVICE_URL"]

# Static files
STATIC_URL = "/static/"
STATICFILES_DIRS = [BASE_DIR / "static"]
STATIC_ROOT = BASE_DIR / "staticfiles"  # for collectstatic on Render
```

## What Django Does NOT Do

- No direct S3 access
- No Bytez API calls
- No embedding or FAISS operations
- No MLflow logging
- No business logic for RAG — all delegated to FastAPI

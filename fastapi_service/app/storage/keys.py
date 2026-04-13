"""
S3 key conventions — one place, no magic strings scattered around the codebase.

Layout:
    uploads/{session_id}/original.pdf
    extracted/{session_id}/raw_text.txt
    extracted/{session_id}/cleaned_text.txt
    extracted/{session_id}/validation_report.json
    extracted/{session_id}/textract_response.json   ← raw Textract output preserved
    chunks/{session_id}/chunks.json
    chunks/{session_id}/embeddings.npy
    chunks/{session_id}/faiss.index
    mlflow/artifacts/  (managed by MLflow)
"""


class S3Keys:
    @staticmethod
    def upload_pdf(session_id: str) -> str:
        return f"uploads/{session_id}/original.pdf"

    @staticmethod
    def raw_text(session_id: str) -> str:
        return f"extracted/{session_id}/raw_text.txt"

    @staticmethod
    def cleaned_text(session_id: str) -> str:
        return f"extracted/{session_id}/cleaned_text.txt"

    @staticmethod
    def validation_report(session_id: str) -> str:
        return f"extracted/{session_id}/validation_report.json"

    @staticmethod
    def textract_response(session_id: str) -> str:
        return f"extracted/{session_id}/textract_response.json"

    @staticmethod
    def chunks(session_id: str) -> str:
        return f"chunks/{session_id}/chunks.json"

    @staticmethod
    def embeddings(session_id: str) -> str:
        return f"chunks/{session_id}/embeddings.npy"

    @staticmethod
    def faiss_index(session_id: str) -> str:
        return f"chunks/{session_id}/faiss.index"

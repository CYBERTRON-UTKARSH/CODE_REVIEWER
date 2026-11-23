import json
import os
from typing import Literal
import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
import cohere
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from pydantic import BaseModel, Field

app = FastAPI(title="RAG Code Reviewer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten to your frontend URL once deployed
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

COHERE_API_KEY = os.environ.get("COHERE_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

co_client = cohere.ClientV2(api_key=COHERE_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)


class CohereQueryEmbedding(EmbeddingFunction[Documents]):
    def __init__(self, model: str = "embed-english-light-v3.0"):
        self.model = model

    def __call__(self, input: Documents) -> Embeddings:
        res = co_client.embed(
            texts=list(input),
            model=self.model,
            input_type="search_query",
            embedding_types=["float"],
        )
        return res.embeddings.float_


DB_DIR = os.path.join(os.path.dirname(__file__), "code_rag_db")
chroma_client = chromadb.PersistentClient(path=DB_DIR)
collection = chroma_client.get_or_create_collection(
    name="code_review_standards",
    embedding_function=CohereQueryEmbedding(),
)

SupportedLang = Literal["python", "java", "javascript", "cpp", "go"]


class ReviewRequest(BaseModel):
    code: str = Field(..., min_length=1, max_length=20000)
    language: SupportedLang


class AspectScores(BaseModel):
    security: int = Field(..., ge=0, le=100)
    correctness: int = Field(..., ge=0, le=100)
    readability: int = Field(..., ge=0, le=100)
    efficiency: int = Field(..., ge=0, le=100)


class ReviewResponse(BaseModel):
    has_errors: bool
    errors: list[str]
    security_issues: list[str]
    aspects: AspectScores
    overall_percent: int = Field(..., ge=0, le=100)
    suggested_code: str
    improvement_notes: list[str]


@app.get("/")
def root():
    return {"status": "online", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/review", response_model=ReviewResponse)
def review_code(req: ReviewRequest):
    if not GROQ_API_KEY or not COHERE_API_KEY:
        raise HTTPException(status_code=500, detail="Missing API keys in environment.")

    try:
        results = collection.query(
            query_texts=[req.code],
            n_results=3,
            where={"language": req.language},
        )
        docs = results["documents"][0] if results.get("documents") else []
        metas = results["metadatas"][0] if results.get("metadatas") else []
        rules_context = "\n".join(
            f"- [{m.get('aspect', 'general')} | {m.get('severity', 'info')}]: {d}"
            for d, m in zip(docs, metas)
        )
    except Exception:
        rules_context = ""

    if not rules_context:
        rules_context = "Apply standard industry best practices for this language."

    system_prompt = (
        "You are an expert code reviewer and refactorer. "
        "Return ONLY a single valid JSON object with exact keys:\n"
        "{\n"
        '  "has_errors": boolean,\n'
        '  "errors": [string],\n'
        '  "security_issues": [string],\n'
        '  "aspects": {"security": int (0-100), "correctness": int (0-100), "readability": int (0-100), "efficiency": int (0-100)},\n'
        '  "overall_percent": int (0-100),\n'
        '  "suggested_code": string,\n'
        '  "improvement_notes": [string]\n'
        "}\n"
        "If syntax/logic errors exist, set has_errors=true and list them in errors. "
        "When writing suggested_code, make it shorter, cleaner, safer, and idiomatic with higher aspect scores."
    )

    user_prompt = (
        f"Language: {req.language}\n"
        f"Retrieved Standards:\n{rules_context}\n\n"
        f"Code to Review:\n```{req.language}\n{req.code}\n```"
    )

    try:
        completion = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        data = json.loads(completion.choices[0].message.content)
        return ReviewResponse(**data)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Model review failed: {exc}")
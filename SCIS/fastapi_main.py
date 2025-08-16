from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import requests

# 导入你的本地重写模块
from rewriter import init_model, rewrite_text

# 创建 FastAPI 应用
app = FastAPI()

# 允许前端跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # 前端 Vite 地址
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化模型（应用启动时加载）
@app.on_event("startup")
def load_model():
    init_model()
    print("本地模型已加载")

# ========================
# 接口 1: 文本改写
# ========================

class RewriteRequest(BaseModel):
    text: str

@app.post("/rewrite1")
def rewrite(request: RewriteRequest):
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text is required and cannot be empty")
        result = rewrite_text(request.text)
        print(f"Rewritten: {result}")
        return {"rewritten_text": result}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
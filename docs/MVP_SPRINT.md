# PaperPilot MVP Sprint

## 架构（本轮）

```text
Vue 3 (5173) → Spring BFF (8080) → Django API (8001) → MySQL
                                      ↓
                                 FastAPI (8000)  AI 推理
```

- **Vue 3**：主界面（科技蓝、中文 UI）
- **Spring BFF**：轻量网关，聚合 `/api/**`（上传走 Vite 直连 Django）
- **Django**：REST API + ORM，`ChatMessage.metadata` 存 trace/chunks
- **MySQL**：`docker compose up -d mysql`（未配置时回退 SQLite）
- **FastAPI**：RAG / LangGraph 核心（不变）

## 启动顺序

### 1. MySQL

```bash
docker compose up -d mysql
```

在仓库根目录 `.env` 增加：

```env
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_DATABASE=paperpilot
MYSQL_USER=paperpilot
MYSQL_PASSWORD=paperpilot
```

### 2. Python 依赖 & 迁移

```bash
source .venv/bin/activate
pip install -r requirements.txt
cd django_shell
python manage.py migrate
```

### 3. FastAPI

```bash
# 仓库根目录
uvicorn app.main:app --reload --port 8000
```

### 4. Django

```bash
cd django_shell
python manage.py runserver 8001
```

### 5. Spring BFF

```bash
cd spring-bff
mvn spring-boot:run
```

### 6. Vue

```bash
cd frontend
npm install
npm run dev
```

打开：http://127.0.0.1:5173

## API 一览

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/health/` | Django 健康检查 |
| GET/POST | `/api/sessions/` | 列表 / 新建会话 |
| GET | `/api/sessions/{id}/` | 会话详情 + messages + metadata |
| POST | `/api/chat/ask/` | 问答并持久化 |
| GET | `/api/documents/` | 知识库 PDF 列表 |
| POST | `/api/documents/upload/` | 上传并 reload_kb |

## 遗留 Django 模板

`http://127.0.0.1:8001/` 旧模板仍可用；新产品界面以 Vue 为准。

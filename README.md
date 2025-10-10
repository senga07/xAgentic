# xAgentic - 智能对话系统

xAgentic 是一个基于 LangGraph 的智能对话系统，集成了规划执行、代码生成、搜索工具和 MCP (Model Context Protocol) 支持。该系统采用前后端分离架构，提供流式对话体验和强大的任务执行能力。

## 🚀 主要功能

### 核心特性
- **智能规划执行**: 基于 LangGraph 的循环式规划执行器，能够将复杂任务分解为可执行的步骤
- **流式对话**: 实时流式响应，提供流畅的对话体验
- **代码执行**: 集成安全的 Python 代码执行环境，支持代码生成和运行
- **搜索集成**: 集成 Tavily 搜索 API，提供实时信息检索能力
- **MCP 支持**: 支持 Model Context Protocol，可扩展工具和功能
- **多模型支持**: 支持 Azure OpenAI、阿里云百炼等多种 LLM 提供商
- **记忆管理**: 集成 LangMem 进行对话记忆管理

### 技术架构
- **后端**: FastAPI + LangGraph + LangChain
- **前端**: Express.js + 静态 HTML/JS
- **AI 框架**: LangGraph 状态图 + LangChain 工具链
- **部署**: Docker 支持，一键启动脚本

## 📁 项目结构

```
xAgentic/
├── xAgentic-backend/          # 后端服务
│   ├── api/                   # API 路由
│   │   ├── chat.py           # 聊天 API
│   │   └── mcp.py            # MCP 管理 API
│   ├── cfg/                  # 配置管理
│   │   ├── config.py         # 配置类
│   │   └── setting.py        # 设置管理
│   ├── graph/                # LangGraph 状态图
│   │   └── plan_executor_graph.py  # 规划执行图
│   ├── llm_provider/         # LLM 提供商
│   ├── mcp_/                 # MCP 客户端和管理
│   ├── memory/               # 记忆管理
│   ├── prompt/               # 提示词模板
│   ├── services/             # 服务管理
│   ├── tools/                # 工具集
│   │   ├── code_tools.py     # 代码执行工具
│   │   ├── search_tools.py   # 搜索工具
│   │   └── time_tools.py     # 时间工具
│   ├── utils/                # 工具函数
│   ├── main.py               # 主入口
│   └── pyproject.toml        # Python 依赖
├── xAgentic-frontend/        # 前端服务
│   ├── public/               # 静态文件
│   │   └── index.html        # 主页面
│   ├── server.js             # Express 服务器
│   └── package.json          # Node.js 依赖
├── start.sh                  # 一键启动脚本
└── README.md                 # 项目说明
```

## 🛠️ 安装和配置

### 环境要求
- Python 3.11+
- Node.js 16+
- 有效的 API 密钥（Azure OpenAI、阿里云百炼、Tavily 等）

### 1. 克隆项目
```bash
git clone <repository-url>
cd xAgentic
```

### 2. 配置环境变量
复制示例配置文件并填入你的 API 密钥：
```bash
cp xAgentic-backend/sample.env xAgentic-backend/.env
```

编辑 `.env` 文件，配置以下参数：
```env
# 服务器配置
HOST=0.0.0.0
PORT=8080

# Azure OpenAI配置
AZURE_OPENAI_API_KEY=your_azure_openai_key
AZURE_OPENAI_ENDPOINT=your_azure_endpoint
AZURE_OPENAI_API_VERSION=2024-02-15-preview

# 阿里云百炼 API KEY
DASHSCOPE_API_KEY=your_dashscope_key

# 模型配置
FAST_LLM=dashscope:qwen-plus
STRATEGIC_LLM=dashscope:qwen-max
CODING_LLM=dashscope:qwen3-coder-plus
EMBEDDING=dashscope:text-embedding-v4

# Tavily 搜索 API
TAVILY_API_KEY=your_tavily_key

# LangSmith 配置（可选）
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_PROJECT=xagentic
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_TRACING_V2=true
```

### 3. 安装依赖

#### 后端依赖
```bash
cd xAgentic-backend
pip install -e .
```

#### 前端依赖
```bash
cd xAgentic-frontend
npm install
```

## 🚀 启动服务

### 方式一：一键启动（推荐）
```bash
chmod +x start.sh
./start.sh
```

### 方式二：手动启动

#### 启动后端
```bash
cd xAgentic-backend
python main.py
```

#### 启动前端
```bash
cd xAgentic-frontend
npm start
```

### 访问地址
- **前端界面**: http://localhost:3000
- **后端 API**: http://localhost:8080
- **API 文档**: http://localhost:8080/docs

## 💡 使用说明

### 基本对话
1. 打开浏览器访问 http://localhost:3000
2. 在输入框中输入你的问题或任务
3. 系统会自动进行规划并执行相应步骤
4. 查看流式响应和详细的执行过程

### 支持的任务类型
- **代码生成和执行**: 生成 Python 代码并安全执行
- **信息搜索**: 实时搜索最新信息
- **数据分析**: 处理和分析数据
- **文件操作**: 读取、写入和操作文件
- **复杂任务规划**: 将复杂任务分解为多个步骤执行

### MCP 工具扩展
系统支持通过 MCP (Model Context Protocol) 扩展功能：
1. 在 MCP 配置页面添加新的工具服务器
2. 配置工具的参数和权限
3. 重启服务以加载新工具

## 🔧 API 接口

### 聊天接口
```http
POST /api/chat/stream
Content-Type: application/json

{
  "message": "你的问题或任务",
  "conversation_history": ["历史对话"],
  "mcp_configs": [{"tool": "配置"}]
}
```

### 健康检查
```http
GET /health
```

### MCP 配置管理
```http
GET /api/mcp/configs          # 获取 MCP 配置
POST /api/mcp/configs         # 更新 MCP 配置
```

## 🏗️ 架构详解

### 规划执行流程
1. **任务分析**: 分析用户输入，理解任务需求
2. **制定计划**: 将复杂任务分解为可执行的步骤
3. **循环执行**: 
   - 检查当前步骤的不确定性
   - 如需要，向用户确认
   - 执行步骤并收集结果
4. **生成回复**: 整合所有步骤结果，生成最终回复

### 状态管理
系统使用 LangGraph 的状态图管理对话状态：
- `messages`: 对话消息历史
- `execution_plan`: 执行计划
- `current_step`: 当前执行步骤
- `step_results`: 步骤执行结果
- `final_response`: 最终回复

### 工具集成
- **代码工具**: 基于 langchain-sandbox 的安全代码执行
- **搜索工具**: Tavily API 集成
- **时间工具**: 时间相关操作
- **MCP 工具**: 可扩展的工具协议

## 🔒 安全特性

- **代码沙箱**: 使用 Pyodide 沙箱安全执行 Python 代码
- **输入验证**: Pydantic 模型验证所有输入
- **错误处理**: 完善的异常处理和日志记录
- **CORS 配置**: 安全的跨域请求配置

## 📊 监控和日志

- **统一日志**: 基于 Python logging 的统一日志系统
- **LangSmith 集成**: 可选的链路追踪和监控
- **健康检查**: 服务状态监控接口
- **错误日志**: 详细的错误记录和调试信息

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🆘 常见问题

### Q: 如何更换 LLM 提供商？
A: 修改 `.env` 文件中的模型配置，支持 Azure OpenAI、阿里云百炼等。

### Q: 如何添加自定义工具？
A: 可以通过 MCP 协议添加自定义工具，或在 `tools/` 目录下添加新的工具模块。

### Q: 代码执行是否安全？
A: 是的，系统使用 Pyodide 沙箱环境，限制网络访问和文件系统操作。

### Q: 如何调试问题？
A: 查看 `logs/` 目录下的日志文件，或启用 LangSmith 追踪。

## 📞 支持

如有问题或建议，请：
1. 查看 [Issues](../../issues) 页面
2. 创建新的 Issue 描述问题
3. 提供详细的错误日志和环境信息

---

**xAgentic** - 让 AI 对话更智能，让任务执行更高效！

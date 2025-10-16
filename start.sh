#!/bin/bash

echo "🚀 启动xAgentic对话系统..."

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到Python3，请先安装Python 3.10+"
    exit 1
fi

# 检查Node.js环境
if ! command -v node &> /dev/null; then
    echo "❌ 错误: 未找到Node.js，请先安装Node.js"
    exit 1
fi

# 启动后端服务
echo "📦 安装后端依赖..."
cd xAgentic-backend

# 创建配置文件
echo "🔧 创建配置文件..."
python config_manager.py

# 检查配置
echo "🔍 检查配置..."
python test_config.py

# 安装Python依赖
pip install -e . || pip install -r requirements.txt || echo "请手动安装依赖: pip install fastapi uvicorn langgraph langchain-openai python-dotenv"

# 启动后端
echo "🔧 启动后端服务..."
python main.py &
BACKEND_PID=$!

# 等待后端启动
sleep 3

# 启动前端服务
echo "🎨 安装前端依赖..."
cd ../xAgentic-frontend
npm install

echo "🌐 启动前端服务..."
npm start &
FRONTEND_PID=$!

echo "✅ 服务启动完成！"
echo "📱 前端地址: http://localhost:3000"
echo "🔧 后端地址: http://localhost:8080"
echo "📚 API文档: http://localhost:8080/docs"
echo ""
echo "按 Ctrl+C 停止所有服务"

# 等待用户中断
trap "echo '🛑 正在停止服务...'; kill $BACKEND_PID $FRONTEND_PID; exit" INT
wait

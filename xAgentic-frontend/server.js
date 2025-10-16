const express = require('express');
const cors = require('cors');
const path = require('path');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;
const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8080';

// 中间件
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));


// 代理流式API请求到后端
app.post('/api/chat/stream', async (req, res) => {
  try {
    const response = await axios.post(`${BACKEND_URL}/api/chat/stream`, req.body, {
      responseType: 'stream'
    });
    
    // 设置流式响应头
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    
    // 转发流数据
    response.data.on('data', (chunk) => {
      res.write(chunk);
    });
    
    response.data.on('end', () => {
      res.end();
    });
    
    response.data.on('error', (error) => {
      console.error('流式响应错误:', error);
      res.status(500).end();
    });
    
  } catch (error) {
    console.error('流式API错误:', error.message);
    res.status(500).json({ 
      error: '无法连接到流式聊天服务',
      details: error.message 
    });
  }
});

// 健康检查
app.get('/api/health', async (req, res) => {
  try {
    const response = await axios.get(`${BACKEND_URL}/health`);
    res.json({ 
      frontend: 'healthy', 
      backend: response.data 
    });
  } catch (error) {
    res.status(500).json({ 
      frontend: 'healthy', 
      backend: 'unhealthy',
      error: error.message 
    });
  }
});

// 代理MCP配置相关API
app.get('/api/chat/mcp/configs', async (req, res) => {
  try {
    const response = await axios.get(`${BACKEND_URL}/api/mcp/configs`);
    res.json(response.data);
  } catch (error) {
    console.error('获取MCP配置失败:', error.message);
    res.status(500).json({ 
      error: '获取MCP配置失败',
      details: error.message 
    });
  }
});

app.post('/api/chat/mcp/configs', async (req, res) => {
  try {
    const response = await axios.post(`${BACKEND_URL}/api/mcp/configs`, req.body);
    res.json(response.data);
  } catch (error) {
    console.error('保存MCP配置失败:', error.message);
    res.status(500).json({ 
      error: '保存MCP配置失败',
      details: error.message 
    });
  }
});

// 代理用户反馈API
app.post('/api/chat/feedback', async (req, res) => {
  try {
    const response = await axios.post(`${BACKEND_URL}/api/chat/feedback`, req.body);
    res.json(response.data);
  } catch (error) {
    console.error('处理用户反馈失败:', error.message);
    res.status(500).json({ 
      error: '处理用户反馈失败',
      details: error.message 
    });
  }
});

// 代理流式用户反馈API
app.post('/api/chat/feedback-stream', async (req, res) => {
  try {
    const response = await axios.post(`${BACKEND_URL}/api/chat/feedback-stream`, req.body, {
      responseType: 'stream'
    });
    
    // 设置流式响应头
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    
    // 转发流数据
    response.data.on('data', (chunk) => {
      res.write(chunk);
    });
    
    response.data.on('end', () => {
      res.end();
    });
    
    response.data.on('error', (error) => {
      console.error('流式反馈响应错误:', error);
      res.status(500).end();
    });
    
  } catch (error) {
    console.error('流式反馈API错误:', error.message);
    res.status(500).json({ 
      error: '无法连接到流式反馈服务',
      details: error.message 
    });
  }
});



// 服务静态文件
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.listen(PORT, () => {
  console.log(`前端服务器运行在 http://localhost:${PORT}`);
  console.log(`后端API地址: ${BACKEND_URL}`);
});

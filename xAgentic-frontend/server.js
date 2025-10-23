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

// 添加对multipart/form-data的支持
const multer = require('multer');
const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 5 * 1024 * 1024 // 5MB
  }
});
app.use(upload.any());


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

// 代理fortune分析API
app.post('/api/fortune/analyze', async (req, res) => {
  try {
    console.log('收到fortune分析请求');
    console.log('req.body:', req.body);
    console.log('req.files:', req.files);
    
    // 构建FormData对象
    const FormData = require('form-data');
    const formData = new FormData();
    
    // 添加文本字段
    if (req.body.birthDateTime) {
      // 转换日期时间格式从 ISO 格式 (2025-02-20T13:18) 到后端期望的格式 (2025-02-20 13:18)
      const isoDateTime = req.body.birthDateTime;
      const formattedDateTime = isoDateTime.replace('T', ' ');
      console.log('原始日期时间:', isoDateTime);
      console.log('转换后日期时间:', formattedDateTime);
      formData.append('birthDateTime', formattedDateTime);
    }
    
    // 添加文件字段（可选）
    if (req.files && req.files.length > 0) {
      req.files.forEach(file => {
        console.log('处理文件:', file.fieldname, file.originalname, file.mimetype);
        if (file.fieldname === 'palmPhoto' && file.originalname) {
          formData.append('palmPhoto', file.buffer, {
            filename: file.originalname,
            contentType: file.mimetype
          });
        } else if (file.fieldname === 'facePhoto' && file.originalname) {
          formData.append('facePhoto', file.buffer, {
            filename: file.originalname,
            contentType: file.mimetype
          });
        }
      });
    }
    
    console.log('发送到后端:', `${BACKEND_URL}/api/fortune/analyze`);
    const response = await axios.post(`${BACKEND_URL}/api/fortune/analyze`, formData, {
      headers: {
        ...formData.getHeaders()
      }
    });
    res.json(response.data);
  } catch (error) {
    console.error('fortune分析失败:', error.message);
    console.error('错误详情:', error.response?.data);
    res.status(500).json({ 
      error: 'fortune分析失败',
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

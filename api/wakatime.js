// 用于存储缓存数据
let cache = {
  total_seconds: null,
  timestamp: null,
};

const CACHE_DURATION = 60 * 60 * 1000; // 缓存持续时间：1小时（60分钟）

export default async function handler(req, res) {
  // 允许跨域访问的 HTTP 头
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");

  const { project } = req.query;
  const apiKey = process.env.WAKATIME_API_KEY;

  if (!project || !apiKey) {
    return res.status(400).json({ error: "Project or API key is missing" });
  }

  const range = "all_time"; // 设置统计范围，例如 'last_7_days', 'last_30_days', 'all_time'

  // 检查缓存是否存在且未过期
  if (
    cache.total_seconds &&
    cache.timestamp &&
    Date.now() - cache.timestamp < CACHE_DURATION
  ) {
    console.log(`Serving from cache: ${cache.total_seconds} seconds`);
    return res.status(200).json({ total_seconds: cache.total_seconds });
  }

  // 构造 WakaTime Stats API URL
  const wakatimeUrl = `https://wakatime.com/api/v1/users/current/stats/${range}?api_key=${apiKey}`;

  try {
    // 增加请求重试机制，最多重试 3 次
    const data = await fetchWithRetry(wakatimeUrl, { timeout: 8000 }, 3, 2000);

    console.log("WakaTime API Response:", data); // 输出完整的 API 返回数据

    const projectData = data.data.projects.find((p) => p.name === project);

    if (!projectData) {
      throw new Error(`Project '${project}' not found in the statistics`);
    }

    const totalSeconds = projectData.total_seconds;
    console.log(`Total seconds for project ${project}:`, totalSeconds); // 输出找到的项目编码时间

    // 更新缓存
    cache.total_seconds = totalSeconds;
    cache.timestamp = Date.now();

    // 返回成功响应
    return res.status(200).json({ total_seconds: totalSeconds });
  } catch (error) {
    console.error("Error in WakaTime Stats API:", error);

    // 返回详细的错误响应，避免等待过长
    return res.status(500).json({
      error: error.message,
      details: error.stack || error.toString(),
    });
  }
}

// 添加带有重试和超时处理的 fetch 函数
async function fetchWithTimeout(resource, options = {}) {
  const { timeout = 8000 } = options; // 默认超时设置为 8 秒
  const controller = new AbortController(); // 用于中止请求
  const id = setTimeout(() => controller.abort(), timeout);

  const response = await fetch(resource, {
    ...options,
    signal: controller.signal, // 将中止信号传递给 fetch
  });

  clearTimeout(id); // 成功后清除计时器
  return response;
}

// 带有重试机制的 fetch 函数
async function fetchWithRetry(resource, options = {}, retries = 3, delay = 2000) {
  for (let i = 0; i < retries; i++) {
    try {
      const response = await fetchWithTimeout(resource, options);
      if (!response.ok) {
        const error = new Error(`Failed to fetch data: ${response.status}`);
        error.response = await response.json();
        throw error;
      }
      return await response.json(); // 返回成功的数据
    } catch (error) {
      console.error(`Attempt ${i + 1} failed. Retrying in ${delay / 1000} seconds...`, error);
      if (i === retries - 1) {
        throw error; // 最后一次重试仍然失败，抛出错误
      }
      await new Promise(resolve => setTimeout(resolve, delay)); // 延迟再重试
    }
  }
}

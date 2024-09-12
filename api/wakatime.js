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
    // 超时机制：如果 WakaTime API 请求超过 5000 毫秒（5 秒），则中止请求
    const response = await fetchWithTimeout(wakatimeUrl, { timeout: 5000 });

    if (!response.ok) {
      throw new Error("Failed to fetch WakaTime stats data");
    }

    const data = await response.json();
    console.log("WakaTime API Response:", data); // 输出完整的 API 返回数据

    // 查找指定项目的编码时间
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

    // 返回错误响应，避免等待过长
    return res.status(500).json({ error: error.message });
  }
}

// 添加带有超时处理的 fetch 函数
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

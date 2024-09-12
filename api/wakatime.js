// api/wakatime.js
export default async function handler(req, res) {
  const { project } = req.query;
  const apiKey = process.env.WAKATIME_API_KEY; // 从环境变量中读取 API 密钥

  if (!project || !apiKey) {
    return res.status(400).json({ error: "Project or API key is missing" });
  }

  const wakatimeUrl = `https://wakatime.com/api/v1/users/current/projects/${project}?api_key=${apiKey}`;

  try {
    const response = await fetch(wakatimeUrl);

    if (!response.ok) {
      throw new Error("Failed to fetch WakaTime data");
    }

    const data = await response.json();

    // 允许跨域请求
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

    res.status(200).json(data);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
}

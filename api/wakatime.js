export default async function handler(req, res) {
  const { project } = req.query;
  const apiKey = process.env.WAKATIME_API_KEY;

  if (!project || !apiKey) {
    return res.status(400).json({ error: "Project or API key is missing" });
  }

  const range = 'all_time';  // 设置统计范围，例如 'last_7_days', 'last_30_days', 'all_time'

  // 构造 WakaTime Stats API URL
  const wakatimeUrl = `https://wakatime.com/api/v1/users/current/stats/${range}?api_key=${apiKey}`;

  try {
    const response = await fetch(wakatimeUrl);
    const data = await response.json();

    // 输出完整的 API 返回数据
    console.log("WakaTime API Response:", data);

    if (!response.ok) {
      throw new Error("Failed to fetch WakaTime stats data");
    }

    // 查找指定项目的编码时间
    const projectData = data.data.projects.find(p => p.name === project);

    if (!projectData) {
      throw new Error(`Project '${project}' not found in the statistics`);
    }

    const totalSeconds = projectData.total_seconds;

    // 输出找到的项目编码时间
    console.log(`Total seconds for project ${project}:`, totalSeconds);

    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  } catch (error) {
    console.error("Error in WakaTime Stats API:", error);
    res.status(500).json({ error: error.message });
  }
}

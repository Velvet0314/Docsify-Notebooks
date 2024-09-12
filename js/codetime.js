(function () {
  let defaultWakatimeStatsOptions = {
    text: "该项目的编码时间为：{coding-hours} 小时 {coding-minutes} 分钟",
    project: "", // WakaTime 项目名称
    whereToPlace: "bottom", // 插入位置，"top" 或 "bottom"
  };

  // 从 Vercel API 获取编码时间
  async function fetchCodingTimeFromVercel(project) {
    const url = `https://api.velvet-notes.org/api/wakatime?project=${project}`;
  
    try {
      console.log("Requesting data from:", url);  // 检查请求 URL
      let response = await fetch(url);
  
      if (!response.ok) {
        throw new Error("Failed to fetch data from Vercel API");
      }
  
      const data = await response.json();
      console.log("Received data:", data);  // 输出返回的完整数据
  
      // 检查数据是否存在 total_seconds
      if (data.total_seconds) {
        const totalSeconds = data.total_seconds;
        const hours = Math.floor(totalSeconds / 3600);  // 计算小时数
        const minutes = Math.floor((totalSeconds % 3600) / 60);  // 计算分钟数
  
        return { coding_hours: hours, coding_minutes: minutes };  // 返回对象，包含小时数和分钟数
      } else {
        throw new Error("Invalid data format or total_seconds not found");
      }
    } catch (error) {
      console.error("Error fetching WakaTime data:", error);
      return { coding_hours: "00", coding_minutes: "00" };  // 如果发生错误，返回默认值
    }
  }

  // Docsify 插件主函数
  function wakatimeStatsPlugin(hook, vm) {
    let options = Object.assign({}, defaultWakatimeStatsOptions, vm.config.wakatimeStats);

    // 异步获取编码时间
    let codingTimePromise = new Promise(async (resolve) => {
      const { project } = options;
      const codingTime = await fetchCodingTimeFromVercel(project);  // codingTime 包含 coding_hours 和 coding_minutes
      resolve(codingTime);
    });

    // 在每次页面加载前插入编码时间
    hook.beforeEach(async function (content, next) {
      const { text, whereToPlace } = options;
      const { coding_hours, coding_minutes } = await codingTimePromise;  // 解构获取小时和分钟

      // 更新占位符
      const updatedText = text
        .replace("{coding-hours}", coding_hours)
        .replace("{coding-minutes}", coding_minutes);

      // 插入到文档内容中
      if (whereToPlace === "top") {
        content = updatedText + "\n\n" + content;
      } else {
        content = content + "\n\n" + updatedText;
      }

      next(content);
    });
  }

  // 注册插件到 Docsify
  window.$docsify = window.$docsify || {};
  window.$docsify.plugins = (window.$docsify.plugins || []).concat(wakatimeStatsPlugin);
})();

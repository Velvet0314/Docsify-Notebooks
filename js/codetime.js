(function () {
  let defaultWakatimeStatsOptions = {
    text: "该项目的编码时间为：{coding-hours} 小时",
    project: "", // WakaTime 项目名称
    whereToPlace: "bottom", // 插入位置，"top" 或 "bottom"
  };

  async function fetchCodingTimeFromVercel(project) {
    const url = `https://docsify-notebooks.vercel.app/api/wakatime?project=${project}`;
  
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
  
        // 格式化为 HH:MM 格式
        const formattedTime = `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}`;
        return formattedTime;
      } else {
        throw new Error("Invalid data format or total_seconds not found");
      }
    } catch (error) {
      console.error("Error fetching WakaTime data:", error);
      return "00:00";  // 如果发生错误，返回默认值
    }
  }  

  function wakatimeStatsPlugin(hook, vm) {
    let options = Object.assign({}, defaultWakatimeStatsOptions, vm.config.wakatimeStats);

    let codingTimePromise = new Promise(async (resolve) => {
      const { project } = options;
      const codingHours = await fetchCodingTimeFromVercel(project);
      resolve(codingHours);
    });

    hook.beforeEach(async function (content, next) {
      const { text, whereToPlace } = options;
      const codingHours = await codingTimePromise;
      const updatedText = text.replace("{coding-hours}", codingHours);

      if (whereToPlace === "top") {
        content = updatedText + "\n\n" + content;
      } else {
        content = content + "\n\n" + updatedText;
      }

      next(content);
    });
  }

  window.$docsify = window.$docsify || {};
  window.$docsify.plugins = (window.$docsify.plugins || []).concat(wakatimeStatsPlugin);
})();

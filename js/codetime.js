(function () {
  let defaultWakatimeStatsOptions = {
    text: "该项目的编码时间为：{coding-hours} 小时",
    project: "", // WakaTime 项目名称
    whereToPlace: "bottom", // 插入位置，"top" 或 "bottom"
  };

  async function fetchCodingTimeFromVercel(project) {
    const url = `https://docsify-notebooks.vercel.app/api/wakatime?project=${project}`;

    try {
      let response = await fetch(url);
      if (!response.ok) {
        throw new Error("Failed to fetch data from Vercel API");
      }

      const data = await response.json();
      const totalSeconds = data.data.total_seconds;
      const totalHours = (totalSeconds / 3600).toFixed(2); // 将秒转换为小时
      return totalHours;
    } catch (error) {
      console.error("Error fetching WakaTime data:", error);
      return 0;
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

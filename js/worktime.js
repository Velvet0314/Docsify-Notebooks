(function () {
  // 默认配置选项
  let defaultCommitStatsOptions = {
    text: "该仓库的提交天数为：{commit-days} 天",
    repo: "", // GitHub 仓库的名称，格式为 'username/repo'
    branch: "main", // 分支名称，默认为 main
    commitsPath: "commits.json", // commits.json 文件在仓库中的路径
    whereToPlace: "bottom", // 插入位置，"top" 或 "bottom"
  };

  function calculateCommitDays(commits) {
    const commitDates = new Set();
  
    // 遍历数组中的每个日期，添加到 Set 中去重
    commits.forEach((date) => {
      commitDates.add(date);
    });
  
    console.log("Unique commit dates:", commitDates);
  
    // 返回独立日期的数量
    return commitDates.size;
  }
  

  // Docsify 插件主函数
  function commitStatsPlugin(hook, vm) {
    let options = Object.assign(
      {},
      defaultCommitStatsOptions,
      vm.config.commitStats
    );

    let commitDaysPromise = new Promise(async (resolve, reject) => {
      const { repo, branch, commitsPath } = options;

      if (!repo) {
        console.error("GitHub repository is not specified.");
        return resolve(0);
      }

      const primaryUrl = `https://raw.githubusercontent.com/${repo}/${branch}/${commitsPath}`;
      const fallbackUrl = `https://cdn.jsdelivr.net/gh/${repo}@${branch}/${commitsPath}?t=${new Date().getTime()}`;

      try {
        let response = await fetch(primaryUrl);
        if (!response.ok) {
          throw new Error("Primary URL failed");
        }
        const commits = await response.json();

        // 统计提交天数
        const commitDays = calculateCommitDays(commits);
        resolve(commitDays);
      } catch (error) {
        console.warn("Primary URL failed, trying fallback URL:", error);
        try {
          const response = await fetch(fallbackUrl);
          if (!response.ok) {
            throw new Error("Fallback URL also failed");
          }
          const commits = await response.json();

          // 统计提交天数
          const commitDays = calculateCommitDays(commits);
          resolve(commitDays);
        } catch (error) {
          console.error("Error fetching commits file from GitHub:", error);
          resolve(0); // 如果有错误，返回 0 天
        }
      }
    });

    hook.beforeEach(async function (content, next) {
      const { text, whereToPlace } = options;

      // 等待异步操作完成并获取 commitDays
      const commitDays = await commitDaysPromise;

      console.log(commitDays);

      const updatedText = text.replace("{commit-days}", commitDays);

      // 插入到文档内容中
      if (whereToPlace === "top") {
        content = updatedText + "\n\n" + content;
      } else {
        content = content + "\n\n" + updatedText;
      }

      next(content);
    });
  }

  // 注册插件
  window.$docsify = window.$docsify || {};
  window.$docsify.plugins = (window.$docsify.plugins || []).concat(
    commitStatsPlugin
  );
})();

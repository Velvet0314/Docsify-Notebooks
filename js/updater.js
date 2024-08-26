let defaultDocsifyUpdatedOptions = {
  text: "> Last Modify: {docsify-updated}",
  formatUpdated: "{YYYY}-{MM}-{DD} {HH}:{mm}:{ss}",
  whereToPlace: "bottom",
  repo: "",  // GitHub 仓库的名称，格式为 'username/repo'
  branch: "main",  // 分支名称，默认为 main
  token: "",  // 可选的 GitHub Personal Access Token，用于提高速率限制
  commitsPath: "last_commit_dates.json",  // last_commit_dates.json 文件在仓库中的路径
};

function formatDate(date, format) {
  const map = {
    YYYY: date.getFullYear(),
    MM: ('0' + (date.getMonth() + 1)).slice(-2),
    DD: ('0' + date.getDate()).slice(-2),
    HH: ('0' + date.getHours()).slice(-2),
    mm: ('0' + date.getMinutes()).slice(-2),
    ss: ('0' + date.getSeconds()).slice(-2),
  };
  return format.replace(/YYYY|MM|DD|HH|mm|ss/gi, matched => map[matched]);
}

function lastModifyPlugin(hook, vm) {
  let options = Object.assign({}, defaultDocsifyUpdatedOptions, vm.config.timeUpdater);

  // 调试输出：打印配置选项
  // console.log("Docsify Time Updater Plugin Options:", options);

  let lastCommitDatesPromise = new Promise(async (resolve, reject) => {
    const { repo, branch, commitsPath } = options;

    if (!repo) {
      console.error("GitHub repository is not specified.");
      return resolve({});
    }

    // const fileUrl = `https://raw.githubusercontent.com/${repo}/${branch}/${commitsPath}`;
    const fileUrl = `https://cdn.jsdelivr.net/gh/Velvet0314/Docsify-Notebooks@gh-pages/${commitsPath}`;
    // console.log("Fetching last commit dates from URL:", fileUrl);

    try {
      const response = await fetch(fileUrl);
      if (!response.ok) {
        throw new Error(`Failed to fetch JSON file: ${response.statusText}`);
      }
      const lastCommitDates = await response.json();
      resolve(lastCommitDates);
    } catch (error) {
      console.error("Error fetching last commit dates JSON file:", error);
      resolve({});  // 如果有错误，返回空对象
    }
  });

  hook.beforeEach(async function(content, next) {
    const { text, formatUpdated, whereToPlace } = options;
    let path = vm.route.file;

    if (path.startsWith('/')) {
      path = path.replace(/^\//, '');
    }
    // console.log("Processing file:", path);

    // 等待异步操作完成并获取 lastCommitDates
    const lastCommitDates = await lastCommitDatesPromise;

    // 调试输出：打印获取的 lastCommitDates 对象
    console.log("Last commit dates data:", lastCommitDates);

    if (lastCommitDates[path]) {
      const lastCommitDate = new Date(lastCommitDates[path]);
      const formattedDate = formatDate(lastCommitDate, formatUpdated);
      const updatedText = text.replace('{docsify-updated}', formattedDate);

      // 调试输出：显示插入的时间信息
      console.log(`Inserting last modify date: ${formattedDate} for file: ${path}`);

      // 插入到文档内容中
      if (whereToPlace === 'top') {
        content = updatedText + "\n\n" + content;
      } else {
        content = content + "\n\n" + updatedText;
      }
    } else {
      console.warn(`No modification data found for this file: ${path}`);
    }

    next(content);
  });
}

// 注册插件
window.$docsify = window.$docsify || {};
window.$docsify.timeUpdater = Object.assign(defaultDocsifyUpdatedOptions, window.$docsify.timeUpdater);
window.$docsify.plugins = (window.$docsify.plugins || []).concat(lastModifyPlugin);

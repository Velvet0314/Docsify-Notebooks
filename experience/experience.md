# Experience 心得

这里用来记录我在维护网站时遇见的值得记录的问题，以及我的解决方式。

### Github API 访问速率限制解决

##### 背景故事

由于我的这个网站基于 Docsify，我个人意在打造一个简单易用的文档网站，所有我并没有租赁任何的云服务器，而是选择将其托管到 Github Pages 上进行快速部署。这是一个十分简易的操作方式，而且基本不需要任何的费用开销。

在随后的文档书写中，我产生了这样的想法：我通过观摩其他的优秀网站或是博客，我发现一般会有文档的最后更新日期。这个想法与我之前就想要实现的，如何记录我在网站上的工作时长一拍即合。于是，我开始思考如何实现。

我在 Docsify 的官方社区找到了用于记录最后更改时间的插件。但是这个插件存在着一些问题——显示的是整个仓库的最后更改时间，而不是每个文件各自的最后更改时间。同时，受到自身的水平限制，我无法完成一个自动化的动态工时记录插件的编写。但是，我发现，之前的插件是利用 Github API 来获取仓库的 commit 的状态，那么我应该就可以通过统计 commit 的天数来实现一个简易的工时统计。当然，这个的实现不是特别的困难，尤其是在 GPT 的帮助下。

这就是我本次心得开始的背景故事，现在让我们进入正题。

##### 问题详情

由于开发调试，我需要频繁发送请求。但是未认证的 GitHub API 的访问速率被限制在了 60 times / h。而通过 Github 的 **PAT （personal access tokens）**，通过获取 token 可以将访问速率提高到 5000 times / h。基本上请求不会出现 401、403。但是我在实际操作过程中遇到了另一个问题。由于我的页面没有"后端"，无法将 token 存放到后端再调取，所以我只能将生成的 token 直接放入前端代码中。但是代码需要上传至 Repo，出于隐私保护，Github 会直接自动撤销该 token。也就是说无法设置 token，那么自然也就无法获取高速率的访问了。

##### 解决方案

下面是我的解决方案：**通过 Github Actions 的 workflow 来帮助我进行 token 的暂存与 Github API 的访问，并将返回的结果作为静态资源托管到仓库分支，而后直接访问静态资源进行处理**。

以下是我的具体操作步骤。

###### 1.创建 PAT 用于访问 Github API

<div style="text-align: center;">
 <a href="https://s21.ax1x.com/2024/08/26/pAkEBE4.png" data-lightbox="image-ex" data-title="ex1-1">
  <img src="https://s21.ax1x.com/2024/08/26/pAkEBE4.png" alt="ex1-1" style="width:100%;max-width:900px;cursor:pointer">
 </a>
</div>

在 Settings -> Developer Settings 中找到 PAT，按自己的需求设置后，会生成对应的 token。

###### 2.保存 PAT 生成的 token

<div style="text-align: center;">
 <a href="https://s21.ax1x.com/2024/08/26/pAkEgv6.png" data-lightbox="image-ex" data-title="ex1-2">
  <img src="https://s21.ax1x.com/2024/08/26/pAkEgv6.png" alt="ex1-2" style="width:100%;max-width:900px;cursor:pointer">
 </a>
</div>

进入对应的 Repo，在 Settings -> Secrets and variables 中将 Actions 中用到的 Token 保存在 Repository Secrets 中。

###### 3.编写 workflow 生成静态文件

以下是我个人的示例。

```bash
name: Github Token Proxy

on:
  push:
    branches:
      - main  # 监听 main 分支的 push 事件
  workflow_dispatch:  # 允许手动触发工作流

jobs:
  get_commits_and_deploy:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Fetch Commits
        run: |
          echo "Fetching commits from GitHub API..."
          PAGE=1
          echo "[" > commits.json
          FIRST=true
          while true; do
              RESPONSE=$(curl -s -H "Authorization: token ${{ secrets.REPO_TOKEN }}" \
                                 -H "Accept: application/vnd.github.v3+json" \
                                 "https://api.github.com/repos/${{ github.repository }}/commits?sha=main&per_page=100&page=${PAGE}")
      
              # 提取并处理每个提交的日期
              DATES=$(echo "$RESPONSE" | jq -r '.[].commit.author.date' | cut -d'T' -f1)
      
              # 写入日期到 commits.json 中
              for DATE in $DATES; do
                if [ "$FIRST" = true ]; then
                  FIRST=false
                else
                  echo "," >> commits.json
                fi
                echo "\"$DATE\"" >> commits.json
              done

              # Check if the next page exists
              if [[ $(echo "$RESPONSE" | jq 'length') -lt 100 ]]; then
                break
              fi
              PAGE=$((PAGE + 1))
          done
          echo "]" >> commits.json

      - name: Fetch Last Modification Times
        run: |
          echo "{}" > last_commit_dates.json
          echo "Fetching file list from GitHub API..."
          
          FILES=$(curl -s -H "Authorization: token ${{ secrets.REPO_TOKEN }}" \
                  -H "Accept: application/vnd.github.v3+json" \
                  "https://api.github.com/repos/${{ github.repository }}/git/trees/main?recursive=true" | jq -r '.tree[] | select(.type=="blob") | .path')
          echo "Fetching last commit dates for each file..."
          for file in $FILES; do
            LAST_COMMIT=$(curl -s -H "Authorization: token ${{ secrets.REPO_TOKEN }}" \
                        -H "Accept: application/vnd.github.v3+json" \
                        "https://api.github.com/repos/${{ github.repository }}/commits?path=$file&per_page=1" | jq -r '.[0].commit.committer.date')
            jq --arg file "$file" --arg date "$LAST_COMMIT" \
               '. + {($file): $date}' last_commit_dates.json > tmp.json && mv tmp.json last_commit_dates.json
          done
          cat last_commit_dates.json
      - name: Setup Git config
        run: |
          git config --global user.name "github-actions[bot]"
          git config --global user.email "github-actions[bot]@users.noreply.github.com"
      - name: Switch to gh-pages branch
        run: |
          git checkout --orphan gh-pages
          git reset --hard
      - name: Deploy to GitHub Pages
        run: |
          git add commits.json last_commit_dates.json
          git commit -m "Deploy commits.json and last_commit_dates.json to GitHub Pages"
          git push -f origin gh-pages
```

###### 4.请求访问静态文件

通过 Github 的 CDN 来访问静态资源。

```js
const fileUrl = `https://raw.githubusercontent.com/${repo}/${branch}/${commitsPath}`;
```

国内可通过 jsDelivr 来加速，因为 Github 的 CDN 一般需要上梯子。

```js
const fileUrl = `https://cdn.jsdelivr.net/gh/${username}/${repo}${branch}/${commitsPath}`;
```

需要对提交记录稍微进行处理，防止 jsDelivr 存放过大的文件导致截断。
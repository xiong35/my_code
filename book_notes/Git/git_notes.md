
# Git操作总结

## 配置环境

    ssh-keygen -t rsa -C "2242787668@qq.com"
    vim ~/.ssh/id_rsa.pub
    # add it to the github

    git config --global user.email "2242787668@qq.com"
    git config --global user.name "xiong35"

## 基本操作

- init：初始化一个仓库
- add：暂存更改
- commit：提交更改
- status：查看工作区状态
- diff：查看更改内容
- log：查看提交记录
- reset --hard \<id\>
  - HEAD：当前版本
  - HEAD^^：前两个版本
  - 978r4g：某个版本
- reflog：查看历史commit/pull
- checkout -- \<file\>：让file回到最近一次add/commit的状态(撤销)
- reset HEAD \<file\>：取消file的暂存
- rm：从版本库里删除

## 分支管理

### 基本命令

- switch -c xxx：创建并切换分支（-c：creat）
- branch [-d xxx]：查看分支（-d：del，删除分支）
- merge [-d] xxx：将xxx合并到当前分支

### 解决冲突


# GridView 执行说明

1. 下载并解压整个包，上传 TIMEOrdinalDAG_GridView 文件夹到集群持久化存储。
2. 在 GridView 申请计算节点资源，再打开 Jupyter，选择 spacetime 内核。
3. 打开 GridView_spacetime.ipynb，配置 PROJECT、BASE、WORKERS。
4. 首次将 INSTALL=True 安装依赖；完成后改回 False。若包已加载，重启内核重新配置。
5. 执行测试、checks、小规模模拟及输出检查。
6. 需要完整模拟时将 RUN_FULL=True，执行启动单元，再重复执行监控单元。
7. 完成后 SUMMARIZE=True 生成汇总、图与审计。
8. 可选真实数据：提供原始 CSV，按 Notebook 说明配置 RUN_REAL。

建议起点：单节点 4 CPU、16 GB 内存，无 GPU；资源须服从集群分配，完整计算应根据 pilot 时间申请足够作业时限。
使用当前内核的 sys.executable 安装和执行，不另建或激活其他虚拟环境。
Python 参考后端已与原 R/BiDAG 后端区分；本包默认 Python，非原 R 数值复现。
experiments/requirements.txt 为范围依赖；environment_lock.txt 是原 Linux Python 3.12 环境版本记录。
不需要历史 experiment_outputs.zip 即可从头运行；真实数据原始 CSV 不包含在代码包里。
长计算不保证在 Jupyter/计算资源回收后继续。请按 GridView 本站规定使用批处理调度；本包不假设 Slurm/PBS 的具体配置。
恢复时保留原始参数和输出路径；不要同时启动两个任务写入同一目录。

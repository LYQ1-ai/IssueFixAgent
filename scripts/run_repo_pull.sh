cd /home/lyq/PycharmProjects/CodeAgentRL

source ~/anaconda3/etc/profile.d/conda.sh

conda activate CodeAgentRL

python scripts/batch_repo_pull.py --cache-dir /data/repo_cache --workers 4 --retries 2
# 中途失败可原样重跑续拉（已成功的自动跳过）
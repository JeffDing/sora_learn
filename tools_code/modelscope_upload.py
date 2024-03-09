from modelscope.hub.api import HubApi

YOUR_ACCESS_TOKEN = '8d4596a6-cd52-4557-84b0-85af535a8244'
# 请注意ModelScope平台针对SDK访问和git访问两种模式，提供两种不同的访问令牌(token)。此处请使用SDK访问令牌。


api = HubApi()
api.login(YOUR_ACCESS_TOKEN)
api.push_model(
    model_id="JeffDing/TCM_DEMO", 
    model_dir="/root/temp/ft-medqa/merged" # 本地模型目录，要求目录中必须包含configuration.json
)
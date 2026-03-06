git add runtime.txt requirements.txt setup.sh .streamlit/config.toml
git commit -m "Fix dependency conflicts with exact version constraints and setup script"
git push#!/bin/bash

# 设置环境变量来解决 protobuf 兼容性问题
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# 升级 pip
pip install --upgrade pip

# 安装依赖
pip install -r requirements.txt

echo "Installation completed successfully!"

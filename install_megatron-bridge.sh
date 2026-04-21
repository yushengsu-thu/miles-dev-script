# RUN pip install git+https://github.com/yushengsu-thu/Megatron-Bridge.git@merged-megatron-0.16.0rc0-miles --no-deps --no-build-isolation
# RUN pip install megatron-energon --no-deps
# RUN pip install multi-storage-client --no-deps

# pip install -e /home/ubuntu/yushengsu/Megatron-Bridge --no-deps --no-build-isolation
# pip install megatron-energon --no-deps
# pip install multi-storage-client --no-deps

cd /home/ubuntu/yushengsu/Megatron-Bridge
git config --global --add safe.directory /home/ubuntu/yushengsu/Megatron-Bridge
git submodule update --init
pip install -e . --no-deps --no-build-isolation --break-system-packages

## test if the installation is successful
python3 -c "from megatron.core.package_info import __version__; print(__version__)"
docker run --rm -it \
  --pull always \
  --gpus all \
  --privileged \
  --network=host \
  --ipc=host \
  --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -v $HOME/.ssh:/root/.ssh \
  -v $HOME:$HOME \
  --shm-size 128G \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --ulimit nofile=1048576:1048576 \
  -w /home/ubuntu/yushengsu \
  --name yusheng_miles \
  radixark/miles:dev \
  /bin/bash
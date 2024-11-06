#!/bin/bash

cd ./cvat || exit

containers=$(docker ps --format "{{.Names}}" | wc -l)

expected_containers=$(docker compose config --services | wc -l)

if [ "$containers" -lt "$expected_containers" ]; then
  echo "CVAT containers are not running. Starting them now..."
  docker compose up -d
else
  echo "All CVAT containers are up and running!"
fi

cd ../ || exit
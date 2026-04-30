#!/bin/bash
find src -type d | while read dir; do
    touch "$dir/__init__.py"
done
echo "Done"
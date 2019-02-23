#!/usr/bin/env bash

./render.py $1
cd out
ls -1 *.ppm | sed 's/.ppm//' | xargs -I % convert %.ppm %.png

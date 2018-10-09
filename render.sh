#!/usr/bin/env bash

./render.py
cd out
ls -1 *.ppm | sed 's/.ppm//' | xargs -I % convert %.ppm %.png

FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn
COPY . .
# warm the solver's pruning tables at build time so the first request is fast
RUN python -c "import kociemba; kociemba.solve('DRLUUBFBRBLURRLRUBLRDDFDLFUFUFFDBRDUBRUFLLFDDBFLUBLRBD')"
ENV PORT=7860 CUBE_MODE=decode
EXPOSE 7860
CMD gunicorn -w 2 --threads 4 --timeout 120 -b 0.0.0.0:${PORT} wsgi:app

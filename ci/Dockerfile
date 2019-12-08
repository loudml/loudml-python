FROM debian:stretch-slim AS pyenv

RUN apt-get update \
    && apt-get install -y curl \
    && apt-get install -y gcc git make zlib1g-dev libffi-dev libssl-dev libbz2-dev libncursesw5-dev libgdbm-dev liblzma-dev libsqlite3-dev tk-dev uuid-dev libreadline-dev \
    && apt-get purge -y

RUN useradd -m user

WORKDIR /home/user

USER user

RUN git clone git://github.com/yyuu/pyenv.git .pyenv

ENV HOME  /home/user
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH

RUN pyenv install 3.6.8 \
    && pyenv install 3.7.4
RUN pyenv global 3.6.8
RUN pyenv rehash

FROM debian:stretch-slim

COPY --from=pyenv /home/user/.pyenv /home/user/.pyenv

ENV HOME  /home/user
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH

RUN apt-get update \
    && apt-get install -y wget \
    && apt-get purge -y

ENV DOCKERIZE_VERSION v0.6.1
RUN wget https://github.com/jwilder/dockerize/releases/download/$DOCKERIZE_VERSION/dockerize-linux-amd64-$DOCKERIZE_VERSION.tar.gz \
    && tar -C /usr/local/bin -xzvf dockerize-linux-amd64-$DOCKERIZE_VERSION.tar.gz \
    && rm dockerize-linux-amd64-$DOCKERIZE_VERSION.tar.gz

RUN useradd user

WORKDIR /app

RUN chown -R user:user /app /home/user

RUN pip install tox
USER user
RUN pyenv local 3.6.8 3.7.4

CMD ["tox"]

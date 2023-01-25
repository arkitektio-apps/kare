# kare

[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://pypi.org/project/kare/)
![Maintainer](https://img.shields.io/badge/maintainer-jhnnsrs-blue)

kare is just CARE enabled ass an arkitekt app

### Inspiration

Mikro is the client app for the mikro-server, a graphql compliant server for hosting your microscopy data. Mikro tries to
facilitate a transition to use modern technologies for the storage and retrieval of microscopy data. It emphasizes the importance
of relations within your data and tries to make them accessible through a GraphQL Interface.

### Installation

In order to run kare you can either use the port automatic installer (which installs kare as a docker container with managed dependencies
into your arkitekt setup) or download this repository and run 

```bash
python app.py
```

### Features

- Ability to train CARE model through the arkitekt web api
- Ability to infer images through the arkitekt web api

### Prerequisits

You need a fully configured arkitekt server to connect to.


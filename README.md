# predict

Dépôt Data d'EnerVision. Quatre exécutables indépendants qui ne communiquent
que par des artefacts — une table, des partitions, un registre de modèles :

| Service | Rôle | Produit |
|---|---|---|
| `collector` | interroge l'API Mock IoT | tables `site`, `mesure`, `ingestion_etat` |
| `etl` | transforme les mesures en variables | colonnes déduites de `mesure`, partitions parquet |
| `training` | apprend, arbitre, promeut | versions MLflow, table `modele` |
| `serving` | sert les prévisions | rien — il calcule et rend |

**Aucun service n'importe le code d'un autre.** Si `etl` faisait
`from collector.sink import write`, il n'y aurait plus quatre services mais un
monolithe à quatre dossiers. Chaque image ne copie que la bibliothèque
partagée et son propre code, et `tests/test_architecture.py` le vérifie à
chaque exécution de la suite.

## Sommaire

- [Démarrer](#démarrer)
- [Les contrats entre services](#les-contrats-entre-services)
- [Les points d'entrée](#les-points-dentrée)
- [Configuration](#configuration)
- [Données](#données)
- [Les variables](#les-variables)
- [Le modèle](#le-modèle)
- [Le service d'inférence](#le-service-dinférence)
- [Exploitation](#exploitation)
- [Arborescence](#arborescence)
- [Dépendances](#dépendances)

## Démarrer

```bash
uv sync                 # bibliothèque partagée + quatre services
cp .env.example .env
make check              # ruff + pytest
```

`uv` gère un workspace : `pyproject.toml` déclare les membres, chaque service
a le sien avec ses propres dépendances. `serving` ne déclare ni pandas-SQL, ni
scikit-learn, ni httpx — un import de l'ETL glissé dans son code ne
s'installerait pas dans son image.

La chaîne complète sur une journée, sans Docker :

```bash
make run-day DATE=2026-09-02 FV=v1     # collect → etl → train
make help                              # toutes les cibles
```

Deux dépendances ne sont pas fournies ici : **l'API Mock IoT**
(`MOCK_API_URL`) et **la base TimescaleDB**, portée par le dépôt `api`, qui
détient le schéma figé v1.0 (`DATABASE_URL`).

## Les contrats entre services

Chaque frontière est un **emplacement** plus un **schéma**, jamais un appel de
fonction.

| Producteur | Artefact | Consommateur |
|---|---|---|
| `collector` | table `site`, depuis `GET /api/v1/sites` | `mesure` (clé étrangère) |
| `collector` | table `mesure`, colonnes de la source | `etl` |
| `etl` | table `mesure`, colonnes déduites | API EnerVision |
| `etl` | `features/v1/dt=.../part-*.parquet` | `training`, `serving` |
| `training` | `models:/enervision_xgboost@champion` | `serving` |
| `training` | table `modele`, version active | API EnerVision |

Les chemins sont construits par `predict_common.paths`, les tables déclarées
dans `predict_common.db`, les schémas dans `predict_common.schemas`. Le
producteur valide à l'écriture, le consommateur à la lecture : validée d'un
seul côté, une rupture de contrat serait imputée trois étapes plus loin, ou
découverte après publication.

`serving` ne produit aucun artefact. Archiver les prévisions est le métier de
l'API EnerVision. `modele`, en revanche, est écrite ici : elle dit quel modèle
est en service, et le seul geste qui change cette réponse est la promotion
d'un alias.

### Qui écrit quoi dans `mesure`

| | Colonnes écrites | Sur conflit |
|---|---|---|
| `collector` | les sept mesures, `null_reasons`, `data_quality` | `DO NOTHING` |
| `etl` | `null_reasons`, `data_quality`, `consumption_kw_imputed`, `imputation_method`, `quality_source` | `DO UPDATE`, et seulement si une valeur change |

Le collecteur n'écrase rien : une recollecte ne défait pas la transformation
déjà faite. L'ETL repose, parce que corriger une règle doit atteindre
l'historique déjà traité — mais son `DO UPDATE` ne liste que les colonnes
déduites, et ne se déclenche que si l'une d'elles diffère réellement.

`quality_source` décrit le traitement, pas la mesure : `source` tant que l'ETL
n'est pas passé, `etl` ensuite. Sans elle, une journée fraîchement collectée
afficherait 0 % de mesures dégradées et paraîtrait parfaite.

### `ingestion_etat`

`max(mesure.inserted_at)` ne distingue pas « la collecte a tourné et il n'y
avait rien » de « la collecte n'a pas tourné » : la panne la plus grave est la
plus discrète. Les deux points d'entrée du collecteur reposent donc leur état,
une ligne par site :

| Colonne | Ce qu'elle montre |
|---|---|
| `last_attempt_at` / `last_success_at` | égales : tout va bien. Écartées : la collecte tourne et échoue. Figées : elle ne tourne plus |
| `last_data_lag_s` | âge de la mesure servie par la source |
| `consecutive_failures` | l'à-coup contre la panne installée |
| `source` | `poller` ou `backfill` |

L'écriture de l'état ne peut jamais interrompre la boucle.

## Les points d'entrée

```bash
python -m collector --start 2026-08-01 --end 2026-09-01   # rattrapage
python -m collector --date 2026-09-02                     # une journée
python -m collector.poller                                # au fil de l'eau
python -m collector.datasets                              # historique de référence
python -m etl       --date 2026-09-02 --feature-version v1
python -m training  --feature-version v1 --history-days 90
python -m training.drift                                  # surveillance
uvicorn serving.api:app
```

Deux propriétés sont tenues absolument.

**Idempotence.** Relancer `--date 2026-09-02` reproduit la journée, il ne la
double pas. En base, chaque écriture est un `ON CONFLICT` sur `(site_id, ts)`.
Sur le stockage objet, la partition est écrite dans un répertoire de travail
voisin puis substituée en bloc : un lecteur ne voit jamais une partition à
moitié écrite. Le rejeu après incident est le mode d'exploitation normal.

**Aucun état en mémoire entre services.** Le seul état, ce sont `mesure`, les
partitions et le registre MLflow.

`--start/--end` nomme une période, ce que fait un analyste ; `--date/--days`
nomme une journée et sa profondeur, ce que fait un ordonnanceur. Les deux
formes s'excluent.

## Configuration

Un seul endroit : `conf/`. `base.yaml` décrit la chaîne,
`conf/{PREDICT_ENV}.yaml` décrit ce qu'une machine en change, et la
superposition se fait **clé par clé** — surcharger
`training.params.n_estimators` ne fait pas disparaître les autres
hyperparamètres.

Aucun secret dans un fichier versionné : les valeurs sensibles s'écrivent
`${VARIABLE}` et se résolvent dans l'environnement au chargement. Un
`${VARIABLE}` sans repli est obligatoire et fait échouer le démarrage, en
nommant la variable.

Un réglage qui n'apparaît pas dans `conf/` n'existe pas, même écrit dans
`.env`.

## Données

### Poser l'historique de référence

Deux années horaires par site, fournies avec la source. C'est la **seule**
origine possible de l'historique d'apprentissage : `GET /api/v1/readings` ne
remonte qu'à 48 heures et répond des mesures nulles au-delà, sans erreur.

Ces fichiers ne sont ni dans le dépôt ni dans l'image — 24 Mo de CSV figés sur
chaque clone et chaque couche publiée, pour une donnée lue une seule fois.

```bash
make storage                    # démarre Garage, pose seaux et clé
make datasets-push              # dépose les sept CSV
python -m collector.datasets    # les charge dans mesure
```

Ou vers n'importe quelle racine :

```bash
uv run python deploy/push-datasets.py datasets s3://enervision-datasets
```

Leur seau est distinct de celui des partitions : un rejeu de l'ETL réécrit les
partitions, et deux années de mesures que rien ne sait régénérer n'ont pas à
partager un préfixe avec ce qui s'efface.

Seuls les `SITE00*.csv` sont lus. `all_sites_combined.csv` porte les mêmes
lignes pour 11 Mo de plus.

### Rattraper l'historique

```bash
python -m collector --start 2026-08-01 --end 2026-09-01
python -m collector --start 2026-08-01 --end 2026-09-01 --site SITE001 --limit 500
```

`--limit` borne les mesures par requête ; le plafond de la source est **1000**,
refusé ici plutôt que découvert dans une réponse 422. Sans `--site`, les sept
sites du référentiel sont collectés.

### La grille à la minute

`mesure` est une grille : au plus une ligne par site et par minute,
`(site_id, ts)` en clé.

| Route | Ce qu'elle sert | Horodatage |
|---|---|---|
| `.../{id}/current` | l'instantané | l'instant de l'appel, sub-seconde |
| `/api/v1/readings` | l'historique, par pas de 30 min | aligné à la minute |

La cadence à la minute vient donc du **poller**, pas du rattrapage : rattraper
un mois donne 48 points par jour et par site.

`sink.snap_to_grid` ramène l'horodatage sur la grille au moment de l'écriture,
et seulement là — le retard d'ingestion se mesure sur l'horodatage brut. Sans
ce calage, une minute couverte par les deux routes entrerait en base sous deux
clés, soit 48 doublons par jour et par site.

### Le rattrapage au démarrage

Le poller comble ce qui manque avant d'entrer dans sa boucle, par
`/api/v1/readings`. La profondeur est déduite de la dernière mesure de chaque
site, donc de la durée réelle de la coupure.

```bash
python -m collector.poller                  # rattrape puis boucle (défaut)
python -m collector.poller --no-catch-up    # boucle seule
python -m collector --catch-up              # rattrapage seul
```

| État de la base | Ce qui est collecté |
|---|---|
| vide | `collector.catch_up_days` journées |
| arrêt de 5 jours | les 5 journées, et celle de la reprise |
| redémarrage à chaud | la journée courante seulement |

L'échec du rattrapage n'empêche pas la boucle de démarrer : collecter le
présent vaut mieux que le passé, qui se relance à la main.

### Valeurs manquantes

Une valeur nulle n'est pas une valeur qui manque : c'est un capteur qui dit
qu'il est tombé. Aucune n'est filtrée, aucune n'est écrasée.

| Étage | Ce qu'il garantit |
|---|---|
| `collector.sink` | la valeur nulle est écrite telle quelle, avec ses motifs |
| `etl.quality` | aucune valeur nulle sans motif, et un `data_quality` que les données ne contredisent pas |
| `etl.impute` | la valeur reconstruite va dans `consumption_kw_imputed`, jamais dans `consumption_kw` |
| `etl.exclude` | une mesure ni brute ni reconstruite est écartée des variables, avec sa cause |

Une valeur encadrée par deux voisines connues est interpolée sur le temps, une
valeur qui n'a qu'un passé est reportée (`locf`), une valeur sans passé ni
futur reste nulle et sort des variables. `data_quality` retient la plus sévère
des deux qualifications, celle de la source et celle que les données imposent.

La couche des variables expose `imputed_ratio` : la part de chaque heure
reconstruite. `training.max_imputed_ratio` décide quoi en faire.

## Les variables

La source produit à la minute, le modèle prédit à l'heure. Le changement de
pas a lieu dans `etl.features`, sur une **grille horaire complète, trous
compris** : un décalage est une position sur cette grille, `lag_24h` est la
veille à la même heure ou rien du tout. Une heure dont un décalage manque est
retirée plutôt que complétée.

La moyenne glissante est décalée d'un pas avant d'être calculée — sans quoi
elle contiendrait la cible de l'heure courante, et le modèle lirait la réponse
dans la question.

Produire une journée demande donc de lire les précédentes : la profondeur est
déduite du plus long décalage (neuf jours pour `lag_168h`), et l'ETL le fait
seul.

### Ce que la partition publie, ce que le modèle apprend

| Colonne | Publiée | Apprise |
|---|---|---|
| `hour`, `day_of_week`, `is_weekend` | oui | oui |
| `lag_1h`, `lag_24h`, `lag_168h`, `roll_mean_24h` | oui | oui |
| `temperature_celsius` | oui | **non** |
| `data_quality`, `imputed_ratio` | oui | non — elles décrivent la ligne |

Le service d'inférence ne connaît pas la météo des heures qu'il prédit : un
modèle entraîné sur la température apprendrait des séparations qu'il ne peut
plus emprunter en production. `published_columns` dit ce que l'ETL écrit,
`feature_columns` ce que le modèle consomme ; la seconde est un sous-ensemble
de la première.

Retirer une colonne de `published_columns` change le contrat de la couche,
donc impose une nouvelle `feature_version`. En retirer une de
`feature_columns` ne change que le modèle, que MLflow versionne déjà.

### Changer de version

`etl.lag_hours`, `etl.rolling_window_h` et `etl.resample_rule` décident des
colonnes produites. Les changer se fait en publiant sous une nouvelle version :

```bash
python -m etl --date 2026-09-02 --feature-version v2
```

Les deux versions coexistent sous deux préfixes. Un modèle entraîné sur `v1`
reste reproductible après la sortie de `v2`.

## Le modèle

### Du training au serving

L'entraînement n'écrit pas un `model.pkl` : un pickle transporte un objet
Python et rien d'autre — ni les colonnes attendues, ni leur ordre, ni la
version des bibliothèques. Il logue dans MLflow avec la **signature** et
l'environnement, et chaque version enregistrée porte des tags qui la décrivent
(`feature_version`, `train_window`, `sites`, ses métriques, `residual_std`).

Un alias désigne un rôle et se déplace ; un tag décrit la version et ne bouge
plus. Le service résout un alias au démarrage — un entraînement produit un
`challenger`, jamais un `champion` : promouvoir est une décision
d'exploitation.

```bash
python -m training --feature-version v1 --promote   # apprendre puis promouvoir
python -m training --promote-version 7              # promouvoir sans réapprendre
docker compose restart serving                      # relire l'alias
```

`--promote` déplace l'alias **et** inscrit la version dans `modele`, en
éteignant les autres, dans une seule transaction. `DATABASE_URL` est donc
obligatoire sous cette option, et vérifiée **avant** l'apprentissage.

### Le banc d'arbitrage

Chaque run mesurait sa qualité sur son propre bloc de test : deux
entraînements espacés d'une semaine produisaient deux MAE que rien
n'autorisait à comparer. Le banc est une fenêtre de journées **retirée de
l'apprentissage**, sur laquelle tout candidat est réévalué — le modèle appris,
les familles concurrentes, les baselines naïves. Ses mesures sont préfixées
`arbitrage_`, sa fenêtre journalisée sous `arbitrage_window`.

Glissant par défaut (`training.arbitration.days`), il suffit à comparer les
candidats d'un même run. Le figer les rend comparables d'un entraînement à
l'autre :

```yaml
training:
  arbitration:
    start: "2026-08-01"
    end: "2026-08-14"
```

Les persistances — la consommation d'il y a 1 h, 24 h, 168 h — sont mesurées
sur le même banc ; la meilleure des trois sert de barre (`naif_mae`).

### La règle de promotion

`--promote` passe par une règle, dans cet ordre :

1. **battre la persistance** — un modèle qui ne bat pas la recopie de la veille
   ne paie ni son entraînement, ni son registre, ni sa surveillance ;
2. **être comparable** — deux mesures faites sur des bancs différents ne se
   comparent pas ;
3. **ne pas dégrader** — `training.promotion.margin`, zéro par défaut.

Un refus sort en **code 3**, distinct du 1 : le modèle a été appris et
enregistré en challenger, seule sa mise en service a été refusée. Il n'y a
rien à réparer, il y a quelque chose à lire.

```bash
python -m training --feature-version v1 --promote --force   # passer outre
```

`--force` porte sur ce seul geste et n'assouplit rien pour les suivants. Une
version enregistrée avant l'existence du banc n'en porte aucune mesure :
la promouvoir demande `--force`.

### Opposer les familles

```bash
make challenge HISTORY_DAYS=90
python -m training --challenge --feature-version v1 --history-days 90
```

Chaque famille de `training.candidates`, plus XGBoost, plus chaque
persistance, est ajustée puis mesurée sur le banc. Chacune a son run MLflow
tagué `challenge`, et le classement sort dans le journal :

```
classement sur le banc 2026-08-21/2026-09-03 (14 journée(s), glissant, 672 heure(s)) :
  1. foret-aleatoire    MAE     2.30 kW   RMSE     2.93   R2  1.000
  2. xgboost            MAE     2.31 kW   RMSE     2.93   R2  1.000
  3. ridge              MAE     3.16 kW   RMSE     3.97   R2  0.999
  4. persistance-24h    MAE     4.57 kW   RMSE     5.84   R2  0.999
```

Le challenge **n'enregistre rien et ne promeut rien** : un classement dit
quelle famille convient, mettre en service est une autre décision, qui passe
par `--promote-version`.

### Surveiller la dérive

```bash
python -m training.drift                              # les 7 derniers jours
python -m training.drift --since 2026-08-26 --until 2026-09-01
```

Chaque exécution rejoue le modèle **en service** — celui de l'alias
`champion` — sur les mesures arrivées depuis, et publie un run dans
l'expérience `enervision-drift`.

L'écart est calculé **à un pas** : chaque heure est prédite depuis ses
décalages réels, la même tâche que celle mesurée à l'entraînement. Ce n'est
donc pas le nombre du dashboard, qui compare des prévisions récursives à
48 heures.

Chaque run mesure aussi **chaque site** — une MAE unique sur le parc est
dominée par le plus gros consommateur — et publie `SITE00X_mae` avec un tag
`verdict_SITE00X`.

| Réglage | Défaut | Ce qu'il décide |
|---|---|---|
| `monitoring.mae_alert_ratio` | **1.5** | rapport toléré entre l'erreur mesurée et celle du jeu de test |
| `monitoring.window_days` | 7 | profondeur évaluée sans borne donnée |
| `monitoring.min_rows` | 168 | en deçà, pas de verdict d'ensemble |
| `monitoring.min_rows_per_site` | 24 | en deçà, pas de verdict de site |

Le seuil est un **rapport**, pas des kilowatts : un écart de 20 kW n'a pas le
même sens sur un bureau de 200 kW et sur une usine de 1000.

| Code de sortie | Sens |
|---|---|
| `0` | stable, ou trop peu d'heures pour conclure |
| `1` | la surveillance elle-même a échoué |
| `2` | **dérive**, d'ensemble ou sur un seul site |

Le `2` est distinct du `1` à dessein : les confondre ferait chercher un
problème d'infrastructure là où le modèle a vieilli.

## Le service d'inférence

```bash
uvicorn serving.api:app --reload
```

`CONTRACT_VERSION` vaut **1.4.0**.

| Méthode | Route | Clé exigée |
|---|---|---|
| GET | `/health` | non — sonde de vivacité de l'hébergeur |
| GET | `/ready` | oui |
| POST | `/api/v1/predict` | oui |
| GET | `/api/v1/sites` | oui |
| POST | `/api/v1/simulate/spike/{site_id}` | oui |

`/health` ne consulte pas le registre : l'y lier ferait redémarrer un service
sain chaque fois que MLflow tousse. `/ready` consulte, et répond **toujours
200**, y compris quand rien n'est prêt — son consommateur est l'API métier,
qui doit pouvoir dire *pourquoi* la prévision manque.

Trois erreurs, qui ne désignent pas la même panne : **404** le site n'a pas
d'historique récent, **422** la requête sort des bornes du contrat, **503** le
registre n'a résolu aucun modèle.

### Clé de service

Le service relaie `POST /api/v1/simulate/spike`, qui **écrit sur la source**.
L'API métier protège la même opération derrière le rôle `writer` : un service
ouvert rendrait ce contrôle contournable.

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Sans `SERVING_API_KEY`, le service **refuse de démarrer**.
`SERVING_AUTH_ENABLED=false` ouvre les routes pour un poste de développement —
journalisé en avertissement à chaque démarrage, jamais déployé.

### Sur quoi la prévision s'appuie

`history_end` donne la dernière heure observée dont la récurrence est partie,
et `feature_lag_hours` son âge. Une prévision calculée sur des variables
vieilles de trois jours n'est pas fausse, elle est **aveugle**.

`lower_bound_kw` et `upper_bound_kw` encadrent chaque point à 95 %. La
demi-largeur vaut `1.96 × residual_std × √pas`, où `residual_std` est posé en
tag sur la version enregistrée — la dispersion suit le modèle, promouvoir une
autre version change les intervalles sans redéployer. Une version sans ce tag
est servie sans bornes plutôt qu'avec une bande inventée.

La largeur croît en **racine** du pas : deux erreurs successives s'additionnent
en variance, pas en écart-type. C'est une approximation optimiste — elle
suppose les erreurs indépendantes — et **globale** : un seul écart-type pour
sept sites allant de 90 à 580 kW.

### Contrat OpenAPI

La source de vérité, ce sont les DTO de `services/serving/src/serving/schemas.py`.
Le fichier gelé qui fait foi entre les équipes est
`enervision/docs/contracts/openapi-predict.json`.

```bash
python scripts/export_openapi.py > ../docs/contracts/openapi-predict.json
```

Les docstrings de `serving/schemas.py` et `serving/api.py` alimentent les
descriptions du contrat : les reformater change le JSON gelé et fait échouer
le job `contract-drift`. C'est la raison pour laquelle ruff est configuré à
88 colonnes et non 80, et pour laquelle ces deux fichiers gardent leurs
docstrings d'origine.

## Exploitation

### Orchestration

Les services ne s'appellent pas entre eux : un ordonnanceur externe les
enchaîne, et c'est aujourd'hui le `Makefile`.

```bash
make run-day DATE=2026-09-02 FV=v1              # collect → etl → train
make backfill START=2026-08-01 END=2026-09-01   # rattrapage d'un mois
```

Tant que l'enchaînement tient en trois commandes séquentielles, un DAG
n'apporte qu'une infrastructure de plus. Ces cibles se transposeront telles
quelles : elles sont déjà des processus indépendants, datés et idempotents.

### Pile Docker

```bash
docker compose up -d mlflow serving
docker compose up -d poller                                   # au fil de l'eau
docker compose --profile jobs run --rm collector --date 2026-09-02
docker compose --profile jobs run --rm etl --date 2026-09-02
docker compose --profile jobs run --rm training --feature-version v1
docker compose --profile jobs run --rm drift
```

Les traitements datés sont derrière le profil `jobs` : sans lui, un
`docker compose up` déclencherait une collecte, une transformation, un
entraînement et une surveillance à chaque démarrage.

`drift` partage l'image de `training` avec un entrypoint différent, et a
vocation à tourner tous les jours — un indicateur de dérive produit à la main
vieillit en silence, ce qui est pire que pas d'indicateur.

### Stockage objet

Le disque local suffit au développement. Pour éprouver le déploiement, un nœud
Garage est disponible derrière le profil `garage` :

```bash
make storage        # démarre garage, pose les seaux, affiche la clé
```

Reporter la clé dans `.env` (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`),
puis poser `PREDICT_STORAGE_ROOT=s3://enervision` et
`AWS_ENDPOINT_URL=http://garage:3900`.

Deux seaux : `enervision` porte les partitions, que l'ETL réécrit ;
`enervision-datasets` porte l'historique de référence, figé.

Les quatre services traversent le même code : `predict_common.io` résout un
chemin nu vers le disque et une URI `s3://` vers le stockage objet. Passer du
poste au déploiement ne change qu'une valeur de `conf/`.

### MLflow

Backend store SQLite dans le volume `mlruns`. Une base et non un file store :
le Model Registry, qui donne son alias au modèle servi, exige un backend
relationnel.

```bash
mlflow server \
  --backend-store-uri sqlite:///mlruns/mlflow.db \
  --artifacts-destination ./mlruns/artifacts \
  --host 127.0.0.1 --port 5000
```

### TimescaleDB

`mesure` est la couche brute de la chaîne, lue aussi par l'API EnerVision. La
base est fournie par le dépôt `api`, qui détient le schéma figé v1.0 — la
dupliquer ici donnerait deux vérités sur la même table.

`DATABASE_URL` n'est pas optionnelle : le collecteur et l'ETL refusent de
démarrer sans, en nommant la variable.

**Migration.** L'ETL repose `consumption_kw_imputed` et `imputation_method`,
absentes du schéma v1.0. La migration vit dans le dépôt de la base
(`initdb/03_mesure_imputation.sql`), est idempotente, et doit être appliquée à
la main sur une base déjà démarrée.

**Référentiel des sites.** `mesure.site_id` référence `site` : une mesure dont
le site n'y figure pas est rejetée. Le collecteur synchronise ce référentiel
depuis la source, avant toute écriture de mesure.

## Arborescence

| Chemin | Rôle |
|---|---|
| `conf/base.yaml` | chemins, colonnes, hyperparamètres — source de vérité |
| `conf/local.yaml` | ce que le poste change, superposé clé par clé |
| `libs/predict_common/` | configuration, chemins, schémas, io, db, source HTTP, horodatages |
| `services/collector/src/collector/__main__.py` | collecte datée d'une période, rattrapage |
| `services/collector/src/collector/poller.py` | collecte continue de `/current` |
| `services/collector/src/collector/datasets.py` | chargement de l'historique de référence |
| `services/collector/src/collector/sink.py` | écriture des mesures, états, alertes, capteurs, sites |
| `services/etl/src/etl/extract.py` | lecture de `mesure` sur la fenêtre des décalages |
| `services/etl/src/etl/clean.py` | typage et dédoublonnage du lot lu |
| `services/etl/src/etl/quality.py` | cause de chaque valeur nulle, qualification |
| `services/etl/src/etl/impute.py` | valeur reconstruite, dans une colonne séparée |
| `services/etl/src/etl/exclude.py` | mise à l'écart de ce qui n'est pas exploitable |
| `services/etl/src/etl/features.py` | grille horaire, décalages, agrégats, calendrier |
| `services/etl/src/etl/validate.py` | pandera : échec immédiat si le schéma est cassé |
| `services/etl/src/etl/load.py` | repose les colonnes déduites dans `mesure` |
| `services/training/src/training/dataset.py` | lecture des partitions, découpe temporelle |
| `services/training/src/training/model.py` | XGBoost, arrêt anticipé, métriques |
| `services/training/src/training/candidates.py` | familles opposées sur le banc |
| `services/training/src/training/arbitration.py` | banc d'arbitrage, fenêtre commune |
| `services/training/src/training/promotion.py` | règle qui décide d'une mise en service |
| `services/training/src/training/tracking.py` | MLflow : runs, versions, tags, alias |
| `services/training/src/training/registry.py` | inscription en base de la version servie |
| `services/training/src/training/drift.py` | écart prédiction/réel, seuil et verdict |
| `services/serving/src/serving/loader.py` | résolution du modèle par alias |
| `services/serving/src/serving/forecast.py` | historique lu, prévision par récurrence |
| `services/serving/src/serving/api.py` | FastAPI, `CONTRACT_VERSION` |
| `services/serving/src/serving/auth.py` | clé de service sur les routes du contrat |
| `services/serving/src/serving/schemas.py` | DTO — source de vérité du contrat |
| `tests/test_architecture.py` | vérifie qu'aucun service n'en importe un autre |
| `deploy/` | amorçage du nœud Garage, dépôt des jeux de données |
| `scripts/export_openapi.py` | export de la spécification OpenAPI |

## Dépendances

Pas de `requirements*.txt` : chaque paquet déclare les siennes dans son
`pyproject.toml`, et `uv.lock` fige la résolution de l'ensemble. La CI
installe exactement ce que les images installeront.

| Paquet | Ce qu'il tire | Pourquoi |
|---|---|---|
| `predict-common` | PyYAML, pandas, pyarrow, pandera, SQLAlchemy, httpx | configuration, chemins, schémas, parquet, tables, source |
| `collector` | psycopg | il écrit la couche brute |
| `etl` | psycopg | il lit et repose la couche brute |
| `training` | scikit-learn, xgboost, mlflow | il apprend et enregistre |
| `serving` | fastapi, uvicorn, mlflow, xgboost | il sert ; xgboost restitue le modèle |

Ni l'entraînement ni le service d'inférence ne déclarent de driver base : pour
l'un, un accès direct court-circuiterait le contrat de la couche des
variables ; pour l'autre, il n'a rien à y écrire. Un test d'architecture
refuse les deux.

`fastapi` et `pydantic` sont épinglés à la version exacte : ce sont eux qui
génèrent la spécification OpenAPI, et une montée même en patch peut produire
un JSON différent alors qu'aucun DTO n'a bougé. **Ne pas dé-épingler sans PR
de contrat.**

Les versions tiennent sur **Python 3.11**, celle de la CI.

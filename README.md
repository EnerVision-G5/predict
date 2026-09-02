# predict

Repo Data d'EnerVision. Il porte quatre exécutables indépendants qui ne
communiquent que par des artefacts — une table, des partitions, un registre de
modèles : la collecte des mesures depuis l'API Mock IoT vers TimescaleDB, leur
transformation en variables d'apprentissage, l'entraînement des modèles suivi
par MLflow, et le service d'inférence FastAPI déployé sur Azure.

La règle centrale tient en une phrase. **Aucun service n'importe le code d'un
autre.** Si `etl` faisait `from collector.client import fetch`, il n'y aurait
plus quatre services mais un monolithe avec quatre dossiers, et le déploiement
séparé serait un mensonge. La règle n'est pas une consigne : chaque image ne
copie que la bibliothèque partagée et son propre code, et un test
d'architecture la vérifie à chaque exécution de la suite.

## Les contrats entre services

C'est la partie qui fait vraiment la séparation. Chaque frontière se définit
par un **emplacement** plus un **schéma**, jamais par un appel de fonction.

| Producteur | Artefact | Consommateur |
|---|---|---|
| `collector` | table `site`, synchronisée depuis `GET /api/v1/sites` | `mesure` (clé étrangère) |
| `collector` | table `mesure`, colonnes de la source | `etl` |
| `etl` | table `mesure`, colonnes déduites | API EnerVision |
| `etl` | `features/v1/dt=2026-09-02/part-*.parquet` (Garage) | `training`, `serving` |
| `training` | `models:/enervision_xgboost@champion` (MLflow) | `serving` |

Le service d'inférence, lui, ne produit aucun artefact : il calcule une
prévision et la rend. L'archiver dans `prediction` et référencer le modèle dans
`modele` sont le métier de l'API EnerVision, qui sert ce contrat à ses
consommateurs et tient sa propre base.

Une frontière est une table ou un chemin ; ce qui compte est qu'elle ne soit
jamais un import. Les chemins sont construits par `predict_common.paths`, la
table est déclarée dans `predict_common.db`, les schémas dans
`predict_common.schemas`. Le producteur valide à l'écriture, le consommateur
valide à la lecture — ce n'est pas une redite : validée à l'écriture seule, une
rupture de contrat serait imputée au consommateur trois étapes plus loin ;
validée à la lecture seule, une partition fausse serait déjà publiée.

### Qui écrit quoi dans `mesure`

Les deux services écrivent la même table sans jamais se marcher dessus, et
c'est la politique d'écriture qui le garantit.

| | Colonnes écrites | Sur conflit |
|---|---|---|
| `collector` | les sept mesures, `null_reasons`, `data_quality` de la source | `DO NOTHING` |
| `etl` | `null_reasons`, `data_quality`, `consumption_kw_imputed`, `imputation_method` | `DO UPDATE` sur ces quatre |

Le collecteur n'écrase rien : une recollecte ne défait donc pas la
transformation déjà faite. L'ETL repose, lui, parce qu'il a du nouveau à dire —
sans quoi corriger une règle de qualification n'aurait aucun effet sur
l'historique déjà traité. Et son `DO UPDATE` ne liste que les colonnes
déduites : même si le lot soumis portait une consommation différente de celle
en base, la base garderait celle de la source. La panne capteur ne peut pas
être effacée par l'étage qui a justement pour métier de la décrire.

Une quatrième frontière existe, de même forme : le service d'inférence lit la
dernière partition de variables pour reconstruire les décalages d'un site. Il
ne les recalcule pas depuis les mesures brutes, ce qui donnerait un second jeu
de règles qui finirait par diverger du premier.

## Rattraper l'historique

C'est ce qui alimente le premier entraînement. Sans historique, la couche des
variables n'a pas de quoi calculer un décalage de 168 heures, et le modèle n'a
rien à apprendre.

```bash
python -m collector --start 2026-08-01 --end 2026-09-01
python -m collector --start 2026-08-01 --end 2026-09-01 --site SITE001 --limit 500
```

Deux façons de dire la période, parce qu'elles ne servent pas au même usage.
`--start/--end` nomme une période, ce que fait un analyste qui rattrape ;
`--date/--days` nomme une journée et sa profondeur, ce que fait un
ordonnanceur, où la date est un paramètre et la profondeur une constante. Les
deux formes s'excluent : les mélanger laisserait deux périodes possibles pour
un même appel, et le run partirait sur l'une des deux sans dire laquelle.

`--limit` borne le nombre de mesures demandées par requête. Le plafond est
celui de la source, **1000**, et il est refusé ici plutôt que découvert dans
une réponse 422 — sans quoi un rattrapage de trois mois échouerait à sa
première page, en laissant chercher la panne du côté du réseau.

Sans `--site`, les sept sites du référentiel sont collectés.

**Le rejeu est sans effet de bord.** Relancer la même période ne double rien :
l'insertion est un `ON CONFLICT DO NOTHING` sur la clé naturelle
`(site_id, ts)`. Elle n'écrase rien non plus, et c'est le point qui compte pour
un rattrapage lancé après coup : les lignes visées peuvent déjà porter les
colonnes que l'ETL a déduites, et un `DO UPDATE` déferait la transformation en
croyant rafraîchir la source.

## Les points d'entrée

Chaque service est lançable seul, sans les autres.

```bash
python -m collector --start 2026-08-01 --end 2026-09-01  # rattrapage d'un mois
python -m collector --date 2026-09-02                    # une journée, forme courte
python -m collector.poller                               # collecte au fil de l'eau
python -m etl       --date 2026-09-02 --feature-version v1
python -m training  --feature-version v1 --history-days 90
uvicorn serving.api:app
```

Deux propriétés sont tenues absolument.

**Idempotence.** Relancer `--date 2026-09-02` reproduit la journée, il ne la
double pas. En base, la clé primaire `(site_id, ts)` porte tout : chaque
écriture est un `ON CONFLICT`. Sur le stockage objet, la partition est écrite
dans un répertoire de travail voisin puis substituée en bloc, si bien qu'un
lecteur ne voit jamais une partition à moitié écrite. C'est ce qui rend le
rejeu après incident sans effet de bord, et le rejeu est le mode d'exploitation
normal — une source indisponible deux heures se rattrape en relançant la
journée, pas en réparant à la main.

**Aucun état en mémoire entre services.** Pas de variable globale, pas de cache
partagé. Le seul état, ce sont la table `mesure`, les partitions parquet et le
registre MLflow.

## Démarrer

```bash
uv sync                     # installe la bibliothèque et les quatre services
cp .env.example .env
make check                  # ruff + pytest
```

`uv` gère un workspace : `pyproject.toml` à la racine déclare les membres,
chaque service a le sien avec ses propres dépendances. Le découpage n'est pas
cosmétique — `serving` ne déclare ni pandas-SQL, ni scikit-learn, ni httpx, si
bien qu'un import de l'ETL glissé dans son code ne s'installerait pas dans son
image.

Sans Docker, la chaîne complète sur une journée :

```bash
make run-day DATE=2026-09-02 FV=v1
```

## Arborescence

| Chemin | Rôle |
|---|---|
| `conf/base.yaml` | Chemins, noms de colonnes, hyperparamètres. Source de vérité |
| `conf/local.yaml` | Ce que le poste change ; superposé clé par clé |
| `libs/predict_common/config.py` | Chargement YAML en couches, résolution des `${}` |
| `libs/predict_common/paths.py` | Construction des chemins de partition |
| `libs/predict_common/schemas.py` | Schémas pandera et pyarrow, par couche |
| `libs/predict_common/io.py` | Lecture, écriture atomique, disque ou Garage |
| `libs/predict_common/db.py` | Déclaration de `site`, `mesure`, `mesure_exclu` |
| `services/collector/client.py` | HTTP : pagination, reprises, débit borné |
| `services/collector/sink.py` | Référentiel `site`, puis écriture brute dans `mesure` |
| `services/collector/poller.py` | Collecte continue de `/current` |
| `services/etl/extract.py` | Lecture de `mesure` sur la fenêtre des décalages |
| `services/etl/clean.py` | Typage et dédoublonnage du lot lu |
| `services/etl/quality.py` | Cause de chaque valeur nulle, qualification |
| `services/etl/impute.py` | Valeur reconstruite, dans une colonne séparée |
| `services/etl/exclude.py` | Mise à l'écart de ce qui n'est pas exploitable |
| `services/etl/features.py` | Grille horaire, décalages, agrégats, calendrier |
| `services/etl/validate.py` | Pandera : échec immédiat si le schéma est cassé |
| `services/etl/load.py` | Repose les colonnes déduites dans `mesure` |
| `services/training/dataset.py` | Lecture des partitions, découpe temporelle |
| `services/training/model.py` | XGBoost, arrêt anticipé, métriques |
| `services/training/tracking.py` | MLflow : paramètres, métriques, signature, tags, alias |
| `services/training/drift.py` | Écart prédiction/réel, seuil et verdict |
| `services/serving/loader.py` | Résolution du modèle par alias |
| `services/serving/forecast.py` | Historique lu, prévision par récurrence |
| `services/serving/api.py` | FastAPI, `CONTRACT_VERSION` |
| `services/serving/schemas.py` | DTO : source de vérité du contrat |
| `tests/test_architecture.py` | Vérifie qu'aucun service n'en importe un autre |
| `data/` | Partitions de variables, jamais commitées — ou un seau Garage |
| `deploy/` | Configuration et amorçage du nœud Garage |

## Configuration

Un seul endroit : `conf/`. `base.yaml` décrit la chaîne, `conf/{PREDICT_ENV}.yaml`
décrit ce qu'une machine en change, et la superposition se fait clé par clé —
surcharger `training.params.n_estimators` ne fait pas disparaître les autres
hyperparamètres.

Aucun secret n'entre dans un fichier versionné. Les valeurs sensibles sont
écrites `${VARIABLE}` et résolues dans l'environnement au chargement. Un
`${VARIABLE}` sans repli est **obligatoire** et fait échouer le démarrage :
mieux vaut un service qui refuse de partir en nommant la variable qu'un run
qui écrit ses artefacts au mauvais endroit sans rien dire.

Un réglage qui n'apparaît pas dans `conf/` n'existe pas, même s'il est écrit
dans `.env`.

## Valeurs manquantes

Une valeur nulle n'est pas une valeur qui manque : c'est un capteur qui dit
qu'il est tombé. Aucune n'est filtrée, aucune n'est écrasée. La chaîne les
range en quatre temps, et chaque temps a son module.

| Étage | Ce qu'il garantit |
|---|---|
| `collector.writer` | La valeur nulle est écrite telle quelle, avec ses motifs |
| `etl.quality` | Aucune valeur nulle sans motif dans `null_reasons`, et un `data_quality` que les données ne contredisent pas |
| `etl.impute` | Une valeur reconstruite dans `consumption_kw_imputed`, jamais dans `consumption_kw` |
| `etl.exclude` | Une mesure ni brute ni reconstruite est écartée des variables, avec sa cause |

`data_quality` retient la plus sévère des deux qualifications, celle de la
source et celle que les données laissent déduire. La source peut alerter
au-delà de ce que l'ETL voit, jamais en deçà : un `good` posé sur une puissance
absente sortirait la panne de `idx_mesure_quality`, l'index d'audit des
capteurs, qui n'indexe justement que les mesures non `good`.

Une valeur encadrée par deux voisines connues est interpolée sur le temps
(`interpolation`), une valeur qui n'a qu'un passé est reportée (`locf`), et une
valeur sans passé ni futur dans le lot n'est pas inventée : elle reste nulle,
`imputation_method` vaut `none`, et la mesure est écartée des variables. C'est
le cas ordinaire du poller, dont chaque tick ne porte qu'une mesure.

La couche des variables expose enfin `imputed_ratio` : la part de chaque heure
que l'ETL a reconstruite. L'entraînement décide quoi en faire —
`training.max_imputed_ratio` — plutôt que de subir un arbitrage caché.

## Les variables

La source produit à la minute, le modèle prédit à l'heure. Le changement de
pas a lieu dans `etl.features`, et il corrige une faute que la chaîne portait
avant la découpe : les décalages étaient calculés en nombre de lignes, ce qui
suppose une série sans trou. Une coupure de capteur d'un quart d'heure décalait
alors tout l'historique, et `lag_24h` désignait autre chose que la veille sans
que rien ne le dise.

La série est désormais rééchantillonnée puis réindexée sur une grille horaire
complète, trous compris. Un décalage est une position sur cette grille :
`lag_24h` est la veille à la même heure, ou rien du tout — et une heure dont un
décalage manque est retirée plutôt que complétée.

La moyenne glissante est décalée d'un pas avant d'être calculée. Sans ce
décalage, elle contiendrait la cible de l'heure courante : le modèle lirait la
réponse dans la question, ses métriques seraient excellentes à l'apprentissage
et fausses en production.

Produire une journée demande donc de lire les journées précédentes. La
profondeur est déduite du plus long décalage — neuf jours pour `lag_168h` — et
l'ETL le fait seul : `--date` reste la seule chose à lui donner.

### Ce que la partition publie, ce que le modèle apprend

Ce ne sont pas les mêmes colonnes, et les confondre coûtait cher. La
température est une mesure réelle, la partition la porte — mais le service
d'inférence ne connaît pas la météo des heures qu'il prédit : il la
présenterait vide à chaque requête. Un modèle entraîné dessus apprend des
séparations qu'il ne peut plus emprunter en production, et chaque arbre qui
teste la température envoie alors toutes les lignes servies dans sa branche
par défaut. Ce n'est pas une information perdue proprement, c'est un biais
fixe que rien ne signale — et la surveillance ne le voyait pas, puisqu'elle
rejoue le modèle sur des partitions où la température, elle, est présente.

Deux listes tiennent donc la distinction, dans `predict_common.schemas` :
`published_columns` dit ce que l'ETL écrit, `feature_columns` ce que le modèle
consomme. La seconde est un sous-ensemble de la première.

| Colonne | Publiée | Apprise |
|---|---|---|
| `hour`, `day_of_week`, `is_weekend` | oui | oui |
| `lag_1h`, `lag_24h`, `lag_168h`, `roll_mean_24h` | oui | oui |
| `temperature_celsius` | oui | **non** |
| `data_quality`, `imputed_ratio` | oui | non — elles décrivent la ligne |

Retirer une colonne de `published_columns` change le contrat de la couche,
donc impose une nouvelle `feature_version`. En retirer une de
`feature_columns` ne change que le modèle, que MLflow versionne déjà : c'est
pour cela que sortir la température n'a pas demandé de `v2`. Le jour où une
prévision météo alimentera l'inférence, la colonne est déjà là.

### Changer de version

`etl.lag_hours`, `etl.rolling_window_h` et `etl.resample_rule` décident des
colonnes produites. Les changer change le contrat, et cela se fait en publiant
sous une nouvelle `feature_version` :

```bash
python -m etl --date 2026-09-02 --feature-version v2
```

Les deux versions coexistent sous deux préfixes. Un modèle entraîné sur `v1`
reste donc reproductible après la sortie de `v2`, et le service continue de
servir `v1` tant qu'on ne l'a pas rebasculé.

## Le lien training → serving

C'est la frontière la plus souvent mal faite. L'entraînement n'écrit pas un
`model.pkl` que le service rechargerait par `pickle.load` : un pickle
transporte un objet Python et rien d'autre — ni les colonnes attendues, ni leur
ordre, ni la version des bibliothèques qui l'ont produit. Le jour où
l'entraînement monte de version, le service charge le pickle sans broncher et
prédit faux.

Il logue dans MLflow, avec la signature et l'environnement :

```python
mlflow.xgboost.log_model(
    model, name="model",
    signature=infer_signature(X_valid, y_pred),
    registered_model_name="enervision_xgboost",
)
mlflow.log_param("feature_version", "v1")
mlflow.log_param("train_window", "2026-06-01/2026-08-31")
```

Chaque version enregistrée porte en plus des **tags** qui la décrivent :
`feature_version`, `train_window`, `sites`, et les trois métriques de son jeu
de test. Un tag et un alias ne disent pas la même chose — l'alias désigne un
rôle et se déplace, le tag décrit la version et ne bouge plus. Quelqu'un qui
ouvre le registre six mois plus tard voit sur quelles variables et sur quelle
période une version a été entraînée sans avoir à retrouver son run. C'est
aussi ce qui rend la surveillance possible : la dérive se mesure par rapport à
ce que la version affichait à l'entraînement.

Le service résout un **alias** au démarrage. Un entraînement produit un
`challenger`, jamais un `champion` : promouvoir est une décision
d'exploitation, pas une conséquence automatique de la fin d'un run.

```bash
# promouvoir depuis l'entraînement, quand on sait déjà qu'on veut la servir
python -m training --feature-version v1 --promote

# ou promouvoir après coup une version déjà enregistrée
mlflow models set-alias -m enervision_xgboost -a champion -v 7
docker compose restart serving
```

Résultat : on promeut sans redéployer, et le retour arrière est le même geste
en sens inverse.

C'est aussi le `loader` qui sait *comment* parler au modèle. Sa signature dit
les colonnes, leur ordre et leurs types, et MLflow refuse une conversion qu'il
ne peut pas garantir sans perte — un `hour` en `int64` présenté à un modèle
entraîné sur de l'`int32` est rejeté. Le service présente donc les variables
exactement comme la signature les déclare, plutôt que d'en tenir une seconde
copie qui divergerait.

## Surveiller la dérive

Un modèle ne se dégrade pas d'un coup : il se dégrade parce que le monde change
sous lui — un site qui déplace sa production, un capteur remplacé, une saison
absente de l'historique. Les métriques du jour de l'entraînement ne disent rien
de cela : elles ont été mesurées sur des données du passé.

```bash
python -m training.drift                              # les 7 derniers jours
python -m training.drift --since 2026-08-26 --until 2026-09-01
```

Chaque exécution rejoue le modèle **en service** — celui que l'alias `champion`
désigne, pas le dernier entraîné — sur les mesures arrivées depuis, et publie
un run dans l'expérience `enervision-drift`.

### Ce qui est mesuré, et ce qui ne l'est pas

L'écart est calculé **à un pas** : chaque heure est prédite à partir de ses
décalages réels. C'est la même tâche que celle mesurée à l'entraînement, donc
la seule comparable. Le service d'inférence, lui, prédit par récurrence sur
48 heures et son erreur s'accumule mécaniquement à chaque pas ; mêler les deux
rendrait la dégradation du modèle indiscernable de l'effet d'horizon.

### Le seuil

| Réglage | Défaut | Ce qu'il décide |
|---|---|---|
| `monitoring.mae_alert_ratio` | **1.5** | Rapport toléré entre l'erreur mesurée et celle du modèle sur son jeu de test |
| `monitoring.window_days` | 7 | Profondeur évaluée quand aucune borne n'est donnée |
| `monitoring.min_rows` | 24 | En deçà, le run est publié sans verdict |

Le seuil est un **rapport**, pas des kilowatts. Un écart de 20 kW n'a pas le
même sens sur un bureau de 200 kW et sur une usine de 1000, et un seuil absolu
serait à réviser à chaque réentraînement. La référence est l'erreur du modèle
sur son jeu de test, lue dans le run qui l'a produit : elle suit le modèle, et
promouvoir une autre version change la référence du même geste.

`1.5` tolère 50 % de dégradation. En dessous de `1.2`, le bruit d'une semaine
calme suffit à déclencher ; au-delà de `2`, on ne détecte plus qu'une panne.
`DRIFT_ALERT_RATIO` permet de l'abaisser le temps d'une démonstration, pour
montrer le chemin d'alerte sans fabriquer de fausses données.

### Le verdict est un code de sortie

Un ordonnanceur n'a pas à lire un journal pour savoir s'il doit alerter.

| Code | Sens |
|---|---|
| `0` | Stable, ou trop peu d'heures pour conclure |
| `1` | La surveillance elle-même a échoué — partitions absentes, registre injoignable |
| `2` | **Dérive** : l'écart dépasse le seuil |

Le `2` est distinct du `1` à dessein : les confondre ferait chercher un
problème d'infrastructure là où le modèle a simplement vieilli.

## Orchestration

Les services ne s'appellent pas entre eux : c'est un ordonnanceur externe qui
les enchaîne, et c'est aujourd'hui le `Makefile`.

```bash
make run-day DATE=2026-09-02 FV=v1     # collect → etl → train
make backfill START=2026-08-01 END=2026-09-01   # rattrapage d'un mois
make help                              # toutes les cibles
```

Commencer par un Makefile plutôt que par Airflow n'est pas un pis-aller. Tant
que l'enchaînement tient en trois commandes séquentielles, un DAG n'apporte
qu'une infrastructure de plus à exploiter. Le jour où il faudra des reprises
partielles, des dépendances entre journées ou un calendrier, ces trois cibles
se transposeront telles quelles — parce qu'elles sont déjà des processus
indépendants, datés et idempotents.

## Pile Docker

```bash
cp .env.example .env
docker compose up -d mlflow serving
docker compose --profile jobs run --rm collector --date 2026-09-02
docker compose --profile jobs run --rm etl --date 2026-09-02
docker compose --profile jobs run --rm training --feature-version v1
docker compose up -d poller                # collecte au fil de l'eau
```

Les trois traitements datés sont derrière le profil `jobs` : sans lui, un
`docker compose up` déclencherait une collecte, une transformation et un
entraînement à chaque démarrage de la pile.

Deux dépendances ne sont volontairement pas déclarées, pour la même raison —
elles ont déjà une vérité ailleurs. **L'API Mock IoT** est fournie par le
formateur, à l'adresse que porte `MOCK_API_URL_CONTAINER`. **La base
TimescaleDB** est portée par `api/enervision-db/docker-compose.yml`, qui
détient le schéma figé v1.0, et que désigne `DATABASE_URL_CONTAINER`.

Les deux variables sont déclarées avec `:?` dans le compose : un conteneur qui
partirait sans elles écrirait dans le vide et ne le dirait qu'au premier
appel.

### Stockage objet

Le disque local suffit au développement. Pour éprouver le déploiement, un nœud
Garage est disponible derrière le profil `garage` :

```bash
make storage        # démarre garage, crée le seau, affiche la clé générée
```

Le script d'amorçage relève une clé et un secret : les reporter dans `.env`
(`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`), puis poser
`PREDICT_STORAGE_ROOT=s3://enervision` et `AWS_ENDPOINT_URL=http://garage:3900`.

Les quatre services traversent exactement le même code : `predict_common.io`
résout un chemin nu vers le disque et une URI `s3://` vers le stockage objet.
Passer du poste au déploiement ne change qu'une valeur de `conf/`.

Un nœud neuf n'a ni disposition, ni seau, ni clé, et une écriture S3 dans cet
état échoue sur une erreur qui ne dit pas pourquoi : c'est ce que
`deploy/garage-init.sh` pose, une fois, de façon rejouable.

## MLflow

Le backend store est une base SQLite, hors Git, dans le volume `mlruns`. Une
base et non un simple file store : le Model Registry, qui donne son alias au
modèle servi, exige un backend relationnel.

En local, sans Docker :

```bash
mlflow server \
  --backend-store-uri sqlite:///mlruns/mlflow.db \
  --artifacts-destination ./mlruns/artifacts \
  --host 127.0.0.1 --port 5000
```

L'interface est sur <http://127.0.0.1:5000>.

## Endpoints du service d'inférence

```bash
uvicorn serving.api:app --reload
```

| Méthode | Route | Réponse |
|---|---|---|
| GET | `/health` | `HealthOut` |
| POST | `/api/v1/predict` | `PredictionOut`, erreurs 404, 422 et 503 |

`/health` n'est volontairement pas préfixé par `/api/v1` : c'est la sonde de
disponibilité utilisée par l'hébergeur. Elle ne consulte pas le registre —
la lier à MLflow ferait redémarrer un service en parfait état chaque fois que
le registre tousse.

Les trois erreurs ne disent pas la même chose, et les confondre enverrait les
exploitants chercher la panne du mauvais côté : **404** le site n'a pas
d'historique récent, **422** la requête sort des bornes du contrat, **503** le
registre n'a résolu aucun modèle.

### L'intervalle de confiance

`lower_bound_kw` et `upper_bound_kw` encadrent chaque point à 95 %. La demi-
largeur vaut `1.96 × residual_std × √pas`, où `residual_std` est la dispersion
de l'erreur du modèle **sur son jeu de test**, mesurée à l'entraînement et
posée en tag sur la version enregistrée.

Un tag et non une constante du service : la dispersion suit le modèle, donc
promouvoir une autre version change la largeur des intervalles du même geste,
sans redéployer. Une version qui ne déclare pas ce tag est servie sans bornes
— le contrat les prévoit optionnelles depuis l'origine — plutôt qu'avec une
bande inventée dont rien ne dirait qu'elle ne repose sur rien.

La largeur croît en **racine** du pas, et non linéairement. Le service prédit
par récurrence : chaque heure repart des heures qu'il vient lui-même de
prédire, et deux erreurs successives s'additionnent en variance, pas en écart-
type. Une croissance linéaire donnerait à l'horizon 48 une bande quatre fois
trop large, que personne ne lirait.

C'est une approximation, et elle est optimiste sur deux points : elle suppose
les erreurs successives indépendantes, et elle ignore que le modèle se trompe
davantage sur ses propres prédictions que sur des mesures. Elle dit un ordre de
grandeur, pas une garantie. Elle est en outre **globale** : un seul écart-type
pour les sept sites, alors qu'ils vont de 90 à 580 kW. La bande est donc large
pour un petit site et étroite pour un gros. Un `residual_std` par site est la
suite naturelle.

## Contrat OpenAPI

La source de vérité, ce sont les DTO de `services/serving/src/serving/schemas.py`.
Le fichier gelé qui fait foi entre les équipes est
`enervision/docs/contracts/openapi-predict.json`.

> **⚠ Le contrat doit être régénéré.** `CONTRACT_VERSION` passe de `1.0.0` à
> `1.1.0` avec la mise en service du modèle. Deux changements l'imposent, tous
> deux additifs : la description du service ne peut plus annoncer un `501`
> qu'il ne renvoie plus, et un `503` est déclaré pour le cas où le registre n'a
> rien à servir. Le job CI `contract-drift` échouera jusqu'à ce que la PR soit
> passée sur `enervision/docs/contracts` — c'est le signal attendu, pas un
> incident.

```bash
python scripts/export_openapi.py ../docs/contracts/openapi-predict.json
```

Les docstrings de `serving/schemas.py` alimentent les descriptions du contrat :
les reformater change le JSON gelé. C'est la raison pour laquelle ruff est
configuré à 88 colonnes et non 80.

## Ingestion continue

`python -m collector` rattrape une fenêtre passée ; `python -m collector.poller`
alimente la couche brute au fil de l'eau. Il interroge `/current` sur chaque
site à cadence fixe, une minute par défaut.

Le service porte `restart: unless-stopped`. Une coupure réseau ne le fait pas
sortir pour autant, elle est absorbée à trois niveaux, du plus fin au plus
grossier.

| Niveau | Mécanisme | Ce qu'il couvre |
|---|---|---|
| Appel | `source.retries` reprises, attente croissante | Micro-coupure |
| Tick | Le site en échec est journalisé, les autres sont collectés | Site indisponible |
| Conteneur | `restart: unless-stopped` | Sortie du processus |

Un site injoignable ne fait donc perdre qu'un tick. Seul un démarrage sans
référentiel des sites fait sortir le processus, avec le code 1, et c'est alors
Docker qui reprend la main.

### Lire le retard dans les journaux

Deux retards sont journalisés, parce qu'ils ne désignent pas la même panne.

Le **retard de données** est l'âge de la mesure servie par la source :

```text
INFO collector.poller site SITE001 : 1 mesure(s), retard 12.4 s, qualité good=1
INFO collector.poller tick : 7/7 site(s), 7 mesure(s), retard données max 12.4 s, durée 0.83 s
```

Au-delà de `collector.lag_warning_s`, la ligne du site passe en `WARNING`.

Le **retard d'ordonnancement** est celui que le poller a pris lui-même. Il
n'apparaît que s'il dépasse cinq secondes, et signale un tick qui déborde de la
cadence :

```text
WARNING collector.poller retard d'ordonnancement : tick démarré 7.2 s trop tard
WARNING collector.poller 1 tick(s) sauté(s) : le tick précédent a dépassé la cadence
```

Un tick sauté n'est pas rejoué : `/current` ne sert que la mesure du moment, un
rattrapage relirait la même valeur. La cadence reste ancrée sur des instants
absolus, donc un tick lent ne décale pas les suivants.

## TimescaleDB

La table `mesure` est la couche brute de la chaîne, et non une sortie annexe :
le collecteur y écrit, l'ETL l'y relit et y repose ce qu'il en déduit. Elle est
en outre lue par l'API EnerVision, qui vit dans un autre repo. La base est
fournie par `api/enervision-db/docker-compose.yml`, qui détient le schéma figé
v1.0 — la dupliquer ici donnerait deux vérités sur la même table.

`DATABASE_URL` n'est donc pas optionnelle. Le collecteur et l'ETL refusent de
démarrer sans, en nommant la variable : un service qui partirait sans base
écrirait dans le vide et ne le dirait qu'à la première insertion.

**Migration.** L'ETL repose `consumption_kw_imputed` et `imputation_method`,
absentes du schéma figé v1.0. La migration existe déjà dans le repo de la
base — `enervision-db/initdb/03_mesure_imputation.sql` — et rien n'est à
écrire ici. Elle est idempotente, donc rejouable ; une base déjà démarrée ne
rejoue pas ses scripts d'initdb et doit donc la recevoir à la main :

```bash
psql -U enervision -d enervision -f initdb/03_mesure_imputation.sql
```

**Référentiel des sites.** `mesure.site_id` référence `site` : une mesure dont
le site n'y figure pas est rejetée, quelle que soit sa qualité. Le seed
`02_seed_sites.sql` pose sept sites, dont quatre avec des capacités qu'il
marque lui-même « à remplacer par les valeurs réelles de `GET /api/v1/sites`
au premier démarrage de l'ETL ». C'est le collecteur qui s'en charge, seul
service à joindre cet endpoint, avant toute écriture de mesure.

Un site que la source décrirait à moitié est écarté plutôt que soumis : trois
colonnes de `site` sont NOT NULL sans défaut, et la version placeholder du
seed vaut mieux qu'une insertion en échec. Avec `--site`, un référentiel
injoignable n'interrompt pas la collecte — l'exploitant a nommé ses sites, et
la clé étrangère tranchera s'il manquait vraiment quelque chose.

## Dépendances

Il n'y a plus de `requirements*.txt` : chaque paquet déclare les siennes dans
son `pyproject.toml`, et `uv.lock` fige la résolution de l'ensemble. La CI
installe exactement ce que les images installeront.

| Paquet | Ce qu'il tire | Pourquoi |
|---|---|---|
| `predict-common` | PyYAML, pandas, pyarrow, pandera, SQLAlchemy | Configuration, chemins, schémas, parquet, définition de `mesure` |
| `collector` | httpx, psycopg | Il parle à la source et écrit la couche brute |
| `etl` | psycopg | Il lit et repose la couche brute |
| `training` | scikit-learn, xgboost, mlflow | Il apprend et enregistre |
| `serving` | fastapi, uvicorn, mlflow, xgboost | Il sert ; xgboost restitue le modèle |

Ni l'entraînement ni le service d'inférence ne déclarent de driver base. Pour
l'entraînement, un accès direct court-circuiterait le contrat de la couche des
variables ; pour le service d'inférence, il n'a rien à y écrire. Un test
d'architecture refuse les deux.

`fastapi` et `pydantic` sont épinglés à la version exacte parce que ce sont eux
qui génèrent la spécification OpenAPI. Une montée de version, même en patch,
peut produire un JSON différent alors qu'aucun DTO n'a bougé, et le job
`contract-drift` échouerait sur une dérive qui n'existe pas.
**Ne pas dé-épingler sans PR de contrat.**

Les versions tiennent sur Python 3.11, version de la CI : monter numpy au-delà
de 2.4 ou xgboost au-delà de 3.2 impose de faire passer la CI en 3.12 d'abord.

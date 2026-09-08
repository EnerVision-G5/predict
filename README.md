# predict

Repo Data d'EnerVision. Il porte quatre exécutables indépendants qui ne
communiquent que par des artefacts — une table, des partitions, un registre de
modèles : la collecte des mesures depuis l'API Mock IoT vers TimescaleDB, leur
transformation en variables d'apprentissage, l'entraînement des modèles suivi
par MLflow, et le service d'inférence FastAPI, déployé on-premise et appelé par
la seule API métier.

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
| `training` | table `modele`, ligne active de la version promue | API EnerVision |

Le service d'inférence, lui, ne produit aucun artefact : il calcule une
prévision et la rend. L'archiver dans `prediction` est le métier de l'API
EnerVision, qui sert ce contrat à ses consommateurs et tient sa propre base.

`modele`, en revanche, est écrite ici. Elle dit quel modèle est en service, et
le seul geste qui change cette réponse est la promotion d'un alias — un geste
de l'entraînement. Laisser l'API la déduire en interrogeant MLflow donnerait
deux sources pour un fait dont une seule est autoritaire, et rendrait la
chaîne de prédiction dépendante d'un service dont ce n'est pas le contrat.
C'est aussi ce qui rend `prediction` remplissable : sa colonne `modele_id` est
NOT NULL et n'aurait sinon aucune ligne à référencer.

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
| `etl` | `null_reasons`, `data_quality`, `consumption_kw_imputed`, `imputation_method`, `quality_source` | `DO UPDATE` sur ces cinq |

Le collecteur n'écrase rien : une recollecte ne défait donc pas la
transformation déjà faite. L'ETL repose, lui, parce qu'il a du nouveau à dire —
sans quoi corriger une règle de qualification n'aurait aucun effet sur
l'historique déjà traité. Et son `DO UPDATE` ne liste que les colonnes
déduites : même si le lot soumis portait une consommation différente de celle
en base, la base garderait celle de la source. La panne capteur ne peut pas
être effacée par l'étage qui a justement pour métier de la décrire.

`quality_source` est la cinquième colonne, et elle ne décrit pas la mesure mais
le traitement. `data_quality` est `NOT NULL DEFAULT 'good'` : le collecteur
retombe sur le défaut quand la source se tait, et l'ETL repose la vraie
qualification à son passage. Entre les deux, les deux `good` sont le même
caractère. La colonne vaut `source` tant que l'ETL n'est pas passé, `etl`
ensuite — sans elle, l'API métier compterait **0 % de mesures dégradées** sur
une journée fraîchement collectée, et le site paraîtrait parfait.

Une quatrième frontière existe, de même forme : le service d'inférence lit la
dernière partition de variables pour reconstruire les décalages d'un site. Il
ne les recalcule pas depuis les mesures brutes, ce qui donnerait un second jeu
de règles qui finirait par diverger du premier.

### `ingestion_etat` : ce que `mesure` ne peut pas dire

`mesure.inserted_at` répond **tant qu'il y a des lignes**. Un capteur mort en
produit encore — nulles, avec leurs motifs — donc `max(inserted_at)` avance. Un
poller arrêté, une source en 500 ou une base injoignable n'en produisent
aucune : `max(inserted_at)` se fige alors exactement comme si le site avait
cessé d'exister. Aucune requête ne distingue « la collecte a tourné et il n'y
avait rien » de « la collecte n'a pas tourné », et c'est la panne la plus grave
qui devient la plus discrète.

Les deux points d'entrée du collecteur reposent donc leur état dans
`ingestion_etat`, **une ligne par site**, en `DO UPDATE` :

| Colonne | Ce qu'elle permet de voir |
|---|---|
| `last_attempt_at` / `last_success_at` | égales, tout va bien ; écartées, la collecte tourne et échoue ; les deux figées, le collecteur ne tourne plus |
| `last_data_lag_s` | âge de la mesure servie par la source, mesuré par le collecteur — pas reconstructible depuis `inserted_at - ts`, qui mélange retard de source et retard d'écriture |
| `consecutive_failures` | l'à-coup contre la panne installée |
| `source` | `poller` ou `backfill` : un rattrapage lancé à la main pendant que le poller est arrêté ne doit pas faire paraître l'ingestion vivante |

Pas de journal par tick : sept sites à la minute feraient dix mille lignes par
jour à purger, pour une question qui est au présent. L'écriture ne peut jamais
interrompre la boucle — un tick dont la base vient de refuser les mesures ne
pourra pas y écrire son échec non plus, et mourir là serait mourir au moment
où le processus a le plus de raisons de continuer.

## Poser l'historique de référence

Deux années horaires par site, fournies avec la source. C'est la SEULE origine
possible de l'historique d'apprentissage : `GET /api/v1/readings` ne remonte
qu'à 48 heures et répond des mesures nulles au-delà, sans erreur.

```bash
python -m collector.datasets                             # racine de conf/
python -m collector.datasets --root s3://enervision-datasets
```

**Ces fichiers ne sont ni dans le dépôt ni dans l'image.** Vingt-quatre
mégaoctets de CSV figés pesaient sur chaque clone et sur chaque couche
publiée, pour une donnée qu'un seul geste lit une seule fois. Ils vivent sur
le stockage objet, et `storage.datasets_root` dit où — un chemin de poste ou
une URI `s3://`, les deux traversant `predict_common.io` comme les partitions.

Leur seau est distinct de celui des partitions, et ce n'est pas de la
symétrie : un rejeu de l'ETL réécrit les partitions, et deux années de mesures
que rien dans la chaîne ne sait régénérer n'ont pas à partager un préfixe avec
ce qui s'efface.

Les y déposer une première fois, depuis un poste qui les a :

```bash
aws --endpoint-url http://localhost:3900 s3 cp datasets/     s3://enervision-datasets/ --recursive --exclude "*" --include "SITE*.csv"
```

Seuls les fichiers par site sont lus. `all_sites_combined.csv` porte
exactement les mêmes lignes pour 11 Mo de plus, et les `*_metadata.json`
décrivent le jeu pour un lecteur humain — les déposer aussi ne coûte que du
stockage, les charger doublerait le travail pour un résultat identique.

**Le rejeu est sans effet de bord**, pour la même raison que le rattrapage :
`sink.write` insère en `ON CONFLICT DO NOTHING`. Sur une base qui collecte
déjà, l'import comble ce qui manque et ne touche à aucune mesure présente.

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
| `services/serving/auth.py` | Clé de service exigée sur les routes du contrat |
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

# ou promouvoir après coup une version déjà enregistrée, sans réapprendre
python -m training --promote-version 7

# le service résout son alias au démarrage : il le relit au redémarrage,
# sans reconstruction d'image ni redéploiement
docker compose restart serving
```

Résultat : on promeut sans redéployer, et le retour arrière est le même geste
en sens inverse.

`--promote` fait deux choses, et la seconde est la raison pour laquelle
l'entraînement connaît la base. Il déplace l'alias, puis inscrit la version
dans `modele` — `nom`, `version`, `mlflow_run_id`, la date du run et `actif` —
et éteint du même coup les autres versions du même modèle, dans une seule
transaction. C'est cette ligne que l'API EnerVision référence depuis
`prediction.modele_id`, et sa colonne `actif` qui dit quel modèle sert
aujourd'hui sans avoir à interroger MLflow.

`DATABASE_URL` est donc obligatoire sous `--promote`, et vérifiée **avant**
l'apprentissage : la découvrir absente au bout d'une heure de calcul laisserait
le choix entre perdre le run et servir un modèle que rien ne référence. Sans
cette option, l'entraînement ne touche jamais la base.

`--promote-version` fait exactement le même geste sur une version déjà
enregistrée, et c'est lui qui remplace le `mlflow models set-alias` d'avant.
Celui-ci fonctionne toujours, mais il déplace l'alias sans rien savoir de
`modele` : le miroir reste alors en arrière, et il dit qu'une autre version
sert. Après une promotion faite ainsi, un `--promote-version` sur la version
concernée remet tout d'aplomb — l'inscription est un `ON CONFLICT (nom,
version) DO UPDATE`, donc rejouable autant de fois qu'on veut.

### Le banc d'arbitrage, et ce que promouvoir veut dire

Le vocabulaire du challenge — `challenger`, `champion` — a longtemps été le
seul morceau de challenge : `--promote` posait l'alias sur ce qui venait
d'être appris sans jamais regarder ce que la version en place savait faire.
Une semaine d'apprentissage dégradée pouvait remplacer un modèle meilleur
qu'elle, et le seul garde-fou était l'attention de qui tapait la commande.

Deux choses manquaient, et elles se tiennent.

**Une fenêtre commune.** Chaque run mesurait sa qualité sur SON bloc de test,
découpé dans SA fenêtre d'apprentissage : deux entraînements espacés d'une
semaine produisaient deux MAE que rien n'autorisait à mettre côte à côte — ce
qu'on faisait pourtant en les regardant dans l'interface. Le **banc
d'arbitrage** est une fenêtre de journées retirée de l'apprentissage, sur
laquelle tout candidat est réévalué : le modèle qu'on vient d'apprendre, les
familles concurrentes, les baselines naïves. Ses mesures sont préfixées
`arbitrage_` dans le run, et la fenêtre elle-même y est journalisée sous
`arbitrage_window`.

Il est glissant par défaut — les `training.arbitration.days` derniers jours de
la fenêtre demandée — ce qui suffit à comparer entre eux les candidats d'un
même run. Le figer rend les mesures comparables **d'un entraînement à
l'autre** :

```yaml
training:
  arbitration:
    start: "2026-08-01"
    end: "2026-08-14"
```

**Une référence gratuite.** ADR-010 décide que XGBoost est « comparé
systématiquement à une baseline naïve » et que « la baseline est un livrable
permanent, pas une étape jetable ». Les persistances — la consommation d'il y
a 1 h, 24 h, 168 h — sont mesurées sur le même banc, et c'est la meilleure des
trois, donc la plus dure à battre, qui sert de barre (`naif_mae`).

`--promote` passe désormais par une règle, dans cet ordre :

1. **battre la persistance** — un modèle qui ne bat pas la recopie de la
   veille ne paie ni son entraînement, ni son registre, ni sa surveillance ;
2. **être comparable** — deux mesures faites sur des bancs différents ne se
   comparent pas, et le refus est franc plutôt que masqué par un classement
   que personne ne pourrait défendre ;
3. **ne pas dégrader** — `training.promotion.margin` dit ce qu'on tolère, et
   vaut zéro par défaut : le candidat doit au moins égaler le champion.

Un refus sort en **code 3**, distinct du 1 : le modèle a été appris et
enregistré en challenger, seule sa mise en service a été refusée. Il n'y a
rien à réparer, il y a quelque chose à lire.

```bash
# refusé si la version dégrade le service, ou ne bat pas la persistance
python -m training --feature-version v1 --promote

# passer outre : décision d'exploitation, journalisée comme telle
python -m training --feature-version v1 --promote --force
```

`--force` existe parce qu'un exploitant peut avoir une raison que la règle n'a
pas — un champion entraîné sur une période aberrante, un banc qu'on sait
faussé. Il porte sur ce seul geste et n'assouplit rien pour les suivants. Une
version enregistrée **avant** l'existence du banc n'en porte aucune mesure :
la promouvoir demande `--force`, et c'est exactement ce qu'elle est, une
décision prise sans comparaison.

### Opposer les familles : `--challenge`

Un seul algorithme était câblé. ADR-010 tranche entre trois options — baseline
simple, XGBoost, réseau séquentiel — et la comparaison n'existait que dans le
document.

```bash
make challenge HISTORY_DAYS=90
# ou
python -m training --challenge --feature-version v1 --history-days 90
```

Chaque famille déclarée dans `training.candidates`, plus XGBoost (qui tire ses
hyperparamètres de `training.params`, pour qu'ils ne soient pas écrits deux
fois), plus chaque persistance, est ajustée puis mesurée sur le banc. Chacune
a son run dans MLflow, taguée `challenge`, et le classement sort dans le
journal :

```
classement sur le banc 2026-08-21/2026-09-03 (14 journée(s), glissant, 672 heure(s)) :
  1. foret-aleatoire    MAE     2.30 kW   RMSE     2.93   R2  1.000
  2. xgboost            MAE     2.31 kW   RMSE     2.93   R2  1.000
  3. ridge              MAE     3.16 kW   RMSE     3.97   R2  0.999
  4. persistance-24h    MAE     4.57 kW   RMSE     5.84   R2  0.999
```

Le challenge **n'enregistre rien et ne promeut rien**, et ce n'est pas une
limite : un classement dit quelle famille convient au problème, mettre en
service est une autre décision, qui se prend après l'avoir lu et passe par
`--promote-version`. Enregistrer chaque candidat remplirait le registre de
versions qu'aucun alias ne désigne — et la version que `serving` résout porte
le nom d'une famille, `enervision_xgboost` : y déposer une forêt serait un
contresens de nommage avant d'être un contresens d'exploitation.

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

**Ce n'est donc pas le nombre que le dashboard affiche.** L'API métier publie
un « écart prédiction / consommation réelle » qui compare les prévisions
réellement servies — récursives — à ce qui est arrivé ensuite. Les deux sont
justes, celui d'ici sera toujours le meilleur des deux, et les afficher sous le
même libellé serait un contresens.

### Site par site, et pas seulement d'ensemble

Une MAE unique sur tout le parc est dominée par le plus gros consommateur :
elle dit ce que le parc coûte en erreur, pas où l'erreur se trouve. Un bureau
de 200 kW qui double la sienne disparaît dans la moyenne d'une usine de 1000.

Chaque run mesure donc aussi chaque site, publie `SITE00X_mae` et un tag
`verdict_SITE00X` dans MLflow, et journalise le détail trié par erreur
décroissante. Le code de sortie `2` sort **dès qu'un site dérive**, sans
attendre que la moyenne d'ensemble bouge — sinon le détail par site serait
publié sans jamais être écouté.

Une limite à connaître : la référence reste celle du modèle, mesurée sur tout
son jeu de test, et n'est pas propre au site. Un site structurellement plus
difficile que la moyenne paraîtra dégradé dès le premier jour. Le verdict par
site sert à ranger les sites entre eux et à voir l'un d'eux se détacher, pas à
juger un site dans l'absolu — une référence par site demanderait que
l'entraînement en enregistre une.

### Le seuil

| Réglage | Défaut | Ce qu'il décide |
|---|---|---|
| `monitoring.mae_alert_ratio` | **1.5** | Rapport toléré entre l'erreur mesurée et celle du modèle sur son jeu de test |
| `monitoring.window_days` | 7 | Profondeur évaluée quand aucune borne n'est donnée |
| `monitoring.min_rows` | 168 | En deçà, le verdict d'ensemble n'est pas rendu |
| `monitoring.min_rows_per_site` | 24 | En deçà, le verdict d'un site n'est pas rendu |

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
| `2` | **Dérive** : l'écart dépasse le seuil, d'ensemble ou sur un seul site |

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

## Accès au service d'inférence

Le service est routé publiquement et relaie `POST /api/v1/simulate/spike`, qui
**écrit sur la source**. L'API métier protège la même opération derrière le
rôle `writer` : un service ouvert rendait ce contrôle contournable, il
suffisait de l'appeler directement.

Les routes du contrat exigent donc une clé de service, présentée en en-tête
`X-API-Key`. Restent servies sans clé : `/health`, la sonde de vivacité de
l'hébergeur, et `/openapi.json` / `/docs`, dont part le scan DAST de la CI.
`/ready` est fermée — elle nomme la version servie et l'âge des variables.

```bash
# Générer la clé, puis la poser des deux côtés : ici, et dans le .env de
# l'API métier, qui la présente à chaque appel.
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Sans `SERVING_API_KEY`, le service **refuse de démarrer**, comme l'API métier
refuse de démarrer sans `JWT_SECRET` : un service qui partirait ouvert
servirait la simulation de pic à qui la demande, et il aurait l'air sain.
Poser `SERVING_AUTH_ENABLED=false` ouvre les routes pour le poste de
développement — le service le journalise en avertissement à chaque démarrage,
et cela ne doit jamais être déployé.

La clé n'est pas déclarée dans le contrat gelé : l'y ajouter ferait échouer
`contract-drift` sur une PR qui n'a rien changé au contrat métier. C'est la
bonne cible, en patch semver, par une PR sur `enervision/docs/contracts`.

## La grille à la minute

`mesure` est une grille : **au plus une ligne par site et par minute**, et
`(site_id, ts)` en est la clé. La minute est la résolution la plus fine que la
chaîne produise — c'est la cadence du poller.

Les deux points d'entrée n'apportent pas la même granularité, et c'est la
source qui le décide :

| Route | Ce qu'elle sert | Horodatage |
| --- | --- | --- |
| `.../{id}/current` | l'instantané | l'instant de l'appel, sub-seconde |
| `/api/v1/readings` | l'historique, par pas de 30 min | aligné à la minute |

La cadence à la minute vient donc du **poller**, pas du rattrapage :
l'historique de la source n'existe qu'à 30 minutes, et aucune option de
`/readings` ne le raffine. Rattraper un mois donne 48 points par jour et par
site ; les 1440 ne s'obtiennent qu'en polling, à partir du moment où il tourne.

## Le rattrapage au démarrage

Un poller qui redémarre reprenait **au présent** : tout ce que la coupure
avait laissé passer restait un trou, et rien ne le signalait — `mesure` n'a
pas de ligne à montrer pour une minute jamais collectée. Le trou ne se voyait
qu'au moment où l'ETL produisait une journée creuse.

Le poller comble donc ce qui manque avant d'entrer dans sa boucle, par
`/api/v1/readings` — la seule route qui serve du passé. La profondeur n'est
pas fixée : elle est **déduite de la dernière mesure de chaque site**, donc de
la durée réelle de la coupure.

```bash
python -m collector.poller                  # rattrape puis boucle (défaut)
python -m collector.poller --no-catch-up    # boucle seule
python -m collector --catch-up              # rattrapage seul, à la main
```

| État de la base | Ce qui est collecté |
| --- | --- |
| vide | `collector.catch_up_days` journées (30 par défaut) |
| arrêt de 5 jours | les 5 journées, et celle de la reprise |
| redémarrage à chaud | la journée courante seulement |
| un site jamais collecté | retour au plancher, pour tous les sites |

La journée de la dernière mesure est **incluse** : c'est celle où le
collecteur s'est arrêté, elle est donc presque toujours incomplète. Et la
fenêtre est commune à tous les sites plutôt que découpée par site —
`ON CONFLICT DO NOTHING` rend le recouvrement gratuit en base, et une journée
coûte une requête par site à la source.

L'échec du rattrapage n'empêche pas la boucle de démarrer : collecter le
présent a plus de valeur que le passé, et le passé se relance à la main.

`collector.sink.snap_to_grid` ramène l'horodatage sur la grille au moment de
l'écriture, et **seulement là** : le retard d'ingestion se mesure sur
l'horodatage brut, sans quoi il serait quantifié à la minute et ferait
paraître en retard un site à l'heure.

Sans ce calage, une minute couverte par les deux routes entrait en base sous
deux clés — le poller à `14:30:14.620789`, le rattrapage à `14:30:00` — et
`ON CONFLICT DO NOTHING` n'avait aucun conflit à arbitrer. Cela se produit sur
les minutes `:00` et `:30` de chaque heure, soit 48 doublons par jour et par
site.

## Pile Docker

```bash
cp .env.example .env
docker compose up -d mlflow serving
docker compose --profile jobs run --rm collector --date 2026-09-02
docker compose --profile jobs run --rm etl --date 2026-09-02
docker compose --profile jobs run --rm training --feature-version v1
docker compose --profile jobs run --rm drift               # surveillance
docker compose up -d poller                # collecte au fil de l'eau
```

Les quatre traitements datés sont derrière le profil `jobs` : sans lui, un
`docker compose up` déclencherait une collecte, une transformation, un
entraînement et une surveillance à chaque démarrage de la pile.

`drift` partage l'image de `training` — c'est le même paquet — avec un
entrypoint différent. Contrairement à un entraînement, il a vocation à tourner
**tous les jours**, déclenché par l'ordonnanceur de l'infrastructure comme la
collecte de la veille : un indicateur de dérive affiché sur un dashboard mais
produit à la main vieillit en silence, ce qui est pire que pas d'indicateur.

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

Il pose DEUX seaux. `enervision` porte les partitions de variables, que
l'ETL réécrit à chaque run ; `enervision-datasets` porte l'historique de
référence, figé et irremplaçable, désigné par `PREDICT_DATASETS_ROOT`. Voir
« Poser l'historique de référence ».

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
| GET | `/ready` | `ReadinessOut` |
| POST | `/api/v1/predict` | `PredictionOut`, erreurs 404, 422 et 503 |

`/health` n'est volontairement pas préfixé par `/api/v1` : c'est la sonde de
disponibilité utilisée par l'hébergeur. Elle ne consulte pas le registre —
la lier à MLflow ferait redémarrer un service en parfait état chaque fois que
le registre tousse.

`/ready`, elle, consulte : elle dit si un modèle est résolu, si des variables
existent, et jusqu'à quelle heure elles vont. Elle répond **toujours 200**, y
compris quand rien n'est prêt — un 503 en ferait une seconde sonde, et
l'hébergeur redémarrerait le service à chaque hoquet de MLflow. Son
consommateur est l'API métier, qui doit pouvoir dire *pourquoi* la prévision
manque : « aucun modèle au registre » et « partitions absentes » sortent tous
deux en 503 sur `/predict`, et ce 503-là ne les distingue pas.

Les trois erreurs ne disent pas la même chose, et les confondre enverrait les
exploitants chercher la panne du mauvais côté : **404** le site n'a pas
d'historique récent, **422** la requête sort des bornes du contrat, **503** le
registre n'a résolu aucun modèle.

### Sur quelles variables la prévision s'appuie

`history_end` donne la dernière heure observée dont la récurrence est partie,
et `feature_lag_hours` son âge au moment de la réponse. Le service lit les
partitions publiées par l'ETL, pas la base : une prévision calculée sur des
variables vieilles de trois jours n'est pas fausse, elle est **aveugle**, et
rien dans le contrat ne le disait jusqu'ici. Un `feature_lag_hours` négatif
n'est pas ramené à zéro — il signale une partition en avance sur l'horloge du
service, ce que masquer ferait passer un problème de fuseau pour une prévision
fraîche.

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

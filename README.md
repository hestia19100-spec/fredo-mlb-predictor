# ⚾ Les Pronos de Fredo — Prédicteur MLB

Application privée destinée à analyser les rencontres MLB et, à terme, à rechercher les paris présentant le meilleur compromis entre probabilité, risque et value.

> Aucun pari ni résultat financier n’est garanti. Le modèle devra être évalué sur des données hors échantillon avant toute utilisation réelle.

## État actuel

Le projet dispose actuellement d’un MVP technique capable de :

- lancer une application web Streamlit ;
- créer une base locale SQLite ;
- récupérer le calendrier depuis l’API MLB ;
- enregistrer les équipes et les matchs sans doublon ;
- actualiser les statuts et les scores ;
- afficher les rencontres enregistrées dans l’interface ;
- exécuter des tests automatiques.

Aucun modèle prédictif, calcul de probabilité ou système de recommandation n’est encore actif.

## Structure

```text
fredo-mlb-predictor/
├── app.py
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── dashboard_data.py
│   ├── database.py
│   ├── game_repository.py
│   └── mlb_api.py
└── tests/
    ├── test_database.py
    └── test_mlb_api.py
```

Le dossier `data` et la base `fredo_mlb.db` sont créés automatiquement. La base locale n’est pas envoyée sur GitHub.

## Installation dans GitHub Codespaces

Dans le terminal du Codespace :

```bash
python -m pip install -r requirements.txt
```

## Initialiser SQLite

```bash
python -m src.database
```

## Récupérer une journée MLB

Remplacer `AAAA-MM-JJ` par la date souhaitée :

```bash
python -m src.game_repository --date AAAA-MM-JJ
```

Exemple :

```bash
python -m src.game_repository --date 2026-08-28
```

## Lancer l’application

```bash
python -m streamlit run app.py
```

Codespaces proposera d’ouvrir l’application dans le navigateur.

## Lancer les tests

```bash
python -m unittest discover -s tests -v
```

Tous les tests doivent se terminer par :

```text
OK
```

## Principes du futur modèle

Le développement devra respecter les règles suivantes :

- validation strictement chronologique ;
- aucune donnée future dans les variables prédictives ;
- décalage temporel avant tout calcul de statistiques glissantes ;
- comparaison objective des modèles ;
- calibration des probabilités ;
- conservation des versions et métriques ;
- possibilité de conclure `NO BET` ;
- séparation entre probabilité de victoire, edge, EV et confiance.

## Prochaines phases

1. Fiabiliser et historiser la collecte MLB.
2. Construire les tables statistiques nécessaires.
3. Reconstituer les informations disponibles avant chaque match.
4. Créer un modèle baseline de régression logistique.
5. Mettre en place la validation chronologique et les backtests.
6. Ajouter les cotes Betclic via The Odds API.
7. Comparer les probabilités du modèle au marché.
8. Améliorer progressivement l’interface.

## Sécurité

Les vraies clés API ne doivent jamais être inscrites dans le code ni envoyées sur GitHub. Elles seront placées dans les secrets Streamlit ou dans des variables d’environnement.
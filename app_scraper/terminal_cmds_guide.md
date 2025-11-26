Using scrape_reviews.py for individual apps

1) Find the package id from the Play URL (it’s the value after id=)
2) Then run:
python scrape_reviews.py \
--store play \
--app <THE_PACKAGE_ID_FROM_URL> \
--countries ca \
--lang en_CA \
--max 10000 \
--regex ".*" \
--out data/name.csv

BC Services Card
python app_scraperscrape_reviews.py \
--store play \
--app ca.bc.gov.id.servicescard \
--countries ca \
--lang en \
--max 10000 \
--regex ".*" \
--out app_scraper/data/bc_services_card_play.csv

BC Services Card
python scrape_reviews.py \
--store appstore \
--app 1234298467 \
--countries ca \
--lang en \
--max 10000 \
--regex ".*" \
--out data/bc_services_card_play.csv


Using batch_scrape for multiple apps
python app_scraper/batch_scrape.py \
  --excel "app_scraper/data/App_IDs_List.xlsx" \
  --outdir app_scraper/data/batch_scrape_results \
  --lang en \
  --max 10000 \
  --regex ".*"
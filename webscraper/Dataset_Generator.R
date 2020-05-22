setwd("/home/swrdata/Dokumente/SummaryBot/Dataset")

library(dplyr)
library(stringr)
library(R.utils)

bbc_long_dir <- list.dirs("./BBC_News_Summary/News_Articles")[-1]

txts <- character()
sums <- character()

for (d in bbc_long_dir) {
  files <- list.files(d)
  for (f in files) {
    fp <- paste0(d, "/", f)
    fpsum <- gsub("News_Articles", "Summaries", fp)
    text <- readLines(fp, warn = FALSE) %>% paste0(., collapse = " ")
    summary <- readLines(fpsum, warn = FALSE) %>% paste0(., collapse = " ")
    txts <- append(txts, text)
    sums <- append(sums, summary)
  }
  message(d)
}
bbc <- data.frame(lead = sums, body = txts, Language = "English", stringsAsFactors = FALSE)
rm(bbc_long_dir, files, f, fp, fpsum, sums, text, txts, d, summary)
gc()

files <- list.files("./cnn_stories_tokenized")

cnn_txts <- character()
cnn_sums <- character()

pb <- txtProgressBar(max = length(files), style = 3)
for (i in c(1:length(files))) {
  f <- files[i]
  tmp <- readLines(paste0("./cnn_stories_tokenized/", f), warn = FALSE) %>% paste0(., collapse = " ")
  tmp <- tmp %>%
    strsplit(., "@highlight") %>%
    unlist() %>%
    str_squish()
  text <- tmp[1]
  summary <- tmp[-1] %>% paste0(., collapse = ". ")
  cnn_sums <- append(cnn_sums, summary)
  cnn_txts <- append(cnn_txts, text)
  setTxtProgressBar(pb, i)
}

cnn <- data.frame(lead = cnn_sums, body = cnn_txts, Language = "English", stringsAsFactors = FALSE)
cnn <- cnn %>% filter(nchar(lead) > 300, nchar(body) > 2 * nchar(lead))
cnn <- sample_n(cnn, 10000)
cnn$body <- gsub("-LRB- CNN -RRB- --", "", cnn$body)
cnn$body <- gsub("-LRB-", "", cnn$body)
cnn$body <- gsub("``", '"', cnn$body)
cnn$body <- gsub("''", '"', cnn$body)
cnn$lead <- gsub("``", '"', cnn$lead)
cnn$lead <- gsub("''", '"', cnn$lead)

cnn$body <- str_squish(cnn$body)
cnn$lead <- str_squish(cnn$lead)

rm(pb, files, cnn_sums, cnn_txts, tmp, i, text, summary, f)

bbc2 <- read.csv("BBC-Artikel-Clean.csv", stringsAsFactors = FALSE)
bbc2 <- bbc2 %>%
  mutate(Language = "English") %>%
  select(lead, body, Language)

br <- read.csv("BR-Artikel-Clean.csv", stringsAsFactors = FALSE)
br <- br %>%
  mutate(lead = teaserText, Language = "German") %>%
  select(lead, body, Language)

franceinfo <- read.csv("Franceinfo-Artikel-Clean.csv", stringsAsFactors = FALSE)
franceinfo <- franceinfo %>%
  mutate(lead = teaserText, Language = "French") %>%
  select(lead, body, Language)

ts <- read.csv("TS-Artikel.csv", stringsAsFactors = FALSE)
ts <- ts %>%
  mutate(Language = "German") %>%
  select(lead, body, Language)

master <- rbind(bbc, bbc2, cnn, br, franceinfo, ts)
names(master) <- c("Lead", "Body", "Language")
master <- master[complete.cases(master),]
master <- master[-1512,]
master <- master %>% filter(nchar(Lead) > 100, nchar(Body) > 2 * nchar(Lead))
master$Lead <- str_squish(master$Lead)
master$Body <- str_squish(master$Body)
con <- file('Raw_Data.csv', encoding = "UTF-8")
write.csv(master, file = con, row.names = FALSE)


br <- read.csv("BR-Artikel-Clean.csv", stringsAsFactors = FALSE)
br <- br %>%
  mutate(lead = teaserText, Language = "German") %>%
  select(lead, body, Language, topic)

franceinfo <- read.csv("Franceinfo-Artikel-Clean.csv", stringsAsFactors = FALSE)
franceinfo <- franceinfo %>%
  mutate(lead = teaserText, Language = "French") %>%
  select(lead, body, Language, topic)

bbc2 <- read.csv("BBC-Artikel-Clean.csv", stringsAsFactors = FALSE)
bbc2 <- bbc2 %>%
  mutate(Language = "English") %>%
  select(lead, body, Language, topic)


klima_topics <- c("Klimawandel", "climate+change", "climat")
corona_topics <- c("coronavirus", "Corona", "COVID")
#trump_topics <- c("Trump")
korea_topics <- c("Nordkorea", "north+korea", "coree+du+nord")
oil_topics <- c("%C3%96lpreis", "oil+price", "petrole", "OPEC", "OPEP")
#crime_topics <- c("Polizei","police")

topicframe <- rbind(br, franceinfo, bbc2)

klima_artikel <- topicframe %>%
  filter(topic %in% klima_topics) %>%
  mutate(Cluster = "Climate")
corona_artikel <- topicframe %>%
  filter(topic %in% corona_topics) %>%
  mutate(Cluster = "Coronavirus")
#trump_artikel <- topicframe %>% filter(topic %in% trump_topics) %>% mutate(Cluster="Trump")
korea_artikel <- topicframe %>%
  filter(topic %in% korea_topics) %>%
  mutate(Cluster = "Korea")
oil_artikel <- topicframe %>%
  filter(topic %in% oil_topics) %>%
  mutate(Cluster = "Oilprice")
#crime_artikel <- topicframe %>% filter(topic %in% crime_topics) %>% mutate(Cluster="Crime")

#topicmaster <- rbind(klima_artikel, corona_artikel, trump_artikel, korea_artikel, oil_artikel, crime_artikel)
topicmaster <- rbind(klima_artikel, corona_artikel, korea_artikel, oil_artikel)
topicmaster <- topicmaster %>% select(lead, body, Language, Cluster)
topicmaster <- topicmaster %>%
  filter(!is.na(body), !is.na(lead)) %>%
  filter(nchar(body) > 0, nchar(lead) > 0)
write.csv(topicmaster, "Topic_Data.csv", row.names = FALSE)

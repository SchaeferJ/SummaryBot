library(R.utils)
library(jsonlite)
library(dplyr)
library(rvest)
cursor <- "null"
adata <- data.frame()
i <- 1

st <- c("Corona", "Polizei", "Trump", "COVID", "Nordkorea", "Merkel", "Schweden", "%C3%96lpreis", "OPEC", "Klimawandel", "Feuerwehr", "Coburg", "Muenchen", "Nuernberg",
        "Regensburg", "Manching", "Ingolstadt", "Forschung", "Regierung", "Berlin", "Motorrad", "Auto", "Bank", "Vogel", "Fernseher", "Computer",
        "Betrug", "Internet", "Bodensee", "Lindau", "Kuchen", "Bundesliga", "Bier", "Wein", "Verkehr")
for (s in st) {
  for (zz in c(1:3)) {
    q <- paste0("https://graphql-br24.br.de/graphql?operationName=SearchContainerQuery&variables=%7B%22searchTerm%22%3A%22", s, "%22%2C%22count%22%3A100%2C%22cursor%22%3A",
                cursor,
                "%7D&extensions=%7B%22persistedQuery%22%3A%7B%22version%22%3A1%2C%22sha256Hash%22%3A%22650cd0d19012efe8f03652b2a769efe8ed722a627347d115734e95fae8433823%22%7D%7D")
    c <- fromJSON(q)
    dt <- c[["data"]][["searchResultArticles"]][["edges"]][["node"]]
    dt <- dt %>%
      select(path, teaserText, title) %>%
      mutate(topic = s)
    adata <- rbind(adata, dt)
    cursor <- gsub("=", "%3D", c[["data"]][["searchResultArticles"]][["pageInfo"]][["startCursor"]])
    cursor <- paste0("%22", cursor, "%22")
  }
  adata <- adata[!duplicated(adata),]

  message(paste("Iteration", s, "Gefundene Links:", nrow(adata)))
}
write.csv(adata, "BR-Links.csv")


BASE_URL <- "https://www.br.de/nachrichten"
txts <- character()
pb <- txtProgressBar(max = nrow(adata), style = 3)
for (i in c(1:nrow(adata))) {
  url <- paste0(BASE_URL, adata$path[i])
  success <- TRUE
  tryCatch({
             article_page <- read_html(url)
           }, error = function(e) {
    success <<- FALSE
  })
  if (!success) {
    txts <- append(txts, NA)
    next()
  }
  article_text <- article_page %>%
    html_node(".css-1jftgse") %>%
    html_nodes("p") %>%
    html_text() %>%
    paste0(., collapse = " ")
  txts <- append(txts, article_text)
  setTxtProgressBar(pb, i)
}

adata$body <- txts

write.csv(adata, "BR-Artikel.csv")

adata_cleaned <- adata[!grepl("ticker", adata$path),]
adata_cleaned <- adata_cleaned[!grepl("ticker", adata_cleaned$title, ignore.case = TRUE),]
adata_cleaned <- adata_cleaned[nchar(adata_cleaned$body) > 0,]
write.csv(adata_cleaned, "BR-Artikel-Clean.csv")

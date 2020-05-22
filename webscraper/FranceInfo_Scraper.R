library(rvest)
library(dplyr)
library(stringr)
library(httr)

st <- c("coronavirus", "police", "Trump", "COVID", "coree+du+nord", "Merkel", "suede", "petrole", "climat", "pompiers", "avion", "Paris", "Strasbourg", "Alsace",
        "Lorraine", "ordinateur", "Toulouse", "Avignon", "Marseille", "vin", "biere", "science", "internet", "reseau", "sport", "ballon", "maintenant", "demain",
        "ans", "depuis", "alors", "mais", "je", "tu", "toi")

s <- st[1]
options(scipen = 999)
adata <- data.frame()

for (s in st) {
  for (i in c(0:50)) {
    url <- paste0("https://www.francetvinfo.fr/recherche/?request=", s, "&from=", as.character(i * 10))
    success <- TRUE
    tryCatch({
               search_page <- read_html(url)
               urls <- search_page %>%
                 html_nodes('[class="flowItem"]') %>%
                 html_nodes("a") %>%
                 html_attr("href")
               teasers <- search_page %>%
                 html_nodes('[class="flowItem"]') %>%
                 html_nodes('[class="description"]') %>%
                 html_text()
             }, error = function(e) {
      success <<- FALSE
    })
    if (!success) {
      next()
    }
    if (length(urls) == 0) {
      break()
    }
    if (length(urls) != length(teasers)) {
      next()
    }
    adata <- rbind(adata, data.frame(path = urls, teaserText = teasers, topic = s, stringsAsFactors = FALSE))
  }
  adata <- adata[!duplicated(adata),]
  message(paste("Iteration", s, "Gefundene Links:", nrow(adata)))
}


adata <- adata %>% filter(path != "https://la1ere.francetvinfo.fr/polynesie/edition-speciale-covid-19-deconfinement-progressif-partir-du-29-avril-polynesie-francaise-827660.html")
i <- 1
bodies <- character()
titles <- character()

pb <- txtProgressBar(max = nrow(adata), style = 3)

for (i in c(1:nrow(adata))) {
  aurl <- adata$path[i]
  success <- TRUE
  if (!grepl("http", aurl)) {
    tryCatch({
               aurl <- paste0("https://www.francetvinfo.fr", aurl)
               article_page <- read_html(aurl)
               body <- article_page %>%
                 html_node('[id="col-middle"]') %>%
                 html_nodes("p") %>%
                 html_text() %>%
                 paste(., collapse = " ")
               title <- article_page %>% html_node("h1") %>% html_text()
             }, error = function(e) {
      success <<- FALSE
    })
  }else if (grepl("la1ere", aurl)) {
    tryCatch({
               article_page <- read_html(aurl)
               body <- article_page %>%
                 html_nodes(".article__body") %>%
                 html_text() %>%
                 str_squish()
               title <- article_page %>%
                 html_nodes(".article__title") %>%
                 html_text() %>%
                 str_squish()
             }, error = function(e) {
      success <<- FALSE
    })
  }else if (grepl("france3-regions", aurl)) {
    tryCatch({
               article_page <- read_html(aurl)
               body <- article_page %>%
                 html_nodes(".lettrine") %>%
                 html_text()
               title <- article_page %>%
                 html_nodes("h1") %>%
                 html_text() %>%
                 str_squish()
             }, error = function(e) {
      success <<- FALSE
    })
  }else {
    body <- NA
    title <- NA
  }
  if (!success) {
    body <- NA
    title <- NA
  }

  bodies <- append(bodies, body[1])
  titles <- append(titles, title[1])
  setTxtProgressBar(pb, i)
}

adata$title <- titles
adata$body <- bodies

write.csv(adata, "Franceinfo-Artikel.csv")

adata_clean <- adata %>% filter(!is.na(body), nchar(body) > 0)
write.csv(adata_clean, "Franceinfo-Artikel-Clean.csv")

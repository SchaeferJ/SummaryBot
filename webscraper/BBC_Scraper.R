library(httr)
library(rvest)
library(dplyr)
library(R.utils)
#q <- "coronavirus"

adata <- data.frame()
st <- c("coronavirus", "police", "Trump", "COVID", "north+korea", "Merkel", "sweden", "oil+price", "OPEC", "climate+change", "fire", "london", "paris", "berlin",
        "manchester", "bogota", "sydney", "new+york", "government", "caracas", "boris+johnson", "car", "bank", "airbus", "volkswagen", "computer",
        "fraud", "internet", "lufthansa", "bailout", "stock", "united+nations", "brexit", "beer", "traffic")

options(scipen = 999)

for (q in st) {
  for (i in c(1:100)) {
    success <- TRUE
    surl <- paste0("https://www.bbc.co.uk/search?q=", q, "&page=", i)
    tryCatch({
               sp <- read_html(surl)
               nl <- sp %>%
                 html_nodes(".ett16tt7") %>%
                 html_attr("href")
               nl <- nl[grepl("/news/", nl)]
             }, error = function(e) {
      success <<- FALSE
    })
    if (!success) {
      next()
    }
    if (length(nl) > 0) {
      tmp <- data.frame(path = nl, topic = q, stringsAsFactors = FALSE)
      adata <- rbind(adata, tmp)
      adata <- adata[!duplicated(adata),]
    }
  }
  message(paste("Iteration", q, "Gefundene Links:", nrow(adata)))

}
write.csv(adata, "BBC_Articles.csv")

txts <- character()
sums <- character()
titles <- character()

pb <- txtProgressBar(max = nrow(adata), style = 3)

for (i in c(1:nrow(adata))) {
  l <- adata$path[i]

  success <- TRUE
  tryCatch({
             ap <- read_html(l)
             title <- ap %>%
               html_nodes(".story-body") %>%
               html_node(".story-body__h1") %>%
               html_text()
             if (length(title) == 0) {
               title <- NA
               summary <- NA
               text <- NA
             }else {
               summary <- ap %>%
                 html_nodes(".story-body") %>%
                 html_node(".story-body__introduction") %>%
                 html_text()
               text <- ap %>%
                 html_nodes(".story-body") %>%
                 html_nodes("p") %>%
                 html_text()
               text <- text[-c(1:which(text == summary[length(summary)]))]
               text <- paste(text, collapse = " ")
               if (length(text) + length(title) + length(summary) != 3) {
                 print("Huh.")
                 title <- NA
                 summary <- NA
                 text <- NA
               }
             }
           }, error = function(e) {
    success <<- FALSE
  })

  if (success) {
    titles <- append(titles, title)
    sums <- append(sums, summary)
    txts <- append(txts, text)
  }else {
    titles <- append(titles, NA)
    sums <- append(sums, NA)
    txts <- append(txts, NA)
  }
  rm(title, summary, text)
  setTxtProgressBar(pb, i)
}

adata$title <- titles
adata$lead <- sums
adata$body <- txts

write.csv(adata, "BBC-Artikel.csv")

adata <- adata[complete.cases(adata),]
write.csv(adata, "BBC-Artikel-Clean.csv", row.names = FALSE)


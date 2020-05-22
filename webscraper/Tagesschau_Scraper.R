# Tagesschau-Scraper

library(rvest)
library(stringr)
library(dplyr)
library(R.utils)

process_article_page <- function(URL) {
  article_page <- read_html(URL)
  headline <- article_page %>%
    html_node(".headline") %>%
    html_text()
  contents <- article_page %>%
    html_nodes('.small') %>%
    html_text()
  summary <- contents[1]
  contents <- contents[-c(1)]
  article <- paste0(contents[which(substr(contents, 1, 1) == " ")], collapse = "")
  re_val <- character()
  re_val[1] <- str_squish(summary)
  re_val[2] <- str_squish(article)
  re_val[3] <- str_squish(headline)
  return(re_val)
}

make_archive_url <- function(current_date) {
  base_url <- "https://www.tagesschau.de/archiv/meldungsarchiv100~_date-"
  url_date <- gsub("-", "", current_date)
  return(paste0(base_url, url_date, ".html"))
}

get_links_from_archive <- function(archive_url) {
  archive_page <- read_html(archive_url)
  archive_page %>%
    html_nodes(xpath = "/html/body/div[3]/div/div/div/div/div/div/div/div/ul") %>%
    html_attr("href")
  links <- archive_page %>%
    html_nodes("li") %>%
    html_nodes("a") %>%
    html_attr('href')
  links <- links[grepl("\\d{3}\\.html", links)]
  linklist <- links[!grepl("archiv|http", links)]
  return(linklist)
}


dates <- character()
headlines <- character()
summaries <- character()
articles <- character()
urls <- character()
current_date <- Sys.Date() - 364
i <- 1

pb <- txtProgressBar(max = 364, style = 3)
while (TRUE) {
  if (current_date == Sys.Date()) {
    message("\nFinished")
    break
  }
  message(paste("\nProcessing", as.character(current_date)))
  archive_url <- make_archive_url(current_date)
  linklist <- get_links_from_archive(archive_url)
  for (link in linklist) {
    current_url <- paste0("https://www.tagesschau.de", link)
    page_contents <- process_article_page(current_url)
    dates <- append(dates, as.character(current_date))
    summaries <- append(summaries, page_contents[1])
    articles <- append(articles, page_contents[2])
    headlines <- append(headlines, page_contents[3])
    urls <- append(urls, current_url)
  }
  current_date <- current_date + 1
  i <- i + 1
  setTxtProgressBar(pb, i)
}
close(pb)
newsdf <- data.frame(publisher = "Tagesschau", type = "Scrape", item_url = urls, author = NA, headline = headlines, lead = summaries, body = articles, first_published = dates, scraped_at = as.character(Sys.Date()),
                     stringsAsFactors = FALSE)

newsdf <- newsdf %>% filter(body != "", lead != "")

write.csv(newsdf, "TS-Artikel.csv")

#write_article_to_db(newsdf)

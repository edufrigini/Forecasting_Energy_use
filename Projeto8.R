# DSA - DATA SCIENCE ACADEMY 
# FORMACAO CIENTISTA DE DADOS
# MACHINE LEARNING
#
# PROJETO 8, Modelagem Preditiva em IoT - Previsão de Uso de Energia
# ALUNO: EDUARDO FRIGINI DE JESUS 

# Goal: criação de modelos preditivos para a previsão de consumo de energia de eletrodomésticos.

setwd("D:/FCD/Machine_Learning/Projeto8")
getwd()

# carregando as bibliotecas, se nao estiver instalada, instalar install.packages("nome do pacote")
library(data.table) # para usar a fread
library("gmodels") # para usar o CrossTable
library(psych) # para usar o pairs.panels
library(lattice) # graficos de correlacao
require(ggplot2)
library(randomForest)
library(DMwR)
library(dplyr)
library(tidyr)
library("ROCR")
library(caret)
library(lattice)
library(corrplot)
library(corrgram)

##  Carregando os dados na memoria
# Usando o arquivo projeto8-training.csv para treinar o modelo para producao
treino <- fread("projeto8-training.csv", sep = ",", header = TRUE, stringsAsFactors = TRUE)
teste <- fread("projeto8-testing.csv", sep =",", header = TRUE, stringsAsFactors = TRUE)

# View(treino)
# View(teste)
str(treino)
# str(teste)

#######################################################################################
###    Tratando os dados
#######################################################################################

## Convertendo variavel date para o tipo data
to.data <- function(df, variables){
  for (variable in variables){
    df[[variable]] <- as.Date(df[[variable]])
  }
  return(df)
}

# Chama a função que converte para o formato data
coluna_data <- c("date")
treino <- to.data(df = treino, variables = coluna_data)
str(treino)

## Convertendo outras variaveis tipo fator para numericas
to.numerico <- function(df, variables){
  for (variable in variables){
    df[[variable]] <- as.numeric(df[[variable]])
  }
  return(df)
}

# Chama a função
colunas_num <- c("Appliances", "T6", "RH_6", "T_out", "RH_out", "Windspeed", "Visibility", "Tdewpoint", "rv1", "rv2", "NSM")
treino <- to.numerico(df = treino, variables = colunas_num)
str(treino)

# corrigir caso hajam dados NA
treino <- na.omit(treino)
str(treino)
head(treino)


###########################################################################################
###    Analise exploratoria dos dados
###########################################################################################

plot(treino$Appliances, main = "Appliances")
plot(treino$lights, main = "Lights")
hist(treino$T1, main = "T1")
hist(treino$RH_1, main = "RH_1")

summary(treino$Appliances)

#mean(treino$Appliances) # 9.798
#median(treino$Appliances) # 6
quantile(treino$Appliances) # 0%  25%  50%  75% 100% 
#                             1    5    6   10   88 
#quantile(treino$Appliances, probs = c(0.01, 0.95)) # 1% = 2.00 e 95% = 33
#quantile(treino$Appliances, seq(from = 0, to = 1, by = 0.10))
#IQR(treino$Appliances) # diferenca entre Q3 e Q1 = 4 --> IQR = 5
#range(treino$Appliances) # 1 a 88
#diff(range(treino$Appliances)) # 87
#sd(treino$Appliances) # 10.26
#var(treino$Appliances) # 105.2819
plot(treino$Appliances)

# A variavel alvo esta com muitos outlyers, dai vou restringir ao valores menores que 0.90 quartil
quartil_90 <- quantile(treino$Appliances, probs = 0.90)
class(quartil_90)
quartil_90[[1]]
treino_sem_ouliers <- treino[Appliances<=quartil_90[[1]],]
plot(treino_sem_ouliers$Appliances)

sd(treino_sem_ouliers$Appliances) #  3.138
var(treino_sem_ouliers$Appliances) # 9.84

## OS DADOS DO TARGET NAO SEGUEM UMA DISTRIBUICAO NORMAL

## Explorando os dados graficamente
hist(treino_sem_ouliers$Appliances)

plot(treino_sem_ouliers$Appliances) # target e esta com outliers
hist(treino_sem_ouliers$Appliances) # dados concentrados no zero
boxplot(treino_sem_ouliers$Appliances)
str(treino_sem_ouliers)

########################################################################
###     FEATURE SELECTION
########################################################################

# Explorando os dados
# variaveis numericas
cols<- c("Appliances", "T1", "RH_1", "T2", "RH_2", "T3", "RH_3", "T4", "RH_4", "T5", "RH_5", "T6", "RH_6", "T7", "RH_7", "T8", "RH_8", "T9", "RH_9", "T_out","Press_mm_hg", "RH_out", "Windspeed", "Visibility", "Tdewpoint", "rv1", "rv2", "NSM")

correlacao = cor(treino_sem_ouliers[, cols, with=FALSE])
correlacao = correlacao[1,]
correlacao

# Variaveis mais relevantes, considerando a Correlacao absoluta > 0.20
# Sao na ordem
# NSM	0.390787903
# T2	0.245819094
# T8 	0.235186419
# T1	0.224515771
# RH_out	-0.237240955
# RH_6	-0.212255449

# Verificando a matriz de correlacao das variaveis mais relevantes
col_selecao <- c("Appliances", "T1", "T2", "RH_6", "T8", "RH_out", "NSM")

# Vetor com os métodos de correlação
metodos <- c("pearson", "spearman")

# Aplicando os métodos de correlação com a função cor()
cors <- lapply(metodos, function(method) 
  (cor(treino_sem_ouliers[, col_selecao, with=FALSE], method = method)))

head(cors)

# AS TEMPERATURAS SAO COLINEARES

# Preparando o plot
plot.cors <- function(x, labs){
  diag(x) <- 0.0 
  plot( levelplot(x, 
                  main = paste("Plot de Correlação usando Método", labs),
                  scales = list(x = list(rot = 90), cex = 1.0)) )
}

# Mapa de Correlação
Map(plot.cors, cors, metodos)


# Avalidando a importância de todas as variaveis com o RANDOM FOREST
modelo <- randomForest(Appliances ~ . , 
                       data = treino_sem_ouliers, 
                       ntree = 100, 
                       nodesize = 10,
                       importance = TRUE)
# Plotando as variáveis por grau de importância
varImpPlot(modelo)

View(modelo)
View(modelo$importanceSD)
str(treino_sem_ouliers)

# Fazer uma copia do Data Set limpo caso haja algum problema depois
treino_ok <- treino_sem_ouliers


# Retirando variaveis menos importantes para o modelo  %IncMSE <= 1.6
treino_ok$rv1 <- NULL
treino_ok$rv2 <- NULL
treino_ok$WeekStatus <- NULL
treino_ok$Visibility <- NULL
treino_ok$Tdewpoint <- NULL
treino_ok$Windspeed <- NULL
treino_ok$T_out <- NULL
treino_ok$RH_5 <- NULL
treino_ok$T4 <- NULL
treino_ok$RH_out <- NULL
treino_ok$RH_7 <- NULL
treino_ok$T5 <- NULL
treino_ok$Press_mm_hg <- NULL
treino_ok$T7 <- NULL
treino_ok$RH_2 <- NULL
treino_ok$RH_4 <- NULL
treino_ok$T6 <- NULL

# Foram selecionadas 14 variaveis para criar o modelo preditivo, as que tiveram maior correlacao com a variavel alvo: Appliances
str(treino_ok)
hist(treino_ok$Appliances)
plot(treino_ok$lights)
hist(treino_ok$T1)
hist(treino_ok$T2)
hist(treino_ok$T3)
hist(treino_ok$T8)
hist(treino_ok$T9)
hist(treino_ok$RH_1)
hist(treino_ok$RH_3)
hist(treino_ok$RH_6)
hist(treino_ok$RH_8)
hist(treino_ok$RH_9)
hist(treino_ok$NSM)
plot(treino_ok$Day_of_week)

# Variaveis numericas selecionadas
# cols<- c("T1", "RH_1", "T2", "T3", "RH_3", "RH_6", "T8", "RH_8", "T9", "RH_9", "NSM")


# Verificando se as temperaturas sao colineares
cols<- c("T1","T2", "T3", "T8", "T9")

correlacao = cor(treino_ok[, cols, with=FALSE])
correlacao = correlacao[1,]
correlacao


# As temperaturas sao colineares, manter apenas T2 que tem correlacao forte com Appliance
treino_ok$T1 <- NULL
treino_ok$T3 <- NULL
treino_ok$T8 <- NULL
treino_ok$T9 <- NULL

# Variaveis selecionadas
cols<- c("RH_1", "T2","RH_3", "RH_6", "RH_8", "RH_9", "NSM")

# Verificar se essas variaveis sao colineares
correlacao = cor(treino_ok[, cols, with=FALSE])
correlacao = correlacao[1,]
correlacao
# Muito pesado para rodar
# pairs.panels(treino_ok[, cols, with=FALSE])

# RH1 tem colinearidade com RH3, RH6, RH8, RH9
# Manter o RH6 que tem correlacao mais forte com Appliances
treino_ok$RH_1 <- NULL
treino_ok$RH_3 <- NULL
treino_ok$RH_8 <- NULL
treino_ok$RH_9 <- NULL
treino_ok$date <- NULL

# Variaveis selecionadas continuas 
cols<- c("T2", "RH_6", "NSM")

# Verificar se essas variaveis sao colineares
correlacao = cor(treino_ok[, cols, with=FALSE])
correlacao = correlacao[1,]
correlacao
# Muito pesado para rodar
# pairs.panels(treino_ok[, cols, with=FALSE])

# T2 e RH_6 sao colineares, mas vou manter no modelo

# Modelo de randomForest com variaveis numericas selecionadas e as variaveis categoricas
modelo_RF <- randomForest(Appliances ~ . , 
                       data = treino_ok, 
                       ntree = 100, 
                       nodesize = 10,
                       importance = TRUE)
varImpPlot(modelo_RF)

modelo_RF$importance

# Previsões
x <- subset(treino_ok, select = -Appliances)
y <- treino_ok$Appliances

pred <- predict(modelo_RF, x)
#pred <- fitted(modelo_RF)

length(pred)
length(treino_ok$Appliances)
plot(pred, treino_ok$Appliances)
cor(pred, treino_ok$Appliances)
# Cor = 0.78


##########################################################################################
###    Criando o modelo de Regressao Logistica Multipla
##########################################################################################

# Carregando os pacotes
library(caret)
library(ROCR) 
library(e1071) 

plot(treino_ok$Appliances)

# Normalização
scale.features <- function(df, variables){
  for (variable in variables){
    df[[variable]] <- scale(df[[variable]], center = T, scale = T)
  }
  return(df)
}

str(treino_ok)


# Normalizando as variáveis numericas
numeric.vars <- c("T2", "RH_6", "NSM")
treino_ok_n <- scale.features(treino_ok, numeric.vars)

median(treino_ok$Appliances) # = 6

# Para utilizar a Regressao Logistica é necessário separar os dados do Applience em duas classes
# Acima de 6 = Consumo alto de energia = 1
# Abaixo de 6 = Consumo baixo de energia = 0
treino_ok$Appliances_bin <- as.factor(ifelse(treino_ok$Appliances>=6, 1, 0))

plot(treino_ok$Appliances_bin)

str(treino_ok)
treino_ok_n <- treino_ok
treino_ok_n$Appliances <- NULL
str(treino_ok_n)

# Preparando os dados de treino e de teste
indexes <- sample(1:nrow(treino_ok_n), size = 0.6 * nrow(treino_ok))
train.data <- treino_ok_n[indexes,]
test.data <- treino_ok_n[-indexes,]

head(test.data)
# Separando os atributos e as classes
test.feature.vars <- test.data[,-6]
head(test.feature.vars)

test.class.var <- test.data[,6]
head(test.class.var)

# Construindo o modelo de regressão logística
formula.init <- "Appliances_bin ~ ."
formula.init <- as.formula(formula.init)
# help(glm)
modelo_LM <- glm(formula = formula.init, data = treino_ok_n, family = "binomial")

# Visualizando os detalhes do modelo
summary(modelo_LM)


# Fazendo previsões e analisando o resultado
previsoes <- predict(modelo_LM, test.data, type = "response")
previsoes <- round(previsoes)
previsoes <- as.factor(previsoes)
length(previsoes)
length(test.class.var$Appliances_bin)


confusionMatrix(previsoes, test.class.var$Appliances_bin)
# Accuracy = 0.71 com 5 variaveis preditoras
str(test.feature.vars)


##########################################################################################
###    Criando o modelo SVM
##########################################################################################

library(e1071)

str(treino_ok_n)

modelo_SVM <- svm(Appliances_bin ~ ., data = treino_ok_n)
print(modelo_SVM)
summary(modelo)


# Previsões no modelo SVM
x <- subset(treino_ok_n, select = -Appliances_bin)
y <- treino_ok_n$Appliances_bin

pred_SVM <- predict(modelo_SVM, x)

head(pred_SVM)

pred_SVM <- fitted(modelo_SVM)
head(pred_SVM)

# Checando a acurácia
table(pred_SVM, y)
plot(pred_SVM)

confusionMatrix(pred_SVM, treino_ok_n$Appliances_bin)
# Accuracy = 0.7428 com poucas variaveis



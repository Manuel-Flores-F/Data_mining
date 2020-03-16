%TAREA 3 NAIVE BAYES
%Enviar a raquel.torresperalta@gmail.com
%subject: TAREA MDD Bayes

%LAB 3. Naive Bayes Clasificador

% En este lab programaremos un clasificador capaz de predecir el

% diagn?stico de un caso nunca antes visto en base a los casos previamente

% observados.

addpath 'C:/Users/leunam/Documents/UNISON/Minería_de_Datos/Bayes'
load('CasosTrainTestWisconsin.mat')

Train(:,1) = []
Test(:,1) = []

% 4 Maligno, 2 Beningno
Map_Maligno = Train(:,size(Train,2))==4;
Malignos=Train((Map_Maligno), :);


Map_Benigno = Train(:,size(Train,2))==2;
Benignos=Train((Map_Benigno), :);

Num_malignos=size(Malignos, 1);
Num_Benignos=size(Benignos, 1);



% Paso 1: Calcule la probabilidad de cada clase (Malignos, Benignos) usando la
% funci?n prob

function proby = Probabilidad(Casos,a,b)
  [m,n] = size(Casos)
  proby = sum(Casos(:,a)==b)/m;
end

Prob_M = Probabilidad( Train , size(Train,2), 4 );
Prob_B = 1-Prob_M;



% Creamos una tabla para contabilizar las instancias para cada valor en los
% distintos campos. Haremos esto por separado para cada clase
% Generamos un array con los valores únicos del campo 2 (clump thikness)
%% Los valores de la segunda columna de Malignos (A ellos vemos los unicos,
%% Tambien debo ver cuantas veces  se repiten

#A=unique(Malignos(:, 2));
% Paso 2: Contamos cu?ntas instancias ocurren para cada caso (1-10)

#col=2;
#Mtx=Malignos;
#for c=1:size(A,1)
#A(c,col)= sum(Mtx(:, col)== A(c,1));
#end

% Modifique este c?digo para que sea una funci?n que regresa el 
% conteo para todas las columnas.

function [A] = Counting( Matrix, Cols, values )

% Cuenta las instancias de cada valor en las columnas indicadas en el

% arreglo cols.

% values es un arreglo con los posibles valores para las columnas
[m,n]=size(Cols)
A = zeros(values,n)

for value = values
 A = cat(1,A,sum(Matrix(:,Cols)==value))
end
A(1,:)=[]
A =cat(2,transpose(values),A)
% Matrix es la matriz de datos

% Cols es un arreglo con el num de columnas a evaluar

% NOTA: Esta funci?n sirve cuando todas las columnas est?n en el mismo

% rango de datos

end




Cols=[1,2,3,4,5,6,7,8,9];
M_U=repmat(Malignos,1)
B_U=repmat(Benignos,1)

%En este caso uso M_U pero en general debería usar todas las variables
%Coinciddeen que M_U tiene todas las variables

values=unique(M_U)';


% o values=[1,2,3,4,5,6,7,8,9,10]' o values=[1:10]'
Conteo_M = Counting( Malignos, Cols, values );
Conteo_B = Counting( Benignos, Cols, values );

% Paso 3:Suma 1 a cada valor de la tabla para evitar errores a la hora de
% normalizar por valores en 0

Conteo_M2 = [Conteo_M(:,1),Conteo_M(:, 2:10) + 1];
Conteo_B2 = [Conteo_B(:,1),Conteo_B(:, 2:10) + 1];

%%%%%%%%%%%%%%%%

% Paso 4: Normalizaci?n. Cada celda se divide por la suma de los valores 
% de la columna, de modo que tengamos valores entre 0 y 1.

% La suma de los valores en este caso ser? igual al num de registros

% original + 10, porque son 10 valores. Es lo mismo que sumar todas las

% instancias de los valores, porque a conteo de instancias del 1 al 10 le sumamos 1

NormFac_M=sum(Conteo_M2(:, 2)); % Es el mismo valor para todas las columnas, as? que tomamos una al azar.
NormFac_B=sum(Conteo_B2(:, 2));

Conteo_M_Norm = [Conteo_M(:,1),Conteo_M2(:, 2:10)/NormFac_M];
Conteo_B_Norm = [Conteo_B(:,1),Conteo_B2(:, 2:10)/NormFac_B];

#Conteo_M_Norm
%Calcule el Conteo_B_Norm para los benignos


%Intentaremos predecir la probabilidad de malignidad de una muestra del set

%de test. Después la probabilidad de ser un tumor benigno y se le asignará

%un diagnóstico basado en la probabilidad más alta.

% Tomaremos la primera muestra del set test



s = Test;
Acuracy=repmat(Test,1)
Acuracy =cat(2,Acuracy,zeros(size(Acuracy,1),2))
num_cols = size(Cols, 2);
num_fils = size(Test, 1);



num_reg=1;

for num_reg= 1:num_fils
  
ProbsM=1;
ProbsB=1;


%empezamos desde la 2da columna para saltarnos el id
for col= 1:num_cols

    valor= s(num_reg, col);

    idx_m = find(Conteo_M_Norm(:, 1)==valor);
    idx_b = find(Conteo_B_Norm(:, 1)==valor);
    %% La matriz CONTEO_M_Norm tiene el atributo ID, por ello 
    %% Es que se suma 1 en col + 1
    ProbsM = ProbsM * Conteo_M_Norm(idx_m, col+1);
    ProbsB = ProbsB * Conteo_B_Norm(idx_b, col+1);


end

  if ProbsB >= ProbsM
      Acuracy(num_reg,size(Acuracy,2)-1 ) = 2;
      if Acuracy(num_reg,size(Acuracy,2)-2 ) == 2
        Acuracy(num_reg,size(Acuracy,2)) = 2;
      else
        Acuracy(num_reg,size(Acuracy,2)) = 4;
       end
       
  else
      Acuracy(num_reg,size(Acuracy,2)-1 ) = 4;
      if Acuracy(num_reg,size(Acuracy,2)-2 ) == 4
        Acuracy(num_reg,size(Acuracy,2)) = 3;
      else
        Acuracy(num_reg,size(Acuracy,2)) = 1;
      end
  end
      

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Nuestro software querra detectar un tumor Maligno
%%% Por lo que si un tumor maligno lo califica como maligno
%%% estaría calisificandolo de manera correcta (TT) 
%%% Por el contrario si a un tumor benigno lo detecta como
%%% Maligno, entonces estamos ante el caso de una Falso 
%%% Positivo
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

TT=sum(Acuracy(:,size(Acuracy,2))== 3)
TF=sum(Acuracy(:,size(Acuracy,2))== 4)
FT=sum(Acuracy(:,size(Acuracy,2))== 1)
FF=sum(Acuracy(:,size(Acuracy,2))== 2)

Reporte_Acuracy=(TT+FF)*100/size(Acuracy,1);




% En este c?digo consideramos que la primera col de las matrices contienen

% etiquetas (valores ?nicos o id), por eso los c?lculos los hacemos desde

% la col 2. Cada base de datos es distinta, este c?digo es para el ejemplo con el que estamos trabajando.

% Despu?s mediremos la efectividad del clasificador sacando un porcentaje
% de acierto.

% Cambios extraños hay en mi

%% Normalizar matriz de test

%% Eliminar duplicados



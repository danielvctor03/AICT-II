close all;
clear all;
clc;
% Metodo da distancia Euclidiana

% Leitura dos arquivos de dados (sinal normal e ruido)

fr = fopen("AICT I/fm_99.1_a1"); % Leitura do arquivo com mensagem
vr = fread(fr,4000000,'float'); % Criacao do vetor de dados


fsm = fopen("AICT I/sem_msg_3"); % Leitura do arquivo sem mensagem (ruido)
vsm = fread(fsm,4000000,'float'); % Criacao do vetor de dados


complex_v_r = vr(1:2:end,:) + vr(2:2:end,:)*i;% Vetor complexo do sinal
complex_v_sm = vsm(1:2:end,:) + vsm(2:2:end,:)*i;% Vetor complexo do ruido

fclose('all');

N = 20; % Quantidade de amostras de dados


autocorr(complex_v_r) % Plotagem da autocorrelacao do sinal 88.9
autocorr(complex_v_sm) % Plotagem do ruido 1

% Reorganizando os dados 
FmrData = reshape(complex_v_r,[],N); 
sm_Data = reshape(complex_v_sm,[],N);


NumLags = 20; % Atrasos


atc_r = zeros(NumLags + 1,N); % Autocorrelacao do sinal
atc_sm = zeros(NumLags + 1,N); % Autocorrelacao do ruido


for i = 1:N

    [atc_r(:,i)] = autocorr(FmrData(:,i) , NumLags);
    [atc_sm(:,i) , lags] = autocorr(sm_Data(:,i) , NumLags);

end

R = ((-1*lags)./N) + 1; % Reta de referencia

figure(2);
stem(lags, atc_r(:,1),'r.', 'MarkerSize', 13)
xlabel('Lags');
ylabel('Sample Autocorrelation');
title('Sample Autocorrelation Function');
grid on;


figure(3);
stem(lags, atc_sm(:,1), 'r.', 'MarkerSize', 13)
xlabel('Lags');
ylabel('Sample Autocorrelation');
title('Noise Sample Autocorrelation Function');
grid on;


[y1 , x1] = autocorr(complex_v_r); % Matriz de autocorrelacao do sinal
[y2 , x2] = autocorr(complex_v_sm); % Matriz de autocorrelacao do ruido


D1 = sqrt(sum((y1 - R) .^ 2)) % Distancia do sinal existente para a reta
D2 = sqrt(sum((y2 - R) .^ 2)) % Distancia do ruido para a reta


figure(4);
hold on
plot(lags,R, lags , y1);


plot(lags,y2);
xlabel('Lags');
ylabel('Autocorrelation');
title('Autocorrelation Function');
grid on;
legend('Reference' , 'Fm 99.1' , 'No message (noise)')
hold off




































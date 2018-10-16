function [ y ] = gaola(x)
y=(1./(1+exp(-x))-0.5)./(2*x+1e-30);

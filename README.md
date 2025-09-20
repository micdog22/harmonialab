# HarmoniaLab — Key, Chords & Sections Analyzer

HarmoniaLab estima tom/escala, sequência de acordes e segmenta a música em seções prováveis. Exporta relatório HTML, diagrama de acordes e MIDI de camadas de apoio.

## Funcionalidades
- Key/escala por janela com suavização temporal.
- Estimativa de acordes triádicos por batida e probabilidade.
- Segmentação por novidade harmônica (fluxo espectral da cromagram).
- Exporta relatório HTML e MIDI de acordes.
- CLI simples e configurável.

## Instalação
```bash
pip install -r requirements.txt
```

## Uso
```bash
# análise completa com relatório e MIDI
python src/harmonialab.py analyze audio/minha_musica.wav --report --midi out.mid
```

## Limitações
É uma abordagem baseada em cromagramas e templates; músicas complexas, modulações frequentes e tensões avançadas podem exigir ajustes e revisão manual.

## Licença
MIT.

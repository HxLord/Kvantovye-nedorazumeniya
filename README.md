# HELIOS GUARD 

## Project Description

**Helios Guard** - это MVP-система защиты спутников от солнечных бурь и космической погоды. Система реализует полный цикл обнаружения, анализа и защитого маневра.


### Назначение

Система предназначена для демонстрации концепции TinyML-based защита спутников:
- Мониторинг солнечной активности в реальном времени
- Раннее обнаружение опасных событий (вспышки Х-Класса)
- Расчет времени прибытия частиц (ТоА)
- Автоматическое выполнение защитных команд

### Ключевые возможности

| Возможность | Описание |
|-------------|----------|
| SensorUnit | Генерация/чтение данных рентгеновского потока |
| Tiny Trigger | Низкопотребляющий пороговый детектор |
| MissionAI | Расчет ТоА и команды защиты |
| Energy Simulation | Моделирование энергопотребления |
| Real-time Visealization | График активности в Matplotlib |

---

## Project Structure

```
solar_shield_ai/
├── main.py                 # Main application (все компоненты)
├── main_visualization.py   # Visualization application
├── README.md               # Этот файл
├── docs/
│   └── technical_doc.md    # Техническая документация
└── data/
    └── (генерируемые данные) 
```

---

## Инструкции по запуску

### Требования

- Python 3.8+
- numpy
- matplotlib

### Установка зависимостей

```bash
pip install numpy matplotlip
```

### Запуск

```bash
cd solar_shiel_ai
python main.py
```

### Ожидаемый вывод

```
+==============================================================================+
|   HELIOS GUARD - SATELLITE PROTECTION SYSTEM                               |
+==============================================================================+
|  DETECTION ======> ANALYSIS ======> MANEUVER                               |
+==============================================================================+

19:25:27 | HELIOS GUARD SYSTEM - INITIALIZATION
19:25:27 | Distance to Sun: 150,000,000 km (1 AU)
19:25:27 | X-class flare threshold: 1.00e-04 W/m^2
19:25:27 | Sleep mode power: 0.01W
19:25:27 | AI Analysis power: 2.0W

19:25:27 | [✓] SensorUnit: X-ray flux monitor
19:25:27 | [✓] TinyML_Trigger: Threshold gate
19:25:27 | [✓] MissionAI: Threat analysis & maneuver control

19:25:27 | SYSTEM ONLINE - Entering MONITORING mode

... (мониторинг и обнаружение вспышек) ...

19:25:33 | HELIOS GUARD SYSTEM - OPERATION SUMMARY
19:25:33 | Total sensor readings: 11
19:25:33 | Total AI analyses: 2
19:25:33 | Total maneuvers executed: 2
19:25:33 | Total power consumed: 2.0450 Wh
```

---


## Visualization

### Описание графика

При запуске открывается окно Matplotlib с графиком солнечной активности:

- **Ось Х**: Время(номер итерации)
- **Ось Y**: Рентгеновсский поток (W/m²) - логарифмическая шкала
- **Горизонтальная линия**: Порог Х-класса (1е-4 W/M²)
- **Точки**: Измерения (цвет по классу вспышки)
- **Красные вертикальные линии**: Моменты срабатывания защиты

### Управление

График обновляется в реальном времени. Закройте окно для завершения demo.

---

### Энергопотребление

| Режим | Мощность |
|-------|----------|
| Sleep Mode | 0.01 W |
| AI Analysis | 2.0 W |
| Standby | 0.05 W |

---

## Лицензия

MIT Licence

---

## Автор

**Helios guard MVP**
Aerospace Software Engineering Demo TinyML - based Solar Storm Protection System

---

## Поддержка

Для вопросов и предложений:
- Создайте issue в репозитории
- Откройте PR с исправлениями

---

**Версия**: 1.0.0 Beta
**Дата**: 2026-02-27
**Автор**: Helios guard MVP
**Лицензия**: MIT

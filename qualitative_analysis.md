# Imperceptibility Analysis with Qualitative Properties

## Immutability

### Selected Features:

- Adult: `race`, `sex`, `marital-status`
- COMPAS: `sex`, `race`
- German: `personal_status_sex`, `foreign_worker`

### Codes

Adult:
```excel
=IF(EXACT([@[origin_input_race]],[@[origin_adv_race]]),0,1)
=IF(EXACT([@[origin_input_sex]],[@[origin_adv_sex]]),0,1)
=IF(EXACT([@[origin_input_marital-status]],[@[origin_adv_marital-status]]),0,1)
```

COMPAS:
```excel
=IF(EXACT([@[origin_input_race]],[@[origin_adv_race]]),0,1)
=IF(EXACT([@[origin_input_sex]],[@[origin_adv_sex]]),0,1)
```

German:
```excel
=IF(EXACT([@[origin_input_personal_status_sex]],[@[origin_adv_personal_status_sex]]),0,1)
=IF(EXACT([@[origin_input_foreign_worker]],[@[origin_adv_foreign_worker]]),0,1)
```

### Results:

#### Adult:
| Model     | Attack   | `race` | `sex` | `marital-status` |
|-----------|----------|--------|-------|------------------|
| LR        | DeepFool | 17     | 16    | 472              |
| LR        | C&W      | 0      | 0     | 0                |
| LR        | FGSM     | 0      | 0     | 0                |
| LR        | PGD      | 0      | 0     | 0                |
| LinearSVC | DeepFool | 0      | 0     | 0                |
| LinearSVC | C&W      | 0      | 0     | 0                |
| LinearSVC | FGSM     | 0      | 0     | 0                |
| LinearSVC | PGD      | 0      | 0     | 0                |
| MLP       | DeepFool | 648    | 345   | 1096             |
| MLP       | C&W      | 0      | 0     | 0                |
| MLP       | FGSM     | 0      | 0     | 0                |
| MLP       | PGD      | 0      | 0     | 0                |

#### COMPAS:
| Model     | Attack   | `sex` | `race` |
|-----------|----------|-------|--------|
| LR        | DeepFool | 0     | 8      |
| LR        | C&W      | 0     | 0      |
| LR        | FGSM     | 0     | 0      |
| LR        | PGD      | 0     | 0      |
| LinearSVC | DeepFool | 0     | 0      |
| LinearSVC | C&W      | 0     | 0      |
| LinearSVC | FGSM     | 0     | 0      |
| LinearSVC | PGD      | 0     | 0      |
| MLP       | DeepFool | 36    | 16     |
| MLP       | C&W      | 0     | 0      |
| MLP       | FGSM     | 0     | 0      |
| MLP       | PGD      | 0     | 0      |

#### German:
| Model     | Attack   | `personal_status_sex` | `foreign_worker` |
|-----------|----------|-----------------------|------------------|
| LR        | DeepFool | 3                     | 2                |
| LR        | C&W      | 0                     | 0                |
| LR        | FGSM     | 0                     | 0                |
| LR        | PGD      | 0                     | 0                |
| LinearSVC | DeepFool | 0                     | 0                |
| LinearSVC | C&W      | 0                     | 0                |
| LinearSVC | FGSM     | 0                     | 0                |
| LinearSVC | PGD      | 0                     | 0                |
| MLP       | DeepFool | 25                    | 1                |
| MLP       | C&W      | 0                     | 0                |
| MLP       | FGSM     | 0                     | 0                |
| MLP       | PGD      | 0                     | 0                |

# DiagFind Filter

## Description

診断ステータスの配列から条件が一致する要素を取り出します。
このフィルターは `DiagnosticsStatus[]` が入力される前提で作成されています。

## Format

| Name  | Type   | Required | Description                                          |
| ----- | ------ | -------- | ---------------------------------------------------- |
| class | string | yes      | プラグインの固有名称である `DiagFind` を指定します。 |
| name  | string | yes      | 検索する診断ステータスの `name` を設定します。       |

## Examples

```yaml
widgets:
  - class: Simple
    input: { class: subscription, topic: /diagnostics, field: status }
    rules:
      - { class: DiagFind, name: /external/remote_commanad }
      - { class: Access, field: level, fails: 9 } # 9 is dummy level
      - class: SetFirstIf
        type: uint
        list:
          - { eq: 0, value: OK }
          - { eq: 1, value: WARN }
          - { eq: 2, value: ERROR }
          - { eq: 3, value: STALE }
          - { value: ----- }
```

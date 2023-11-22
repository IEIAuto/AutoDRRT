# Access Filter

## Description

データの特定のフィールドにアクセスするフィルターを作成します。

## Format

| Name  | Type                    | Required | Description                                        |
| ----- | ----------------------- | -------- | -------------------------------------------------- |
| class | string                  | yes      | プラグインの固有名称である `Access` を指定します。 |
| field | string / list of string | yes      | アクセスしたいフィールドを指定します。             |
| fails | object                  | no       | アクセスの失敗時に代用する値を指定します。         |

## Note

階層的にフィールドにアクセスする場合は `[foo, bar, baz]` のように配列で指定します。
また、配列にアクセスする際は `[data, 0]` のようにインデックスを独立したフィールドとして扱います。

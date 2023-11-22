# ConfigFile

## Description

コンフィグファイルを構成する最上位のオブジェクトです。コンフィグファイル全体は辞書型のデータである必要があります。

## Format

| Key Name      | Value Type                                               | Description                                                     |
| ------------- | -------------------------------------------------------- | --------------------------------------------------------------- |
| version       | string                                                   | コンフィグファイルのバージョン。現在は 2.0 のみ対応しています。 |
| stylesheets   | list of [Stylesheet](../common/stylesheet.md)            | 使用する stylesheet を指定します。                              |
| widgets       | list of [Widget](../common/widget.md)                    | 使用する widget を定義します。                                  |
| streams       | list of [Stream](../common/stream.md)                    | 使用する stream を定義します。                                  |
| filters       | list of [Filter](../common/filter.md)                    | 使用する filter を定義します。                                  |
| subscriptions | list of [TopicDefinition](../others/topic-definition.md) | 使用する subscription を定義します。                            |

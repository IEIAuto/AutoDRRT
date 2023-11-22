// Copyright 2021 Takagi, Isamu
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef CORE__RVIZ__RVIZ_PANEL_HPP_
#define CORE__RVIZ__RVIZ_PANEL_HPP_

#include "rviz_manager.hpp"
#include <rviz_common/panel.hpp>
#include <QWidget>
#include <memory>
#include <string>

class QMouseEvent;
class QLineEdit;
class QPushButton;

namespace multi_data_monitor
{

class MultiDataMonitor;

class SettingWidget : public QWidget
{
  Q_OBJECT

public:
  explicit SettingWidget(MultiDataMonitor * panel);
  void save(rviz_common::Config config) const;
  void load(const rviz_common::Config & config);
  std::string getPath() const;

private:
  QLineEdit * config_path_edit_;
  QPushButton * config_path_load_;
};

class MultiDataMonitor : public rviz_common::Panel
{
  Q_OBJECT

public:
  explicit MultiDataMonitor(QWidget * parent = nullptr);
  void save(rviz_common::Config config) const override;
  void load(const rviz_common::Config & config) override;
  void onInitialize() override;
  void mousePressEvent(QMouseEvent * event) override;
  void updateMultiDataMonitor();

private:
  RvizManager manager_;
  QWidget * monitor_;
  SettingWidget * setting_;
};

}  // namespace multi_data_monitor

#endif  // CORE__RVIZ__RVIZ_PANEL_HPP_

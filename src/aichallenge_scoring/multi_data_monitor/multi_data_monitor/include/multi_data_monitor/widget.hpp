// Copyright 2022 Takagi, Isamu
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

#ifndef MULTI_DATA_MONITOR__WIDGET_HPP_
#define MULTI_DATA_MONITOR__WIDGET_HPP_

#include <multi_data_monitor/packet.hpp>
#include <string>
#include <vector>

class QString;
class QWidget;
class QLayout;

namespace multi_data_monitor
{

class BasicWidget
{
protected:
  virtual void setup(YAML::Node yaml, const std::vector<QWidget *> & items) = 0;
  virtual void apply([[maybe_unused]] YAML::Node yaml) {}
  void register_root_widget(QWidget * widget);
  void register_root_layout(QLayout * layout);
  void register_stylesheet_widget(QWidget * widget, const std::string & target = "");

public:
  QWidget * call_get_widget();
  void call_setup(YAML::Node yaml, const std::vector<QWidget *> & items);
  void call_apply(const Packet & packet);
  void call_set_stylesheet(const QString & stylesheet);

  // TODO(Takagi, Isamu): Remove debug code.
  // virtual ~BasicWidget() = default;
  static inline int created = 0;
  static inline int removed = 0;
  BasicWidget() { ++created; }
  virtual ~BasicWidget() { ++removed; }

private:
  QWidget * root_ = nullptr;
  std::vector<QWidget *> stylesheet_widgets_;
  Packet::Attrs prev_attrs_;
};

}  // namespace multi_data_monitor

#endif  // MULTI_DATA_MONITOR__WIDGET_HPP_

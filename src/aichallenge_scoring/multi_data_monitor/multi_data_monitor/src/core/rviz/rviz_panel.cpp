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

#include "rviz_panel.hpp"
#include <rclcpp/rclcpp.hpp>
#include <rviz_common/display_context.hpp>
#include <rviz_common/ros_integration/ros_node_abstraction_iface.hpp>
#include <QDockWidget>
#include <QGridLayout>
#include <QLineEdit>
#include <QMouseEvent>
#include <QPushButton>
#include <QTextEdit>
#include <memory>
#include <string>

namespace multi_data_monitor
{

SettingWidget::SettingWidget(MultiDataMonitor * panel) : QWidget(panel)
{
  const auto layout = new QGridLayout();
  config_path_load_ = new QPushButton("OK");
  config_path_edit_ = new QLineEdit();
  config_path_edit_->setPlaceholderText("package://<package>/<path>  or  file://<path>");
  connect(config_path_load_, &QPushButton::clicked, panel, &MultiDataMonitor::updateMultiDataMonitor);
  connect(config_path_edit_, &QLineEdit::editingFinished, panel, &MultiDataMonitor::configChanged);

  layout->addWidget(config_path_edit_, 0, 0);
  layout->addWidget(config_path_load_, 0, 1);
  setLayout(layout);
}

void SettingWidget::save(rviz_common::Config config) const
{
  config.mapSetValue("Path", config_path_edit_->text());
}

void SettingWidget::load(const rviz_common::Config & config)
{
  config_path_edit_->setText(config.mapGetChild("Path").getValue().toString());
}

std::string SettingWidget::getPath() const
{
  return config_path_edit_->text().toStdString();
}

MultiDataMonitor::MultiDataMonitor(QWidget * parent) : rviz_common::Panel(parent)
{
  const auto layout = new QGridLayout();
  setting_ = new SettingWidget(this);
  monitor_ = new QWidget();
  monitor_->setVisible(false);
  layout->addWidget(monitor_);
  layout->addWidget(setting_);
  layout->setSpacing(0);
  layout->setContentsMargins(0, 0, 0, 0);
  setLayout(layout);
}

void MultiDataMonitor::save(rviz_common::Config config) const
{
  Panel::save(config);
  setting_->save(config);
}

void MultiDataMonitor::load(const rviz_common::Config & config)
{
  Panel::load(config);
  setting_->load(config);
  updateMultiDataMonitor();
}

void MultiDataMonitor::onInitialize()
{
  const auto parent = dynamic_cast<QDockWidget *>(this->parent());
  if (parent)
  {
    const auto layout = new QGridLayout();
    const auto widget = new QWidget();
    layout->setContentsMargins(0, 0, 0, 0);
    widget->setContentsMargins(0, 0, 0, 0);
    layout->addWidget(parent->titleBarWidget());
    widget->setLayout(layout);
    parent->setTitleBarWidget(widget);
  }
}

void MultiDataMonitor::mousePressEvent(QMouseEvent * event)
{
  if (event->modifiers() & Qt::ControlModifier)
  {
    setting_->setVisible(!setting_->isVisible());
  }

  if (event->modifiers() & Qt::ShiftModifier)
  {
    const auto parent = dynamic_cast<QDockWidget *>(this->parent());
    if (parent)
    {
      const auto title = parent->titleBarWidget()->layout()->itemAt(0)->widget();
      title->setVisible(!title->isVisible());
    }
  }
}

void MultiDataMonitor::updateMultiDataMonitor()
{
  auto rviz_node = getDisplayContext()->getRosNodeAbstraction().lock()->get_raw_node();
  QWidget * widget = nullptr;
  try
  {
    const auto path = setting_->getPath();
    widget = manager_.build(path, rviz_node);
    setting_->setVisible(false);
  }
  catch (const std::exception & error)
  {
    RCLCPP_ERROR_STREAM(rviz_node->get_logger(), error.what());
    QTextEdit * text = new QTextEdit();
    text->setReadOnly(true);
    text->setText(error.what());
    widget = text;
    setting_->setVisible(true);
  }

  if (widget)
  {
    layout()->replaceWidget(monitor_, widget);
    delete monitor_;
    monitor_ = widget;
    monitor_->setVisible(true);
  }
}

}  // namespace multi_data_monitor

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(multi_data_monitor::MultiDataMonitor, rviz_common::Panel)

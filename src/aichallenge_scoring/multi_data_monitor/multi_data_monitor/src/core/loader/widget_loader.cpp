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

#include "widget_loader.hpp"
#include "common/exceptions.hpp"
#include "config/types.hpp"
#include <QGridLayout>
#include <QWidget>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

// TODO(Takagi, Isamu): exception or warning
#include <iostream>

// TODO(Takagi, Isamu): merge filter loader
namespace
{
std::string get_full_plugin_name(const std::string & klass)
{
  if (klass.find("::") != std::string::npos)
  {
    return klass;
  }
  return multi_data_monitor::plugin::name::package + std::string("::") + klass;
}

}  // namespace

namespace multi_data_monitor
{

QString get_stylesheet(const DesignList & designs, const std::string & klass = "")
{
  std::stringstream stylesheet;
  for (const auto & design : designs)
  {
    if (get_full_plugin_name(design->klass) == get_full_plugin_name(klass))
    {
      stylesheet << design->stylesheet << std::endl;
    }
  }
  return QString::fromStdString(stylesheet.str());
}

WidgetLoader::WidgetLoader() : plugins_(plugin::name::package, plugin::name::widget)
{
}

WidgetLoader::~WidgetLoader()
{
}

WidgetMaps WidgetLoader::create(const WidgetList & configs, const DesignList & designs)
{
  // Place the dummy root object in stack memory to automatically release Qt objects.
  QWidget dummy_root_widget;
  WidgetMaps mapping;

  for (const auto & config : configs)
  {
    const auto node = create_widget(config);
    mapping[config] = widgets_.emplace_back(node);

    std::vector<QWidget *> items;
    for (const auto & item : config->items)
    {
      items.push_back(mapping.at(item)->call_get_widget());
    }
    node->call_setup(config->yaml, items);
    node->call_get_widget()->setParent(&dummy_root_widget);
    node->call_set_stylesheet(get_stylesheet(designs, config->klass));
  }

  root_widget_ = std::make_unique<QWidget>();
  root_widget_->setStyleSheet(get_stylesheet(designs));
  root_widget_->setContentsMargins(0, 0, 0, 0);

  if (!configs.empty())
  {
    QWidget * widget = mapping.at(configs.back())->call_get_widget();
    QLayout * layout = new QGridLayout();
    layout->addWidget(widget);
    // layout->setSpacing(0);
    // layout->setContentsMargins(0, 0, 0, 0);
    root_widget_->setLayout(layout);
  }

  for (const auto & node : widgets_)
  {
    QWidget * widget = node->call_get_widget();
    if (widget && widget->parent() == &dummy_root_widget)
    {
      // TODO(Takagi, Isamu): exception or warning
      std::cerr << "unused widget is detected: " << widget << std::endl;
    }
  }
  return mapping;
}

Widget WidgetLoader::create_widget(const WidgetLink & config)
{
  // Search in default plugins if namespace is omitted.
  std::string klass = get_full_plugin_name(config->klass);
  if (!plugins_.isClassAvailable(klass))
  {
    throw ConfigError("unknown widget type: " + config->klass);
  }
  return plugins_.createSharedInstance(klass);
}

void WidgetLoader::release()
{
  // TODO(Takagi, Isamu): check use count
  // Release shared_ptr to unload plugins.
  widgets_.clear();
  root_widget_.reset();
}

QWidget * WidgetLoader::take_root_widget()
{
  return root_widget_.release();
}

}  // namespace multi_data_monitor

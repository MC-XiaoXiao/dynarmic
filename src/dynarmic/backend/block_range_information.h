/* This file is part of the dynarmic project.
 * Copyright (c) 2018 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <boost/icl/interval_set.hpp>
#include <mcl/container/ihmap.hpp>
#include <tsl/robin_map.h>
#include <tsl/robin_set.h>

#include "dynarmic/ir/location_descriptor.h"

namespace Dynarmic::Backend {

template<typename ProgramCounterType>
class BlockRangeInformation {
public:
    struct Stats {
        std::size_t range_count{};
        std::size_t descriptor_count{};
        std::uint64_t invalidated_descriptors{};
    };

    void AddRange(boost::icl::discrete_interval<ProgramCounterType> range, IR::LocationDescriptor location);
    void ClearCache();
    tsl::robin_set<IR::LocationDescriptor> InvalidateRanges(const boost::icl::interval_set<ProgramCounterType>& ranges);
    std::vector<IR::LocationDescriptor> InvalidateAll();
    void InvalidateLocations(
        const tsl::robin_set<IR::LocationDescriptor>& locations);
    [[nodiscard]] Stats GetStats() const noexcept;

private:
    using Interval = boost::icl::discrete_interval<ProgramCounterType>;
    using IntervalSet = boost::icl::interval_set<ProgramCounterType>;

    // A compiled descriptor normally covers one interval. Allocate the full
    // ICL set only when it is registered again; ICL retains responsibility for
    // joining adjacent/overlapping ranges and handling interval boundaries.
    class DescriptorRanges {
    public:
        explicit DescriptorRanges(Interval range) : first(range) {}

        void Add(Interval range) {
            if (!multiple) {
                multiple = std::make_unique<IntervalSet>();
                multiple->add(first);
            }
            multiple->add(range);
        }

        void Swap(DescriptorRanges& other) noexcept {
            std::swap(first, other.first);
            multiple.swap(other.multiple);
        }

        std::size_t Size() const {
            return multiple ? multiple->iterative_size() : !boost::icl::is_empty(first);
        }

        template<typename Function>
        void ForEach(Function&& function) const {
            if (multiple) {
                for (const auto& range : *multiple) {
                    function(range);
                }
            } else if (!boost::icl::is_empty(first)) {
                function(first);
            }
        }

    private:
        Interval first;
        std::unique_ptr<IntervalSet> multiple;
    };

    // Store one exact interval per descriptor range. Unlike a segmented
    // codomain map, overlapping blocks do not split nodes or copy sets of
    // descriptors. The subtree maximum prunes precise byte-overlap queries.
    class RangeIndex {
        struct Node {
            ProgramCounterType lower, upper, maximum;
            IR::LocationDescriptor location;
            int height = 1;
            std::unique_ptr<Node> left, right;
            Node(ProgramCounterType first, ProgramCounterType last, IR::LocationDescriptor descriptor)
                : lower(first), upper(last), maximum(last), location(descriptor) {}
        };
        using Link = std::unique_ptr<Node>;
        Link root;

        static int Height(const Link& node) { return node ? node->height : 0; }
        static void Update(Node& node) {
            node.height = 1 + std::max(Height(node.left), Height(node.right));
            node.maximum = node.upper;
            if (node.left) node.maximum = std::max(node.maximum, node.left->maximum);
            if (node.right) node.maximum = std::max(node.maximum, node.right->maximum);
        }
        static int Compare(ProgramCounterType first, ProgramCounterType last,
                           IR::LocationDescriptor descriptor, const Node& node) {
            if (first != node.lower) return first < node.lower ? -1 : 1;
            if (last != node.upper) return last < node.upper ? -1 : 1;
            if (descriptor != node.location) return descriptor.Value() < node.location.Value() ? -1 : 1;
            return 0;
        }
        static void RotateLeft(Link& node) {
            auto pivot = std::move(node->right);
            node->right = std::move(pivot->left);
            Update(*node);
            pivot->left = std::move(node);
            Update(*pivot);
            node = std::move(pivot);
        }
        static void RotateRight(Link& node) {
            auto pivot = std::move(node->left);
            node->left = std::move(pivot->right);
            Update(*node);
            pivot->right = std::move(node);
            Update(*pivot);
            node = std::move(pivot);
        }
        static void Balance(Link& node) {
            if (!node) return;
            Update(*node);
            if (Height(node->left) > Height(node->right) + 1) {
                if (Height(node->left->right) > Height(node->left->left)) RotateLeft(node->left);
                RotateRight(node);
            } else if (Height(node->right) > Height(node->left) + 1) {
                if (Height(node->right->left) > Height(node->right->right)) RotateRight(node->right);
                RotateLeft(node);
            }
        }
        static void Insert(Link& node, Link& prepared) {
            if (!node) {
                node = std::move(prepared);
                return;
            }
            const auto order = Compare(prepared->lower, prepared->upper, prepared->location, *node);
            if (order < 0) Insert(node->left, prepared);
            else if (order > 0) Insert(node->right, prepared);
            else return;
            Balance(node);
        }
        static void Erase(Link& node, ProgramCounterType first, ProgramCounterType last,
                          IR::LocationDescriptor descriptor) {
            if (!node) return;
            const auto order = Compare(first, last, descriptor, *node);
            if (order < 0) Erase(node->left, first, last, descriptor);
            else if (order > 0) Erase(node->right, first, last, descriptor);
            else {
                if (!node->left) {
                    auto retired = std::move(node);
                    node = std::move(retired->right);
                    return;
                }
                if (!node->right) {
                    auto retired = std::move(node);
                    node = std::move(retired->left);
                    return;
                }
                const auto* next = node->right.get();
                while (next->left) next = next->left.get();
                node->lower = next->lower;
                node->upper = next->upper;
                node->location = next->location;
                Erase(node->right, node->lower, node->upper, node->location);
            }
            Balance(node);
        }
        static void Overlaps(const Node* node, ProgramCounterType first, ProgramCounterType last,
                             tsl::robin_set<IR::LocationDescriptor>& found) {
            if (!node || node->maximum < first) return;
            Overlaps(node->left.get(), first, last, found);
            if (node->lower <= last && first <= node->upper) found.insert(node->location);
            if (node->lower <= last) Overlaps(node->right.get(), first, last, found);
        }
        // Filtering can remove many levels at once. Join with a pivot rather
        // than treating that arbitrary height change as a single deletion.
        static Link JoinWithPivot(Link left, Link pivot, Link right) {
            if (Height(left) > Height(right) + 1) {
                left->right = JoinWithPivot(std::move(left->right), std::move(pivot), std::move(right));
                Balance(left);
                return left;
            }
            if (Height(right) > Height(left) + 1) {
                right->left = JoinWithPivot(std::move(left), std::move(pivot), std::move(right->left));
                Balance(right);
                return right;
            }
            pivot->left = std::move(left);
            pivot->right = std::move(right);
            Update(*pivot);
            return pivot;
        }
        static Link DetachFirst(Link& node) {
            if (!node->left) {
                auto first = std::move(node);
                node = std::move(first->right);
                return first;
            }
            auto first = DetachFirst(node->left);
            Balance(node);
            return first;
        }
        static Link Join(Link left, Link right) {
            if (!left) return right;
            if (!right) return left;
            auto pivot = DetachFirst(right);
            return JoinWithPivot(std::move(left), std::move(pivot), std::move(right));
        }
        static Link Filter(Link node, const tsl::robin_set<IR::LocationDescriptor>& retired) {
            if (!node) return {};
            auto left = Filter(std::move(node->left), retired);
            auto right = Filter(std::move(node->right), retired);
            if (retired.contains(node->location)) return Join(std::move(left), std::move(right));
            return JoinWithPivot(std::move(left), std::move(node), std::move(right));
        }

    public:
        using PreparedRange = Link;
        static PreparedRange Prepare(Interval range, IR::LocationDescriptor location) {
            return boost::icl::is_empty(range) ? nullptr :
                std::make_unique<Node>(boost::icl::first(range), boost::icl::last(range), location);
        }
        void AddPrepared(PreparedRange prepared) {
            if (prepared) Insert(root, prepared);
        }
        void Add(Interval range, IR::LocationDescriptor location) {
            AddPrepared(Prepare(range, location));
        }
        void Remove(Interval range, IR::LocationDescriptor location) {
            if (!boost::icl::is_empty(range)) Erase(root, boost::icl::first(range), boost::icl::last(range), location);
        }
        void Find(Interval range, tsl::robin_set<IR::LocationDescriptor>& found) const {
            if (!boost::icl::is_empty(range)) Overlaps(root.get(), boost::icl::first(range), boost::icl::last(range), found);
        }
        void RemoveLocations(const tsl::robin_set<IR::LocationDescriptor>& retired) {
            root = Filter(std::move(root), retired);
        }
        void Clear() { root.reset(); }
    };
    RangeIndex block_ranges;
    mcl::ihmap<IR::LocationDescriptor, DescriptorRanges> ranges_by_descriptor;
    std::size_t range_count{};
    std::uint64_t invalidated_descriptors{};
};

}  // namespace Dynarmic::Backend

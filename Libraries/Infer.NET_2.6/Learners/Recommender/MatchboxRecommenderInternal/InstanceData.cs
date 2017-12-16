﻿/********************************************************
*                                                       *
*   Copyright (C) Microsoft. All rights reserved.       *
*                                                       *
********************************************************/

namespace MicrosoftResearch.Infer.Learners.MatchboxRecommenderInternal
{
    using System.Collections.Generic;
    using System.Diagnostics;

    /// <summary>
    /// Represents the instance data - users, items and ratings.
    /// </summary>
    internal class InstanceData
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="InstanceData"/> class.
        /// </summary>
        /// <param name="userIds">The user identifiers.</param>
        /// <param name="itemIds">The item identifiers.</param>
        /// <param name="ratings">The ratings.</param>
        public InstanceData(IList<int> userIds, IList<int> itemIds, IList<int> ratings)
        {
            Debug.Assert(
                userIds != null && itemIds != null && ratings != null,
                "Valid instance arrays must be provided.");
            Debug.Assert(
                userIds.Count == itemIds.Count && userIds.Count == ratings.Count,
                "The instance arrays must be of the same length.");

            this.UserIds = userIds;
            this.ItemIds = itemIds;
            this.Ratings = ratings;
        }

        /// <summary>
        /// Gets the user identifiers.
        /// </summary>
        public IList<int> UserIds { get; private set; }

        /// <summary>
        /// Gets the item identifiers.
        /// </summary>
        public IList<int> ItemIds { get; private set; }

        /// <summary>
        /// Gets the ratings.
        /// </summary>
        public IList<int> Ratings { get; private set; }
    }
}

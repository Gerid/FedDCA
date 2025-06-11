    def _initial_k_estimation(self, client_embeddings, client_label_profiles_np, client_weights_np):
        """Phase 1: Estimate initial K_t using silhouette analysis."""
        best_k = self.vwc_K_t # Fallback to user-defined K_t
        best_silhouette_score = -1
        best_concepts = None
        best_centroids = None

        print(f"Dynamic K: Initial K estimation phase. Range: [{self.dynamic_k_min}, {self.dynamic_k_max}]")

        for k_candidate in range(self.dynamic_k_min, min(self.dynamic_k_max + 1, len(client_embeddings) +1), self.dynamic_k_silhouette_range_step):
            if k_candidate <= 1: continue # Silhouette needs at least 2 clusters
            
            # Run VWC clustering for k_candidate (simplified or full)
            # For this, we need a version of run_vwc_clustering that can be called with a specific K
            # and returns client_concepts and centroids.
            # The following is a simplified GMM approach for demonstration.
            # In a real scenario, you'd call a modified run_vwc_clustering or its core logic.
            
            current_silhouette_score, concepts, centroids = self._calculate_avg_silhouette_for_k(
                client_embeddings, k_candidate, client_label_profiles_np, client_weights_np
            )
            
            print(f"  K={k_candidate}, Silhouette Score: {current_silhouette_score:.4f}")

            if current_silhouette_score > best_silhouette_score:
                best_silhouette_score = current_silhouette_score
                best_k = k_candidate
                best_concepts = concepts
                best_centroids = centroids
        
        if best_concepts is None or best_centroids is None:
            # Fallback if no valid K was found (e.g., all silhouette scores were -1)
            print(f"Dynamic K: Initial estimation failed to find a suitable K. Falling back to K={self.vwc_K_t}.")
            # Perform a default clustering with self.vwc_K_t to get initial concepts/centroids
            _, best_concepts, best_centroids = self._calculate_avg_silhouette_for_k(
                client_embeddings, self.vwc_K_t, client_label_profiles_np, client_weights_np
            )
            if best_concepts is None:
                 print("CRITICAL: Fallback K also failed in initial_k_estimation. Using K=2 as emergency.")
                 best_k = 2
                 _, best_concepts, best_centroids = self._calculate_avg_silhouette_for_k(
                    client_embeddings, best_k, client_label_profiles_np, client_weights_np
                 )
                 if best_concepts is None:
                    # Ultimate fallback: assign all to one cluster if everything else fails
                    print("CRITICAL: Emergency K=2 also failed. Assigning all clients to cluster 0.")
                    best_k = 1
                    best_concepts = np.zeros(len(client_embeddings), dtype=int)
                    best_centroids = np.mean(client_embeddings, axis=0, keepdims=True)
            else:
                best_k = self.vwc_K_t # Ensure best_k reflects the fallback K used

        print(f"Dynamic K: Initial K estimated to {best_k} with silhouette score: {best_silhouette_score:.4f}")
        return best_k, best_concepts, best_centroids

    def _attempt_cluster_merge(self, current_k, client_embeddings, client_concepts, centroids, client_label_profiles_np):
        """Phase 2: Attempt to merge clusters."""
        if current_k <= self.dynamic_k_min:
            return current_k, client_concepts, centroids, False 

        effective_centroid_profiles = []
        for i in range(current_k):
            cluster_client_indices = np.where(client_concepts == i)[0]
            if len(cluster_client_indices) > 0:
                avg_profile = np.mean(client_label_profiles_np[cluster_client_indices], axis=0)
                effective_centroid_profiles.append(avg_profile)
            else:
                effective_centroid_profiles.append(np.full(client_label_profiles_np.shape[1], np.inf))

        if len(effective_centroid_profiles) < 2: 
            return current_k, client_concepts, centroids, False

        min_dist = float('inf')
        merge_candidates = (-1, -1)

        for i in range(len(effective_centroid_profiles)):
            for j in range(i + 1, len(effective_centroid_profiles)):
                if np.all(np.isinf(effective_centroid_profiles[i])) or np.all(np.isinf(effective_centroid_profiles[j])):
                    continue
                dist = self._calculate_inter_centroid_wasserstein_distance(effective_centroid_profiles[i], effective_centroid_profiles[j])
                if dist < min_dist:
                    min_dist = dist
                    merge_candidates = (i, j)
        
        if merge_candidates != (-1,-1) and min_dist < self.dynamic_k_merge_wasserstein_threshold:
            c1_idx, c2_idx = merge_candidates
            print(f"Dynamic K: Merging clusters {c1_idx} and {c2_idx} (Wasserstein dist: {min_dist:.4f})")
            
            client_concepts[client_concepts == c2_idx] = c1_idx
            client_concepts[client_concepts > c2_idx] = client_concepts[client_concepts > c2_idx] - 1
            current_k -= 1
            
            new_centroids = np.zeros((current_k, client_embeddings.shape[1]))
            for i in range(current_k):
                cluster_client_indices = np.where(client_concepts == i)[0]
                if len(cluster_client_indices) > 0:
                    new_centroids[i] = np.mean(client_embeddings[cluster_client_indices], axis=0)

            return current_k, client_concepts, new_centroids, True 
        
        return current_k, client_concepts, centroids, False 

    def _attempt_cluster_split(self, current_k, client_embeddings, client_concepts, centroids, client_label_profiles_np):
        """Phase 2: Attempt to split a cluster."""
        if current_k >= self.dynamic_k_max:
            return current_k, client_concepts, centroids, False 

        max_dispersion = -1
        split_candidate_idx = -1

        for i in range(current_k):
            cluster_client_indices = np.where(client_concepts == i)[0]
            if len(cluster_client_indices) < self.dynamic_k_split_min_clients:
                continue 
            
            embeddings_in_cluster = client_embeddings[cluster_client_indices]
            current_cluster_centroid_embedding = np.mean(embeddings_in_cluster, axis=0)
            dispersion = self._calculate_cluster_dispersion(embeddings_in_cluster, current_cluster_centroid_embedding)
            if dispersion > max_dispersion:
                max_dispersion = dispersion
                split_candidate_idx = i
        
        if split_candidate_idx != -1 and max_dispersion > self.dynamic_k_split_dispersion_threshold:
            print(f"Dynamic K: Attempting to split cluster {split_candidate_idx} (Dispersion: {max_dispersion:.4f})")
            
            indices_to_split = np.where(client_concepts == split_candidate_idx)[0]
            embeddings_to_split = client_embeddings[indices_to_split]
            
            if len(embeddings_to_split) < self.dynamic_k_split_subcluster_k: 
                 print(f"Dynamic K: Not enough samples in cluster {split_candidate_idx} to split into {self.dynamic_k_split_subcluster_k} sub-clusters.")
                 return current_k, client_concepts, centroids, False

            gmm_split = GaussianMixture(n_components=self.dynamic_k_split_subcluster_k, random_state=self.args.seed, covariance_type='diag')
            try:
                sub_labels = gmm_split.fit_predict(embeddings_to_split)
            except ValueError:
                print(f"Dynamic K: GMM fit failed for splitting cluster {split_candidate_idx}.")
                return current_k, client_concepts, centroids, False

            if len(np.unique(sub_labels)) < self.dynamic_k_split_subcluster_k:
                print(f"Dynamic K: Split of cluster {split_candidate_idx} did not result in {self.dynamic_k_split_subcluster_k} distinct sub-clusters.")
                return current_k, client_concepts, centroids, False

            print(f"Dynamic K: Splitting cluster {split_candidate_idx} into {self.dynamic_k_split_subcluster_k} sub-clusters.")
            
            new_cluster_label_start = current_k
            for sub_cluster_id in range(self.dynamic_k_split_subcluster_k):
                sub_indices_original = indices_to_split[sub_labels == sub_cluster_id]
                if sub_cluster_id == 0: # First sub-cluster reuses original label
                    client_concepts[sub_indices_original] = split_candidate_idx
                else:
                    # Check if we exceed max_k before assigning new label
                    if new_cluster_label_start + (sub_cluster_id -1) >= self.dynamic_k_max:
                        print(f"Dynamic K: Splitting cluster {split_candidate_idx} aborted, would exceed dynamic_k_max.")
                        # This requires a rollback or careful handling. For now, we abort the split.
                        # To properly handle, we'd need to revert any partial relabeling or not proceed.
                        # Simplest is to return False here if the *next* label would exceed max_k.
                        return current_k, client_concepts, centroids, False # Abort split
                    client_concepts[sub_indices_original] = new_cluster_label_start + (sub_cluster_id - 1)
            
            num_new_clusters_added = self.dynamic_k_split_subcluster_k - 1
            current_k += num_new_clusters_added
            
            new_centroids = np.zeros((current_k, client_embeddings.shape[1]))
            for i in range(current_k):
                cluster_client_indices = np.where(client_concepts == i)[0]
                if len(cluster_client_indices) > 0:
                    new_centroids[i] = np.mean(client_embeddings[cluster_client_indices], axis=0)

            return current_k, client_concepts, new_centroids, True 
            
        return current_k, client_concepts, centroids, False 

    def _refine_k_iteratively(self, current_k, client_embeddings, client_concepts, centroids, client_label_profiles_np):
        """Phase 2: Iteratively merge and split clusters."""
        print(f"Dynamic K: Refining K iteratively. Start K: {current_k}")
        for i in range(self.dynamic_k_refinement_iterations):
            made_change_in_iteration = False
            # Attempt merge
            if current_k > self.dynamic_k_min:
                new_k, new_concepts, new_centroids, merged = self._attempt_cluster_merge(
                    current_k, client_embeddings, client_concepts, centroids, client_label_profiles_np
                )
                if merged:
                    current_k, client_concepts, centroids = new_k, new_concepts, new_centroids
                    made_change_in_iteration = True
                    print(f"  Refinement iter {i+1}: Merged. New K: {current_k}")
            
            # Attempt split
            if current_k < self.dynamic_k_max:
                new_k, new_concepts, new_centroids, split = self._attempt_cluster_split(
                    current_k, client_embeddings, client_concepts, centroids, client_label_profiles_np
                )
                if split:
                    current_k, client_concepts, centroids = new_k, new_concepts, new_centroids
                    made_change_in_iteration = True
                    print(f"  Refinement iter {i+1}: Split. New K: {current_k}")
            
            if not made_change_in_iteration:
                print(f"Dynamic K: Refinement converged after {i+1} iterations.")
                break
        
        print(f"Dynamic K: Final K after refinement: {current_k}")
        return current_k, client_concepts, centroids


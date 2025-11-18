import torch
from ...enums import *
from logging import getLogger
logger = getLogger(__name__)

class Connections_Tensor:
    """Class for holding the connections between tokens"""
    def __init__(self, connections: torch.Tensor):
        self.connections: torch.Tensor = connections
        assert connections.dtype == torch.bool, "Connections tensor must be a boolean tensor"
    
    def connect(self, parent_idxs: torch.Tensor, child_idxs: torch.Tensor, value=True):
        """
        Connect from parent token to to child token.
        If multiple parents and children are provided, creates all combinations.
        """
        logger.info(f"Connecting {len(parent_idxs)} parents to {len(child_idxs)} children")
        logger.debug(f"Parent indices: {parent_idxs}")
        logger.debug(f"Child indices: {child_idxs}")
        # Create all combinations of parents and children
        parent_expanded = parent_idxs.unsqueeze(1).expand(-1, len(child_idxs))
        child_expanded = child_idxs.unsqueeze(0).expand(len(parent_idxs), -1)
        self.connections[parent_expanded, child_expanded] = value
    
    def connect_bi(self, from_idxs: torch.Tensor, to_idxs: torch.Tensor, value=True):
        """
        Connect from from_idxs to to_idxs and vice versa.
        If multiple indices are provided, creates all combinations.
        """
        logger.info(f"Connecting {len(from_idxs)} indices to {len(to_idxs)} indices")
        logger.debug(f"From indices: {from_idxs}")
        logger.debug(f"To indices: {to_idxs}")
        # Create all combinations
        from_expanded = from_idxs.unsqueeze(1).expand(-1, len(to_idxs))
        to_expanded = to_idxs.unsqueeze(0).expand(len(from_idxs), -1)
        self.connections[from_expanded, to_expanded] = value
        self.connections[to_expanded, from_expanded] = value

    def get_parents(self, child_idxs: torch.Tensor) -> torch.Tensor:
        """
        Get the direct parents of the given child_idxs
        Args:
            child_idxs: torch.Tensor - The indices of the children to get the parents of.
        Returns:
            torch.Tensor - The indices of the parents of the given child_idxs.
        """
        logger.info(f"Getting parents of {len(child_idxs)} children")
        logger.debug(f"Child indices: {child_idxs}")
        # connections[i, j] = True means i -> j (parent i connects to child j)
        # To get parents of child j, we need all i where connections[i, j] = True
        # This is the column j, so we check connections[:, child_idxs]
        mask = self.connections[:, child_idxs] == True
        # Get row indices (parents) where any of the specified children have connections
        parent_indices = torch.where(mask.any(dim=1))[0]
        return parent_indices
    
    def get_parents_recursive(self, child_idxs: torch.Tensor) -> torch.Tensor:
        """
        Get all parents recursively (parents, parents' parents, etc.) from the given child indices.
        Handles circular connections by tracking visited nodes.
        
        Args:
            child_idxs: torch.Tensor - The indices of the children to get the parents of.
        Returns:
            torch.Tensor - The indices of all ancestors (parents, grandparents, etc.) of the given child_idxs.
        """
        logger.info(f"Getting parents recursively of {len(child_idxs)} children")
        logger.debug(f"Child indices: {child_idxs}")
        if len(child_idxs) == 0:
            return torch.tensor([], dtype=torch.long)
        
        # Convert starting indices to set to exclude them from results
        starting_nodes = set(child_idxs.tolist() if isinstance(child_idxs, torch.Tensor) else list(child_idxs))
        
        # Convert to set for efficient lookup and to handle duplicates
        visited = set()
        all_parents = set()
        # Use a queue for breadth-first traversal
        queue = child_idxs.tolist() if isinstance(child_idxs, torch.Tensor) else list(child_idxs)
        
        while queue:
            current = queue.pop(0)
            # Skip if we've already visited this node (handles cycles)
            if current in visited:
                continue
            
            visited.add(current)
            
            # Get direct parents of current node
            current_tensor = torch.tensor([current], dtype=torch.long)
            direct_parents = self.get_parents(current_tensor)
            
            # Add parents to the result set (excluding starting nodes)
            for parent in direct_parents.tolist():
                if parent not in starting_nodes:
                    all_parents.add(parent)
                # Add to queue to explore their parents (if not already visited)
                if parent not in visited:
                    queue.append(parent)
        
        # Convert back to tensor and return
        if len(all_parents) == 0:
            return torch.tensor([], dtype=torch.long)
        return torch.tensor(list(all_parents), dtype=torch.long)
    
    def get_children(self, parent_idxs: torch.Tensor) -> torch.Tensor:
        """
        Get the direct children of the given parent_idxs
        Args:
            parent_idxs: torch.Tensor - The indices of the parents to get the children of.
        Returns:
            torch.Tensor - The indices of the children of the given parent_idxs.
        """
        logger.info(f"Getting children of {len(parent_idxs)} parents")
        logger.debug(f"Parent indices: {parent_idxs}")
        # connections[i, j] = True means i -> j (parent i connects to child j)
        # To get children of parent i, we need all j where connections[i, j] = True
        # This is the row i, so we check connections[parent_idxs, :]
        mask = self.connections[parent_idxs, :] == True
        # Get column indices (children) where any of the specified parents have connections
        child_indices = torch.where(mask.any(dim=0))[0]
        return child_indices
    
    def get_children_recursive(self, parent_idxs: torch.Tensor) -> torch.Tensor:
        """
        Get all children recursively (children, children's children, etc.) from the given parent indices.
        Handles circular connections by tracking visited nodes.
        
        Args:
            parent_idxs: torch.Tensor - The indices of the parents to get the children of.
        Returns:
            torch.Tensor - The indices of all descendants (children, grandchildren, etc.) of the given parent_idxs.
        """
        logger.info(f"Getting children recursively of {len(parent_idxs)} parents")
        logger.debug(f"Parent indices: {parent_idxs}")
        if len(parent_idxs) == 0:
            return torch.tensor([], dtype=torch.long)
        
        # Convert starting indices to set to exclude them from results
        starting_nodes = set(parent_idxs.tolist() if isinstance(parent_idxs, torch.Tensor) else list(parent_idxs))
        
        # Convert to set for efficient lookup and to handle duplicates
        visited = set()
        all_children = set()
        # Use a queue for breadth-first traversal
        queue = parent_idxs.tolist() if isinstance(parent_idxs, torch.Tensor) else list(parent_idxs)
        
        while queue:
            current = queue.pop(0)
            # Skip if we've already visited this node (handles cycles)
            if current in visited:
                continue
            
            visited.add(current)
            
            # Get direct children of current node
            current_tensor = torch.tensor([current], dtype=torch.long)
            direct_children = self.get_children(current_tensor)
            
            # Add children to the result set (excluding starting nodes)
            for child in direct_children.tolist():
                if child not in starting_nodes:
                    all_children.add(child)
                # Add to queue to explore their children (if not already visited)
                if child not in visited:
                    queue.append(child)
        
        # Convert back to tensor and return
        if len(all_children) == 0:
            return torch.tensor([], dtype=torch.long)
        return torch.tensor(list(all_children), dtype=torch.long)

    
    def get_all_connected(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Get both parents and children of the given idx
        Args:
            idx: torch.Tensor - The index of the token to get the connected tokens of.
        Returns:
            torch.Tensor - The indices of the connected tokens of the given idx.
        """
        logger.info(f"Getting all connected of {len(idx)} indices")
        logger.debug(f"Indices: {idx}")
        return torch.unique(torch.cat([self.get_parents(idx), self.get_children(idx)]))
    
    def get_all_connected_recursive(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Get all connected tokens recursively (parents, parents' parents, etc., and children, children's children, etc.) from the given idx.
        Handles circular connections by tracking visited nodes.
        Args:
            idx: torch.Tensor - The index of the token to get the connected tokens of.
        Returns:
            torch.Tensor - The indices of all connected tokens of the given idx.
        """
        logger.info(f"Getting all connected recursively of {len(idx)} indices")
        logger.debug(f"Indices: {idx}")
        return torch.unique(torch.cat([self.get_parents_recursive(idx), self.get_children_recursive(idx)]))
    
    def get_connected_set(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Get all connected tokens from the given idx. This looks at parents, children, and both their parents and children.
        This returns all tokens that have any path of parents/children that lead to the input token(s).
        Essentially finds the connected component in the undirected graph formed by parent-child relationships.
        
        Args:
            idx: torch.Tensor - The index (or indices) of the token(s) to get the connected set of.
        Returns:
            torch.Tensor - The indices of all tokens in the connected component (including the input token(s)).
        """
        logger.info(f"Getting connected set of {len(idx)} indices")
        logger.debug(f"Indices: {idx}")
        if len(idx) == 0:
            return torch.tensor([], dtype=torch.long)
        
        # Convert to set for efficient lookup and to handle duplicates
        visited = set()
        connected_set = set()
        # Use a queue for breadth-first traversal
        queue = idx.tolist() if isinstance(idx, torch.Tensor) else list(idx)
        
        while queue:
            current = queue.pop(0)
            # Skip if we've already visited this node (handles cycles)
            if current in visited:
                continue
            
            visited.add(current)
            connected_set.add(current)
            
            # Get both direct parents and children of current node
            current_tensor = torch.tensor([current], dtype=torch.long)
            direct_parents = self.get_parents(current_tensor)
            direct_children = self.get_children(current_tensor)
            
            # Add parents and children to the result set and queue
            for neighbor in direct_parents.tolist() + direct_children.tolist():
                connected_set.add(neighbor)
                # Add to queue to explore their connections (if not already visited)
                if neighbor not in visited:
                    queue.append(neighbor)
        
        # Convert back to tensor and return
        if len(connected_set) == 0:
            return torch.tensor([], dtype=torch.long)
        return torch.tensor(list(connected_set), dtype=torch.long)
        
    
    def expand_to(self, new_size: int):
        """
        Expand the connections tensor to the given size
        Args:
            new_size: int - The new size of the connections tensor.
        """
        old_size = self.connections.shape[0]
        new_connections = torch.zeros(new_size, new_size, dtype=torch.bool)
        # Only copy if new size is larger than current size
        if new_size >= self.connections.shape[0] and new_size >= self.connections.shape[1]:
            new_connections[:self.connections.shape[0], :self.connections.shape[1]] = self.connections
        elif new_size < self.connections.shape[0] or new_size < self.connections.shape[1]:
            # If shrinking, only copy what fits
            copy_size = min(new_size, self.connections.shape[0], self.connections.shape[1])
            new_connections[:copy_size, :copy_size] = self.connections[:copy_size, :copy_size]
        self.connections = new_connections
        logger.info(f"Expanded connections tensor: {old_size} -> {new_size}")
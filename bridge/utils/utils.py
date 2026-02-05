from typing import Dict, Any

from torch._refs import T

from nodes.enums import B
from nodes.utils import TablePrinter, OutputType
from ..old_net import TestDataGenerator
from ..new_net import NetworkLoader
from logging import getLogger

logger = getLogger("UTIL")

def load_network_from_json(file_path):
    """
    Convenience function to load a Network from a JSON state file.
    
    Args:
        file_path: Path to the JSON state file
        
    Returns:
        Network: The reconstructed Network object
        
    Example:
        >>> network = load_network_from_json('test_data/state.json')
    """
    return NetworkLoader().load(file_path)

def compare_connections(old_state: Dict, new_state: Dict, verbose: bool = False) -> Dict[str, Any]:
    """
    Compare two states, to see if they have the same connections between tokens.
    Compares the structural hierarchy: P→RB, RB→PO (pred/obj), and RB→childP.
    
    Args:
        old_state: First network state dict
        new_state: Second network state dict
        verbose: Whether to print detailed comparison results
        
    Returns:
        Dict containing:
            - match: bool - Whether connections match
            - differences: list of difference descriptions
            - p_to_rb_diffs: dict with missing_in_old and missing_in_new
            - rb_to_po_diffs: dict with missing_in_old and missing_in_new
            - rb_to_childp_diffs: dict with missing_in_old and missing_in_new
    """
    results = {
        'match': True,
        'differences': [],
        'p_to_rb_diffs': {'missing_in_old': [], 'missing_in_new': []},
        'rb_to_po_diffs': {'missing_in_old': [], 'missing_in_new': []},
        'rb_to_childp_diffs': {'missing_in_old': [], 'missing_in_new': []},
    }
    
    old_conns = old_state.get('connections', {})
    new_conns = new_state.get('connections', {})
    
    # Compare P_to_RB connections
    old_p_to_rb = set((c['parent'], c['child']) for c in old_conns.get('P_to_RB', []))
    new_p_to_rb = set((c['parent'], c['child']) for c in new_conns.get('P_to_RB', []))
    
    missing_in_new = old_p_to_rb - new_p_to_rb
    missing_in_old = new_p_to_rb - old_p_to_rb
    
    if missing_in_new:
        results['match'] = False
        results['p_to_rb_diffs']['missing_in_new'] = list(missing_in_new)
        diff = f"P→RB connections missing in new: {len(missing_in_new)}"
        results['differences'].append(diff)
        if verbose:
            print(f"  ❌ {diff}")
            for conn in list(missing_in_new)[:5]:
                print(f"      {conn[0]} → {conn[1]}")
    
    if missing_in_old:
        results['match'] = False
        results['p_to_rb_diffs']['missing_in_old'] = list(missing_in_old)
        diff = f"P→RB connections missing in old: {len(missing_in_old)}"
        results['differences'].append(diff)
        if verbose:
            print(f"  ❌ {diff}")
            for conn in list(missing_in_old)[:5]:
                print(f"      {conn[0]} → {conn[1]}")
    
    # Compare RB_to_PO connections (include role)
    old_rb_to_po = set((c['parent'], c['child'], c.get('role', '')) for c in old_conns.get('RB_to_PO', []))
    new_rb_to_po = set((c['parent'], c['child'], c.get('role', '')) for c in new_conns.get('RB_to_PO', []))
    
    missing_in_new = old_rb_to_po - new_rb_to_po
    missing_in_old = new_rb_to_po - old_rb_to_po
    
    if missing_in_new:
        results['match'] = False
        results['rb_to_po_diffs']['missing_in_new'] = list(missing_in_new)
        diff = f"RB→PO connections missing in new: {len(missing_in_new)}"
        results['differences'].append(diff)
        if verbose:
            print(f"  ❌ {diff}")
            for conn in list(missing_in_new)[:5]:
                print(f"      {conn[0]} → {conn[1]} ({conn[2]})")
    
    if missing_in_old:
        results['match'] = False
        results['rb_to_po_diffs']['missing_in_old'] = list(missing_in_old)
        diff = f"RB→PO connections missing in old: {len(missing_in_old)}"
        results['differences'].append(diff)
        if verbose:
            print(f"  ❌ {diff}")
            for conn in list(missing_in_old)[:5]:
                print(f"      {conn[0]} → {conn[1]} ({conn[2]})")
    
    # Compare RB_to_childP connections
    old_rb_to_childp = set((c['parent'], c['child']) for c in old_conns.get('RB_to_childP', []))
    new_rb_to_childp = set((c['parent'], c['child']) for c in new_conns.get('RB_to_childP', []))
    
    missing_in_new = old_rb_to_childp - new_rb_to_childp
    missing_in_old = new_rb_to_childp - old_rb_to_childp
    
    if missing_in_new:
        results['match'] = False
        results['rb_to_childp_diffs']['missing_in_new'] = list(missing_in_new)
        diff = f"RB→childP connections missing in new: {len(missing_in_new)}"
        results['differences'].append(diff)
        if verbose:
            print(f"  ❌ {diff}")
            for conn in list(missing_in_new)[:5]:
                print(f"      {conn[0]} → {conn[1]}")
    
    if missing_in_old:
        results['match'] = False
        results['rb_to_childp_diffs']['missing_in_old'] = list(missing_in_old)
        diff = f"RB→childP connections missing in old: {len(missing_in_old)}"
        results['differences'].append(diff)
        if verbose:
            print(f"  ❌ {diff}")
            for conn in list(missing_in_old)[:5]:
                print(f"      {conn[0]} → {conn[1]}")
    
    if results['match'] and verbose:
        print("  ✅ Token connections match!")
    
    return results


def compare_links_connections(old_state: Dict, new_state: Dict, verbose: bool = False) -> Dict[str, Any]:
    """
    Compare two states, to see if they have the same links between tokens and semantics.
    This is a binary check (i.e links>0.0 is true, 0.0 is false).
    This checks only the structure of the links, not the values.
    
    Args:
        old_state: First network state dict
        new_state: Second network state dict
        verbose: Whether to print detailed comparison results
        
    Returns:
        Dict containing:
            - match: bool - Whether link structure matches
            - differences: list of difference descriptions
            - missing_in_old: list of (po_name, sem_name) tuples present in new but not old
            - missing_in_new: list of (po_name, sem_name) tuples present in old but not new
    """
    results = {
        'match': True,
        'differences': [],
        'missing_in_old': [],
        'missing_in_new': [],
    }
    
    old_links = old_state.get('links', {}).get('links_list', [])
    new_links = new_state.get('links', {}).get('links_list', [])
    
    # Convert to sets of (po_name, sem_name) where weight > 0
    old_link_set = set((l['po_name'], l['sem_name']) for l in old_links if l.get('weight', 0) > 0)
    new_link_set = set((l['po_name'], l['sem_name']) for l in new_links if l.get('weight', 0) > 0)
    
    missing_in_new = old_link_set - new_link_set
    missing_in_old = new_link_set - old_link_set
    
    if missing_in_new:
        results['match'] = False
        results['missing_in_new'] = list(missing_in_new)
        diff = f"Links missing in new: {len(missing_in_new)}"
        results['differences'].append(diff)
        if verbose:
            print(f"  ❌ {diff}")
            for link in list(missing_in_new)[:5]:
                print(f"      {link[0]} ↔ {link[1]}")
            if len(missing_in_new) > 5:
                print(f"      ... and {len(missing_in_new) - 5} more")
    
    if missing_in_old:
        results['match'] = False
        results['missing_in_old'] = list(missing_in_old)
        diff = f"Links missing in old: {len(missing_in_old)}"
        results['differences'].append(diff)
        if verbose:
            print(f"  ❌ {diff}")
            for link in list(missing_in_old)[:5]:
                print(f"      {link[0]} ↔ {link[1]}")
            if len(missing_in_old) > 5:
                print(f"      ... and {len(missing_in_old) - 5} more")
    
    if results['match'] and verbose:
        print(f"  ✅ Link connections match! ({len(old_link_set)} links)")
    
    return results


def compare_links_weights(old_state: Dict, new_state: Dict, tolerance: float = 1e-6, verbose: bool = False) -> Dict[str, Any]:
    """
    Compare two states, to see if they have the same weights between tokens and semantics.
    This checks the values of the links.
    
    Args:
        old_state: First network state dict
        new_state: Second network state dict
        tolerance: Maximum allowed difference between weights to be considered equal
        verbose: Whether to print detailed comparison results
        
    Returns:
        Dict containing:
            - match: bool - Whether link weights match (within tolerance)
            - differences: list of difference descriptions
            - weight_mismatches: list of dicts with po_name, sem_name, old_weight, new_weight
            - missing_in_old: list of (po_name, sem_name) present in new but not old
            - missing_in_new: list of (po_name, sem_name) present in old but not new
    """
    results = {
        'match': True,
        'differences': [],
        'weight_mismatches': [],
        'missing_in_old': [],
        'missing_in_new': [],
    }
    
    old_links = old_state.get('links', {}).get('links_list', [])
    new_links = new_state.get('links', {}).get('links_list', [])
    
    # Build dicts keyed by (po_name, sem_name)
    old_links_dict = {(l['po_name'], l['sem_name']): l.get('weight', 0) for l in old_links}
    new_links_dict = {(l['po_name'], l['sem_name']): l.get('weight', 0) for l in new_links}
    
    all_keys = set(old_links_dict.keys()) | set(new_links_dict.keys())
    
    for key in all_keys:
        old_weight = old_links_dict.get(key, 0.0)
        new_weight = new_links_dict.get(key, 0.0)
        
        if key not in old_links_dict:
            results['match'] = False
            results['missing_in_old'].append(key)
        elif key not in new_links_dict:
            results['match'] = False
            results['missing_in_new'].append(key)
        elif abs(old_weight - new_weight) > tolerance:
            results['match'] = False
            results['weight_mismatches'].append({
                'po_name': key[0],
                'sem_name': key[1],
                'old_weight': old_weight,
                'new_weight': new_weight,
                'diff': abs(old_weight - new_weight),
            })
    
    if results['missing_in_new']:
        diff = f"Links missing in new: {len(results['missing_in_new'])}"
        results['differences'].append(diff)
        if verbose:
            print(f"  ❌ {diff}")
    
    if results['missing_in_old']:
        diff = f"Links missing in old: {len(results['missing_in_old'])}"
        results['differences'].append(diff)
        if verbose:
            print(f"  ❌ {diff}")
    
    if results['weight_mismatches']:
        diff = f"Link weight mismatches: {len(results['weight_mismatches'])}"
        results['differences'].append(diff)
        if verbose:
            print(f"  ❌ {diff}")
            for mismatch in results['weight_mismatches'][:5]:
                print(f"      {mismatch['po_name']} ↔ {mismatch['sem_name']}: "
                      f"old={mismatch['old_weight']:.6f}, new={mismatch['new_weight']:.6f} "
                      f"(Δ={mismatch['diff']:.6f})")
            if len(results['weight_mismatches']) > 5:
                print(f"      ... and {len(results['weight_mismatches']) - 5} more")
    
    if results['match'] and verbose:
        print(f"  ✅ Link weights match! ({len(old_links_dict)} links, tolerance={tolerance})")
    
    return results


def compare_mappings_connections(old_state: Dict, new_state: Dict, verbose: bool = False) -> Dict[str, Any]:
    """
    Compare two states, to see if they have the same connections between driver and recipient tokens.
    This checks the structure of the mappings (i.e mapping weights>0.0 is true, 0.0 is false).
    
    Args:
        old_state: First network state dict
        new_state: Second network state dict
        verbose: Whether to print detailed comparison results
        
    Returns:
        Dict containing:
            - match: bool - Whether mapping structure matches
            - differences: list of difference descriptions
            - missing_in_old: list of (type, driver_name, recipient_name) present in new but not old
            - missing_in_new: list of (type, driver_name, recipient_name) present in old but not new
            - by_type: dict with P, RB, PO keys, each containing missing_in_old and missing_in_new
    """
    results = {
        'match': True,
        'differences': [],
        'missing_in_old': [],
        'missing_in_new': [],
        'by_type': {
            'P': {'missing_in_old': [], 'missing_in_new': []},
            'RB': {'missing_in_old': [], 'missing_in_new': []},
            'PO': {'missing_in_old': [], 'missing_in_new': []},
        },
    }
    
    old_mappings = old_state.get('mappings', {})
    new_mappings = new_state.get('mappings', {})
    
    # Compare by type
    for token_type in ['P', 'RB', 'PO']:
        type_key = f'{token_type}_mappings'
        old_type_mappings = old_mappings.get(type_key, [])
        new_type_mappings = new_mappings.get(type_key, [])
        
        # Convert to sets of (driver_name, recipient_name) where weight > 0
        old_set = set((m['driver_name'], m['recipient_name']) 
                      for m in old_type_mappings if m.get('weight', 0) > 0)
        new_set = set((m['driver_name'], m['recipient_name']) 
                      for m in new_type_mappings if m.get('weight', 0) > 0)
        
        missing_in_new = old_set - new_set
        missing_in_old = new_set - old_set
        
        if missing_in_new:
            results['match'] = False
            for mapping in missing_in_new:
                results['missing_in_new'].append((token_type, mapping[0], mapping[1]))
                results['by_type'][token_type]['missing_in_new'].append(mapping)
        
        if missing_in_old:
            results['match'] = False
            for mapping in missing_in_old:
                results['missing_in_old'].append((token_type, mapping[0], mapping[1]))
                results['by_type'][token_type]['missing_in_old'].append(mapping)
    
    if results['missing_in_new']:
        diff = f"Mappings missing in new: {len(results['missing_in_new'])}"
        results['differences'].append(diff)
        if verbose:
            print(f"  ❌ {diff}")
            for mapping in results['missing_in_new'][:5]:
                print(f"      [{mapping[0]}] {mapping[1]} ↔ {mapping[2]}")
            if len(results['missing_in_new']) > 5:
                print(f"      ... and {len(results['missing_in_new']) - 5} more")
    
    if results['missing_in_old']:
        diff = f"Mappings missing in old: {len(results['missing_in_old'])}"
        results['differences'].append(diff)
        if verbose:
            print(f"  ❌ {diff}")
            for mapping in results['missing_in_old'][:5]:
                print(f"      [{mapping[0]}] {mapping[1]} ↔ {mapping[2]}")
            if len(results['missing_in_old']) > 5:
                print(f"      ... and {len(results['missing_in_old']) - 5} more")
    
    if results['match'] and verbose:
        total = sum(len(old_mappings.get(f'{t}_mappings', [])) for t in ['P', 'RB', 'PO'])
        print(f"  ✅ Mapping connections match! ({total} mappings)")
    
    return results


def compare_mappings_weights(old_state: Dict, new_state: Dict, tolerance: float = 1e-6, verbose: bool = False) -> Dict[str, Any]:
    """
    Compare two states, to see if they have the same weights between driver and recipient tokens.
    This checks the values of the mappings.
    
    Args:
        old_state: First network state dict
        new_state: Second network state dict
        tolerance: Maximum allowed difference between weights to be considered equal
        verbose: Whether to print detailed comparison results
        
    Returns:
        Dict containing:
            - match: bool - Whether mapping weights match (within tolerance)
            - differences: list of difference descriptions
            - weight_mismatches: list of dicts with type, driver_name, recipient_name, old_weight, new_weight
            - missing_in_old: list of (type, driver_name, recipient_name) present in new but not old
            - missing_in_new: list of (type, driver_name, recipient_name) present in old but not new
            - by_type: dict with P, RB, PO keys, each containing mismatches and missing entries
    """
    results = {
        'match': True,
        'differences': [],
        'weight_mismatches': [],
        'missing_in_old': [],
        'missing_in_new': [],
        'by_type': {
            'P': {'weight_mismatches': [], 'missing_in_old': [], 'missing_in_new': []},
            'RB': {'weight_mismatches': [], 'missing_in_old': [], 'missing_in_new': []},
            'PO': {'weight_mismatches': [], 'missing_in_old': [], 'missing_in_new': []},
        },
    }
    
    old_mappings = old_state.get('mappings', {})
    new_mappings = new_state.get('mappings', {})
    
    # Compare by type
    for token_type in ['P', 'RB', 'PO']:
        type_key = f'{token_type}_mappings'
        old_type_mappings = old_mappings.get(type_key, [])
        new_type_mappings = new_mappings.get(type_key, [])
        
        # Build dicts keyed by (driver_name, recipient_name)
        old_dict = {(m['driver_name'], m['recipient_name']): m.get('weight', 0) 
                    for m in old_type_mappings}
        new_dict = {(m['driver_name'], m['recipient_name']): m.get('weight', 0) 
                    for m in new_type_mappings}
        
        all_keys = set(old_dict.keys()) | set(new_dict.keys())
        
        for key in all_keys:
            old_weight = old_dict.get(key, 0.0)
            new_weight = new_dict.get(key, 0.0)
            
            if key not in old_dict:
                results['match'] = False
                entry = (token_type, key[0], key[1])
                results['missing_in_old'].append(entry)
                results['by_type'][token_type]['missing_in_old'].append(key)
            elif key not in new_dict:
                results['match'] = False
                entry = (token_type, key[0], key[1])
                results['missing_in_new'].append(entry)
                results['by_type'][token_type]['missing_in_new'].append(key)
            elif abs(old_weight - new_weight) > tolerance:
                results['match'] = False
                mismatch = {
                    'type': token_type,
                    'driver_name': key[0],
                    'recipient_name': key[1],
                    'old_weight': old_weight,
                    'new_weight': new_weight,
                    'diff': abs(old_weight - new_weight),
                }
                results['weight_mismatches'].append(mismatch)
                results['by_type'][token_type]['weight_mismatches'].append(mismatch)
    
    if results['missing_in_new']:
        diff = f"Mappings missing in new: {len(results['missing_in_new'])}"
        results['differences'].append(diff)
        if verbose:
            print(f"  ❌ {diff}")
    
    if results['missing_in_old']:
        diff = f"Mappings missing in old: {len(results['missing_in_old'])}"
        results['differences'].append(diff)
        if verbose:
            print(f"  ❌ {diff}")
    
    if results['weight_mismatches']:
        diff = f"Mapping weight mismatches: {len(results['weight_mismatches'])}"
        results['differences'].append(diff)
        if verbose:
            print(f"  ❌ {diff}")
            for mismatch in results['weight_mismatches'][:5]:
                print(f"      [{mismatch['type']}] {mismatch['driver_name']} ↔ {mismatch['recipient_name']}: "
                      f"old={mismatch['old_weight']:.6f}, new={mismatch['new_weight']:.6f} "
                      f"(Δ={mismatch['diff']:.6f})")
            if len(results['weight_mismatches']) > 5:
                print(f"      ... and {len(results['weight_mismatches']) - 5} more")
    
    if results['match'] and verbose:
        total = sum(len(old_mappings.get(f'{t}_mappings', [])) for t in ['P', 'RB', 'PO'])
        print(f"  ✅ Mapping weights match! ({total} mappings, tolerance={tolerance})")
    
    return results


def compare_states(old_state: Dict, new_state: Dict, verbose: bool = False) -> Dict[str, Any]:
    """
    Compare two network states and identify differences.
    
    Args:
        old_state: State from TestDataGenerator (old implementation)
        new_state: State from NewNetworkStateGenerator (new implementation)
        verbose: Whether to print detailed comparison results
        
    Returns:
        Dict containing comparison results with:
            - match: bool - Whether states match
            - differences: list of difference descriptions
            - token_diffs: dict of token differences
            - semantic_diffs: dict of semantic differences
            - link_diffs: dict of link differences
            - mapping_diffs: dict of mapping differences
    """
    results = {
        'match': True,
        'differences': [],
        'token_diffs': {},
        'semantic_diffs': {},
        'link_diffs': {},
        'mapping_diffs': {},
    }

    # Compare params
    old_params = old_state['metadata']['parameters']
    new_params = new_state['metadata']['parameters']
    for key in old_params.keys():
        if old_params[key] != new_params[key]:
            results['match'] = False
            results['differences'].append(f"Param {key} mismatch: old={old_params[key]}, new={new_params[key]}")
            if verbose:
                print(f"  ❌ {diff}")
    
    # Compare token counts
    old_counts = old_state['metadata']['token_counts']
    new_counts = new_state['metadata']['token_counts']
    
    for key in ['Ps', 'RBs', 'POs', 'semantics']:
        if old_counts.get(key) != new_counts.get(key):
            now_not_mach = False
            if key == 'semantics':
                if old_counts.get(key) > new_counts.get(key):
                    # extra semantics are probably the comparative semantics, so we don't count them as a mismatch
                    # get the names of mismatching semantics
                    old_names = set()
                    for sem in old_state['semantics']:
                        old_names.add(sem['name'])
                    new_names = set()
                    for sem in new_state['semantics']:
                        new_names.add(sem['name'])
                    mismatching_names = old_names - new_names
                    for name in ['MORE', 'LESS', 'SAME', 'DIFF']:
                        try:
                            old_names.remove(name)
                        except KeyError:
                            pass
                    if len(mismatching_names>0):
                        results['match'] = False
                        now_not_mach = True
            else:
                now_not_mach = True
                results['match'] = False
            if now_not_mach:
                diff = f"Token count mismatch for {key}: old={old_counts.get(key)}, new={new_counts.get(key)}"
                results['differences'].append(diff)
                if verbose:
                    print(f"  ❌ {diff}")
    
    # Compare tokens by name
    for token_type in ['Ps', 'RBs', 'POs']:
        old_tokens = {t['name']: t for t in old_state['tokens'][token_type]}
        new_tokens = {t['name']: t for t in new_state['tokens'][token_type]}
        
        # Check for missing tokens
        old_names = set(old_tokens.keys())
        new_names = set(new_tokens.keys())
        
        missing_in_new = old_names - new_names
        missing_in_old = new_names - old_names
        
        if missing_in_new:
            results['match'] = False
            diff = f"{token_type} missing in new: {missing_in_new}"
            results['differences'].append(diff)
            results['token_diffs'][f'{token_type}_missing_in_new'] = list(missing_in_new)
            if verbose:
                print(f"  ❌ {diff}")
        
        if missing_in_old:
            results['match'] = False
            diff = f"{token_type} missing in old: {missing_in_old}"
            results['differences'].append(diff)
            results['token_diffs'][f'{token_type}_missing_in_old'] = list(missing_in_old)
            if verbose:
                print(f"  ❌ {diff}")
        
        # Compare matching tokens
        for name in old_names & new_names:
            old_t = old_tokens[name]
            new_t = new_tokens[name]
            token_fields = list(old_t.keys())
            token_fields.remove('name')
            token_fields.remove('set')

            old_set = old_t.get('set')
            new_set = new_t.get('set')

            if old_set != new_set:
                results['match'] = False
                diff = f"{token_type} {name}.set: old={old_set}, new={new_set}"
                results['differences'].append(diff)
                if verbose:
                    print(f"  ❌ {diff}")
            
            for field in token_fields:
                if field == 'index':
                    continue
                old_val = old_t.get(field)
                new_val = new_t.get(field)
                if type(old_val) != type(new_val):
                    logger.critical(f"Type mismatch for {field}: {type(old_val)} != {type(new_val)}")
                    raise ValueError(f"Type mismatch for {field}: {type(old_val)} != {type(new_val)}")
                matching = True
                match old_val:
                    case list():
                        pass
                    case dict():
                        pass
                    case str():
                        matching = (old_val == new_val)
                    case None:
                        pass
                    case bool():
                        matching = (old_val == new_val)
                    case int():
                        matching = (old_val == new_val)
                    case float():
                        try:
                            matching = (abs(float(old_val) - float(new_val)) <= 0.001)
                        except:
                            logger.error(f"Error converting ({field}) to float: {old_t.get(field)} or {new_t.get(field)}")
                            matching = (old_t.get(field) == new_t.get(field))
                if not matching:
                    logger.error(f"[COMP]: ({token_type}) {name}.{field}: old={old_val}, new={new_val}")
                    results['match'] = False
                    diff = [token_type, [old_t.get('index'), new_t.get('index')], name, field, old_val, new_val]
                    results['differences'].append(diff)
    
    # Compare semantics by name
    old_semantics = {s['name']: s for s in old_state['semantics']}
    new_semantics = {s['name']: s for s in new_state['semantics']}
    old_names = set(old_semantics.keys())
    new_names = set(new_semantics.keys())
    missing_in_new = old_names - new_names
    missing_in_old = new_names - old_names - set(['MORE', 'LESS', 'SAME', 'DIFF'])

    if missing_in_new:
        results['match'] = False
        diff = f"Semantics missing in new: {missing_in_new}"
        results['differences'].append(diff)
        results['semantic_diffs']['missing_in_new'] = list(missing_in_new)
        if verbose:
            print(f"  ❌ {diff}")
    if missing_in_old:
        results['match'] = False
        diff = f"Semantics missing in old: {missing_in_old}"
        results['differences'].append(diff)
        results['semantic_diffs']['missing_in_old'] = list(missing_in_old)
        if verbose:
            print(f"  ❌ {diff}")
    for name in old_names & new_names:
        old_s = old_semantics[name]
        new_s = new_semantics[name]
        old_fields = list(old_s.keys())
        new_fields = list(new_s.keys())
        if not (old_fields == new_fields):
            logger.critical(f"Semantic fields mismatch for {name}: {old_fields} != {new_fields}")
        for field in old_fields:
            old_val = old_s.get(field)
            new_val = new_s.get(field)
            if old_val in [None, 'nil'] and new_val in [None, 'nil']:
                continue
            if old_val != new_val:
                if isinstance(old_val, float) or isinstance(new_val, float):
                    if abs(float(old_val) - float(new_val)) <= 1e-6:
                        continue
                results['match'] = False
                diff = ["SEM", [old_s.get('index'), new_s.get('index')], name,field, old_val, new_val]
                results['differences'].append(diff)
                if verbose:
                    print(f"  ❌ {diff}")



    # Compare links
    old_links = set((l['po_name'], l['sem_name']) for l in old_state['links']['links_list'])
    new_links = set((l['po_name'], l['sem_name']) for l in new_state['links']['links_list'])
    
    missing_links_in_new = old_links - new_links
    missing_links_in_old = new_links - old_links
    
    if missing_links_in_new:
        results['match'] = False
        diff = f"Links missing in new: {len(missing_links_in_new)}"
        results['differences'].append(diff)
        results['link_diffs']['missing_in_new'] = list(missing_links_in_new)
        if verbose:
            print(f"  ❌ {diff}")
    
    if missing_links_in_old:
        results['match'] = False
        diff = f"Links missing in old: {len(missing_links_in_old)}"
        results['differences'].append(diff)
        results['link_diffs']['missing_in_old'] = list(missing_links_in_old)
        if verbose:
            print(f"  ❌ {diff}")
    
    if results['match'] and verbose:
        print("  ✅ States match!")
    return results

def print_diffs(diffs: list, header=False):
    """ Print a table of differences."""
    try:
        tp = TablePrinter(headers=["DIFF"],columns=['Type','Idxs', 'Name', 'Field', 'Old', 'New'], rows=diffs, output_type=OutputType.PRINT_CONSOLE)
        tp.print_table(header=header)
    except Exception as e:
        for diff in diffs:
            print(diff)


def load_net_from_sim(sim_path: str):
    """
    Load a new network object from a sim file, by first loading the old network, then saving the state, 
    and loading the new network from the saved state.

    Currently the builder has some issues with handling analogs, and when to create new tokens/merge them. 
    This is just easier for now, and ensures that the network structure is the same for testing.

    Args:
        sim_path: Path to the sim file.
    
    Returns:
        Network: The new network object.
    """
    # load the old network:
    gen = TestDataGenerator()
    gen.load_sim(sim_path)
    old_state = gen.get_state()
    new_network = NetworkLoader().load_from_state(old_state)
    return new_network

def get_indices(state: dict):
    indices = []
    for type in ['Ps', 'RBs', 'POs']:
        type_tokens = state['tokens'][type]
        for token in type_tokens:
            indices.append(token['index'])
    return sorted(indices)


def map_dict(state, indent=0):
    """"""
    if isinstance(state, dict):
        for key, value in state.items():
            print(f"{' ' * indent}{key}")
            map_dict(value, indent+2)
    elif isinstance(state, list) and len(state) > 0 and isinstance(state[0], dict) and False:
        print(f"{' ' * indent}List:")
        for item in state:
            map_dict(item, indent+2)
//
// QLState.java
//
// Data structure representing a possible game state
// Stores lists of friendly and enemy units
//

import edu.cwru.sepia.environment.model.state.State.StateView;
import edu.cwru.sepia.environment.model.state.Unit.UnitView;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

public class QLState {

    private List<QLUnit>            friendly;   // list of friendly footmen
    private List<QLUnit>            enemy;      // list of enemy footmen
    private Map<Integer, Integer>   targets;    // maps id of footmen to their targets

    public QLState(StateView state) {
        friendly = new ArrayList<QLUnit>();
        enemy = new ArrayList<QLUnit>();
        targets = new HashMap<Integer, Integer>();

        for(UnitView unit: state.getUnits(0)) friendly.add(new QLUnit(unit));
        for(UnitView unit: state.getUnits(1)) enemy.add(new QLUnit(unit));
    }

    public List<QLUnit> getFriendly() { return friendly; }

    public List<QLUnit> getEnemy() { return enemy; }

    public Map<Integer, Integer> getTargets() { return targets; }

    // returns the friendly footman with id = fid, or null if not found
    public QLUnit getF(int fid) {
        for(QLUnit f: friendly)
            if(f.getID() == fid) return f;
        return null;
    }

    // returns the enemy footman with id = eid, or null if not found
    public QLUnit getE(int eid) {
        for(QLUnit e: enemy)
            if(e.getID() == eid) return e;
        return null;
    }

    // sets a random enemy as the target of friendly footman with id = fid
    public void setRandomTargetFor(int fid) {
        int rand = QLearningAgent.getRng().nextInt(enemy.size());
        int eid = enemy.get(rand).getID();
        targets.put(fid, eid);
    }

    // sets enemy with id = fid as target for footman with id = fid
    public void setTarget(int fid, int eid) {
        targets.put(fid, eid);
    }

    // returns the id of the enemy that friendly footman with id = fid is targeting
    public Integer getTargetFor(int fid) {
        if(!targets.containsKey(fid)) return null;
        return targets.get(fid);
    }

    // returns the number of friendly footmen targeting enemy with id = eid
    public int getNumAttackersFor(int eid) {
        int attackers = 0;
        for(Entry<Integer, Integer> e: targets.entrySet())
            if(e.getValue() == eid) attackers++;
        return attackers;
    }

    // returns the number of friendly units adjacent to the footman
    public int getFriendlyNeighborsFor(int id) {
        int adjacent = 0;
        QLUnit u = getF(id);                            // see if unit is friendly
        if(u == null) u = getE(id);                     // see if unit is enemy
        if(u == null) {                                 // error out if unit does not exist
            System.err.printf("No unit #%d found for state\n", id);
            System.exit(1);
        }

        for(QLUnit f: friendly) {
            if(f.getID() == id) continue;               // don't count self
            if(f.getLoc().distanceSq(u.getLoc()) <= 2d)
                adjacent++;                             // adjacent if distance squared <= 2
        }

        return adjacent;
    }

    // returns the number of enemy units adjacent to the footman
    public int getEnemyNeighborsFor(int id) {
        int adjacent = 0;
        QLUnit u = getF(id);                            // see if unit is friendly
        if(u == null) u = getE(id);                     // see if unit is enemy
        if(u == null) {                                 // error out if unit does not exist
            System.err.printf("No unit #%d found for state\n", id);
            System.exit(1);
        }

        for(QLUnit e: enemy) {
            if(e.getID() == id) continue;               // don't count self
            if(e.getLoc().distanceSq(u.getLoc()) <= 2d)
                adjacent++;                             // adjacent if distance squared <= 2
        }

        return adjacent;
    }

    // copy over target mapping from another state for units in common
    public void copyTargetsFrom(QLState s) {
        for(Entry<Integer, Integer> e: s.targets.entrySet())
            if(getF(e.getKey()) != null &&
               getE(e.getValue()) != null)
                targets.put(e.getKey(), e.getValue());
    }

    // print the lists of friendly and enemy footmen in two columns
    public void print() {
        String str = String.format("%-15s%7s%-15s\n", "FRIENDLY", "", "ENEMY");
        int i = 0;
        while(i < friendly.size() || i < enemy.size()) {
            // friendly footman in first column
            QLUnit f = friendly.size() > i ? friendly.get(i) : null;
            if(f != null) {
                str += f;
                // print the friendly footman's target
                Integer t = getTargetFor(f.getID());
                str += t == null ? String.format("%7s", "") : String.format(" -> %-2d ", t);
            } else str += String.format("%-22s", "");

            // enemy footman in second column
            QLUnit e = enemy.size() > i ? enemy.get(i) : null;
            if(e != null) str += e;

            str += "\n";
            i++;
        }
        System.out.print(str);
    }
}

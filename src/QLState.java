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
        System.out.printf("Set random target: F%d -> E%d\n", fid, eid);
    }

    // sets enemy with id = fid as target for footman with id = fid
    public void setTarget(int fid, int eid) {
        targets.put(fid, eid);
        System.out.printf("Set target: F%d -> E%d\n", fid, eid);
    }

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

    @Override
    public String toString() {
        String str = "";
        return str;
    }
}

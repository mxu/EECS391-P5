//
// QLUnit.java
//
// Data structure representing a single unit
// Stores its unit ID, HP, and location
//

import edu.cwru.sepia.environment.model.state.Unit.UnitView;

import java.awt.Point;

public class QLUnit {

    private int ID;
    private int HP;
    private Point loc;

    public QLUnit(UnitView unit) {
        ID = unit.getID();
        HP = unit.getHP();
        loc = new Point(unit.getXPosition(), unit.getYPosition());
    }

    public int getID() { return ID; }

    public int getHP() { return HP; }

    public Point getLoc() { return loc; }
}

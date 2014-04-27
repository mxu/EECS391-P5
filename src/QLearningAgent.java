//
// QLearningAgent.java
//
// Implements a Q-learning algorithm to teach an agent how to control footmen in combat
//


import edu.cwru.sepia.action.Action;
import edu.cwru.sepia.action.ActionType;
import edu.cwru.sepia.action.TargetedAction;
import edu.cwru.sepia.agent.Agent;
import edu.cwru.sepia.environment.model.history.DamageLog;
import edu.cwru.sepia.environment.model.history.DeathLog;
import edu.cwru.sepia.environment.model.history.History.HistoryView;
import edu.cwru.sepia.environment.model.state.State.StateView;

import java.io.InputStream;
import java.io.OutputStream;
import java.util.*;

public class QLearningAgent extends Agent {

    private static final long serialVersionUID = 0L;

    // q-learning constants
    private static final float  DISCOUNT_FACTOR = 0.9f;
    private static final float  LEARNING_RATE   = 0.0001f;
    private static final float  EPSILON         = 0.02f;

    // random number generator
    private static Random              rng;

    // agent parameters
    private int                 maxEpisodes;            // max number of episodes to play
    private int                 numEpisodes;            // number of episodes played
    private boolean             freezeQ;                // do not update Q when set to true
    private List<Float>         weights;                // weights of features to learn

    // episode parameters
    private float               cumulativeReward;       // current total reward for this episode
    private QLState             lastQls;                // cache the last state for comparing to current

    public QLearningAgent(int playernum, String[] args) {
        super(playernum);

        rng = new Random(12345L);                       // seed the RNG for consistent testing

        // parse argument and handle errors
        if(args.length > 0)
            try {
                maxEpisodes = new Integer(args[0]);
                if(maxEpisodes < 1) printUsage();       // arg must be > 0
            } catch(NumberFormatException e) {          // arg must be a number
                e.printStackTrace();
                printUsage();
            }
        else printUsage();                              // requires 1 argument

        // initialize agent parameters
        numEpisodes = 0;
        freezeQ = false;
        weights = new ArrayList<Float>();
        for(int i = 0; i < 7; i++)
            weights.add(rng.nextFloat() * 2 - 1);       // set initial weights to random value between -1 and 1

        System.out.printf("Q-Learning Agent initialized for %d episodes\n", maxEpisodes);
    }

    private void printUsage() {
        System.out.println("Usage: QLearningAgent [eps]" +
                           "\teps: number of episodes to run (must be > 0)");
        System.exit(1);
    }

    @Override
    public Map<Integer, Action> initialStep(StateView state, HistoryView stateHistory) {
        // initialize episode parameters
        cumulativeReward = 0f;

        // initialize step parameters
        Map<Integer, Action> actions = new HashMap<Integer, Action>();
        QLState qls = new QLState(state);

        // analyze the state and determine next action
        for(QLUnit f: qls.getFriendly()) {              // handle each friendly footman
            if(rng.nextFloat() < EPSILON) {             // choose a random target with probability EPSILON
                qls.setRandomTargetFor(f.getID());
            } else {                                    // otherwise follow current policy (greedy)
                float qMax = Float.NEGATIVE_INFINITY;
                Integer bestTarget = null;
                for(QLUnit e: qls.getEnemy()) {         // evaluate each enemy footman
                    float q = Q(qls, f, e);
                    if(q > qMax) {                      // found a better action, update
                        qMax = q;
                        bestTarget = e.getID();
                    }
                }
                if(bestTarget == null) {
                    System.err.printf("Could not find target for F%d\n", f.getID());
                    System.exit(1);
                }
                qls.setTarget(f.getID(), bestTarget);   // targets the best footman according to current policy
            }
        }

        lastQls = qls;                                  // cache this state

        // translate and instantiate actions
        Map<Integer, Integer> targets = qls.getTargets();
        for(int fid: targets.keySet()) {                // generate action for each targeting footman
            int eid = targets.get(fid);
            actions.put(fid, new TargetedAction(fid, ActionType.COMPOUNDATTACK, eid));
        }

        return actions;
    }

    @Override
    public Map<Integer, Action> middleStep(StateView state, HistoryView stateHistory) {
        // initialize step parameters
        Map<Integer, Action> actions = new HashMap<Integer, Action>();
        QLState qls = new QLState(state);
        boolean eventOccurred = false;

        int stepNumber = state.getTurnNumber();
        System.out.println("Step Number " + stepNumber);
        for(DamageLog log: stateHistory.getDamageLogs(stepNumber - 1)) {
            boolean harm = log.getDefenderController() == 0;
            System.out.printf("%s%d atk %s%d for %d\n",
                              harm ? "E" : "F",
                              log.getAttackerID(),
                              harm ? "F" : "E",
                              log.getDefenderID(),
                              log.getDamage());
        }
        for(DeathLog log: stateHistory.getDeathLogs(stepNumber - 1)) {
            boolean harm = log.getController() == 0;
            System.out.printf("%s%d died\n",
                              harm ? "F" : "E",
                              log.getDeadUnitID());
        }

        /* @TODO:
            to update Q for state qls and each action ATTACK(f, e)
            loop through each f
                loop through combat log for last step
                    +damage if f is attacker and e is defender
                    -damage if f is defender and e is attacker
                loop through death logs for last step
                    +100 if e died
                    -100 if f died
         */

        return actions;
    }

    @Override
    public void terminalStep(StateView state, HistoryView stateHistory) {

    }

    // evaluate Q function for s = qls and a = ATTACK(f,e)
    private float Q(QLState qls, QLUnit f, QLUnit e) {
        float q = 0;

        // linear approximation
        q += weights.get(0) +
             weights.get(1) * f.getLoc().distanceSq(e.getLoc()) +
             weights.get(2) * f.getHP() +
             weights.get(3) * e.getHP() +
             weights.get(4) * f.getLoc().x +
             weights.get(5) * f.getLoc().y +
             weights.get(6) * qls.getNumAttackersFor(f.getID());

        return q;
    }

    @Override
    public void savePlayerData(OutputStream outputStream) { }

    @Override
    public void loadPlayerData(InputStream inputStream) { }

    public static Random getRng() {
        return rng;
    }
}

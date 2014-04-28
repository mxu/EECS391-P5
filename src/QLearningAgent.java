//
// QLearningAgent.java
//
// Implements a Q-learning algorithm to teach an agent how to control footmen in combat
// - Starts with a policy of randomly weighted features
// - Plays a set of episodes during a training session to execute Q-learning
// - Freezes the policy and plays a set of episodes during an evaluation session

import edu.cwru.sepia.action.Action;
import edu.cwru.sepia.action.ActionType;
import edu.cwru.sepia.action.TargetedAction;
import edu.cwru.sepia.agent.Agent;
import edu.cwru.sepia.environment.model.history.DamageLog;
import edu.cwru.sepia.environment.model.history.DeathLog;
import edu.cwru.sepia.environment.model.history.History.HistoryView;
import edu.cwru.sepia.environment.model.state.State.StateView;
import edu.cwru.sepia.util.Pair;

import java.io.InputStream;
import java.io.OutputStream;
import java.util.*;

public class QLearningAgent extends Agent {

    private static final long   serialVersionUID    = 0L;

    private static Random       rng;
    private static boolean      debug;

    // q-learning constants
    private static final float  DISCOUNT_FACTOR     = 0.9f;
    private static final float  LEARNING_RATE       = 0.0001f;
    private static final float  EPSILON             = 0.02f;
    private static final int    NUM_FEATURES        = 9;
    private static final int    TRAINING_EPS        = 10;
    private static final int    EVALUATION_EPS      = 5;

    // agent parameters
    private int                 maxEpisodes;            // max number of episodes to play
    private int                 numEpisodes;            // number of episodes played
    private int                 evalEpisodes;           // number of episodes left for evaluation
    private float               evalTotalReward;        // cumulative reward during evaluation
    private boolean             freezeQ;                // do not update Q when set to true
    private List<Float>         weights;                // weights of features to learn

    // episode parameters
    private float               episodeReward;          // current total reward for this episode
    private QLState             lastQls;                // cache the last state for comparing to current

    public QLearningAgent(int playernum, String[] args) {
        super(playernum);

        rng                     = new Random(12345L);   // seed the RNG for consistent testing

        // parse argument and handle errors
        if(args.length > 0) {
            try {
                maxEpisodes     = Integer.parseInt(args[0]);
                if(maxEpisodes < 1) printUsage();       // first arg must be > 0
            } catch(NumberFormatException e) {          // first arg must be a number
                e.printStackTrace();
                printUsage();
            }

            debug               = args.length > 1 && Boolean.parseBoolean(args[1]);
        } else printUsage();

        // initialize agent parameters
        numEpisodes             = 0;
        evalEpisodes            = 0;
        evalTotalReward         = 0f;
        freezeQ                 = false;
        weights                 = new ArrayList<Float>();
        for(int i = 0; i < NUM_FEATURES; i++)
            weights.add(rng.nextFloat() * 2f - 1f);    // set initial weights to random value between -1 and 1
        printWeights();
        normalizeWeights();

        System.out.printf("Q-Learning Agent initialized for %d episodes\n", maxEpisodes);
    }

    @Override
    public Map<Integer, Action> initialStep(StateView state, HistoryView stateHistory) {
        // initialize episode parameters
        episodeReward = 0f;

        // initialize step parameters
        QLState qls = new QLState(state);

        if(debug) System.out.printf("Initial step:\n");

        // analyze the state and determine next action for each footman
        for(QLUnit f: qls.getFriendly()) {
            int fid = f.getID();
            // choose a random target with probability EPSILON
            if(rng.nextFloat() < EPSILON) qls.setRandomTargetFor(fid);
            // otherwise choose greedy option (target footman with best Q score for current policy)
            else qls.setTarget(fid, getQMaxTarget(qls, fid).b);
        }

        lastQls = qls;                                  // cache this state
        if(debug) qls.print();

        return translateActions(qls);
    }

    @Override
    public Map<Integer, Action> middleStep(StateView state, HistoryView stateHistory) {
        // initialize step parameters
        Map<Integer, Action> actions = new HashMap<Integer, Action>();
        Map<Integer, Integer> bestTargets = new HashMap<Integer, Integer>();
        int step = state.getTurnNumber();
        QLState qls = new QLState(state);
        // get the combat logs for the last step
        List<DamageLog> damageLogs = stateHistory.getDamageLogs(step - 1);
        List<DeathLog> deathLogs = stateHistory.getDeathLogs(step - 1);
        // mark that event has occurred if any friendly units took damage or enemies died
        boolean eventOccurred = false;
        for(DamageLog log: damageLogs) {
            if(log.getDefenderController() == 0) {
                eventOccurred = true;
                break;
            }
        }

        if(!eventOccurred) {
            for(DeathLog log: deathLogs) {
                if(log.getController() == 1) {
                    eventOccurred = true;
                    break;
                }
            }
        }

        if(debug) System.out.printf("\nStep %d:\n", step);

        // decompose reward for each footman and determine best target for next move
        for(QLUnit f: lastQls.getFriendly()) {
            boolean died = false;                       // did the friendly footman die this turn?
            int fid = f.getID();                        // friendly footman id
            int tid = lastQls.getTargetFor(fid);        // enemy target id

            float reward = -0.1f;                       // each step costs -0.1

            if(debug) System.out.printf("Reward for ATTACK(%d,%d) = %.1f", fid, tid, reward);

            // check for damage dealt / taken between fid and tid
            for(DamageLog log: damageLogs) {
                if(log.getAttackerID() == fid) {
                    reward += (float)log.getDamage();   // damage dealt rewards +damage
                    if(debug) System.out.printf(" + %d", log.getDamage());
                }
                if(log.getDefenderID() == fid) {
                    reward -= (float)log.getDamage();   // damage taken costs -damage
                    if(debug) System.out.printf(" - %d", log.getDamage());
                }
            }

            // check for death of fid and tid
            for(DeathLog log: deathLogs) {
                if(log.getDeadUnitID() == tid) {
                    reward += 100f;                     // killing enemy rewards +100
                    if(debug) System.out.print(" + 100");
                }
                if(log.getDeadUnitID() == fid) {
                    reward -= 100f;                     // dying costs -100
                    died = true;
                    if(debug) System.out.print(" - 100");
                }
            }

            if(debug) System.out.printf(" = %.1f\n", reward);

            episodeReward += reward;                    // accumulate reward

            float qLast = Q(lastQls, fid, tid);
            float qMax;
            int bestTarget;

            if(died) {
                qMax = qLast;
            } else {
                // identify best targets for next move
                Pair<Float, Integer> qMaxTarget = getQMaxTarget(qls, fid);
                qMax = qMaxTarget.a;
                bestTarget = qMaxTarget.b;
                bestTargets.put(fid, bestTarget);
            }

            // update the Q function with feedback
            if(!freezeQ) {
                if(debug) System.out.printf("Difference = %.1f + %.1f * %.3f - %.3f = ", reward, DISCOUNT_FACTOR, qMax, qLast);
                float difference = reward + DISCOUNT_FACTOR * qMax - qLast;
                if(debug) System.out.printf("%.5f\n", difference);
                List<Float> features = getFeatures(lastQls, fid, tid);
                // update weights for each feature
                for(int i = 0; i < NUM_FEATURES; i++) {
                    if(debug) System.out.printf("w%d = %.5f + %.5f * %.5f * %.1f = ", i, weights.get(i), LEARNING_RATE, difference, features.get(i));
                    weights.set(i, weights.get(i) + LEARNING_RATE * difference * features.get(i));
                    if(debug) System.out.printf("%.5f\n", weights.get(i));
                }
                normalizeWeights();
            }
        }

        if(eventOccurred) {                             // reallocate targets at event point
            for(QLUnit f: qls.getFriendly()) {
                int fid = f.getID();
                // choose a random target with probability EPSILON
                if(rng.nextFloat() < EPSILON) qls.setRandomTargetFor(fid);
                // otherwise choose greedy option (target footman with best Q score for current policy)
                else qls.setTarget(fid, bestTargets.get(fid));
            }

            actions = translateActions(qls);            // generate actions
        } else {                                        // otherwise maintain same targets
            qls.copyTargetsFrom(lastQls);
        }

        lastQls = qls;                                  // update last state
        if(debug) qls.print();

        return actions;
    }

    @Override
    public void terminalStep(StateView state, HistoryView stateHistory) {
        if(evalEpisodes == 0) {
            if(debug) {
                System.out.printf("Completed episode %d with reward %.1f\n", numEpisodes, episodeReward);
                printWeights();
            } else {
                System.out.print("|");
            }
            numEpisodes++;
            if(numEpisodes % TRAINING_EPS == 0) {       // finished training
                freezeQ = true;                         // stop Q from updating
                evalEpisodes = EVALUATION_EPS;          // evaluate for next EVALUATION_EPS episodes
                evalTotalReward = 0;                    // reset cumulative reward
            }
        } else {                                        // evaluation
            if(debug) System.out.printf("Completed evaluation episode with reward %.1f\n", episodeReward);
            else System.out.print("*");
            evalTotalReward += episodeReward;           // accumulate reward
            if(--evalEpisodes == 0) {                   // finished evaluating
                System.out.println();
                freezeQ = false;
                float avgReward = evalTotalReward / EVALUATION_EPS;
                System.out.printf("Episodes Played: %d, Average Reward: %.1f\n", numEpisodes, avgReward);
                if(debug) printWeights();
                if(numEpisodes == maxEpisodes) System.exit(0);
            }
        }
    }

    // returns a list of feature values for a = ATTACK(fid, eid)
    private List<Float> getFeatures(QLState qls, int fid, int eid) {
        ArrayList<Float> features = new ArrayList<Float>();
        QLUnit f = qls.getF(fid);
        QLUnit e = qls.getE(eid);

        // first feature is constant so w0 is not modified
        features.add(1f);
        features.add((float) f.getHP());
        features.add((float) e.getHP());
        features.add((float) f.getLoc().distanceSq(e.getLoc()));
        features.add((float) qls.getNumAttackersFor(eid));
        features.add((float) qls.getFriendlyNeighborsFor(fid));
        features.add((float) qls.getEnemyNeighborsFor(fid));
        features.add((float) qls.getFriendlyNeighborsFor(eid));
        features.add((float) qls.getEnemyNeighborsFor(eid));
        return features;
    }

    // normalize weights
    private void normalizeWeights() {
        float total = 0;
        for(float w: weights)
            total += w;
        for(int i = 0; i < NUM_FEATURES; i++)
            weights.set(i, weights.get(i) / total);
    }

    // evaluate linear approximation Q function for s = qls and a = ATTACK(fid, eid)
    private float Q(QLState qls, int fid, int eid) {
        float q = 0;
        List<Float> features = getFeatures(qls, fid, eid);

        // w0 + w * f(s, a)
        for(int i = 0; i < NUM_FEATURES; i++)
            q += weights.get(i) * features.get(i);

        return q;
    }

    // returns a pair containing the max Q score for friendly footman with id = fid
    // and the id of the enemy it should attack to achieve that score
    private Pair<Float, Integer> getQMaxTarget(QLState qls, int fid) {
        float qMax = Float.NEGATIVE_INFINITY;
        Integer bestTarget = null;
        for(QLUnit e: qls.getEnemy()) {                 // evaluate each enemy footman
            int eid = e.getID();
            float q = Q(qls, fid, eid);
            if(q > qMax) {                              // found a better action, update
                qMax = q;
                bestTarget = eid;
            }
        }

        if(bestTarget == null) {                        // error out if no target found
            System.err.printf("Could not find target for F%d\n", fid);
            System.exit(1);
        }

        return new Pair<Float, Integer>(qMax, bestTarget);
    }

    // generate SEPIA actions for targets mapped in the QLState
    private Map<Integer, Action> translateActions(QLState qls) {
        Map<Integer, Action> actions = new HashMap<Integer, Action>();
        Map<Integer, Integer> targets = qls.getTargets();

        for(int fid: targets.keySet()) {                // generate action for each targeting footman
            int eid = targets.get(fid);
            actions.put(fid, new TargetedAction(fid, ActionType.COMPOUNDATTACK, eid));
        }

        return actions;
    }

    @Override
    public void savePlayerData(OutputStream outputStream) { }

    @Override
    public void loadPlayerData(InputStream inputStream) { }

    private void printUsage() {
        System.out.println("Usage: QLearningAgent [eps] [debug]");
        System.out.println("\teps: number of episodes to run (must be > 0)");
        System.out.println("\tdebug: set to true to display debug messages");
        System.exit(1);
    }

    private void printWeights() {
        System.out.println("Weights for ATTACK(F,E):");
        System.out.printf("%f\tw0\n", weights.get(0));
        System.out.printf("%f\tHP of F\n", weights.get(1));
        System.out.printf("%f\tHP of E\n", weights.get(2));
        System.out.printf("%f\tSquare distance between F and E\n", weights.get(3));
        System.out.printf("%f\tNumber of units attacking E\n", weights.get(4));
        System.out.printf("%f\tNumber of friendly units adjacent to F\n", weights.get(5));
        System.out.printf("%f\tNumber of enemy units adjacent to F\n", weights.get(6));
        System.out.printf("%f\tNumber of friendly units adjacent to E\n", weights.get(7));
        System.out.printf("%f\tNumber of enemy units adjacent to E\n", weights.get(8));
    }

    public static Random getRng() { return rng; }

    public static boolean getDebug() { return debug; }
}

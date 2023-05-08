CREATE TABLE test (
    id serial,
    workout_num INTEGER NOT NULL,
    PRIMARY KEY (id)
);

CREATE TABLE client (
    id serial, 
    name VARCHAR(200) NOT NULL,
    PRIMARY KEY (id)
);

CREATE TABLE blocks(
    id serial,
    client_id INTEGER NOT NULL REFERENCES client(id),
    workouts INTEGER NOT NULL, 
    PRIMARY KEY (id)
);

CREATE TABLE prescriptions (
    id serial,
    block_id INTEGER NOT NULL REFERENCES blocks(id),
    workout_number INTEGER NOT NULL,
    exercise_id INTEGER NOT NULL REFERENCES exercises(id),
    sets INTEGER,
    reps INTEGER,
    weight INTEGER,
    PRIMARY KEY (id)
);


---Modified Prescriptions table for new commands for in_progress workout
ALTER TABLE prescriptions ADD COLUMN client_id integer REFERENCES client(id) ON DELETE CASCADE;


UPDATE prescriptions
SET client_id = blocks.client_id
FROM blocks
WHERE blocks.id = prescriptions.block_id;



CREATE TABLE sessions (
    id serial,
    session_date DATE,
    client_id INTEGER REFERENCES client(id) on DELETE CASCADE,
    PRIMARY KEY (id)
);

CREATE TABLE workout_exercises (
    id serial,
    workout_id INTEGER NOT NULL REFERENCES sessions(id),
    exercise_id INTEGER NOT NULL REFERENCES exercises(id),
    sets INTEGER,
    reps INTEGER,
    weight INTEGER,
    block_id INTEGER NOT NULL REFERENCES blocks(id),
    client_id INTEGER NOT NULL REFERENCES client(id),
    PRIMARY KEY (id)
);

SELECT e.exercise, sets, reps, weight
FROM workout_exercises AS we
JOIN exercises AS e ON we.exercise_id = e.id
WHERE we.workout_id = 90


SELECT id FROM blocks 
    WHERE client_id = (SELECT id FROM client WHERE name = '{name}')

CREATE TABLE actual_to_prescription (
    id serial,
    block_id INTEGER REFERENCES blocks(id) ON DELETE CASCADE,
    workout_number INTEGER,
    session_id INTEGER REFERENCES sessions(id) on DELETE CASCADE,
    PRIMARY KEY (block_id, workout_number, session_id)
);

INSERT INTO actual_to_prescription(
    block_id, workout_number, session_id VALUES (%s, %s, %s)
);

SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'actual_to_prescription'
AND column_name IN (
  SELECT column_name
  FROM information_schema.constraint_column_usage
  WHERE table_name = 'actual_to_prescription'
  AND constraint_name = 'actual_to_prescription_pkey'
);

DELETE FROM actual_to_prescription
WHERE id NOT IN (
  SELECT MIN(id)
  FROM actual_to_prescription
  GROUP BY block_id, workout_number
);

ALTER TABLE actual_to_prescription
ADD CONSTRAINT unique_block_workout
UNIQUE (block_id, workout_number);



CREATE TABLE exercises(
    id serial,
    exercise VARCHAR(200),
    PRIMARY KEY (id)
)
INSERT INTO blocks (client_id, workouts) VALUES (%s, %s), 
            (client_id, len(block))


SELECT s.session_date, we.weight
FROM sessions s
JOIN workout_exercises we ON s.id = we.workout_id
JOIN exercises e ON we.exercise_id = e.id
WHERE s.client_id = 47 AND e.exercise = 'BB Bench Press'
ORDER BY s.session_date ASC


def track_weight_changes(conn, name, exercise):
    name=st.text_input('Name')
    exercise=st.text_input('Exercise')

    cursor.execute("""
    SELECT s.session_date, we.weight
    FROM sessions s
    JOIN workout_exercises we ON s.id = we.workout_id
    JOIN exercises e ON we.exercise_id = e.id
    WHERE s.client_id = 47 AND e.exercise = 'BB Bench Press'
    ORDER BY s.session_date ASC;
    """
    )

SELECT workout_exercises.exercise_id, workout_exercises.workout_id, exercises.exercise, workout_exercises.sets, workout_exercises.reps, workout_exercises.weight
    FROM workout_exercises
    JOIN exercises ON workout_exercises."exercise_id" = exercises.id
    WHERE workout_exercises.workout_id = 74;

SELECT workout_exercises.exercise_id, exercises.exercise 
FROM workout_exercises 
JOIN exercises ON workout_exercises."exercise_id" = exercises.id
WHERE workout_id = 74;
ORDER BY workout_exercises.exercise_id;

SELECT exercise_id from workout_exercises WHERE workout_id=74;


--Aggregation command
SELECT 
    p.exercise_id, 
    SUM(we.weight) AS total_actual_weight,
    SUM(p.weight) AS total_prescribed_weight,
    SUM(we.weight) - SUM(p.weight) AS weight_difference
FROM 
    prescriptions p 
    JOIN workout_exercises we ON p.block_id = we.block_id 
    JOIN actual_to_prescription atp ON p.block_id = atp.block_id 
        AND we.workout_id = atp.session_id 
        AND p.workout_number = atp.workout_number
WHERE 
    p.block_id = 23 
        AND atp.block_id = 23
        AND atp.workout_number = 0
GROUP BY 
    p.exercise_id



--Analysis Command
SELECT
    p.workout_number as workout_number,
    p.exercise_id AS prescribed_exercises,
    p.sets AS prescribed_sets,
    p.reps AS prescribed_reps,
    p.weight AS prescribed_weight,
    we.exercise_id AS performed_exercises,
    we.sets AS actual_sets,
    we.reps AS actual_reps,
    we.weight AS actual_weight
FROM 
    prescriptions p
    JOIN workout_exercises we ON p.block_id = we.block_id 
    JOIN actual_to_prescription atp ON p.block_id = atp.block_id 
        AND we.workout_id = atp.session_id 
        AND p.workout_number = atp.workout_number
WHERE 
    p.block_id = 8
        AND atp.block_id = 8
        AND atp.workout_number = 2;


---Trigger Function that clear in_progress when a workout is submitted
CREATE OR REPLACE FUNCTION public.delete_in_progress()
  RETURNS trigger
  LANGUAGE plpgsql
AS $function$
BEGIN
  DELETE FROM in_progress WHERE client_id = NEW.client_id;
  RETURN NEW;
END;
$function$;

CREATE TRIGGER workout_exercises_insert_trigger
AFTER INSERT ON workout_exercises
FOR EACH ROW EXECUTE PROCEDURE delete_in_progress();


-- Add foreign key constraint to ensure the workout_id exists in the workout table
ALTER TABLE in_progress ADD CONSTRAINT fk_in_progress_workout_id 
FOREIGN KEY (workout_id) REFERENCES workout(id);

-- Add a partial unique index to enforce that each client can have only one in-progress workout at a time
CREATE INDEX index_workout_id ON in_progress (workout_id, client_id);

CREATE TABLE in_progress (
    id serial,
    workout_number INTEGER,
    exercise_id INTEGER NOT NULL REFERENCES exercises(id),
    sets INTEGER,
    reps INTEGER,
    weight INTEGER,
    block_id INTEGER NOT NULL REFERENCES blocks(id),
    notes TEXT,
    client_id INTEGER REFERENCES client(id) on DELETE CASCADE,
    PRIMARY KEY (id)
);


SELECT *
FROM prescriptions
WHERE block_id IN (
  SELECT 54 FROM in_progress
)
AND client_id IN (
  SELECT 71 FROM in_progress
) AND workout_number IN (SELECT 4 FROM in_progress);

---Updating workout_exercises table with client_id

UPDATE workout_exercises
SET client_id = sessions.client_id
FROM sessions
WHERE sessions.id = workout_exercises.workout_id;

UPDATE prescriptions
SET client_id = blocks.client_id
FROM blocks
WHERE blocks.id = prescriptions.block_id;


ALTER TABLE prescriptions ADD COLUMN client_id INTEGER REFERENCES client(id) on DELETE CASCADE;

ALTER TABLE workout_exercises ADD COLUMN client_id INTEGER REFERENCES client(id) on DELETE CASCADE;

---Trying to fix the new_dfs[index] error that sometimes combined people's workouts together
DELETE FROM workout_exercises WHERE block_id IS NULL;

ALTER TABLE workout_exercises 
ALTER COLUMN client_id SET NOT NULL,
ADD CONSTRAINT workout_exercises_client_id_not_null CHECK (client_id IS NOT NULL);

ALTER TABLE in_progress DROP CONSTRAINT IF EXISTS in_progress_workout_id_fkey;

-- Change the name of the workout_id column to a new name, e.g., new_workout_id
ALTER TABLE in_progress RENAME COLUMN workout_id TO new_workout_id;


-- Drop the unique index idx_exercise_sets_reps_weight because it wont allow duplicates later on
DROP INDEX IF EXISTS idx_exercise_sets_reps_weight;


-- Create a unique index on (exercise_id, sets, reps, weight) columns
-- but only for rows where exercise_id, sets, reps, and weight are not null
CREATE UNIQUE INDEX IF NOT EXISTS idx_exercise_sets_reps_weight
ON in_progress (exercise_id, sets, reps, weight, client_id, block_id, workout_number)
WHERE exercise_id IS NOT NULL AND sets IS NOT NULL AND reps IS NOT NULL AND weight IS NOT NULL AND client_id IS NOT NULL AND block_id IS NOT NULL AND workout_number IS NOT NULL;


---Creating a New Table for my model's training data
CREATE TABLE training_data (
id serial,
client_id INTEGER REFERENCES client(id) ON DELETE CASCADE,
exercise_id INTEGER NOT NULL REFERENCES exercises(exercise) ON DELETE CASCADE,
sets INTEGER NOT NULL,
reps INTEGER NOT NULL,
weight INTEGER NOT NULL
);

SELECT 
    c.name AS client_name, 
    e.exercise AS exercise_name, 
    td.weight, 
    td.sets, 
    td.reps
FROM training_data td
LEFT JOIN client c ON td.client_id = c.id
LEFT JOIN exercises e ON td.exercise_id = e.id;


----Create Trigger and Trigger Function to add new training data to the model everytime unique values for a client are performed
CREATE OR REPLACE FUNCTION insert_into_training_data()
RETURNS TRIGGER AS $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM training_data
        WHERE client_id = NEW.client_id
        AND exercise_id = NEW.exercise_id
        AND weight = NEW.weight
        AND sets = NEW.sets
        AND reps = NEW.reps
    ) THEN
        INSERT INTO training_data (client_id, exercise_id, weight, sets, reps)
        VALUES (NEW.client_id, NEW.exercise_id, NEW.weight, NEW.sets, NEW.reps);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
CREATE FUNCTION
cabral_fitness=> CREATE TRIGGER trg_insert_into_training_data
AFTER INSERT ON workout_exercises
FOR EACH ROW
EXECUTE FUNCTION insert_into_training_data();
CREATE TRIGGER
import Mathlib

namespace pink_tulips_count_l4096_409668

theorem pink_tulips_count (total : ℕ) (red_fraction blue_fraction : ℚ) : 
  total = 56 →
  red_fraction = 3 / 7 →
  blue_fraction = 3 / 8 →
  ↑total - (↑total * red_fraction + ↑total * blue_fraction) = 11 := by
  sorry

end pink_tulips_count_l4096_409668


namespace arithmetic_sequence_a9_l4096_409656

/-- An arithmetic sequence {aₙ} where a₂ = -3 and a₃ = -5 has a₉ = -17 -/
theorem arithmetic_sequence_a9 (a : ℕ → ℤ) :
  (∀ n : ℕ, a (n + 2) - a (n + 1) = a (n + 1) - a n) →  -- arithmetic sequence condition
  a 2 = -3 →
  a 3 = -5 →
  a 9 = -17 := by
  sorry

end arithmetic_sequence_a9_l4096_409656


namespace inequality_solution_l4096_409678

theorem inequality_solution (x : ℝ) : 
  (x + 3) / (x + 4) > (4 * x + 5) / (3 * x + 10) ↔ 
  (x > -10/3 ∧ x < -4) ∨ 
  (x > (-1 - Real.sqrt 41) / 4 ∧ x < (-1 + Real.sqrt 41) / 4) :=
by sorry

end inequality_solution_l4096_409678


namespace always_solution_never_solution_l4096_409641

-- Define the polynomial function
def f (a x : ℝ) : ℝ := (1 + a) * x^4 + x^3 - (3*a + 2) * x^2 - 4*a

-- Theorem 1: For all real a, x = -2 is a solution
theorem always_solution : ∀ a : ℝ, f a (-2) = 0 := by sorry

-- Theorem 2: For all real a, x = 2 is not a solution
theorem never_solution : ∀ a : ℝ, f a 2 ≠ 0 := by sorry

end always_solution_never_solution_l4096_409641


namespace second_term_arithmetic_sequence_l4096_409613

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_arithmetic_sequence (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  n * a 1 + (n * (n - 1) / 2) * (a n - a 1)

theorem second_term_arithmetic_sequence 
  (a : ℕ → ℤ) 
  (h_arithmetic : arithmetic_sequence a) 
  (h_first_term : a 1 = -2010) 
  (h_sum_condition : sum_arithmetic_sequence a 2010 / 2010 - sum_arithmetic_sequence a 2008 / 2008 = 2) :
  a 2 = -2008 :=
sorry

end second_term_arithmetic_sequence_l4096_409613


namespace stolen_newspapers_weight_l4096_409616

/-- Calculates the total weight of stolen newspapers in tons over a given number of weeks. -/
def total_weight_in_tons (weeks : ℕ) : ℚ :=
  let daily_papers : ℕ := 250
  let weekday_paper_weight : ℚ := 8 / 16
  let sunday_paper_weight : ℚ := 2 * weekday_paper_weight
  let weekdays_per_week : ℕ := 6
  let sundays_per_week : ℕ := 1
  let weekday_weight : ℚ := (weeks * weekdays_per_week * daily_papers * weekday_paper_weight)
  let sunday_weight : ℚ := (weeks * sundays_per_week * daily_papers * sunday_paper_weight)
  (weekday_weight + sunday_weight) / 2000

/-- Theorem stating that the total weight of stolen newspapers over 10 weeks is 5 tons. -/
theorem stolen_newspapers_weight : total_weight_in_tons 10 = 5 := by
  sorry

end stolen_newspapers_weight_l4096_409616


namespace total_homework_time_l4096_409666

-- Define the time left for each person
def jacob_time : ℕ := 18
def greg_time : ℕ := jacob_time - 6
def patrick_time : ℕ := 2 * greg_time - 4

-- Theorem to prove
theorem total_homework_time :
  jacob_time + greg_time + patrick_time = 50 :=
by sorry

end total_homework_time_l4096_409666


namespace art_club_officer_selection_l4096_409632

/-- Represents the Art Club with its members and officer selection rules -/
structure ArtClub where
  totalMembers : ℕ
  officerCount : ℕ
  andyIndex : ℕ
  breeIndex : ℕ
  carlosIndex : ℕ
  danaIndex : ℕ

/-- Calculates the number of ways to choose officers in the Art Club -/
def chooseOfficers (club : ArtClub) : ℕ :=
  sorry

/-- Theorem stating the correct number of ways to choose officers -/
theorem art_club_officer_selection (club : ArtClub) :
  club.totalMembers = 30 ∧
  club.officerCount = 4 ∧
  club.andyIndex ≠ club.breeIndex ∧
  club.carlosIndex ≠ club.danaIndex ∧
  club.andyIndex ≤ club.totalMembers ∧
  club.breeIndex ≤ club.totalMembers ∧
  club.carlosIndex ≤ club.totalMembers ∧
  club.danaIndex ≤ club.totalMembers →
  chooseOfficers club = 369208 :=
sorry

end art_club_officer_selection_l4096_409632


namespace equation_solutions_l4096_409694

-- Define the equation
def equation (a : ℝ) (t : ℝ) : Prop :=
  (4*a*(Real.sin t)^2 + 4*a*(1 + 2*Real.sqrt 2)*Real.cos t - 4*(a - 1)*Real.sin t - 5*a + 2) / 
  (2*Real.sqrt 2*Real.cos t - Real.sin t) = 4*a

-- Define the solution set
def solution_set (a : ℝ) : Set ℝ :=
  {t : ℝ | equation a t ∧ 0 < t ∧ t < Real.pi/2}

-- Define the condition for exactly two distinct solutions
def has_two_distinct_solutions (a : ℝ) : Prop :=
  ∃ (t₁ t₂ : ℝ), t₁ ∈ solution_set a ∧ t₂ ∈ solution_set a ∧ t₁ ≠ t₂ ∧
  ∀ (t : ℝ), t ∈ solution_set a → t = t₁ ∨ t = t₂

-- The main theorem
theorem equation_solutions (a : ℝ) :
  has_two_distinct_solutions a ↔ (a > 6 ∧ a < 18 + 24*Real.sqrt 2) ∨ a > 18 + 24*Real.sqrt 2 :=
sorry

end equation_solutions_l4096_409694


namespace train_length_l4096_409652

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 90 → time = 5 → speed * time * (1000 / 3600) = 125 := by
  sorry

end train_length_l4096_409652


namespace remainder_comparison_l4096_409658

theorem remainder_comparison (P P' : ℕ) (h1 : P = P' + 10) (h2 : P % 10 = 0) (h3 : P' % 10 = 0) :
  (P^2 - P'^2) % 10 = 0 ∧ 0 % 10 = 0 := by
  sorry

end remainder_comparison_l4096_409658


namespace intersection_of_M_and_N_l4096_409605

def M : Set ℝ := {x : ℝ | -2 < x ∧ x < 2}
def N : Set ℝ := {x : ℝ | x > 1}

theorem intersection_of_M_and_N : M ∩ N = {x : ℝ | 1 < x ∧ x < 2} := by sorry

end intersection_of_M_and_N_l4096_409605


namespace fraction_simplification_l4096_409655

theorem fraction_simplification (x : ℝ) (h : x ≠ 1) :
  (x^2 - 1) / (x^2 - 2*x + 1) = (x + 1) / (x - 1) := by
  sorry

end fraction_simplification_l4096_409655


namespace pen_cost_l4096_409693

-- Define the number of pens bought by Robert
def robert_pens : ℕ := 4

-- Define the number of pens bought by Julia in terms of Robert's
def julia_pens : ℕ := 3 * robert_pens

-- Define the number of pens bought by Dorothy in terms of Julia's
def dorothy_pens : ℕ := julia_pens / 2

-- Define the total amount spent
def total_spent : ℚ := 33

-- Define the total number of pens bought
def total_pens : ℕ := dorothy_pens + julia_pens + robert_pens

-- Theorem: The cost of one pen is $1.50
theorem pen_cost : total_spent / total_pens = 3/2 := by
  sorry

end pen_cost_l4096_409693


namespace traffic_light_change_probability_l4096_409688

/-- Represents a traffic light cycle -/
structure TrafficLightCycle where
  green : ℕ
  yellow : ℕ
  red : ℕ

/-- Calculates the total cycle time -/
def cycleDuration (cycle : TrafficLightCycle) : ℕ :=
  cycle.green + cycle.yellow + cycle.red

/-- Calculates the number of seconds where a color change can be observed -/
def changeObservationWindow (cycle : TrafficLightCycle) (observationDuration : ℕ) : ℕ :=
  3 * observationDuration  -- 3 color changes per cycle

/-- The probability of observing a color change -/
def probabilityOfChange (cycle : TrafficLightCycle) (observationDuration : ℕ) : ℚ :=
  changeObservationWindow cycle observationDuration / cycleDuration cycle

theorem traffic_light_change_probability :
  let cycle := TrafficLightCycle.mk 45 5 40
  let observationDuration := 4
  probabilityOfChange cycle observationDuration = 2 / 15 := by
  sorry


end traffic_light_change_probability_l4096_409688


namespace population_growth_l4096_409612

theorem population_growth (initial_population : ℝ) :
  let growth_factor1 := 1 + 0.05
  let growth_factor2 := 1 + 0.10
  let growth_factor3 := 1 + 0.15
  let final_population := initial_population * growth_factor1 * growth_factor2 * growth_factor3
  (final_population - initial_population) / initial_population * 100 = 33.075 := by
sorry

end population_growth_l4096_409612


namespace min_value_geometric_sequence_l4096_409643

/-- Given a geometric sequence with first term b₁ = 2, the minimum value of 3b₂ + 6b₃ is -3/4,
    where b₂ and b₃ are the second and third terms of the sequence respectively. -/
theorem min_value_geometric_sequence (b₁ b₂ b₃ : ℝ) : 
  b₁ = 2 → (∃ r : ℝ, b₂ = b₁ * r ∧ b₃ = b₂ * r) → 
  (∀ r : ℝ, b₂ = 2 * r → b₃ = 2 * r^2 → 3 * b₂ + 6 * b₃ ≥ -3/4) ∧ 
  (∃ r : ℝ, b₂ = 2 * r ∧ b₃ = 2 * r^2 ∧ 3 * b₂ + 6 * b₃ = -3/4) :=
by sorry

end min_value_geometric_sequence_l4096_409643


namespace log_sum_minus_exp_equality_l4096_409602

theorem log_sum_minus_exp_equality : Real.log 2 / Real.log 10 + Real.log 5 / Real.log 10 - 42 * 8^(1/4) - 2017^0 = -2 := by
  sorry

end log_sum_minus_exp_equality_l4096_409602


namespace percentage_of_men_in_class_l4096_409687

theorem percentage_of_men_in_class (
  women_science_major_percentage : Real)
  (non_science_major_percentage : Real)
  (men_science_major_percentage : Real)
  (h1 : women_science_major_percentage = 0.1)
  (h2 : non_science_major_percentage = 0.6)
  (h3 : men_science_major_percentage = 0.8500000000000001)
  : Real :=
by
  sorry

#check percentage_of_men_in_class

end percentage_of_men_in_class_l4096_409687


namespace arithmetic_sequence_triangle_cosine_identity_l4096_409618

theorem arithmetic_sequence_triangle_cosine_identity (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  a / (Real.sin A) = b / (Real.sin B) ∧
  b / (Real.sin B) = c / (Real.sin C) ∧
  2 * b = a + c →
  5 * Real.cos A - 4 * Real.cos A * Real.cos C + 5 * Real.cos C = 8 :=
by sorry

end arithmetic_sequence_triangle_cosine_identity_l4096_409618


namespace circle_and_tangent_line_l4096_409610

-- Define the vertices of the triangle and point P
def A : ℝ × ℝ := (8, 0)
def B : ℝ × ℝ := (0, 6)
def O : ℝ × ℝ := (0, 0)
def P : ℝ × ℝ := (-1, 5)

-- Define the circumcircle equation
def is_circumcircle_equation (eq : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, eq x y ↔ (x - 4)^2 + (y - 3)^2 = 25

-- Define the tangent line equation
def is_tangent_line_equation (eq : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, eq x y ↔ (x = -1 ∨ 21*x - 20*y + 121 = 0)

-- Theorem statement
theorem circle_and_tangent_line (eq_circle eq_line : ℝ → ℝ → Prop) : 
  is_circumcircle_equation eq_circle ∧ is_tangent_line_equation eq_line :=
sorry

end circle_and_tangent_line_l4096_409610


namespace second_number_equality_l4096_409683

theorem second_number_equality : ∃ x : ℤ, (9548 + x = 3362 + 13500) ∧ (x = 7314) := by
  sorry

end second_number_equality_l4096_409683


namespace base_prime_repr_360_l4096_409609

/-- Base prime representation of a natural number -/
def base_prime_repr (n : ℕ) : ℕ := sorry

/-- Prime factorization of a natural number -/
def prime_factorization (n : ℕ) : List (ℕ × ℕ) := sorry

theorem base_prime_repr_360 :
  base_prime_repr 360 = 321 := by sorry

end base_prime_repr_360_l4096_409609


namespace harry_milk_bottles_l4096_409606

theorem harry_milk_bottles (initial : ℕ) (jason_bought : ℕ) (final : ℕ) 
  (h1 : initial = 35) 
  (h2 : jason_bought = 5) 
  (h3 : final = 24) : 
  initial - jason_bought - final = 6 := by
  sorry

#check harry_milk_bottles

end harry_milk_bottles_l4096_409606


namespace loan_principal_calculation_l4096_409623

/-- Calculates the total interest paid on a loan with varying interest rates over different periods. -/
def total_interest (principal : ℝ) : ℝ :=
  principal * (0.08 * 4 + 0.10 * 6 + 0.12 * 5)

/-- Proves that for the given interest rates and periods, a principal of 8000 results in a total interest of 12160. -/
theorem loan_principal_calculation :
  ∃ (principal : ℝ), principal > 0 ∧ total_interest principal = 12160 ∧ principal = 8000 :=
by
  sorry

#eval total_interest 8000

end loan_principal_calculation_l4096_409623


namespace increasing_on_negative_reals_l4096_409657

theorem increasing_on_negative_reals (x₁ x₂ : ℝ) (h1 : x₁ < 0) (h2 : x₂ < 0) (h3 : x₁ < x₂) :
  2 * x₁ < 2 * x₂ := by
  sorry

end increasing_on_negative_reals_l4096_409657


namespace trig_identity_l4096_409670

theorem trig_identity : (Real.cos (10 * π / 180) - 2 * Real.sin (20 * π / 180)) / Real.sin (10 * π / 180) = Real.sqrt 3 := by
  sorry

end trig_identity_l4096_409670


namespace quadratic_factorization_l4096_409628

theorem quadratic_factorization (x : ℝ) : 16 * x^2 - 40 * x + 25 = (4 * x - 5)^2 := by
  sorry

end quadratic_factorization_l4096_409628


namespace mangoes_per_jar_l4096_409630

/-- Given the following conditions about Jordan's mangoes:
    - Jordan picked 54 mangoes
    - One-third of the mangoes were ripe
    - Two-thirds of the mangoes were unripe
    - Jordan kept 16 unripe mangoes
    - The remainder was given to his sister
    - His sister made 5 jars of pickled mangoes
    Prove that it takes 4 mangoes to fill one jar. -/
theorem mangoes_per_jar :
  ∀ (total_mangoes : ℕ)
    (ripe_fraction : ℚ)
    (unripe_fraction : ℚ)
    (kept_unripe : ℕ)
    (num_jars : ℕ),
  total_mangoes = 54 →
  ripe_fraction = 1/3 →
  unripe_fraction = 2/3 →
  kept_unripe = 16 →
  num_jars = 5 →
  (total_mangoes : ℚ) * unripe_fraction - kept_unripe = num_jars * 4 :=
by sorry

end mangoes_per_jar_l4096_409630


namespace twenty_team_tournament_games_l4096_409679

/-- Calculates the number of games in a single-elimination tournament. -/
def tournament_games (num_teams : ℕ) : ℕ :=
  num_teams - 1

/-- Theorem: A single-elimination tournament with 20 teams requires 19 games. -/
theorem twenty_team_tournament_games :
  tournament_games 20 = 19 := by
  sorry

#eval tournament_games 20

end twenty_team_tournament_games_l4096_409679


namespace smallest_in_S_l4096_409635

def S : Set ℕ := {5, 8, 1, 2, 6}

theorem smallest_in_S : ∃ m ∈ S, ∀ n ∈ S, m ≤ n ∧ m = 1 := by
  sorry

end smallest_in_S_l4096_409635


namespace min_value_sum_l4096_409619

theorem min_value_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2/x + 1/y = 1) :
  x + 2*y ≥ 8 := by
sorry

end min_value_sum_l4096_409619


namespace larry_jogging_time_l4096_409674

/-- Represents the number of days Larry jogs in the first week -/
def days_first_week : ℕ := 3

/-- Represents the number of days Larry jogs in the second week -/
def days_second_week : ℕ := 5

/-- Represents the total number of hours Larry jogs in two weeks -/
def total_hours : ℕ := 4

/-- Calculates the total number of days Larry jogs in two weeks -/
def total_days : ℕ := days_first_week + days_second_week

/-- Converts hours to minutes -/
def hours_to_minutes (hours : ℕ) : ℕ := hours * 60

/-- Represents the total jogging time in minutes -/
def total_minutes : ℕ := hours_to_minutes total_hours

/-- Theorem: Larry jogs for 30 minutes each day -/
theorem larry_jogging_time :
  total_minutes / total_days = 30 := by sorry

end larry_jogging_time_l4096_409674


namespace joshua_borrowed_amount_l4096_409669

/-- The cost of the pen in dollars -/
def pen_cost : ℚ := 6

/-- The amount Joshua has in dollars -/
def joshua_has : ℚ := 5

/-- The additional amount Joshua needs in cents -/
def additional_cents : ℚ := 32 / 100

/-- The amount Joshua borrowed in cents -/
def borrowed_amount : ℚ := 132 / 100

theorem joshua_borrowed_amount :
  borrowed_amount = (pen_cost - joshua_has) * 100 + additional_cents := by
  sorry

end joshua_borrowed_amount_l4096_409669


namespace proportion_problem_l4096_409699

theorem proportion_problem (x : ℝ) : (18 / 12 = x / (6 * 60)) → x = 540 := by sorry

end proportion_problem_l4096_409699


namespace train_passing_time_l4096_409661

/-- Proves that a train of given length and speed takes the calculated time to pass a stationary point. -/
theorem train_passing_time (train_length : ℝ) (train_speed_kmh : ℝ) (passing_time : ℝ) : 
  train_length = 500 → 
  train_speed_kmh = 72 → 
  passing_time = 25 → 
  passing_time = train_length / (train_speed_kmh * (5/18)) := by
  sorry

#check train_passing_time

end train_passing_time_l4096_409661


namespace first_player_winning_strategy_l4096_409665

/-- Represents the game board -/
def GameBoard := Fin 2020 → Fin 2020 → Option Bool

/-- Checks if there are k consecutive cells of the same color in a row or column -/
def has_k_consecutive (board : GameBoard) (k : ℕ) : Prop :=
  sorry

/-- Represents a strategy for a player -/
def Strategy := GameBoard → Fin 2020 × Fin 2020

/-- Checks if a strategy is winning for the first player -/
def is_winning_strategy (strategy : Strategy) (k : ℕ) : Prop :=
  sorry

/-- The main theorem stating the condition for the first player's winning strategy -/
theorem first_player_winning_strategy :
  ∀ k : ℕ, (∃ strategy : Strategy, is_winning_strategy strategy k) ↔ k ≤ 1011 :=
sorry

end first_player_winning_strategy_l4096_409665


namespace largest_number_proof_l4096_409689

theorem largest_number_proof (x y z : ℝ) 
  (h_order : x < y ∧ y < z)
  (h_sum : x + y + z = 102)
  (h_diff1 : z - y = 10)
  (h_diff2 : y - x = 5) :
  z = 127 / 3 := by
sorry

end largest_number_proof_l4096_409689


namespace stratified_sampling_size_l4096_409634

/-- Represents the total population size -/
def total_population : ℕ := 260

/-- Represents the elderly population size -/
def elderly_population : ℕ := 60

/-- Represents the number of elderly people selected in the sample -/
def elderly_sample : ℕ := 3

/-- Represents the total sample size -/
def total_sample : ℕ := 13

/-- Theorem stating that the total sample size is correct given the stratified sampling conditions -/
theorem stratified_sampling_size :
  (elderly_sample : ℚ) / elderly_population = (total_sample : ℚ) / total_population :=
by sorry

end stratified_sampling_size_l4096_409634


namespace same_problem_different_algorithms_l4096_409676

-- Define the characteristics of algorithms
structure AlgorithmCharacteristics where
  finiteness : Bool
  determinacy : Bool
  sequentiality : Bool
  correctness : Bool
  nonUniqueness : Bool
  universality : Bool

-- Define the possible representations of algorithms
inductive AlgorithmRepresentation
  | NaturalLanguage
  | GraphicalLanguage
  | ProgrammingLanguage

-- Define a problem that can be solved by algorithms
structure Problem where
  description : String

-- Define an algorithm
structure Algorithm where
  steps : List String
  representation : AlgorithmRepresentation

-- Theorem: The same problem can have different algorithms
theorem same_problem_different_algorithms 
  (p : Problem) 
  (chars : AlgorithmCharacteristics) 
  (reprs : List AlgorithmRepresentation) :
  ∃ (a1 a2 : Algorithm), a1 ≠ a2 ∧ 
  (∀ (input : String), 
    (a1.steps.foldl (λ acc step => step ++ acc) input) = 
    (a2.steps.foldl (λ acc step => step ++ acc) input)) :=
sorry

end same_problem_different_algorithms_l4096_409676


namespace adas_original_seat_l4096_409617

/-- Represents the seats in the movie theater. -/
inductive Seat
| one
| two
| three
| four
| five
| six

/-- Represents the friends. -/
inductive Friend
| Ada
| Bea
| Ceci
| Dee
| Edie
| Fran

/-- Represents a seating arrangement. -/
def Arrangement := Friend → Seat

/-- Defines what it means for a seat to be an end seat. -/
def is_end_seat (s : Seat) : Prop :=
  s = Seat.one ∨ s = Seat.six

/-- Defines the movement of friends after Ada left. -/
def moved (initial final : Arrangement) : Prop :=
  (∃ s, initial Friend.Bea = s ∧ final Friend.Bea = Seat.six) ∧
  (initial Friend.Ceci = final Friend.Ceci) ∧
  (initial Friend.Dee = final Friend.Edie ∧ initial Friend.Edie = final Friend.Dee) ∧
  (∃ s t, initial Friend.Fran = s ∧ final Friend.Fran = t ∧ 
    (s = Seat.two ∧ t = Seat.one ∨
     s = Seat.three ∧ t = Seat.two ∨
     s = Seat.four ∧ t = Seat.three ∨
     s = Seat.five ∧ t = Seat.four ∨
     s = Seat.six ∧ t = Seat.five))

/-- The main theorem stating Ada's original seat. -/
theorem adas_original_seat (initial final : Arrangement) :
  moved initial final →
  is_end_seat (final Friend.Ada) →
  initial Friend.Ada = Seat.three :=
sorry

end adas_original_seat_l4096_409617


namespace fifth_term_of_geometric_sequence_l4096_409690

def geometric_sequence (a : ℕ → ℝ) (r : ℝ) :=
  ∀ n : ℕ, a (n + 1) = r * a n

theorem fifth_term_of_geometric_sequence (a : ℕ → ℝ) (y : ℝ) :
  geometric_sequence a (3 * y) →
  a 0 = 3 →
  a 1 = 9 * y →
  a 2 = 27 * y^2 →
  a 3 = 81 * y^3 →
  a 4 = 243 * y^4 :=
by
  sorry

end fifth_term_of_geometric_sequence_l4096_409690


namespace coin_difference_is_six_l4096_409672

/-- Represents the available coin denominations in cents -/
def CoinDenominations : List ℕ := [5, 10, 25]

/-- The amount Paul needs to pay in cents -/
def AmountToPay : ℕ := 45

/-- Calculates the minimum number of coins needed to make the payment -/
def MinCoins (amount : ℕ) (denominations : List ℕ) : ℕ :=
  sorry

/-- Calculates the maximum number of coins needed to make the payment -/
def MaxCoins (amount : ℕ) (denominations : List ℕ) : ℕ :=
  sorry

/-- Theorem stating the difference between max and min coins is 6 -/
theorem coin_difference_is_six :
  MaxCoins AmountToPay CoinDenominations - MinCoins AmountToPay CoinDenominations = 6 :=
sorry

end coin_difference_is_six_l4096_409672


namespace tessa_initial_apples_l4096_409650

/-- The number of apples needed to make a pie -/
def apples_for_pie : ℕ := 10

/-- The number of apples Anita gave to Tessa -/
def apples_from_anita : ℕ := 5

/-- The number of additional apples Tessa needs after receiving apples from Anita -/
def additional_apples_needed : ℕ := 1

/-- Tessa's initial number of apples -/
def initial_apples : ℕ := apples_for_pie - apples_from_anita - additional_apples_needed

theorem tessa_initial_apples :
  initial_apples = 4 := by sorry

end tessa_initial_apples_l4096_409650


namespace pool_surface_area_l4096_409654

/-- A rectangular swimming pool with given dimensions. -/
structure RectangularPool where
  length : ℝ
  width : ℝ

/-- Calculate the surface area of a rectangular pool. -/
def surfaceArea (pool : RectangularPool) : ℝ :=
  pool.length * pool.width

/-- Theorem: The surface area of a rectangular pool with length 20 meters and width 15 meters is 300 square meters. -/
theorem pool_surface_area :
  let pool : RectangularPool := { length := 20, width := 15 }
  surfaceArea pool = 300 := by
  sorry

end pool_surface_area_l4096_409654


namespace existence_of_special_sequence_l4096_409677

/-- A sequence of complex numbers -/
def ComplexSequence := ℕ → ℂ

/-- Predicate to check if a natural number is prime -/
def IsPrime (p : ℕ) : Prop := sorry

/-- Predicate to check if a series converges -/
def Converges (s : ℕ → ℂ) : Prop := sorry

/-- The main theorem -/
theorem existence_of_special_sequence :
  ∃ (a : ComplexSequence), ∀ (p : ℕ), p > 0 →
    (Converges (fun n => (a n)^p) ↔ ¬(IsPrime p)) := by sorry

end existence_of_special_sequence_l4096_409677


namespace simplify_expression_l4096_409651

theorem simplify_expression : 
  let x := 2 / (Real.sqrt 2 + Real.sqrt 3)
  let y := Real.sqrt 2 / (4 * Real.sqrt (97 + 56 * Real.sqrt 3))
  x + y = (3 * Real.sqrt 2) / 4 :=
by
  sorry

end simplify_expression_l4096_409651


namespace power_equation_solution_l4096_409603

theorem power_equation_solution (n : ℕ) : 5^29 * 4^15 = 2 * 10^n → n = 29 := by
  sorry

end power_equation_solution_l4096_409603


namespace trigonometric_equation_solution_l4096_409637

theorem trigonometric_equation_solution (x : Real) :
  8.419 * Real.sin x + Real.sqrt (2 - Real.sin x ^ 2) + Real.sin x * Real.sqrt (2 - Real.sin x ^ 2) = 3 ↔
  ∃ k : ℤ, x = π / 2 + 2 * π * k :=
by sorry

end trigonometric_equation_solution_l4096_409637


namespace count_multiples_6_or_8_not_both_l4096_409646

theorem count_multiples_6_or_8_not_both : Nat := by
  -- Define the set of positive integers less than 201
  let S := Finset.range 201

  -- Define the subsets of multiples of 6 and 8
  let mult6 := S.filter (λ n => n % 6 = 0)
  let mult8 := S.filter (λ n => n % 8 = 0)

  -- Define the set of numbers that are multiples of either 6 or 8, but not both
  let result := (mult6 ∪ mult8) \ (mult6 ∩ mult8)

  -- The theorem statement
  have : result.card = 42 := by sorry

  -- Return the result
  exact result.card

end count_multiples_6_or_8_not_both_l4096_409646


namespace brownies_left_l4096_409608

/-- Calculates the number of brownies left after consumption by Tina, her husband, and dinner guests. -/
theorem brownies_left (total : ℝ) (tina_lunch : ℝ) (tina_dinner : ℝ) (husband : ℝ) (guests : ℝ) 
  (days : ℕ) (guest_days : ℕ) : 
  total = 24 ∧ 
  tina_lunch = 1.5 ∧ 
  tina_dinner = 0.5 ∧ 
  husband = 0.75 ∧ 
  guests = 2.5 ∧ 
  days = 5 ∧ 
  guest_days = 2 → 
  total - ((tina_lunch + tina_dinner) * days + husband * days + guests * guest_days) = 5.25 := by
  sorry

end brownies_left_l4096_409608


namespace gcf_36_54_l4096_409604

theorem gcf_36_54 : Nat.gcd 36 54 = 18 := by
  sorry

end gcf_36_54_l4096_409604


namespace complex_fraction_simplification_l4096_409615

theorem complex_fraction_simplification :
  (7 + 18 * Complex.I) / (4 - 5 * Complex.I) = -62/41 + (107/41) * Complex.I :=
by sorry

end complex_fraction_simplification_l4096_409615


namespace stamps_total_proof_l4096_409649

/-- The number of stamps Lizette has -/
def lizette_stamps : ℕ := 813

/-- The number of stamps Lizette has more than Minerva -/
def lizette_minerva_diff : ℕ := 125

/-- The number of stamps Jermaine has more than Lizette -/
def jermaine_lizette_diff : ℕ := 217

/-- The total number of stamps Minerva, Lizette, and Jermaine have -/
def total_stamps : ℕ := lizette_stamps + (lizette_stamps - lizette_minerva_diff) + (lizette_stamps + jermaine_lizette_diff)

theorem stamps_total_proof : total_stamps = 2531 := by
  sorry

end stamps_total_proof_l4096_409649


namespace book_original_price_l4096_409692

theorem book_original_price (discounted_price original_price : ℝ) : 
  discounted_price = 5 → 
  discounted_price = (1 / 10) * original_price → 
  original_price = 50 := by
sorry

end book_original_price_l4096_409692


namespace no_triple_prime_l4096_409600

theorem no_triple_prime (p : ℕ) : ¬(Nat.Prime p ∧ Nat.Prime (p^2 + 4) ∧ Nat.Prime (p^2 + 6)) := by
  sorry

end no_triple_prime_l4096_409600


namespace shark_sightings_relationship_l4096_409642

/-- The number of shark sightings in Daytona Beach per year. -/
def daytona_sightings : ℕ := 26

/-- The number of shark sightings in Cape May per year. -/
def cape_may_sightings : ℕ := 7

/-- Theorem stating the relationship between shark sightings in Daytona Beach and Cape May. -/
theorem shark_sightings_relationship :
  daytona_sightings = 3 * cape_may_sightings + 5 ∧ cape_may_sightings = 7 := by
  sorry

end shark_sightings_relationship_l4096_409642


namespace inverse_of_P_l4096_409620

def P (a : ℕ) : Prop := Odd a → Prime a

theorem inverse_of_P : 
  (∀ a : ℕ, P a) ↔ (∀ a : ℕ, Prime a → Odd a) :=
sorry

end inverse_of_P_l4096_409620


namespace prob_even_modified_die_l4096_409614

/-- Represents a standard 6-sided die -/
def StandardDie : Finset ℕ := {1, 2, 3, 4, 5, 6}

/-- Total number of dots on a standard die -/
def TotalDots : ℕ := (StandardDie.sum id)

/-- Probability of a specific dot not being removed in one attempt -/
def ProbNotRemoved : ℚ := (TotalDots - 1) / TotalDots

/-- Probability of a specific dot not being removed in two attempts -/
def ProbNotRemovedTwice : ℚ := ProbNotRemoved ^ 2

/-- Probability of a specific dot being removed in two attempts -/
def ProbRemovedTwice : ℚ := 1 - ProbNotRemovedTwice

/-- Probability of losing exactly one dot from a face with n dots -/
def ProbLoseOneDot (n : ℕ) : ℚ := 2 * (n / TotalDots) * ProbNotRemoved

/-- Probability of a face with n dots remaining even after dot removal -/
def ProbRemainsEven (n : ℕ) : ℚ :=
  if n % 2 = 0
  then ProbNotRemovedTwice + (if n ≥ 2 then ProbRemovedTwice else 0)
  else ProbLoseOneDot n

/-- Theorem: The probability of rolling an even number on the modified die -/
theorem prob_even_modified_die :
  (StandardDie.sum (λ n => (1 : ℚ) / 6 * ProbRemainsEven n)) =
  (StandardDie.sum (λ n => if n % 2 = 0 then (1 : ℚ) / 6 * ProbNotRemovedTwice else 0)) +
  (StandardDie.sum (λ n => if n % 2 = 0 ∧ n ≥ 2 then (1 : ℚ) / 6 * ProbRemovedTwice else 0)) +
  (StandardDie.sum (λ n => if n % 2 = 1 then (1 : ℚ) / 6 * ProbLoseOneDot n else 0)) :=
by sorry


end prob_even_modified_die_l4096_409614


namespace rationalize_sqrt_five_twelfths_l4096_409667

theorem rationalize_sqrt_five_twelfths : 
  Real.sqrt (5 / 12) = Real.sqrt 15 / 6 := by
  sorry

end rationalize_sqrt_five_twelfths_l4096_409667


namespace football_team_size_l4096_409638

/-- The number of players on a football team -/
def total_players : ℕ := 70

/-- The number of throwers on the team -/
def throwers : ℕ := 46

/-- The number of right-handed players on the team -/
def right_handed : ℕ := 62

/-- All throwers are right-handed -/
axiom throwers_are_right_handed : throwers ≤ right_handed

/-- One third of non-throwers are left-handed -/
axiom one_third_left_handed :
  3 * (total_players - throwers - (right_handed - throwers)) = total_players - throwers

theorem football_team_size :
  total_players = 70 :=
sorry

end football_team_size_l4096_409638


namespace jade_transactions_l4096_409698

theorem jade_transactions (mabel anthony cal jade : ℕ) : 
  mabel = 90 →
  anthony = mabel + (mabel / 10) →
  cal = (2 * anthony) / 3 →
  jade = cal + 15 →
  jade = 81 :=
by
  sorry

end jade_transactions_l4096_409698


namespace quadratic_sum_zero_l4096_409682

/-- A quadratic function f(x) = ax^2 + bx + c with specific properties -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := λ x => a * x^2 + b * x + c

theorem quadratic_sum_zero 
  (a b c : ℝ) 
  (h1 : QuadraticFunction a b c 1 = 0)
  (h2 : QuadraticFunction a b c 5 = 0)
  (h_min : ∀ x, QuadraticFunction a b c x ≥ 36)
  (h_reaches_min : ∃ x, QuadraticFunction a b c x = 36) :
  a + b + c = 0 := by
  sorry

end quadratic_sum_zero_l4096_409682


namespace sqrt_x_minus_one_real_l4096_409622

theorem sqrt_x_minus_one_real (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 1) ↔ x ≥ 1 := by
  sorry

end sqrt_x_minus_one_real_l4096_409622


namespace triangle_perimeter_range_l4096_409662

/-- Given vectors a and b, function f, and triangle ABC, prove the perimeter range -/
theorem triangle_perimeter_range (x : ℝ) :
  let a : ℝ × ℝ := (1, Real.sin x)
  let b : ℝ × ℝ := (Real.cos (2*x + π/3), Real.sin x)
  let f : ℝ → ℝ := λ x => a.1 * b.1 + a.2 * b.2 - (1/2) * Real.cos (2*x)
  let c : ℝ := Real.sqrt 3
  ∃ (A B C : ℝ), 
    0 < A ∧ 0 < B ∧ 0 < C ∧
    A + B + C = π ∧
    f C = 0 ∧
    2 * Real.sqrt 3 < A + B + c ∧ A + B + c ≤ 3 * Real.sqrt 3 :=
by sorry

end triangle_perimeter_range_l4096_409662


namespace domain_f_real_iff_a_gt_one_range_f_real_iff_a_between_zero_and_one_decreasing_interval_g_l4096_409660

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) := Real.log (a * x^2 + 2 * x + 1)
noncomputable def g (x : ℝ) := Real.log (x^2 - 4 * x - 5) / Real.log (1/2)

-- Theorem 1: Domain of f is ℝ iff a > 1
theorem domain_f_real_iff_a_gt_one (a : ℝ) :
  (∀ x, ∃ y, f a x = y) ↔ a > 1 :=
sorry

-- Theorem 2: Range of f is ℝ iff 0 ≤ a ≤ 1
theorem range_f_real_iff_a_between_zero_and_one (a : ℝ) :
  (∀ y, ∃ x, f a x = y) ↔ 0 ≤ a ∧ a ≤ 1 :=
sorry

-- Theorem 3: Decreasing interval of g is (5, +∞)
theorem decreasing_interval_g :
  ∀ x₁ x₂, x₁ > 5 → x₂ > 5 → x₁ < x₂ → g x₁ > g x₂ :=
sorry

end domain_f_real_iff_a_gt_one_range_f_real_iff_a_between_zero_and_one_decreasing_interval_g_l4096_409660


namespace negation_equivalence_l4096_409601

theorem negation_equivalence : 
  (¬ ∃ x₀ : ℝ, x₀^2 + 2*x₀ + 2 < 0) ↔ (∀ x : ℝ, x^2 + 2*x + 2 ≥ 0) := by
  sorry

end negation_equivalence_l4096_409601


namespace abc_inequality_l4096_409631

theorem abc_inequality (a b c : ℝ) (ha : a = Real.rpow 0.8 0.8) 
  (hb : b = Real.rpow 0.8 0.9) (hc : c = Real.rpow 1.2 0.8) : c > a ∧ a > b := by
  sorry

end abc_inequality_l4096_409631


namespace emmas_average_speed_l4096_409685

theorem emmas_average_speed 
  (distance1 : ℝ) (time1 : ℝ) (distance2 : ℝ) (time2 : ℝ)
  (h1 : distance1 = 420)
  (h2 : time1 = 7)
  (h3 : distance2 = 480)
  (h4 : time2 = 8) :
  (distance1 + distance2) / (time1 + time2) = 60 := by
sorry

end emmas_average_speed_l4096_409685


namespace complex_equation_solution_l4096_409691

theorem complex_equation_solution (z : ℂ) : z * (1 + Complex.I)^2 = Complex.abs (1 + Complex.I)^2 → z = -Complex.I := by
  sorry

end complex_equation_solution_l4096_409691


namespace figure_to_square_l4096_409626

/-- Represents a figure on a grid --/
structure GridFigure where
  width : ℕ
  height : ℕ
  area : ℕ

/-- Represents a part of the figure after cutting --/
structure FigurePart where
  area : ℕ

/-- Represents a square --/
structure Square where
  side_length : ℕ

/-- Function to cut the figure into parts --/
def cut_figure (f : GridFigure) (n : ℕ) : List FigurePart :=
  sorry

/-- Function to check if parts can form a square --/
def can_form_square (parts : List FigurePart) : Bool :=
  sorry

/-- Main theorem statement --/
theorem figure_to_square (f : GridFigure) 
  (h1 : f.width = 6) 
  (h2 : f.height = 6) 
  (h3 : f.area = 36) : 
  ∃ (parts : List FigurePart), 
    (parts.length = 4) ∧ 
    (∀ p ∈ parts, p.area = 9) ∧
    (can_form_square parts = true) :=
  sorry

end figure_to_square_l4096_409626


namespace emma_cookies_problem_l4096_409671

theorem emma_cookies_problem :
  ∃! (N : ℕ), N < 150 ∧ N % 13 = 7 ∧ N % 8 = 5 ∧ N = 85 := by
  sorry

end emma_cookies_problem_l4096_409671


namespace three_planes_six_parts_l4096_409695

/-- A plane in 3D space -/
structure Plane3D where
  -- Define a plane (this is a simplified representation)
  dummy : Unit

/-- The number of parts that a set of planes divides the space into -/
def num_parts (planes : List Plane3D) : Nat :=
  sorry

/-- Defines if three planes are collinear -/
def are_collinear (p1 p2 p3 : Plane3D) : Prop :=
  sorry

/-- Defines if two planes are parallel -/
def are_parallel (p1 p2 : Plane3D) : Prop :=
  sorry

/-- Defines if two planes intersect -/
def intersect (p1 p2 : Plane3D) : Prop :=
  sorry

theorem three_planes_six_parts 
  (p1 p2 p3 : Plane3D) 
  (h : num_parts [p1, p2, p3] = 6) :
  (are_collinear p1 p2 p3) ∨ 
  ((are_parallel p1 p2 ∧ intersect p1 p3 ∧ intersect p2 p3) ∨
   (are_parallel p1 p3 ∧ intersect p1 p2 ∧ intersect p3 p2) ∨
   (are_parallel p2 p3 ∧ intersect p2 p1 ∧ intersect p3 p1)) :=
by
  sorry

end three_planes_six_parts_l4096_409695


namespace cube_root_simplification_l4096_409648

theorem cube_root_simplification :
  (2^9 * 3^3 * 5^3 * 11^3 : ℝ)^(1/3) = 1320 := by sorry

end cube_root_simplification_l4096_409648


namespace frustum_views_l4096_409686

/-- A frustum is a portion of a solid (usually a cone or pyramid) lying between two parallel planes cutting the solid. -/
structure Frustum where
  -- Add necessary fields to define a frustum

/-- Represents a 2D view of a 3D object -/
inductive View
  | IsoscelesTrapezoid
  | ConcentricCircles

/-- Front view of a frustum -/
def front_view (f : Frustum) : View := sorry

/-- Side view of a frustum -/
def side_view (f : Frustum) : View := sorry

/-- Top view of a frustum -/
def top_view (f : Frustum) : View := sorry

/-- Two views are congruent -/
def congruent (v1 v2 : View) : Prop := sorry

theorem frustum_views (f : Frustum) : 
  front_view f = View.IsoscelesTrapezoid ∧ 
  side_view f = View.IsoscelesTrapezoid ∧
  congruent (front_view f) (side_view f) ∧
  top_view f = View.ConcentricCircles := by sorry

end frustum_views_l4096_409686


namespace sin_sum_to_product_l4096_409627

theorem sin_sum_to_product (x : ℝ) : 
  Real.sin (3 * x) + Real.sin (5 * x) = 2 * Real.sin (4 * x) * Real.cos x := by
  sorry

end sin_sum_to_product_l4096_409627


namespace manuels_savings_l4096_409640

/-- Calculates the total savings amount given initial deposit, weekly savings, and number of weeks -/
def totalSavings (initialDeposit weeklyAmount numWeeks : ℕ) : ℕ :=
  initialDeposit + weeklyAmount * numWeeks

/-- Theorem: Manuel's total savings after 19 weeks is $500 -/
theorem manuels_savings :
  totalSavings 177 17 19 = 500 := by
  sorry

end manuels_savings_l4096_409640


namespace projection_theorem_l4096_409611

def v : Fin 2 → ℝ := ![6, -3]
def u : Fin 2 → ℝ := ![3, 0]

theorem projection_theorem :
  (((v • u) / (u • u)) • u) = ![6, 0] := by
  sorry

end projection_theorem_l4096_409611


namespace fifth_basket_price_l4096_409645

/-- Given 4 baskets with an average cost of $4 and a fifth basket that makes
    the average cost of all 5 baskets $4.8, the price of the fifth basket is $8. -/
theorem fifth_basket_price (num_initial_baskets : Nat) (initial_avg_cost : ℝ)
    (total_baskets : Nat) (new_avg_cost : ℝ) :
    num_initial_baskets = 4 →
    initial_avg_cost = 4 →
    total_baskets = 5 →
    new_avg_cost = 4.8 →
    (total_baskets * new_avg_cost - num_initial_baskets * initial_avg_cost) = 8 := by
  sorry

#check fifth_basket_price

end fifth_basket_price_l4096_409645


namespace cos_neg_570_deg_l4096_409696

-- Define the cosine function for degrees
noncomputable def cos_deg (x : ℝ) : ℝ := Real.cos (x * Real.pi / 180)

-- State the theorem
theorem cos_neg_570_deg : cos_deg (-570) = -Real.sqrt 3 / 2 := by
  sorry

end cos_neg_570_deg_l4096_409696


namespace fraction_equivalence_l4096_409644

theorem fraction_equivalence : 
  ∀ (n : ℚ), (4 + n) / (7 + n) = 7 / 9 ↔ n = 13 / 2 := by sorry

end fraction_equivalence_l4096_409644


namespace smallest_n_perfect_square_and_cube_l4096_409625

/-- A number is a perfect square if it's equal to some integer multiplied by itself. -/
def IsPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

/-- A number is a perfect cube if it's equal to some integer multiplied by itself twice. -/
def IsPerfectCube (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m * m

/-- 45 is the smallest positive integer n such that 5n is a perfect square and 3n is a perfect cube. -/
theorem smallest_n_perfect_square_and_cube :
  (∀ n : ℕ, 0 < n ∧ n < 45 → ¬(IsPerfectSquare (5 * n) ∧ IsPerfectCube (3 * n))) ∧
  (IsPerfectSquare (5 * 45) ∧ IsPerfectCube (3 * 45)) :=
sorry

end smallest_n_perfect_square_and_cube_l4096_409625


namespace smaller_cube_edge_length_l4096_409659

/-- Given a cube with volume 1000 cm³ divided into 8 equal smaller cubes,
    prove that the edge length of each smaller cube is 5 cm. -/
theorem smaller_cube_edge_length :
  ∀ (original_volume smaller_volume : ℝ) (original_edge smaller_edge : ℝ),
  original_volume = 1000 →
  smaller_volume = original_volume / 8 →
  original_volume = original_edge ^ 3 →
  smaller_volume = smaller_edge ^ 3 →
  smaller_edge = 5 := by
sorry

end smaller_cube_edge_length_l4096_409659


namespace min_value_xyz_l4096_409675

theorem min_value_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x^2 + y^2 + z^2 = 1) : 
  (x*y/z + y*z/x + z*x/y) ≥ 1 := by
  sorry

end min_value_xyz_l4096_409675


namespace power_function_through_point_l4096_409647

/-- A power function that passes through the point (2,8) -/
def f (x : ℝ) : ℝ := x^3

theorem power_function_through_point (x : ℝ) :
  f 2 = 8 ∧ f 3 = 27 := by sorry

end power_function_through_point_l4096_409647


namespace sum_calculation_l4096_409673

def sum_integers (a b : ℕ) : ℕ := (b - a + 1) * (a + b) / 2

def count_even_integers (a b : ℕ) : ℕ := (b - a) / 2 + 1

def sum_squares_odd_integers (a b : ℕ) : ℕ :=
  List.sum (List.map (λ x => x * x) (List.filter (λ x => x % 2 = 1) (List.range (b - a + 1) |>.map (λ x => x + a))))

theorem sum_calculation :
  sum_integers 30 50 + count_even_integers 30 50 + sum_squares_odd_integers 30 50 = 17661 := by
  sorry

end sum_calculation_l4096_409673


namespace proportion_equality_l4096_409629

theorem proportion_equality (h : (7 : ℚ) / 11 * 66 = 42) : (13 : ℚ) / 17 * 289 = 221 := by
  sorry

end proportion_equality_l4096_409629


namespace sin_390_degrees_l4096_409681

theorem sin_390_degrees : Real.sin (390 * π / 180) = 1 / 2 := by
  sorry

end sin_390_degrees_l4096_409681


namespace boat_journey_distance_l4096_409697

-- Define the constants
def total_time : ℝ := 4
def boat_speed : ℝ := 7.5
def stream_speed : ℝ := 2.5
def distance_AC : ℝ := 10

-- Define the theorem
theorem boat_journey_distance :
  ∃ (x : ℝ), 
    (x / (boat_speed + stream_speed) + (x + distance_AC) / (boat_speed - stream_speed) = total_time ∧ x = 20) ∨
    (x / (boat_speed + stream_speed) + (x - distance_AC) / (boat_speed - stream_speed) = total_time ∧ x = 20/3) :=
by sorry


end boat_journey_distance_l4096_409697


namespace min_value_expression_l4096_409663

theorem min_value_expression (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 1) :
  y / x + 4 / y ≥ 8 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + y₀ = 1 ∧ y₀ / x₀ + 4 / y₀ = 8 :=
by sorry

end min_value_expression_l4096_409663


namespace divisor_problem_l4096_409664

theorem divisor_problem (d : ℕ) : 
  (∃ q₁ q₂ : ℕ, 100 = q₁ * d + 4 ∧ 90 = q₂ * d + 18) → d = 24 :=
by sorry

end divisor_problem_l4096_409664


namespace baseball_team_members_l4096_409680

theorem baseball_team_members (
  pouches_per_pack : ℕ)
  (num_coaches : ℕ)
  (num_helpers : ℕ)
  (num_packs : ℕ)
  (h1 : pouches_per_pack = 6)
  (h2 : num_coaches = 3)
  (h3 : num_helpers = 2)
  (h4 : num_packs = 3)
  : ∃ (team_members : ℕ),
    team_members = num_packs * pouches_per_pack - num_coaches - num_helpers ∧
    team_members = 13 :=
by sorry

end baseball_team_members_l4096_409680


namespace product_even_implies_one_even_one_odd_l4096_409633

/-- Represents a polynomial with integer coefficients -/
def IntPolynomial (n : ℕ) := Fin n → ℤ

/-- Checks if all coefficients of a polynomial are even -/
def allEven (p : IntPolynomial n) : Prop :=
  ∀ i, 2 ∣ p i

/-- Checks if at least one coefficient of a polynomial is odd -/
def hasOddCoeff (p : IntPolynomial n) : Prop :=
  ∃ i, ¬(2 ∣ p i)

/-- Represents the product of two polynomials -/
def polyProduct (a : IntPolynomial n) (b : IntPolynomial m) : IntPolynomial (n + m - 1) :=
  sorry

/-- Main theorem -/
theorem product_even_implies_one_even_one_odd
  (n m : ℕ)
  (a : IntPolynomial n)
  (b : IntPolynomial m)
  (h1 : allEven (polyProduct a b))
  (h2 : ∃ i, ¬(4 ∣ (polyProduct a b) i)) :
  (allEven a ∧ hasOddCoeff b) ∨ (allEven b ∧ hasOddCoeff a) :=
sorry

end product_even_implies_one_even_one_odd_l4096_409633


namespace bianca_received_30_dollars_l4096_409653

/-- The amount of money Bianca received for her birthday -/
def biancas_birthday_money (num_friends : ℕ) (dollars_per_friend : ℕ) : ℕ :=
  num_friends * dollars_per_friend

/-- Theorem stating that Bianca received 30 dollars for her birthday -/
theorem bianca_received_30_dollars :
  biancas_birthday_money 5 6 = 30 := by
  sorry

end bianca_received_30_dollars_l4096_409653


namespace max_product_sum_l4096_409684

theorem max_product_sum (A M C : ℕ) (h : A + M + C = 15) :
  A * M * C + A * M + M * C + C * A ≤ 200 :=
sorry

end max_product_sum_l4096_409684


namespace z_range_difference_l4096_409624

theorem z_range_difference (x y z : ℝ) 
  (sum_eq : x + y + z = 2) 
  (prod_eq : x * y + y * z + x * z = 0) : 
  ∃ (a b : ℝ), (∀ z', (∃ x' y', x' + y' + z' = 2 ∧ x' * y' + y' * z' + x' * z' = 0) → a ≤ z' ∧ z' ≤ b) ∧ b - a = 8/3 := by
  sorry

end z_range_difference_l4096_409624


namespace work_completion_time_l4096_409636

theorem work_completion_time 
  (original_men : ℕ) 
  (original_days : ℕ) 
  (absent_men : ℕ) 
  (final_days : ℕ) :
  original_men = 15 →
  original_days = 40 →
  absent_men = 5 →
  final_days = 60 →
  original_men * original_days = (original_men - absent_men) * final_days :=
by sorry

end work_completion_time_l4096_409636


namespace factorization_problem_1_l4096_409607

theorem factorization_problem_1 (a : ℝ) : -2*a^2 + 4*a = -2*a*(a - 2) := by sorry

end factorization_problem_1_l4096_409607


namespace sum_of_distance_and_reciprocal_l4096_409639

theorem sum_of_distance_and_reciprocal (a b : ℝ) : 
  (|a| = 5 ∧ b = -(1 / (-1/3))) → (a + b = 2 ∨ a + b = -8) :=
by sorry

end sum_of_distance_and_reciprocal_l4096_409639


namespace club_selection_count_l4096_409621

/-- A club with members of different genders and service lengths -/
structure Club where
  total_members : Nat
  boys : Nat
  girls : Nat
  longest_serving_boys : Nat
  longest_serving_girls : Nat

/-- The conditions for selecting president and vice-president -/
def valid_selection (c : Club) : Prop :=
  c.total_members = 30 ∧
  c.boys = 15 ∧
  c.girls = 15 ∧
  c.longest_serving_boys = 6 ∧
  c.longest_serving_girls = 6

/-- The number of ways to select a president and vice-president -/
def selection_count (c : Club) : Nat :=
  c.longest_serving_boys * c.girls + c.longest_serving_girls * c.boys

/-- Theorem stating the number of ways to select president and vice-president -/
theorem club_selection_count (c : Club) :
  valid_selection c → selection_count c = 180 := by
  sorry

end club_selection_count_l4096_409621

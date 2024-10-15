import Mathlib

namespace NUMINAMATH_CALUDE_bus_system_stops_l2472_247275

/-- Represents a bus system in a city -/
structure BusSystem where
  num_routes : ℕ
  stops_per_route : ℕ
  travel_without_transfer : Prop
  unique_intersection : Prop
  min_stops : Prop

/-- Theorem: In a bus system with 57 routes, where you can travel between any two stops without transferring,
    each pair of routes intersects at exactly one stop, and each route has at least three stops,
    the number of stops on each route is 40. -/
theorem bus_system_stops (bs : BusSystem) 
  (h1 : bs.num_routes = 57)
  (h2 : bs.travel_without_transfer)
  (h3 : bs.unique_intersection)
  (h4 : bs.min_stops)
  : bs.stops_per_route = 40 := by
  sorry

end NUMINAMATH_CALUDE_bus_system_stops_l2472_247275


namespace NUMINAMATH_CALUDE_newspaper_prices_l2472_247282

theorem newspaper_prices :
  ∃ (x y : ℕ) (k : ℚ),
    x < 30 ∧ y < 30 ∧ 0 < k ∧ k < 1 ∧
    k * 30 = y ∧ k * x = 15 ∧
    ((x = 25 ∧ y = 18) ∨ (x = 18 ∧ y = 25)) ∧
    ∀ (x' y' : ℕ) (k' : ℚ),
      x' < 30 → y' < 30 → 0 < k' → k' < 1 →
      k' * 30 = y' → k' * x' = 15 →
      ((x' = 25 ∧ y' = 18) ∨ (x' = 18 ∧ y' = 25)) :=
by sorry

end NUMINAMATH_CALUDE_newspaper_prices_l2472_247282


namespace NUMINAMATH_CALUDE_savings_calculation_l2472_247249

theorem savings_calculation (income : ℕ) (ratio_income : ℕ) (ratio_expenditure : ℕ) 
  (h1 : income = 16000)
  (h2 : ratio_income = 5)
  (h3 : ratio_expenditure = 4) :
  income - (income * ratio_expenditure / ratio_income) = 3200 := by
  sorry

end NUMINAMATH_CALUDE_savings_calculation_l2472_247249


namespace NUMINAMATH_CALUDE_power_of_two_representation_l2472_247240

theorem power_of_two_representation (n : ℕ) (h : n ≥ 3) :
  ∃ (x y : ℤ), Odd x ∧ Odd y ∧ 2^n = 7*x^2 + y^2 :=
by sorry

end NUMINAMATH_CALUDE_power_of_two_representation_l2472_247240


namespace NUMINAMATH_CALUDE_certain_number_is_negative_eleven_l2472_247244

def binary_op (n : ℤ) : ℤ := n - (n * 5)

theorem certain_number_is_negative_eleven :
  ∃ (certain_number : ℤ),
    (binary_op 3 < certain_number) ∧
    (certain_number ≤ binary_op 4) ∧
    (∀ m : ℤ, (binary_op 3 < m) ∧ (m ≤ binary_op 4) → certain_number ≤ m) ∧
    certain_number = -11 :=
by
  sorry

end NUMINAMATH_CALUDE_certain_number_is_negative_eleven_l2472_247244


namespace NUMINAMATH_CALUDE_max_silver_tokens_l2472_247276

/-- Represents the state of tokens Alex has -/
structure TokenState where
  red : ℕ
  blue : ℕ
  silver : ℕ

/-- Represents the exchange rules -/
inductive ExchangeRule
  | RedToSilver : ExchangeRule  -- 3 red for 1 silver and 2 blue
  | BlueToSilver : ExchangeRule -- 4 blue for 1 silver and 2 red

/-- Applies an exchange rule to a token state -/
def applyExchange (state : TokenState) (rule : ExchangeRule) : Option TokenState :=
  match rule with
  | ExchangeRule.RedToSilver =>
      if state.red ≥ 3 then
        some ⟨state.red - 3, state.blue + 2, state.silver + 1⟩
      else
        none
  | ExchangeRule.BlueToSilver =>
      if state.blue ≥ 4 then
        some ⟨state.red + 2, state.blue - 4, state.silver + 1⟩
      else
        none

/-- Checks if any exchange is possible -/
def canExchange (state : TokenState) : Bool :=
  state.red ≥ 3 ∨ state.blue ≥ 4

/-- The main theorem to prove -/
theorem max_silver_tokens (initialState : TokenState) 
    (h_initial : initialState = ⟨100, 100, 0⟩) :
    ∃ (finalState : TokenState), 
      (¬canExchange finalState) ∧ 
      (finalState.silver = 88) ∧
      (∃ (exchanges : List ExchangeRule), 
        finalState = exchanges.foldl (λ s r => (applyExchange s r).getD s) initialState) :=
  sorry


end NUMINAMATH_CALUDE_max_silver_tokens_l2472_247276


namespace NUMINAMATH_CALUDE_count_valid_numbers_l2472_247251

/-- The set of digits that can be used to form the numbers. -/
def digits : Finset Nat := {0, 1, 2, 3, 4, 5}

/-- A function that checks if a number is even. -/
def isEven (n : Nat) : Bool := n % 2 = 0

/-- A function that checks if a three-digit number is less than 600. -/
def lessThan600 (n : Nat) : Bool := n < 600 ∧ n ≥ 100

/-- The set of valid hundreds digits (1 to 5). -/
def validHundreds : Finset Nat := {1, 2, 3, 4, 5}

/-- The set of valid units digits (0, 2, 4). -/
def validUnits : Finset Nat := {0, 2, 4}

/-- The main theorem stating the number of valid three-digit numbers. -/
theorem count_valid_numbers : 
  (validHundreds.card * digits.card * validUnits.card : Nat) = 90 := by sorry

end NUMINAMATH_CALUDE_count_valid_numbers_l2472_247251


namespace NUMINAMATH_CALUDE_ball_ratio_problem_l2472_247283

theorem ball_ratio_problem (white_balls red_balls : ℕ) : 
  (white_balls : ℚ) / red_balls = 3 / 2 →
  white_balls = 9 →
  red_balls = 6 := by
sorry

end NUMINAMATH_CALUDE_ball_ratio_problem_l2472_247283


namespace NUMINAMATH_CALUDE_always_possible_to_sell_tickets_l2472_247270

/-- Represents the amount a child pays (5 or 10 yuan) -/
inductive Payment
| five : Payment
| ten : Payment

/-- A queue of children represented by their payments -/
def Queue := List Payment

/-- Counts the number of each type of payment in a queue -/
def countPayments (q : Queue) : ℕ × ℕ :=
  q.foldl (λ (five, ten) p => match p with
    | Payment.five => (five + 1, ten)
    | Payment.ten => (five, ten + 1)
  ) (0, 0)

/-- Checks if it's possible to give change at each step -/
def canGiveChange (q : Queue) : Prop :=
  q.foldl (λ acc p => match p with
    | Payment.five => acc + 1
    | Payment.ten => acc - 1
  ) 0 ≥ 0

/-- The main theorem stating that it's always possible to sell tickets without running out of change -/
theorem always_possible_to_sell_tickets (q : Queue) :
  let (fives, tens) := countPayments q
  fives = tens → q.length = 2 * fives → canGiveChange q :=
sorry

#check always_possible_to_sell_tickets

end NUMINAMATH_CALUDE_always_possible_to_sell_tickets_l2472_247270


namespace NUMINAMATH_CALUDE_smallest_integer_x_l2472_247273

theorem smallest_integer_x : ∃ x : ℤ, (∀ y : ℤ, 3 * |y|^3 + 5 < 56 → x ≤ y) ∧ (3 * |x|^3 + 5 < 56) := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_x_l2472_247273


namespace NUMINAMATH_CALUDE_bucket_water_difference_l2472_247291

/-- Given two buckets with initial volumes and a water transfer between them,
    prove the resulting volume difference. -/
theorem bucket_water_difference 
  (large_initial small_initial transfer : ℕ)
  (h1 : large_initial = 7)
  (h2 : small_initial = 5)
  (h3 : transfer = 2)
  : large_initial + transfer - (small_initial - transfer) = 6 := by
  sorry

end NUMINAMATH_CALUDE_bucket_water_difference_l2472_247291


namespace NUMINAMATH_CALUDE_kim_earrings_proof_l2472_247259

/-- The number of gumballs Kim gets for each pair of earrings -/
def gumballs_per_earring : ℕ := 9

/-- The number of pairs of earrings Kim brings on the first day -/
def first_day_earrings : ℕ := 3

/-- The number of gumballs Kim eats per day -/
def gumballs_eaten_per_day : ℕ := 3

/-- The number of days the gumballs last -/
def days_gumballs_last : ℕ := 42

theorem kim_earrings_proof :
  (first_day_earrings * gumballs_per_earring + 
   2 * first_day_earrings * gumballs_per_earring + 
   (2 * first_day_earrings - 1) * gumballs_per_earring) = 
  (gumballs_eaten_per_day * days_gumballs_last) := by
  sorry

end NUMINAMATH_CALUDE_kim_earrings_proof_l2472_247259


namespace NUMINAMATH_CALUDE_closest_integer_to_6_sqrt_35_l2472_247205

theorem closest_integer_to_6_sqrt_35 : 
  ∃ n : ℤ, ∀ m : ℤ, |6 * Real.sqrt 35 - n| ≤ |6 * Real.sqrt 35 - m| ∧ n = 36 :=
sorry

end NUMINAMATH_CALUDE_closest_integer_to_6_sqrt_35_l2472_247205


namespace NUMINAMATH_CALUDE_age_ratio_change_l2472_247232

theorem age_ratio_change (father_age : ℕ) (man_age : ℕ) (years : ℕ) : 
  father_age = 60 → 
  man_age = (2 * father_age) / 5 → 
  (man_age + years) * 2 = father_age + years → 
  years = 12 := by
sorry

end NUMINAMATH_CALUDE_age_ratio_change_l2472_247232


namespace NUMINAMATH_CALUDE_expression_evaluation_l2472_247226

theorem expression_evaluation (a b : ℝ) 
  (h : |a - 2| + (b - 1/2)^2 = 0) : 
  2*(a^2*b - 3*a*b^2) - (5*a^2*b - 3*(2*a*b^2 - a^2*b) - 2) = -10 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2472_247226


namespace NUMINAMATH_CALUDE_min_value_quadratic_l2472_247219

theorem min_value_quadratic (x y : ℝ) :
  y = x^2 + 12*x + 5 →
  ∀ z : ℝ, y ≥ -31 ∧ (∃ w : ℝ, w^2 + 12*w + 5 = -31) :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l2472_247219


namespace NUMINAMATH_CALUDE_cafeteria_total_l2472_247253

/-- The total number of people in a cafeteria with checkered, horizontal, and vertical striped shirts -/
def total_people (checkered : ℕ) (horizontal : ℕ) (vertical : ℕ) : ℕ :=
  checkered + horizontal + vertical

/-- Theorem: The total number of people in the cafeteria is 40 -/
theorem cafeteria_total : 
  ∃ (checkered horizontal vertical : ℕ),
    checkered = 7 ∧ 
    horizontal = 4 * checkered ∧ 
    vertical = 5 ∧ 
    total_people checkered horizontal vertical = 40 := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_total_l2472_247253


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2472_247242

def M : Set ℝ := {x | (x + 2) * (x - 1) < 0}
def N : Set ℝ := {x | x + 1 < 0}

theorem intersection_of_M_and_N : M ∩ N = Set.Ioo (-2) (-1) := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2472_247242


namespace NUMINAMATH_CALUDE_vector_b_magnitude_l2472_247295

def a : ℝ × ℝ := (-2, -1)

theorem vector_b_magnitude (b : ℝ × ℝ) 
  (h1 : a.1 * b.1 + a.2 * b.2 = 10)
  (h2 : Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = Real.sqrt 5) : 
  Real.sqrt (b.1^2 + b.2^2) = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_b_magnitude_l2472_247295


namespace NUMINAMATH_CALUDE_mrs_hilt_dogs_l2472_247223

/-- The number of dogs Mrs. Hilt saw -/
def num_dogs : ℕ := 2

/-- The number of chickens Mrs. Hilt saw -/
def num_chickens : ℕ := 2

/-- The total number of legs Mrs. Hilt saw -/
def total_legs : ℕ := 12

/-- The number of legs each dog has -/
def dog_legs : ℕ := 4

/-- The number of legs each chicken has -/
def chicken_legs : ℕ := 2

theorem mrs_hilt_dogs :
  num_dogs * dog_legs + num_chickens * chicken_legs = total_legs :=
by sorry

end NUMINAMATH_CALUDE_mrs_hilt_dogs_l2472_247223


namespace NUMINAMATH_CALUDE_election_vote_count_l2472_247258

/-- Represents the number of votes in an election round -/
structure ElectionRound where
  totalVotes : ℕ
  firstCandidateVotes : ℕ
  secondCandidateVotes : ℕ

/-- Represents a two-round election -/
structure TwoRoundElection where
  firstRound : ElectionRound
  secondRound : ElectionRound

theorem election_vote_count (election : TwoRoundElection) : election.firstRound.totalVotes = 48000 :=
  by
  have h1 : election.firstRound.firstCandidateVotes = election.firstRound.secondCandidateVotes :=
    sorry
  have h2 : election.secondRound.totalVotes = election.firstRound.totalVotes := sorry
  have h3 : election.secondRound.firstCandidateVotes =
    election.firstRound.firstCandidateVotes - 16000 := sorry
  have h4 : election.secondRound.secondCandidateVotes =
    election.firstRound.secondCandidateVotes + 16000 := sorry
  have h5 : election.secondRound.secondCandidateVotes =
    5 * election.secondRound.firstCandidateVotes := sorry
  sorry

end NUMINAMATH_CALUDE_election_vote_count_l2472_247258


namespace NUMINAMATH_CALUDE_die_roll_frequency_l2472_247204

/-- The frequency of a specific outcome in a series of trials -/
def frequency (successful_outcomes : ℕ) (total_trials : ℕ) : ℚ :=
  successful_outcomes / total_trials

/-- The number of times the die was rolled -/
def total_rolls : ℕ := 60

/-- The number of times six appeared -/
def six_appearances : ℕ := 10

theorem die_roll_frequency :
  frequency six_appearances total_rolls = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_die_roll_frequency_l2472_247204


namespace NUMINAMATH_CALUDE_binomial_expansion_theorem_l2472_247297

theorem binomial_expansion_theorem (y b : ℚ) (m : ℕ) : 
  (Nat.choose m 4 : ℚ) * y^(m-4) * b^4 = 210 →
  (Nat.choose m 5 : ℚ) * y^(m-5) * b^5 = 462 →
  (Nat.choose m 6 : ℚ) * y^(m-6) * b^6 = 792 →
  m = 7 := by sorry

end NUMINAMATH_CALUDE_binomial_expansion_theorem_l2472_247297


namespace NUMINAMATH_CALUDE_latest_time_82_degrees_l2472_247257

-- Define the temperature function
def T (t : ℝ) : ℝ := -t^2 + 12*t + 55

-- Define the derivative of the temperature function
def T' (t : ℝ) : ℝ := -2*t + 12

-- Theorem statement
theorem latest_time_82_degrees (t : ℝ) :
  (T t = 82) ∧ (T' t < 0) →
  t = 6 + (3 * Real.sqrt 28) / 2 :=
by sorry

end NUMINAMATH_CALUDE_latest_time_82_degrees_l2472_247257


namespace NUMINAMATH_CALUDE_difference_of_numbers_l2472_247288

theorem difference_of_numbers (x y : ℝ) 
  (sum_eq : x + y = 10) 
  (diff_squares : x^2 - y^2 = 80) : 
  x - y = 8 := by
sorry

end NUMINAMATH_CALUDE_difference_of_numbers_l2472_247288


namespace NUMINAMATH_CALUDE_B_power_150_is_identity_l2472_247245

def B : Matrix (Fin 3) (Fin 3) ℝ := !![0, 1, 0; 0, 0, 1; 1, 0, 0]

theorem B_power_150_is_identity :
  B^150 = (1 : Matrix (Fin 3) (Fin 3) ℝ) := by sorry

end NUMINAMATH_CALUDE_B_power_150_is_identity_l2472_247245


namespace NUMINAMATH_CALUDE_chess_tournament_score_l2472_247281

theorem chess_tournament_score (total_games wins draws losses : ℕ) 
  (old_score : ℚ) : 
  total_games = wins + draws + losses →
  old_score = wins + (1/2 : ℚ) * draws →
  total_games = 52 →
  old_score = 35 →
  (wins : ℤ) - losses = 18 :=
by sorry

end NUMINAMATH_CALUDE_chess_tournament_score_l2472_247281


namespace NUMINAMATH_CALUDE_equation_solution_l2472_247264

theorem equation_solution : ∃ x : ℝ, 
  (x + 2 ≠ 0) ∧ 
  (x - 2 ≠ 0) ∧ 
  (2 * x / (x + 2) + x / (x - 2) = 3) ∧ 
  (x = 6) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2472_247264


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2472_247201

theorem quadratic_inequality_solution_set 
  (a b : ℝ) 
  (h : Set.Ioo (-3 : ℝ) 2 = {x : ℝ | a * x^2 + 5 * x + b < 0}) : 
  Set.Ioo (-1/3 : ℝ) (1/2 : ℝ) = {x : ℝ | b * x^2 + 5 * x + a > 0} :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2472_247201


namespace NUMINAMATH_CALUDE_intersection_A_B_l2472_247200

def A : Set ℤ := {-1, 0, 1, 2, 3}
def B : Set ℤ := {x | 3 - x < 1}

theorem intersection_A_B : A ∩ B = {3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l2472_247200


namespace NUMINAMATH_CALUDE_symmetrical_parabola_directrix_l2472_247299

/-- Given a parabola y = 2x², prove that the equation of the directrix of the parabola
    symmetrical to it with respect to the line y = x is x = -1/8 -/
theorem symmetrical_parabola_directrix (x y : ℝ) :
  (y = 2 * x^2) →  -- Original parabola
  ∃ (x₀ : ℝ), 
    (∀ (x' y' : ℝ), y'^2 = (1/2) * x' ↔ (y = x ∧ x' = y ∧ y' = x)) →  -- Symmetry condition
    (x₀ = -1/8 ∧ ∀ (x' y' : ℝ), y'^2 = (1/2) * x' → |x' - x₀| = (1/4)) :=  -- Directrix equation
sorry

end NUMINAMATH_CALUDE_symmetrical_parabola_directrix_l2472_247299


namespace NUMINAMATH_CALUDE_min_set_size_l2472_247268

theorem min_set_size (n : ℕ) : 
  let set_size := 2 * n + 1
  let median := 10
  let arithmetic_mean := 6
  let sum := arithmetic_mean * set_size
  let lower_bound := n * 1 + (n + 1) * 10
  sum ≥ lower_bound → n ≥ 4 :=
by
  sorry

end NUMINAMATH_CALUDE_min_set_size_l2472_247268


namespace NUMINAMATH_CALUDE_faster_by_plane_l2472_247272

-- Define the driving time in minutes
def driving_time : ℕ := 3 * 60 + 15

-- Define the components of the airplane trip
def airport_drive_time : ℕ := 10
def boarding_wait_time : ℕ := 20
def offboarding_time : ℕ := 10

-- Define the flight time as one-third of the driving time
def flight_time : ℕ := driving_time / 3

-- Define the total airplane trip time
def airplane_trip_time : ℕ := airport_drive_time + boarding_wait_time + flight_time + offboarding_time

-- Theorem to prove
theorem faster_by_plane : driving_time - airplane_trip_time = 90 := by
  sorry

end NUMINAMATH_CALUDE_faster_by_plane_l2472_247272


namespace NUMINAMATH_CALUDE_tangent_perpendicular_point_l2472_247221

theorem tangent_perpendicular_point (x y : ℝ) : 
  y = 1 / x →  -- P is on the curve y = 1/x
  ((-1 / x^2) * (1 / 4) = -1) →  -- Tangent line is perpendicular to x - 4y - 8 = 0
  ((x = -1/2 ∧ y = -2) ∨ (x = 1/2 ∧ y = 2)) := by
  sorry

end NUMINAMATH_CALUDE_tangent_perpendicular_point_l2472_247221


namespace NUMINAMATH_CALUDE_profit_difference_is_640_l2472_247218

/-- Calculates the difference between profit shares of two partners given their investments and the profit share of a third partner. -/
def profit_share_difference (invest_a invest_b invest_c b_profit : ℕ) : ℕ :=
  let total_parts := invest_a + invest_b + invest_c
  let part_value := b_profit * total_parts / invest_b
  let a_profit := part_value * invest_a / total_parts
  let c_profit := part_value * invest_c / total_parts
  c_profit - a_profit

/-- Theorem stating that given the investments and b's profit share, the difference between a's and c's profit shares is 640. -/
theorem profit_difference_is_640 :
  profit_share_difference 8000 10000 12000 1600 = 640 := by
  sorry

end NUMINAMATH_CALUDE_profit_difference_is_640_l2472_247218


namespace NUMINAMATH_CALUDE_double_inequality_proof_l2472_247230

theorem double_inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  let f := (1 / (x + y + z + 1)) - (1 / ((x + 1) * (y + 1) * (z + 1)))
  (0 < f) ∧ 
  (f ≤ 1/8) ∧ 
  (f = 1/8 ↔ x = 1 ∧ y = 1 ∧ z = 1) :=
by sorry


end NUMINAMATH_CALUDE_double_inequality_proof_l2472_247230


namespace NUMINAMATH_CALUDE_order_of_abc_l2472_247246

theorem order_of_abc (a b c : ℝ) 
  (ha : a = (1.1 : ℝ)^10)
  (hb : (5 : ℝ)^b = 3^a + 4^a)
  (hc : c = Real.exp a - a) : 
  b < a ∧ a < c := by sorry

end NUMINAMATH_CALUDE_order_of_abc_l2472_247246


namespace NUMINAMATH_CALUDE_quadratic_properties_l2472_247228

-- Define the quadratic function
def quadratic (a b x : ℝ) : ℝ := a * x^2 - b * x

-- State the theorem
theorem quadratic_properties
  (a b m n : ℝ)
  (h_a : a ≠ 0)
  (h_point : quadratic a b m = 2)
  (h_range : ∀ x, quadratic a b x ≥ -2/3 → x ≤ n - 1 ∨ x ≥ -3 - n) :
  (∃ x, ∀ y, quadratic a b y = quadratic a b x → y = x ∨ y = -4 - x) ∧
  (quadratic a b 1 = 2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_properties_l2472_247228


namespace NUMINAMATH_CALUDE_distance_to_SFL_is_81_miles_l2472_247274

/-- The distance to Super Fun-tastic Land -/
def distance_to_SFL (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem: The distance to Super Fun-tastic Land is 81 miles -/
theorem distance_to_SFL_is_81_miles :
  distance_to_SFL 27 3 = 81 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_SFL_is_81_miles_l2472_247274


namespace NUMINAMATH_CALUDE_ellipse_a_plus_k_equals_nine_l2472_247294

/-- An ellipse with given properties -/
structure Ellipse where
  foci1 : ℝ × ℝ
  foci2 : ℝ × ℝ
  point : ℝ × ℝ
  h : ℝ
  k : ℝ
  a : ℝ
  b : ℝ

/-- The ellipse satisfies the given conditions -/
def satisfies_conditions (e : Ellipse) : Prop :=
  e.foci1 = (1, 5) ∧
  e.foci2 = (1, 1) ∧
  e.point = (7, 3) ∧
  e.a > 0 ∧
  e.b > 0 ∧
  (e.point.1 - e.h)^2 / e.a^2 + (e.point.2 - e.k)^2 / e.b^2 = 1

/-- The theorem stating that a + k equals 9 for the given ellipse -/
theorem ellipse_a_plus_k_equals_nine (e : Ellipse) 
  (h : satisfies_conditions e) : e.a + e.k = 9 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_a_plus_k_equals_nine_l2472_247294


namespace NUMINAMATH_CALUDE_rain_probability_l2472_247265

/-- The probability of rain on Friday -/
def prob_friday : ℝ := 0.4

/-- The probability of rain on Saturday -/
def prob_saturday : ℝ := 0.5

/-- The probability of rain on Sunday, given it didn't rain on both Friday and Saturday -/
def prob_sunday_normal : ℝ := 0.3

/-- The probability of rain on Sunday, given it rained on both Friday and Saturday -/
def prob_sunday_conditional : ℝ := 0.6

/-- The probability of rain on all three days -/
def prob_all_days : ℝ := prob_friday * prob_saturday * prob_sunday_conditional

theorem rain_probability :
  prob_all_days = 0.12 :=
sorry

end NUMINAMATH_CALUDE_rain_probability_l2472_247265


namespace NUMINAMATH_CALUDE_expansion_properties_l2472_247213

/-- Given n, returns the sum of the binomial coefficients of the last three terms in (1-3x)^n -/
def sumLastThreeCoefficients (n : ℕ) : ℕ :=
  Nat.choose n (n-2) + Nat.choose n (n-1) + Nat.choose n n

/-- Returns the coefficient of the (r+1)-th term in the expansion of (1-3x)^n -/
def coefficientOfTerm (n : ℕ) (r : ℕ) : ℤ :=
  (Nat.choose n r : ℤ) * (-3) ^ r

/-- Returns the absolute value of the coefficient of the (r+1)-th term in the expansion of (1-3x)^n -/
def absCoefficient (n : ℕ) (r : ℕ) : ℕ :=
  Nat.choose n r * 3 ^ r

/-- The main theorem about the expansion of (1-3x)^n -/
theorem expansion_properties (n : ℕ) (h : sumLastThreeCoefficients n = 121) :
  (∃ r : ℕ, r = 12 ∧ ∀ k : ℕ, absCoefficient n k ≤ absCoefficient n r) ∧
  (∀ k : ℕ, Nat.choose n k ≤ Nat.choose n 7 ∧ Nat.choose n k ≤ Nat.choose n 8) :=
sorry

end NUMINAMATH_CALUDE_expansion_properties_l2472_247213


namespace NUMINAMATH_CALUDE_patrick_has_25_dollars_l2472_247254

/-- Calculates the amount of money Patrick has after saving for a bicycle and lending money to a friend. -/
def patricks_money (bicycle_price : ℕ) (amount_lent : ℕ) : ℕ :=
  bicycle_price / 2 - amount_lent

/-- Proves that Patrick has $25 after saving for a $150 bicycle and lending $50 to a friend. -/
theorem patrick_has_25_dollars :
  patricks_money 150 50 = 25 := by
  sorry

end NUMINAMATH_CALUDE_patrick_has_25_dollars_l2472_247254


namespace NUMINAMATH_CALUDE_amy_remaining_money_l2472_247227

theorem amy_remaining_money (initial_amount : ℝ) 
  (doll_cost doll_quantity : ℝ)
  (board_game_cost board_game_quantity : ℝ)
  (comic_book_cost comic_book_quantity : ℝ) :
  initial_amount = 100 ∧
  doll_cost = 1.25 ∧
  doll_quantity = 3 ∧
  board_game_cost = 12.75 ∧
  board_game_quantity = 2 ∧
  comic_book_cost = 3.50 ∧
  comic_book_quantity = 4 →
  initial_amount - (doll_cost * doll_quantity + board_game_cost * board_game_quantity + comic_book_cost * comic_book_quantity) = 56.75 := by
sorry

end NUMINAMATH_CALUDE_amy_remaining_money_l2472_247227


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l2472_247261

theorem partial_fraction_decomposition (N₁ N₂ : ℚ) :
  (∀ x : ℚ, x ≠ 1 → x ≠ 3 → (42 * x - 37) / (x^2 - 4*x + 3) = N₁ / (x - 1) + N₂ / (x - 3)) →
  N₁ * N₂ = -445/4 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l2472_247261


namespace NUMINAMATH_CALUDE_other_diagonal_length_l2472_247277

/-- A trapezoid with diagonals intersecting at a right angle -/
structure RightAngleDiagonalTrapezoid where
  midline : ℝ
  diagonal1 : ℝ
  diagonal2 : ℝ
  diagonals_perpendicular : diagonal1 * diagonal2 = midline * midline * 4

/-- Theorem: In a trapezoid with diagonals intersecting at a right angle,
    if the midline is 6.5 and one diagonal is 12, then the other diagonal is 5 -/
theorem other_diagonal_length
  (t : RightAngleDiagonalTrapezoid)
  (h1 : t.midline = 6.5)
  (h2 : t.diagonal1 = 12) :
  t.diagonal2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_other_diagonal_length_l2472_247277


namespace NUMINAMATH_CALUDE_possible_values_of_x_l2472_247224

def S (x : ℝ) : Set ℝ := {1, 2, x^2}

theorem possible_values_of_x : {x : ℝ | x ∈ S x} = {0, 2} := by sorry

end NUMINAMATH_CALUDE_possible_values_of_x_l2472_247224


namespace NUMINAMATH_CALUDE_problem_solution_l2472_247247

theorem problem_solution : 
  (((Real.sqrt 48) / (Real.sqrt 3) - (Real.sqrt (1/2)) * (Real.sqrt 12) + (Real.sqrt 24)) = 4 + Real.sqrt 6) ∧
  ((Real.sqrt 5 + Real.sqrt 2) * (Real.sqrt 5 - Real.sqrt 2) + (Real.sqrt 3 - 1)^2 = 7 - 2 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2472_247247


namespace NUMINAMATH_CALUDE_cubic_root_sum_l2472_247293

theorem cubic_root_sum (a b c d : ℝ) (ha : a ≠ 0) 
  (h1 : a * 5^3 + b * 5^2 + c * 5 + d = 0)
  (h2 : a * (-3)^3 + b * (-3)^2 + c * (-3) + d = 0) :
  (b + c) / a = -19 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l2472_247293


namespace NUMINAMATH_CALUDE_multiply_whole_and_mixed_number_l2472_247250

theorem multiply_whole_and_mixed_number :
  7 * (9 + 2 / 5) = 65 + 4 / 5 := by sorry

end NUMINAMATH_CALUDE_multiply_whole_and_mixed_number_l2472_247250


namespace NUMINAMATH_CALUDE_calculate_savings_l2472_247207

/-- Calculates a person's savings given their income and income-to-expenditure ratio --/
theorem calculate_savings (income : ℕ) (income_ratio expenditure_ratio : ℕ) : 
  income_ratio > 0 ∧ expenditure_ratio > 0 ∧ income = 21000 ∧ income_ratio = 3 ∧ expenditure_ratio = 2 →
  income - (income * expenditure_ratio / income_ratio) = 7000 := by
sorry

end NUMINAMATH_CALUDE_calculate_savings_l2472_247207


namespace NUMINAMATH_CALUDE_fold_theorem_l2472_247210

/-- Represents a point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line on a 2D plane -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Defines a fold on graph paper -/
def Fold (p1 p2 p3 p4 : Point) : Prop :=
  ∃ (foldLine : Line),
    -- The fold line is perpendicular to the line connecting p1 and p2
    foldLine.slope * ((p2.x - p1.x) / (p2.y - p1.y)) = -1 ∧
    -- The midpoint of p1 and p2 is on the fold line
    (p1.y + p2.y) / 2 = foldLine.slope * ((p1.x + p2.x) / 2) + foldLine.yIntercept ∧
    -- The midpoint of p3 and p4 is on the fold line
    (p3.y + p4.y) / 2 = foldLine.slope * ((p3.x + p4.x) / 2) + foldLine.yIntercept ∧
    -- The line connecting p3 and p4 is perpendicular to the fold line
    foldLine.slope * ((p4.x - p3.x) / (p4.y - p3.y)) = -1

/-- The main theorem to prove -/
theorem fold_theorem (m n : ℝ) :
  Fold ⟨0, 3⟩ ⟨5, 0⟩ ⟨8, 5⟩ ⟨m, n⟩ → m + n = 10.3 := by
  sorry

end NUMINAMATH_CALUDE_fold_theorem_l2472_247210


namespace NUMINAMATH_CALUDE_tangent_slope_at_point_A_l2472_247286

-- Define the function
def f (x : ℝ) : ℝ := x^2 + 3*x

-- Define the point A
def point_A : ℝ × ℝ := (2, 10)

-- Theorem statement
theorem tangent_slope_at_point_A :
  (deriv f) point_A.1 = 7 :=
sorry

end NUMINAMATH_CALUDE_tangent_slope_at_point_A_l2472_247286


namespace NUMINAMATH_CALUDE_weight_lifting_problem_l2472_247280

theorem weight_lifting_problem (total_weight first_lift second_lift : ℕ) : 
  total_weight = 1500 →
  2 * first_lift = second_lift + 300 →
  first_lift + second_lift = total_weight →
  first_lift = 600 := by
sorry

end NUMINAMATH_CALUDE_weight_lifting_problem_l2472_247280


namespace NUMINAMATH_CALUDE_complex_equality_modulus_l2472_247202

theorem complex_equality_modulus (x y : ℝ) (i : ℂ) : 
  i * i = -1 →
  (2 + i) * (3 - x * i) = 3 + (y + 5) * i →
  Complex.abs (x + y * i) = 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_equality_modulus_l2472_247202


namespace NUMINAMATH_CALUDE_sequence_general_term_l2472_247241

theorem sequence_general_term (a : ℕ → ℝ) :
  (∀ n, a n > 0) →
  a 1 = 2 →
  (∀ n, a (n + 1)^2 = (a n)^2 + 2) →
  ∀ n, a n = Real.sqrt (2 * n + 2) :=
by sorry

end NUMINAMATH_CALUDE_sequence_general_term_l2472_247241


namespace NUMINAMATH_CALUDE_fraction_problem_l2472_247271

theorem fraction_problem (N : ℝ) (F : ℝ) (h : F * (1/3 * N) = 30) :
  ∃ G : ℝ, G * N = 75 ∧ G = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l2472_247271


namespace NUMINAMATH_CALUDE_strawberry_picking_l2472_247206

theorem strawberry_picking (total strawberries_JM strawberries_Z : ℕ) 
  (h1 : total = 550)
  (h2 : strawberries_JM = 350)
  (h3 : strawberries_Z = 200) :
  total - (strawberries_JM - strawberries_Z) = 400 := by
  sorry

#check strawberry_picking

end NUMINAMATH_CALUDE_strawberry_picking_l2472_247206


namespace NUMINAMATH_CALUDE_factors_of_M_l2472_247289

/-- The number of natural-number factors of M, where M = 2^5 · 3^4 · 5^3 · 7^3 · 11^2 -/
def num_factors (M : ℕ) : ℕ :=
  (5 + 1) * (4 + 1) * (3 + 1) * (3 + 1) * (2 + 1)

/-- Theorem stating that the number of natural-number factors of M is 1440 -/
theorem factors_of_M :
  let M : ℕ := 2^5 * 3^4 * 5^3 * 7^3 * 11^2
  num_factors M = 1440 := by sorry

end NUMINAMATH_CALUDE_factors_of_M_l2472_247289


namespace NUMINAMATH_CALUDE_bouquet_cost_60_l2472_247296

/-- The cost of a bouquet of tulips at Tony's Tulip Tower -/
def bouquet_cost (n : ℕ) : ℚ :=
  let base_rate := 36 / 18
  let threshold := 40
  let extra_rate := base_rate * (3/2)
  if n ≤ threshold then
    n * base_rate
  else
    threshold * base_rate + (n - threshold) * extra_rate

/-- The theorem stating the cost of a bouquet of 60 tulips -/
theorem bouquet_cost_60 : bouquet_cost 60 = 140 := by
  sorry

#eval bouquet_cost 60

end NUMINAMATH_CALUDE_bouquet_cost_60_l2472_247296


namespace NUMINAMATH_CALUDE_object_height_properties_l2472_247235

-- Define the height function
def h (t : ℝ) : ℝ := -14 * (t - 3)^2 + 140

-- Theorem statement
theorem object_height_properties :
  (∀ t : ℝ, h t ≤ h 3) ∧ (h 5 = 84) := by
  sorry

end NUMINAMATH_CALUDE_object_height_properties_l2472_247235


namespace NUMINAMATH_CALUDE_at_least_one_not_less_than_two_l2472_247269

theorem at_least_one_not_less_than_two (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  max ((b + c) / a) (max ((a + c) / b) ((a + b) / c)) ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_not_less_than_two_l2472_247269


namespace NUMINAMATH_CALUDE_range_of_m_l2472_247208

-- Define the sets A and B
def A : Set ℝ := {x | (x + 2) * (x - 5) > 0}
def B (m : ℝ) : Set ℝ := {x | m ≤ x ∧ x < m + 1}

-- Define the theorem
theorem range_of_m (m : ℝ) : 
  (B m ⊆ (Set.univ \ A)) → (-2 ≤ m ∧ m ≤ 4) :=
by
  sorry

-- Note: The proof is omitted as per the instructions

end NUMINAMATH_CALUDE_range_of_m_l2472_247208


namespace NUMINAMATH_CALUDE_crayons_difference_is_seven_l2472_247279

/-- The number of crayons Nori gave to Lea more than Mae -/
def crayons_difference : ℕ :=
  let initial_boxes : ℕ := 4
  let crayons_per_box : ℕ := 8
  let crayons_to_mae : ℕ := 5
  let crayons_left : ℕ := 15
  let initial_crayons : ℕ := initial_boxes * crayons_per_box
  let crayons_after_mae : ℕ := initial_crayons - crayons_to_mae
  let crayons_to_lea : ℕ := crayons_after_mae - crayons_left
  crayons_to_lea - crayons_to_mae

theorem crayons_difference_is_seven :
  crayons_difference = 7 := by
  sorry

end NUMINAMATH_CALUDE_crayons_difference_is_seven_l2472_247279


namespace NUMINAMATH_CALUDE_second_rectangle_perimeter_l2472_247256

theorem second_rectangle_perimeter (a b : ℝ) : 
  (a + 3) * (b + 3) - a * b = 48 →
  2 * ((a + 3) + (b + 3)) = 38 := by
sorry

end NUMINAMATH_CALUDE_second_rectangle_perimeter_l2472_247256


namespace NUMINAMATH_CALUDE_initial_men_count_l2472_247214

/-- Represents a road construction project --/
structure RoadProject where
  length : ℝ  -- Length of the road in km
  duration : ℝ  -- Total duration of the project in days
  initialProgress : ℝ  -- Length of road completed after 10 days
  initialDays : ℝ  -- Number of days for initial progress
  extraMen : ℕ  -- Number of extra men needed to finish on time

/-- Calculates the initial number of men employed in the project --/
def initialMen (project : RoadProject) : ℕ :=
  sorry

/-- Theorem stating the initial number of men for the given project --/
theorem initial_men_count (project : RoadProject) 
  (h1 : project.length = 10)
  (h2 : project.duration = 30)
  (h3 : project.initialProgress = 2)
  (h4 : project.initialDays = 10)
  (h5 : project.extraMen = 30) :
  initialMen project = 75 :=
sorry

end NUMINAMATH_CALUDE_initial_men_count_l2472_247214


namespace NUMINAMATH_CALUDE_cube_64_sqrt_is_plus_minus_2_l2472_247229

theorem cube_64_sqrt_is_plus_minus_2 (x : ℝ) (h : x^3 = 64) : 
  Real.sqrt x = 2 ∨ Real.sqrt x = -2 := by
sorry

end NUMINAMATH_CALUDE_cube_64_sqrt_is_plus_minus_2_l2472_247229


namespace NUMINAMATH_CALUDE_earnings_difference_l2472_247255

/-- Represents the delivery areas --/
inductive DeliveryArea
  | A
  | B
  | C

/-- Represents a delivery worker --/
structure DeliveryWorker where
  name : String
  deliveries : DeliveryArea → Nat

/-- Get the fee for a specific delivery area --/
def areaFee (area : DeliveryArea) : Nat :=
  match area with
  | DeliveryArea.A => 100
  | DeliveryArea.B => 125
  | DeliveryArea.C => 150

/-- Calculate the total earnings for a worker --/
def totalEarnings (worker : DeliveryWorker) : Nat :=
  (worker.deliveries DeliveryArea.A * areaFee DeliveryArea.A) +
  (worker.deliveries DeliveryArea.B * areaFee DeliveryArea.B) +
  (worker.deliveries DeliveryArea.C * areaFee DeliveryArea.C)

/-- Oula's delivery data --/
def oula : DeliveryWorker :=
  { name := "Oula"
    deliveries := fun
      | DeliveryArea.A => 48
      | DeliveryArea.B => 32
      | DeliveryArea.C => 16 }

/-- Tona's delivery data --/
def tona : DeliveryWorker :=
  { name := "Tona"
    deliveries := fun
      | DeliveryArea.A => 27
      | DeliveryArea.B => 18
      | DeliveryArea.C => 9 }

/-- The main theorem to prove --/
theorem earnings_difference : totalEarnings oula - totalEarnings tona = 4900 := by
  sorry


end NUMINAMATH_CALUDE_earnings_difference_l2472_247255


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l2472_247215

theorem pure_imaginary_complex_number (m : ℝ) : 
  (((m^2 - 5*m + 6) : ℂ) + (m^2 - 3*m)*I = (0 : ℂ) + (m^2 - 3*m)*I) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l2472_247215


namespace NUMINAMATH_CALUDE_distance_to_karasuk_proof_l2472_247298

/-- The distance from Novosibirsk to Karasuk -/
def distance_to_karasuk : ℝ := 140

/-- The speed of the bus -/
def bus_speed : ℝ := 1

/-- The speed of the car -/
def car_speed : ℝ := 2 * bus_speed

/-- The initial distance the bus traveled before the car started -/
def initial_bus_distance : ℝ := 70

/-- The distance the bus traveled after Karasuk -/
def bus_distance_after_karasuk : ℝ := 20

/-- The distance the car traveled after Karasuk -/
def car_distance_after_karasuk : ℝ := 40

theorem distance_to_karasuk_proof :
  distance_to_karasuk = initial_bus_distance + 
    (car_distance_after_karasuk * bus_speed / car_speed) :=
by sorry

end NUMINAMATH_CALUDE_distance_to_karasuk_proof_l2472_247298


namespace NUMINAMATH_CALUDE_hash_six_eight_l2472_247236

-- Define the # operation
def hash (a b : ℤ) : ℤ := 3*a - 3*b + 4

-- Theorem statement
theorem hash_six_eight : hash 6 8 = -2 := by
  sorry

end NUMINAMATH_CALUDE_hash_six_eight_l2472_247236


namespace NUMINAMATH_CALUDE_height_conversion_l2472_247262

/-- Converts a height from inches to centimeters given the conversion factors. -/
def height_in_cm (height_in : ℚ) (in_per_ft : ℚ) (cm_per_ft : ℚ) : ℚ :=
  height_in * (cm_per_ft / in_per_ft)

/-- Theorem stating that 65 inches is equivalent to 162.5 cm given the conversion factors. -/
theorem height_conversion :
  height_in_cm 65 10 25 = 162.5 := by sorry

end NUMINAMATH_CALUDE_height_conversion_l2472_247262


namespace NUMINAMATH_CALUDE_cylinder_lateral_surface_area_l2472_247287

/-- Given a cylinder formed by rotating a square around one of its sides,
    if the volume of the cylinder is 27π cm³,
    then its lateral surface area is 18π cm². -/
theorem cylinder_lateral_surface_area 
  (side : ℝ) 
  (h_cylinder : side > 0) 
  (h_volume : π * side^2 * side = 27 * π) : 
  2 * π * side * side = 18 * π :=
sorry

end NUMINAMATH_CALUDE_cylinder_lateral_surface_area_l2472_247287


namespace NUMINAMATH_CALUDE_product_equality_l2472_247233

theorem product_equality : 2.5 * 8.5 * (5.2 - 0.2) = 106.25 := by
  sorry

end NUMINAMATH_CALUDE_product_equality_l2472_247233


namespace NUMINAMATH_CALUDE_binomial_7_2_l2472_247211

theorem binomial_7_2 : Nat.choose 7 2 = 21 := by
  sorry

end NUMINAMATH_CALUDE_binomial_7_2_l2472_247211


namespace NUMINAMATH_CALUDE_angle_terminal_side_trig_sum_l2472_247260

theorem angle_terminal_side_trig_sum (α : Real) :
  (∃ (P : Real × Real), P = (-4/5, 3/5) ∧ P.1 = -4/5 ∧ P.2 = 3/5 ∧ 
   P.1^2 + P.2^2 = 1 ∧ 
   P.1 = Real.cos α ∧ P.2 = Real.sin α) →
  2 * Real.sin α + Real.cos α = 2/5 := by
sorry

end NUMINAMATH_CALUDE_angle_terminal_side_trig_sum_l2472_247260


namespace NUMINAMATH_CALUDE_garden_ant_count_l2472_247220

/-- Represents the dimensions and ant density of a rectangular garden --/
structure Garden where
  width : ℝ  -- width in feet
  length : ℝ  -- length in feet
  antDensity : ℝ  -- ants per square inch

/-- Conversion factor from feet to inches --/
def feetToInches : ℝ := 12

/-- Calculates the approximate number of ants in the garden --/
def approximateAntCount (g : Garden) : ℝ :=
  g.width * feetToInches * g.length * feetToInches * g.antDensity

/-- Theorem stating that the number of ants in the given garden is approximately 30 million --/
theorem garden_ant_count :
  let g : Garden := { width := 350, length := 300, antDensity := 2 }
  ∃ ε > 0, |approximateAntCount g - 30000000| < ε := by
  sorry

end NUMINAMATH_CALUDE_garden_ant_count_l2472_247220


namespace NUMINAMATH_CALUDE_greatest_integer_solution_two_satisfies_inequality_three_exceeds_inequality_greatest_integer_value_l2472_247231

theorem greatest_integer_solution (x : ℤ) : x^2 + 5*x < 30 → x ≤ 2 :=
by
  sorry

theorem two_satisfies_inequality : 2^2 + 5*2 < 30 :=
by
  sorry

theorem three_exceeds_inequality : ¬(3^2 + 5*3 < 30) :=
by
  sorry

theorem greatest_integer_value : ∃ (x : ℤ), x^2 + 5*x < 30 ∧ ∀ (y : ℤ), y^2 + 5*y < 30 → y ≤ x :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_solution_two_satisfies_inequality_three_exceeds_inequality_greatest_integer_value_l2472_247231


namespace NUMINAMATH_CALUDE_problem_solution_l2472_247290

theorem problem_solution (a : ℝ) (h : a = 2 / (3 - Real.sqrt 7)) :
  -2 * a^2 + 12 * a + 3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2472_247290


namespace NUMINAMATH_CALUDE_tightrope_length_calculation_l2472_247266

-- Define the length of the tightrope
def tightrope_length : ℝ := 320

-- Define the probability of breaking in the first 50 meters
def break_probability : ℝ := 0.15625

-- Theorem statement
theorem tightrope_length_calculation :
  (∀ x : ℝ, x ≥ 0 ∧ x ≤ tightrope_length → 
    (50 / tightrope_length = break_probability)) →
  tightrope_length = 320 := by
sorry

end NUMINAMATH_CALUDE_tightrope_length_calculation_l2472_247266


namespace NUMINAMATH_CALUDE_total_distance_walked_and_run_l2472_247248

/-- Calculates the total distance traveled when walking and running at given rates and times. -/
theorem total_distance_walked_and_run
  (walking_time : ℝ) (walking_rate : ℝ) (running_time : ℝ) (running_rate : ℝ) :
  walking_time = 30 / 60 →
  walking_rate = 3.5 →
  running_time = 45 / 60 →
  running_rate = 8 →
  walking_time * walking_rate + running_time * running_rate = 7.75 := by
  sorry

#check total_distance_walked_and_run

end NUMINAMATH_CALUDE_total_distance_walked_and_run_l2472_247248


namespace NUMINAMATH_CALUDE_profit_percentage_calculation_l2472_247263

/-- Calculate the profit percentage given the selling price and profit -/
theorem profit_percentage_calculation (selling_price profit : ℝ) :
  selling_price = 900 →
  profit = 225 →
  (profit / (selling_price - profit)) * 100 = 33.33333333333333 :=
by sorry

end NUMINAMATH_CALUDE_profit_percentage_calculation_l2472_247263


namespace NUMINAMATH_CALUDE_percentage_loss_calculation_l2472_247234

theorem percentage_loss_calculation (cost_price selling_price : ℝ) : 
  cost_price = 1400 →
  selling_price = 1120 →
  (cost_price - selling_price) / cost_price * 100 = 20 := by
sorry

end NUMINAMATH_CALUDE_percentage_loss_calculation_l2472_247234


namespace NUMINAMATH_CALUDE_six_digit_numbers_with_zero_count_six_digit_numbers_with_zero_l2472_247243

theorem six_digit_numbers_with_zero (total : ℕ) (no_zero : ℕ) : ℕ :=
  total - no_zero

theorem count_six_digit_numbers_with_zero :
  six_digit_numbers_with_zero 900000 531441 = 368559 := by
  sorry

end NUMINAMATH_CALUDE_six_digit_numbers_with_zero_count_six_digit_numbers_with_zero_l2472_247243


namespace NUMINAMATH_CALUDE_roots_in_intervals_l2472_247278

/-- The quadratic function f(x) = 7x^2 - (k+13)x + k^2 - k - 2 -/
def f (k : ℝ) (x : ℝ) : ℝ := 7 * x^2 - (k + 13) * x + k^2 - k - 2

/-- Theorem stating the range of k for which f(x) has roots in (0,1) and (1,2) -/
theorem roots_in_intervals (k : ℝ) : 
  (∃ x y, 0 < x ∧ x < 1 ∧ 1 < y ∧ y < 2 ∧ f k x = 0 ∧ f k y = 0) ↔ 
  ((3 < k ∧ k < 4) ∨ (-2 < k ∧ k < -1)) :=
sorry

end NUMINAMATH_CALUDE_roots_in_intervals_l2472_247278


namespace NUMINAMATH_CALUDE_expand_product_l2472_247212

theorem expand_product (x : ℝ) : (2*x + 3) * (4*x - 5) = 8*x^2 + 2*x - 15 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l2472_247212


namespace NUMINAMATH_CALUDE_farm_rent_calculation_l2472_247284

-- Define the constants
def rent_per_acre_per_month : ℝ := 60
def plot_length : ℝ := 360
def plot_width : ℝ := 1210
def square_feet_per_acre : ℝ := 43560

-- Define the theorem
theorem farm_rent_calculation :
  let plot_area : ℝ := plot_length * plot_width
  let acres : ℝ := plot_area / square_feet_per_acre
  let monthly_rent : ℝ := rent_per_acre_per_month * acres
  monthly_rent = 600 := by sorry

end NUMINAMATH_CALUDE_farm_rent_calculation_l2472_247284


namespace NUMINAMATH_CALUDE_sum_minimized_at_Q₅_l2472_247225

/-- A type representing points on a line -/
structure PointOnLine where
  position : ℝ

/-- The distance between two points on a line -/
def distance (p q : PointOnLine) : ℝ := |p.position - q.position|

/-- The sum of distances from a point Q to points Q₁, ..., Q₉ -/
def sumOfDistances (Q Q₁ Q₂ Q₃ Q₄ Q₅ Q₆ Q₇ Q₈ Q₉ : PointOnLine) : ℝ :=
  distance Q Q₁ + distance Q Q₂ + distance Q Q₃ + distance Q Q₄ + 
  distance Q Q₅ + distance Q Q₆ + distance Q Q₇ + distance Q Q₈ + distance Q Q₉

/-- The theorem stating that the sum of distances is minimized when Q is at Q₅ -/
theorem sum_minimized_at_Q₅ 
  (Q₁ Q₂ Q₃ Q₄ Q₅ Q₆ Q₇ Q₈ Q₉ : PointOnLine) 
  (h : Q₁.position < Q₂.position ∧ Q₂.position < Q₃.position ∧ 
       Q₃.position < Q₄.position ∧ Q₄.position < Q₅.position ∧ 
       Q₅.position < Q₆.position ∧ Q₆.position < Q₇.position ∧ 
       Q₇.position < Q₈.position ∧ Q₈.position < Q₉.position) :
  ∀ Q : PointOnLine, sumOfDistances Q Q₁ Q₂ Q₃ Q₄ Q₅ Q₆ Q₇ Q₈ Q₉ ≥ 
                     sumOfDistances Q₅ Q₁ Q₂ Q₃ Q₄ Q₅ Q₆ Q₇ Q₈ Q₉ :=
sorry

end NUMINAMATH_CALUDE_sum_minimized_at_Q₅_l2472_247225


namespace NUMINAMATH_CALUDE_percentage_of_x_pay_to_y_l2472_247292

/-- The percentage of X's pay compared to Y's, given their total pay and Y's pay -/
theorem percentage_of_x_pay_to_y (total_pay y_pay x_pay : ℚ) : 
  total_pay = 528 →
  y_pay = 240 →
  x_pay + y_pay = total_pay →
  (x_pay / y_pay) * 100 = 120 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_x_pay_to_y_l2472_247292


namespace NUMINAMATH_CALUDE_bus_ride_difference_l2472_247203

-- Define the lengths of the bus rides
def oscar_ride : ℝ := 0.75
def charlie_ride : ℝ := 0.25

-- Theorem statement
theorem bus_ride_difference : oscar_ride - charlie_ride = 0.50 := by
  sorry

end NUMINAMATH_CALUDE_bus_ride_difference_l2472_247203


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l2472_247217

theorem sum_of_reciprocals_of_roots (p q : ℝ) : 
  p^2 - 16*p + 9 = 0 → 
  q^2 - 16*q + 9 = 0 → 
  p ≠ q → 
  1/p + 1/q = 16/9 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l2472_247217


namespace NUMINAMATH_CALUDE_sales_solution_l2472_247222

def sales_problem (s1 s2 s3 s4 s6 : ℕ) (average : ℕ) : Prop :=
  let total := average * 6
  let known_sum := s1 + s2 + s3 + s4 + s6
  let s5 := total - known_sum
  s5 = 6562

theorem sales_solution (s1 s2 s3 s4 s6 average : ℕ) 
  (h1 : s1 = 6435) (h2 : s2 = 6927) (h3 : s3 = 6855) (h4 : s4 = 7230) 
  (h5 : s6 = 6791) (h6 : average = 6800) :
  sales_problem s1 s2 s3 s4 s6 average :=
by
  sorry

end NUMINAMATH_CALUDE_sales_solution_l2472_247222


namespace NUMINAMATH_CALUDE_max_cake_pieces_l2472_247209

/-- Represents the dimensions of a rectangular cake -/
structure CakeDimensions where
  m : ℕ
  n : ℕ

/-- Checks if the given dimensions satisfy the required condition -/
def satisfiesCondition (d : CakeDimensions) : Prop :=
  (d.m - 2) * (d.n - 2) = (d.m * d.n) / 2

/-- Calculates the total number of cake pieces -/
def totalPieces (d : CakeDimensions) : ℕ :=
  d.m * d.n

/-- Theorem stating the maximum number of cake pieces possible -/
theorem max_cake_pieces :
  ∃ (d : CakeDimensions), satisfiesCondition d ∧ 
    (∀ (d' : CakeDimensions), satisfiesCondition d' → totalPieces d' ≤ totalPieces d) ∧
    totalPieces d = 60 := by
  sorry

end NUMINAMATH_CALUDE_max_cake_pieces_l2472_247209


namespace NUMINAMATH_CALUDE_no_prime_divisor_8t_plus_5_l2472_247238

theorem no_prime_divisor_8t_plus_5 (x : ℕ+) :
  ∀ p : ℕ, Prime p → p % 8 = 5 →
    ¬(p ∣ (8 * x^4 - 2)) ∧
    ¬(p ∣ (8 * x^4 - 1)) ∧
    ¬(p ∣ (8 * x^4)) ∧
    ¬(p ∣ (8 * x^4 + 1)) :=
sorry

end NUMINAMATH_CALUDE_no_prime_divisor_8t_plus_5_l2472_247238


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2472_247237

theorem quadratic_inequality_solution_set 
  (a b c α β : ℝ) 
  (h1 : ∀ x, ax^2 + b*x + c > 0 ↔ α < x ∧ x < β) 
  (h2 : α > 0) :
  ∀ x, c*x^2 + b*x + a < 0 ↔ x < 1/β ∨ x > 1/α :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2472_247237


namespace NUMINAMATH_CALUDE_continuity_at_4_l2472_247285

def f (x : ℝ) : ℝ := 2 * x^2 - 3

theorem continuity_at_4 :
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - 4| < δ → |f x - f 4| < ε :=
by sorry

end NUMINAMATH_CALUDE_continuity_at_4_l2472_247285


namespace NUMINAMATH_CALUDE_expression_value_l2472_247267

theorem expression_value (m n : ℝ) (h : m * n = m + 3) : 3 * m - 3 * (m * n) + 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2472_247267


namespace NUMINAMATH_CALUDE_increasing_interval_of_f_l2472_247239

-- Define the function
def f (x : ℝ) : ℝ := 3*x - x^3

-- State the theorem
theorem increasing_interval_of_f :
  ∀ x : ℝ, (∀ y : ℝ, x < y → f x < f y) ↔ x ∈ Set.Ioo (-1 : ℝ) 1 := by sorry

end NUMINAMATH_CALUDE_increasing_interval_of_f_l2472_247239


namespace NUMINAMATH_CALUDE_completing_square_result_l2472_247216

theorem completing_square_result (x : ℝ) : 
  (x^2 - 4*x + 2 = 0) → ((x - 2)^2 = 2) :=
by
  sorry

end NUMINAMATH_CALUDE_completing_square_result_l2472_247216


namespace NUMINAMATH_CALUDE_greatest_integer_value_l2472_247252

theorem greatest_integer_value (x : ℤ) : 
  (∀ y : ℤ, y > x → ¬(∃ z : ℤ, (y^2 + 5*y + 6) / (y - 2) = z)) →
  (∃ z : ℤ, (x^2 + 5*x + 6) / (x - 2) = z) →
  x = 22 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_value_l2472_247252

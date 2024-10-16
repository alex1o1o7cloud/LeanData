import Mathlib

namespace NUMINAMATH_CALUDE_pie_slices_l557_55724

/-- Proves that if 3/4 of a pie is given away and 2 slices are left, then the pie was sliced into 8 pieces. -/
theorem pie_slices (total_slices : ℕ) : 
  (3 : ℚ) / 4 * total_slices + 2 = total_slices → total_slices = 8 := by
  sorry

#check pie_slices

end NUMINAMATH_CALUDE_pie_slices_l557_55724


namespace NUMINAMATH_CALUDE_expected_value_of_sum_is_seven_l557_55757

def marbles : Finset Nat := {1, 2, 3, 4, 5, 6}

def pairs : Finset (Nat × Nat) :=
  (marbles.product marbles).filter (fun (a, b) => a < b)

def sum_pair (p : Nat × Nat) : Nat := p.1 + p.2

theorem expected_value_of_sum_is_seven :
  (pairs.sum sum_pair) / pairs.card = 7 := by sorry

end NUMINAMATH_CALUDE_expected_value_of_sum_is_seven_l557_55757


namespace NUMINAMATH_CALUDE_lagrange_mean_value_theorem_l557_55741

theorem lagrange_mean_value_theorem {f : ℝ → ℝ} {a b : ℝ} (hf : Differentiable ℝ f) (hab : a < b) :
  ∃ x₀ ∈ Set.Ioo a b, deriv f x₀ = (f a - f b) / (a - b) := by
  sorry

end NUMINAMATH_CALUDE_lagrange_mean_value_theorem_l557_55741


namespace NUMINAMATH_CALUDE_f_positive_iff_l557_55719

/-- The function f(x) = 2x + 5 -/
def f (x : ℝ) : ℝ := 2 * x + 5

/-- Theorem: f(x) > 0 if and only if x > -5/2 -/
theorem f_positive_iff (x : ℝ) : f x > 0 ↔ x > -5/2 := by
  sorry

end NUMINAMATH_CALUDE_f_positive_iff_l557_55719


namespace NUMINAMATH_CALUDE_root_in_interval_l557_55788

def f (x : ℝ) := x^3 + x^2 - 2*x - 2

theorem root_in_interval :
  f 1 = -2 →
  f 1.5 = 0.65 →
  f 1.25 = -0.984 →
  f 1.375 = -0.260 →
  f 1.4375 = 0.162 →
  f 1.40625 = -0.054 →
  ∃ x, x > 1.3 ∧ x < 1.5 ∧ f x = 0 :=
by sorry

end NUMINAMATH_CALUDE_root_in_interval_l557_55788


namespace NUMINAMATH_CALUDE_circle_graph_proportions_l557_55783

theorem circle_graph_proportions :
  ∀ (total : ℝ) (blue : ℝ),
    blue > 0 →
    total = blue + 3 * blue + 0.5 * blue →
    (3 * blue / total = 2 / 3) ∧
    (blue / total = 1 / 4.5) ∧
    (0.5 * blue / total = 1 / 9) := by
  sorry

end NUMINAMATH_CALUDE_circle_graph_proportions_l557_55783


namespace NUMINAMATH_CALUDE_race_result_l557_55751

-- Define the runners
structure Runner :=
  (name : String)
  (speed : ℝ)

-- Define the race
def race (mengmeng lujia wangyu : Runner) : Prop :=
  -- All runners have constant, positive speed
  mengmeng.speed > 0 ∧ lujia.speed > 0 ∧ wangyu.speed > 0 ∧
  -- When Mengmeng finishes, Lujia is 10m away and Wang Yu is 20m away
  10 / lujia.speed = 20 / wangyu.speed

-- Theorem statement
theorem race_result (mengmeng lujia wangyu : Runner) 
  (h : race mengmeng lujia wangyu) : 
  ∃ (x : ℝ), x = 100 / 9 ∧ 
  (100 - x) / wangyu.speed = 100 / lujia.speed :=
sorry

end NUMINAMATH_CALUDE_race_result_l557_55751


namespace NUMINAMATH_CALUDE_curve_constants_sum_l557_55791

/-- Given a curve y = ax² + b/x passing through the point (2, -5) with a tangent at this point
    parallel to the line 7x + 2y + 3 = 0, prove that a + b = -43/20 -/
theorem curve_constants_sum (a b : ℝ) : 
  (4 * a + b / 2 = -5) →  -- Curve passes through (2, -5)
  (4 * a - b / 4 = -7/2) →  -- Tangent at (2, -5) is parallel to 7x + 2y + 3 = 0
  a + b = -43/20 := by
  sorry

end NUMINAMATH_CALUDE_curve_constants_sum_l557_55791


namespace NUMINAMATH_CALUDE_problem_solution_l557_55775

/-- The function f defined on real numbers --/
noncomputable def f (b : ℝ) (x : ℝ) : ℝ := (-2^x + b) / (2^(x+1) + 2)

/-- f is an odd function --/
axiom f_odd (b : ℝ) : ∀ x, f b (-x) = -(f b x)

theorem problem_solution :
  ∃ b : ℝ,
  (∀ x, f b (-x) = -(f b x)) ∧  -- f is odd
  (b = 1) ∧  -- part 1
  (∀ x y, x < y → f b x > f b y) ∧  -- part 2: f is decreasing
  (∀ k, (∀ t, f b (t^2 - 2*t) + f b (2*t^2 - k) < 0) → k < -1/3)  -- part 3
  := by sorry

end NUMINAMATH_CALUDE_problem_solution_l557_55775


namespace NUMINAMATH_CALUDE_quotient_of_powers_l557_55743

theorem quotient_of_powers (a b c : ℕ) (ha : a = 50) (hb : b = 25) (hc : c = 100) :
  (a ^ 50) / (b ^ 25) = c ^ 25 := by
  sorry

end NUMINAMATH_CALUDE_quotient_of_powers_l557_55743


namespace NUMINAMATH_CALUDE_log_sawing_time_l557_55712

theorem log_sawing_time (log_length : ℕ) (section_length : ℕ) (saw_time : ℕ) 
  (h1 : log_length = 10)
  (h2 : section_length = 1)
  (h3 : saw_time = 3) :
  (log_length - 1) * saw_time = 27 :=
by sorry

end NUMINAMATH_CALUDE_log_sawing_time_l557_55712


namespace NUMINAMATH_CALUDE_tan_3_degrees_decomposition_l557_55722

theorem tan_3_degrees_decomposition :
  ∃ (p q r s : ℕ+),
    (Real.tan (3 * Real.pi / 180) = Real.sqrt p - Real.sqrt q + Real.sqrt r - s) ∧
    (p ≥ q) ∧ (q ≥ r) ∧ (r ≥ s) →
    p + q + r + s = 20 := by
  sorry

end NUMINAMATH_CALUDE_tan_3_degrees_decomposition_l557_55722


namespace NUMINAMATH_CALUDE_tournament_games_l557_55705

/-- The number of games needed in a single-elimination tournament -/
def games_in_tournament (n : ℕ) : ℕ := n - 1

/-- Theorem: A single-elimination tournament with 32 teams requires 31 games -/
theorem tournament_games : games_in_tournament 32 = 31 := by
  sorry

end NUMINAMATH_CALUDE_tournament_games_l557_55705


namespace NUMINAMATH_CALUDE_solve_system_l557_55700

theorem solve_system (x y : ℤ) (h1 : x + y = 300) (h2 : x - y = 200) : y = 50 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l557_55700


namespace NUMINAMATH_CALUDE_arithmetic_sequence_m_value_l557_55708

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_m_value
  (a : ℕ → ℝ)
  (m : ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_roots : (a 2)^2 + m * (a 2) - 8 = 0 ∧ (a 8)^2 + m * (a 8) - 8 = 0)
  (h_sum : a 4 + a 6 = (a 5)^2 + 1) :
  m = -2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_m_value_l557_55708


namespace NUMINAMATH_CALUDE_third_cat_weight_calculation_l557_55740

/-- The weight of the third cat given the weights of the other cats and their average -/
def third_cat_weight (cat1 : ℝ) (cat2 : ℝ) (cat4 : ℝ) (avg_weight : ℝ) : ℝ :=
  4 * avg_weight - (cat1 + cat2 + cat4)

theorem third_cat_weight_calculation :
  third_cat_weight 12 12 9.3 12 = 14.7 := by
  sorry

end NUMINAMATH_CALUDE_third_cat_weight_calculation_l557_55740


namespace NUMINAMATH_CALUDE_inequality_iff_solution_set_l557_55725

-- Define the inequality function
def inequality (x : ℝ) : Prop :=
  Real.log (1 + 27 * x^5) / Real.log (1 + x^2) +
  Real.log (1 + x^2) / Real.log (1 - 2*x^2 + 27*x^4) ≤
  1 + Real.log (1 + 27*x^5) / Real.log (1 - 2*x^2 + 27*x^4)

-- Define the solution set
def solution_set (x : ℝ) : Prop :=
  (x > -Real.rpow 27 (-1/5) ∧ x ≤ -1/3) ∨
  (x > -Real.sqrt (2/27) ∧ x < 0) ∨
  (x > 0 ∧ x < Real.sqrt (2/27)) ∨
  x = 1/3

-- State the theorem
theorem inequality_iff_solution_set :
  ∀ x : ℝ, inequality x ↔ solution_set x :=
sorry

end NUMINAMATH_CALUDE_inequality_iff_solution_set_l557_55725


namespace NUMINAMATH_CALUDE_time_difference_1200_miles_l557_55770

/-- Calculates the time difference for a 1200-mile trip between two given speeds -/
theorem time_difference_1200_miles (speed1 speed2 : ℝ) (h1 : speed1 > 0) (h2 : speed2 > 0) :
  (1200 / speed1 - 1200 / speed2) = 4 ↔ speed1 = 60 ∧ speed2 = 50 := by sorry

end NUMINAMATH_CALUDE_time_difference_1200_miles_l557_55770


namespace NUMINAMATH_CALUDE_find_M_l557_55703

theorem find_M : ∃ M : ℕ+, (36 ^ 2 : ℕ) * (75 ^ 2) = (30 ^ 2) * (M.val ^ 2) ∧ M.val = 90 := by
  sorry

end NUMINAMATH_CALUDE_find_M_l557_55703


namespace NUMINAMATH_CALUDE_jake_weight_proof_l557_55769

/-- Jake's present weight in pounds -/
def jake_weight : ℝ := 108

/-- Jake's sister's weight in pounds -/
def sister_weight : ℝ := 48

/-- The combined weight of Jake and his sister in pounds -/
def combined_weight : ℝ := 156

theorem jake_weight_proof :
  (jake_weight - 12 = 2 * sister_weight) ∧
  (jake_weight + sister_weight = combined_weight) →
  jake_weight = 108 :=
by sorry

end NUMINAMATH_CALUDE_jake_weight_proof_l557_55769


namespace NUMINAMATH_CALUDE_chess_tournament_players_l557_55785

/-- Represents a chess tournament with the given conditions -/
structure ChessTournament where
  n : ℕ  -- number of players
  winner_wins : ℕ  -- number of wins by the winner
  winner_draws : ℕ  -- number of draws by the winner

/-- The conditions of the tournament are satisfied -/
def valid_tournament (t : ChessTournament) : Prop :=
  t.n > 1 ∧  -- more than one player
  t.winner_wins = t.winner_draws ∧  -- winner won half and drew half
  t.winner_wins + t.winner_draws = t.n - 1 ∧  -- winner played against every other player once
  (t.winner_wins : ℚ) + (t.winner_draws : ℚ) / 2 = (t.n * (t.n - 1) : ℚ) / 20  -- winner's points are 9 times less than others'

theorem chess_tournament_players (t : ChessTournament) :
  valid_tournament t → t.n = 15 := by
  sorry

#check chess_tournament_players

end NUMINAMATH_CALUDE_chess_tournament_players_l557_55785


namespace NUMINAMATH_CALUDE_asterisk_replacement_l557_55764

theorem asterisk_replacement : (42 / 21) * (42 / 84) = 1 := by
  sorry

end NUMINAMATH_CALUDE_asterisk_replacement_l557_55764


namespace NUMINAMATH_CALUDE_dan_minimum_speed_l557_55738

/-- Proves the minimum speed Dan must exceed to arrive before Cara -/
theorem dan_minimum_speed (distance : ℝ) (cara_speed : ℝ) (dan_delay : ℝ) :
  distance = 180 →
  cara_speed = 30 →
  dan_delay = 1 →
  ∃ (min_speed : ℝ), min_speed > 36 ∧
    ∀ (dan_speed : ℝ), dan_speed > min_speed →
      distance / dan_speed < distance / cara_speed - dan_delay := by
  sorry

#check dan_minimum_speed

end NUMINAMATH_CALUDE_dan_minimum_speed_l557_55738


namespace NUMINAMATH_CALUDE_inverse_proportion_order_l557_55731

theorem inverse_proportion_order (y₁ y₂ y₃ : ℝ) : 
  y₁ = -4 / 1 → y₂ = -4 / 2 → y₃ = -4 / (-3) → y₁ < y₂ ∧ y₂ < y₃ := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_order_l557_55731


namespace NUMINAMATH_CALUDE_rower_downstream_speed_l557_55730

/-- Calculates the downstream speed of a rower given their upstream and still water speeds -/
def downstreamSpeed (upstreamSpeed stillWaterSpeed : ℝ) : ℝ :=
  2 * stillWaterSpeed - upstreamSpeed

theorem rower_downstream_speed
  (upstreamSpeed : ℝ)
  (stillWaterSpeed : ℝ)
  (h1 : upstreamSpeed = 25)
  (h2 : stillWaterSpeed = 33) :
  downstreamSpeed upstreamSpeed stillWaterSpeed = 41 := by
  sorry

#eval downstreamSpeed 25 33

end NUMINAMATH_CALUDE_rower_downstream_speed_l557_55730


namespace NUMINAMATH_CALUDE_employee_age_when_hired_l557_55713

/-- Represents the retirement eligibility rule where age plus years of employment must equal 70 -/
def retirement_rule (age : ℕ) (years_employed : ℕ) : Prop :=
  age + years_employed = 70

/-- Represents the fact that the employee worked for 19 years before retirement eligibility -/
def years_worked : ℕ := 19

/-- Proves that the employee's age when hired was 51 -/
theorem employee_age_when_hired :
  ∃ (age_when_hired : ℕ),
    retirement_rule (age_when_hired + years_worked) years_worked ∧
    age_when_hired = 51 := by
  sorry

end NUMINAMATH_CALUDE_employee_age_when_hired_l557_55713


namespace NUMINAMATH_CALUDE_train_length_problem_l557_55758

theorem train_length_problem (v_fast v_slow : ℝ) (t : ℝ) (h1 : v_fast = 46) (h2 : v_slow = 36) (h3 : t = 144) :
  let rel_speed := (v_fast - v_slow) * (5 / 18)
  let train_length := rel_speed * t / 2
  train_length = 200 := by
sorry

end NUMINAMATH_CALUDE_train_length_problem_l557_55758


namespace NUMINAMATH_CALUDE_compound_interest_calculation_l557_55714

/-- Given a principal amount that yields $50 in simple interest over 2 years at 5% per annum,
    the compound interest for the same principal, rate, and time is $51.25. -/
theorem compound_interest_calculation (P : ℝ) : 
  (P * 0.05 * 2 = 50) →  -- Simple interest condition
  (P * (1 + 0.05)^2 - P = 51.25) :=  -- Compound interest calculation
by
  sorry


end NUMINAMATH_CALUDE_compound_interest_calculation_l557_55714


namespace NUMINAMATH_CALUDE_can_divide_into_12_l557_55707

-- Define a circular cake
structure CircularCake where
  radius : ℝ
  center : ℝ × ℝ

-- Define a function to represent dividing a cake into equal pieces
def divide_cake (cake : CircularCake) (n : ℕ) : Set (ℝ × ℝ) :=
  sorry

-- Define our given cakes
def cake1 : CircularCake := sorry
def cake2 : CircularCake := sorry
def cake3 : CircularCake := sorry

-- State that cake1 is divided into 3 pieces
axiom cake1_division : divide_cake cake1 3

-- State that cake2 is divided into 4 pieces
axiom cake2_division : divide_cake cake2 4

-- State that all cakes have the same radius
axiom same_radius : cake1.radius = cake2.radius ∧ cake2.radius = cake3.radius

-- State that we know the center of cake3
axiom known_center3 : cake3.center = (0, 0)

-- Theorem to prove
theorem can_divide_into_12 : 
  ∃ (division : Set (ℝ × ℝ)), division = divide_cake cake3 12 :=
sorry

end NUMINAMATH_CALUDE_can_divide_into_12_l557_55707


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_l557_55749

theorem quadratic_distinct_roots (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 9 = 0 ∧ y^2 + m*y + 9 = 0) ↔ 
  (m < -6 ∨ m > 6) :=
sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_l557_55749


namespace NUMINAMATH_CALUDE_exchange_point_configuration_exists_multiple_configurations_exist_l557_55716

/-- A planar graph representing a city map -/
structure CityMap where
  -- The number of edges (roads) in the map
  num_edges : ℕ
  -- The initial number of vertices (exchange points)
  initial_vertices : ℕ
  -- The number of faces in the planar graph (city parts)
  num_faces : ℕ
  -- Euler's formula for planar graphs
  euler_formula : num_faces = num_edges - initial_vertices + 2

/-- The configuration of exchange points in the city -/
structure ExchangePointConfig where
  -- The total number of exchange points after adding new ones
  total_points : ℕ
  -- The number of points in each face
  points_per_face : ℕ
  -- Condition that each face has exactly two points
  two_points_per_face : points_per_face = 2
  -- The total number of points is consistent with the number of faces
  total_points_condition : total_points = num_faces * points_per_face

/-- Theorem stating that it's possible to add three exchange points to satisfy the conditions -/
theorem exchange_point_configuration_exists (m : CityMap) (h : m.initial_vertices = 1) :
  ∃ (config : ExchangePointConfig), config.total_points = m.initial_vertices + 3 :=
sorry

/-- Theorem stating that multiple valid configurations exist -/
theorem multiple_configurations_exist (m : CityMap) (h : m.initial_vertices = 1) :
  ∃ (config1 config2 config3 config4 : ExchangePointConfig),
    config1 ≠ config2 ∧ config1 ≠ config3 ∧ config1 ≠ config4 ∧
    config2 ≠ config3 ∧ config2 ≠ config4 ∧ config3 ≠ config4 ∧
    (∀ c ∈ [config1, config2, config3, config4], c.total_points = m.initial_vertices + 3) :=
sorry

end NUMINAMATH_CALUDE_exchange_point_configuration_exists_multiple_configurations_exist_l557_55716


namespace NUMINAMATH_CALUDE_farm_animal_count_l557_55778

/-- Represents the count of animals on a farm -/
structure FarmCount where
  chickens : ℕ
  ducks : ℕ
  geese : ℕ
  quails : ℕ
  turkeys : ℕ
  cow_sheds : ℕ
  cows_per_shed : ℕ
  pigs : ℕ

/-- Calculates the total number of animals on the farm -/
def total_animals (farm : FarmCount) : ℕ :=
  farm.chickens + farm.ducks + farm.geese + farm.quails + farm.turkeys +
  (farm.cow_sheds * farm.cows_per_shed) + farm.pigs

/-- Theorem stating that the total number of animals on the given farm is 219 -/
theorem farm_animal_count :
  let farm := FarmCount.mk 60 40 20 50 10 3 8 15
  total_animals farm = 219 := by
  sorry

#eval total_animals (FarmCount.mk 60 40 20 50 10 3 8 15)

end NUMINAMATH_CALUDE_farm_animal_count_l557_55778


namespace NUMINAMATH_CALUDE_passenger_trips_scientific_notation_l557_55787

/-- The number of operating passenger trips in millions -/
def passenger_trips : ℝ := 56.99

/-- The scientific notation representation of the passenger trips -/
def scientific_notation : ℝ := 5.699 * (10^7)

/-- Theorem stating that the number of passenger trips in millions 
    is equal to its scientific notation representation -/
theorem passenger_trips_scientific_notation : 
  passenger_trips * 10^6 = scientific_notation := by sorry

end NUMINAMATH_CALUDE_passenger_trips_scientific_notation_l557_55787


namespace NUMINAMATH_CALUDE_die_roll_probabilities_l557_55711

theorem die_roll_probabilities :
  let n : ℕ := 7  -- number of rolls
  let p : ℝ := 1/6  -- probability of rolling a 4
  let q : ℝ := 1 - p  -- probability of not rolling a 4

  -- (a) Probability of rolling at least one 4 in 7 rolls
  let prob_at_least_one : ℝ := 1 - q^n

  -- (b) Probability of rolling exactly one 4 in 7 rolls
  let prob_exactly_one : ℝ := n * p * q^(n-1)

  -- (c) Probability of rolling at most one 4 in 7 rolls
  let prob_at_most_one : ℝ := q^n + n * p * q^(n-1)

  -- Prove that the calculated probabilities are correct
  (prob_at_least_one = 1 - (5/6)^7) ∧
  (prob_exactly_one = 7 * (1/6) * (5/6)^6) ∧
  (prob_at_most_one = (5/6)^7 + 7 * (1/6) * (5/6)^6) :=
by
  sorry

end NUMINAMATH_CALUDE_die_roll_probabilities_l557_55711


namespace NUMINAMATH_CALUDE_cube_volume_from_diagonal_l557_55734

/-- The volume of a cube with a given space diagonal -/
theorem cube_volume_from_diagonal (d : ℝ) (h : d = 5 * Real.sqrt 3) : 
  let s := d / Real.sqrt 3
  s ^ 3 = 125 := by sorry

end NUMINAMATH_CALUDE_cube_volume_from_diagonal_l557_55734


namespace NUMINAMATH_CALUDE_sixth_power_of_z_l557_55768

theorem sixth_power_of_z (z : ℂ) : z = (Real.sqrt 3 - Complex.I) / 2 → z^6 = -1 := by
  sorry

end NUMINAMATH_CALUDE_sixth_power_of_z_l557_55768


namespace NUMINAMATH_CALUDE_last_four_digits_of_2_to_15000_l557_55752

theorem last_four_digits_of_2_to_15000 (h : 2^500 ≡ 1 [ZMOD 1250]) :
  2^15000 ≡ 1 [ZMOD 1250] := by
  sorry

end NUMINAMATH_CALUDE_last_four_digits_of_2_to_15000_l557_55752


namespace NUMINAMATH_CALUDE_total_value_is_71_rupees_l557_55781

/-- Represents the value of a coin in paise -/
inductive CoinValue
  | paise20 : CoinValue
  | paise25 : CoinValue

/-- Calculates the total value in rupees given the number of coins and their values -/
def totalValueInRupees (totalCoins : ℕ) (coins20paise : ℕ) : ℚ :=
  let coins25paise := totalCoins - coins20paise
  let value20paise := 20 * coins20paise
  let value25paise := 25 * coins25paise
  (value20paise + value25paise : ℚ) / 100

/-- Theorem stating that the total value of the given coins is 71 rupees -/
theorem total_value_is_71_rupees :
  totalValueInRupees 334 250 = 71 := by
  sorry


end NUMINAMATH_CALUDE_total_value_is_71_rupees_l557_55781


namespace NUMINAMATH_CALUDE_ellipse_properties_and_max_area_l557_55709

noncomputable def ellipse_equation (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

noncomputable def eccentricity (c a : ℝ) : ℝ :=
  c / a

noncomputable def distance (x y : ℝ) : ℝ :=
  Real.sqrt (x^2 + y^2)

theorem ellipse_properties_and_max_area 
  (a b c : ℝ) 
  (h1 : a > b) (h2 : b > 0) (h3 : c > 0)
  (h4 : eccentricity c a = Real.sqrt 3 / 2)
  (h5 : ellipse_equation a b c (b^2/a))
  (h6 : distance c (b^2/a) = Real.sqrt 13 / 2) :
  (∃ (x y : ℝ), ellipse_equation 2 1 x y) ∧
  (∃ (S : ℝ), S = 4 ∧ 
    ∀ (m : ℝ), abs m < Real.sqrt 2 → 
      2 * Real.sqrt (m^2 * (8 - 4 * m^2)) ≤ S) := by
sorry

end NUMINAMATH_CALUDE_ellipse_properties_and_max_area_l557_55709


namespace NUMINAMATH_CALUDE_zero_clever_numbers_l557_55771

def is_zero_clever (n : ℕ) : Prop :=
  ∃ (a b c : ℕ),
    n = a * 1000 + b * 10 + c ∧
    n = (a * 100 + b * 10 + c) * 9 ∧
    a ≥ 1 ∧ a ≤ 9 ∧ b ≥ 0 ∧ b ≤ 9 ∧ c ≥ 0 ∧ c ≤ 9

theorem zero_clever_numbers :
  {n : ℕ | is_zero_clever n} = {2025, 4050, 6075} :=
sorry

end NUMINAMATH_CALUDE_zero_clever_numbers_l557_55771


namespace NUMINAMATH_CALUDE_chocolate_distribution_l557_55729

def is_valid_distribution (n m : ℕ) : Prop :=
  n ≤ m ∨ (m < n ∧ m ∣ (n - m))

def possible_n_for_m_9 : Set ℕ :=
  {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 18}

theorem chocolate_distribution (n m : ℕ) :
  (m = 9 → n ∈ possible_n_for_m_9) ↔ is_valid_distribution n m :=
sorry

end NUMINAMATH_CALUDE_chocolate_distribution_l557_55729


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l557_55736

theorem diophantine_equation_solution : 
  {(x, y) : ℕ+ × ℕ+ | x.val - y.val - (x.val / y.val) - (x.val^3 / y.val^3) + (x.val^4 / y.val^4) = 2017} = 
  {(⟨2949, by norm_num⟩, ⟨983, by norm_num⟩), (⟨4022, by norm_num⟩, ⟨2011, by norm_num⟩)} :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l557_55736


namespace NUMINAMATH_CALUDE_ellipse_condition_l557_55795

/-- The equation of the graph -/
def equation (x y k : ℝ) : Prop := x^2 + 9*y^2 - 6*x + 27*y = k

/-- The graph is a non-degenerate ellipse -/
def is_non_degenerate_ellipse (k : ℝ) : Prop := k > -29.25

theorem ellipse_condition (k : ℝ) : 
  (∃ x y, equation x y k) ↔ is_non_degenerate_ellipse k :=
sorry

end NUMINAMATH_CALUDE_ellipse_condition_l557_55795


namespace NUMINAMATH_CALUDE_apple_sale_profit_percentage_l557_55772

/-- Represents the shopkeeper's apple selling scenario -/
structure AppleSale where
  total_apples : ℝ
  sell_percent_1 : ℝ
  profit_percent_1 : ℝ
  sell_percent_2 : ℝ
  profit_percent_2 : ℝ
  sell_percent_3 : ℝ
  profit_percent_3 : ℝ
  unsold_percent : ℝ
  additional_expenses : ℝ

/-- Calculates the effective profit percentage for the given apple sale scenario -/
def effectiveProfitPercentage (sale : AppleSale) : ℝ :=
  sorry

/-- Theorem stating the effective profit percentage for the given scenario -/
theorem apple_sale_profit_percentage :
  let sale : AppleSale := {
    total_apples := 120,
    sell_percent_1 := 0.4,
    profit_percent_1 := 0.25,
    sell_percent_2 := 0.3,
    profit_percent_2 := 0.35,
    sell_percent_3 := 0.2,
    profit_percent_3 := 0.2,
    unsold_percent := 0.1,
    additional_expenses := 20
  }
  ∃ (ε : ℝ), ε > 0 ∧ abs (effectiveProfitPercentage sale + 2.407) < ε :=
sorry

end NUMINAMATH_CALUDE_apple_sale_profit_percentage_l557_55772


namespace NUMINAMATH_CALUDE_gcd_factorial_problem_l557_55792

theorem gcd_factorial_problem : Nat.gcd (Nat.factorial 7) ((Nat.factorial 10) / (Nat.factorial 4)) = 2520 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_problem_l557_55792


namespace NUMINAMATH_CALUDE_fourth_intersection_point_l557_55760

/-- The curve xy = 2 intersects a circle at four points. Three of these points are given. -/
def intersection_points : Finset (ℚ × ℚ) :=
  {(4, 1/2), (-2, -1), (2/5, 5)}

/-- The fourth intersection point -/
def fourth_point : ℚ × ℚ := (-16/5, -5/8)

/-- All points satisfy the equation xy = 2 -/
def on_curve (p : ℚ × ℚ) : Prop :=
  p.1 * p.2 = 2

theorem fourth_intersection_point :
  (∀ p ∈ intersection_points, on_curve p) →
  on_curve fourth_point →
  ∃ (a b r : ℚ),
    (∀ p ∈ intersection_points, (p.1 - a)^2 + (p.2 - b)^2 = r^2) ∧
    (fourth_point.1 - a)^2 + (fourth_point.2 - b)^2 = r^2 :=
by sorry

end NUMINAMATH_CALUDE_fourth_intersection_point_l557_55760


namespace NUMINAMATH_CALUDE_assignment_validity_l557_55773

-- Define what constitutes a valid assignment statement
def is_valid_assignment (stmt : String) : Prop :=
  ∃ (var : String) (expr : String),
    stmt = var ++ " = " ++ expr ∧
    var.length > 0 ∧
    expr.length > 0 ∧
    var.all Char.isAlpha

theorem assignment_validity :
  is_valid_assignment "x = x + 1" ∧
  ¬is_valid_assignment "b =" ∧
  ¬is_valid_assignment "x = y = 10" ∧
  ¬is_valid_assignment "x + y = 10" :=
by sorry


end NUMINAMATH_CALUDE_assignment_validity_l557_55773


namespace NUMINAMATH_CALUDE_space_line_relations_l557_55710

-- Define a type for lines in space
variable (Line : Type)

-- Define the parallel relation
variable (parallel : Line → Line → Prop)

-- Define the perpendicular relation
variable (perpendicular : Line → Line → Prop)

-- Define the intersects relation
variable (intersects : Line → Line → Prop)

-- Define a type for planes
variable (Plane : Type)

-- Define a relation for a line being in a plane
variable (line_in_plane : Line → Plane → Prop)

-- Define three non-intersecting lines
variable (a b c : Line)

-- Define two planes
variable (α β : Plane)

-- State that the lines are non-intersecting
variable (h_non_intersect : ¬(intersects a b ∨ intersects b c ∨ intersects a c))

theorem space_line_relations :
  (∀ x y z, parallel x y → parallel y z → parallel x z) ∧
  ¬(∀ x y z, perpendicular x y → perpendicular y z → parallel x z) ∧
  ¬(∀ x y z, intersects x y → intersects y z → intersects x z) ∧
  ¬(∀ x y p q, line_in_plane x p → line_in_plane y q → x ≠ y → ¬(parallel x y ∨ intersects x y)) :=
by sorry

end NUMINAMATH_CALUDE_space_line_relations_l557_55710


namespace NUMINAMATH_CALUDE_derivative_of_even_is_odd_l557_55790

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem derivative_of_even_is_odd
  (f : ℝ → ℝ) (hf : IsEven f) (g : ℝ → ℝ) (hg : ∀ x, HasDerivAt f (g x) x) :
  ∀ x, g (-x) = -g x :=
sorry

end NUMINAMATH_CALUDE_derivative_of_even_is_odd_l557_55790


namespace NUMINAMATH_CALUDE_least_lcm_a_c_l557_55718

theorem least_lcm_a_c (a b c : ℕ) (h1 : Nat.lcm a b = 12) (h2 : Nat.lcm b c = 15) :
  ∃ (a' c' : ℕ), Nat.lcm a' c' = 20 ∧ (∀ (x y : ℕ), Nat.lcm x b = 12 → Nat.lcm b y = 15 → Nat.lcm a' c' ≤ Nat.lcm x y) :=
sorry

end NUMINAMATH_CALUDE_least_lcm_a_c_l557_55718


namespace NUMINAMATH_CALUDE_midpoint_between_fractions_l557_55735

theorem midpoint_between_fractions :
  (1 / 12 + 1 / 15) / 2 = 3 / 40 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_between_fractions_l557_55735


namespace NUMINAMATH_CALUDE_unique_last_digit_for_divisibility_by_6_l557_55748

def is_divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

def last_digit (n : ℕ) : ℕ := n % 10

def replace_last_digit (n : ℕ) (d : ℕ) : ℕ := (n / 10) * 10 + d

theorem unique_last_digit_for_divisibility_by_6 :
  ∀ d : ℕ, d < 10 →
    (is_divisible_by_6 (replace_last_digit 314270 d) ↔ d = last_digit 314274) :=
sorry

end NUMINAMATH_CALUDE_unique_last_digit_for_divisibility_by_6_l557_55748


namespace NUMINAMATH_CALUDE_sqrt_132_plus_46_sqrt_11_l557_55720

theorem sqrt_132_plus_46_sqrt_11 :
  ∃ (a b c : ℤ), 
    (132 + 46 * Real.sqrt 11 : ℝ).sqrt = a + b * Real.sqrt c ∧
    ¬ ∃ (d : ℤ), c = d * d ∧
    ∃ (e f : ℤ), c = e * f ∧ (∀ (g : ℤ), g * g ∣ e → g = 1 ∨ g = -1) ∧
                             (∀ (h : ℤ), h * h ∣ f → h = 1 ∨ h = -1) :=
sorry

end NUMINAMATH_CALUDE_sqrt_132_plus_46_sqrt_11_l557_55720


namespace NUMINAMATH_CALUDE_reciprocal_location_l557_55776

theorem reciprocal_location (a b : ℝ) (h1 : a < 0) (h2 : b < 0) (h3 : a^2 + b^2 < 1) :
  let F := Complex.mk a b
  let recip := F⁻¹
  (Complex.re recip > 0) ∧ (Complex.im recip > 0) ∧ (Complex.abs recip > 1) := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_location_l557_55776


namespace NUMINAMATH_CALUDE_pascal_39th_number_40th_row_l557_55715

-- Define Pascal's triangle coefficient
def pascal (n k : ℕ) : ℕ := Nat.choose n k

-- Theorem statement
theorem pascal_39th_number_40th_row : pascal 40 38 = 780 := by
  sorry

end NUMINAMATH_CALUDE_pascal_39th_number_40th_row_l557_55715


namespace NUMINAMATH_CALUDE_perfect_square_existence_l557_55774

theorem perfect_square_existence (k : ℕ+) :
  ∃ (n m : ℕ+), n * 2^(k : ℕ) - 7 = m^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_existence_l557_55774


namespace NUMINAMATH_CALUDE_symmetric_point_proof_l557_55717

/-- A point in a 2D plane represented by its x and y coordinates. -/
structure Point where
  x : ℝ
  y : ℝ

/-- The origin point (0, 0) in a 2D plane. -/
def origin : Point := ⟨0, 0⟩

/-- Determines if two points are symmetric with respect to the origin. -/
def isSymmetricToOrigin (p1 p2 : Point) : Prop :=
  p1.x = -p2.x ∧ p1.y = -p2.y

/-- The given point (3, -1). -/
def givenPoint : Point := ⟨3, -1⟩

/-- The point to be proven symmetric to the given point. -/
def symmetricPoint : Point := ⟨-3, 1⟩

/-- Theorem stating that the symmetricPoint is symmetric to the givenPoint with respect to the origin. -/
theorem symmetric_point_proof : isSymmetricToOrigin givenPoint symmetricPoint := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_proof_l557_55717


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l557_55782

theorem quadratic_equation_solution : 
  ∃ x₁ x₂ : ℝ, (x₁ = -2 + Real.sqrt 2 ∧ x₂ = -2 - Real.sqrt 2) ∧ 
  (∀ x : ℝ, x^2 + 4*x + 2 = 0 ↔ (x = x₁ ∨ x = x₂)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l557_55782


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l557_55728

theorem least_addition_for_divisibility (n : ℕ) : 
  (∀ m : ℕ, m < n → ¬ (9 ∣ (4499 + m))) ∧ (9 ∣ (4499 + n)) → n = 1 := by
  sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l557_55728


namespace NUMINAMATH_CALUDE_combined_future_age_l557_55737

-- Define the current age of Hurley
def hurley_current_age : ℕ := 14

-- Define the age difference between Richard and Hurley
def age_difference : ℕ := 20

-- Define the number of years into the future
def years_future : ℕ := 40

-- Theorem to prove
theorem combined_future_age :
  (hurley_current_age + years_future) + (hurley_current_age + age_difference + years_future) = 128 := by
  sorry

end NUMINAMATH_CALUDE_combined_future_age_l557_55737


namespace NUMINAMATH_CALUDE_line_equations_l557_55704

/-- Given a line passing through (-b, c) that cuts a triangular region with area U from the second quadrant,
    this theorem states the equations of the inclined line and the horizontal line passing through its y-intercept. -/
theorem line_equations (b c U : ℝ) (h_b : b > 0) (h_c : c > 0) (h_U : U > 0) :
  ∃ (m k : ℝ),
    (∀ x y, y = m * x + k ↔ 2 * U * x - b^2 * y + 2 * U * b + c * b^2 = 0) ∧
    (k = 2 * U / b + c) := by
  sorry

end NUMINAMATH_CALUDE_line_equations_l557_55704


namespace NUMINAMATH_CALUDE_digit_relation_l557_55706

theorem digit_relation (a b : ℕ) : 
  a ≤ 9 → b ≤ 9 → (a : ℚ) / b = b + (a : ℚ) / 10 → a = 5 ∧ b = 2 := by
sorry

end NUMINAMATH_CALUDE_digit_relation_l557_55706


namespace NUMINAMATH_CALUDE_max_abs_z5_l557_55777

open Complex

theorem max_abs_z5 (z₁ z₂ z₃ z₄ z₅ : ℂ) 
  (h1 : abs z₁ ≤ 1) (h2 : abs z₂ ≤ 1)
  (h3 : abs (2 * z₃ - (z₁ + z₂)) ≤ abs (z₁ - z₂))
  (h4 : abs (2 * z₄ - (z₁ + z₂)) ≤ abs (z₁ - z₂))
  (h5 : abs (2 * z₅ - (z₃ + z₄)) ≤ abs (z₃ - z₄)) :
  abs z₅ ≤ Real.sqrt 3 ∧ ∃ z₁ z₂ z₃ z₄ z₅ : ℂ, abs z₅ = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_max_abs_z5_l557_55777


namespace NUMINAMATH_CALUDE_negation_of_existence_ln_positive_l557_55767

theorem negation_of_existence_ln_positive :
  (¬ ∃ x : ℝ, x > 0 ∧ Real.log x > 0) ↔ (∀ x : ℝ, x > 0 → Real.log x ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_ln_positive_l557_55767


namespace NUMINAMATH_CALUDE_incorrect_number_calculation_l557_55798

theorem incorrect_number_calculation (n : ℕ) (correct_num incorrect_num : ℝ) 
  (incorrect_avg correct_avg : ℝ) :
  n = 10 ∧ 
  correct_num = 75 ∧
  n * incorrect_avg = n * correct_avg - (correct_num - incorrect_num) →
  incorrect_num = 25 :=
by sorry

end NUMINAMATH_CALUDE_incorrect_number_calculation_l557_55798


namespace NUMINAMATH_CALUDE_jennie_rental_cost_l557_55793

/-- Calculates the rental cost for a given number of days --/
def rental_cost (daily_rate : ℕ) (weekly_rate : ℕ) (days : ℕ) : ℕ :=
  if days ≤ 7 then
    daily_rate * days
  else
    weekly_rate + daily_rate * (days - 7)

theorem jennie_rental_cost :
  let daily_rate := 30
  let weekly_rate := 190
  let rental_days := 11
  rental_cost daily_rate weekly_rate rental_days = 310 := by
  sorry

end NUMINAMATH_CALUDE_jennie_rental_cost_l557_55793


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l557_55721

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 9/y = 1) :
  ∀ a b : ℝ, a > 0 → b > 0 → 1/a + 9/b = 1 → x + y ≤ a + b :=
by sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l557_55721


namespace NUMINAMATH_CALUDE_three_fractions_product_one_l557_55726

def S : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

theorem three_fractions_product_one :
  ∃ (a b c d e f : ℕ),
    a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ e ∈ S ∧ f ∈ S ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
    d ≠ e ∧ d ≠ f ∧
    e ≠ f ∧
    Nat.gcd a b = 1 ∧ Nat.gcd c d = 1 ∧ Nat.gcd e f = 1 ∧
    (a * c * e : ℚ) / (b * d * f) = 1 := by
  sorry

end NUMINAMATH_CALUDE_three_fractions_product_one_l557_55726


namespace NUMINAMATH_CALUDE_inequality_solution_l557_55723

theorem inequality_solution (x : ℝ) : 
  (-1 ≤ (x^2 + 3*x - 1) / (4 - x^2) ∧ (x^2 + 3*x - 1) / (4 - x^2) < 1) ↔ 
  (x < -5/2 ∨ (-1 ≤ x ∧ x < 1)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l557_55723


namespace NUMINAMATH_CALUDE_pentagonal_pillar_faces_l557_55786

/-- Represents a pentagonal pillar -/
structure PentagonalPillar :=
  (rectangular_faces : Nat)
  (pentagonal_faces : Nat)

/-- The total number of faces of a pentagonal pillar -/
def total_faces (p : PentagonalPillar) : Nat :=
  p.rectangular_faces + p.pentagonal_faces

/-- Theorem stating that a pentagonal pillar has 7 faces -/
theorem pentagonal_pillar_faces :
  ∀ (p : PentagonalPillar),
  p.rectangular_faces = 5 ∧ p.pentagonal_faces = 2 →
  total_faces p = 7 := by
  sorry

#check pentagonal_pillar_faces

end NUMINAMATH_CALUDE_pentagonal_pillar_faces_l557_55786


namespace NUMINAMATH_CALUDE_f_properties_l557_55756

def f (x : ℝ) := 4 - x^2

theorem f_properties :
  (∀ x : ℝ, f (-x) = f x) ∧
  (∀ x₁ x₂ : ℝ, 0 ≤ x₁ → x₁ < x₂ → f x₁ > f x₂) ∧
  (∀ x : ℝ, f x ≥ 3*x ↔ -4 ≤ x ∧ x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l557_55756


namespace NUMINAMATH_CALUDE_complement_of_intersection_l557_55766

-- Define the universe set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define set A
def A : Set ℕ := {1, 2, 3}

-- Define set B
def B : Set ℕ := {2, 3, 4}

-- Theorem statement
theorem complement_of_intersection :
  (Aᶜ : Set ℕ) ∩ B = {1, 4, 5} := by sorry

end NUMINAMATH_CALUDE_complement_of_intersection_l557_55766


namespace NUMINAMATH_CALUDE_percentage_of_hindu_boys_l557_55750

theorem percentage_of_hindu_boys (total : ℕ) (muslim_percent : ℚ) (sikh_percent : ℚ) (other : ℕ) :
  total = 300 →
  muslim_percent = 44 / 100 →
  sikh_percent = 10 / 100 →
  other = 54 →
  (total - (muslim_percent * total + sikh_percent * total + other)) / total = 28 / 100 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_hindu_boys_l557_55750


namespace NUMINAMATH_CALUDE_minimum_score_for_average_l557_55763

def exam_scores : List ℕ := [92, 85, 89, 93]
def desired_average : ℕ := 90
def num_exams : ℕ := 5

theorem minimum_score_for_average (scores : List ℕ) (avg : ℕ) (n : ℕ) :
  scores.length + 1 = n →
  (scores.sum + (n * avg - scores.sum)) / n = avg →
  n * avg - scores.sum = 91 :=
by sorry

#check minimum_score_for_average exam_scores desired_average num_exams

end NUMINAMATH_CALUDE_minimum_score_for_average_l557_55763


namespace NUMINAMATH_CALUDE_smallest_n_for_probability_threshold_l557_55745

/-- The probability of drawing n-1 white marbles followed by a red marble -/
def P (n : ℕ) : ℚ := 1 / (n * (n^2 + 1))

/-- The number of boxes -/
def num_boxes : ℕ := 3000

theorem smallest_n_for_probability_threshold :
  ∀ n : ℕ, n > 0 → n < 15 → P n ≥ 1 / num_boxes ∧
  P 15 < 1 / num_boxes :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_probability_threshold_l557_55745


namespace NUMINAMATH_CALUDE_pizza_dough_milk_calculation_l557_55747

/-- Given a ratio of milk to flour for pizza dough, calculate the amount of milk needed for a given amount of flour -/
theorem pizza_dough_milk_calculation 
  (milk_per_portion : ℝ) 
  (flour_per_portion : ℝ) 
  (total_flour : ℝ) 
  (h1 : milk_per_portion = 50) 
  (h2 : flour_per_portion = 250) 
  (h3 : total_flour = 750) :
  (total_flour / flour_per_portion) * milk_per_portion = 150 :=
by sorry

end NUMINAMATH_CALUDE_pizza_dough_milk_calculation_l557_55747


namespace NUMINAMATH_CALUDE_product_of_largest_unachievable_scores_l557_55794

def score_system : List ℕ := [19, 9, 8]

def is_achievable (n : ℕ) : Prop :=
  ∃ a b c : ℕ, n = 19 * a + 9 * b + 8 * c

def largest_unachievable_scores : List ℕ :=
  [31, 39]

theorem product_of_largest_unachievable_scores :
  (List.prod largest_unachievable_scores) = 1209 ∧
  (∀ n ∈ largest_unachievable_scores, ¬ is_achievable n) ∧
  (∀ m : ℕ, m > (List.maximum largest_unachievable_scores).getD 0 → is_achievable m) :=
by sorry

end NUMINAMATH_CALUDE_product_of_largest_unachievable_scores_l557_55794


namespace NUMINAMATH_CALUDE_discount_theorem_l557_55761

/-- Calculates the final price and equivalent discount for a given original price and discounts --/
def discount_calculation (original_price : ℝ) (store_discount : ℝ) (vip_discount : ℝ) : ℝ × ℝ :=
  let final_price := original_price * (1 - store_discount) * (1 - vip_discount)
  let equivalent_discount := 1 - (1 - store_discount) * (1 - vip_discount)
  (final_price, equivalent_discount * 100)

theorem discount_theorem :
  discount_calculation 1500 0.8 0.05 = (1140, 76) := by
  sorry

end NUMINAMATH_CALUDE_discount_theorem_l557_55761


namespace NUMINAMATH_CALUDE_total_cookies_count_l557_55744

/-- Represents a pack of cookies -/
structure CookiePack where
  name : String
  cookies : Nat

/-- Represents a person's cookie purchase -/
structure Purchase where
  packs : List (CookiePack × Nat)

def packA : CookiePack := ⟨"A", 15⟩
def packB : CookiePack := ⟨"B", 30⟩
def packC : CookiePack := ⟨"C", 45⟩
def packD : CookiePack := ⟨"D", 60⟩

def paulPurchase : Purchase := ⟨[(packB, 2), (packA, 1)]⟩
def paulaPurchase : Purchase := ⟨[(packA, 1), (packC, 1)]⟩

def countCookies (purchase : Purchase) : Nat :=
  purchase.packs.foldl (fun acc (pack, quantity) => acc + pack.cookies * quantity) 0

theorem total_cookies_count :
  countCookies paulPurchase + countCookies paulaPurchase = 135 := by
  sorry

#eval countCookies paulPurchase + countCookies paulaPurchase

end NUMINAMATH_CALUDE_total_cookies_count_l557_55744


namespace NUMINAMATH_CALUDE_expansion_properties_l557_55755

theorem expansion_properties :
  let f := fun x => (1 - 2*x)^6
  ∃ (c : ℚ) (p : Polynomial ℚ), 
    (f = fun x => p.eval x) ∧ 
    (p.coeff 2 = 60) ∧ 
    (p.eval 1 = 1) := by
  sorry

end NUMINAMATH_CALUDE_expansion_properties_l557_55755


namespace NUMINAMATH_CALUDE_sum_of_squares_with_hcf_lcm_constraint_l557_55746

theorem sum_of_squares_with_hcf_lcm_constraint 
  (a b c : ℕ+) 
  (sum_of_squares : a^2 + b^2 + c^2 = 2011)
  (x : ℕ) 
  (hx : x = Nat.gcd a (Nat.gcd b c))
  (y : ℕ) 
  (hy : y = Nat.lcm a (Nat.lcm b c))
  (hxy : x + y = 388) : 
  a + b + c = 61 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_with_hcf_lcm_constraint_l557_55746


namespace NUMINAMATH_CALUDE_molecular_weight_proof_l557_55780

/-- Given 4 moles of a compound with a total molecular weight of 304 g/mol,
    prove that the molecular weight of 1 mole of the compound is 76 g/mol. -/
theorem molecular_weight_proof (total_weight : ℝ) (total_moles : ℝ) 
  (h1 : total_weight = 304)
  (h2 : total_moles = 4) :
  total_weight / total_moles = 76 := by
  sorry

end NUMINAMATH_CALUDE_molecular_weight_proof_l557_55780


namespace NUMINAMATH_CALUDE_distribute_objects_l557_55727

theorem distribute_objects (n : ℕ) (k : ℕ) (m : ℕ) (h1 : n = 12) (h2 : k = 3) (h3 : m = 4) (h4 : n = k * m) :
  (Nat.factorial n) / ((Nat.factorial m)^k) = 34650 :=
sorry

end NUMINAMATH_CALUDE_distribute_objects_l557_55727


namespace NUMINAMATH_CALUDE_abs_sum_eq_sum_abs_iff_product_pos_l557_55784

theorem abs_sum_eq_sum_abs_iff_product_pos (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  |a + b| = |a| + |b| ↔ a * b > 0 := by sorry

end NUMINAMATH_CALUDE_abs_sum_eq_sum_abs_iff_product_pos_l557_55784


namespace NUMINAMATH_CALUDE_range_of_m_for_line_intersecting_semicircle_l557_55702

/-- A line intersecting a semicircle at exactly two points -/
structure LineIntersectingSemicircle where
  m : ℝ
  intersects_twice : ∃ (x₁ y₁ x₂ y₂ : ℝ),
    x₁ + y₁ = m ∧ y₁ = Real.sqrt (9 - x₁^2) ∧ y₁ ≥ 0 ∧
    x₂ + y₂ = m ∧ y₂ = Real.sqrt (9 - x₂^2) ∧ y₂ ≥ 0 ∧
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂)

/-- The range of m values for lines intersecting the semicircle at exactly two points -/
theorem range_of_m_for_line_intersecting_semicircle (l : LineIntersectingSemicircle) :
  l.m ≥ 3 ∧ l.m < 3 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_for_line_intersecting_semicircle_l557_55702


namespace NUMINAMATH_CALUDE_milburg_adults_l557_55799

theorem milburg_adults (total_population children : ℕ) 
  (h1 : total_population = 5256)
  (h2 : children = 2987) :
  total_population - children = 2269 := by
  sorry

end NUMINAMATH_CALUDE_milburg_adults_l557_55799


namespace NUMINAMATH_CALUDE_unique_divisible_by_11_l557_55733

/-- A number is divisible by 11 if the alternating sum of its digits is divisible by 11 -/
def isDivisibleBy11 (n : ℕ) : Prop :=
  (n / 100 - (n / 10 % 10) + n % 10) % 11 = 0

/-- The set of three-digit numbers with units digit 5 and hundreds digit 6 -/
def validNumbers : Set ℕ :=
  {n : ℕ | 600 ≤ n ∧ n < 700 ∧ n % 10 = 5 ∧ n / 100 = 6}

theorem unique_divisible_by_11 :
  ∃! n : ℕ, n ∈ validNumbers ∧ isDivisibleBy11 n ∧ n = 605 := by
  sorry

#check unique_divisible_by_11

end NUMINAMATH_CALUDE_unique_divisible_by_11_l557_55733


namespace NUMINAMATH_CALUDE_original_price_calculation_l557_55789

def discount_rate : ℝ := 0.2
def discounted_price : ℝ := 56

theorem original_price_calculation :
  ∃ (original_price : ℝ), 
    original_price * (1 - discount_rate) = discounted_price ∧ 
    original_price = 70 :=
by sorry

end NUMINAMATH_CALUDE_original_price_calculation_l557_55789


namespace NUMINAMATH_CALUDE_parallelogram_height_l557_55739

/-- Given a parallelogram with area 384 cm² and base 24 cm, its height is 16 cm -/
theorem parallelogram_height (area : ℝ) (base : ℝ) (height : ℝ) : 
  area = 384 ∧ base = 24 ∧ area = base * height → height = 16 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_height_l557_55739


namespace NUMINAMATH_CALUDE_chrysanthemum_arrangement_count_l557_55762

/-- Represents the number of pots for each color of chrysanthemums -/
structure ChrysanthemumPots where
  yellow : Nat
  white : Nat
  red : Nat

/-- Calculates the number of arrangements for chrysanthemum pots -/
def countArrangements (pots : ChrysanthemumPots) : Nat :=
  sorry

/-- Theorem stating the number of arrangements for the given conditions -/
theorem chrysanthemum_arrangement_count :
  let pots : ChrysanthemumPots := { yellow := 2, white := 2, red := 1 }
  countArrangements pots = 24 := by
  sorry

end NUMINAMATH_CALUDE_chrysanthemum_arrangement_count_l557_55762


namespace NUMINAMATH_CALUDE_smallest_area_special_square_l557_55754

/-- A square with vertices on a line and parabola -/
structure SpecialSquare where
  -- One pair of opposite vertices lie on this line
  line : Real → Real
  -- The other pair of opposite vertices lie on this parabola
  parabola : Real → Real
  -- The line is y = -2x + 17
  line_eq : line = fun x => -2 * x + 17
  -- The parabola is y = x^2 - 2
  parabola_eq : parabola = fun x => x^2 - 2

/-- The smallest possible area of a SpecialSquare is 160 -/
theorem smallest_area_special_square (s : SpecialSquare) :
  ∃ (area : Real), area = 160 ∧ 
  (∀ (other_area : Real), other_area ≥ area) :=
by sorry

end NUMINAMATH_CALUDE_smallest_area_special_square_l557_55754


namespace NUMINAMATH_CALUDE_jessica_withdrawal_l557_55759

/-- Proves that given the conditions of Jessica's bank transactions, 
    the amount she initially withdrew was $200. -/
theorem jessica_withdrawal (B : ℚ) : 
  (3/5 * B + 1/5 * (3/5 * B) = 360) → 
  (2/5 * B = 200) := by
  sorry

#eval (2/5 : ℚ) * 500 -- Optional: to verify the result

end NUMINAMATH_CALUDE_jessica_withdrawal_l557_55759


namespace NUMINAMATH_CALUDE_francie_remaining_money_l557_55779

/-- Calculates the remaining money after Francie's savings and purchases -/
def remaining_money (initial_allowance : ℕ) (initial_weeks : ℕ) 
  (raised_allowance : ℕ) (raised_weeks : ℕ) (video_game_cost : ℕ) : ℕ :=
  let total_savings := initial_allowance * initial_weeks + raised_allowance * raised_weeks
  let after_clothes := total_savings / 2
  after_clothes - video_game_cost

/-- Theorem stating that Francie's remaining money is $3 -/
theorem francie_remaining_money :
  remaining_money 5 8 6 6 35 = 3 := by
  sorry

#eval remaining_money 5 8 6 6 35

end NUMINAMATH_CALUDE_francie_remaining_money_l557_55779


namespace NUMINAMATH_CALUDE_money_equalization_l557_55701

theorem money_equalization (xiaoli_money xiaogang_money : ℕ) : 
  xiaoli_money = 18 → xiaogang_money = 24 → 
  (xiaogang_money - (xiaoli_money + xiaogang_money) / 2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_money_equalization_l557_55701


namespace NUMINAMATH_CALUDE_fruit_drink_total_amount_l557_55797

/-- Represents the composition of a fruit drink -/
structure FruitDrink where
  orange_percent : Real
  watermelon_percent : Real
  grape_percent : Real
  pineapple_percent : Real
  grape_ounces : Real
  total_ounces : Real

/-- The theorem stating the total amount of the drink given its composition -/
theorem fruit_drink_total_amount (drink : FruitDrink) 
  (h1 : drink.orange_percent = 0.1)
  (h2 : drink.watermelon_percent = 0.55)
  (h3 : drink.grape_percent = 0.2)
  (h4 : drink.pineapple_percent = 1 - (drink.orange_percent + drink.watermelon_percent + drink.grape_percent))
  (h5 : drink.grape_ounces = 40)
  (h6 : drink.total_ounces * drink.grape_percent = drink.grape_ounces) :
  drink.total_ounces = 200 := by
  sorry

end NUMINAMATH_CALUDE_fruit_drink_total_amount_l557_55797


namespace NUMINAMATH_CALUDE_parabola_point_shift_l557_55742

/-- Given a point P(m,n) on the parabola y = ax^2 (a ≠ 0), 
    prove that (m-1, n) lies on y = a(x+1)^2 -/
theorem parabola_point_shift (a m n : ℝ) (h1 : a ≠ 0) (h2 : n = a * m^2) :
  n = a * ((m - 1) + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_shift_l557_55742


namespace NUMINAMATH_CALUDE_project_time_allocation_l557_55796

theorem project_time_allocation (worker1 worker2 worker3 : ℚ) 
  (h1 : worker1 = 1/2)
  (h2 : worker3 = 1/3)
  (h_total : worker1 + worker2 + worker3 = 1) :
  worker2 = 1/6 := by
sorry

end NUMINAMATH_CALUDE_project_time_allocation_l557_55796


namespace NUMINAMATH_CALUDE_double_inequality_solution_l557_55732

theorem double_inequality_solution (x : ℝ) : 
  -1 < (x^2 - 20*x + 21) / (x^2 - 4*x + 5) ∧ 
  (x^2 - 20*x + 21) / (x^2 - 4*x + 5) < 1 ↔ 
  (2 < x ∧ x < 1) ∨ (26 < x) :=
sorry

end NUMINAMATH_CALUDE_double_inequality_solution_l557_55732


namespace NUMINAMATH_CALUDE_first_row_desks_l557_55753

/-- Calculates the number of desks in the first row given the total number of rows,
    the increase in desks per row, and the total number of students that can be seated. -/
def desks_in_first_row (total_rows : ℕ) (increase_per_row : ℕ) (total_students : ℕ) : ℕ :=
  (2 * total_students - total_rows * (total_rows - 1) * increase_per_row) / (2 * total_rows)

/-- Theorem stating that given 8 rows of desks, where each subsequent row has 2 more desks
    than the previous row, and a total of 136 students can be seated, the number of desks
    in the first row is 10. -/
theorem first_row_desks :
  desks_in_first_row 8 2 136 = 10 := by
  sorry

end NUMINAMATH_CALUDE_first_row_desks_l557_55753


namespace NUMINAMATH_CALUDE_area_of_triangle_MOI_l557_55765

/-- Triangle ABC with given side lengths -/
structure Triangle where
  AB : ℝ
  AC : ℝ
  BC : ℝ

/-- The given triangle -/
def givenTriangle : Triangle := { AB := 15, AC := 8, BC := 17 }

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Circumcenter of the triangle -/
noncomputable def O : Point := sorry

/-- Incenter of the triangle -/
noncomputable def I : Point := sorry

/-- Center of the circle tangent to AC, BC, and the circumcircle -/
noncomputable def M : Point := sorry

/-- Area of a triangle given three points -/
noncomputable def triangleArea (p1 p2 p3 : Point) : ℝ := sorry

/-- The main theorem -/
theorem area_of_triangle_MOI (t : Triangle) (h : t = givenTriangle) : 
  triangleArea O I M = 3.4 := by sorry

end NUMINAMATH_CALUDE_area_of_triangle_MOI_l557_55765

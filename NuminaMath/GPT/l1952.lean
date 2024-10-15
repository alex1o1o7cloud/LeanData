import Mathlib

namespace NUMINAMATH_GPT_max_checkers_on_chessboard_l1952_195297

theorem max_checkers_on_chessboard (n : ℕ) : 
  ∃ k : ℕ, k = 2 * n * (n / 2) := sorry

end NUMINAMATH_GPT_max_checkers_on_chessboard_l1952_195297


namespace NUMINAMATH_GPT_arithmetic_mean_solve_x_l1952_195228

theorem arithmetic_mean_solve_x (x : ℚ) :
  (x + 10 + 20 + 3 * x + 15 + 3 * x + 6) / 5 = 30 → x = 99 / 7 :=
by 
sorry

end NUMINAMATH_GPT_arithmetic_mean_solve_x_l1952_195228


namespace NUMINAMATH_GPT_julia_total_food_cost_l1952_195284

-- Definitions based on conditions
def weekly_total_cost : ℕ := 30
def rabbit_weeks : ℕ := 5
def rabbit_food_cost : ℕ := 12
def parrot_weeks : ℕ := 3
def parrot_food_cost : ℕ := weekly_total_cost - rabbit_food_cost

-- Proof statement
theorem julia_total_food_cost : 
  rabbit_weeks * rabbit_food_cost + parrot_weeks * parrot_food_cost = 114 := 
by 
  sorry

end NUMINAMATH_GPT_julia_total_food_cost_l1952_195284


namespace NUMINAMATH_GPT_student_l1952_195267

-- Definition of the conditions
def mistaken_calculation (x : ℤ) : ℤ :=
  x + 10

def correct_calculation (x : ℤ) : ℤ :=
  x + 5

-- Theorem statement: Prove that the student's result is 10 more than the correct result
theorem student's_error {x : ℤ} : mistaken_calculation x = correct_calculation x + 5 :=
by
  sorry

end NUMINAMATH_GPT_student_l1952_195267


namespace NUMINAMATH_GPT_find_n_such_that_4_pow_n_plus_5_pow_n_is_perfect_square_l1952_195254

theorem find_n_such_that_4_pow_n_plus_5_pow_n_is_perfect_square :
  ∃ n : ℕ, (4^n + 5^n) = k^2 ↔ n = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_n_such_that_4_pow_n_plus_5_pow_n_is_perfect_square_l1952_195254


namespace NUMINAMATH_GPT_ordered_pairs_1944_l1952_195270

theorem ordered_pairs_1944 :
  ∃ n : ℕ, (∀ x y : ℕ, (x * y = 1944 ↔ x > 0 ∧ y > 0)) → n = 24 :=
by
  sorry

end NUMINAMATH_GPT_ordered_pairs_1944_l1952_195270


namespace NUMINAMATH_GPT_circle_center_and_radius_l1952_195242

-- Define the equation of the circle
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 6*x + 5 = 0

-- Statement of the center and radius of the circle
theorem circle_center_and_radius :
  (∀ x y : ℝ, circle_equation x y) →
  (∃ (h k r : ℝ), (∀ x y : ℝ, circle_equation x y ↔ (x - h)^2 + (y - k)^2 = r^2) ∧ h = 3 ∧ k = 0 ∧ r = 2) :=
by
  sorry

end NUMINAMATH_GPT_circle_center_and_radius_l1952_195242


namespace NUMINAMATH_GPT_tourists_count_l1952_195227

theorem tourists_count (n k : ℕ) (h1 : 2 * n % k = 1) (h2 : 3 * n % k = 13) : k = 23 := 
sorry

end NUMINAMATH_GPT_tourists_count_l1952_195227


namespace NUMINAMATH_GPT_minimum_value_m_l1952_195257

noncomputable def f (x : ℝ) (phi : ℝ) : ℝ :=
  Real.sin (2 * x + phi)

theorem minimum_value_m (phi : ℝ) (m : ℝ) (h1 : |phi| < Real.pi / 2)
  (h2 : ∃ x ∈ Set.Icc 0 (Real.pi / 2), f x (Real.pi / 6) ≤ m) :
  m = -1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_m_l1952_195257


namespace NUMINAMATH_GPT_license_plates_count_l1952_195241

noncomputable def num_license_plates : Nat :=
  let num_w := 26 * 26      -- number of combinations for w
  let num_w_orders := 2     -- two possible orders for w
  let num_digits := 10 ^ 5  -- number of combinations for 5 digits
  let num_positions := 6    -- number of valid positions for w
  2 * num_positions * num_digits * num_w

theorem license_plates_count : num_license_plates = 809280000 := by
  sorry

end NUMINAMATH_GPT_license_plates_count_l1952_195241


namespace NUMINAMATH_GPT_find_value_of_a_l1952_195275

-- Define the setting for triangle ABC with sides a, b, c opposite to angles A, B, C respectively
variables {a b c : ℝ}
variables {A B C : ℝ}
variables (h1 : b^2 - c^2 + 2 * a = 0)
variables (h2 : Real.tan C / Real.tan B = 3)

-- Given conditions and conclusion for the proof problem
theorem find_value_of_a 
  (h1 : b^2 - c^2 + 2 * a = 0) 
  (h2 : Real.tan C / Real.tan B = 3) : 
  a = 4 := 
sorry

end NUMINAMATH_GPT_find_value_of_a_l1952_195275


namespace NUMINAMATH_GPT_evaluate_f_at_2_l1952_195210

def f (x : ℝ) : ℝ := x^3 - 3 * x + 2

theorem evaluate_f_at_2 : f 2 = 4 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_evaluate_f_at_2_l1952_195210


namespace NUMINAMATH_GPT_clock_chime_time_l1952_195298

/-- The proven time it takes for a wall clock to strike 12 times at 12 o'clock -/
theorem clock_chime_time :
  (∃ (interval_time : ℝ), (interval_time = 3) ∧ (∃ (time_12_times : ℝ), (time_12_times = interval_time * (12 - 1)) ∧ (time_12_times = 33))) :=
by
  sorry

end NUMINAMATH_GPT_clock_chime_time_l1952_195298


namespace NUMINAMATH_GPT_percent_of_a_is_20_l1952_195219

variable {a b c : ℝ}

theorem percent_of_a_is_20 (h1 : c = (x / 100) * a)
                          (h2 : c = 0.1 * b)
                          (h3 : b = 2 * a) :
  c = 0.2 * a := sorry

end NUMINAMATH_GPT_percent_of_a_is_20_l1952_195219


namespace NUMINAMATH_GPT_distance_between_trees_l1952_195223

theorem distance_between_trees (yard_length : ℕ) (num_trees : ℕ)
  (h_yard : yard_length = 400) (h_trees : num_trees = 26) : 
  (yard_length / (num_trees - 1)) = 16 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_trees_l1952_195223


namespace NUMINAMATH_GPT_boys_in_class_l1952_195265

-- Define the conditions given in the problem
def ratio_boys_to_girls (boys girls : ℕ) : Prop := boys = 4 * (boys + girls) / 7 ∧ girls = 3 * (boys + girls) / 7
def total_students (boys girls : ℕ) : Prop := boys + girls = 49

-- Define the statement to be proved
theorem boys_in_class (boys girls : ℕ) (h1 : ratio_boys_to_girls boys girls) (h2 : total_students boys girls) : boys = 28 :=
by
  sorry

end NUMINAMATH_GPT_boys_in_class_l1952_195265


namespace NUMINAMATH_GPT_candy_problem_l1952_195282

theorem candy_problem (N a S : ℕ) 
  (h1 : ∀ i : ℕ, i < N → a = S - 7 - a)
  (h2 : ∀ i : ℕ, i < N → a > 1)
  (h3 : S = N * a) : 
  S = 21 :=
by
  sorry

end NUMINAMATH_GPT_candy_problem_l1952_195282


namespace NUMINAMATH_GPT_evaluate_expression_l1952_195200

theorem evaluate_expression : 2 + (2 / (2 + (2 / (2 + 3)))) = 17 / 6 := 
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1952_195200


namespace NUMINAMATH_GPT_train_passes_jogger_l1952_195213

noncomputable def speed_of_jogger_kmph := 9
noncomputable def speed_of_train_kmph := 45
noncomputable def jogger_lead_m := 270
noncomputable def train_length_m := 120

noncomputable def speed_of_jogger_mps := speed_of_jogger_kmph * (1000 / 3600) -- converting km/hr to m/s
noncomputable def speed_of_train_mps := speed_of_train_kmph * (1000 / 3600) -- converting km/hr to m/s
noncomputable def relative_speed_mps := speed_of_train_mps - speed_of_jogger_mps
noncomputable def total_distance_m := jogger_lead_m + train_length_m
noncomputable def time_to_pass_jogger := total_distance_m / relative_speed_mps

theorem train_passes_jogger : time_to_pass_jogger = 39 :=
  by
    -- Proof steps would be provided here
    sorry

end NUMINAMATH_GPT_train_passes_jogger_l1952_195213


namespace NUMINAMATH_GPT_remainder_of_eggs_is_2_l1952_195249

-- Define the number of eggs each person has
def david_eggs : ℕ := 45
def emma_eggs : ℕ := 52
def fiona_eggs : ℕ := 25

-- Define total eggs and remainder function
def total_eggs : ℕ := david_eggs + emma_eggs + fiona_eggs
def remainder (a b : ℕ) : ℕ := a % b

-- Prove that the remainder of total eggs divided by 10 is 2
theorem remainder_of_eggs_is_2 : remainder total_eggs 10 = 2 := by
  sorry

end NUMINAMATH_GPT_remainder_of_eggs_is_2_l1952_195249


namespace NUMINAMATH_GPT_solve_system_of_equations_l1952_195258

theorem solve_system_of_equations : ∃ (x y : ℝ), 4 * x + y = 6 ∧ 3 * x - y = 1 ∧ x = 1 ∧ y = 2 :=
by
  existsi (1 : ℝ)
  existsi (2 : ℝ)
  sorry

end NUMINAMATH_GPT_solve_system_of_equations_l1952_195258


namespace NUMINAMATH_GPT_questionnaires_drawn_from_D_l1952_195294

theorem questionnaires_drawn_from_D (a b c d : ℕ) (A_s B_s C_s D_s: ℕ) (common_diff: ℕ)
  (h1 : a + b + c + d = 1000)
  (h2 : b = a + common_diff)
  (h3 : c = a + 2 * common_diff)
  (h4 : d = a + 3 * common_diff)
  (h5 : A_s = 30 - common_diff)
  (h6 : B_s = 30)
  (h7 : C_s = 30 + common_diff)
  (h8 : D_s = 30 + 2 * common_diff)
  (h9 : A_s + B_s + C_s + D_s = 150)
  : D_s = 60 := sorry

end NUMINAMATH_GPT_questionnaires_drawn_from_D_l1952_195294


namespace NUMINAMATH_GPT_initial_distance_planes_l1952_195216

theorem initial_distance_planes (speed_A speed_B : ℝ) (time_seconds : ℝ) (time_hours : ℝ) (distance_A distance_B : ℝ) (total_distance : ℝ) :
  speed_A = 240 ∧ speed_B = 360 ∧ time_seconds = 72000 ∧ time_hours = 20 ∧ 
  time_hours = time_seconds / 3600 ∧
  distance_A = speed_A * time_hours ∧ 
  distance_B = speed_B * time_hours ∧ 
  total_distance = distance_A + distance_B →
  total_distance = 12000 :=
by
  intros
  sorry

end NUMINAMATH_GPT_initial_distance_planes_l1952_195216


namespace NUMINAMATH_GPT_rotameter_gas_phase_measurement_l1952_195273

theorem rotameter_gas_phase_measurement
  (liquid_inch_per_lpm : ℝ) (liquid_liter_per_minute : ℝ) (gas_inch_movement_ratio : ℝ) (gas_liter_passed : ℝ) :
  liquid_inch_per_lpm = 2.5 → liquid_liter_per_minute = 60 → gas_inch_movement_ratio = 0.5 → gas_liter_passed = 192 →
  (gas_inch_movement_ratio * liquid_inch_per_lpm * gas_liter_passed / liquid_liter_per_minute) = 4 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_rotameter_gas_phase_measurement_l1952_195273


namespace NUMINAMATH_GPT_find_number_l1952_195209

theorem find_number (N Q : ℕ) (h1 : N = 5 * Q) (h2 : Q + N + 5 = 65) : N = 50 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l1952_195209


namespace NUMINAMATH_GPT_senate_arrangement_l1952_195283

def countArrangements : ℕ :=
  let totalSeats : ℕ := 14
  let democrats : ℕ := 6
  let republicans : ℕ := 6
  let independents : ℕ := 2
  -- The calculation for arrangements considering fixed elements, and permutations adjusted for rotation
  12 * (Nat.factorial 10 / 2)

theorem senate_arrangement :
  let totalSeats : ℕ := 14
  let democrats : ℕ := 6
  let republicans : ℕ := 6
  let independents : ℕ := 2
  -- Total ways to arrange the members around the table under the given conditions
  countArrangements = 21772800 :=
by
  sorry

end NUMINAMATH_GPT_senate_arrangement_l1952_195283


namespace NUMINAMATH_GPT_jack_received_emails_in_the_morning_l1952_195208

theorem jack_received_emails_in_the_morning
  (total_emails : ℕ)
  (afternoon_emails : ℕ)
  (morning_emails : ℕ) 
  (h1 : total_emails = 8)
  (h2 : afternoon_emails = 5)
  (h3 : total_emails = morning_emails + afternoon_emails) :
  morning_emails = 3 :=
  by
    -- proof omitted
    sorry

end NUMINAMATH_GPT_jack_received_emails_in_the_morning_l1952_195208


namespace NUMINAMATH_GPT_initial_marbles_count_l1952_195215

-- Definitions as per conditions in the problem
variables (x y z : ℕ)

-- Condition 1: Removing one black marble results in one-eighth of the remaining marbles being black
def condition1 : Prop := (x - 1) * 8 = (x + y - 1)

-- Condition 2: Removing three white marbles results in one-sixth of the remaining marbles being black
def condition2 : Prop := x * 6 = (x + y - 3)

-- Proof that initial total number of marbles is 9 given conditions
theorem initial_marbles_count (h1 : condition1 x y) (h2 : condition2 x y) : x + y = 9 :=
by 
  sorry

end NUMINAMATH_GPT_initial_marbles_count_l1952_195215


namespace NUMINAMATH_GPT_problem_statement_l1952_195293

-- Let's define the conditions
def num_blue_balls : ℕ := 8
def num_green_balls : ℕ := 7
def total_balls : ℕ := num_blue_balls + num_green_balls

-- Function to calculate combinations (binomial coefficients)
def combination (n r : ℕ) : ℕ :=
  n.choose r

-- Specific combinations for this problem
def blue_ball_ways : ℕ := combination num_blue_balls 3
def green_ball_ways : ℕ := combination num_green_balls 2
def total_ways : ℕ := combination total_balls 5

-- The number of favorable outcomes
def favorable_outcomes : ℕ := blue_ball_ways * green_ball_ways

-- The probability
def probability : ℚ := favorable_outcomes / total_ways

-- The theorem stating our result
theorem problem_statement : probability = 1176/3003 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l1952_195293


namespace NUMINAMATH_GPT_fred_balloon_count_l1952_195235

variable (Fred_balloons Sam_balloons Mary_balloons total_balloons : ℕ)

/-- 
  Given:
  - Fred has some yellow balloons
  - Sam has 6 yellow balloons
  - Mary has 7 yellow balloons
  - Total number of yellow balloons (Fred's, Sam's, and Mary's balloons) is 18

  Prove: Fred has 5 yellow balloons.
-/
theorem fred_balloon_count :
  Sam_balloons = 6 →
  Mary_balloons = 7 →
  total_balloons = 18 →
  Fred_balloons = total_balloons - (Sam_balloons + Mary_balloons) →
  Fred_balloons = 5 :=
by
  sorry

end NUMINAMATH_GPT_fred_balloon_count_l1952_195235


namespace NUMINAMATH_GPT_two_bags_remainder_l1952_195279

-- Given conditions
variables (n : ℕ)

-- Assume n ≡ 8 (mod 11)
def satisfied_mod_condition : Prop := n % 11 = 8

-- Prove that 2n ≡ 5 (mod 11)
theorem two_bags_remainder (h : satisfied_mod_condition n) : (2 * n) % 11 = 5 :=
by 
  unfold satisfied_mod_condition at h
  sorry

end NUMINAMATH_GPT_two_bags_remainder_l1952_195279


namespace NUMINAMATH_GPT_maximum_t_l1952_195260

theorem maximum_t {a b t : ℝ} (ha : 0 < a) (hb : a < b) (ht : b < t)
  (h_condition : b * Real.log a < a * Real.log b) : t ≤ Real.exp 1 :=
sorry

end NUMINAMATH_GPT_maximum_t_l1952_195260


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l1952_195262

theorem sufficient_but_not_necessary (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c > 0) :
  (a > b ∧ b > 0 ∧ c > 0) → (a / (a + c) > b / (b + c)) :=
by
  intros
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l1952_195262


namespace NUMINAMATH_GPT_remainder_of_16_pow_2048_mod_11_l1952_195229

theorem remainder_of_16_pow_2048_mod_11 : (16^2048) % 11 = 4 := by
  sorry

end NUMINAMATH_GPT_remainder_of_16_pow_2048_mod_11_l1952_195229


namespace NUMINAMATH_GPT_solution_l1952_195271

namespace ProofProblem

variables (a b : ℝ)

def five_times_a_minus_b_eq_60 := 5 * a - b = 60
def six_times_a_plus_b_lt_90 := 6 * a + b < 90

theorem solution (h1 : five_times_a_minus_b_eq_60 a b) (h2 : six_times_a_plus_b_lt_90 a b) :
  a < 150 / 11 ∧ b < 8.18 :=
sorry

end ProofProblem

end NUMINAMATH_GPT_solution_l1952_195271


namespace NUMINAMATH_GPT_part1_part2_l1952_195221

open Real

noncomputable def f (x : ℝ) (a : ℝ) := |2 * x - 1| - |x - a|

theorem part1 (a : ℝ) (h : a = 0) :
  {x : ℝ | f x a < 1} = {x : ℝ | 0 < x ∧ x < 2} :=
by
  sorry

theorem part2 (a : ℝ) (h : ∀ x : ℝ, f x a < 1 → |(1 - 2 * a)^2 / 6| > 3 / 2) 
  : a < -1 :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l1952_195221


namespace NUMINAMATH_GPT_total_yards_in_marathons_eq_495_l1952_195202

-- Definitions based on problem conditions
def marathon_miles : ℕ := 26
def marathon_yards : ℕ := 385
def yards_in_mile : ℕ := 1760
def marathons_run : ℕ := 15

-- Main proof statement
theorem total_yards_in_marathons_eq_495
  (miles_per_marathon : ℕ := marathon_miles)
  (yards_per_marathon : ℕ := marathon_yards)
  (yards_per_mile : ℕ := yards_in_mile)
  (marathons : ℕ := marathons_run) :
  let total_yards := marathons * yards_per_marathon
  let remaining_yards := total_yards % yards_per_mile
  remaining_yards = 495 :=
by
  sorry

end NUMINAMATH_GPT_total_yards_in_marathons_eq_495_l1952_195202


namespace NUMINAMATH_GPT_minimum_workers_in_team_A_l1952_195234

variable (a b c : ℤ)

theorem minimum_workers_in_team_A (h1 : b + 90 = 2 * (a - 90))
                               (h2 : a + c = 6 * (b - c)) :
  ∃ a ≥ 148, a = 153 :=
by
  sorry

end NUMINAMATH_GPT_minimum_workers_in_team_A_l1952_195234


namespace NUMINAMATH_GPT_range_of_a_l1952_195269

theorem range_of_a (a : ℝ) :
  (¬ ∃ x₀ : ℝ, x₀^2 - a*x₀ + 2 < 0) ↔ (a^2 ≤ 8) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1952_195269


namespace NUMINAMATH_GPT_speed_in_still_water_l1952_195255

def upstream_speed : ℝ := 25
def downstream_speed : ℝ := 35

theorem speed_in_still_water : (upstream_speed + downstream_speed) / 2 = 30 :=
by
  sorry

end NUMINAMATH_GPT_speed_in_still_water_l1952_195255


namespace NUMINAMATH_GPT_log_equation_l1952_195299

theorem log_equation :
  (3 / (Real.log 1000^4 / Real.log 8)) + (4 / (Real.log 1000^4 / Real.log 9)) = 3 :=
by
  sorry

end NUMINAMATH_GPT_log_equation_l1952_195299


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_equals_product_l1952_195286

theorem arithmetic_sequence_sum_equals_product :
  ∃ (a_1 a_2 a_3 : ℤ), (a_2 = a_1 + d) ∧ (a_3 = a_1 + 2 * d) ∧ 
    a_1 ≠ 0 ∧ (a_1 + a_2 + a_3 = a_1 * a_2 * a_3) ∧ 
    (∃ d x : ℤ, x ≠ 0 ∧ d ≠ 0 ∧ 
    ((x = 1 ∧ d = 1) ∨ (x = -3 ∧ d = 1) ∨ (x = 3 ∧ d = -1) ∨ (x = -1 ∧ d = -1))) :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_equals_product_l1952_195286


namespace NUMINAMATH_GPT_solver_inequality_l1952_195247

theorem solver_inequality (x : ℝ) :
  (2 * x - 1 ≥ x + 2) ∧ (x + 5 < 4 * x - 1) → (x ≥ 3) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_solver_inequality_l1952_195247


namespace NUMINAMATH_GPT_A_eq_B_l1952_195217

namespace SetsEquality

open Set

def A : Set ℝ := {x | ∃ a : ℝ, x = 5 - 4 * a + a^2}
def B : Set ℝ := {y | ∃ b : ℝ, y = 4 * b^2 + 4 * b + 2}

theorem A_eq_B : A = B := by
  sorry

end SetsEquality

end NUMINAMATH_GPT_A_eq_B_l1952_195217


namespace NUMINAMATH_GPT_triangle_obtuse_l1952_195278

theorem triangle_obtuse (α β γ : ℝ) 
  (h1 : α ≤ β) (h2 : β < γ) 
  (h3 : α + β + γ = 180) 
  (h4 : α + β < γ) : 
  γ > 90 :=
  sorry

end NUMINAMATH_GPT_triangle_obtuse_l1952_195278


namespace NUMINAMATH_GPT_train_length_l1952_195289

-- Define the given conditions
def train_cross_time : ℕ := 40 -- time in seconds
def train_speed_kmh : ℕ := 144 -- speed in km/h

-- Convert the speed from km/h to m/s
def kmh_to_ms (speed_kmh : ℕ) : ℕ :=
  (speed_kmh * 5) / 18 

def train_speed_ms : ℕ := kmh_to_ms train_speed_kmh

-- Theorem statement
theorem train_length :
  train_speed_ms * train_cross_time = 1600 :=
by
  sorry

end NUMINAMATH_GPT_train_length_l1952_195289


namespace NUMINAMATH_GPT_length_of_other_parallel_side_l1952_195237

theorem length_of_other_parallel_side 
  (a : ℝ) (h : ℝ) (A : ℝ) (x : ℝ) 
  (h_a : a = 16) (h_h : h = 15) (h_A : A = 270) 
  (h_area_formula : A = 1 / 2 * (a + x) * h) : 
  x = 20 :=
sorry

end NUMINAMATH_GPT_length_of_other_parallel_side_l1952_195237


namespace NUMINAMATH_GPT_johns_hats_cost_l1952_195248

theorem johns_hats_cost 
  (weeks : ℕ)
  (days_in_week : ℕ)
  (cost_per_hat : ℕ) 
  (h : weeks = 2 ∧ days_in_week = 7 ∧ cost_per_hat = 50) 
  : (weeks * days_in_week * cost_per_hat) = 700 :=
by
  sorry

end NUMINAMATH_GPT_johns_hats_cost_l1952_195248


namespace NUMINAMATH_GPT_negation_of_proposition_l1952_195245

theorem negation_of_proposition (x : ℝ) : ¬(∀ x : ℝ, x^2 ≥ 0) ↔ ∃ x : ℝ, x^2 < 0 :=
sorry

end NUMINAMATH_GPT_negation_of_proposition_l1952_195245


namespace NUMINAMATH_GPT_eight_bees_have_48_legs_l1952_195231

  def legs_per_bee : ℕ := 6
  def number_of_bees : ℕ := 8
  def total_legs : ℕ := 48

  theorem eight_bees_have_48_legs :
    number_of_bees * legs_per_bee = total_legs :=
  by
    sorry
  
end NUMINAMATH_GPT_eight_bees_have_48_legs_l1952_195231


namespace NUMINAMATH_GPT_relationship_among_a_b_c_l1952_195295

-- Defining the properties and conditions of the function
def is_even (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x

-- Defining the function f based on the condition
noncomputable def f (x m : ℝ) : ℝ := 2 ^ |x - m| - 1

-- Defining the constants a, b, c
noncomputable def a : ℝ := f (Real.log 3 / Real.log 0.5) 0
noncomputable def b : ℝ := f (Real.log 5 / Real.log 2) 0
noncomputable def c : ℝ := f 0 0

-- The theorem stating the relationship among a, b, and c
theorem relationship_among_a_b_c : c < a ∧ a < b := by
  sorry

end NUMINAMATH_GPT_relationship_among_a_b_c_l1952_195295


namespace NUMINAMATH_GPT_sum_of_roots_l1952_195288

theorem sum_of_roots (b : ℝ) (x : ℝ) (y : ℝ) :
  (x^2 - b * x + 20 = 0) ∧ (y^2 - b * y + 20 = 0) ∧ (x * y = 20) -> (x + y = b) := 
by
  sorry

end NUMINAMATH_GPT_sum_of_roots_l1952_195288


namespace NUMINAMATH_GPT_find_principal_amount_l1952_195230

-- Define the conditions as constants and assumptions
def monthly_interest_payment : ℝ := 216
def annual_interest_rate : ℝ := 0.09

-- Define the Lean statement to show that the amount of the investment is 28800
theorem find_principal_amount (monthly_payment : ℝ) (annual_rate : ℝ) (P : ℝ) :
  monthly_payment = 216 →
  annual_rate = 0.09 →
  P = 28800 :=
by
  intros 
  sorry

end NUMINAMATH_GPT_find_principal_amount_l1952_195230


namespace NUMINAMATH_GPT_value_of_a_l1952_195296

-- Declare and define the given conditions.
def line1 (y : ℝ) := y = 13
def line2 (x t y : ℝ) := y = 3 * x + t

-- Define the proof statement.
theorem value_of_a (a b t : ℝ) (h1 : line1 b) (h2 : line2 a t b) (ht : t = 1) : a = 4 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_l1952_195296


namespace NUMINAMATH_GPT_pencil_distribution_l1952_195238

theorem pencil_distribution (x : ℕ) 
  (Alice Bob Charles : ℕ)
  (h1 : Alice = 2 * Bob)
  (h2 : Charles = Bob + 3)
  (h3 : Bob = x)
  (total_pencils : 53 = Alice + Bob + Charles) : 
  Bob = 13 ∧ Alice = 26 ∧ Charles = 16 :=
by
  sorry

end NUMINAMATH_GPT_pencil_distribution_l1952_195238


namespace NUMINAMATH_GPT_geometric_sequence_sum_l1952_195220

variable {a b : ℝ} -- Parameters for real numbers a and b
variable (a_ne_zero : a ≠ 0) -- condition a ≠ 0

/-- Proof that in the geometric sequence {a_n}, given a_5 + a_6 = a and a_15 + a_16 = b, 
    a_25 + a_26 = b^2 / a --/
theorem geometric_sequence_sum (a5_plus_a6 : ℕ → ℝ) (a15_plus_a16 : ℕ → ℝ) (a25_plus_a26 : ℕ → ℝ)
  (h1 : a5_plus_a6 5 + a5_plus_a6 6 = a)
  (h2 : a15_plus_a16 15 + a15_plus_a16 16 = b) :
  a25_plus_a26 25 + a25_plus_a26 26 = b^2 / a :=
  sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l1952_195220


namespace NUMINAMATH_GPT_max_x_satisfying_ineq_l1952_195268

theorem max_x_satisfying_ineq : ∃ (x : ℤ), (x ≤ 1 ∧ ∀ (y : ℤ), (y > x → y > 1) ∧ (y ≤ 1 → (y : ℚ) / 3 + 7 / 4 < 9 / 4)) := 
by
  sorry

end NUMINAMATH_GPT_max_x_satisfying_ineq_l1952_195268


namespace NUMINAMATH_GPT_complement_union_example_l1952_195201

open Set

universe u

variable (U : Set ℕ) (A B : Set ℕ)

def U_def : Set ℕ := {0, 1, 2, 3, 4}
def A_def : Set ℕ := {0, 1, 2}
def B_def : Set ℕ := {2, 3}

theorem complement_union_example :
  (U \ A) ∪ B = {2, 3, 4} := 
by
  -- Proving the theorem considering
  -- complement and union operations on sets
  sorry

end NUMINAMATH_GPT_complement_union_example_l1952_195201


namespace NUMINAMATH_GPT_prob_union_of_mutually_exclusive_l1952_195277

-- Let's denote P as a probability function
variable {Ω : Type} (P : Set Ω → ℝ)

-- Define the mutually exclusive condition
def mutually_exclusive (A B : Set Ω) : Prop :=
  (A ∩ B) = ∅

-- State the theorem that we want to prove
theorem prob_union_of_mutually_exclusive (A B : Set Ω) 
  (h : mutually_exclusive A B) : P (A ∪ B) = P A + P B :=
sorry

end NUMINAMATH_GPT_prob_union_of_mutually_exclusive_l1952_195277


namespace NUMINAMATH_GPT_total_potatoes_l1952_195280

theorem total_potatoes (cooked_potatoes : ℕ) (time_per_potato : ℕ) (remaining_time : ℕ) (H1 : cooked_potatoes = 7) (H2 : time_per_potato = 5) (H3 : remaining_time = 45) : (cooked_potatoes + (remaining_time / time_per_potato) = 16) :=
by
  sorry

end NUMINAMATH_GPT_total_potatoes_l1952_195280


namespace NUMINAMATH_GPT_base_length_of_isosceles_triangle_l1952_195274

noncomputable def isosceles_triangle_base_length (height : ℝ) (radius : ℝ) : ℝ :=
  if height = 25 ∧ radius = 8 then 80 / 3 else 0

theorem base_length_of_isosceles_triangle :
  isosceles_triangle_base_length 25 8 = 80 / 3 :=
by
  -- skipping the proof
  sorry

end NUMINAMATH_GPT_base_length_of_isosceles_triangle_l1952_195274


namespace NUMINAMATH_GPT_solve_eq1_solve_eq2_l1952_195226

theorem solve_eq1 (x : ℝ) : (x - 2)^2 - 16 = 0 ↔ x = 6 ∨ x = -2 :=
by sorry

theorem solve_eq2 (x : ℝ) : (x + 3)^3 = -27 ↔ x = -6 :=
by sorry

end NUMINAMATH_GPT_solve_eq1_solve_eq2_l1952_195226


namespace NUMINAMATH_GPT_simplify_and_evaluate_division_l1952_195211

theorem simplify_and_evaluate_division (m : ℕ) (h : m = 10) : 
  (1 - (m / (m + 2))) / ((m^2 - 4 * m + 4) / (m^2 - 4)) = 1 / 4 :=
by sorry

end NUMINAMATH_GPT_simplify_and_evaluate_division_l1952_195211


namespace NUMINAMATH_GPT_abs_sum_zero_l1952_195212

theorem abs_sum_zero (a b : ℝ) (h : |a - 5| + |b + 8| = 0) : a + b = -3 := 
sorry

end NUMINAMATH_GPT_abs_sum_zero_l1952_195212


namespace NUMINAMATH_GPT_rate_per_sq_meter_l1952_195239

def length : ℝ := 5.5
def width : ℝ := 3.75
def totalCost : ℝ := 14437.5

theorem rate_per_sq_meter : (totalCost / (length * width)) = 700 := 
by sorry

end NUMINAMATH_GPT_rate_per_sq_meter_l1952_195239


namespace NUMINAMATH_GPT_complement_is_empty_l1952_195264

def U : Set ℕ := {1, 3}
def A : Set ℕ := {1, 3}

theorem complement_is_empty : (U \ A) = ∅ := 
by 
  sorry

end NUMINAMATH_GPT_complement_is_empty_l1952_195264


namespace NUMINAMATH_GPT_A_inter_B_empty_iff_A_union_B_eq_B_iff_l1952_195285

open Set

variable (a x : ℝ)

def A (a : ℝ) : Set ℝ := {x | 1 - a ≤ x ∧ x ≤ 1 + a}
def B : Set ℝ := {x | x < -1 ∨ x > 5}

theorem A_inter_B_empty_iff {a : ℝ} :
  (A a ∩ B = ∅) ↔ 0 ≤ a ∧ a ≤ 4 :=
by 
  sorry

theorem A_union_B_eq_B_iff {a : ℝ} :
  (A a ∪ B = B) ↔ a < -4 :=
by
  sorry

end NUMINAMATH_GPT_A_inter_B_empty_iff_A_union_B_eq_B_iff_l1952_195285


namespace NUMINAMATH_GPT_least_5_digit_divisible_l1952_195261

theorem least_5_digit_divisible (n : ℕ) (h1 : n ≥ 10000) (h2 : n < 100000)
  (h3 : 15 ∣ n) (h4 : 12 ∣ n) (h5 : 18 ∣ n) : n = 10080 :=
by
  sorry

end NUMINAMATH_GPT_least_5_digit_divisible_l1952_195261


namespace NUMINAMATH_GPT_noncongruent_triangles_count_l1952_195256

/-- Prove that the number of noncongruent integer-sided triangles 
with positive area and perimeter less than 20, 
which are neither equilateral, isosceles, nor right triangles, is 15. -/
theorem noncongruent_triangles_count : 
  ∃ n : ℕ, 
  (∀ (a b c : ℕ) (h : a ≤ b ∧ b ≤ c),
    a + b + c < 20 ∧ a + b > c ∧ a^2 + b^2 ≠ c^2 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c → n ≥ 15) :=
sorry

end NUMINAMATH_GPT_noncongruent_triangles_count_l1952_195256


namespace NUMINAMATH_GPT_electricity_average_l1952_195225

-- Define the daily electricity consumptions
def electricity_consumptions : List ℕ := [110, 101, 121, 119, 114]

-- Define the function to calculate the average
def average (l : List ℕ) : ℕ := l.sum / l.length

-- Formalize the proof problem
theorem electricity_average :
  average electricity_consumptions = 113 :=
  sorry

end NUMINAMATH_GPT_electricity_average_l1952_195225


namespace NUMINAMATH_GPT_line_intersects_xaxis_at_l1952_195204

theorem line_intersects_xaxis_at (x y : ℝ) 
  (h : 4 * y - 5 * x = 15) 
  (hy : y = 0) : (x, y) = (-3, 0) :=
by
  sorry

end NUMINAMATH_GPT_line_intersects_xaxis_at_l1952_195204


namespace NUMINAMATH_GPT_tim_tasks_per_day_l1952_195232

theorem tim_tasks_per_day (earnings_per_task : ℝ) (days_per_week : ℕ) (weekly_earnings : ℝ) :
  earnings_per_task = 1.2 ∧ days_per_week = 6 ∧ weekly_earnings = 720 → (weekly_earnings / days_per_week / earnings_per_task = 100) :=
by
  sorry

end NUMINAMATH_GPT_tim_tasks_per_day_l1952_195232


namespace NUMINAMATH_GPT_geom_seq_root_product_l1952_195246

theorem geom_seq_root_product
  (a : ℕ → ℝ)
  (h_geom : ∀ n, a (n + 1) = a n * a 1)
  (h_root1 : 3 * (a 1)^2 + 7 * a 1 - 9 = 0)
  (h_root10 : 3 * (a 10)^2 + 7 * a 10 - 9 = 0) :
  a 4 * a 7 = -3 := 
by
  sorry

end NUMINAMATH_GPT_geom_seq_root_product_l1952_195246


namespace NUMINAMATH_GPT_find_rate_of_interest_l1952_195252

variable (P : ℝ) (R : ℝ) (T : ℕ := 2)

-- Condition for Simple Interest (SI = Rs. 660 for 2 years)
def simple_interest :=
  P * R * ↑T / 100 = 660

-- Condition for Compound Interest (CI = Rs. 696.30 for 2 years)
def compound_interest :=
  P * ((1 + R / 100) ^ T - 1) = 696.30

-- We need to prove that R = 11
theorem find_rate_of_interest (P : ℝ) (h1 : simple_interest P R) (h2 : compound_interest P R) : 
  R = 11 := by
  sorry

end NUMINAMATH_GPT_find_rate_of_interest_l1952_195252


namespace NUMINAMATH_GPT_right_angle_case_acute_angle_case_obtuse_angle_case_l1952_195236

-- Definitions
def circumcenter (O : Type) (A B C : Type) : Prop := sorry -- Definition of circumcenter.

def orthocenter (H : Type) (A B C : Type) : Prop := sorry -- Definition of orthocenter.

noncomputable def R : ℝ := sorry -- Circumradius of the triangle.

-- Conditions
variables {A B C O H : Type}
  (h_circumcenter : circumcenter O A B C)
  (h_orthocenter : orthocenter H A B C)

-- The angles α β γ represent the angles of triangle ABC.
variables {α β γ : ℝ}

-- Statements
-- Case 1: ∠C = 90°
theorem right_angle_case (h_angle_C : γ = 90) (h_H_eq_C : H = C) (h_AB_eq_2R : AB = 2 * R) : AH + BH >= AB := by
  sorry

-- Case 2: ∠C < 90°
theorem acute_angle_case (h_angle_C_lt_90 : γ < 90) : O_in_triangle_AHB := by
  sorry

-- Case 3: ∠C > 90°
theorem obtuse_angle_case (h_angle_C_gt_90 : γ > 90) : AH + BH > 2 * R := by
  sorry

end NUMINAMATH_GPT_right_angle_case_acute_angle_case_obtuse_angle_case_l1952_195236


namespace NUMINAMATH_GPT_anna_grams_l1952_195291

-- Definitions based on conditions
def gary_grams : ℕ := 30
def gary_cost_per_gram : ℝ := 15
def anna_cost_per_gram : ℝ := 20
def combined_cost : ℝ := 1450

-- Statement to prove
theorem anna_grams : (combined_cost - (gary_grams * gary_cost_per_gram)) / anna_cost_per_gram = 50 :=
by 
  sorry

end NUMINAMATH_GPT_anna_grams_l1952_195291


namespace NUMINAMATH_GPT_smallest_n_l1952_195287

theorem smallest_n 
    (h1 : ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a * r = b ∧ b * r = c ∧ 7 * n + 1 = a + b + c)
    (h2 : ∃ (x y z : ℕ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x * s = y ∧ y * s = z ∧ 8 * n + 1 = x + y + z) :
    n = 22 :=
sorry

end NUMINAMATH_GPT_smallest_n_l1952_195287


namespace NUMINAMATH_GPT_profit_percentage_with_discount_correct_l1952_195233

variable (CP SP_without_discount Discounted_SP : ℝ)
variable (profit_without_discount profit_with_discount : ℝ)
variable (discount_percentage profit_percentage_without_discount profit_percentage_with_discount : ℝ)
variable (h1 : CP = 100)
variable (h2 : SP_without_discount = CP + profit_without_discount)
variable (h3 : profit_without_discount = 1.20 * CP)
variable (h4 : Discounted_SP = SP_without_discount - discount_percentage * SP_without_discount)
variable (h5 : discount_percentage = 0.05)
variable (h6 : profit_with_discount = Discounted_SP - CP)
variable (h7 : profit_percentage_with_discount = (profit_with_discount / CP) * 100)

theorem profit_percentage_with_discount_correct : profit_percentage_with_discount = 109 := by
  sorry

end NUMINAMATH_GPT_profit_percentage_with_discount_correct_l1952_195233


namespace NUMINAMATH_GPT_part1_part2_l1952_195205

open Set

def A (x : ℝ) : Prop := -1 < x ∧ x < 6
def B (x : ℝ) (a : ℝ) : Prop := x ≥ 1 + a ∨ x ≤ 1 - a

theorem part1 (a : ℝ) (hpos : 0 < a) :
  (∀ x, A x → ¬ B x a) ↔ a ≥ 5 :=
sorry

theorem part2 (a : ℝ) (hpos : 0 < a) :
  (∀ x, (¬ A x → B x a) ∧ ∃ x, ¬ A x ∧ ¬ B x a) ↔ (0 < a ∧ a ≤ 2) :=
sorry

end NUMINAMATH_GPT_part1_part2_l1952_195205


namespace NUMINAMATH_GPT_circle_radius_l1952_195214

theorem circle_radius (d : ℝ) (h : d = 10) : d / 2 = 5 :=
by
  sorry

end NUMINAMATH_GPT_circle_radius_l1952_195214


namespace NUMINAMATH_GPT_parabola_coeff_sum_l1952_195224

theorem parabola_coeff_sum (a b c : ℤ) (h₁ : a * (1:ℤ)^2 + b * 1 + c = 3)
                                      (h₂ : a * (-1)^2 + b * (-1) + c = 5)
                                      (vertex : ∀ x, a * (x + 1)^2 + 1 = a * x^2 + bx + c) :
a + b + c = 3 := 
sorry

end NUMINAMATH_GPT_parabola_coeff_sum_l1952_195224


namespace NUMINAMATH_GPT_line_through_intersection_parallel_to_given_line_l1952_195222

theorem line_through_intersection_parallel_to_given_line :
  ∃ k : ℝ, (∀ x y : ℝ, (2 * x + 3 * y + k = 0 ↔ (x, y) = (2, 1)) ∧
  (∀ m n : ℝ, (2 * m + 3 * n + 5 = 0 → 2 * m + 3 * n + k = 0))) →
  2 * x + 3 * y - 7 = 0 :=
sorry

end NUMINAMATH_GPT_line_through_intersection_parallel_to_given_line_l1952_195222


namespace NUMINAMATH_GPT_inequality_solution_l1952_195266

theorem inequality_solution 
  (a b c d e f : ℕ) 
  (h1 : a * d * f > b * c * f)
  (h2 : c * f * b > d * e * b) 
  (h3 : a * f - b * e = 1) 
  : d ≥ b + f := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_inequality_solution_l1952_195266


namespace NUMINAMATH_GPT_remainder_when_subtracted_l1952_195203

theorem remainder_when_subtracted (s t : ℕ) (hs : s % 6 = 2) (ht : t % 6 = 3) (h : s > t) : (s - t) % 6 = 5 :=
by
  sorry -- Proof not required

end NUMINAMATH_GPT_remainder_when_subtracted_l1952_195203


namespace NUMINAMATH_GPT_gcd_45045_30030_l1952_195206

/-- The greatest common divisor of 45045 and 30030 is 15015. -/
theorem gcd_45045_30030 : Nat.gcd 45045 30030 = 15015 :=
by 
  sorry

end NUMINAMATH_GPT_gcd_45045_30030_l1952_195206


namespace NUMINAMATH_GPT_talent_show_l1952_195250

theorem talent_show (B G : ℕ) (h1 : G = B + 22) (h2 : G + B = 34) : G = 28 :=
by
  sorry

end NUMINAMATH_GPT_talent_show_l1952_195250


namespace NUMINAMATH_GPT_find_x_value_l1952_195281

-- Define the condition as a hypothesis
def condition (x : ℝ) : Prop := (x / 4) - x - (3 / 6) = 1

-- State the theorem
theorem find_x_value (x : ℝ) (h : condition x) : x = -2 := 
by sorry

end NUMINAMATH_GPT_find_x_value_l1952_195281


namespace NUMINAMATH_GPT_no_real_intersection_l1952_195276

theorem no_real_intersection (f : ℝ → ℝ) 
  (h1 : ∀ x y : ℝ, x * f y = y * f x) 
  (h2 : f 1 = -1) : ¬∃ x : ℝ, f x = x^2 + 1 :=
by
  sorry

end NUMINAMATH_GPT_no_real_intersection_l1952_195276


namespace NUMINAMATH_GPT_graph_union_l1952_195240

-- Definitions of the conditions from part a)
def graph1 (z y : ℝ) : Prop := z^4 - 6 * y^4 = 3 * z^2 - 2

def graph_hyperbola (z y : ℝ) : Prop := z^2 - 3 * y^2 = 2

def graph_ellipse (z y : ℝ) : Prop := z^2 - 2 * y^2 = 1

-- Lean statement to prove the question is equivalent to the answer
theorem graph_union (z y : ℝ) : graph1 z y ↔ (graph_hyperbola z y ∨ graph_ellipse z y) := 
sorry

end NUMINAMATH_GPT_graph_union_l1952_195240


namespace NUMINAMATH_GPT_sum_of_D_coordinates_l1952_195290

-- Definition of the midpoint condition
def is_midpoint (N C D : ℝ × ℝ) : Prop :=
  N.1 = (C.1 + D.1) / 2 ∧ N.2 = (C.2 + D.2) / 2

-- Given points
def N : ℝ × ℝ := (5, -1)
def C : ℝ × ℝ := (11, 10)

-- Statement of the problem
theorem sum_of_D_coordinates :
  ∃ D : ℝ × ℝ, is_midpoint N C D ∧ (D.1 + D.2 = -13) :=
  sorry

end NUMINAMATH_GPT_sum_of_D_coordinates_l1952_195290


namespace NUMINAMATH_GPT_periodic_odd_function_l1952_195207

theorem periodic_odd_function (f : ℝ → ℝ) (period : ℝ) (h_periodic : ∀ x, f (x + period) = f x) (h_odd : ∀ x, f (-x) = -f x) (h_value : f (-3) = 1) (α : ℝ) (h_tan : Real.tan α = 2) :
  f (20 * Real.sin α * Real.cos α) = -1 := 
sorry

end NUMINAMATH_GPT_periodic_odd_function_l1952_195207


namespace NUMINAMATH_GPT_probability_xavier_yvonne_not_zelda_wendell_l1952_195218

theorem probability_xavier_yvonne_not_zelda_wendell
  (P_Xavier_solves : ℚ)
  (P_Yvonne_solves : ℚ)
  (P_Zelda_solves : ℚ)
  (P_Wendell_solves : ℚ) :
  P_Xavier_solves = 1/4 →
  P_Yvonne_solves = 1/3 →
  P_Zelda_solves = 5/8 →
  P_Wendell_solves = 1/2 →
  (P_Xavier_solves * P_Yvonne_solves * (1 - P_Zelda_solves) * (1 - P_Wendell_solves)) = 1/64 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  simp
  sorry

end NUMINAMATH_GPT_probability_xavier_yvonne_not_zelda_wendell_l1952_195218


namespace NUMINAMATH_GPT_inequality_trig_l1952_195292

theorem inequality_trig 
  (x y z : ℝ) 
  (hx : 0 < x ∧ x < (π / 2)) 
  (hy : 0 < y ∧ y < (π / 2)) 
  (hz : 0 < z ∧ z < (π / 2)) :
  (π / 2) + 2 * (Real.sin x) * (Real.cos y) + 2 * (Real.sin y) * (Real.cos z) > 
  (Real.sin (2 * x)) + (Real.sin (2 * y)) + (Real.sin (2 * z)) :=
by
  sorry  -- The proof is omitted

end NUMINAMATH_GPT_inequality_trig_l1952_195292


namespace NUMINAMATH_GPT_express_b_c_range_a_not_monotonic_l1952_195253

noncomputable def f (a b c x : ℝ) : ℝ := (a * x^2 + b * x + c) * Real.exp (-x)
noncomputable def f' (a b c x : ℝ) : ℝ := 
    (a * x^2 + b * x + c) * (-Real.exp (-x)) + (2 * a * x + b) * Real.exp (-x)

theorem express_b_c (a : ℝ) : 
    (∃ b c : ℝ, f a b c 0 = 2 * a ∧ f' a b c 0 = Real.pi / 4) → 
    (∃ b c : ℝ, b = 1 + 2 * a ∧ c = 2 * a) := 
sorry

noncomputable def g (a x : ℝ) : ℝ := -a * x^2 - x + 1

theorem range_a_not_monotonic (a : ℝ) : 
    (¬ (∀ x y : ℝ, x ∈ Set.Ici (1 / 2) → y ∈ Set.Ici (1 / 2) → x < y → g a x ≤ g a y)) → 
    (-1 / 4 < a ∧ a < 2) := 
sorry

end NUMINAMATH_GPT_express_b_c_range_a_not_monotonic_l1952_195253


namespace NUMINAMATH_GPT_frame_percentage_l1952_195244

theorem frame_percentage : 
  let side_length := 80
  let frame_width := 4
  let total_area := side_length * side_length
  let picture_side_length := side_length - 2 * frame_width
  let picture_area := picture_side_length * picture_side_length
  let frame_area := total_area - picture_area
  let frame_percentage := (frame_area * 100) / total_area
  frame_percentage = 19 := 
by
  sorry

end NUMINAMATH_GPT_frame_percentage_l1952_195244


namespace NUMINAMATH_GPT_republicans_in_house_l1952_195259

theorem republicans_in_house (D R : ℕ) (h1 : D + R = 434) (h2 : R = D + 30) : R = 232 :=
by sorry

end NUMINAMATH_GPT_republicans_in_house_l1952_195259


namespace NUMINAMATH_GPT_at_least_one_true_l1952_195263

theorem at_least_one_true (p q : Prop) (h : ¬(p ∨ q) = false) : p ∨ q :=
by
  sorry

end NUMINAMATH_GPT_at_least_one_true_l1952_195263


namespace NUMINAMATH_GPT_problem_statement_l1952_195251

theorem problem_statement : ((26.3 * 12 * 20) / 3 + 125 - Real.sqrt 576 = 21141) :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1952_195251


namespace NUMINAMATH_GPT_findAnalyticalExpression_l1952_195272

-- Defining the point A as a structure with x and y coordinates
structure Point where
  x : ℝ
  y : ℝ

-- Defining a line as having a slope and y-intercept
structure Line where
  slope : ℝ
  intercept : ℝ

-- Condition: Line 1 is parallel to y = 2x - 3
def line1 : Line := {slope := 2, intercept := -3}

-- Condition: Line 2 passes through point A
def point_A : Point := {x := -2, y := -1}

-- The theorem statement:
theorem findAnalyticalExpression : 
  ∃ b : ℝ, (∀ x : ℝ, (point_A.y = line1.slope * point_A.x + b) → b = 3) ∧ 
            ∀ x : ℝ, (line1.slope * x + b = 2 * x + 3) :=
sorry

end NUMINAMATH_GPT_findAnalyticalExpression_l1952_195272


namespace NUMINAMATH_GPT_MaryIncomeIs64PercentOfJuanIncome_l1952_195243

variable {J T M : ℝ}

-- Conditions
def TimIncome (J : ℝ) : ℝ := 0.40 * J
def MaryIncome (T : ℝ) : ℝ := 1.60 * T

-- Theorem to prove
theorem MaryIncomeIs64PercentOfJuanIncome (J : ℝ) :
  MaryIncome (TimIncome J) = 0.64 * J :=
by
  sorry

end NUMINAMATH_GPT_MaryIncomeIs64PercentOfJuanIncome_l1952_195243

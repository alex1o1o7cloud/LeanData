import Mathlib

namespace NUMINAMATH_CALUDE_shirt_discount_price_l1073_107347

theorem shirt_discount_price (original_price discount_percentage : ℝ) 
  (h1 : original_price = 80)
  (h2 : discount_percentage = 15) : 
  original_price * (1 - discount_percentage / 100) = 68 := by
  sorry

end NUMINAMATH_CALUDE_shirt_discount_price_l1073_107347


namespace NUMINAMATH_CALUDE_square_area_ratio_l1073_107399

theorem square_area_ratio (y : ℝ) (h : y > 0) : 
  (y^2) / ((5*y)^2) = 1 / 25 := by sorry

end NUMINAMATH_CALUDE_square_area_ratio_l1073_107399


namespace NUMINAMATH_CALUDE_magazine_subscription_cost_l1073_107372

/-- If a 35% reduction in a cost results in a decrease of $611, then the original cost was $1745.71 -/
theorem magazine_subscription_cost (C : ℝ) : (0.35 * C = 611) → C = 1745.71 := by
  sorry

end NUMINAMATH_CALUDE_magazine_subscription_cost_l1073_107372


namespace NUMINAMATH_CALUDE_max_concentration_time_l1073_107352

def drug_concentration_peak_time : ℝ := 0.65
def time_uncertainty : ℝ := 0.15

theorem max_concentration_time :
  drug_concentration_peak_time + time_uncertainty = 0.8 := by sorry

end NUMINAMATH_CALUDE_max_concentration_time_l1073_107352


namespace NUMINAMATH_CALUDE_magnitude_of_3_minus_4i_l1073_107325

theorem magnitude_of_3_minus_4i :
  Complex.abs (3 - 4*Complex.I) = 5 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_3_minus_4i_l1073_107325


namespace NUMINAMATH_CALUDE_johnny_work_hours_l1073_107361

theorem johnny_work_hours (hourly_wage : ℝ) (total_earned : ℝ) (hours_worked : ℝ) : 
  hourly_wage = 2.35 →
  total_earned = 11.75 →
  hours_worked = total_earned / hourly_wage →
  hours_worked = 5 := by
sorry

end NUMINAMATH_CALUDE_johnny_work_hours_l1073_107361


namespace NUMINAMATH_CALUDE_multiple_optimal_solutions_l1073_107369

/-- The feasible region defined by the given linear constraints -/
def FeasibleRegion (x y : ℝ) : Prop :=
  2 * x - y + 2 ≥ 0 ∧ x - 3 * y + 1 ≤ 0 ∧ x + y - 2 ≤ 0

/-- The objective function z -/
def ObjectiveFunction (a x y : ℝ) : ℝ := a * x - y

/-- The theorem stating that a = 1/3 results in multiple optimal solutions -/
theorem multiple_optimal_solutions :
  ∃ (a : ℝ), a > 0 ∧
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧
    FeasibleRegion x₁ y₁ ∧ FeasibleRegion x₂ y₂ ∧
    ObjectiveFunction a x₁ y₁ = ObjectiveFunction a x₂ y₂ ∧
    (∀ (x y : ℝ), FeasibleRegion x y → ObjectiveFunction a x y ≤ ObjectiveFunction a x₁ y₁)) ∧
  a = 1/3 :=
sorry

end NUMINAMATH_CALUDE_multiple_optimal_solutions_l1073_107369


namespace NUMINAMATH_CALUDE_probability_neither_correct_l1073_107389

theorem probability_neither_correct (P_A P_B P_AB : ℝ) 
  (h1 : P_A = 0.75)
  (h2 : P_B = 0.70)
  (h3 : P_AB = 0.65)
  (h4 : 0 ≤ P_A ∧ P_A ≤ 1)
  (h5 : 0 ≤ P_B ∧ P_B ≤ 1)
  (h6 : 0 ≤ P_AB ∧ P_AB ≤ 1) :
  1 - (P_A + P_B - P_AB) = 0.20 := by
  sorry

#check probability_neither_correct

end NUMINAMATH_CALUDE_probability_neither_correct_l1073_107389


namespace NUMINAMATH_CALUDE_min_sum_arithmetic_sequence_l1073_107391

/-- An arithmetic sequence with a_n = 2n - 19 -/
def a (n : ℕ) : ℤ := 2 * n - 19

/-- Sum of the first n terms of the arithmetic sequence -/
def S (n : ℕ) : ℤ := n^2 - 18 * n

theorem min_sum_arithmetic_sequence :
  ∃ k : ℕ, k > 0 ∧ 
  (∀ n : ℕ, n > 0 → S n ≥ S k) ∧
  S k = -81 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_arithmetic_sequence_l1073_107391


namespace NUMINAMATH_CALUDE_cafeteria_pies_correct_l1073_107331

def cafeteria_pies (initial_apples : ℕ) (handed_out : ℕ) (apples_per_pie : ℕ) : ℕ :=
  (initial_apples - handed_out) / apples_per_pie

theorem cafeteria_pies_correct :
  cafeteria_pies 50 5 5 = 9 := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_pies_correct_l1073_107331


namespace NUMINAMATH_CALUDE_distribute_five_balls_four_boxes_l1073_107362

/-- The number of ways to distribute n indistinguishable balls into k distinguishable boxes,
    with at least one box containing a ball -/
def distribute_balls (n k : ℕ) : ℕ :=
  sorry

theorem distribute_five_balls_four_boxes :
  distribute_balls 5 4 = 52 := by
  sorry

end NUMINAMATH_CALUDE_distribute_five_balls_four_boxes_l1073_107362


namespace NUMINAMATH_CALUDE_product_inequality_l1073_107301

theorem product_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (habc : a * b * c = 1) :
  (a - 1 + 1 / b) * (b - 1 + 1 / c) * (c - 1 + 1 / a) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_product_inequality_l1073_107301


namespace NUMINAMATH_CALUDE_rectangle_golden_ratio_l1073_107395

/-- A rectangle with sides x and y, where x > y, can be cut in half parallel to the longer side
    to produce scaled-down versions of the original if and only if x/y = √2 -/
theorem rectangle_golden_ratio (x y : ℝ) (h : x > y) (h' : x > 0) (h'' : y > 0) :
  (x / 2 : ℝ) / y = x / y ↔ x / y = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_rectangle_golden_ratio_l1073_107395


namespace NUMINAMATH_CALUDE_T_equals_x_to_fourth_l1073_107383

theorem T_equals_x_to_fourth (x : ℝ) : 
  (x - 2)^4 + 5*(x - 2)^3 + 10*(x - 2)^2 + 10*(x - 2) + 5 = x^4 := by
  sorry

end NUMINAMATH_CALUDE_T_equals_x_to_fourth_l1073_107383


namespace NUMINAMATH_CALUDE_molecular_weight_problem_l1073_107374

/-- Given that 3 moles of a compound have a molecular weight of 222,
    prove that the molecular weight of 1 mole of the compound is 74. -/
theorem molecular_weight_problem (moles : ℕ) (total_weight : ℝ) :
  moles = 3 →
  total_weight = 222 →
  total_weight / moles = 74 := by
  sorry

end NUMINAMATH_CALUDE_molecular_weight_problem_l1073_107374


namespace NUMINAMATH_CALUDE_set_equality_l1073_107315

theorem set_equality : {x : ℤ | -3 < x ∧ x < 1} = {-2, -1, 0} := by
  sorry

end NUMINAMATH_CALUDE_set_equality_l1073_107315


namespace NUMINAMATH_CALUDE_garden_length_l1073_107306

theorem garden_length (width : ℝ) (length : ℝ) (perimeter : ℝ) : 
  width > 0 → 
  length = 2 * width → 
  perimeter = 2 * length + 2 * width → 
  perimeter = 900 → 
  length = 300 := by
sorry

end NUMINAMATH_CALUDE_garden_length_l1073_107306


namespace NUMINAMATH_CALUDE_carolyn_sum_is_18_l1073_107308

/-- Represents the game state -/
structure GameState where
  remaining : List Nat
  carolyn_sum : Nat

/-- Represents a player's move -/
inductive Move
  | Remove (n : Nat)

/-- Applies Carolyn's move to the game state -/
def apply_carolyn_move (state : GameState) (move : Move) : GameState :=
  match move with
  | Move.Remove n =>
    { remaining := state.remaining.filter (· ≠ n),
      carolyn_sum := state.carolyn_sum + n }

/-- Applies Paul's move to the game state -/
def apply_paul_move (state : GameState) (move : List Move) : GameState :=
  match move with
  | [] => state
  | (Move.Remove n) :: rest =>
    apply_paul_move
      { remaining := state.remaining.filter (· ≠ n),
        carolyn_sum := state.carolyn_sum }
      rest

/-- Checks if a number has a divisor in the list -/
def has_divisor_in_list (n : Nat) (list : List Nat) : Bool :=
  list.any (fun m => m ≠ n && n % m == 0)

/-- Simulates the game -/
def play_game (initial_state : GameState) : Nat :=
  let state1 := apply_carolyn_move initial_state (Move.Remove 4)
  let state2 := apply_paul_move state1 [Move.Remove 1, Move.Remove 2]
  let state3 := apply_carolyn_move state2 (Move.Remove 6)
  let state4 := apply_paul_move state3 [Move.Remove 3]
  let state5 := apply_carolyn_move state4 (Move.Remove 8)
  let final_state := apply_paul_move state5 [Move.Remove 5, Move.Remove 7]
  final_state.carolyn_sum

theorem carolyn_sum_is_18 :
  let initial_state : GameState := { remaining := [1, 2, 3, 4, 5, 6, 7, 8], carolyn_sum := 0 }
  play_game initial_state = 18 := by
  sorry

end NUMINAMATH_CALUDE_carolyn_sum_is_18_l1073_107308


namespace NUMINAMATH_CALUDE_frog_escape_probability_l1073_107390

/-- Probability of frog escaping from pad N -/
noncomputable def P (N : ℕ) : ℝ :=
  sorry

/-- Total number of lily pads -/
def total_pads : ℕ := 15

/-- Starting pad for the frog -/
def start_pad : ℕ := 2

theorem frog_escape_probability :
  (∀ N, 0 < N → N < total_pads - 1 →
    P N = (N : ℝ) / total_pads * P (N - 1) + (1 - (N : ℝ) / total_pads) * P (N + 1)) →
  P 0 = 0 →
  P (total_pads - 1) = 1 →
  P start_pad = 163 / 377 :=
sorry

end NUMINAMATH_CALUDE_frog_escape_probability_l1073_107390


namespace NUMINAMATH_CALUDE_system_solution_l1073_107357

theorem system_solution : ∃ (x y : ℚ), 
  (7 * x - 50 * y = 3) ∧ 
  (3 * y - x = 5) ∧ 
  (x = -259/29) ∧ 
  (y = -38/29) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l1073_107357


namespace NUMINAMATH_CALUDE_add_12345_seconds_to_5_45_00_l1073_107363

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat

/-- Adds seconds to a given time -/
def addSeconds (t : Time) (s : Nat) : Time :=
  sorry

theorem add_12345_seconds_to_5_45_00 :
  addSeconds { hours := 5, minutes := 45, seconds := 0 } 12345 =
  { hours := 9, minutes := 10, seconds := 45 } :=
sorry

end NUMINAMATH_CALUDE_add_12345_seconds_to_5_45_00_l1073_107363


namespace NUMINAMATH_CALUDE_sum_of_powers_l1073_107305

theorem sum_of_powers (x y : ℂ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x^2 + x*y + y^2 = 0) : 
  (x / (x + y))^2013 + (y / (x + y))^2013 = -2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_l1073_107305


namespace NUMINAMATH_CALUDE_faye_crayons_count_l1073_107332

/-- The number of rows of crayons --/
def num_rows : ℕ := 15

/-- The number of crayons in each row --/
def crayons_per_row : ℕ := 42

/-- The total number of crayons --/
def total_crayons : ℕ := num_rows * crayons_per_row

theorem faye_crayons_count : total_crayons = 630 := by
  sorry

end NUMINAMATH_CALUDE_faye_crayons_count_l1073_107332


namespace NUMINAMATH_CALUDE_intersection_range_l1073_107330

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 ≤ 4}
def N (r : ℝ) : Set (ℝ × ℝ) := {p | (p.1 - 1)^2 + (p.2 - 1)^2 ≤ r^2}

-- State the theorem
theorem intersection_range (r : ℝ) (h1 : r > 0) (h2 : M ∩ N r = N r) :
  r ∈ Set.Ioo 0 (2 - Real.sqrt 2) := by
  sorry

-- Note: Set.Ioo represents an open interval (a, b)

end NUMINAMATH_CALUDE_intersection_range_l1073_107330


namespace NUMINAMATH_CALUDE_square_division_perimeter_counterexample_l1073_107382

theorem square_division_perimeter_counterexample :
  ∃ (s : ℚ), 
    s > 0 ∧ 
    (∃ (w h : ℚ), w > 0 ∧ h > 0 ∧ w + h = s ∧ (2 * (w + h)).isInt) ∧ 
    ¬(4 * s).isInt :=
by sorry

end NUMINAMATH_CALUDE_square_division_perimeter_counterexample_l1073_107382


namespace NUMINAMATH_CALUDE_max_term_binomial_expansion_l1073_107318

theorem max_term_binomial_expansion :
  let n : ℕ := 212
  let x : ℝ := Real.sqrt 11
  let term (k : ℕ) : ℝ := (n.choose k) * (x ^ k)
  ∃ k : ℕ, k = 163 ∧ ∀ j : ℕ, j ≠ k → j ≤ n → term k ≥ term j :=
by sorry

end NUMINAMATH_CALUDE_max_term_binomial_expansion_l1073_107318


namespace NUMINAMATH_CALUDE_probability_human_given_id_as_human_l1073_107334

-- Define the total population
def total_population : ℝ := 1000

-- Define the proportion of vampires and humans
def vampire_proportion : ℝ := 0.99
def human_proportion : ℝ := 1 - vampire_proportion

-- Define the correct identification rates
def vampire_correct_id_rate : ℝ := 0.9
def human_correct_id_rate : ℝ := 0.9

-- Define the number of vampires and humans
def num_vampires : ℝ := vampire_proportion * total_population
def num_humans : ℝ := human_proportion * total_population

-- Define the number of correctly and incorrectly identified vampires and humans
def vampires_id_as_vampires : ℝ := vampire_correct_id_rate * num_vampires
def vampires_id_as_humans : ℝ := (1 - vampire_correct_id_rate) * num_vampires
def humans_id_as_humans : ℝ := human_correct_id_rate * num_humans
def humans_id_as_vampires : ℝ := (1 - human_correct_id_rate) * num_humans

-- Define the total number of individuals identified as humans
def total_id_as_humans : ℝ := vampires_id_as_humans + humans_id_as_humans

-- Theorem statement
theorem probability_human_given_id_as_human :
  humans_id_as_humans / total_id_as_humans = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_probability_human_given_id_as_human_l1073_107334


namespace NUMINAMATH_CALUDE_money_duration_l1073_107341

def lawn_money : ℕ := 9
def weed_eating_money : ℕ := 18
def weekly_spending : ℕ := 3

theorem money_duration : 
  (lawn_money + weed_eating_money) / weekly_spending = 9 := by
  sorry

end NUMINAMATH_CALUDE_money_duration_l1073_107341


namespace NUMINAMATH_CALUDE_symmetry_xoz_plane_l1073_107322

-- Define a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the xOz plane
def xOzPlane : Set Point3D := {p : Point3D | p.y = 0}

-- Define symmetry with respect to the xOz plane
def symmetricPointXOZ (p : Point3D) : Point3D :=
  { x := p.x, y := -p.y, z := p.z }

theorem symmetry_xoz_plane :
  let P := Point3D.mk 3 1 5
  let Q := Point3D.mk 3 (-1) 5
  symmetricPointXOZ P = Q := by sorry

end NUMINAMATH_CALUDE_symmetry_xoz_plane_l1073_107322


namespace NUMINAMATH_CALUDE_base_conversion_440_to_octal_l1073_107333

theorem base_conversion_440_to_octal :
  (440 : ℕ) = 6 * 8^2 + 7 * 8^1 + 0 * 8^0 :=
by sorry

end NUMINAMATH_CALUDE_base_conversion_440_to_octal_l1073_107333


namespace NUMINAMATH_CALUDE_multiply_divide_sqrt_l1073_107387

theorem multiply_divide_sqrt (x y : ℝ) (hx : x = 1.4) (hx_neq_zero : x ≠ 0) :
  Real.sqrt ((x * y) / 5) = x → y = 7 := by
  sorry

end NUMINAMATH_CALUDE_multiply_divide_sqrt_l1073_107387


namespace NUMINAMATH_CALUDE_f_max_min_difference_l1073_107336

noncomputable def f (x : ℝ) := Real.exp (Real.sin x + Real.cos x) - (1/2) * Real.sin (2 * x)

theorem f_max_min_difference :
  (⨆ (x : ℝ), f x) - (⨅ (x : ℝ), f x) = Real.exp (Real.sqrt 2) - Real.exp (-Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_f_max_min_difference_l1073_107336


namespace NUMINAMATH_CALUDE_smallest_perfect_square_multiplier_l1073_107378

def y : ℕ := 2^5 * 3^5 * 4^5 * 5^5 * 6^5 * 7^5 * 8^5 * 9^5

theorem smallest_perfect_square_multiplier (k : ℕ) : 
  (∀ m : ℕ, m < 105 → ¬∃ n : ℕ, m * y = n^2) ∧ 
  (∃ n : ℕ, 105 * y = n^2) := by
  sorry

end NUMINAMATH_CALUDE_smallest_perfect_square_multiplier_l1073_107378


namespace NUMINAMATH_CALUDE_cookie_eating_contest_l1073_107329

theorem cookie_eating_contest (first_student second_student : ℚ) 
  (h1 : first_student = 5/6)
  (h2 : second_student = 7/12) :
  first_student - second_student = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_cookie_eating_contest_l1073_107329


namespace NUMINAMATH_CALUDE_jim_age_in_two_years_l1073_107384

theorem jim_age_in_two_years :
  let tom_age_five_years_ago : ℕ := 32
  let years_since_tom_age : ℕ := 5
  let years_to_past_reference : ℕ := 7
  let jim_age_difference : ℕ := 5
  let years_to_future : ℕ := 2

  let tom_current_age : ℕ := tom_age_five_years_ago + years_since_tom_age
  let tom_age_at_reference : ℕ := tom_current_age - years_to_past_reference
  let jim_age_at_reference : ℕ := (tom_age_at_reference / 2) + jim_age_difference
  let jim_current_age : ℕ := jim_age_at_reference + years_to_past_reference
  let jim_future_age : ℕ := jim_current_age + years_to_future

  jim_future_age = 29 := by sorry

end NUMINAMATH_CALUDE_jim_age_in_two_years_l1073_107384


namespace NUMINAMATH_CALUDE_expression_evaluation_l1073_107321

theorem expression_evaluation :
  let x : ℚ := -1/4
  (2*x + 1) * (2*x - 1) - (x - 2)^2 - 3*x^2 = -6 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1073_107321


namespace NUMINAMATH_CALUDE_three_balls_selected_l1073_107394

def num_balls : ℕ := 100
def prob_odd_first : ℚ := 2/3

theorem three_balls_selected 
  (h1 : num_balls = 100)
  (h2 : prob_odd_first = 2/3)
  (h3 : ∃ (odd_count even_count : ℕ), 
    odd_count = 2 ∧ even_count = 1 ∧ 
    odd_count + even_count = num_selected) :
  num_selected = 3 := by
  sorry

end NUMINAMATH_CALUDE_three_balls_selected_l1073_107394


namespace NUMINAMATH_CALUDE_proportional_relation_l1073_107381

/-- Given that x is directly proportional to y^2 and y is inversely proportional to z,
    prove that if x = 5 when z = 20, then x = 40/81 when z = 45. -/
theorem proportional_relation (x y z : ℝ) (c d : ℝ) (h1 : x = c * y^2) (h2 : y * z = d)
  (h3 : z = 20 → x = 5) : z = 45 → x = 40 / 81 := by
  sorry

end NUMINAMATH_CALUDE_proportional_relation_l1073_107381


namespace NUMINAMATH_CALUDE_square_diagonal_ratio_l1073_107310

theorem square_diagonal_ratio (a b : ℝ) (h : a > 0 ∧ b > 0) 
  (area_ratio : a^2 / b^2 = 49 / 64) : 
  (a * Real.sqrt 2) / (b * Real.sqrt 2) = 7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_square_diagonal_ratio_l1073_107310


namespace NUMINAMATH_CALUDE_sum_of_valid_numbers_l1073_107380

def digits : List Nat := [1, 3, 5]

def isValidNumber (n : Nat) : Bool :=
  n ≥ 100 ∧ n < 1000 ∧
  (n / 100) ∈ digits ∧
  ((n / 10) % 10) ∈ digits ∧
  (n % 10) ∈ digits ∧
  (n / 100) ≠ ((n / 10) % 10) ∧
  (n / 100) ≠ (n % 10) ∧
  ((n / 10) % 10) ≠ (n % 10)

def validNumbers : List Nat :=
  (List.range 1000).filter isValidNumber

theorem sum_of_valid_numbers :
  validNumbers.sum = 1998 := by sorry

end NUMINAMATH_CALUDE_sum_of_valid_numbers_l1073_107380


namespace NUMINAMATH_CALUDE_perpendicular_bisector_value_l1073_107323

/-- The perpendicular bisector of a line segment from (x₁, y₁) to (x₂, y₂) is defined as
    the line that passes through the midpoint of the segment and is perpendicular to it. --/
def is_perpendicular_bisector (a b c : ℝ) (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  let midpoint_x := (x₁ + x₂) / 2
  let midpoint_y := (y₁ + y₂) / 2
  -- The line ax + by + c = 0 passes through the midpoint
  a * midpoint_x + b * midpoint_y + c = 0 ∧
  -- The line is perpendicular to the segment
  a * (x₂ - x₁) + b * (y₂ - y₁) = 0

/-- Given that the line x + y = b is the perpendicular bisector of the line segment 
    from (0, 5) to (8, 10), prove that b = 11.5 --/
theorem perpendicular_bisector_value : 
  is_perpendicular_bisector 1 1 (-b) 0 5 8 10 → b = 11.5 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_value_l1073_107323


namespace NUMINAMATH_CALUDE_min_A_over_C_is_zero_l1073_107307

theorem min_A_over_C_is_zero (A C x : ℝ) (hA : A > 0) (hC : C > 0) (hx : x > 0)
  (hAx : x^2 + 1/x^2 = A) (hCx : x + 1/x = C) :
  ∀ ε > 0, ∃ A' C' x', A' > 0 ∧ C' > 0 ∧ x' > 0 ∧
    x'^2 + 1/x'^2 = A' ∧ x' + 1/x' = C' ∧ A' / C' < ε :=
sorry

end NUMINAMATH_CALUDE_min_A_over_C_is_zero_l1073_107307


namespace NUMINAMATH_CALUDE_cos_double_angle_special_case_l1073_107302

/-- Given a vector a = (cos α, 1/2) with magnitude √2/2, prove that cos(2α) = -1/2 -/
theorem cos_double_angle_special_case (α : ℝ) :
  let a : ℝ × ℝ := (Real.cos α, 1/2)
  (a.1^2 + a.2^2 = 1/2) →
  Real.cos (2 * α) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_double_angle_special_case_l1073_107302


namespace NUMINAMATH_CALUDE_diploma_monthly_pay_l1073_107312

/-- The annual salary of a person with a degree -/
def annual_salary_degree : ℕ := 144000

/-- The ratio of salary between a person with a degree and a diploma holder -/
def salary_ratio : ℕ := 3

/-- The number of months in a year -/
def months_per_year : ℕ := 12

/-- The monthly pay for a person holding a diploma certificate -/
def monthly_pay_diploma : ℚ := annual_salary_degree / (salary_ratio * months_per_year)

theorem diploma_monthly_pay :
  monthly_pay_diploma = 4000 := by sorry

end NUMINAMATH_CALUDE_diploma_monthly_pay_l1073_107312


namespace NUMINAMATH_CALUDE_average_favorable_draws_l1073_107366

def lottery_size : ℕ := 90
def draw_size : ℕ := 5

def favorable_draws : ℕ :=
  (86^2 * 85) / 2 + 87 * 85 + 86

def total_draws : ℕ :=
  lottery_size * (lottery_size - 1) * (lottery_size - 2) * (lottery_size - 3) * (lottery_size - 4) / 120

theorem average_favorable_draws :
  (total_draws : ℚ) / favorable_draws = 5874 / 43 := by sorry

end NUMINAMATH_CALUDE_average_favorable_draws_l1073_107366


namespace NUMINAMATH_CALUDE_selling_price_calculation_l1073_107356

def cost_price : ℝ := 975
def profit_percentage : ℝ := 20

theorem selling_price_calculation :
  let profit := (profit_percentage / 100) * cost_price
  let selling_price := cost_price + profit
  selling_price = 1170 := by sorry

end NUMINAMATH_CALUDE_selling_price_calculation_l1073_107356


namespace NUMINAMATH_CALUDE_non_shaded_perimeter_l1073_107342

/-- Given a rectangle with dimensions 12 × 10 inches, containing an inner rectangle
    of 6 × 2 inches, and a shaded area of 116 square inches, prove that the
    perimeter of the non-shaded region is 10 inches. -/
theorem non_shaded_perimeter (outer_length outer_width inner_length inner_width shaded_area : ℝ)
  (h_outer_length : outer_length = 12)
  (h_outer_width : outer_width = 10)
  (h_inner_length : inner_length = 6)
  (h_inner_width : inner_width = 2)
  (h_shaded_area : shaded_area = 116)
  (h_right_angles : ∀ angle, angle = 90) :
  let total_area := outer_length * outer_width
  let inner_area := inner_length * inner_width
  let non_shaded_area := total_area - shaded_area
  let non_shaded_length := 4
  let non_shaded_width := 1
  2 * (non_shaded_length + non_shaded_width) = 10 := by
    sorry

end NUMINAMATH_CALUDE_non_shaded_perimeter_l1073_107342


namespace NUMINAMATH_CALUDE_opposite_of_three_l1073_107337

theorem opposite_of_three : 
  (-(3 : ℤ) : ℤ) = -3 := by sorry

end NUMINAMATH_CALUDE_opposite_of_three_l1073_107337


namespace NUMINAMATH_CALUDE_divisible_by_four_and_six_percentage_l1073_107365

theorem divisible_by_four_and_six_percentage (n : ℕ) : 
  (↑(Finset.filter (fun x => x % 4 = 0 ∧ x % 6 = 0) (Finset.range (n + 1))).card / n) * 100 = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_divisible_by_four_and_six_percentage_l1073_107365


namespace NUMINAMATH_CALUDE_opposite_numbers_l1073_107367

theorem opposite_numbers : -4^2 = -((- 4)^2) := by sorry

end NUMINAMATH_CALUDE_opposite_numbers_l1073_107367


namespace NUMINAMATH_CALUDE_certain_number_proof_l1073_107393

theorem certain_number_proof :
  let first_number : ℝ := 15
  let certain_number : ℝ := (0.4 * first_number) - (0.8 * 5)
  certain_number = 2 := by
sorry

end NUMINAMATH_CALUDE_certain_number_proof_l1073_107393


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1073_107311

/-- The eccentricity of the hyperbola (x²/4) - (y²/2) = 1 is √6/2 -/
theorem hyperbola_eccentricity : 
  let C : Set (ℝ × ℝ) := {(x, y) | x^2/4 - y^2/2 = 1}
  ∃ e : ℝ, e = Real.sqrt 6 / 2 ∧ 
    ∀ (x y : ℝ), (x, y) ∈ C → 
      e = Real.sqrt ((x^2/4 + y^2/2) / (x^2/4)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1073_107311


namespace NUMINAMATH_CALUDE_problem_statement_l1073_107335

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  (∀ m : ℝ, (∀ a b : ℝ, a > 0 → b > 0 → a + b = 2 → a * b < m / 2) ↔ m > 2) ∧
  (∀ x : ℝ, (∀ a b : ℝ, a > 0 → b > 0 → a + b = 2 → 9 / a + 1 / b ≥ |x - 1| + |x + 2|) ↔ -9/2 ≤ x ∧ x ≤ 7/2) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l1073_107335


namespace NUMINAMATH_CALUDE_derangement_probability_five_l1073_107373

/-- The number of derangements of n elements -/
def derangement (n : ℕ) : ℕ := sorry

/-- The probability of a derangement of n elements -/
def derangementProbability (n : ℕ) : ℚ :=
  (derangement n : ℚ) / (Nat.factorial n)

theorem derangement_probability_five :
  derangementProbability 5 = 11 / 30 := by sorry

end NUMINAMATH_CALUDE_derangement_probability_five_l1073_107373


namespace NUMINAMATH_CALUDE_ceiling_neg_sqrt_100_over_9_l1073_107354

theorem ceiling_neg_sqrt_100_over_9 : ⌈-Real.sqrt (100 / 9)⌉ = -3 := by sorry

end NUMINAMATH_CALUDE_ceiling_neg_sqrt_100_over_9_l1073_107354


namespace NUMINAMATH_CALUDE_fraction_multiplication_l1073_107350

theorem fraction_multiplication : (1 : ℚ) / 3 * (3 : ℚ) / 5 * (5 : ℚ) / 7 = (1 : ℚ) / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l1073_107350


namespace NUMINAMATH_CALUDE_simplify_polynomial_l1073_107349

theorem simplify_polynomial (y : ℝ) : 
  2*y*(4*y^3 - 3*y + 5) - 4*(y^3 - 3*y^2 + 4*y - 6) = 
  8*y^4 - 4*y^3 + 6*y^2 - 6*y + 24 := by
sorry

end NUMINAMATH_CALUDE_simplify_polynomial_l1073_107349


namespace NUMINAMATH_CALUDE_M_superset_P_l1073_107300

-- Define the sets M and P
def M : Set ℝ := {y | ∃ x, y = x^2 - 4}
def P : Set ℝ := {y | |y - 3| ≤ 1}

-- State the theorem
theorem M_superset_P : M ⊇ P := by
  sorry

end NUMINAMATH_CALUDE_M_superset_P_l1073_107300


namespace NUMINAMATH_CALUDE_max_intersections_12_6_l1073_107359

/-- The maximum number of intersection points in the first quadrant 
    given the number of points on x and y axes -/
def max_intersections (x_points y_points : ℕ) : ℕ :=
  (x_points.choose 2) * (y_points.choose 2)

/-- Theorem stating the maximum number of intersections for 12 x-axis points
    and 6 y-axis points -/
theorem max_intersections_12_6 :
  max_intersections 12 6 = 990 := by
  sorry

#eval max_intersections 12 6

end NUMINAMATH_CALUDE_max_intersections_12_6_l1073_107359


namespace NUMINAMATH_CALUDE_greatest_three_digit_divisible_by_3_5_6_l1073_107320

theorem greatest_three_digit_divisible_by_3_5_6 : ∃ n : ℕ, 
  n < 1000 ∧ 
  n ≥ 100 ∧ 
  n % 3 = 0 ∧ 
  n % 5 = 0 ∧ 
  n % 6 = 0 ∧
  ∀ m : ℕ, m < 1000 ∧ m ≥ 100 ∧ m % 3 = 0 ∧ m % 5 = 0 ∧ m % 6 = 0 → m ≤ n :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_greatest_three_digit_divisible_by_3_5_6_l1073_107320


namespace NUMINAMATH_CALUDE_genevieve_errors_fixed_l1073_107344

/-- Represents the number of errors fixed by a programmer -/
def errors_fixed (total_lines : ℕ) (debug_interval : ℕ) (errors_per_debug : ℕ) : ℕ :=
  (total_lines / debug_interval) * errors_per_debug

/-- Theorem stating the number of errors fixed by Genevieve -/
theorem genevieve_errors_fixed :
  errors_fixed 4300 100 3 = 129 := by
  sorry

end NUMINAMATH_CALUDE_genevieve_errors_fixed_l1073_107344


namespace NUMINAMATH_CALUDE_green_shirt_pairs_l1073_107328

theorem green_shirt_pairs (total_students : ℕ) (red_students : ℕ) (green_students : ℕ) 
  (total_pairs : ℕ) (red_red_pairs : ℕ) :
  total_students = 140 →
  red_students = 60 →
  green_students = 80 →
  total_pairs = 70 →
  red_red_pairs = 10 →
  ∃ (green_green_pairs : ℕ), green_green_pairs = 20 ∧ 
    green_green_pairs + red_red_pairs + (total_pairs - green_green_pairs - red_red_pairs) = total_pairs :=
by sorry

end NUMINAMATH_CALUDE_green_shirt_pairs_l1073_107328


namespace NUMINAMATH_CALUDE_book_purchase_total_price_l1073_107345

/-- Calculates the total price of books given the following conditions:
  * Total number of books
  * Number of math books
  * Price of a math book
  * Price of a history book
-/
def total_price (total_books : ℕ) (math_books : ℕ) (math_price : ℕ) (history_price : ℕ) : ℕ :=
  math_books * math_price + (total_books - math_books) * history_price

/-- Theorem stating that given the specific conditions in the problem,
    the total price of books is $390. -/
theorem book_purchase_total_price :
  total_price 80 10 4 5 = 390 := by
  sorry

end NUMINAMATH_CALUDE_book_purchase_total_price_l1073_107345


namespace NUMINAMATH_CALUDE_young_worker_proportion_is_three_fifths_l1073_107324

/-- The proportion of young workers in a steel works -/
def young_worker_proportion : ℚ := 3/5

/-- The statement that the proportion of young workers is three-fifths -/
theorem young_worker_proportion_is_three_fifths : 
  young_worker_proportion = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_young_worker_proportion_is_three_fifths_l1073_107324


namespace NUMINAMATH_CALUDE_count_integer_pairs_l1073_107316

theorem count_integer_pairs : ∃ (count : ℕ),
  count = (Finset.filter (fun p : ℕ × ℕ => 
    let m := p.1
    let n := p.2
    1 ≤ m ∧ m ≤ 2012 ∧ 
    (5 : ℝ)^n < (2 : ℝ)^m ∧ 
    (2 : ℝ)^m < (2 : ℝ)^(m+2) ∧ 
    (2 : ℝ)^(m+2) < (5 : ℝ)^(n+1))
  (Finset.product (Finset.range 2013) (Finset.range (2014 + 1)))).card ∧
  (2 : ℝ)^2013 < (5 : ℝ)^867 ∧ (5 : ℝ)^867 < (2 : ℝ)^2014 ∧
  count = 279 := by
  sorry

end NUMINAMATH_CALUDE_count_integer_pairs_l1073_107316


namespace NUMINAMATH_CALUDE_square_rectangle_area_relation_l1073_107304

theorem square_rectangle_area_relation : 
  ∃ (x₁ x₂ : ℝ), 
    (∀ x : ℝ, 3 * (x - 4)^2 = (x - 5) * (x + 6) ↔ x = x₁ ∨ x = x₂) ∧ 
    x₁ + x₂ = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_square_rectangle_area_relation_l1073_107304


namespace NUMINAMATH_CALUDE_gcd_294_84_l1073_107360

theorem gcd_294_84 : Nat.gcd 294 84 = 42 := by
  sorry

end NUMINAMATH_CALUDE_gcd_294_84_l1073_107360


namespace NUMINAMATH_CALUDE_equation_solution_l1073_107338

theorem equation_solution : ∃ N : ℝ,
  (∃ e₁ e₂ : ℝ, 2 * |2 - e₁| = N ∧ 2 * |2 - e₂| = N ∧ e₁ + e₂ = 4) →
  N = 0 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1073_107338


namespace NUMINAMATH_CALUDE_quadratic_roots_problem_l1073_107396

theorem quadratic_roots_problem (x₁ x₂ b : ℝ) : 
  (∀ x, x^2 + b*x + 4 = 0 ↔ x = x₁ ∨ x = x₂) →
  x₁ - x₁*x₂ + x₂ = 2 →
  b = -6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_problem_l1073_107396


namespace NUMINAMATH_CALUDE_mr_green_potato_yield_l1073_107397

/-- Represents the dimensions of a rectangular garden in steps -/
structure GardenDimensions where
  length : ℕ
  width : ℕ

/-- Calculates the expected potato yield from a rectangular garden -/
def expected_potato_yield (garden : GardenDimensions) (step_length : ℝ) (yield_per_sqft : ℝ) : ℝ :=
  (garden.length : ℝ) * step_length * (garden.width : ℝ) * step_length * yield_per_sqft

/-- Theorem stating the expected potato yield for Mr. Green's garden -/
theorem mr_green_potato_yield :
  let garden := GardenDimensions.mk 18 25
  let step_length := 2.5
  let yield_per_sqft := 0.5
  expected_potato_yield garden step_length yield_per_sqft = 1406.25 := by
  sorry


end NUMINAMATH_CALUDE_mr_green_potato_yield_l1073_107397


namespace NUMINAMATH_CALUDE_exists_48_good_perfect_square_l1073_107340

/-- A number is k-good if it can be split into two parts y and z where y = k * z -/
def is_k_good (k : ℕ) (n : ℕ) : Prop :=
  ∃ y z : ℕ, y * (10^(Nat.log 10 z + 1)) + z = n ∧ y = k * z

/-- The main theorem: there exists a 48-good perfect square -/
theorem exists_48_good_perfect_square : ∃ n : ℕ, is_k_good 48 n ∧ ∃ m : ℕ, n = m^2 :=
sorry

end NUMINAMATH_CALUDE_exists_48_good_perfect_square_l1073_107340


namespace NUMINAMATH_CALUDE_equation_solution_l1073_107314

theorem equation_solution : ∃ (x y : ℝ), 
  (1 / 6 + 6 / x = 14 / x + 1 / 14 + y) ∧ (x = 84) ∧ (y = 0) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1073_107314


namespace NUMINAMATH_CALUDE_locus_of_A_is_hyperbola_l1073_107377

/-- Triangle ABC with special properties -/
structure SpecialTriangle where
  -- Points of the triangle
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  -- H is the orthocenter
  H : ℝ × ℝ
  -- G is the centroid
  G : ℝ × ℝ
  -- B and C are fixed points
  B_fixed : B.1 = -a ∧ B.2 = 0
  C_fixed : C.1 = a ∧ C.2 = 0
  -- Midpoint of HG lies on BC
  HG_midpoint_on_BC : ∃ m : ℝ, (H.1 + G.1) / 2 = m ∧ (H.2 + G.2) / 2 = 0
  -- G is the centroid
  G_is_centroid : G.1 = (A.1 + B.1 + C.1) / 3 ∧ G.2 = (A.2 + B.2 + C.2) / 3
  -- H is the orthocenter
  H_is_orthocenter : (A.1 - B.1) * (H.1 - C.1) + (A.2 - B.2) * (H.2 - C.2) = 0 ∧
                     (B.1 - C.1) * (H.1 - A.1) + (B.2 - C.2) * (H.2 - A.2) = 0

/-- The locus of A in a special triangle is a hyperbola -/
theorem locus_of_A_is_hyperbola (t : SpecialTriangle) : 
  t.A.1^2 - t.A.2^2/3 = a^2 := by sorry

end NUMINAMATH_CALUDE_locus_of_A_is_hyperbola_l1073_107377


namespace NUMINAMATH_CALUDE_investment_rate_is_five_percent_l1073_107376

/-- Represents an investment account --/
structure Account where
  balance : ℝ
  rate : ℝ

/-- Calculates the interest earned on an account in one year --/
def interest (a : Account) : ℝ := a.balance * a.rate

/-- Represents the investment scenario --/
structure InvestmentScenario where
  account1 : Account
  account2 : Account
  totalInterest : ℝ

/-- The given investment scenario --/
def scenario : InvestmentScenario where
  account1 := { balance := 8000, rate := 0.05 }
  account2 := { balance := 2000, rate := 0.06 }
  totalInterest := 520

/-- Theorem stating that the given scenario satisfies all conditions --/
theorem investment_rate_is_five_percent : 
  scenario.account1.balance = 4 * scenario.account2.balance ∧
  scenario.account2.rate = 0.06 ∧
  interest scenario.account1 + interest scenario.account2 = scenario.totalInterest ∧
  scenario.account1.rate = 0.05 := by
  sorry

#check investment_rate_is_five_percent

end NUMINAMATH_CALUDE_investment_rate_is_five_percent_l1073_107376


namespace NUMINAMATH_CALUDE_function_properties_l1073_107317

-- Define the function f
noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x + φ)

-- State the theorem
theorem function_properties
  (ω φ : ℝ)
  (h_ω : ω > 0)
  (h_φ : 0 < φ ∧ φ < π)
  (h_sym1 : ∀ x, f ω φ x = f ω φ ((2 * π) / 3 - x))
  (h_sym2 : ∀ x, f ω φ x = -f ω φ (π - x))
  (h_period : ∃ T > π / 2, ∀ x, f ω φ (x + T) = f ω φ x) :
  (∀ x, f ω φ (x + (2 * π) / 3) = f ω φ x) ∧
  (∀ x, f ω φ x = f ω φ (-x)) :=
sorry

end NUMINAMATH_CALUDE_function_properties_l1073_107317


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l1073_107385

/-- For a quadratic equation px^2 - 16x + 5 = 0, where p is nonzero,
    the equation has only one solution if and only if p = 64/5 -/
theorem quadratic_one_solution (p : ℝ) (hp : p ≠ 0) :
  (∃! x, p * x^2 - 16 * x + 5 = 0) ↔ p = 64/5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l1073_107385


namespace NUMINAMATH_CALUDE_candidate_vote_percentage_l1073_107313

theorem candidate_vote_percentage 
  (total_votes : ℕ) 
  (invalid_percentage : ℚ) 
  (candidate_valid_votes : ℕ) 
  (h1 : total_votes = 560000) 
  (h2 : invalid_percentage = 15/100) 
  (h3 : candidate_valid_votes = 404600) : 
  (candidate_valid_votes : ℚ) / ((1 - invalid_percentage) * total_votes) = 85/100 := by
sorry

end NUMINAMATH_CALUDE_candidate_vote_percentage_l1073_107313


namespace NUMINAMATH_CALUDE_pure_imaginary_fraction_l1073_107348

theorem pure_imaginary_fraction (α : ℝ) : 
  (∃ (y : ℝ), (α + 3 * Complex.I) / (1 + 2 * Complex.I) = y * Complex.I) → α = -6 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_fraction_l1073_107348


namespace NUMINAMATH_CALUDE_expected_cases_correct_l1073_107327

/-- The probability of an American having the disease -/
def disease_probability : ℚ := 1 / 3

/-- The total number of Americans in the sample -/
def sample_size : ℕ := 450

/-- The expected number of Americans with the disease in the sample -/
def expected_cases : ℕ := 150

/-- Theorem stating that the expected number of cases is correct -/
theorem expected_cases_correct : 
  ↑expected_cases = ↑sample_size * disease_probability := by sorry

end NUMINAMATH_CALUDE_expected_cases_correct_l1073_107327


namespace NUMINAMATH_CALUDE_imaginary_part_of_one_minus_i_squared_l1073_107339

theorem imaginary_part_of_one_minus_i_squared :
  Complex.im ((1 - Complex.I) ^ 2) = -2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_one_minus_i_squared_l1073_107339


namespace NUMINAMATH_CALUDE_carbonated_water_percentage_l1073_107358

/-- Represents a solution with percentages of lemonade and carbonated water -/
structure Solution where
  lemonade : ℝ
  carbonated : ℝ
  sum_to_one : lemonade + carbonated = 1

/-- Represents a mixture of two solutions -/
structure Mixture where
  solution1 : Solution
  solution2 : Solution
  proportion1 : ℝ
  proportion2 : ℝ
  sum_to_one : proportion1 + proportion2 = 1

theorem carbonated_water_percentage
  (sol1 : Solution)
  (sol2 : Solution)
  (mix : Mixture)
  (h1 : sol1.carbonated = 0.8)
  (h2 : sol2.lemonade = 0.45)
  (h3 : mix.solution1 = sol1)
  (h4 : mix.solution2 = sol2)
  (h5 : mix.proportion1 = 0.5)
  (h6 : mix.proportion2 = 0.5)
  (h7 : mix.proportion1 * sol1.carbonated + mix.proportion2 * sol2.carbonated = 0.675) :
  sol2.carbonated = 0.55 := by
  sorry


end NUMINAMATH_CALUDE_carbonated_water_percentage_l1073_107358


namespace NUMINAMATH_CALUDE_exists_n_plus_sum_of_digits_eq_125_l1073_107326

/-- Sum of digits function -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- Theorem stating the existence of a natural number n such that n + S(n) = 125 -/
theorem exists_n_plus_sum_of_digits_eq_125 :
  ∃ n : ℕ, n + sumOfDigits n = 125 ∧ n = 121 :=
sorry

end NUMINAMATH_CALUDE_exists_n_plus_sum_of_digits_eq_125_l1073_107326


namespace NUMINAMATH_CALUDE_derek_age_is_42_l1073_107392

-- Define the ages as natural numbers
def anne_age : ℕ := 36
def brianna_age : ℕ := (2 * anne_age) / 3
def caitlin_age : ℕ := brianna_age - 3
def derek_age : ℕ := 2 * caitlin_age

-- Theorem to prove Derek's age is 42
theorem derek_age_is_42 : derek_age = 42 := by
  sorry

end NUMINAMATH_CALUDE_derek_age_is_42_l1073_107392


namespace NUMINAMATH_CALUDE_inscribed_sphere_volume_l1073_107309

/-- A right circular cone with a sphere inscribed inside it. -/
structure ConeWithSphere where
  /-- The diameter of the cone's base in inches. -/
  base_diameter : ℝ
  /-- The vertex angle of the cross-section triangle in degrees. -/
  vertex_angle : ℝ
  /-- The sphere is tangent to the sides of the cone and sits on the table. -/
  sphere_tangent : Bool

/-- The volume of the inscribed sphere in cubic inches. -/
def sphere_volume (cone : ConeWithSphere) : ℝ := sorry

/-- Theorem stating the volume of the inscribed sphere for the given conditions. -/
theorem inscribed_sphere_volume (cone : ConeWithSphere) 
  (h1 : cone.base_diameter = 24)
  (h2 : cone.vertex_angle = 90)
  (h3 : cone.sphere_tangent = true) :
  sphere_volume cone = 576 * Real.sqrt 2 * Real.pi := by sorry

end NUMINAMATH_CALUDE_inscribed_sphere_volume_l1073_107309


namespace NUMINAMATH_CALUDE_smallest_resolvable_debt_is_correct_l1073_107364

/-- The value of a pig in dollars -/
def pig_value : ℕ := 400

/-- The value of a goat in dollars -/
def goat_value : ℕ := 240

/-- A debt is resolvable if it can be expressed as a linear combination of pig and goat values -/
def is_resolvable (debt : ℕ) : Prop :=
  ∃ (p g : ℤ), debt = p * pig_value + g * goat_value

/-- The smallest positive resolvable debt -/
def smallest_resolvable_debt : ℕ := 80

theorem smallest_resolvable_debt_is_correct :
  (is_resolvable smallest_resolvable_debt) ∧
  (∀ d : ℕ, 0 < d → d < smallest_resolvable_debt → ¬(is_resolvable d)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_resolvable_debt_is_correct_l1073_107364


namespace NUMINAMATH_CALUDE_fraction_equality_l1073_107355

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (4 * x + y) / (x - 4 * y) = -3) : 
  (x + 3 * y) / (3 * x - y) = 16 / 13 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l1073_107355


namespace NUMINAMATH_CALUDE_min_steps_even_correct_min_steps_odd_correct_l1073_107353

-- Define the stone arrangement
structure StoneArrangement where
  k : Nat
  n : Nat
  stones : List Nat

-- Define a step
def step (arrangement : StoneArrangement) : StoneArrangement := sorry

-- Define the minimum number of steps for even n
def min_steps_even (k : Nat) (n : Nat) : Nat :=
  (n^2 * k * (k-1)) / 4

-- Define the minimum number of steps for odd n and k = 3
def min_steps_odd (n : Nat) : Nat :=
  let q := (n - 1) / 2
  n^2 + 2 * q * (q + 1)

-- Theorem for even n
theorem min_steps_even_correct (k n : Nat) (h1 : k ≥ 2) (h2 : n % 2 = 0) :
  ∀ (arrangement : StoneArrangement),
    arrangement.k = k ∧ arrangement.n = n →
    ∃ (m : Nat), m ≤ min_steps_even k n ∧
      ∃ (final_arrangement : StoneArrangement),
        final_arrangement = (step^[m] arrangement) ∧
        -- The n stones of the same color are together in final_arrangement
        sorry := by sorry

-- Theorem for odd n and k = 3
theorem min_steps_odd_correct (n : Nat) (h1 : n % 2 = 1) :
  ∀ (arrangement : StoneArrangement),
    arrangement.k = 3 ∧ arrangement.n = n →
    ∃ (m : Nat), m ≤ min_steps_odd n ∧
      ∃ (final_arrangement : StoneArrangement),
        final_arrangement = (step^[m] arrangement) ∧
        -- The n stones of the same color are together in final_arrangement
        sorry := by sorry

end NUMINAMATH_CALUDE_min_steps_even_correct_min_steps_odd_correct_l1073_107353


namespace NUMINAMATH_CALUDE_place_four_men_five_women_l1073_107379

/-- The number of ways to place men and women into groups -/
def placeInGroups (numMen numWomen : ℕ) : ℕ :=
  let twoGroup := numMen * numWomen
  let threeGroup := (numMen - 1) * (numWomen.choose 2)
  let fourGroup := 1  -- As all remaining people form this group
  twoGroup * threeGroup * fourGroup

/-- Theorem stating the number of ways to place 4 men and 5 women into specific groups -/
theorem place_four_men_five_women :
  placeInGroups 4 5 = 360 := by
  sorry

#eval placeInGroups 4 5

end NUMINAMATH_CALUDE_place_four_men_five_women_l1073_107379


namespace NUMINAMATH_CALUDE_factorial_ratio_l1073_107319

def factorial (n : ℕ) : ℕ := 
  if n = 0 then 1 else n * factorial (n - 1)

theorem factorial_ratio (n : ℕ) (h : n ≥ 2) : 
  (factorial n) / (factorial (n - 2)) = n * (n - 1) := by
  sorry

#eval factorial 100 / factorial 98  -- Should output 9900

end NUMINAMATH_CALUDE_factorial_ratio_l1073_107319


namespace NUMINAMATH_CALUDE_solve_equation_l1073_107343

theorem solve_equation (n : ℤ) : (n + 1999) / 2 = -1 → n = -2001 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1073_107343


namespace NUMINAMATH_CALUDE_lunch_percentage_l1073_107388

theorem lunch_percentage (total_students : ℕ) (total_students_pos : total_students > 0) :
  let boys := (6 : ℚ) / 10 * total_students
  let girls := (4 : ℚ) / 10 * total_students
  let boys_lunch := (60 : ℚ) / 100 * boys
  let girls_lunch := (40 : ℚ) / 100 * girls
  let total_lunch := boys_lunch + girls_lunch
  (total_lunch / total_students) * 100 = 52 := by
  sorry

end NUMINAMATH_CALUDE_lunch_percentage_l1073_107388


namespace NUMINAMATH_CALUDE_apple_redistribution_theorem_l1073_107398

/-- Represents the state of apples in baskets -/
structure AppleBaskets where
  total_apples : ℕ
  baskets : List ℕ
  deriving Repr

/-- Checks if all non-empty baskets have the same number of apples -/
def all_equal (ab : AppleBaskets) : Prop :=
  let non_empty := ab.baskets.filter (· > 0)
  non_empty.all (· = non_empty.head!)

/-- Checks if the total number of apples is at least 100 -/
def at_least_100 (ab : AppleBaskets) : Prop :=
  ab.total_apples ≥ 100

/-- Represents a valid redistribution of apples -/
def is_valid_redistribution (initial final : AppleBaskets) : Prop :=
  final.total_apples ≤ initial.total_apples ∧
  final.baskets.length ≤ initial.baskets.length

/-- The main theorem to prove -/
theorem apple_redistribution_theorem (initial : AppleBaskets) :
  initial.total_apples = 2000 →
  ∃ (final : AppleBaskets), 
    is_valid_redistribution initial final ∧
    all_equal final ∧
    at_least_100 final := by
  sorry

end NUMINAMATH_CALUDE_apple_redistribution_theorem_l1073_107398


namespace NUMINAMATH_CALUDE_books_sold_l1073_107346

/-- Given Paul's initial and final number of books, prove that he sold 42 books. -/
theorem books_sold (initial_books final_books : ℕ) 
  (h1 : initial_books = 108) 
  (h2 : final_books = 66) : 
  initial_books - final_books = 42 := by
  sorry

#check books_sold

end NUMINAMATH_CALUDE_books_sold_l1073_107346


namespace NUMINAMATH_CALUDE_parallelogram_area_theorem_l1073_107371

/-- The area of a parallelogram with given base and height -/
def parallelogram_area (base height : Real) : Real :=
  base * height

/-- The inclination of the parallelogram -/
def inclination : Real := 6

theorem parallelogram_area_theorem (base height : Real) 
  (h_base : base = 20) 
  (h_height : height = 4) :
  parallelogram_area base height = 80 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_theorem_l1073_107371


namespace NUMINAMATH_CALUDE_exists_terrorist_with_eleven_raids_l1073_107375

/-- Represents a terrorist in the band -/
structure Terrorist : Type :=
  (id : Nat)

/-- Represents a raid -/
structure Raid : Type :=
  (id : Nat)

/-- Represents the participation of a terrorist in a raid -/
def Participation : Type := Terrorist → Raid → Prop

/-- The total number of terrorists in the band -/
def num_terrorists : Nat := 101

/-- Axiom: Each pair of terrorists has met exactly once in a raid -/
axiom met_once (p : Participation) (t1 t2 : Terrorist) :
  t1 ≠ t2 → ∃! r : Raid, p t1 r ∧ p t2 r

/-- Axiom: No two terrorists have participated in more than one raid together -/
axiom no_multiple_raids (p : Participation) (t1 t2 : Terrorist) (r1 r2 : Raid) :
  t1 ≠ t2 → p t1 r1 → p t2 r1 → p t1 r2 → p t2 r2 → r1 = r2

/-- Theorem: There exists a terrorist who participated in at least 11 different raids -/
theorem exists_terrorist_with_eleven_raids (p : Participation) :
  ∃ t : Terrorist, ∃ (raids : Finset Raid), raids.card ≥ 11 ∧ ∀ r ∈ raids, p t r :=
sorry

end NUMINAMATH_CALUDE_exists_terrorist_with_eleven_raids_l1073_107375


namespace NUMINAMATH_CALUDE_detergent_water_ratio_change_l1073_107370

-- Define the original ratio
def original_ratio : Fin 3 → ℚ
  | 0 => 2  -- bleach
  | 1 => 40 -- detergent
  | 2 => 100 -- water

-- Define the altered ratio
def altered_ratio : Fin 3 → ℚ
  | 0 => 6  -- bleach (tripled)
  | 1 => 40 -- detergent
  | 2 => 200 -- water

-- Theorem to prove
theorem detergent_water_ratio_change :
  (altered_ratio 1 / altered_ratio 2) / (original_ratio 1 / original_ratio 2) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_detergent_water_ratio_change_l1073_107370


namespace NUMINAMATH_CALUDE_inequality_proof_l1073_107303

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^2 + y^3 ≥ x^3 + y^4) : x^3 + y^3 ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1073_107303


namespace NUMINAMATH_CALUDE_soccer_balls_count_l1073_107386

/-- The cost of a football in dollars -/
def football_cost : ℝ := 35

/-- The cost of a soccer ball in dollars -/
def soccer_ball_cost : ℝ := 50

/-- The cost of 2 footballs and some soccer balls in dollars -/
def first_set_cost : ℝ := 220

/-- The cost of 3 footballs and 1 soccer ball in dollars -/
def second_set_cost : ℝ := 155

/-- The number of soccer balls in the second set -/
def soccer_balls_in_second_set : ℕ := 1

theorem soccer_balls_count : 
  2 * football_cost + soccer_balls_in_second_set * soccer_ball_cost = first_set_cost ∧
  3 * football_cost + soccer_ball_cost = second_set_cost →
  soccer_balls_in_second_set = 1 := by
  sorry

end NUMINAMATH_CALUDE_soccer_balls_count_l1073_107386


namespace NUMINAMATH_CALUDE_no_perfect_square_with_conditions_l1073_107351

def is_nine_digit (n : ℕ) : Prop := 10^8 ≤ n ∧ n < 10^9

def contains_all_nonzero_digits (n : ℕ) : Prop :=
  ∀ d : ℕ, 1 ≤ d ∧ d ≤ 9 → ∃ k : ℕ, n / 10^k % 10 = d

def last_digit_is_five (n : ℕ) : Prop := n % 10 = 5

theorem no_perfect_square_with_conditions :
  ¬ ∃ n : ℕ, is_nine_digit n ∧ contains_all_nonzero_digits n ∧ last_digit_is_five n ∧ ∃ m : ℕ, n = m^2 := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_square_with_conditions_l1073_107351


namespace NUMINAMATH_CALUDE_max_third_side_of_triangle_l1073_107368

theorem max_third_side_of_triangle (a b c : ℝ) : 
  a = 7 → b = 10 → c > 0 → a + b + c ≤ 30 → 
  a + b > c → a + c > b → b + c > a → 
  ∀ n : ℕ, (n : ℝ) > c → n ≤ 13 :=
by sorry

end NUMINAMATH_CALUDE_max_third_side_of_triangle_l1073_107368

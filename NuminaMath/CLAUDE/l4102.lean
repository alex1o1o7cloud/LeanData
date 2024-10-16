import Mathlib

namespace NUMINAMATH_CALUDE_sue_answer_formula_l4102_410278

/-- Given Ben's initial number, calculate Sue's final answer -/
def sueAnswer (x : ℕ) : ℕ :=
  let benResult := 2 * (2 * x + 1)
  2 * (benResult - 1)

/-- Theorem: Sue's answer is always 4x + 2, where x is Ben's initial number -/
theorem sue_answer_formula (x : ℕ) : sueAnswer x = 4 * x + 2 := by
  sorry

#eval sueAnswer 8  -- Should output 66

end NUMINAMATH_CALUDE_sue_answer_formula_l4102_410278


namespace NUMINAMATH_CALUDE_jasons_stove_repair_cost_l4102_410256

theorem jasons_stove_repair_cost (stove_cost : ℝ) (wall_repair_ratio : ℝ) : 
  stove_cost = 1200 →
  wall_repair_ratio = 1 / 6 →
  stove_cost + (wall_repair_ratio * stove_cost) = 1400 := by
  sorry

end NUMINAMATH_CALUDE_jasons_stove_repair_cost_l4102_410256


namespace NUMINAMATH_CALUDE_largest_expression_l4102_410237

theorem largest_expression : 
  let a := 1 - 2 + 3 + 4
  let b := 1 + 2 - 3 + 4
  let c := 1 + 2 + 3 - 4
  let d := 1 + 2 - 3 - 4
  let e := 1 - 2 - 3 + 4
  (a ≥ b) ∧ (a ≥ c) ∧ (a ≥ d) ∧ (a ≥ e) := by
  sorry

#eval (1 - 2 + 3 + 4)
#eval (1 + 2 - 3 + 4)
#eval (1 + 2 + 3 - 4)
#eval (1 + 2 - 3 - 4)
#eval (1 - 2 - 3 + 4)

end NUMINAMATH_CALUDE_largest_expression_l4102_410237


namespace NUMINAMATH_CALUDE_range_of_a_minus_b_l4102_410213

theorem range_of_a_minus_b (a b : ℝ) (ha : 1 < a ∧ a < 4) (hb : -2 < b ∧ b < 4) :
  ∀ x, x ∈ Set.Ioo (-3 : ℝ) 6 ↔ ∃ (a' b' : ℝ), 1 < a' ∧ a' < 4 ∧ -2 < b' ∧ b' < 4 ∧ x = a' - b' :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_minus_b_l4102_410213


namespace NUMINAMATH_CALUDE_temperature_conversion_l4102_410289

theorem temperature_conversion (t k : ℝ) : 
  t = 5 / 9 * (k - 32) → k = 95 → t = 35 := by
  sorry

end NUMINAMATH_CALUDE_temperature_conversion_l4102_410289


namespace NUMINAMATH_CALUDE_gcd_condition_iff_special_form_l4102_410254

theorem gcd_condition_iff_special_form (n m : ℕ) (hn : n > 0) (hm : m > 0) :
  Nat.gcd ((n + 1)^m - n) ((n + 1)^(m+3) - n) > 1 ↔
  ∃ (k l : ℕ), k > 0 ∧ l > 0 ∧ n = 7*k - 6 ∧ m = 3*l :=
sorry

end NUMINAMATH_CALUDE_gcd_condition_iff_special_form_l4102_410254


namespace NUMINAMATH_CALUDE_quadratic_equation_equivalence_l4102_410219

theorem quadratic_equation_equivalence :
  ∀ x : ℝ, x^2 - 6*x + 7 = 0 ↔ (x - 3)^2 = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_equivalence_l4102_410219


namespace NUMINAMATH_CALUDE_degree_to_radian_90_l4102_410224

theorem degree_to_radian_90 : 
  (90 : ℝ) * (π / 180) = π / 2 := by sorry

end NUMINAMATH_CALUDE_degree_to_radian_90_l4102_410224


namespace NUMINAMATH_CALUDE_square_difference_of_integers_l4102_410204

theorem square_difference_of_integers (x y : ℕ+) 
  (sum_eq : x + y = 50) 
  (diff_eq : x - y = 12) : 
  x^2 - y^2 = 600 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_of_integers_l4102_410204


namespace NUMINAMATH_CALUDE_element_in_two_pairs_l4102_410247

/-- A system of elements and pairs satisfying the given conditions -/
structure PairSystem (n : ℕ) where
  -- The set of elements
  elements : Fin n → Type
  -- The set of pairs
  pairs : Fin n → Set (Fin n)
  -- Two pairs share exactly one element iff they form a pair
  share_condition : ∀ i j : Fin n, 
    (∃! k : Fin n, k ∈ pairs i ∧ k ∈ pairs j) ↔ j ∈ pairs i

/-- Every element is in exactly two pairs -/
theorem element_in_two_pairs {n : ℕ} (sys : PairSystem n) :
  ∀ k : Fin n, ∃! (i j : Fin n), i ≠ j ∧ k ∈ sys.pairs i ∧ k ∈ sys.pairs j :=
sorry

end NUMINAMATH_CALUDE_element_in_two_pairs_l4102_410247


namespace NUMINAMATH_CALUDE_farm_chickens_l4102_410253

theorem farm_chickens (total : ℕ) 
  (h1 : (total : ℚ) * (1/5) = (total : ℚ) * (20/100))  -- 20% of chickens are BCM
  (h2 : ((total : ℚ) * (1/5)) * (4/5) = ((total : ℚ) * (20/100)) * (80/100))  -- 80% of BCM are hens
  (h3 : ((total : ℚ) * (1/5)) * (4/5) = 16)  -- There are 16 BCM hens
  : total = 100 := by
sorry

end NUMINAMATH_CALUDE_farm_chickens_l4102_410253


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l4102_410236

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 > 0) ↔ (∃ x : ℝ, x^2 ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l4102_410236


namespace NUMINAMATH_CALUDE_spade_nested_calc_l4102_410280

-- Define the spade operation
def spade (x y : ℚ) : ℚ := x - 1 / y

-- Theorem statement
theorem spade_nested_calc : spade 3 (spade 3 (3/2)) = 18/7 := by sorry

end NUMINAMATH_CALUDE_spade_nested_calc_l4102_410280


namespace NUMINAMATH_CALUDE_gcd_sequence_a_odd_l4102_410208

def sequence_a (a₁ : ℤ) : ℕ → ℤ
  | 0 => a₁
  | n + 1 => (sequence_a a₁ n)^2 - (sequence_a a₁ n) - 1

theorem gcd_sequence_a_odd (a₁ : ℤ) (n : ℕ) :
  Nat.gcd (Int.natAbs (sequence_a a₁ (n + 1))) (2 * (n + 1) + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_sequence_a_odd_l4102_410208


namespace NUMINAMATH_CALUDE_central_cell_value_l4102_410273

theorem central_cell_value (a b c d e f g h i : ℝ) 
  (row_prod : a * b * c = 10 ∧ d * e * f = 10 ∧ g * h * i = 10)
  (col_prod : a * d * g = 10 ∧ b * e * h = 10 ∧ c * f * i = 10)
  (square_prod : a * b * d * e = 3 ∧ b * c * e * f = 3 ∧ d * e * g * h = 3 ∧ e * f * h * i = 3) :
  e = 0.00081 := by
  sorry

end NUMINAMATH_CALUDE_central_cell_value_l4102_410273


namespace NUMINAMATH_CALUDE_marshas_pay_per_mile_l4102_410265

theorem marshas_pay_per_mile :
  let first_package_miles : ℝ := 10
  let second_package_miles : ℝ := 28
  let third_package_miles : ℝ := second_package_miles / 2
  let total_miles : ℝ := first_package_miles + second_package_miles + third_package_miles
  let total_pay : ℝ := 104
  total_pay / total_miles = 2 := by
sorry

end NUMINAMATH_CALUDE_marshas_pay_per_mile_l4102_410265


namespace NUMINAMATH_CALUDE_cable_program_cost_per_roommate_l4102_410284

/-- Cable program cost calculation -/
def cable_cost (tier1_cost tier2_cost : ℚ) : ℚ :=
  let tier3_cost := tier2_cost / 2
  let tier4_cost := tier3_cost * 1.25
  tier1_cost + tier2_cost + (tier3_cost * 1.5) + (tier4_cost * 2)

/-- Cost per roommate calculation -/
def cost_per_roommate (total_cost : ℚ) (num_roommates : ℕ) : ℚ :=
  total_cost / num_roommates

/-- Theorem stating the cost per roommate for the given cable program -/
theorem cable_program_cost_per_roommate :
  cost_per_roommate (cable_cost 100 75) 4 = 81.25 := by
  sorry

#eval cable_cost 100 75
#eval cost_per_roommate (cable_cost 100 75) 4

end NUMINAMATH_CALUDE_cable_program_cost_per_roommate_l4102_410284


namespace NUMINAMATH_CALUDE_inner_square_area_l4102_410214

/-- Represents a square with side length and area -/
structure Square where
  side_length : ℝ
  area : ℝ

/-- Represents the configuration of two squares -/
structure SquareConfiguration where
  outer : Square
  inner : Square
  wi_length : ℝ

/-- Checks if the configuration is valid -/
def is_valid_configuration (config : SquareConfiguration) : Prop :=
  config.outer.side_length = 10 ∧
  config.wi_length = 3 ∧
  config.inner.area = config.inner.side_length ^ 2 ∧
  config.outer.area = config.outer.side_length ^ 2 ∧
  config.inner.side_length < config.outer.side_length

/-- The main theorem -/
theorem inner_square_area (config : SquareConfiguration) :
  is_valid_configuration config →
  config.inner.area = 21.16 := by
  sorry

end NUMINAMATH_CALUDE_inner_square_area_l4102_410214


namespace NUMINAMATH_CALUDE_rockets_won_38_games_l4102_410296

/-- Represents the number of wins for each team -/
structure TeamWins where
  sharks : ℕ
  dolphins : ℕ
  rockets : ℕ
  wolves : ℕ
  comets : ℕ

/-- The set of possible win numbers -/
def winNumbers : Finset ℕ := {28, 33, 38, 43}

/-- The conditions of the problem -/
def validTeamWins (tw : TeamWins) : Prop :=
  tw.sharks > tw.dolphins ∧
  tw.rockets > tw.wolves ∧
  tw.comets > tw.rockets ∧
  tw.wolves > 25 ∧
  tw.sharks ∈ winNumbers ∧
  tw.dolphins ∈ winNumbers ∧
  tw.rockets ∈ winNumbers ∧
  tw.wolves ∈ winNumbers ∧
  tw.comets ∈ winNumbers

/-- Theorem: Given the conditions, the Rockets won 38 games -/
theorem rockets_won_38_games (tw : TeamWins) (h : validTeamWins tw) : tw.rockets = 38 := by
  sorry

end NUMINAMATH_CALUDE_rockets_won_38_games_l4102_410296


namespace NUMINAMATH_CALUDE_complex_square_eq_neg45_neg48i_l4102_410259

theorem complex_square_eq_neg45_neg48i (z : ℂ) : 
  z^2 = -45 - 48*I ↔ z = 3 - 8*I ∨ z = -3 + 8*I := by
  sorry

end NUMINAMATH_CALUDE_complex_square_eq_neg45_neg48i_l4102_410259


namespace NUMINAMATH_CALUDE_polynomial_expansion_l4102_410266

theorem polynomial_expansion (t : ℝ) : 
  (3 * t^2 - 4 * t + 3) * (-4 * t^2 + 2 * t - 6) = 
  -12 * t^4 + 22 * t^3 - 38 * t^2 + 30 * t - 18 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l4102_410266


namespace NUMINAMATH_CALUDE_initial_students_count_l4102_410245

/-- The number of students at the start of the year. -/
def initial_students : ℕ := sorry

/-- The number of students who left during the year. -/
def students_left : ℕ := 5

/-- The number of new students who came during the year. -/
def new_students : ℕ := 8

/-- The number of students at the end of the year. -/
def final_students : ℕ := 11

/-- Theorem stating that the initial number of students is 8. -/
theorem initial_students_count :
  initial_students = final_students - (new_students - students_left) := by sorry

end NUMINAMATH_CALUDE_initial_students_count_l4102_410245


namespace NUMINAMATH_CALUDE_min_balls_to_guarantee_color_l4102_410249

theorem min_balls_to_guarantee_color (red green yellow blue white black : ℕ) 
  (h_red : red = 30) (h_green : green = 25) (h_yellow : yellow = 22) 
  (h_blue : blue = 15) (h_white : white = 12) (h_black : black = 10) : 
  ∃ (n : ℕ), n = 95 ∧ 
  (∀ m : ℕ, m < n → 
    ∃ (r g y b w k : ℕ), r < 20 ∧ g < 20 ∧ y < 20 ∧ b < 20 ∧ w < 20 ∧ k < 20 ∧
    r + g + y + b + w + k = m ∧
    r ≤ red ∧ g ≤ green ∧ y ≤ yellow ∧ b ≤ blue ∧ w ≤ white ∧ k ≤ black) ∧
  (∀ (r g y b w k : ℕ), r + g + y + b + w + k = n →
    r ≤ red ∧ g ≤ green ∧ y ≤ yellow ∧ b ≤ blue ∧ w ≤ white ∧ k ≤ black →
    r ≥ 20 ∨ g ≥ 20 ∨ y ≥ 20 ∨ b ≥ 20 ∨ w ≥ 20 ∨ k ≥ 20) :=
by sorry

end NUMINAMATH_CALUDE_min_balls_to_guarantee_color_l4102_410249


namespace NUMINAMATH_CALUDE_magical_green_knights_fraction_l4102_410250

theorem magical_green_knights_fraction (total : ℕ) (total_pos : 0 < total) :
  let green := total / 3
  let yellow := total - green
  let magical := total / 5
  let green_magical_fraction := magical_green / green
  let yellow_magical_fraction := magical_yellow / yellow
  green_magical_fraction = 3 * yellow_magical_fraction →
  magical_green + magical_yellow = magical →
  green_magical_fraction = 9 / 25 :=
by sorry

end NUMINAMATH_CALUDE_magical_green_knights_fraction_l4102_410250


namespace NUMINAMATH_CALUDE_largest_power_of_two_divisor_l4102_410292

theorem largest_power_of_two_divisor (n : ℕ) :
  (∃ (k : ℕ), 2^k ∣ ⌊(3 + Real.sqrt 11)^(2*n - 1)⌋ ∧
    ∀ (m : ℕ), m > k → ¬(2^m ∣ ⌊(3 + Real.sqrt 11)^(2*n - 1)⌋)) →
  (∃! (k : ℕ), k = n ∧ 2^k ∣ ⌊(3 + Real.sqrt 11)^(2*n - 1)⌋ ∧
    ∀ (m : ℕ), m > k → ¬(2^m ∣ ⌊(3 + Real.sqrt 11)^(2*n - 1)⌋)) :=
by sorry

#check largest_power_of_two_divisor

end NUMINAMATH_CALUDE_largest_power_of_two_divisor_l4102_410292


namespace NUMINAMATH_CALUDE_exam_mean_score_l4102_410233

theorem exam_mean_score (morning_mean : ℝ) (afternoon_mean : ℝ) (ratio : ℚ) 
  (h1 : morning_mean = 90)
  (h2 : afternoon_mean = 75)
  (h3 : ratio = 5 / 7) : 
  ∃ (overall_mean : ℝ), 
    (overall_mean ≥ 81 ∧ overall_mean < 82) ∧ 
    (∀ (m a : ℕ), m / a = ratio → 
      (m * morning_mean + a * afternoon_mean) / (m + a) = overall_mean) :=
sorry

end NUMINAMATH_CALUDE_exam_mean_score_l4102_410233


namespace NUMINAMATH_CALUDE_games_played_calculation_l4102_410212

/-- Represents the gambler's poker game statistics -/
structure GamblerStats where
  gamesPlayed : ℝ
  initialWinRate : ℝ
  newWinRate : ℝ
  targetWinRate : ℝ
  additionalGames : ℝ

/-- Theorem stating the number of games played given the conditions -/
theorem games_played_calculation (stats : GamblerStats)
  (h1 : stats.initialWinRate = 0.4)
  (h2 : stats.newWinRate = 0.8)
  (h3 : stats.targetWinRate = 0.6)
  (h4 : stats.additionalGames = 19.999999999999993)
  (h5 : stats.initialWinRate * stats.gamesPlayed + stats.newWinRate * stats.additionalGames = 
        stats.targetWinRate * (stats.gamesPlayed + stats.additionalGames)) :
  stats.gamesPlayed = 20 := by
  sorry

end NUMINAMATH_CALUDE_games_played_calculation_l4102_410212


namespace NUMINAMATH_CALUDE_sum_of_integers_l4102_410203

theorem sum_of_integers (x y : ℕ+) (h1 : x^2 + y^2 = 130) (h2 : x * y = 27) : 
  x + y = 14 := by
sorry

end NUMINAMATH_CALUDE_sum_of_integers_l4102_410203


namespace NUMINAMATH_CALUDE_max_b_for_zero_in_range_l4102_410241

/-- The maximum value of b for which the quadratic function g(x) = x^2 - 7x + b has 0 in its range -/
theorem max_b_for_zero_in_range : 
  ∃ (b_max : ℝ), b_max = 49 / 4 ∧ 
  (∀ b : ℝ, (∃ x : ℝ, x^2 - 7*x + b = 0) → b ≤ b_max) ∧
  (∃ x : ℝ, x^2 - 7*x + b_max = 0) :=
sorry

end NUMINAMATH_CALUDE_max_b_for_zero_in_range_l4102_410241


namespace NUMINAMATH_CALUDE_gear_speed_proportion_l4102_410244

/-- Represents a gear in the system -/
structure Gear where
  teeth : ℕ
  angular_speed : ℝ

/-- Represents the system of gears -/
structure GearSystem where
  P : Gear
  Q : Gear
  R : Gear
  efficiency : ℝ

/-- The theorem stating the correct proportion of angular speeds -/
theorem gear_speed_proportion (sys : GearSystem) 
  (h1 : sys.efficiency = 0.9)
  (h2 : sys.P.teeth * sys.P.angular_speed = sys.Q.teeth * sys.Q.angular_speed)
  (h3 : sys.R.angular_speed = sys.efficiency * sys.Q.angular_speed) :
  ∃ (k : ℝ), k > 0 ∧ 
    sys.P.angular_speed = k * sys.Q.teeth ∧
    sys.Q.angular_speed = k * sys.P.teeth ∧
    sys.R.angular_speed = k * sys.efficiency * sys.P.teeth :=
sorry

end NUMINAMATH_CALUDE_gear_speed_proportion_l4102_410244


namespace NUMINAMATH_CALUDE_l_shaped_area_l4102_410290

/-- The area of an L-shaped region formed by subtracting two smaller squares
    from a larger square -/
theorem l_shaped_area (side_large : ℝ) (side_small1 : ℝ) (side_small2 : ℝ)
    (h1 : side_large = side_small1 + side_small2)
    (h2 : side_small1 = 4)
    (h3 : side_small2 = 2) :
    side_large^2 - (side_small1^2 + side_small2^2) = 16 := by
  sorry

end NUMINAMATH_CALUDE_l_shaped_area_l4102_410290


namespace NUMINAMATH_CALUDE_fraction_change_l4102_410279

theorem fraction_change (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0) :
  1 - (a * 0.8) / (b * 1.28) / (a / b) = 0.375 := by sorry

end NUMINAMATH_CALUDE_fraction_change_l4102_410279


namespace NUMINAMATH_CALUDE_black_balls_count_l4102_410261

theorem black_balls_count (total : ℕ) (red white black : ℕ) 
  (h1 : red + white + black = total)
  (h2 : (red : ℚ) / total = 42 / 100)
  (h3 : (white : ℚ) / total = 28 / 100)
  (h4 : red = 21) : 
  black = 15 := by
  sorry

end NUMINAMATH_CALUDE_black_balls_count_l4102_410261


namespace NUMINAMATH_CALUDE_pear_weight_proof_l4102_410246

/-- The weight of one pear in grams -/
def pear_weight : ℝ := 120

theorem pear_weight_proof :
  let apple_weight : ℝ := 530
  let apple_count : ℕ := 12
  let pear_count : ℕ := 8
  let weight_difference : ℝ := 5400
  apple_count * apple_weight = pear_count * pear_weight + weight_difference →
  pear_weight = 120 := by
sorry

end NUMINAMATH_CALUDE_pear_weight_proof_l4102_410246


namespace NUMINAMATH_CALUDE_four_points_no_obtuse_triangle_l4102_410257

noncomputable def probability_no_obtuse_triangle (n : ℕ) : ℝ :=
  sorry

theorem four_points_no_obtuse_triangle :
  probability_no_obtuse_triangle 4 = 3 / 32 :=
sorry

end NUMINAMATH_CALUDE_four_points_no_obtuse_triangle_l4102_410257


namespace NUMINAMATH_CALUDE_restaurant_cooks_count_l4102_410222

theorem restaurant_cooks_count :
  ∀ (C W : ℕ),
  (C : ℚ) / W = 3 / 10 →
  (C : ℚ) / (W + 12) = 3 / 14 →
  C = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_restaurant_cooks_count_l4102_410222


namespace NUMINAMATH_CALUDE_largest_divisor_of_expression_l4102_410281

theorem largest_divisor_of_expression (y : ℤ) (h : Even y) :
  (∃ (k : ℤ), (8*y+4)*(8*y+8)*(4*y+6)*(4*y+2) = 96 * k) ∧
  (∀ (n : ℤ), n > 96 → ¬(∀ (y : ℤ), Even y → ∃ (k : ℤ), (8*y+4)*(8*y+8)*(4*y+6)*(4*y+2) = n * k)) :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_expression_l4102_410281


namespace NUMINAMATH_CALUDE_mary_marbles_left_l4102_410223

/-- The number of yellow marbles Mary has left after a series of exchanges -/
def marblesLeft (initial : ℝ) (giveJoan : ℝ) (receiveJoan : ℝ) (giveSam : ℝ) : ℝ :=
  initial - giveJoan + receiveJoan - giveSam

/-- Theorem stating that Mary will have 4.7 yellow marbles left -/
theorem mary_marbles_left :
  marblesLeft 9.5 2.3 1.1 3.6 = 4.7 := by
  sorry

end NUMINAMATH_CALUDE_mary_marbles_left_l4102_410223


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l4102_410262

theorem sum_of_roots_quadratic (x : ℝ) (h : x^2 + 12*x = 64) : 
  ∃ y : ℝ, y^2 + 12*y = 64 ∧ x + y = -12 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l4102_410262


namespace NUMINAMATH_CALUDE_additional_space_needed_l4102_410272

theorem additional_space_needed (available_space backup_size software_size : ℕ) : 
  available_space = 28 → 
  backup_size = 26 → 
  software_size = 4 → 
  (backup_size + software_size) - available_space = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_additional_space_needed_l4102_410272


namespace NUMINAMATH_CALUDE_B_eq_A_pow2_l4102_410263

def A : ℕ → ℚ
  | 0 => 1
  | n + 1 => (A n + 2) / (A n + 1)

def B : ℕ → ℚ
  | 0 => 1
  | n + 1 => (B n^2 + 2) / (2 * B n)

theorem B_eq_A_pow2 (n : ℕ) : B (n + 1) = A (2^n) := by
  sorry

end NUMINAMATH_CALUDE_B_eq_A_pow2_l4102_410263


namespace NUMINAMATH_CALUDE_arithmetic_progression_coprime_terms_l4102_410226

theorem arithmetic_progression_coprime_terms :
  ∃ (a r : ℕ), 
    (∀ i j, 0 ≤ i ∧ i < j ∧ j < 100 → 
      (a + i * r).gcd (a + j * r) = 1) ∧
    (∀ i, 0 ≤ i ∧ i < 99 → a + i * r < a + (i + 1) * r) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_coprime_terms_l4102_410226


namespace NUMINAMATH_CALUDE_total_pencils_l4102_410285

/-- Given that each child has 4 pencils and there are 8 children, 
    prove that the total number of pencils is 32. -/
theorem total_pencils (pencils_per_child : ℕ) (num_children : ℕ) 
  (h1 : pencils_per_child = 4) (h2 : num_children = 8) : 
  pencils_per_child * num_children = 32 := by
  sorry

end NUMINAMATH_CALUDE_total_pencils_l4102_410285


namespace NUMINAMATH_CALUDE_hex_to_binary_max_bits_l4102_410291

theorem hex_to_binary_max_bits :
  ∀ (A B C D : Nat),
  A < 16 → B < 16 → C < 16 → D < 16 →
  ∃ (n : Nat),
  n ≤ 16 ∧
  A * 16^3 + B * 16^2 + C * 16^1 + D < 2^n :=
by sorry

end NUMINAMATH_CALUDE_hex_to_binary_max_bits_l4102_410291


namespace NUMINAMATH_CALUDE_new_person_weight_l4102_410270

-- Define the initial number of people
def initial_people : ℕ := 8

-- Define the weight increase per person
def weight_increase_per_person : ℝ := 3

-- Define the weight of the person being replaced
def replaced_person_weight : ℝ := 65

-- Theorem to prove
theorem new_person_weight (new_weight : ℝ) :
  new_weight = replaced_person_weight + (initial_people * weight_increase_per_person) :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l4102_410270


namespace NUMINAMATH_CALUDE_solution_to_system_of_equations_l4102_410283

theorem solution_to_system_of_equations :
  ∃ x y : ℚ, x + 2*y = 3 ∧ 9*x - 8*y = 5 ∧ x = 17/13 ∧ y = 11/13 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_system_of_equations_l4102_410283


namespace NUMINAMATH_CALUDE_shaded_fraction_of_specific_rectangle_l4102_410268

/-- Represents a rectangle divided into equal squares -/
structure DividedRectangle where
  total_squares : ℕ
  shaded_half_squares : ℕ

/-- Calculates the fraction of a divided rectangle that is shaded -/
def shaded_fraction (rect : DividedRectangle) : ℚ :=
  rect.shaded_half_squares / (2 * rect.total_squares)

theorem shaded_fraction_of_specific_rectangle : 
  ∀ (rect : DividedRectangle), 
    rect.total_squares = 6 → 
    rect.shaded_half_squares = 5 → 
    shaded_fraction rect = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_shaded_fraction_of_specific_rectangle_l4102_410268


namespace NUMINAMATH_CALUDE_major_axis_length_is_13_l4102_410276

/-- A configuration of a cylinder and two spheres -/
structure CylinderSphereConfig where
  cylinder_radius : ℝ
  sphere_radius : ℝ
  sphere_centers_distance : ℝ

/-- The length of the major axis of the ellipse formed by the intersection of a plane 
    touching both spheres and the cylinder surface -/
def major_axis_length (config : CylinderSphereConfig) : ℝ :=
  config.sphere_centers_distance

/-- Theorem stating that for the given configuration, the major axis length is 13 -/
theorem major_axis_length_is_13 :
  let config := CylinderSphereConfig.mk 6 6 13
  major_axis_length config = 13 := by
  sorry

#eval major_axis_length (CylinderSphereConfig.mk 6 6 13)

end NUMINAMATH_CALUDE_major_axis_length_is_13_l4102_410276


namespace NUMINAMATH_CALUDE_arithmetic_square_root_sum_l4102_410294

theorem arithmetic_square_root_sum (a b c : ℝ) : 
  a^(1/3) = 2 → 
  b = ⌊Real.sqrt 5⌋ → 
  c^2 = 16 → 
  (Real.sqrt (a + b + c) = Real.sqrt 14) ∨ (Real.sqrt (a + b + c) = Real.sqrt 6) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_sum_l4102_410294


namespace NUMINAMATH_CALUDE_consecutive_integers_equation_l4102_410200

theorem consecutive_integers_equation (x y z n : ℤ) : 
  x = y + 1 → 
  y = z + 1 → 
  x > y → 
  y > z → 
  z = 3 → 
  2*x + 3*y + 3*z = 5*y + n → 
  n = 11 := by sorry

end NUMINAMATH_CALUDE_consecutive_integers_equation_l4102_410200


namespace NUMINAMATH_CALUDE_log_expression_equality_l4102_410238

theorem log_expression_equality : 
  (Real.log 3 / Real.log 2 + Real.log 3 / Real.log 8) / (Real.log 9 / Real.log 4) = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_log_expression_equality_l4102_410238


namespace NUMINAMATH_CALUDE_quadratic_function_problem_l4102_410215

/-- A quadratic function f(x) = ax^2 + bx + c with integer coefficients -/
def f (a b c : ℤ) (x : ℤ) : ℤ := a * x^2 + b * x + c

/-- The value of j that satisfies the given conditions -/
def j : ℤ := 36

theorem quadratic_function_problem (a b c : ℤ) :
  f a b c 2 = 0 ∧
  200 < f a b c 10 ∧ f a b c 10 < 300 ∧
  400 < f a b c 9 ∧ f a b c 9 < 500 ∧
  1000 * j < f a b c 100 ∧ f a b c 100 < 1000 * (j + 1) →
  j = 36 := by sorry

end NUMINAMATH_CALUDE_quadratic_function_problem_l4102_410215


namespace NUMINAMATH_CALUDE_evaluate_expression_l4102_410299

theorem evaluate_expression (x y : ℕ) (h1 : x = 3) (h2 : y = 4) :
  5 * x^(y-1) + 2 * y^(x+1) = 647 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l4102_410299


namespace NUMINAMATH_CALUDE_solution_exists_for_a_in_range_l4102_410202

/-- The system of equations has a solution for a given 'a' -/
def has_solution (a : ℝ) : Prop :=
  ∃ b x y : ℝ, x^2 + y^2 + 2*a*(a + y - x) = 49 ∧ y = 8 / ((x - b)^2 + 1)

/-- The theorem stating the range of 'a' for which the system has a solution -/
theorem solution_exists_for_a_in_range :
  ∀ a : ℝ, -15 ≤ a ∧ a < 7 → has_solution a :=
sorry

end NUMINAMATH_CALUDE_solution_exists_for_a_in_range_l4102_410202


namespace NUMINAMATH_CALUDE_annieka_made_14_throws_l4102_410232

/-- The number of free-throws made by DeShawn -/
def deshawn_throws : ℕ := 12

/-- The number of free-throws made by Kayla -/
def kayla_throws : ℕ := (deshawn_throws * 3) / 2

/-- The number of free-throws made by Annieka -/
def annieka_throws : ℕ := kayla_throws - 4

/-- Theorem: Annieka made 14 free-throws -/
theorem annieka_made_14_throws : annieka_throws = 14 := by
  sorry

end NUMINAMATH_CALUDE_annieka_made_14_throws_l4102_410232


namespace NUMINAMATH_CALUDE_solve_equation_l4102_410269

theorem solve_equation (x : ℝ) : 3 * x + 36 = 48 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l4102_410269


namespace NUMINAMATH_CALUDE_triangle_inequality_l4102_410264

theorem triangle_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b) :
  (a / (b + c - a) + b / (c + a - b) + c / (a + b - c) ≥
   (b + c - a) / a + (c + a - b) / b + (a + b - c) / c) ∧
  ((b + c - a) / a + (c + a - b) / b + (a + b - c) / c ≥ 3) := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l4102_410264


namespace NUMINAMATH_CALUDE_work_completion_time_l4102_410210

theorem work_completion_time 
  (a_rate : ℚ) 
  (b_rate : ℚ) 
  (joint_work_days : ℕ) 
  (h1 : a_rate = 1 / 5) 
  (h2 : b_rate = 1 / 15) 
  (h3 : joint_work_days = 2) : 
  ℕ :=
by
  sorry

#check work_completion_time

end NUMINAMATH_CALUDE_work_completion_time_l4102_410210


namespace NUMINAMATH_CALUDE_euclidean_division_l4102_410274

theorem euclidean_division (a b : ℕ) (hb : b > 0) :
  ∃ q r : ℤ, 0 ≤ r ∧ r < b ∧ (a : ℤ) = b * q + r :=
sorry

end NUMINAMATH_CALUDE_euclidean_division_l4102_410274


namespace NUMINAMATH_CALUDE_jellybean_ratio_l4102_410206

/-- Proves the ratio of jellybeans Lorelai has eaten to the total jellybeans Rory and Gigi have -/
theorem jellybean_ratio (gigi_jellybeans : ℕ) (rory_extra_jellybeans : ℕ) (lorelai_jellybeans : ℕ) :
  gigi_jellybeans = 15 →
  rory_extra_jellybeans = 30 →
  lorelai_jellybeans = 180 →
  ∃ (m : ℕ), m * (gigi_jellybeans + (gigi_jellybeans + rory_extra_jellybeans)) = lorelai_jellybeans →
  (lorelai_jellybeans : ℚ) / (gigi_jellybeans + (gigi_jellybeans + rory_extra_jellybeans) : ℚ) = 3 := by
  sorry

#check jellybean_ratio

end NUMINAMATH_CALUDE_jellybean_ratio_l4102_410206


namespace NUMINAMATH_CALUDE_inverse_proportion_point_value_l4102_410242

/-- Prove that for an inverse proportion function y = k/x (k ≠ 0),
    if points A(2,m) and B(m,n) lie on its graph, then n = 2. -/
theorem inverse_proportion_point_value (k m n : ℝ) : 
  k ≠ 0 → m = k / 2 → n = k / m → n = 2 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_point_value_l4102_410242


namespace NUMINAMATH_CALUDE_certain_number_problem_l4102_410275

theorem certain_number_problem :
  ∃ x : ℝ, (1/10 : ℝ) * x - (1/1000 : ℝ) * x = 693 ∧ x = 7000 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l4102_410275


namespace NUMINAMATH_CALUDE_eugene_pencils_l4102_410217

theorem eugene_pencils (x : ℕ) (h1 : x + 6 = 57) : x = 51 := by
  sorry

end NUMINAMATH_CALUDE_eugene_pencils_l4102_410217


namespace NUMINAMATH_CALUDE_point_outside_circle_l4102_410277

/-- The line ax + by = 1 intersects with the circle x^2 + y^2 = 1 -/
def line_intersects_circle (a b : ℝ) : Prop :=
  ∃ x y : ℝ, a * x + b * y = 1 ∧ x^2 + y^2 = 1

theorem point_outside_circle (a b : ℝ) :
  line_intersects_circle a b → a^2 + b^2 > 1 := by
  sorry

end NUMINAMATH_CALUDE_point_outside_circle_l4102_410277


namespace NUMINAMATH_CALUDE_attractions_permutations_l4102_410205

theorem attractions_permutations : Nat.factorial 5 = 120 := by
  sorry

end NUMINAMATH_CALUDE_attractions_permutations_l4102_410205


namespace NUMINAMATH_CALUDE_c_plus_d_equals_negative_two_l4102_410287

theorem c_plus_d_equals_negative_two 
  (c d : ℝ) 
  (h1 : 2 = c + d / 3) 
  (h2 : 6 = c + d / (-3)) : 
  c + d = -2 := by
sorry

end NUMINAMATH_CALUDE_c_plus_d_equals_negative_two_l4102_410287


namespace NUMINAMATH_CALUDE_ice_cream_sales_for_video_games_l4102_410201

theorem ice_cream_sales_for_video_games :
  let game_cost : ℕ := 60
  let ice_cream_price : ℕ := 5
  let num_games : ℕ := 2
  let total_cost : ℕ := game_cost * num_games
  let ice_creams_needed : ℕ := total_cost / ice_cream_price
  ice_creams_needed = 24 := by
sorry

end NUMINAMATH_CALUDE_ice_cream_sales_for_video_games_l4102_410201


namespace NUMINAMATH_CALUDE_intersection_and_union_of_A_and_B_l4102_410211

-- Define the universal set U
def U : Set Int := {-3, -1, 0, 1, 2, 3, 4, 6}

-- Define set A
def A : Set Int := {0, 2, 4, 6}

-- Define the complement of A with respect to U
def C_UA : Set Int := {-1, -3, 1, 3}

-- Define the complement of B with respect to U
def C_UB : Set Int := {-1, 0, 2}

-- Define set B
def B : Set Int := U \ C_UB

-- Theorem to prove
theorem intersection_and_union_of_A_and_B :
  (A ∩ B = {4, 6}) ∧ (A ∪ B = {-3, 0, 1, 2, 3, 4, 6}) := by
  sorry

end NUMINAMATH_CALUDE_intersection_and_union_of_A_and_B_l4102_410211


namespace NUMINAMATH_CALUDE_original_decimal_l4102_410240

theorem original_decimal : ∃ x : ℝ, x * 10 = x + 2.7 ∧ x = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_original_decimal_l4102_410240


namespace NUMINAMATH_CALUDE_negative_five_times_three_l4102_410260

theorem negative_five_times_three : (-5 : ℤ) * 3 = -15 := by
  sorry

end NUMINAMATH_CALUDE_negative_five_times_three_l4102_410260


namespace NUMINAMATH_CALUDE_max_value_expression_max_value_achievable_l4102_410295

theorem max_value_expression (a b : ℝ) (ha : a ≥ 1) (hb : b ≥ 1) :
  (|7*a + 8*b - a*b| + |2*a + 8*b - 6*a*b|) / (a * Real.sqrt (1 + b^2)) ≤ 9 * Real.sqrt 2 :=
by sorry

theorem max_value_achievable :
  ∃ (a b : ℝ), a ≥ 1 ∧ b ≥ 1 ∧
  (|7*a + 8*b - a*b| + |2*a + 8*b - 6*a*b|) / (a * Real.sqrt (1 + b^2)) = 9 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_max_value_achievable_l4102_410295


namespace NUMINAMATH_CALUDE_unique_A_for_club_equation_l4102_410239

-- Define the ♣ operation
def club (A B : ℝ) : ℝ := 4 * A + 2 * B + 6

-- Theorem statement
theorem unique_A_for_club_equation : ∃! A : ℝ, club A 6 = 70 ∧ A = 13 := by
  sorry

end NUMINAMATH_CALUDE_unique_A_for_club_equation_l4102_410239


namespace NUMINAMATH_CALUDE_complex_sixth_root_of_negative_sixteen_l4102_410230

theorem complex_sixth_root_of_negative_sixteen :
  ∀ z : ℂ, z^6 = -16 ↔ z = Complex.I * 2 ∨ z = Complex.I * (-2) := by
  sorry

end NUMINAMATH_CALUDE_complex_sixth_root_of_negative_sixteen_l4102_410230


namespace NUMINAMATH_CALUDE_parabola_decreasing_condition_l4102_410218

/-- Represents a parabola of the form y = -5(x + m)² - 3 -/
def Parabola (m : ℝ) : ℝ → ℝ := λ x ↦ -5 * (x + m)^2 - 3

/-- States that the parabola is decreasing for x ≥ 2 -/
def IsDecreasingForXGeq2 (m : ℝ) : Prop :=
  ∀ x₁ x₂, x₁ ≥ 2 → x₂ ≥ 2 → x₁ < x₂ → Parabola m x₁ > Parabola m x₂

theorem parabola_decreasing_condition (m : ℝ) :
  IsDecreasingForXGeq2 m → m ≥ -2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_decreasing_condition_l4102_410218


namespace NUMINAMATH_CALUDE_triangle_perimeter_l4102_410252

/-- Given a right-angled triangle PQR with R at the right angle, 
    PR = 4000, and PQ = 5000, prove that PQ + QR + PR = 16500 -/
theorem triangle_perimeter (PR PQ QR : ℝ) : 
  PR = 4000 → 
  PQ = 5000 → 
  QR^2 = PQ^2 - PR^2 → 
  PQ + QR + PR = 16500 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l4102_410252


namespace NUMINAMATH_CALUDE_ellipse_intersection_relation_l4102_410235

/-- Theorem: Relationship between y-coordinates of intersection points on an ellipse --/
theorem ellipse_intersection_relation (a b m : ℝ) (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ) : 
  a > b ∧ b > 0 ∧ m > a →  -- Conditions on a, b, and m
  (x₁^2 / a^2 + y₁^2 / b^2 = 1) ∧  -- A is on the ellipse
  (x₂^2 / a^2 + y₂^2 / b^2 = 1) ∧  -- B is on the ellipse
  (∃ k : ℝ, x₁ = k * y₁ + m ∧ x₂ = k * y₂ + m) →  -- A and B are on a line through M
  x₃ = a^2 / m ∧ x₄ = a^2 / m →  -- P and Q are on the line x = a^2/m
  (y₃ * (x₁ + a) = y₁ * (x₃ + a)) ∧  -- P is on line A₁A
  (y₄ * (x₂ + a) = y₂ * (x₄ + a)) →  -- Q is on line A₁B
  1 / y₁ + 1 / y₂ = 1 / y₃ + 1 / y₄ :=
by sorry


end NUMINAMATH_CALUDE_ellipse_intersection_relation_l4102_410235


namespace NUMINAMATH_CALUDE_angle_315_same_terminal_side_as_negative_45_l4102_410293

-- Define a function to represent angles with the same terminal side
def sameTerminalSide (θ : ℝ) (α : ℝ) : Prop :=
  ∃ k : ℤ, α = k * 360 + θ

-- State the theorem
theorem angle_315_same_terminal_side_as_negative_45 :
  sameTerminalSide 315 (-45) := by
  sorry

end NUMINAMATH_CALUDE_angle_315_same_terminal_side_as_negative_45_l4102_410293


namespace NUMINAMATH_CALUDE_heracles_age_l4102_410255

/-- Proves that Heracles' age is 10 years old given the conditions of the problem -/
theorem heracles_age : 
  ∀ (heracles_age : ℕ) (audrey_age : ℕ),
  audrey_age = heracles_age + 7 →
  audrey_age + 3 = 2 * heracles_age →
  heracles_age = 10 := by
sorry

end NUMINAMATH_CALUDE_heracles_age_l4102_410255


namespace NUMINAMATH_CALUDE_ice_cream_flavors_l4102_410228

theorem ice_cream_flavors (cone_types : ℕ) (total_combinations : ℕ) (h1 : cone_types = 2) (h2 : total_combinations = 8) :
  total_combinations / cone_types = 4 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_flavors_l4102_410228


namespace NUMINAMATH_CALUDE_passengers_from_other_continents_l4102_410209

theorem passengers_from_other_continents : 
  ∀ (total : ℕ) (north_america europe africa asia other : ℚ),
    total = 96 →
    north_america = 1/4 →
    europe = 1/8 →
    africa = 1/12 →
    asia = 1/6 →
    other = 1 - (north_america + europe + africa + asia) →
    (other * total : ℚ) = 36 := by
  sorry

end NUMINAMATH_CALUDE_passengers_from_other_continents_l4102_410209


namespace NUMINAMATH_CALUDE_west_east_correspondence_l4102_410221

-- Define a type for directions
inductive Direction
| East
| West

-- Define a function to represent distance with direction
def distance_with_direction (d : ℝ) (dir : Direction) : ℝ :=
  match dir with
  | Direction.East => d
  | Direction.West => -d

-- State the theorem
theorem west_east_correspondence :
  (distance_with_direction 2023 Direction.West = -2023) →
  (distance_with_direction 2023 Direction.East = 2023) :=
by
  sorry

end NUMINAMATH_CALUDE_west_east_correspondence_l4102_410221


namespace NUMINAMATH_CALUDE_victor_remaining_lives_l4102_410248

def calculate_lives_remaining (initial_lives : ℕ) 
                               (first_level_loss : ℕ) 
                               (second_level_gain_rate : ℕ) 
                               (second_level_duration : ℕ) 
                               (third_level_loss_rate : ℕ) 
                               (third_level_duration : ℕ) : ℕ :=
  let lives_after_first := initial_lives - first_level_loss
  let second_level_intervals := second_level_duration / 45
  let lives_after_second := lives_after_first + second_level_gain_rate * second_level_intervals
  let third_level_intervals := third_level_duration / 20
  lives_after_second - third_level_loss_rate * third_level_intervals

theorem victor_remaining_lives : 
  calculate_lives_remaining 246 14 3 135 4 80 = 225 := by
  sorry

end NUMINAMATH_CALUDE_victor_remaining_lives_l4102_410248


namespace NUMINAMATH_CALUDE_factorial_calculation_l4102_410225

theorem factorial_calculation : (Nat.factorial 11) / (Nat.factorial 10) * 12 = 132 := by
  sorry

end NUMINAMATH_CALUDE_factorial_calculation_l4102_410225


namespace NUMINAMATH_CALUDE_terminal_side_angle_l4102_410297

/-- Given a point P(-4,3) on the terminal side of angle θ, prove that 2sin θ + cos θ = 2/5 -/
theorem terminal_side_angle (θ : ℝ) : 
  let P : ℝ × ℝ := (-4, 3)
  (P.1 = -4 ∧ P.2 = 3) →  -- Point P(-4,3)
  (P.1 = Real.cos θ * Real.sqrt (P.1^2 + P.2^2) ∧ 
   P.2 = Real.sin θ * Real.sqrt (P.1^2 + P.2^2)) →  -- P is on the terminal side of θ
  2 * Real.sin θ + Real.cos θ = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_terminal_side_angle_l4102_410297


namespace NUMINAMATH_CALUDE_number_problem_l4102_410227

theorem number_problem (x y : ℝ) : 
  (x^2)/2 + 5*y = 15 ∧ x + y = 10 → x = 5 ∧ y = 5 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l4102_410227


namespace NUMINAMATH_CALUDE_tank_capacity_l4102_410271

theorem tank_capacity (initial_buckets : ℕ) (initial_capacity : ℚ) (new_buckets : ℕ) :
  initial_buckets = 26 →
  initial_capacity = 13.5 →
  new_buckets = 39 →
  (initial_buckets : ℚ) * initial_capacity / (new_buckets : ℚ) = 9 :=
by sorry

end NUMINAMATH_CALUDE_tank_capacity_l4102_410271


namespace NUMINAMATH_CALUDE_sum_of_quotient_digits_l4102_410231

def dividend : ℕ := 111111
def divisor : ℕ := 3

def quotient : ℕ := dividend / divisor

def sum_of_digits (n : ℕ) : ℕ :=
  if n = 0 then 0 else n % 10 + sum_of_digits (n / 10)

theorem sum_of_quotient_digits :
  sum_of_digits quotient = 20 := by sorry

end NUMINAMATH_CALUDE_sum_of_quotient_digits_l4102_410231


namespace NUMINAMATH_CALUDE_unique_base_for_special_palindrome_l4102_410251

def is_palindrome (n : ℕ) (base : ℕ) : Prop :=
  ∃ (digits : List ℕ), digits.length > 0 ∧ 
    (digits.reverse = digits) ∧ 
    (n = digits.foldl (λ acc d => acc * base + d) 0)

theorem unique_base_for_special_palindrome : 
  ∃! (r : ℕ), 
    r % 2 = 0 ∧ 
    r ≥ 18 ∧ 
    (∃ (x : ℕ), 
      x = 5 * r^3 + 5 * r^2 + 5 * r + 5 ∧
      is_palindrome (x^2) r ∧
      (∃ (a b c d : ℕ), 
        x^2 = a * r^7 + b * r^6 + c * r^5 + d * r^4 + 
              d * r^3 + c * r^2 + b * r + a ∧
        d - c = 2)) ∧
    r = 24 :=
sorry

end NUMINAMATH_CALUDE_unique_base_for_special_palindrome_l4102_410251


namespace NUMINAMATH_CALUDE_science_club_board_selection_l4102_410207

theorem science_club_board_selection (total_members : Nat) (prev_served : Nat) (board_size : Nat)
  (h1 : total_members = 20)
  (h2 : prev_served = 9)
  (h3 : board_size = 6) :
  (Nat.choose total_members board_size) - (Nat.choose (total_members - prev_served) board_size) = 38298 := by
  sorry

end NUMINAMATH_CALUDE_science_club_board_selection_l4102_410207


namespace NUMINAMATH_CALUDE_square_binomial_k_l4102_410243

theorem square_binomial_k (k : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, x^2 - 20*x + k = (a*x + b)^2) → k = 100 := by
sorry

end NUMINAMATH_CALUDE_square_binomial_k_l4102_410243


namespace NUMINAMATH_CALUDE_negative_quartic_count_l4102_410216

theorem negative_quartic_count : 
  (∃ (S : Finset Int), (∀ x : Int, x ∈ S ↔ x^4 - 63*x^2 + 126 < 0) ∧ Finset.card S = 12) :=
by sorry

end NUMINAMATH_CALUDE_negative_quartic_count_l4102_410216


namespace NUMINAMATH_CALUDE_jills_salary_l4102_410220

theorem jills_salary (discretionary_income : ℝ) (net_salary : ℝ) : 
  discretionary_income = net_salary / 5 →
  discretionary_income * 0.15 = 105 →
  net_salary = 3500 := by
  sorry

end NUMINAMATH_CALUDE_jills_salary_l4102_410220


namespace NUMINAMATH_CALUDE_rabbit_travel_time_l4102_410298

/-- Proves that a rabbit running at a constant speed of 6 miles per hour will take 20 minutes to travel 2 miles. -/
theorem rabbit_travel_time :
  let rabbit_speed : ℝ := 6 -- miles per hour
  let distance : ℝ := 2 -- miles
  let time_in_hours : ℝ := distance / rabbit_speed
  let time_in_minutes : ℝ := time_in_hours * 60
  time_in_minutes = 20 := by sorry

end NUMINAMATH_CALUDE_rabbit_travel_time_l4102_410298


namespace NUMINAMATH_CALUDE_plane_perpendicularity_l4102_410229

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (contained_in : Line → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem plane_perpendicularity 
  (m n : Line) (α β : Plane) :
  parallel m n → 
  contained_in m α → 
  perpendicular n β → 
  plane_perpendicular α β := by
sorry

end NUMINAMATH_CALUDE_plane_perpendicularity_l4102_410229


namespace NUMINAMATH_CALUDE_world_grain_supply_l4102_410267

/-- World grain supply problem -/
theorem world_grain_supply :
  let world_grain_demand : ℝ := 2400000
  let supply_ratio : ℝ := 0.75
  let world_grain_supply : ℝ := supply_ratio * world_grain_demand
  world_grain_supply = 1800000 := by
  sorry

end NUMINAMATH_CALUDE_world_grain_supply_l4102_410267


namespace NUMINAMATH_CALUDE_product_of_roots_product_of_roots_specific_equation_l4102_410282

theorem product_of_roots (a b c : ℝ) (h : a ≠ 0) :
  let p := (- b + Real.sqrt (b ^ 2 - 4 * a * c)) / (2 * a)
  let q := (- b - Real.sqrt (b ^ 2 - 4 * a * c)) / (2 * a)
  p * q = c / a :=
by sorry

theorem product_of_roots_specific_equation :
  let p := (9 + Real.sqrt (81 + 4 * 36)) / 2
  let q := (9 - Real.sqrt (81 + 4 * 36)) / 2
  p * q = -36 :=
by sorry

end NUMINAMATH_CALUDE_product_of_roots_product_of_roots_specific_equation_l4102_410282


namespace NUMINAMATH_CALUDE_nandan_earning_is_2000_l4102_410258

/-- Represents the business investment scenario of Krishan and Nandan -/
structure BusinessInvestment where
  nandan_investment : ℝ
  nandan_time : ℝ
  total_gain : ℝ

/-- Calculates Nandan's earning based on the given business investment scenario -/
def nandan_earning (b : BusinessInvestment) : ℝ :=
  b.nandan_investment * b.nandan_time

/-- Theorem stating that Nandan's earning is 2000 given the specified conditions -/
theorem nandan_earning_is_2000 (b : BusinessInvestment) 
  (h1 : b.total_gain = 26000)
  (h2 : b.nandan_investment * b.nandan_time + 
        (4 * b.nandan_investment) * (3 * b.nandan_time) = b.total_gain) :
  nandan_earning b = 2000 := by
  sorry

#check nandan_earning_is_2000

end NUMINAMATH_CALUDE_nandan_earning_is_2000_l4102_410258


namespace NUMINAMATH_CALUDE_complex_expression_equals_nine_l4102_410234

theorem complex_expression_equals_nine :
  (Real.sqrt 2 - 3) ^ (0 : ℝ) - Real.sqrt 9 + |(-2 : ℝ)| + (-1/3 : ℝ) ^ (-2 : ℝ) = 9 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_equals_nine_l4102_410234


namespace NUMINAMATH_CALUDE_inkblot_area_bound_l4102_410288

/-- Represents an inkblot on a square sheet of paper -/
structure Inkblot where
  area : ℝ
  x_extent : ℝ
  y_extent : ℝ

/-- The theorem stating that the total area of inkblots does not exceed the side length of the square paper -/
theorem inkblot_area_bound (a : ℝ) (inkblots : List Inkblot) : a > 0 →
  (∀ i ∈ inkblots, i.area ≤ 1) →
  (∀ i ∈ inkblots, i.x_extent ≤ a ∧ i.y_extent ≤ a) →
  (∀ x : ℝ, x ≥ 0 ∧ x ≤ a → (inkblots.filter (fun i => i.x_extent > x)).length ≤ 1) →
  (∀ y : ℝ, y ≥ 0 ∧ y ≤ a → (inkblots.filter (fun i => i.y_extent > y)).length ≤ 1) →
  (inkblots.map (fun i => i.area)).sum ≤ a :=
by sorry

end NUMINAMATH_CALUDE_inkblot_area_bound_l4102_410288


namespace NUMINAMATH_CALUDE_millet_exceeds_half_on_day_three_l4102_410286

/-- Represents the proportion of millet in the feeder on a given day -/
def milletProportion (day : ℕ) : ℝ :=
  0.4 * (1 - (0.5 ^ day))

/-- The day when millet first exceeds half of the seeds -/
def milletExceedsHalfDay : ℕ :=
  3

theorem millet_exceeds_half_on_day_three :
  milletProportion milletExceedsHalfDay > 0.5 ∧
  ∀ d : ℕ, d < milletExceedsHalfDay → milletProportion d ≤ 0.5 :=
by sorry

end NUMINAMATH_CALUDE_millet_exceeds_half_on_day_three_l4102_410286

import Mathlib

namespace NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l1613_161356

theorem hyperbola_asymptote_slope (m : ℝ) : m > 0 →
  (∀ x y : ℝ, x^2/16 - y^2/25 = 1 ↔ (y = m*x ∨ y = -m*x)) →
  m = 5/4 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l1613_161356


namespace NUMINAMATH_CALUDE_smallest_value_in_different_bases_l1613_161311

/-- Converts a number from base b to decimal --/
def to_decimal (digits : List Nat) (b : Nat) : Nat :=
  digits.foldr (fun d acc => d + b * acc) 0

theorem smallest_value_in_different_bases :
  let base_9 := to_decimal [8, 5] 9
  let base_6 := to_decimal [2, 1, 0] 6
  let base_4 := to_decimal [1, 0, 0, 0] 4
  let base_2 := to_decimal [1, 1, 1, 1, 1, 1] 2
  base_2 = min base_9 (min base_6 (min base_4 base_2)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_value_in_different_bases_l1613_161311


namespace NUMINAMATH_CALUDE_coordinate_difference_of_P_l1613_161341

/-- Triangle ABC with vertices A(0,10), B(4,0), C(12,0) -/
def triangle_ABC : Set (ℝ × ℝ) :=
  {⟨0, 10⟩, ⟨4, 0⟩, ⟨12, 0⟩}

/-- Point P on line AC -/
def P : ℝ × ℝ := ⟨6, 5⟩

/-- Point Q on line BC -/
def Q : ℝ × ℝ := ⟨6, 0⟩

/-- Area of triangle PQC -/
def area_PQC : ℝ := 16

/-- Theorem: The positive difference between x and y coordinates of P is 1 -/
theorem coordinate_difference_of_P :
  P ∈ Set.Icc (0 : ℝ) 12 ×ˢ Set.Icc (0 : ℝ) 10 →
  Q.1 = P.1 →
  Q.2 = 0 →
  area_PQC = 16 →
  |P.1 - P.2| = 1 := by sorry

end NUMINAMATH_CALUDE_coordinate_difference_of_P_l1613_161341


namespace NUMINAMATH_CALUDE_lcm_hcf_problem_l1613_161393

theorem lcm_hcf_problem (a b : ℕ+) : 
  Nat.lcm a b = 2310 →
  Nat.gcd a b = 30 →
  b = 330 →
  a = 210 := by
sorry

end NUMINAMATH_CALUDE_lcm_hcf_problem_l1613_161393


namespace NUMINAMATH_CALUDE_find_k_n_l1613_161344

theorem find_k_n : ∃ (k n : ℕ), k * n^2 - k * n - n^2 + n = 94 ∧ k = 48 ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_find_k_n_l1613_161344


namespace NUMINAMATH_CALUDE_dan_marbles_l1613_161335

theorem dan_marbles (initial_marbles given_marbles : ℕ) 
  (h1 : initial_marbles = 64)
  (h2 : given_marbles = 14) :
  initial_marbles - given_marbles = 50 := by
  sorry

end NUMINAMATH_CALUDE_dan_marbles_l1613_161335


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1613_161308

theorem quadratic_inequality_solution (x : ℝ) : 
  -9 * x^2 + 6 * x + 1 < 0 ↔ (1 - Real.sqrt 2) / 3 < x ∧ x < (1 + Real.sqrt 2) / 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1613_161308


namespace NUMINAMATH_CALUDE_three_digit_factorial_sum_l1613_161399

theorem three_digit_factorial_sum : ∃ (x y z : ℕ), 
  (1 ≤ x ∧ x ≤ 9) ∧ 
  (0 ≤ y ∧ y ≤ 9) ∧ 
  (0 ≤ z ∧ z ≤ 9) ∧
  (100 * x + 10 * y + z = Nat.factorial x + Nat.factorial y + Nat.factorial z) ∧
  (x + y + z = 10) := by
  sorry

end NUMINAMATH_CALUDE_three_digit_factorial_sum_l1613_161399


namespace NUMINAMATH_CALUDE_gcd_of_quadratic_and_linear_l1613_161361

theorem gcd_of_quadratic_and_linear (a : ℤ) (h : ∃ k : ℤ, a = 2100 * k) :
  Int.gcd (a^2 + 14*a + 49) (a + 7) = 7 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_quadratic_and_linear_l1613_161361


namespace NUMINAMATH_CALUDE_extreme_value_implies_a_equals_5_l1613_161340

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + 3*x + 5

-- Define the derivative of f(x)
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + 3

-- Theorem statement
theorem extreme_value_implies_a_equals_5 (a : ℝ) :
  (∃ (ε : ℝ), ∀ (x : ℝ), |x + 3| < ε → f a (-3) ≥ f a x) →
  f' a (-3) = 0 →
  a = 5 :=
sorry

end NUMINAMATH_CALUDE_extreme_value_implies_a_equals_5_l1613_161340


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l1613_161382

theorem min_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = 1) :
  1/x + 1/y ≥ 3 + 2*Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l1613_161382


namespace NUMINAMATH_CALUDE_evaluate_expression_l1613_161305

theorem evaluate_expression : (16 ^ 24) / (64 ^ 8) = 16 ^ 12 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1613_161305


namespace NUMINAMATH_CALUDE_sum_of_fractions_equals_41_20_l1613_161323

theorem sum_of_fractions_equals_41_20 : 
  (2 + 4 + 6 + 8) / (1 + 3 + 5 + 7) + (1 + 3 + 5 + 7) / (2 + 4 + 6 + 8) = 41 / 20 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_equals_41_20_l1613_161323


namespace NUMINAMATH_CALUDE_chlorine_used_equals_chloromethane_formed_l1613_161387

/-- Represents the chemical reaction between Methane and Chlorine to form Chloromethane -/
structure ChemicalReaction where
  methane_initial : ℝ
  chloromethane_formed : ℝ

/-- Theorem stating that the moles of Chlorine used equals the moles of Chloromethane formed -/
theorem chlorine_used_equals_chloromethane_formed (reaction : ChemicalReaction)
  (h : reaction.methane_initial = reaction.chloromethane_formed) :
  reaction.chloromethane_formed = reaction.methane_initial :=
by sorry

end NUMINAMATH_CALUDE_chlorine_used_equals_chloromethane_formed_l1613_161387


namespace NUMINAMATH_CALUDE_dvd_packs_total_cost_l1613_161360

/-- Calculates the total cost of purchasing two packs of DVDs with given prices, discounts, and an additional discount for buying both. -/
def total_cost (price1 price2 discount1 discount2 additional_discount : ℕ) : ℕ :=
  (price1 - discount1) + (price2 - discount2) - additional_discount

/-- Theorem stating that the total cost of purchasing the two DVD packs is 111 dollars. -/
theorem dvd_packs_total_cost : 
  total_cost 76 85 25 15 10 = 111 := by
  sorry

end NUMINAMATH_CALUDE_dvd_packs_total_cost_l1613_161360


namespace NUMINAMATH_CALUDE_soda_survey_result_l1613_161357

/-- Given a survey of 520 people and a central angle of 220° for the "Soda" sector,
    prove that 317 people chose "Soda". -/
theorem soda_survey_result (total_surveyed : ℕ) (soda_angle : ℝ) :
  total_surveyed = 520 →
  soda_angle = 220 →
  ∃ (soda_count : ℕ),
    soda_count = 317 ∧
    (soda_count : ℝ) / total_surveyed * 360 ≥ soda_angle - 0.5 ∧
    (soda_count : ℝ) / total_surveyed * 360 < soda_angle + 0.5 :=
by sorry


end NUMINAMATH_CALUDE_soda_survey_result_l1613_161357


namespace NUMINAMATH_CALUDE_F_is_second_from_left_l1613_161306

-- Define a structure for rectangles
structure Rectangle where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ

-- Define the four rectangles
def F : Rectangle := ⟨7, 2, 5, 9⟩
def G : Rectangle := ⟨6, 9, 1, 3⟩
def H : Rectangle := ⟨2, 5, 7, 10⟩
def J : Rectangle := ⟨3, 1, 6, 8⟩

-- Define a function to check if two rectangles can connect
def canConnect (r1 r2 : Rectangle) : Prop :=
  (r1.a = r2.a) ∨ (r1.a = r2.b) ∨ (r1.a = r2.c) ∨ (r1.a = r2.d) ∨
  (r1.b = r2.a) ∨ (r1.b = r2.b) ∨ (r1.b = r2.c) ∨ (r1.b = r2.d) ∨
  (r1.c = r2.a) ∨ (r1.c = r2.b) ∨ (r1.c = r2.c) ∨ (r1.c = r2.d) ∨
  (r1.d = r2.a) ∨ (r1.d = r2.b) ∨ (r1.d = r2.c) ∨ (r1.d = r2.d)

-- Theorem stating that F is second from the left
theorem F_is_second_from_left :
  ∃ (left right : Rectangle), left ≠ F ∧ right ≠ F ∧
  canConnect left F ∧ canConnect F right ∧
  (∀ r : Rectangle, r ≠ F → r ≠ left → r ≠ right → ¬(canConnect left r ∧ canConnect r right)) :=
by
  sorry

end NUMINAMATH_CALUDE_F_is_second_from_left_l1613_161306


namespace NUMINAMATH_CALUDE_sum_a_plus_d_l1613_161379

theorem sum_a_plus_d (a b c d : ℤ) 
  (eq1 : a + b = 12) 
  (eq2 : b + c = 9) 
  (eq3 : c + d = 3) : 
  a + d = 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_a_plus_d_l1613_161379


namespace NUMINAMATH_CALUDE_quadratic_root_values_l1613_161315

theorem quadratic_root_values : 
  (Real.sqrt (9 - 8 * 0) = 3) ∧ 
  (Real.sqrt (9 - 8 * (1/2)) = Real.sqrt 5) ∧ 
  (Real.sqrt (9 - 8 * (-2)) = 5) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_values_l1613_161315


namespace NUMINAMATH_CALUDE_intersection_trajectory_l1613_161384

/-- 
Given points A(a,0) and B(b,0) on the x-axis and a point C(0,c) on the y-axis,
prove that the trajectory of the intersection point of line l (passing through O(0,0) 
and perpendicular to AC) and line BC is described by the given equation.
-/
theorem intersection_trajectory 
  (a b : ℝ) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (hab : a ≠ b) :
  ∃ (x y : ℝ → ℝ), 
    ∀ (c : ℝ), c ≠ 0 →
      (x c - b/2)^2 / (b^2/4) + (y c)^2 / (a*b/4) = 1 :=
sorry

end NUMINAMATH_CALUDE_intersection_trajectory_l1613_161384


namespace NUMINAMATH_CALUDE_inverse_composition_equals_neg_eight_ninths_l1613_161383

-- Define the function g
def g (x : ℝ) : ℝ := 3 * x + 7

-- Define the inverse function g⁻¹
noncomputable def g_inv (y : ℝ) : ℝ := (y - 7) / 3

-- Theorem statement
theorem inverse_composition_equals_neg_eight_ninths :
  g_inv (g_inv 20) = -8/9 := by
  sorry

end NUMINAMATH_CALUDE_inverse_composition_equals_neg_eight_ninths_l1613_161383


namespace NUMINAMATH_CALUDE_same_distance_different_time_l1613_161363

/-- Proves that given Joann's average speed and time, Fran needs to ride at a specific speed to cover the same distance in less time -/
theorem same_distance_different_time (joann_speed : ℝ) (joann_time : ℝ) (fran_time : ℝ) 
  (h1 : joann_speed = 14)
  (h2 : joann_time = 4)
  (h3 : fran_time = 2) :
  joann_speed * joann_time = (joann_speed * joann_time / fran_time) * fran_time :=
by sorry

end NUMINAMATH_CALUDE_same_distance_different_time_l1613_161363


namespace NUMINAMATH_CALUDE_real_solution_implies_m_positive_l1613_161307

theorem real_solution_implies_m_positive (x m : ℝ) : 
  (∃ x : ℝ, 3^x - m = 0) → m > 0 := by
sorry

end NUMINAMATH_CALUDE_real_solution_implies_m_positive_l1613_161307


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l1613_161362

theorem expression_simplification_and_evaluation :
  ∀ x y : ℝ, (x - 2)^2 + |y + 1| = 0 →
  3 * x^2 * y - (2 * x^2 * y - 3 * (2 * x * y - x^2 * y) + 5 * x * y) = 6 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l1613_161362


namespace NUMINAMATH_CALUDE_arithmetic_and_geometric_sequence_equal_l1613_161364

/-- A sequence is both arithmetic and geometric if and only if all its terms are equal -/
theorem arithmetic_and_geometric_sequence_equal (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) - a n = a 1 - a 0) ∧ 
  (∀ n : ℕ, a n ≠ 0 → a (n + 1) / a n = a 1 / a 0) ↔ 
  (∀ n m : ℕ, a n = a m) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_and_geometric_sequence_equal_l1613_161364


namespace NUMINAMATH_CALUDE_break_difference_l1613_161398

def work_duration : ℕ := 240
def water_break_interval : ℕ := 20
def sitting_break_interval : ℕ := 120

def water_breaks : ℕ := work_duration / water_break_interval
def sitting_breaks : ℕ := work_duration / sitting_break_interval

theorem break_difference : water_breaks - sitting_breaks = 10 := by
  sorry

end NUMINAMATH_CALUDE_break_difference_l1613_161398


namespace NUMINAMATH_CALUDE_pi_irrational_in_set_l1613_161324

theorem pi_irrational_in_set : ∃ x ∈ ({-2, 0, Real.sqrt 9, Real.pi} : Set ℝ), Irrational x :=
by sorry

end NUMINAMATH_CALUDE_pi_irrational_in_set_l1613_161324


namespace NUMINAMATH_CALUDE_student_skills_l1613_161374

theorem student_skills (total : ℕ) (chess_unable : ℕ) (puzzle_unable : ℕ) (code_unable : ℕ) :
  total = 120 →
  chess_unable = 50 →
  puzzle_unable = 75 →
  code_unable = 40 →
  (∃ (two_skills : ℕ), two_skills = 75 ∧
    two_skills = (total - chess_unable) + (total - puzzle_unable) + (total - code_unable) - total) :=
by sorry

end NUMINAMATH_CALUDE_student_skills_l1613_161374


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1613_161353

def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := {x : ℤ | x^2 ≤ 2}

theorem intersection_of_A_and_B : A ∩ B = {-1, 0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1613_161353


namespace NUMINAMATH_CALUDE_largest_integer_m_l1613_161300

theorem largest_integer_m (x m : ℝ) : 
  (3 : ℝ) / 3 + 2 * m < -3 → 
  ∀ k : ℤ, (k : ℝ) > m → k ≤ -3 :=
sorry

end NUMINAMATH_CALUDE_largest_integer_m_l1613_161300


namespace NUMINAMATH_CALUDE_product_sum_equality_l1613_161309

theorem product_sum_equality : 15 * 35 + 45 * 15 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_equality_l1613_161309


namespace NUMINAMATH_CALUDE_paint_cost_contribution_l1613_161367

-- Define the given conditions
def wall_area : ℝ := 1600
def paint_coverage : ℝ := 400
def paint_cost_per_gallon : ℝ := 45
def number_of_coats : ℕ := 2

-- Define the theorem
theorem paint_cost_contribution :
  let total_gallons := (wall_area / paint_coverage) * number_of_coats
  let total_cost := total_gallons * paint_cost_per_gallon
  let individual_contribution := total_cost / 2
  individual_contribution = 180 := by sorry

end NUMINAMATH_CALUDE_paint_cost_contribution_l1613_161367


namespace NUMINAMATH_CALUDE_range_of_a_for_always_positive_quadratic_l1613_161322

theorem range_of_a_for_always_positive_quadratic :
  {a : ℝ | ∀ x : ℝ, x^2 + 2*a*x + a > 0} = Set.Ioo 0 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_for_always_positive_quadratic_l1613_161322


namespace NUMINAMATH_CALUDE_fathers_age_is_45_l1613_161348

/-- Proves that the father's age is 45 given the problem conditions -/
theorem fathers_age_is_45 (F C : ℕ) : 
  F = 3 * C →  -- Father's age is three times the sum of the ages of his two children
  F + 5 = 2 * (C + 10) →  -- After 5 years, father's age will be twice the sum of age of two children
  F = 45 := by
sorry

end NUMINAMATH_CALUDE_fathers_age_is_45_l1613_161348


namespace NUMINAMATH_CALUDE_investment_problem_l1613_161359

/-- Given two investors P and Q, where the profit is divided in the ratio 3:5
    and P invested 12000, prove that Q invested 20000. -/
theorem investment_problem (P Q : ℕ) (profit_ratio : ℚ) (P_investment : ℕ) :
  profit_ratio = 3 / 5 →
  P_investment = 12000 →
  Q = 20000 :=
by sorry

end NUMINAMATH_CALUDE_investment_problem_l1613_161359


namespace NUMINAMATH_CALUDE_expand_product_l1613_161380

theorem expand_product (x : ℝ) : (3 * x - 2) * (2 * x + 4) = 6 * x^2 + 8 * x - 8 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l1613_161380


namespace NUMINAMATH_CALUDE_train_cost_XY_is_900_l1613_161320

/-- Represents the cost of a train journey in dollars -/
def train_cost (distance : ℝ) : ℝ := 0.20 * distance

/-- The cities and their distances -/
structure Cities where
  XY : ℝ
  XZ : ℝ

/-- The problem setup -/
def piravena_journey : Cities where
  XY := 4500
  XZ := 4000

theorem train_cost_XY_is_900 :
  train_cost piravena_journey.XY = 900 := by sorry

end NUMINAMATH_CALUDE_train_cost_XY_is_900_l1613_161320


namespace NUMINAMATH_CALUDE_green_ball_fraction_l1613_161355

theorem green_ball_fraction (total : ℕ) (green blue yellow white : ℕ) :
  blue = total / 8 →
  yellow = total / 12 →
  white = 26 →
  blue = 6 →
  green + blue + yellow + white = total →
  green = total / 4 := by
  sorry

end NUMINAMATH_CALUDE_green_ball_fraction_l1613_161355


namespace NUMINAMATH_CALUDE_greatest_x_value_l1613_161321

theorem greatest_x_value : 
  let f (x : ℝ) := ((5*x - 20) / (4*x - 5))^2 + (5*x - 20) / (4*x - 5)
  ∃ (x_max : ℝ), x_max = 50/29 ∧ 
    (∀ (x : ℝ), f x = 18 → x ≤ x_max) ∧
    (f x_max = 18) :=
by sorry

end NUMINAMATH_CALUDE_greatest_x_value_l1613_161321


namespace NUMINAMATH_CALUDE_unique_pin_l1613_161396

def is_valid_pin (pin : Nat) : Prop :=
  pin ≥ 1000 ∧ pin < 10000 ∧
  let first_digit := pin / 1000
  let last_three_digits := pin % 1000
  10 * last_three_digits + first_digit = 3 * pin - 6

theorem unique_pin : ∃! pin, is_valid_pin pin ∧ pin = 2856 := by
  sorry

end NUMINAMATH_CALUDE_unique_pin_l1613_161396


namespace NUMINAMATH_CALUDE_problem_1_l1613_161392

theorem problem_1 : 2 + (-5) - (-4) + |(-3)| = 4 := by sorry

end NUMINAMATH_CALUDE_problem_1_l1613_161392


namespace NUMINAMATH_CALUDE_inequality_chain_l1613_161338

theorem inequality_chain (x : ℝ) (h : 1 < x ∧ x < 2) :
  ((Real.log x) / x) ^ 2 < (Real.log x) / x ∧ (Real.log x) / x < (Real.log (x^2)) / (x^2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_chain_l1613_161338


namespace NUMINAMATH_CALUDE_binomial_coefficient_sum_squared_difference_l1613_161318

theorem binomial_coefficient_sum_squared_difference (a₀ a₁ a₂ a₃ : ℝ) : 
  (∀ x, (Real.sqrt 5 * x - 1)^3 = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3) →
  (a₀ + a₂)^2 - (a₁ + a₃)^2 = -64 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_sum_squared_difference_l1613_161318


namespace NUMINAMATH_CALUDE_chest_contents_l1613_161329

/-- Represents the types of coins that can be in a chest -/
inductive CoinType
  | Gold
  | Silver
  | Copper

/-- Represents a chest with its inscription and actual content -/
structure Chest where
  inscription : CoinType → Prop
  content : CoinType

/-- The problem setup -/
def chestProblem (c1 c2 c3 : Chest) : Prop :=
  -- Inscriptions
  c1.inscription = fun t => t = CoinType.Gold ∧
  c2.inscription = fun t => t = CoinType.Silver ∧
  c3.inscription = fun t => t = CoinType.Gold ∨ t = CoinType.Silver ∧
  -- All inscriptions are incorrect
  ¬c1.inscription c1.content ∧
  ¬c2.inscription c2.content ∧
  ¬c3.inscription c3.content ∧
  -- One of each type of coin
  c1.content ≠ c2.content ∧
  c2.content ≠ c3.content ∧
  c3.content ≠ c1.content

/-- The theorem to prove -/
theorem chest_contents (c1 c2 c3 : Chest) :
  chestProblem c1 c2 c3 →
  c1.content = CoinType.Silver ∧
  c2.content = CoinType.Gold ∧
  c3.content = CoinType.Copper :=
by
  sorry

end NUMINAMATH_CALUDE_chest_contents_l1613_161329


namespace NUMINAMATH_CALUDE_fraction_simplification_l1613_161301

theorem fraction_simplification (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : a ≠ b) :
  (a^2 - b^2) / (a * b) - (a * b - b^2) / (a * b - a^2) = a / b :=
by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1613_161301


namespace NUMINAMATH_CALUDE_two_minus_i_in_fourth_quadrant_l1613_161354

/-- A complex number z is in the fourth quadrant if its real part is positive and its imaginary part is negative. -/
def in_fourth_quadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im < 0

/-- The complex number 2 - i is in the fourth quadrant. -/
theorem two_minus_i_in_fourth_quadrant :
  in_fourth_quadrant (2 - I) := by
  sorry

end NUMINAMATH_CALUDE_two_minus_i_in_fourth_quadrant_l1613_161354


namespace NUMINAMATH_CALUDE_commemorative_book_sales_l1613_161389

/-- Profit function for commemorative book sales -/
def profit (x : ℝ) : ℝ := (x - 20) * (-2 * x + 80)

/-- Theorem for commemorative book sales problem -/
theorem commemorative_book_sales 
  (x : ℝ) 
  (h1 : 20 ≤ x ∧ x ≤ 28) : 
  (∃ (x : ℝ), profit x = 150 ∧ x = 25) ∧ 
  (∀ (y : ℝ), 20 ≤ y ∧ y ≤ 28 → profit y ≤ profit 28) ∧
  profit 28 = 192 := by
  sorry


end NUMINAMATH_CALUDE_commemorative_book_sales_l1613_161389


namespace NUMINAMATH_CALUDE_investment_interest_theorem_l1613_161328

/-- Calculates the total interest earned from two investments -/
def totalInterest (totalAmount : ℝ) (amount1 : ℝ) (rate1 : ℝ) (rate2 : ℝ) : ℝ :=
  let amount2 := totalAmount - amount1
  let interest1 := amount1 * rate1
  let interest2 := amount2 * rate2
  interest1 + interest2

/-- Proves that investing $9000 with $4000 at 8% and the rest at 9% yields $770 in interest -/
theorem investment_interest_theorem :
  totalInterest 9000 4000 0.08 0.09 = 770 := by
  sorry

end NUMINAMATH_CALUDE_investment_interest_theorem_l1613_161328


namespace NUMINAMATH_CALUDE_sum_24_probability_l1613_161352

/-- The number of ways to achieve a sum of 24 with 10 fair standard 6-sided dice -/
def ways_to_sum_24 : ℕ := 817190

/-- The number of possible outcomes when throwing 10 fair standard 6-sided dice -/
def total_outcomes : ℕ := 6^10

/-- The probability of achieving a sum of 24 when throwing 10 fair standard 6-sided dice -/
def prob_sum_24 : ℚ := ways_to_sum_24 / total_outcomes

theorem sum_24_probability :
  ways_to_sum_24 = 817190 ∧
  total_outcomes = 6^10 ∧
  prob_sum_24 = ways_to_sum_24 / total_outcomes :=
sorry

end NUMINAMATH_CALUDE_sum_24_probability_l1613_161352


namespace NUMINAMATH_CALUDE_emberly_walk_distance_l1613_161391

/-- Emberly's walking problem -/
theorem emberly_walk_distance :
  ∀ (total_days : ℕ) (days_not_walked : ℕ) (total_miles : ℕ),
    total_days = 31 →
    days_not_walked = 4 →
    total_miles = 108 →
    (total_miles : ℚ) / (total_days - days_not_walked : ℚ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_emberly_walk_distance_l1613_161391


namespace NUMINAMATH_CALUDE_edward_spent_thirteen_l1613_161337

/-- The amount of money Edward spent -/
def amount_spent (initial_amount current_amount : ℕ) : ℕ :=
  initial_amount - current_amount

/-- Theorem: Edward spent $13 -/
theorem edward_spent_thirteen : amount_spent 19 6 = 13 := by
  sorry

end NUMINAMATH_CALUDE_edward_spent_thirteen_l1613_161337


namespace NUMINAMATH_CALUDE_only_7_3_1_wins_for_second_player_l1613_161343

/-- Represents a wall configuration in the game --/
structure WallConfig :=
  (walls : List Nat)

/-- Calculates the nim-value of a single wall --/
def nimValue (wall : Nat) : Nat :=
  sorry

/-- Calculates the nim-sum of a list of nim-values --/
def nimSum (values : List Nat) : Nat :=
  sorry

/-- Determines if a given configuration is a winning position for the second player --/
def isWinningForSecondPlayer (config : WallConfig) : Prop :=
  nimSum (config.walls.map nimValue) = 0

/-- The main theorem stating that (7,3,1) is the only winning configuration for the second player --/
theorem only_7_3_1_wins_for_second_player :
  let configs := [
    WallConfig.mk [7, 1, 1],
    WallConfig.mk [7, 2, 1],
    WallConfig.mk [7, 2, 2],
    WallConfig.mk [7, 3, 1],
    WallConfig.mk [7, 3, 2]
  ]
  ∀ config ∈ configs, isWinningForSecondPlayer config ↔ config = WallConfig.mk [7, 3, 1] :=
  sorry

end NUMINAMATH_CALUDE_only_7_3_1_wins_for_second_player_l1613_161343


namespace NUMINAMATH_CALUDE_specific_figure_perimeter_l1613_161316

/-- A figure composed of unit squares arranged in a specific pattern -/
structure UnitSquareFigure where
  horizontalSegments : ℕ
  verticalSegments : ℕ

/-- The perimeter of a UnitSquareFigure -/
def perimeter (figure : UnitSquareFigure) : ℕ :=
  figure.horizontalSegments + figure.verticalSegments

/-- The specific figure from the problem -/
def specificFigure : UnitSquareFigure :=
  { horizontalSegments := 16, verticalSegments := 10 }

/-- Theorem stating that the perimeter of the specific figure is 26 -/
theorem specific_figure_perimeter :
  perimeter specificFigure = 26 := by sorry

end NUMINAMATH_CALUDE_specific_figure_perimeter_l1613_161316


namespace NUMINAMATH_CALUDE_inequality_range_l1613_161314

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, |x - 3| + |x + 1| > a) → a < 4 := by
sorry

end NUMINAMATH_CALUDE_inequality_range_l1613_161314


namespace NUMINAMATH_CALUDE_smallest_battleship_board_l1613_161342

/-- Represents the types of ships in Battleship -/
inductive ShipType
  | OneByFour
  | OneByThree
  | OneByTwo
  | OneByOne

/-- The set of ships in a standard Battleship game -/
def battleshipSet : List ShipType :=
  [ShipType.OneByFour] ++
  List.replicate 2 ShipType.OneByThree ++
  List.replicate 3 ShipType.OneByTwo ++
  List.replicate 4 ShipType.OneByOne

/-- Calculates the number of nodes a ship occupies, including its surrounding space -/
def nodesOccupied (ship : ShipType) : Nat :=
  match ship with
  | ShipType.OneByFour => 10
  | ShipType.OneByThree => 8
  | ShipType.OneByTwo => 6
  | ShipType.OneByOne => 4

/-- The smallest square board size for Battleship -/
def smallestBoardSize : Nat := 7

theorem smallest_battleship_board :
  (∀ n : Nat, n < smallestBoardSize → 
    (List.sum (List.map nodesOccupied battleshipSet) > (n + 1)^2)) ∧
  (List.sum (List.map nodesOccupied battleshipSet) ≤ (smallestBoardSize + 1)^2) := by
  sorry

#eval smallestBoardSize  -- Should output 7

end NUMINAMATH_CALUDE_smallest_battleship_board_l1613_161342


namespace NUMINAMATH_CALUDE_product_sum_fractions_l1613_161370

theorem product_sum_fractions : (2 * 3 * 4) * (1 / 2 + 1 / 3 + 1 / 4) = 26 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_fractions_l1613_161370


namespace NUMINAMATH_CALUDE_lcm_of_9_12_15_l1613_161317

theorem lcm_of_9_12_15 : Nat.lcm 9 (Nat.lcm 12 15) = 180 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_9_12_15_l1613_161317


namespace NUMINAMATH_CALUDE_u_difference_divisible_l1613_161349

/-- Sequence u defined recursively -/
def u (a : ℕ+) : ℕ → ℕ
  | 0 => 1
  | n + 1 => a.val ^ u a n

/-- Theorem stating that n! divides u_{n+1} - u_n for all n ≥ 1 -/
theorem u_difference_divisible (a : ℕ+) (n : ℕ) (h : n ≥ 1) :
  ∃ k : ℕ, u a (n + 1) - u a n = k * n! :=
sorry

end NUMINAMATH_CALUDE_u_difference_divisible_l1613_161349


namespace NUMINAMATH_CALUDE_distinct_fm_pairs_count_l1613_161345

/-- Represents the gender of a person -/
inductive Gender
| Male
| Female

/-- Represents a seating arrangement of 5 people around a round table -/
def SeatingArrangement := Vector Gender 5

/-- Counts the number of people sitting next to at least one female -/
def count_next_to_female (arrangement : SeatingArrangement) : Nat :=
  sorry

/-- Counts the number of people sitting next to at least one male -/
def count_next_to_male (arrangement : SeatingArrangement) : Nat :=
  sorry

/-- Generates all distinct seating arrangements -/
def all_distinct_arrangements : List SeatingArrangement :=
  sorry

/-- The main theorem stating that there are exactly 8 distinct (f, m) pairs -/
theorem distinct_fm_pairs_count :
  (all_distinct_arrangements.map (λ arr => (count_next_to_female arr, count_next_to_male arr))).toFinset.card = 8 :=
  sorry

end NUMINAMATH_CALUDE_distinct_fm_pairs_count_l1613_161345


namespace NUMINAMATH_CALUDE_sin_50_plus_sqrt3_tan_10_eq_1_l1613_161331

theorem sin_50_plus_sqrt3_tan_10_eq_1 :
  Real.sin (50 * π / 180) * (1 + Real.sqrt 3 * Real.tan (10 * π / 180)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_50_plus_sqrt3_tan_10_eq_1_l1613_161331


namespace NUMINAMATH_CALUDE_colors_needed_l1613_161350

/-- The number of people coloring the planets -/
def num_people : ℕ := 3

/-- The number of planets to be colored -/
def num_planets : ℕ := 8

/-- The total number of colors needed -/
def total_colors : ℕ := num_people * num_planets

/-- Theorem stating that the total number of colors needed is 24 -/
theorem colors_needed : total_colors = 24 := by sorry

end NUMINAMATH_CALUDE_colors_needed_l1613_161350


namespace NUMINAMATH_CALUDE_max_a_value_l1613_161373

theorem max_a_value (a b : ℤ) (h1 : a > b) (h2 : b > 0) (h3 : a + b + a * b = 119) :
  a ≤ 59 ∧ ∃ (a' b' : ℤ), a' = 59 ∧ b' = 1 ∧ a' > b' ∧ b' > 0 ∧ a' + b' + a' * b' = 119 :=
sorry

end NUMINAMATH_CALUDE_max_a_value_l1613_161373


namespace NUMINAMATH_CALUDE_total_farm_tax_collected_l1613_161327

/-- Theorem: Total farm tax collected from a village
Given:
- Farm tax is levied on 75% of the cultivated land
- Mr. William paid $480 as farm tax
- Mr. William's land represents 16.666666666666668% of the total taxable land in the village

Prove: The total amount collected through farm tax from the village is $2880 -/
theorem total_farm_tax_collected (william_tax : ℝ) (william_land_percentage : ℝ) 
  (h1 : william_tax = 480)
  (h2 : william_land_percentage = 16.666666666666668) :
  william_tax / (william_land_percentage / 100) = 2880 := by
  sorry

#check total_farm_tax_collected

end NUMINAMATH_CALUDE_total_farm_tax_collected_l1613_161327


namespace NUMINAMATH_CALUDE_negative_cube_squared_l1613_161378

theorem negative_cube_squared (a : ℝ) : -(-3*a)^2 = -9*a^2 := by
  sorry

end NUMINAMATH_CALUDE_negative_cube_squared_l1613_161378


namespace NUMINAMATH_CALUDE_geometric_number_difference_l1613_161351

/-- A 4-digit number is geometric if it has 4 distinct digits forming a geometric sequence from left to right. -/
def IsGeometric (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧
  ∃ a b c d r : ℕ,
    n = a * 1000 + b * 100 + c * 10 + d ∧
    a ≠ 0 ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    b = a * r ∧ c = a * r^2 ∧ d = a * r^3

/-- The largest 4-digit geometric number -/
def LargestGeometric : ℕ := 9648

/-- The smallest 4-digit geometric number -/
def SmallestGeometric : ℕ := 1248

theorem geometric_number_difference :
  IsGeometric LargestGeometric ∧
  IsGeometric SmallestGeometric ∧
  (∀ n : ℕ, IsGeometric n → SmallestGeometric ≤ n ∧ n ≤ LargestGeometric) ∧
  LargestGeometric - SmallestGeometric = 8400 := by
  sorry

end NUMINAMATH_CALUDE_geometric_number_difference_l1613_161351


namespace NUMINAMATH_CALUDE_simplify_power_expression_l1613_161346

theorem simplify_power_expression (x : ℝ) : (3 * x^4)^4 = 81 * x^16 := by
  sorry

end NUMINAMATH_CALUDE_simplify_power_expression_l1613_161346


namespace NUMINAMATH_CALUDE_arithmetic_sequence_perfect_squares_l1613_161372

/-- An arithmetic sequence of natural numbers -/
def ArithmeticSequence (a d : ℕ) : ℕ → ℕ := fun n ↦ a + n * d

/-- A number is a perfect square -/
def IsPerfectSquare (n : ℕ) : Prop := ∃ k : ℕ, n = k * k

/-- The theorem stating that if an arithmetic sequence of natural numbers contains
    one perfect square, it contains infinitely many perfect squares -/
theorem arithmetic_sequence_perfect_squares
  (a d : ℕ) -- First term and common difference of the arithmetic sequence
  (h : ∃ n : ℕ, IsPerfectSquare (ArithmeticSequence a d n)) :
  ∀ m : ℕ, ∃ k > m, IsPerfectSquare (ArithmeticSequence a d k) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_perfect_squares_l1613_161372


namespace NUMINAMATH_CALUDE_league_teams_count_league_teams_count_proof_l1613_161313

theorem league_teams_count : ℕ → Prop :=
  fun n => (n * (n - 1) / 2 = 45) → n = 10

-- The proof is omitted
theorem league_teams_count_proof : league_teams_count 10 := by
  sorry

end NUMINAMATH_CALUDE_league_teams_count_league_teams_count_proof_l1613_161313


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l1613_161334

theorem complex_modulus_problem (z : ℂ) (h : z * (1 + Complex.I) = 2 * Complex.I) : 
  Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l1613_161334


namespace NUMINAMATH_CALUDE_fourth_term_of_geometric_sequence_l1613_161302

/-- Given a geometric sequence where:
    - The first term is 4
    - The second term is 12y
    - The third term is 36y^3
    Prove that the fourth term is 108y^5 -/
theorem fourth_term_of_geometric_sequence (y : ℝ) :
  let a₁ : ℝ := 4
  let a₂ : ℝ := 12 * y
  let a₃ : ℝ := 36 * y^3
  let a₄ : ℝ := 108 * y^5
  (∃ (r : ℝ), a₂ = a₁ * r ∧ a₃ = a₂ * r ∧ a₄ = a₃ * r) :=
by
  sorry

end NUMINAMATH_CALUDE_fourth_term_of_geometric_sequence_l1613_161302


namespace NUMINAMATH_CALUDE_equal_roots_implies_k_eq_four_l1613_161371

/-- 
A quadratic equation ax^2 + bx + c = 0 has two equal real roots if and only if 
its discriminant b^2 - 4ac is equal to 0.
-/
def has_two_equal_real_roots (a b c : ℝ) : Prop :=
  b^2 - 4*a*c = 0

/-- 
Given a quadratic equation kx^2 - 2kx + 4 = 0 with two equal real roots,
prove that k = 4.
-/
theorem equal_roots_implies_k_eq_four :
  ∀ k : ℝ, k ≠ 0 → has_two_equal_real_roots k (-2*k) 4 → k = 4 :=
by sorry

end NUMINAMATH_CALUDE_equal_roots_implies_k_eq_four_l1613_161371


namespace NUMINAMATH_CALUDE_max_product_sum_l1613_161395

theorem max_product_sum (a b c d : ℕ) : 
  a ∈ ({2, 3, 4, 5} : Set ℕ) → 
  b ∈ ({2, 3, 4, 5} : Set ℕ) → 
  c ∈ ({2, 3, 4, 5} : Set ℕ) → 
  d ∈ ({2, 3, 4, 5} : Set ℕ) → 
  a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d → 
  (a * b + b * c + c * d + d * a) ≤ 49 :=
by sorry

end NUMINAMATH_CALUDE_max_product_sum_l1613_161395


namespace NUMINAMATH_CALUDE_jeans_pricing_l1613_161325

theorem jeans_pricing (C : ℝ) (R : ℝ) :
  (C > 0) →
  (R > C) →
  (1.40 * R = 1.96 * C) →
  ((R - C) / C * 100 = 40) :=
by sorry

end NUMINAMATH_CALUDE_jeans_pricing_l1613_161325


namespace NUMINAMATH_CALUDE_star_commutative_l1613_161385

/-- Binary operation ★ defined for integers -/
def star (a b : ℤ) : ℤ := a^2 + b^2

/-- Theorem stating that ★ is commutative for all integers -/
theorem star_commutative : ∀ (a b : ℤ), star a b = star b a := by
  sorry

end NUMINAMATH_CALUDE_star_commutative_l1613_161385


namespace NUMINAMATH_CALUDE_new_year_money_distribution_l1613_161339

/-- Represents the distribution of money to three grandsons --/
structure MoneyDistribution :=
  (grandson1 : ℕ)
  (grandson2 : ℕ)
  (grandson3 : ℕ)

/-- Checks if a distribution is valid according to the problem conditions --/
def is_valid_distribution (d : MoneyDistribution) : Prop :=
  -- Total sum is 300
  d.grandson1 + d.grandson2 + d.grandson3 = 300 ∧
  -- Each amount is divisible by 10 (smallest denomination)
  d.grandson1 % 10 = 0 ∧ d.grandson2 % 10 = 0 ∧ d.grandson3 % 10 = 0 ∧
  -- Each amount is one of the allowed denominations (50, 20, or 10)
  (d.grandson1 % 50 = 0 ∨ d.grandson1 % 20 = 0 ∨ d.grandson1 % 10 = 0) ∧
  (d.grandson2 % 50 = 0 ∨ d.grandson2 % 20 = 0 ∨ d.grandson2 % 10 = 0) ∧
  (d.grandson3 % 50 = 0 ∨ d.grandson3 % 20 = 0 ∨ d.grandson3 % 10 = 0) ∧
  -- Number of bills condition
  (d.grandson1 / 10 = (d.grandson2 / 20) * (d.grandson3 / 50) ∨
   d.grandson2 / 20 = (d.grandson1 / 10) * (d.grandson3 / 50) ∨
   d.grandson3 / 50 = (d.grandson1 / 10) * (d.grandson2 / 20))

/-- The theorem to be proved --/
theorem new_year_money_distribution :
  ∀ d : MoneyDistribution,
    is_valid_distribution d →
    (d = ⟨100, 100, 100⟩ ∨ d = ⟨90, 60, 150⟩ ∨ d = ⟨90, 150, 60⟩ ∨
     d = ⟨60, 90, 150⟩ ∨ d = ⟨60, 150, 90⟩ ∨ d = ⟨150, 60, 90⟩ ∨
     d = ⟨150, 90, 60⟩) :=
by sorry


end NUMINAMATH_CALUDE_new_year_money_distribution_l1613_161339


namespace NUMINAMATH_CALUDE_circumscribed_sphere_radius_rectangular_solid_l1613_161386

/-- For a rectangular solid with edges a, b, and c, the radius R of its circumscribed sphere
    satisfies the equation 4R² = a² + b² + c². -/
theorem circumscribed_sphere_radius_rectangular_solid
  (a b c R : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  4 * R^2 = a^2 + b^2 + c^2 :=
sorry

end NUMINAMATH_CALUDE_circumscribed_sphere_radius_rectangular_solid_l1613_161386


namespace NUMINAMATH_CALUDE_ellipse_properties_l1613_161368

/-- An ellipse with equation x^2/2 + y^2 = 1 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p | p.1^2 / 2 + p.2^2 = 1}

/-- The foci of the ellipse -/
def F₁ : ℝ × ℝ := (-1, 0)
def F₂ : ℝ × ℝ := (1, 0)

/-- The circle with diameter F₁F₂ -/
def Circle : Set (ℝ × ℝ) :=
  {p | (p.1^2 + p.2^2) = 1}

/-- The line x + y - √2 = 0 -/
def Line : Set (ℝ × ℝ) :=
  {p | p.1 + p.2 = Real.sqrt 2}

/-- The dot product of vectors PF₁ and PF₂ -/
def dotProduct (P : ℝ × ℝ) : ℝ :=
  (F₁.1 - P.1) * (F₂.1 - P.1) + (F₁.2 - P.2) * (F₂.2 - P.2)

theorem ellipse_properties :
  (∀ p ∈ Circle, p ∈ Line) ∧
  (∀ P ∈ Ellipse, dotProduct P ≥ 0 ∧ ∃ Q ∈ Ellipse, dotProduct Q = 0) :=
sorry

end NUMINAMATH_CALUDE_ellipse_properties_l1613_161368


namespace NUMINAMATH_CALUDE_percentage_less_than_l1613_161332

theorem percentage_less_than (w x y z : ℝ) 
  (hw : w = 0.60 * x) 
  (hz1 : z = 0.54 * y) 
  (hz2 : z = 1.50 * w) : 
  x = 0.60 * y := by
sorry

end NUMINAMATH_CALUDE_percentage_less_than_l1613_161332


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l1613_161310

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (x^2 = 10*x - 24) → (∃ y : ℝ, y^2 = 10*y - 24 ∧ x + y = 10) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l1613_161310


namespace NUMINAMATH_CALUDE_rectangle_areas_sum_l1613_161312

theorem rectangle_areas_sum : 
  let widths : List ℕ := [2, 3, 4, 5, 6, 7, 8]
  let lengths : List ℕ := [5, 8, 11, 14, 17, 20, 23]
  (widths.zip lengths).map (fun (w, l) => w * l) |>.sum = 574 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_areas_sum_l1613_161312


namespace NUMINAMATH_CALUDE_square_sum_equals_37_l1613_161388

theorem square_sum_equals_37 (x y : ℝ) 
  (h1 : y + 3 = (x - 3)^2)
  (h2 : x + 3 = (y - 3)^2)
  (h3 : x ≠ y) :
  x^2 + y^2 = 37 := by
sorry

end NUMINAMATH_CALUDE_square_sum_equals_37_l1613_161388


namespace NUMINAMATH_CALUDE_sum_even_not_square_or_cube_l1613_161333

theorem sum_even_not_square_or_cube (n : ℕ+) :
  ∀ k m : ℕ+, (n : ℕ) * (n + 1) ≠ k ^ 2 ∧ (n : ℕ) * (n + 1) ≠ m ^ 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_even_not_square_or_cube_l1613_161333


namespace NUMINAMATH_CALUDE_k_squared_test_probability_two_males_l1613_161394

-- Define the contingency table
def contingency_table : Matrix (Fin 2) (Fin 2) ℕ :=
  ![![30, 45],
    ![15, 10]]

-- Define the total number of people surveyed
def total_surveyed : ℕ := 100

-- Define the K² formula
def k_squared (a b c d : ℕ) : ℚ :=
  (total_surveyed * (a * d - b * c)^2 : ℚ) /
  ((a + b) * (c + d) * (a + c) * (b + d))

-- Define the critical value for 95% confidence
def critical_value : ℚ := 3841 / 1000

-- Theorem for the K² test
theorem k_squared_test :
  k_squared (contingency_table 0 0) (contingency_table 0 1)
            (contingency_table 1 0) (contingency_table 1 1) < critical_value := by
  sorry

-- Define the number of healthy living people
def healthy_living : ℕ := 45

-- Define the number of healthy living males
def healthy_males : ℕ := 30

-- Theorem for the probability of selecting two males
theorem probability_two_males :
  (Nat.choose healthy_males 2 : ℚ) / (Nat.choose healthy_living 2) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_k_squared_test_probability_two_males_l1613_161394


namespace NUMINAMATH_CALUDE_exponential_equation_and_inequality_l1613_161336

/-- Given a > 0 and a ≠ 1, this theorem proves the conditions for equality and inequality
    between a^(3x+1) and a^(-2x) -/
theorem exponential_equation_and_inequality (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x : ℝ, a^(3*x + 1) = a^(-2*x) ↔ x = 1/5) ∧
  (∀ x : ℝ, (a > 1 → (a^(3*x + 1) < a^(-2*x) ↔ x < 1/5)) ∧
            (a < 1 → (a^(3*x + 1) < a^(-2*x) ↔ x > 1/5))) :=
by sorry

end NUMINAMATH_CALUDE_exponential_equation_and_inequality_l1613_161336


namespace NUMINAMATH_CALUDE_acid_solution_concentration_l1613_161397

theorem acid_solution_concentration 
  (x : ℝ) -- original concentration
  (h1 : 0.5 * x + 0.5 * 30 = 40) -- mixing equation
  : x = 50 := by
  sorry

end NUMINAMATH_CALUDE_acid_solution_concentration_l1613_161397


namespace NUMINAMATH_CALUDE_root_relationship_l1613_161319

theorem root_relationship (m n a b : ℝ) : 
  (∀ x, 3 - (x - m) * (x - n) = 0 ↔ x = a ∨ x = b) →
  a < m ∧ m < n ∧ n < b :=
sorry

end NUMINAMATH_CALUDE_root_relationship_l1613_161319


namespace NUMINAMATH_CALUDE_missing_number_is_eight_l1613_161358

-- Define the structure of the pyramid
def Pyramid (a b c d e : ℕ) : Prop :=
  b * c = d ∧ c * a = e ∧ d * e = 3360

-- Theorem statement
theorem missing_number_is_eight :
  ∃ (x : ℕ), Pyramid 8 6 7 42 x ∧ x > 0 :=
by sorry

end NUMINAMATH_CALUDE_missing_number_is_eight_l1613_161358


namespace NUMINAMATH_CALUDE_juice_cost_calculation_l1613_161366

/-- The cost of a sandwich in dollars -/
def sandwich_cost : ℚ := 0.30

/-- The total amount Lyle has in dollars -/
def total_amount : ℚ := 2.50

/-- The number of friends Lyle is buying for -/
def num_friends : ℕ := 4

/-- The cost of a pack of juice in dollars -/
def juice_cost : ℚ := 0.325

theorem juice_cost_calculation : 
  sandwich_cost * num_friends + juice_cost * num_friends = total_amount :=
by sorry

end NUMINAMATH_CALUDE_juice_cost_calculation_l1613_161366


namespace NUMINAMATH_CALUDE_right_triangles_shared_hypotenuse_l1613_161330

theorem right_triangles_shared_hypotenuse (a : ℝ) (h : a ≥ Real.sqrt 7) :
  let BC : ℝ := 3
  let AC : ℝ := a
  let AD : ℝ := 4
  let AB : ℝ := Real.sqrt (AC^2 + BC^2)
  let BD : ℝ := Real.sqrt (AB^2 - AD^2)
  BD = Real.sqrt (a^2 - 7) :=
by sorry

end NUMINAMATH_CALUDE_right_triangles_shared_hypotenuse_l1613_161330


namespace NUMINAMATH_CALUDE_current_speed_calculation_l1613_161303

theorem current_speed_calculation (boat_speed : ℝ) (upstream_time : ℝ) (downstream_time : ℝ)
  (h1 : boat_speed = 16)
  (h2 : upstream_time = 20 / 60)
  (h3 : downstream_time = 15 / 60) :
  ∃ (current_speed : ℝ),
    (boat_speed - current_speed) * upstream_time = (boat_speed + current_speed) * downstream_time ∧
    current_speed = 16 / 7 := by
  sorry

end NUMINAMATH_CALUDE_current_speed_calculation_l1613_161303


namespace NUMINAMATH_CALUDE_nancy_homework_l1613_161381

def homework_problem (math_problems : ℝ) (problems_per_hour : ℝ) (total_hours : ℝ) : Prop :=
  let total_problems := problems_per_hour * total_hours
  let spelling_problems := total_problems - math_problems
  spelling_problems = 15.0

theorem nancy_homework :
  homework_problem 17.0 8.0 4.0 := by
  sorry

end NUMINAMATH_CALUDE_nancy_homework_l1613_161381


namespace NUMINAMATH_CALUDE_sqrt_two_plus_sqrt_l1613_161390

theorem sqrt_two_plus_sqrt : ∃ y : ℝ, y = Real.sqrt (2 + y) → y = 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_two_plus_sqrt_l1613_161390


namespace NUMINAMATH_CALUDE_smartphone_loss_percentage_l1613_161376

/-- Calculates the percentage loss when selling an item -/
def percentageLoss (initialCost sellPrice : ℚ) : ℚ :=
  (initialCost - sellPrice) / initialCost * 100

/-- Proves that selling a $300 item for $255 results in a 15% loss -/
theorem smartphone_loss_percentage :
  percentageLoss 300 255 = 15 := by
  sorry

end NUMINAMATH_CALUDE_smartphone_loss_percentage_l1613_161376


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1613_161326

/-- A geometric sequence with positive first term and a specific condition on its terms -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  (∃ q : ℝ, ∀ n, a (n + 1) = a n * q) ∧ 
  a 1 > 0 ∧
  a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25

/-- The sum of the third and fifth terms of the geometric sequence is 5 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (h : geometric_sequence a) : 
  a 3 + a 5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1613_161326


namespace NUMINAMATH_CALUDE_jake_initial_cats_jake_initial_cats_is_one_l1613_161365

/-- Proves that Jake initially had 1 cat given the conditions of the problem -/
theorem jake_initial_cats : ℝ → Prop :=
  fun initial_cats =>
    let food_per_cat : ℝ := 0.5
    let total_food_after : ℝ := 0.9
    let extra_food : ℝ := 0.4
    (initial_cats * food_per_cat + food_per_cat = total_food_after) ∧
    (food_per_cat = extra_food) →
    initial_cats = 1

/-- The theorem is true -/
theorem jake_initial_cats_is_one : jake_initial_cats 1 := by
  sorry

end NUMINAMATH_CALUDE_jake_initial_cats_jake_initial_cats_is_one_l1613_161365


namespace NUMINAMATH_CALUDE_slower_walk_delay_l1613_161369

/-- Proves that walking at 4/5 of the usual speed results in a 6-minute delay -/
theorem slower_walk_delay (usual_time : ℝ) (h : usual_time = 24) : 
  let slower_time := usual_time / (4/5)
  slower_time - usual_time = 6 := by
  sorry

#check slower_walk_delay

end NUMINAMATH_CALUDE_slower_walk_delay_l1613_161369


namespace NUMINAMATH_CALUDE_kiwi_apple_equivalence_l1613_161377

/-- The value of kiwis in terms of apples -/
def kiwi_value (k : ℚ) : ℚ := k * 2

theorem kiwi_apple_equivalence :
  kiwi_value (1/4 * 20) = 10 →
  kiwi_value (3/4 * 12) = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_kiwi_apple_equivalence_l1613_161377


namespace NUMINAMATH_CALUDE_dartboard_section_angle_l1613_161347

theorem dartboard_section_angle (p : ℝ) (θ : ℝ) : 
  p = 1 / 4 →  -- probability of dart landing in a section
  p = θ / 360 →  -- probability equals ratio of central angle to full circle
  θ = 90 :=  -- central angle is 90 degrees
by sorry

end NUMINAMATH_CALUDE_dartboard_section_angle_l1613_161347


namespace NUMINAMATH_CALUDE_power_of_two_plus_one_l1613_161375

theorem power_of_two_plus_one (a : ℤ) (b : ℝ) (h : 2^a = b) : 2^(a+1) = 2*b := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_plus_one_l1613_161375


namespace NUMINAMATH_CALUDE_rectangular_field_area_l1613_161304

/-- Represents the ratio of the sides of a rectangular field -/
def side_ratio : ℚ := 3 / 4

/-- Cost of fencing per metre in paise -/
def fencing_cost_per_metre : ℚ := 25

/-- Total cost of fencing in rupees -/
def total_fencing_cost : ℚ := 101.5

/-- Conversion factor from rupees to paise -/
def rupees_to_paise : ℕ := 100

theorem rectangular_field_area (length width : ℚ) :
  length / width = side_ratio →
  2 * (length + width) * fencing_cost_per_metre = total_fencing_cost * rupees_to_paise →
  length * width = 10092 := by sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l1613_161304

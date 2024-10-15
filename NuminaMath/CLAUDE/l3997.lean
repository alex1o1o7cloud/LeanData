import Mathlib

namespace NUMINAMATH_CALUDE_fourth_column_is_quadratic_l3997_399751

/-- A quadruple of real numbers is quadratic if it satisfies the quadratic condition. -/
def is_quadratic (y : Fin 4 → ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ n : Fin 4, y n = a * (n.val + 1)^2 + b * (n.val + 1) + c

/-- A 4×4 grid of real numbers. -/
def Grid := Fin 4 → Fin 4 → ℝ

/-- All rows of the grid are quadratic. -/
def all_rows_quadratic (g : Grid) : Prop :=
  ∀ i : Fin 4, is_quadratic (λ j => g i j)

/-- The first three columns of the grid are quadratic. -/
def first_three_columns_quadratic (g : Grid) : Prop :=
  ∀ j : Fin 3, is_quadratic (λ i => g i j)

/-- The fourth column of the grid is quadratic. -/
def fourth_column_quadratic (g : Grid) : Prop :=
  is_quadratic (λ i => g i 3)

/-- 
If all rows and the first three columns of a 4×4 grid are quadratic,
then the fourth column is also quadratic.
-/
theorem fourth_column_is_quadratic (g : Grid)
  (h_rows : all_rows_quadratic g)
  (h_cols : first_three_columns_quadratic g) :
  fourth_column_quadratic g :=
sorry

end NUMINAMATH_CALUDE_fourth_column_is_quadratic_l3997_399751


namespace NUMINAMATH_CALUDE_sqrt_equation_sum_l3997_399719

theorem sqrt_equation_sum (n a t : ℝ) (hn : n ≥ 2) (ha : a > 0) (ht : t > 0) :
  Real.sqrt (n + a / t) = n * Real.sqrt (a / t) → a + t = n^2 + n - 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_sum_l3997_399719


namespace NUMINAMATH_CALUDE_min_bad_work_percentage_l3997_399720

/-- Represents the grading system for student work -/
inductive Grade
  | Accepted
  | NotAccepted

/-- Represents the true quality of student work -/
inductive Quality
  | Good
  | Bad

/-- Neural network classification result -/
def neuralNetworkClassify (work : Quality) : Grade :=
  sorry

/-- Expert classification result -/
def expertClassify (work : Quality) : Grade :=
  sorry

/-- Probability of neural network error -/
def neuralNetworkErrorRate : ℝ := 0.1

/-- Probability of work being bad -/
def badWorkProbability : ℝ := 0.2

/-- Probability of work being good -/
def goodWorkProbability : ℝ := 1 - badWorkProbability

/-- Percentage of work rechecked by experts -/
def recheckedPercentage : ℝ :=
  badWorkProbability * (1 - neuralNetworkErrorRate) + goodWorkProbability * neuralNetworkErrorRate

/-- Theorem: The minimum percentage of bad works among those rechecked by experts is 66% -/
theorem min_bad_work_percentage :
  (badWorkProbability * (1 - neuralNetworkErrorRate)) / recheckedPercentage ≥ 0.66 := by
  sorry

end NUMINAMATH_CALUDE_min_bad_work_percentage_l3997_399720


namespace NUMINAMATH_CALUDE_podcast5_duration_theorem_l3997_399799

def total_drive_time : ℕ := 6 * 60  -- in minutes
def podcast1_duration : ℕ := 45     -- in minutes
def podcast2_duration : ℕ := 2 * podcast1_duration
def podcast3_duration : ℕ := 105    -- 1 hour and 45 minutes in minutes
def podcast4_duration : ℕ := 60     -- 1 hour in minutes

def total_podcast_time : ℕ := podcast1_duration + podcast2_duration + podcast3_duration + podcast4_duration

theorem podcast5_duration_theorem :
  total_drive_time - total_podcast_time = 60 := by sorry

end NUMINAMATH_CALUDE_podcast5_duration_theorem_l3997_399799


namespace NUMINAMATH_CALUDE_sum_of_P_roots_l3997_399761

variable (a b c d : ℂ)

def P (X : ℂ) : ℂ := X^6 - X^5 - X^4 - X^3 - X

theorem sum_of_P_roots :
  (a^4 - a^3 - a^2 - 1 = 0) →
  (b^4 - b^3 - b^2 - 1 = 0) →
  (c^4 - c^3 - c^2 - 1 = 0) →
  (d^4 - d^3 - d^2 - 1 = 0) →
  P a + P b + P c + P d = -2 := by sorry

end NUMINAMATH_CALUDE_sum_of_P_roots_l3997_399761


namespace NUMINAMATH_CALUDE_num_acceptance_configs_prove_num_acceptance_configs_l3997_399790

/-- Represents the number of students -/
def num_students : ℕ := 4

/-- Represents the minimum number of companies -/
def min_companies : ℕ := 3

/-- Represents the acceptance configuration -/
structure AcceptanceConfig where
  student_acceptances : Fin num_students → ℕ
  company_acceptances : ℕ → Fin num_students → Bool
  each_student_diff : ∀ (i j : Fin num_students), i ≠ j → student_acceptances i ≠ student_acceptances j
  student_order : ∀ (i j : Fin num_students), i < j → student_acceptances i < student_acceptances j
  company_nonempty : ∀ (c : ℕ), c < min_companies → ∃ (s : Fin num_students), company_acceptances c s = true

/-- The main theorem stating the number of valid acceptance configurations -/
theorem num_acceptance_configs : (AcceptanceConfig → Prop) → ℕ := 60

/-- Proof of the theorem -/
theorem prove_num_acceptance_configs : num_acceptance_configs = 60 := by sorry

end NUMINAMATH_CALUDE_num_acceptance_configs_prove_num_acceptance_configs_l3997_399790


namespace NUMINAMATH_CALUDE_negative_two_and_negative_half_reciprocal_l3997_399701

/-- Two non-zero real numbers are reciprocal if their product is 1 -/
def IsReciprocal (a b : ℝ) : Prop := a ≠ 0 ∧ b ≠ 0 ∧ a * b = 1

/-- -2 and -1/2 are reciprocal -/
theorem negative_two_and_negative_half_reciprocal : IsReciprocal (-2) (-1/2) := by
  sorry

end NUMINAMATH_CALUDE_negative_two_and_negative_half_reciprocal_l3997_399701


namespace NUMINAMATH_CALUDE_ellipse_m_value_l3997_399723

/-- An ellipse with equation mx^2 + y^2 = 1, foci on the y-axis, and major axis length three times the minor axis length has m = 4/9 --/
theorem ellipse_m_value (m : ℝ) : 
  (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
    (∀ (x y : ℝ), m * x^2 + y^2 = 1 ↔ x^2 / b^2 + y^2 / a^2 = 1) ∧ 
    (∃ (c : ℝ), c > 0 ∧ a^2 = b^2 + c^2) ∧ 
    2 * a = 3 * (2 * b)) →
  m = 4/9 := by
sorry

end NUMINAMATH_CALUDE_ellipse_m_value_l3997_399723


namespace NUMINAMATH_CALUDE_student_A_pass_probability_l3997_399765

/-- Probability that student A passes the exam --/
def prob_pass (pA pB pDAC pDB : ℝ) : ℝ :=
  pA * pDAC + pB * pDB + (1 - pA - pB) * pDAC

theorem student_A_pass_probability :
  let pA := 0.3
  let pB := 0.3
  let pDAC := 0.8
  let pDB := 0.6
  prob_pass pA pB pDAC pDB = 0.74 := by
  sorry

#eval prob_pass 0.3 0.3 0.8 0.6

end NUMINAMATH_CALUDE_student_A_pass_probability_l3997_399765


namespace NUMINAMATH_CALUDE_intersecting_lines_theorem_l3997_399756

/-- Given two lines that intersect at (-7, 9), prove that the line passing through their coefficients as points has equation -7x + 9y = 1 -/
theorem intersecting_lines_theorem (A₁ B₁ A₂ B₂ : ℝ) : 
  (A₁ * (-7) + B₁ * 9 = 1) → 
  (A₂ * (-7) + B₂ * 9 = 1) → 
  ∃ (k : ℝ), k * (A₂ - A₁) = B₂ - B₁ ∧ 
             ∀ (x y : ℝ), y - B₁ = k * (x - A₁) → -7 * x + 9 * y = 1 :=
by sorry

end NUMINAMATH_CALUDE_intersecting_lines_theorem_l3997_399756


namespace NUMINAMATH_CALUDE_negation_of_universal_quantifier_negation_of_inequality_negation_of_proposition_l3997_399750

theorem negation_of_universal_quantifier (P : ℝ → Prop) :
  (¬ ∀ x ∈ Set.Ioo 0 1, P x) ↔ (∃ x ∈ Set.Ioo 0 1, ¬ P x) := by sorry

theorem negation_of_inequality (x : ℝ) : ¬(x^2 - x < 0) ↔ x^2 - x ≥ 0 := by sorry

theorem negation_of_proposition :
  (¬ ∀ x ∈ Set.Ioo 0 1, x^2 - x < 0) ↔ (∃ x ∈ Set.Ioo 0 1, x^2 - x ≥ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_quantifier_negation_of_inequality_negation_of_proposition_l3997_399750


namespace NUMINAMATH_CALUDE_cos_54_degrees_l3997_399721

theorem cos_54_degrees : Real.cos (54 * π / 180) = (3 - Real.sqrt 5) / 8 := by
  sorry

end NUMINAMATH_CALUDE_cos_54_degrees_l3997_399721


namespace NUMINAMATH_CALUDE_smallest_solution_congruence_l3997_399764

theorem smallest_solution_congruence :
  ∃ (x : ℕ), x > 0 ∧ (5 * x) % 33 = 14 % 33 ∧
  ∀ (y : ℕ), y > 0 → (5 * y) % 33 = 14 % 33 → x ≤ y :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_smallest_solution_congruence_l3997_399764


namespace NUMINAMATH_CALUDE_solution_set_part1_solution_set_part2_l3997_399796

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 2| + |x + a|

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f 1 x ≥ 4} = {x : ℝ | x ≤ -3/2 ∨ x ≥ 5/2} :=
sorry

-- Part 2
theorem solution_set_part2 (a : ℝ) :
  ({x : ℝ | f a x ≤ 2*x} = {x : ℝ | x ≥ 1}) → (a = 0 ∨ a = -2) :=
sorry

end NUMINAMATH_CALUDE_solution_set_part1_solution_set_part2_l3997_399796


namespace NUMINAMATH_CALUDE_triangle_area_l3997_399714

/-- The area of the right triangle formed by the x-axis, the line y = 2, and the line x = 1 + √3y --/
theorem triangle_area : ℝ := by
  -- Define the lines
  let x_axis : Set (ℝ × ℝ) := {p | p.2 = 0}
  let line_y2 : Set (ℝ × ℝ) := {p | p.2 = 2}
  let line_x1_sqrt3y : Set (ℝ × ℝ) := {p | p.1 = 1 + Real.sqrt 3 * p.2}

  -- Define the vertices of the triangle
  let origin : ℝ × ℝ := (0, 0)
  let vertex_on_x_axis : ℝ × ℝ := (1, 0)
  let vertex_on_y_axis : ℝ × ℝ := (0, 2)

  -- Calculate the area of the triangle
  let base : ℝ := vertex_on_x_axis.1 - origin.1
  let height : ℝ := vertex_on_y_axis.2 - origin.2
  let area : ℝ := (1 / 2) * base * height

  -- Prove that the area equals 1
  sorry

end NUMINAMATH_CALUDE_triangle_area_l3997_399714


namespace NUMINAMATH_CALUDE_similar_triangles_height_l3997_399798

theorem similar_triangles_height (h_small : ℝ) (area_ratio : ℝ) :
  h_small > 0 → area_ratio = 9 →
  ∃ h_large : ℝ, h_large = h_small * Real.sqrt area_ratio ∧ h_large = 15 :=
by sorry

end NUMINAMATH_CALUDE_similar_triangles_height_l3997_399798


namespace NUMINAMATH_CALUDE_max_expressions_greater_than_one_two_expressions_can_be_greater_than_one_max_expressions_greater_than_one_is_two_l3997_399727

theorem max_expressions_greater_than_one (a b c : ℝ) (h : a * b * c = 1) :
  (∃ (x y z : ℝ) (p : Fin 3 → ℝ), 
    (p 0 = 2*a - 1/b) ∧ (p 1 = 2*b - 1/c) ∧ (p 2 = 2*c - 1/a) ∧
    x ∈ Set.range p ∧ y ∈ Set.range p ∧ z ∈ Set.range p ∧
    x > 1 ∧ y > 1 ∧ z > 1) → False :=
by sorry

theorem two_expressions_can_be_greater_than_one (a b c : ℝ) (h : a * b * c = 1) :
  ∃ (x y : ℝ) (p : Fin 3 → ℝ), 
    (p 0 = 2*a - 1/b) ∧ (p 1 = 2*b - 1/c) ∧ (p 2 = 2*c - 1/a) ∧
    x ∈ Set.range p ∧ y ∈ Set.range p ∧
    x > 1 ∧ y > 1 :=
by sorry

theorem max_expressions_greater_than_one_is_two (a b c : ℝ) (h : a * b * c = 1) :
  (∃ (x y : ℝ) (p : Fin 3 → ℝ), 
    (p 0 = 2*a - 1/b) ∧ (p 1 = 2*b - 1/c) ∧ (p 2 = 2*c - 1/a) ∧
    x ∈ Set.range p ∧ y ∈ Set.range p ∧
    x > 1 ∧ y > 1) ∧
  (∃ (x y z : ℝ) (p : Fin 3 → ℝ), 
    (p 0 = 2*a - 1/b) ∧ (p 1 = 2*b - 1/c) ∧ (p 2 = 2*c - 1/a) ∧
    x ∈ Set.range p ∧ y ∈ Set.range p ∧ z ∈ Set.range p ∧
    x > 1 ∧ y > 1 ∧ z > 1) → False :=
by sorry

end NUMINAMATH_CALUDE_max_expressions_greater_than_one_two_expressions_can_be_greater_than_one_max_expressions_greater_than_one_is_two_l3997_399727


namespace NUMINAMATH_CALUDE_area_code_combinations_l3997_399779

/-- The number of digits in the area code -/
def n : ℕ := 4

/-- The set of digits used in the area code -/
def digits : Finset ℕ := {9, 8, 7, 6}

/-- The number of possible combinations for the area code -/
def num_combinations : ℕ := n.factorial

theorem area_code_combinations :
  Finset.card (Finset.powerset digits) = n ∧ num_combinations = 24 := by sorry

end NUMINAMATH_CALUDE_area_code_combinations_l3997_399779


namespace NUMINAMATH_CALUDE_investment_dividend_theorem_l3997_399793

/-- Calculates the dividend received from an investment in shares with premium and dividend rate --/
def calculate_dividend (investment : ℚ) (share_value : ℚ) (premium_rate : ℚ) (dividend_rate : ℚ) : ℚ :=
  let share_cost := share_value * (1 + premium_rate)
  let num_shares := investment / share_cost
  let dividend_per_share := share_value * dividend_rate
  num_shares * dividend_per_share

/-- Theorem: Given the specified investment conditions, the dividend received is 600 --/
theorem investment_dividend_theorem (investment : ℚ) (share_value : ℚ) (premium_rate : ℚ) (dividend_rate : ℚ)
  (h1 : investment = 14400)
  (h2 : share_value = 100)
  (h3 : premium_rate = 1/5)
  (h4 : dividend_rate = 1/20) :
  calculate_dividend investment share_value premium_rate dividend_rate = 600 := by
  sorry

#eval calculate_dividend 14400 100 (1/5) (1/20)

end NUMINAMATH_CALUDE_investment_dividend_theorem_l3997_399793


namespace NUMINAMATH_CALUDE_sum_of_consecutive_odds_l3997_399753

def is_valid_sum (n : ℕ) : Prop :=
  ∃ (k : ℤ), 
    (4 * k + 12 = n) ∧ 
    (k % 2 = 1) ∧ 
    ((4 * k + 4) % 10 = 0)

theorem sum_of_consecutive_odds : 
  is_valid_sum 28 ∧ 
  is_valid_sum 52 ∧ 
  is_valid_sum 84 ∧ 
  is_valid_sum 220 ∧ 
  ¬(is_valid_sum 112) :=
sorry

end NUMINAMATH_CALUDE_sum_of_consecutive_odds_l3997_399753


namespace NUMINAMATH_CALUDE_problems_solved_l3997_399725

theorem problems_solved (first last : ℕ) (h : first = 55) (h' : last = 150) :
  (Finset.range (last - first + 1)).card = 96 := by
  sorry

end NUMINAMATH_CALUDE_problems_solved_l3997_399725


namespace NUMINAMATH_CALUDE_alternating_series_ratio_l3997_399770

theorem alternating_series_ratio : 
  (1 - 2 + 4 - 8 + 16 - 32 + 64 - 128) / 
  (1^2 + 2^2 - 4^2 + 8^2 + 16^2 - 32^2 + 64^2 - 128^2) = 1 / 113 := by
  sorry

end NUMINAMATH_CALUDE_alternating_series_ratio_l3997_399770


namespace NUMINAMATH_CALUDE_shopkeeper_gain_l3997_399735

/-- Calculates the percentage gain for a shopkeeper given markup and discount percentages -/
theorem shopkeeper_gain (cost_price : ℝ) (markup_percent : ℝ) (discount_percent : ℝ) : 
  markup_percent = 35 →
  discount_percent = 20 →
  let marked_price := cost_price * (1 + markup_percent / 100)
  let selling_price := marked_price * (1 - discount_percent / 100)
  let gain := selling_price - cost_price
  gain / cost_price * 100 = 8 := by
  sorry

end NUMINAMATH_CALUDE_shopkeeper_gain_l3997_399735


namespace NUMINAMATH_CALUDE_kth_roots_sum_power_real_l3997_399733

theorem kth_roots_sum_power_real (k : ℕ) (x y : ℂ) 
  (hx : x^k = 1) (hy : y^k = 1) : 
  ∃ (r : ℝ), (x + y)^k = r := by sorry

end NUMINAMATH_CALUDE_kth_roots_sum_power_real_l3997_399733


namespace NUMINAMATH_CALUDE_largest_cube_in_sphere_l3997_399712

theorem largest_cube_in_sphere (a b c : ℝ) (ha : a = 22) (hb : b = 2) (hc : c = 10) :
  let cuboid_diagonal := Real.sqrt (a^2 + b^2 + c^2)
  let cube_side := Real.sqrt ((a^2 + b^2 + c^2) / 3)
  cube_side = 14 :=
sorry

end NUMINAMATH_CALUDE_largest_cube_in_sphere_l3997_399712


namespace NUMINAMATH_CALUDE_cookie_cost_claire_cookie_cost_l3997_399777

/-- The cost of a cookie given Claire's spending habits and gift card balance --/
theorem cookie_cost (gift_card : ℝ) (latte_cost : ℝ) (croissant_cost : ℝ) 
  (days : ℕ) (num_cookies : ℕ) (remaining_balance : ℝ) : ℝ :=
  let daily_treat_cost := latte_cost + croissant_cost
  let weekly_treat_cost := daily_treat_cost * days
  let total_spent := gift_card - remaining_balance
  let cookie_total_cost := total_spent - weekly_treat_cost
  cookie_total_cost / num_cookies

/-- Proof that each cookie costs $1.25 given Claire's spending habits --/
theorem claire_cookie_cost : 
  cookie_cost 100 3.75 3.50 7 5 43 = 1.25 := by
  sorry

end NUMINAMATH_CALUDE_cookie_cost_claire_cookie_cost_l3997_399777


namespace NUMINAMATH_CALUDE_ginkgo_field_length_l3997_399707

/-- The length of a field with evenly spaced trees -/
def field_length (num_trees : ℕ) (interval : ℕ) : ℕ :=
  (num_trees - 1) * interval

/-- Theorem: The length of a field with 10 ginkgo trees planted at 10-meter intervals, 
    including trees at both ends, is 90 meters. -/
theorem ginkgo_field_length : field_length 10 10 = 90 := by
  sorry

end NUMINAMATH_CALUDE_ginkgo_field_length_l3997_399707


namespace NUMINAMATH_CALUDE_x_squared_plus_reciprocal_l3997_399776

theorem x_squared_plus_reciprocal (x : ℝ) (h : 15 = x^4 + 1/x^4) : x^2 + 1/x^2 = Real.sqrt 17 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_plus_reciprocal_l3997_399776


namespace NUMINAMATH_CALUDE_percentage_difference_l3997_399754

theorem percentage_difference (x y : ℝ) (h : y = x * (1 + 1/3)) :
  x = y * (1 - 1/4) :=
sorry

end NUMINAMATH_CALUDE_percentage_difference_l3997_399754


namespace NUMINAMATH_CALUDE_exists_43_move_strategy_l3997_399728

/-- The number of boxes and chosen numbers -/
def n : ℕ := 2017

/-- A strategy for distributing stones -/
structure Strategy where
  numbers : Fin n → ℕ

/-- The state of the game after some moves -/
def GameState := Fin n → ℕ

/-- Apply a strategy for one move -/
def applyStrategy (s : Strategy) (state : GameState) : GameState :=
  fun i => state i + s.numbers i

/-- Apply a strategy for k moves -/
def applyStrategyKTimes (s : Strategy) (k : ℕ) : GameState :=
  fun i => k * (s.numbers i)

/-- Check if all boxes have the same number of stones -/
def allEqual (state : GameState) : Prop :=
  ∀ i j, state i = state j

/-- The main theorem -/
theorem exists_43_move_strategy :
  ∃ (s : Strategy),
    (allEqual (applyStrategyKTimes s 43)) ∧
    (∀ k, 0 < k → k < 43 → ¬(allEqual (applyStrategyKTimes s k))) := by
  sorry

end NUMINAMATH_CALUDE_exists_43_move_strategy_l3997_399728


namespace NUMINAMATH_CALUDE_equal_area_rectangles_width_l3997_399792

/-- Given two rectangles of equal area, where one rectangle measures 12 inches by 15 inches
    and the other has a length of 9 inches, prove that the width of the second rectangle is 20 inches. -/
theorem equal_area_rectangles_width (area carol_length carol_width jordan_length jordan_width : ℝ) :
  area = carol_length * carol_width →
  area = jordan_length * jordan_width →
  carol_length = 12 →
  carol_width = 15 →
  jordan_length = 9 →
  jordan_width = 20 := by
sorry

end NUMINAMATH_CALUDE_equal_area_rectangles_width_l3997_399792


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3997_399731

/-- Represents a hyperbola in the xy-plane -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  equation : (x y : ℝ) → Prop := fun x y => (x^2 / a^2) - (y^2 / b^2) = 1

/-- The focus of a hyperbola -/
def Focus := ℝ × ℝ

/-- Theorem stating the equation of a specific hyperbola -/
theorem hyperbola_equation (H : Hyperbola) (F : Focus) :
  H.equation = fun x y => (y^2 / 12) - (x^2 / 24) = 1 ↔
    F = (0, 6) ∧
    ∃ (K : Hyperbola), K.a^2 = 2 ∧ K.b^2 = 1 ∧
      (∀ x y, H.equation x y ↔ K.equation x y) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3997_399731


namespace NUMINAMATH_CALUDE_exists_k_for_special_sequence_l3997_399743

/-- A sequence of non-negative integers satisfying certain conditions -/
def SpecialSequence (c : Fin 1997 → ℕ) : Prop :=
  (c 1 ≥ 0) ∧
  (∀ m n : Fin 1997, m > 0 → n > 0 → m + n < 1998 →
    c m + c n ≤ c (m + n) ∧ c (m + n) ≤ c m + c n + 1)

/-- Theorem stating the existence of k for the special sequence -/
theorem exists_k_for_special_sequence (c : Fin 1997 → ℕ) (h : SpecialSequence c) :
  ∃ k : ℝ, ∀ n : Fin 1997, c n = ⌊n * k⌋ :=
sorry

end NUMINAMATH_CALUDE_exists_k_for_special_sequence_l3997_399743


namespace NUMINAMATH_CALUDE_largest_band_size_l3997_399773

/-- Represents a rectangular band formation --/
structure BandFormation where
  rows : ℕ
  membersPerRow : ℕ

/-- Checks if a band formation is valid according to the problem conditions --/
def isValidFormation (original : BandFormation) (total : ℕ) : Prop :=
  total < 100 ∧
  total = original.rows * original.membersPerRow + 3 ∧
  total = (original.rows - 3) * (original.membersPerRow + 1)

/-- Finds the largest valid band formation --/
def largestValidFormation : Option (BandFormation × ℕ) :=
  sorry

theorem largest_band_size :
  ∀ bf : BandFormation,
  ∀ m : ℕ,
  isValidFormation bf m →
  m ≤ 75 :=
sorry

end NUMINAMATH_CALUDE_largest_band_size_l3997_399773


namespace NUMINAMATH_CALUDE_chloe_points_per_treasure_l3997_399732

/-- The number of treasures Chloe found on the first level -/
def treasures_level1 : ℕ := 6

/-- The number of treasures Chloe found on the second level -/
def treasures_level2 : ℕ := 3

/-- Chloe's total score -/
def total_score : ℕ := 81

/-- The number of points Chloe scores for each treasure -/
def points_per_treasure : ℕ := total_score / (treasures_level1 + treasures_level2)

theorem chloe_points_per_treasure :
  points_per_treasure = 9 := by
  sorry

end NUMINAMATH_CALUDE_chloe_points_per_treasure_l3997_399732


namespace NUMINAMATH_CALUDE_fraction_multiplication_equals_decimal_l3997_399758

theorem fraction_multiplication_equals_decimal : 
  (1 : ℚ) / 3 * (3 : ℚ) / 7 * (7 : ℚ) / 8 = 0.12499999999999997 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_equals_decimal_l3997_399758


namespace NUMINAMATH_CALUDE_vacation_pictures_l3997_399747

def remaining_pictures (zoo_pics : ℕ) (museum_pics : ℕ) (deleted_pics : ℕ) : ℕ :=
  zoo_pics + museum_pics - deleted_pics

theorem vacation_pictures (zoo_pics : ℕ) (museum_pics : ℕ) (deleted_pics : ℕ) 
  (h1 : zoo_pics = 50)
  (h2 : museum_pics = 8)
  (h3 : deleted_pics = 38) :
  remaining_pictures zoo_pics museum_pics deleted_pics = 20 := by
  sorry

end NUMINAMATH_CALUDE_vacation_pictures_l3997_399747


namespace NUMINAMATH_CALUDE_triangle_max_perimeter_l3997_399700

theorem triangle_max_perimeter (a b c : ℝ) : 
  1 ≤ a ∧ a ≤ 3 ∧ 3 ≤ b ∧ b ≤ 5 ∧ 5 ≤ c ∧ c ≤ 7 →
  ∃ (p : ℝ), p = 8 + Real.sqrt 34 ∧ 
  ∀ (a' b' c' : ℝ), 1 ≤ a' ∧ a' ≤ 3 ∧ 3 ≤ b' ∧ b' ≤ 5 ∧ 5 ≤ c' ∧ c' ≤ 7 →
  (a' + b' + c' ≤ p ∧ 
   ∃ (s : ℝ), s = (a' + b' + c') / 2 ∧ 
   a' * b' * c' / (4 * s * (s - a') * (s - b') * (s - c')).sqrt ≤ 
   3 * 5 / 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_perimeter_l3997_399700


namespace NUMINAMATH_CALUDE_tabitha_honey_days_l3997_399786

/-- Represents the number of days Tabitha can enjoy honey in her tea --/
def honey_days (servings_per_cup : ℕ) (evening_cups : ℕ) (morning_cups : ℕ) 
               (container_ounces : ℕ) (servings_per_ounce : ℕ) : ℕ :=
  (container_ounces * servings_per_ounce) / (servings_per_cup * (evening_cups + morning_cups))

/-- Theorem stating that Tabitha can enjoy honey in her tea for 32 days --/
theorem tabitha_honey_days : 
  honey_days 1 2 1 16 6 = 32 := by
  sorry

end NUMINAMATH_CALUDE_tabitha_honey_days_l3997_399786


namespace NUMINAMATH_CALUDE_parallelogram_area_specific_vectors_l3997_399771

/-- The area of a parallelogram formed by two 2D vectors -/
def parallelogramArea (v w : ℝ × ℝ) : ℝ :=
  |v.1 * w.2 - v.2 * w.1|

theorem parallelogram_area_specific_vectors :
  let v : ℝ × ℝ := (8, -5)
  let w : ℝ × ℝ := (14, -3)
  parallelogramArea v w = 46 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_area_specific_vectors_l3997_399771


namespace NUMINAMATH_CALUDE_total_shoes_count_l3997_399711

theorem total_shoes_count (bonny becky bobby cherry diane : ℕ) : 
  bonny = 13 ∧
  bonny = 2 * becky - 5 ∧
  bobby = 3 * becky ∧
  cherry = bonny + becky + 4 ∧
  diane = 2 * cherry - 2 →
  bonny + becky + bobby + cherry + diane = 125 := by
  sorry

end NUMINAMATH_CALUDE_total_shoes_count_l3997_399711


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3997_399705

theorem polynomial_factorization (a b c : ℝ) :
  a * (b - c)^3 - 2 * b * (c - a)^3 + 3 * c * (a - b)^3 =
  (a - b) * (b - c) * (c - a) * (5 * a - 4 * b - 3 * c) := by
sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3997_399705


namespace NUMINAMATH_CALUDE_binary_1101_equals_13_l3997_399729

/-- Converts a list of binary digits to its decimal representation -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 1101₂ -/
def binary_1101 : List Bool := [true, false, true, true]

theorem binary_1101_equals_13 : binary_to_decimal binary_1101 = 13 := by
  sorry

end NUMINAMATH_CALUDE_binary_1101_equals_13_l3997_399729


namespace NUMINAMATH_CALUDE_function_composition_l3997_399746

theorem function_composition (f : ℝ → ℝ) (x : ℝ) :
  (∀ y, f y = y^2 + 2*y) →
  f (2*x + 1) = 4*x^2 + 8*x + 3 := by
sorry

end NUMINAMATH_CALUDE_function_composition_l3997_399746


namespace NUMINAMATH_CALUDE_salad_dressing_calories_l3997_399737

/-- Calculates the calories in the salad dressing given the total calories consumed and the calories from other ingredients. -/
theorem salad_dressing_calories :
  let lettuce_calories : ℝ := 50
  let carrot_calories : ℝ := 2 * lettuce_calories
  let pizza_crust_calories : ℝ := 600
  let pepperoni_calories : ℝ := (1 / 3) * pizza_crust_calories
  let cheese_calories : ℝ := 400
  let salad_portion : ℝ := 1 / 4
  let pizza_portion : ℝ := 1 / 5
  let total_calories_consumed : ℝ := 330

  let salad_calories_without_dressing : ℝ := (lettuce_calories + carrot_calories) * salad_portion
  let pizza_calories : ℝ := (pizza_crust_calories + pepperoni_calories + cheese_calories) * pizza_portion
  let calories_without_dressing : ℝ := salad_calories_without_dressing + pizza_calories
  let dressing_calories : ℝ := total_calories_consumed - calories_without_dressing

  dressing_calories = 52.5 := by sorry

end NUMINAMATH_CALUDE_salad_dressing_calories_l3997_399737


namespace NUMINAMATH_CALUDE_hotel_rooms_available_l3997_399785

theorem hotel_rooms_available (total_floors : ℕ) (rooms_per_floor : ℕ) (unavailable_floors : ℕ) :
  total_floors = 10 →
  rooms_per_floor = 10 →
  unavailable_floors = 1 →
  (total_floors - unavailable_floors) * rooms_per_floor = 90 := by
  sorry

end NUMINAMATH_CALUDE_hotel_rooms_available_l3997_399785


namespace NUMINAMATH_CALUDE_quadratic_minimum_l3997_399782

theorem quadratic_minimum (p q : ℝ) : 
  (∀ x, x^2 + p*x + q ≥ 1) → q = 1 + p^2/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l3997_399782


namespace NUMINAMATH_CALUDE_triangle_inequality_l3997_399702

theorem triangle_inequality (α β γ a b c : ℝ) 
  (h_positive : α > 0 ∧ β > 0 ∧ γ > 0 ∧ a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : α + β + γ = π) : 
  a * (1/β + 1/γ) + b * (1/γ + 1/α) + c * (1/α + 1/β) ≥ 2 * (a/α + b/β + c/γ) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3997_399702


namespace NUMINAMATH_CALUDE_complex_subtraction_multiplication_l3997_399742

theorem complex_subtraction_multiplication (i : ℂ) : 
  (7 - 3 * i) - 3 * (2 - 5 * i) = 1 + 12 * i := by sorry

end NUMINAMATH_CALUDE_complex_subtraction_multiplication_l3997_399742


namespace NUMINAMATH_CALUDE_tangent_line_equations_l3997_399706

/-- The function for which we're finding the tangent line -/
def f (x : ℝ) : ℝ := x^3 - 2*x

/-- The derivative of f -/
def f' (x : ℝ) : ℝ := 3*x^2 - 2

/-- The point through which the tangent line passes -/
def P : ℝ × ℝ := (2, 4)

/-- Theorem: The equations of the tangent lines to y = x³ - 2x passing through (2,4) -/
theorem tangent_line_equations :
  ∃ (m : ℝ), (f m = m^3 - 2*m) ∧
             ((4 - f m = (f' m) * (2 - m)) ∧
              (∀ x, f' m * (x - m) + f m = 10*x - 16 ∨
                    f' m * (x - m) + f m = x + 2)) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equations_l3997_399706


namespace NUMINAMATH_CALUDE_dog_bone_collection_l3997_399744

/-- Calculates the final number of bones in a dog's collection after finding and giving away some bones. -/
def final_bone_count (initial_bones : ℕ) (found_multiplier : ℕ) (bones_given_away : ℕ) : ℕ :=
  initial_bones + (initial_bones * found_multiplier) - bones_given_away

/-- Theorem stating that given the specific conditions, the dog ends up with 3380 bones. -/
theorem dog_bone_collection : final_bone_count 350 9 120 = 3380 := by
  sorry

end NUMINAMATH_CALUDE_dog_bone_collection_l3997_399744


namespace NUMINAMATH_CALUDE_rectangle_area_constant_l3997_399775

theorem rectangle_area_constant (d : ℝ) (h : d > 0) : 
  ∃ (w l : ℝ), w > 0 ∧ l > 0 ∧ w / l = 3 / 5 ∧ w ^ 2 + l ^ 2 = (10 * d) ^ 2 ∧ w * l = (750 / 17) * d ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_constant_l3997_399775


namespace NUMINAMATH_CALUDE_simplify_absolute_value_l3997_399730

theorem simplify_absolute_value : |(-4)^2 - 3^2 + 2| = 9 := by
  sorry

end NUMINAMATH_CALUDE_simplify_absolute_value_l3997_399730


namespace NUMINAMATH_CALUDE_arithmetic_sequence_difference_l3997_399766

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- The sum of specific terms in the sequence equals 80 -/
def SumCondition (a : ℕ → ℝ) : Prop :=
  a 2 + a 4 + a 6 + a 8 + a 10 = 80

theorem arithmetic_sequence_difference
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_sum : SumCondition a) :
  ∃ d : ℝ, a 7 - a 8 = -d ∧ ArithmeticSequence a ∧ ∀ n : ℕ, a (n + 1) - a n = d :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_difference_l3997_399766


namespace NUMINAMATH_CALUDE_increasing_sine_function_bound_l3997_399784

open Real

/-- A function f is increasing on ℝ if for all x, y ∈ ℝ, x < y implies f(x) < f(y) -/
def IncreasingOn (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

/-- The main theorem: If f(x) = x + a*sin(x) is increasing on ℝ, then -1 ≤ a ≤ 1 -/
theorem increasing_sine_function_bound (a : ℝ) :
  IncreasingOn (fun x => x + a * sin x) → -1 ≤ a ∧ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_increasing_sine_function_bound_l3997_399784


namespace NUMINAMATH_CALUDE_ratio_is_two_to_one_l3997_399734

/-- An isosceles right-angled triangle with an inscribed square -/
structure IsoscelesRightTriangleWithSquare where
  /-- The side length of the isosceles right triangle -/
  x : ℝ
  /-- The distance from O to P on OB -/
  a : ℝ
  /-- The distance from O to Q on OA -/
  b : ℝ
  /-- The side length of the inscribed square PQRS -/
  s : ℝ
  /-- The side length of the triangle is positive -/
  x_pos : 0 < x
  /-- a and b are positive and their sum equals x -/
  ab_sum : 0 < a ∧ 0 < b ∧ a + b = x
  /-- The side length of the square is the sum of a and b -/
  square_side : s = a + b
  /-- The area of the square is 2/5 of the area of the triangle -/
  area_ratio : s^2 = (2/5) * (x^2/2)

/-- 
If an isosceles right-angled triangle AOB has a square PQRS inscribed as described, 
and the area of PQRS is 2/5 of the area of AOB, then the ratio of OP to OQ is 2:1.
-/
theorem ratio_is_two_to_one (t : IsoscelesRightTriangleWithSquare) : 
  t.a / t.b = 2 := by sorry

end NUMINAMATH_CALUDE_ratio_is_two_to_one_l3997_399734


namespace NUMINAMATH_CALUDE_functional_equation_implies_linear_l3997_399755

/-- A function satisfying the given functional equation -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^2 - y^2) = x * f x - y * f y

/-- The main theorem stating that any function satisfying the functional equation is linear -/
theorem functional_equation_implies_linear (f : ℝ → ℝ) 
  (h : SatisfiesFunctionalEquation f) : 
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x :=
sorry

end NUMINAMATH_CALUDE_functional_equation_implies_linear_l3997_399755


namespace NUMINAMATH_CALUDE_book_reading_time_l3997_399718

/-- Given a book with 500 pages, prove that reading the first half at 10 pages per day
    and the second half at 5 pages per day results in a total of 75 days spent reading. -/
theorem book_reading_time (total_pages : ℕ) (first_half_speed second_half_speed : ℕ) :
  total_pages = 500 →
  first_half_speed = 10 →
  second_half_speed = 5 →
  (total_pages / 2 / first_half_speed) + (total_pages / 2 / second_half_speed) = 75 :=
by sorry


end NUMINAMATH_CALUDE_book_reading_time_l3997_399718


namespace NUMINAMATH_CALUDE_box_dimensions_l3997_399710

theorem box_dimensions (a h : ℝ) (b : ℝ) : 
  h = a / 2 →
  6 * a + b = 156 →
  7 * a + b = 178 →
  a = 22 ∧ h = 11 :=
by sorry

end NUMINAMATH_CALUDE_box_dimensions_l3997_399710


namespace NUMINAMATH_CALUDE_locus_of_perpendicular_foot_l3997_399736

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- Represents a rectangular hyperbola -/
structure RectangularHyperbola where
  k : ℝ

/-- Represents the locus of points -/
def locus (h : RectangularHyperbola) : Set PolarPoint :=
  {p : PolarPoint | p.r^2 = 2 * h.k^2 * Real.sin (2 * p.θ)}

/-- The main theorem stating that the locus of the foot of the perpendicular
    from the center of a rectangular hyperbola to a tangent is given by
    the polar equation r^2 = 2k^2 sin 2θ -/
theorem locus_of_perpendicular_foot (h : RectangularHyperbola) :
  ∀ p : PolarPoint, p ∈ locus h ↔
    ∃ (t : ℝ), -- t represents the parameter of a point on the hyperbola
      let tangent_point := (t, h.k^2 / t)
      let tangent_slope := -h.k^2 / t^2
      let perpendicular_slope := -1 / tangent_slope
      p.r * (Real.cos p.θ) = 2 * h.k^4 / (t * (t^4 + h.k^4)) ∧
      p.r * (Real.sin p.θ) = 2 * t * h.k^2 / (t^4 + h.k^4) :=
by
  sorry

end NUMINAMATH_CALUDE_locus_of_perpendicular_foot_l3997_399736


namespace NUMINAMATH_CALUDE_johns_total_cost_l3997_399780

/-- Calculates the total cost of a cell phone plan --/
def calculate_total_cost (base_cost : ℝ) (text_cost : ℝ) (extra_minute_cost : ℝ) 
  (texts_sent : ℕ) (hours_talked : ℕ) : ℝ :=
  let text_charge := text_cost * texts_sent
  let extra_hours := max (hours_talked - 50) 0
  let extra_minutes := extra_hours * 60
  let extra_minute_charge := extra_minute_cost * extra_minutes
  base_cost + text_charge + extra_minute_charge

/-- Theorem stating that John's total cost is $69.00 --/
theorem johns_total_cost : 
  calculate_total_cost 30 0.10 0.20 150 52 = 69 := by
  sorry

end NUMINAMATH_CALUDE_johns_total_cost_l3997_399780


namespace NUMINAMATH_CALUDE_octagon_area_division_l3997_399740

theorem octagon_area_division (CO OM MP PU UT TE : ℝ) (D : ℝ) :
  CO = 1 ∧ OM = 1 ∧ MP = 1 ∧ PU = 1 ∧ UT = 1 ∧ TE = 1 →
  (∃ (COMPUTER_area COMPUTED_area CDR_area : ℝ),
    COMPUTER_area = 6 ∧
    COMPUTED_area = 3 ∧
    CDR_area = 3 ∧
    COMPUTED_area = CDR_area) →
  (∃ (CD DR : ℝ),
    CD = 3 ∧
    CDR_area = 1/2 * CD * DR) →
  DR = 2 :=
sorry

end NUMINAMATH_CALUDE_octagon_area_division_l3997_399740


namespace NUMINAMATH_CALUDE_line_equation_proof_l3997_399741

/-- A line parameterized by t ∈ ℝ -/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- The specific line given in the problem -/
def given_line : ParametricLine where
  x := λ t => 3 * t + 6
  y := λ t => 5 * t - 7

/-- The equation of a line in slope-intercept form -/
structure LineEquation where
  slope : ℝ
  intercept : ℝ

theorem line_equation_proof (L : ParametricLine) 
    (h : L = given_line) : 
    ∃ (eq : LineEquation), 
      eq.slope = 5/3 ∧ 
      eq.intercept = -17 ∧
      ∀ t, L.y t = eq.slope * (L.x t) + eq.intercept :=
  sorry

end NUMINAMATH_CALUDE_line_equation_proof_l3997_399741


namespace NUMINAMATH_CALUDE_number_puzzle_l3997_399739

theorem number_puzzle : ∃ x : ℝ, (x / 7 - x / 11 = 100) ∧ (x = 1925) := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l3997_399739


namespace NUMINAMATH_CALUDE_company_plants_1500_trees_l3997_399783

/-- Represents the number of trees chopped down in the first half of the year -/
def trees_chopped_first_half : ℕ := 200

/-- Represents the number of trees chopped down in the second half of the year -/
def trees_chopped_second_half : ℕ := 300

/-- Represents the number of trees to be planted for each tree chopped down -/
def trees_planted_per_chopped : ℕ := 3

/-- Calculates the total number of trees that need to be planted -/
def trees_to_plant : ℕ := (trees_chopped_first_half + trees_chopped_second_half) * trees_planted_per_chopped

/-- Theorem stating that the company needs to plant 1500 trees -/
theorem company_plants_1500_trees : trees_to_plant = 1500 := by
  sorry

end NUMINAMATH_CALUDE_company_plants_1500_trees_l3997_399783


namespace NUMINAMATH_CALUDE_highlighters_count_l3997_399797

/-- The number of pink highlighters in the desk -/
def pink_highlighters : ℕ := 7

/-- The number of yellow highlighters in the desk -/
def yellow_highlighters : ℕ := 4

/-- The number of blue highlighters in the desk -/
def blue_highlighters : ℕ := 5

/-- The number of green highlighters in the desk -/
def green_highlighters : ℕ := 3

/-- The number of orange highlighters in the desk -/
def orange_highlighters : ℕ := 6

/-- The total number of highlighters in the desk -/
def total_highlighters : ℕ := pink_highlighters + yellow_highlighters + blue_highlighters + green_highlighters + orange_highlighters

theorem highlighters_count : total_highlighters = 25 := by
  sorry

end NUMINAMATH_CALUDE_highlighters_count_l3997_399797


namespace NUMINAMATH_CALUDE_min_selling_price_A_is_190_l3997_399788

/-- Represents the number of units and prices of water purifiers --/
structure WaterPurifiers where
  units_A : ℕ
  units_B : ℕ
  cost_A : ℕ
  cost_B : ℕ
  total_cost : ℕ

/-- Calculates the minimum selling price of model A --/
def min_selling_price_A (w : WaterPurifiers) : ℕ :=
  w.cost_A + (w.total_cost - w.units_A * w.cost_A - w.units_B * w.cost_B) / w.units_A

/-- Theorem stating the minimum selling price of model A --/
theorem min_selling_price_A_is_190 (w : WaterPurifiers) 
  (h1 : w.units_A + w.units_B = 100)
  (h2 : w.units_A * w.cost_A + w.units_B * w.cost_B = w.total_cost)
  (h3 : w.cost_A = 150)
  (h4 : w.cost_B = 250)
  (h5 : w.total_cost = 19000)
  (h6 : ∀ (sell_A : ℕ), 
    (sell_A - w.cost_A) * w.units_A + 2 * (sell_A - w.cost_A) * w.units_B ≥ 5600 → 
    min_selling_price_A w ≤ sell_A) :
  min_selling_price_A w = 190 := by
  sorry

#eval min_selling_price_A ⟨60, 40, 150, 250, 19000⟩

end NUMINAMATH_CALUDE_min_selling_price_A_is_190_l3997_399788


namespace NUMINAMATH_CALUDE_min_coach_handshakes_correct_l3997_399722

/-- The minimum number of handshakes by coaches in a basketball tournament --/
def min_coach_handshakes : ℕ := 60

/-- Total number of handshakes in the tournament --/
def total_handshakes : ℕ := 495

/-- Number of teams in the tournament --/
def num_teams : ℕ := 2

/-- Function to calculate the number of player-to-player handshakes --/
def player_handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem stating the minimum number of handshakes by coaches --/
theorem min_coach_handshakes_correct :
  ∃ (n : ℕ), n % num_teams = 0 ∧
  player_handshakes n + (n / num_teams) * num_teams = total_handshakes ∧
  (n / num_teams) * num_teams = min_coach_handshakes :=
sorry

end NUMINAMATH_CALUDE_min_coach_handshakes_correct_l3997_399722


namespace NUMINAMATH_CALUDE_vector_magnitude_problem_l3997_399752

/-- Given two vectors in 2D space satisfying certain conditions, prove that the magnitude of one vector is 3. -/
theorem vector_magnitude_problem (a b : ℝ × ℝ) : 
  (∃ θ : ℝ, θ = Real.pi / 3 ∧ a.fst * b.fst + a.snd * b.snd = Real.cos θ * ‖a‖ * ‖b‖) →  -- angle between a and b is 60°
  ‖a‖ = 1 →  -- |a| = 1
  ‖2 • a - b‖ = Real.sqrt 7 →  -- |2a - b| = √7
  ‖b‖ = 3 := by
sorry


end NUMINAMATH_CALUDE_vector_magnitude_problem_l3997_399752


namespace NUMINAMATH_CALUDE_team_a_win_probability_l3997_399794

def number_of_matches : ℕ := 7
def wins_required : ℕ := 4
def win_probability : ℚ := 1/2

theorem team_a_win_probability :
  (number_of_matches.choose wins_required) * win_probability ^ wins_required * (1 - win_probability) ^ (number_of_matches - wins_required) = 35/128 := by
  sorry

end NUMINAMATH_CALUDE_team_a_win_probability_l3997_399794


namespace NUMINAMATH_CALUDE_circle_equation_proof_l3997_399717

/-- Prove that the equation (x-1)^2 + (y-1)^2 = 4 represents the circle that passes through
    points A(1,-1) and B(-1,1), and has its center on the line x+y-2=0. -/
theorem circle_equation_proof (x y : ℝ) : 
  (∀ (cx cy : ℝ), cx + cy = 2 →
    (cx - 1)^2 + (cy - 1)^2 = 4 ∧
    (1 - cx)^2 + (-1 - cy)^2 = 4 ∧
    (-1 - cx)^2 + (1 - cy)^2 = 4) ↔
  (x - 1)^2 + (y - 1)^2 = 4 := by
sorry

end NUMINAMATH_CALUDE_circle_equation_proof_l3997_399717


namespace NUMINAMATH_CALUDE_inequality_proof_l3997_399789

theorem inequality_proof (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  ((x + y + z) / 3) ^ (x + y + z) ≤ x^x * y^y * z^z ∧
  x^x * y^y * z^z ≤ ((x^2 + y^2 + z^2) / (x + y + z)) ^ (x + y + z) := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3997_399789


namespace NUMINAMATH_CALUDE_divisor_problem_l3997_399724

theorem divisor_problem (dividend quotient remainder divisor : ℕ) : 
  dividend = 158 →
  quotient = 9 →
  remainder = 5 →
  dividend = divisor * quotient + remainder →
  divisor = 17 := by
sorry

end NUMINAMATH_CALUDE_divisor_problem_l3997_399724


namespace NUMINAMATH_CALUDE_sqrt_simplification_l3997_399787

theorem sqrt_simplification : (5 - 3 * Real.sqrt 2) ^ 2 = 45 - 28 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_simplification_l3997_399787


namespace NUMINAMATH_CALUDE_difference_of_squares_special_case_l3997_399774

theorem difference_of_squares_special_case : (503 : ℤ) * 503 - 502 * 504 = 1 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_special_case_l3997_399774


namespace NUMINAMATH_CALUDE_data_set_properties_l3997_399778

def data_set : List ℕ := [3, 5, 4, 5, 6, 7]

def mode (list : List ℕ) : ℕ := sorry

def median (list : List ℕ) : ℚ := sorry

def mean (list : List ℕ) : ℚ := sorry

theorem data_set_properties :
  mode data_set = 5 ∧ 
  median data_set = 5 ∧ 
  mean data_set = 5 := by sorry

end NUMINAMATH_CALUDE_data_set_properties_l3997_399778


namespace NUMINAMATH_CALUDE_zeros_difference_quadratic_l3997_399716

theorem zeros_difference_quadratic (m : ℝ) : 
  (∃ α β : ℝ, 2 * α^2 - m * α - 8 = 0 ∧ 
              2 * β^2 - m * β - 8 = 0 ∧ 
              α - β = m - 1) ↔ 
  (m = 6 ∨ m = -10/3) := by
sorry

end NUMINAMATH_CALUDE_zeros_difference_quadratic_l3997_399716


namespace NUMINAMATH_CALUDE_visible_cubes_count_l3997_399709

/-- Represents a cube with gaps -/
structure CubeWithGaps where
  size : ℕ
  unit_cubes : ℕ
  has_gaps : Bool

/-- Calculates the number of visible or partially visible unit cubes from a corner -/
def visible_cubes (c : CubeWithGaps) : ℕ :=
  sorry

/-- The specific cube in the problem -/
def problem_cube : CubeWithGaps :=
  { size := 12
  , unit_cubes := 12^3
  , has_gaps := true }

theorem visible_cubes_count :
  visible_cubes problem_cube = 412 :=
sorry

end NUMINAMATH_CALUDE_visible_cubes_count_l3997_399709


namespace NUMINAMATH_CALUDE_larger_number_problem_l3997_399749

theorem larger_number_problem (L S : ℕ) (h1 : L > S) (h2 : L - S = 1345) (h3 : L = 6 * S + 15) : L = 1611 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l3997_399749


namespace NUMINAMATH_CALUDE_geometric_sequence_value_l3997_399703

theorem geometric_sequence_value (x : ℝ) : 
  (∃ r : ℝ, x / 12 = r ∧ 3 / x = r) → x = 6 ∨ x = -6 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_value_l3997_399703


namespace NUMINAMATH_CALUDE_max_peak_consumption_l3997_399757

/-- Proves that the maximum average monthly electricity consumption during peak hours is 118 kw•h -/
theorem max_peak_consumption (original_price peak_price off_peak_price total_consumption : ℝ)
  (h1 : original_price = 0.52)
  (h2 : peak_price = 0.55)
  (h3 : off_peak_price = 0.35)
  (h4 : total_consumption = 200)
  (h5 : ∀ x : ℝ, 0 ≤ x ∧ x ≤ total_consumption →
    (original_price - peak_price) * x + (original_price - off_peak_price) * (total_consumption - x) ≥ 
    total_consumption * original_price * 0.1) :
  ∃ max_peak : ℝ, max_peak = 118 ∧ 
    ∀ x : ℝ, 0 ≤ x ∧ x ≤ total_consumption →
      (original_price - peak_price) * x + (original_price - off_peak_price) * (total_consumption - x) ≥ 
      total_consumption * original_price * 0.1 → 
      x ≤ max_peak :=
sorry

end NUMINAMATH_CALUDE_max_peak_consumption_l3997_399757


namespace NUMINAMATH_CALUDE_jack_hand_in_amount_l3997_399759

def calculate_amount_to_hand_in (hundred_bills two_hundred_bills fifty_bills twenty_bills ten_bills five_bills one_bills quarters dimes nickels pennies : ℕ) (amount_to_leave : ℝ) : ℝ :=
  let total_notes := 100 * hundred_bills + 50 * fifty_bills + 20 * twenty_bills + 10 * ten_bills + 5 * five_bills + one_bills
  let amount_to_hand_in := total_notes - amount_to_leave
  amount_to_hand_in

theorem jack_hand_in_amount :
  calculate_amount_to_hand_in 2 1 5 3 7 27 42 19 36 47 300 = 142 := by
  sorry

end NUMINAMATH_CALUDE_jack_hand_in_amount_l3997_399759


namespace NUMINAMATH_CALUDE_max_tokens_on_chessboard_max_tokens_on_chessboard_with_diagonals_l3997_399772

/-- Represents a chessboard configuration -/
def Chessboard := Fin 8 → Fin 8 → Bool

/-- Checks if a chessboard configuration is valid according to the given constraints -/
def is_valid_configuration (board : Chessboard) : Prop :=
  -- At most 4 tokens per row
  (∀ row, (Finset.filter (λ col => board row col) Finset.univ).card ≤ 4) ∧
  -- At most 4 tokens per column
  (∀ col, (Finset.filter (λ row => board row col) Finset.univ).card ≤ 4)

/-- Checks if a chessboard configuration is valid including diagonal constraints -/
def is_valid_configuration_with_diagonals (board : Chessboard) : Prop :=
  is_valid_configuration board ∧
  -- At most 4 tokens on main diagonal
  (Finset.filter (λ i => board i i) Finset.univ).card ≤ 4 ∧
  -- At most 4 tokens on anti-diagonal
  (Finset.filter (λ i => board i (7 - i)) Finset.univ).card ≤ 4

/-- The total number of tokens on the board -/
def token_count (board : Chessboard) : Nat :=
  (Finset.filter (λ p => board p.1 p.2) (Finset.univ.product Finset.univ)).card

theorem max_tokens_on_chessboard :
  (∃ board : Chessboard, is_valid_configuration board ∧ token_count board = 32) ∧
  (∀ board : Chessboard, is_valid_configuration board → token_count board ≤ 32) :=
sorry

theorem max_tokens_on_chessboard_with_diagonals :
  (∃ board : Chessboard, is_valid_configuration_with_diagonals board ∧ token_count board = 32) ∧
  (∀ board : Chessboard, is_valid_configuration_with_diagonals board → token_count board ≤ 32) :=
sorry

end NUMINAMATH_CALUDE_max_tokens_on_chessboard_max_tokens_on_chessboard_with_diagonals_l3997_399772


namespace NUMINAMATH_CALUDE_dinner_arrangement_count_l3997_399760

def number_of_friends : ℕ := 5
def number_of_cooks : ℕ := 2

theorem dinner_arrangement_count :
  Nat.choose number_of_friends number_of_cooks = 10 := by
  sorry

end NUMINAMATH_CALUDE_dinner_arrangement_count_l3997_399760


namespace NUMINAMATH_CALUDE_square_sum_from_difference_and_product_l3997_399704

theorem square_sum_from_difference_and_product (a b : ℝ) 
  (h1 : a - b = 8) 
  (h2 : a * b = 50) : 
  a^2 + b^2 = 164 := by
sorry

end NUMINAMATH_CALUDE_square_sum_from_difference_and_product_l3997_399704


namespace NUMINAMATH_CALUDE_triangle_area_in_square_grid_l3997_399763

theorem triangle_area_in_square_grid :
  let square_side : ℝ := 4
  let square_area : ℝ := square_side ^ 2
  let triangle1_area : ℝ := 4
  let triangle2_area : ℝ := 2
  let triangle3_area : ℝ := 3
  let total_triangles_area : ℝ := triangle1_area + triangle2_area + triangle3_area
  let triangle_abc_area : ℝ := square_area - total_triangles_area
  triangle_abc_area = 7 := by sorry

end NUMINAMATH_CALUDE_triangle_area_in_square_grid_l3997_399763


namespace NUMINAMATH_CALUDE_min_value_and_y_l3997_399781

theorem min_value_and_y (x y z : ℝ) (h : 2*x - 3*y + z = 3) :
  ∃ (min_val : ℝ), 
    (∀ x' y' z' : ℝ, 2*x' - 3*y' + z' = 3 → x'^2 + (y'-1)^2 + z'^2 ≥ min_val) ∧
    (x^2 + (y-1)^2 + z^2 = min_val) ∧
    (min_val = 18/7) ∧
    (y = -2/7) := by
  sorry

end NUMINAMATH_CALUDE_min_value_and_y_l3997_399781


namespace NUMINAMATH_CALUDE_quadratic_roots_ratio_l3997_399726

theorem quadratic_roots_ratio (k : ℝ) : 
  (∃ r : ℝ, r ≠ 0 ∧ 
    (∀ x : ℝ, x^2 + 6*x + k = 0 ↔ (x = 2*r ∨ x = r)) ∧
    (2*r : ℝ) / r = 2) → 
  k = 8 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_ratio_l3997_399726


namespace NUMINAMATH_CALUDE_square_root_sum_equals_four_root_six_l3997_399795

theorem square_root_sum_equals_four_root_six :
  Real.sqrt (16 - 8 * Real.sqrt 6) + Real.sqrt (16 + 8 * Real.sqrt 6) = 4 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_square_root_sum_equals_four_root_six_l3997_399795


namespace NUMINAMATH_CALUDE_expression_simplification_l3997_399769

theorem expression_simplification (x : ℝ) (h : x = 2 * Real.sin (60 * π / 180) - Real.tan (45 * π / 180)) :
  (x / (x - 1) - 1 / (x^2 - x)) / ((x + 1)^2 / x) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3997_399769


namespace NUMINAMATH_CALUDE_sum_reciprocals_of_roots_l3997_399762

theorem sum_reciprocals_of_roots (x : ℝ) : 
  x^2 - 7*x + 2 = 0 → 
  ∃ a b : ℝ, (x = a ∨ x = b) ∧ (1/a + 1/b = 7/2) :=
by sorry

end NUMINAMATH_CALUDE_sum_reciprocals_of_roots_l3997_399762


namespace NUMINAMATH_CALUDE_car_speed_l3997_399791

/-- Given a car that travels 375 km in 3 hours, its speed is 125 km/h -/
theorem car_speed (distance : ℝ) (time : ℝ) (speed : ℝ) : 
  distance = 375 ∧ time = 3 → speed = distance / time → speed = 125 := by
sorry

end NUMINAMATH_CALUDE_car_speed_l3997_399791


namespace NUMINAMATH_CALUDE_cone_apex_angle_l3997_399748

theorem cone_apex_angle (R : ℝ) (h : R > 0) :
  let lateral_surface := π * R^2 / 2
  let base_circumference := π * R
  lateral_surface = base_circumference * R / 2 →
  let base_diameter := R
  let apex_angle := 2 * Real.arcsin (base_diameter / (2 * R))
  apex_angle = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_cone_apex_angle_l3997_399748


namespace NUMINAMATH_CALUDE_max_triangle_area_l3997_399738

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < b ∧ b < a

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle formed by three points -/
structure Triangle where
  p1 : Point
  p2 : Point
  p3 : Point

def Ellipse.equation (e : Ellipse) (p : Point) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

def isosceles_trapezoid (m n f2 f1 : Point) : Prop :=
  ∃ (height area : ℝ), height = Real.sqrt 3 ∧ area = 3 * Real.sqrt 3

def line_through_point (p : Point) : Set Point :=
  {q : Point | ∃ (k : ℝ), q.x = k * q.y + p.x}

def intersect_ellipse_line (e : Ellipse) (l : Set Point) : Set Point :=
  {p : Point | p ∈ l ∧ e.equation p}

def triangle_area (t : Triangle) : ℝ :=
  sorry

theorem max_triangle_area (e : Ellipse) (m n f2 f1 : Point) :
  isosceles_trapezoid m n f2 f1 →
  m = Point.mk (-e.a) e.b →
  n = Point.mk e.a e.b →
  (∀ (l : Set Point), f1 ∈ l →
    let intersection := intersect_ellipse_line e l
    ∀ (a b : Point), a ∈ intersection → b ∈ intersection →
      triangle_area (Triangle.mk f2 a b) ≤ 3) ∧
  (∃ (l : Set Point), f1 ∈ l →
    let intersection := intersect_ellipse_line e l
    ∃ (a b : Point), a ∈ intersection ∧ b ∈ intersection ∧
      triangle_area (Triangle.mk f2 a b) = 3) :=
sorry

end NUMINAMATH_CALUDE_max_triangle_area_l3997_399738


namespace NUMINAMATH_CALUDE_lcm_of_ratio_and_hcf_l3997_399768

theorem lcm_of_ratio_and_hcf (a b : ℕ+) : 
  (a : ℚ) / b = 3 / 4 → 
  Nat.gcd a b = 3 → 
  Nat.lcm a b = 36 := by
sorry

end NUMINAMATH_CALUDE_lcm_of_ratio_and_hcf_l3997_399768


namespace NUMINAMATH_CALUDE_sequence_inequality_l3997_399767

theorem sequence_inequality (n : ℕ) (a : ℕ → ℝ) 
  (h1 : a 0 = 0 ∧ a (n + 1) = 0)
  (h2 : ∀ k : ℕ, k ≥ 1 ∧ k ≤ n → |a (k - 1) - 2 * a k + a (k + 1)| ≤ 1) :
  ∀ k : ℕ, k ≤ n + 1 → |a k| ≤ k * (n + 1 - k) / 2 :=
by sorry

end NUMINAMATH_CALUDE_sequence_inequality_l3997_399767


namespace NUMINAMATH_CALUDE_chess_tournament_games_l3997_399745

theorem chess_tournament_games (n : ℕ) (total_games : ℕ) :
  n = 12 →
  total_games = 132 →
  ∃ (games_per_pair : ℕ),
    total_games = games_per_pair * (n * (n - 1) / 2) ∧
    games_per_pair = 2 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l3997_399745


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3997_399715

def A : Set ℝ := {0, 1, 2}
def B : Set ℝ := {x | x^2 - x ≤ 0}

theorem intersection_of_A_and_B : A ∩ B = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3997_399715


namespace NUMINAMATH_CALUDE_trigonometric_sum_l3997_399713

theorem trigonometric_sum (a b : ℝ) (θ : ℝ) (h : 0 < a) (k : 0 < b) :
  (Real.sin θ ^ 6 / a + Real.cos θ ^ 6 / b = 1 / (a + b)) →
  (Real.sin θ ^ 12 / a ^ 5 + Real.cos θ ^ 12 / b ^ 5 = 1 / (a + b) ^ 5) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_sum_l3997_399713


namespace NUMINAMATH_CALUDE_parabola_directrix_quadratic_roots_as_eccentricities_l3997_399708

-- Define the parabola
def parabola (x y : ℝ) : Prop := x = 2 * y^2

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop := 2 * x^2 - 5 * x + 2 = 0

-- Theorem for the parabola directrix
theorem parabola_directrix : ∃ (p : ℝ), ∀ (x y : ℝ), parabola x y → (x = -1/8 ↔ x = -p) := by sorry

-- Theorem for the quadratic equation roots as eccentricities
theorem quadratic_roots_as_eccentricities :
  ∃ (e₁ e₂ : ℝ), quadratic_equation e₁ ∧ quadratic_equation e₂ ∧
  (0 < e₁ ∧ e₁ < 1) ∧ (e₂ > 1) := by sorry

end NUMINAMATH_CALUDE_parabola_directrix_quadratic_roots_as_eccentricities_l3997_399708

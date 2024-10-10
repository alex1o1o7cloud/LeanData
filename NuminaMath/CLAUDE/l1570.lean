import Mathlib

namespace standard_form_of_given_equation_l1570_157070

/-- Standard form of a quadratic equation -/
def standard_form (a b c : ℝ) : ℝ → Prop :=
  fun x ↦ a * x^2 + b * x + c = 0

/-- The given quadratic equation -/
def given_equation (x : ℝ) : Prop :=
  3 * x^2 + 1 = 7 * x

/-- Theorem stating that the standard form of 3x^2 + 1 = 7x is 3x^2 - 7x + 1 = 0 -/
theorem standard_form_of_given_equation :
  ∃ a b c : ℝ, a ≠ 0 ∧ (∀ x, given_equation x ↔ standard_form a b c x) ∧ 
  a = 3 ∧ b = -7 ∧ c = 1 :=
sorry

end standard_form_of_given_equation_l1570_157070


namespace christmas_discount_problem_l1570_157015

/-- Represents the Christmas discount problem for an air-conditioning unit. -/
theorem christmas_discount_problem (original_price : ℝ) (price_increase : ℝ) (final_price : ℝ) 
  (h1 : original_price = 470)
  (h2 : price_increase = 0.12)
  (h3 : final_price = 442.18) :
  ∃ (x : ℝ), 
    x ≥ 0 ∧ 
    x ≤ 100 ∧ 
    abs (x - 1.11) < 0.01 ∧
    original_price * (1 - x / 100) * (1 + price_increase) = final_price :=
sorry

end christmas_discount_problem_l1570_157015


namespace transistors_in_2000_l1570_157053

/-- Moore's law states that the number of transistors doubles every two years -/
def moores_law (t : ℕ) : ℕ := 2^(t/2)

/-- The number of transistors in a typical CPU in 1990 -/
def transistors_1990 : ℕ := 1000000

/-- The year we're calculating for -/
def target_year : ℕ := 2000

/-- The starting year -/
def start_year : ℕ := 1990

theorem transistors_in_2000 : 
  transistors_1990 * moores_law (target_year - start_year) = 32000000 := by
  sorry

end transistors_in_2000_l1570_157053


namespace paper_towel_savings_l1570_157017

theorem paper_towel_savings (package_price : ℚ) (individual_price : ℚ) (rolls : ℕ) : 
  package_price = 9 → individual_price = 1 → rolls = 12 →
  (1 - package_price / (individual_price * rolls)) * 100 = 25 := by
sorry

end paper_towel_savings_l1570_157017


namespace curve_self_intersects_l1570_157014

/-- The x-coordinate of a point on the curve given a parameter t -/
def x (t : ℝ) : ℝ := t^2 - 4

/-- The y-coordinate of a point on the curve given a parameter t -/
def y (t : ℝ) : ℝ := t^3 - 6*t + 7

/-- The curve intersects itself if there exist two distinct real numbers that yield the same point -/
def self_intersects : Prop :=
  ∃ a b : ℝ, a ≠ b ∧ x a = x b ∧ y a = y b

/-- The point of self-intersection -/
def intersection_point : ℝ × ℝ := (2, 7)

/-- Theorem stating that the curve intersects itself at (2, 7) -/
theorem curve_self_intersects :
  self_intersects ∧ ∃ a b : ℝ, a ≠ b ∧ x a = (intersection_point.1) ∧ y a = (intersection_point.2) :=
sorry

end curve_self_intersects_l1570_157014


namespace winning_strategy_l1570_157076

/-- Represents the colors used in the chocolate table -/
inductive Color
  | Red
  | Green
  | Blue

/-- Represents a cell in the chocolate table -/
structure Cell :=
  (row : Nat)
  (col : Nat)
  (color : Color)

/-- Represents the chocolate table -/
def ChocolateTable (n : Nat) := Array (Array Cell)

/-- Creates a colored n × n chocolate table -/
def createTable (n : Nat) : ChocolateTable n := sorry

/-- Removes a cell from the chocolate table -/
def removeCell (table : ChocolateTable n) (cell : Cell) : ChocolateTable n := sorry

/-- Checks if a 3 × 1 or 1 × 3 rectangle contains one of each color -/
def validRectangle (rect : Array Cell) : Bool := sorry

/-- Checks if the table can be partitioned into valid rectangles -/
def canPartition (table : ChocolateTable n) : Bool := sorry

theorem winning_strategy 
  (n : Nat) 
  (h1 : n > 3) 
  (h2 : ¬(3 ∣ n)) : 
  ∃ (cell : Cell), cell.color ≠ Color.Red ∧ 
    ¬(canPartition (removeCell (createTable n) cell)) := by sorry

end winning_strategy_l1570_157076


namespace prob_two_good_in_four_draws_l1570_157060

/-- Represents the number of light bulbs in the box -/
def total_bulbs : ℕ := 10

/-- Represents the number of good quality bulbs -/
def good_bulbs : ℕ := 8

/-- Represents the number of defective bulbs -/
def defective_bulbs : ℕ := 2

/-- Represents the number of draws -/
def num_draws : ℕ := 4

/-- Represents the number of good quality bulbs to be drawn -/
def target_good_bulbs : ℕ := 2

/-- Calculates the probability of drawing exactly 2 good quality bulbs in 4 draws -/
theorem prob_two_good_in_four_draws :
  (defective_bulbs * (defective_bulbs - 1) * good_bulbs * (good_bulbs - 1)) / 
  (total_bulbs * (total_bulbs - 1) * (total_bulbs - 2) * (total_bulbs - 3)) = 1 / 15 := by
  sorry

end prob_two_good_in_four_draws_l1570_157060


namespace parking_lot_valid_tickets_percentage_l1570_157049

theorem parking_lot_valid_tickets_percentage 
  (total_cars : ℕ) 
  (unpaid_cars : ℕ) 
  (valid_ticket_percentage : ℝ) :
  total_cars = 300 →
  unpaid_cars = 30 →
  (valid_ticket_percentage / 5 + valid_ticket_percentage) * total_cars / 100 = total_cars - unpaid_cars →
  valid_ticket_percentage = 75 := by
sorry

end parking_lot_valid_tickets_percentage_l1570_157049


namespace sum_of_coefficients_l1570_157092

def expansion (x : ℝ) (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) : Prop :=
  (1 - 2*x)^5 = a₀ + 2*a₁*x + 4*a₂*x^2 + 8*a₃*x^3 + 16*a₄*x^4 + 32*a₅*x^5

theorem sum_of_coefficients 
  (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) 
  (h : ∀ x, expansion x a₀ a₁ a₂ a₃ a₄ a₅) : 
  a₁ + a₂ + a₃ + a₄ + a₅ = -1 := by
  sorry

end sum_of_coefficients_l1570_157092


namespace f_is_quadratic_l1570_157054

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation x² - 5 = 0 -/
def f (x : ℝ) : ℝ := x^2 - 5

/-- Theorem: f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry

end f_is_quadratic_l1570_157054


namespace arithmetic_geometric_mean_inequality_l1570_157098

theorem arithmetic_geometric_mean_inequality {a b : ℝ} (ha : 0 < a) (hb : 0 < b) :
  (a + b) / 2 ≥ Real.sqrt (a * b) ∧
  ((a + b) / 2 = Real.sqrt (a * b) ↔ a = b) :=
sorry

end arithmetic_geometric_mean_inequality_l1570_157098


namespace fraction_sum_ratio_l1570_157090

theorem fraction_sum_ratio : (1 / 3 + 1 / 4) / (1 / 2) = 7 / 6 := by
  sorry

end fraction_sum_ratio_l1570_157090


namespace solution_set_of_inequality_l1570_157072

open Set

theorem solution_set_of_inequality (f : ℝ → ℝ) (f' : ℝ → ℝ) :
  (∀ x, f (-x) = -f x) →  -- f is odd
  (∀ x, HasDerivAt f (f' x) x) →  -- f' is the derivative of f
  (∀ x > 0, x * f' x + 3 * f x > 0) →  -- given condition
  {x : ℝ | x^3 * f x + (2*x - 1)^3 * f (1 - 2*x) < 0} = Iic (1/3) ∪ Ioi 1 :=
sorry

end solution_set_of_inequality_l1570_157072


namespace eunice_pots_l1570_157005

/-- Given a total number of seeds and a number of seeds per pot (except for the last pot),
    calculate the number of pots needed. -/
def calculate_pots (total_seeds : ℕ) (seeds_per_pot : ℕ) : ℕ :=
  (total_seeds - 1) / seeds_per_pot + 1

/-- Theorem stating that with 10 seeds and 3 seeds per pot (except for the last pot),
    the number of pots needed is 4. -/
theorem eunice_pots : calculate_pots 10 3 = 4 := by
  sorry

end eunice_pots_l1570_157005


namespace sliding_window_is_only_translation_l1570_157043

/-- Represents a type of movement --/
inductive Movement
  | PingPongBall
  | SlidingWindow
  | Kite
  | Basketball

/-- Predicate to check if a movement is a translation --/
def isTranslation (m : Movement) : Prop :=
  match m with
  | Movement.SlidingWindow => True
  | _ => False

/-- Theorem stating that only the sliding window movement is a translation --/
theorem sliding_window_is_only_translation :
  ∀ m : Movement, isTranslation m ↔ m = Movement.SlidingWindow :=
sorry

#check sliding_window_is_only_translation

end sliding_window_is_only_translation_l1570_157043


namespace max_value_problem_l1570_157073

theorem max_value_problem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_sum : 2 * a + b = 1) :
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ 2 * x + y = 1 ∧
    2 * Real.sqrt (a * b) - 4 * a^2 - b^2 ≤ 2 * Real.sqrt (x * y) - 4 * x^2 - y^2) ∧
  2 * Real.sqrt (a * b) - 4 * a^2 - b^2 ≤ (Real.sqrt 2 - 1) / 2 :=
sorry

end max_value_problem_l1570_157073


namespace square_sum_difference_l1570_157027

theorem square_sum_difference (n : ℕ) : 
  (2*n + 1)^2 - (2*n - 1)^2 + (2*n - 1)^2 - (2*n - 3)^2 + (2*n - 3)^2 - (2*n - 5)^2 + 
  (2*n - 5)^2 - (2*n - 7)^2 + (2*n - 7)^2 - (2*n - 9)^2 + (2*n - 9)^2 - (2*n - 11)^2 = 288 :=
by sorry

end square_sum_difference_l1570_157027


namespace max_S_at_7_or_8_l1570_157095

/-- Represents the sum of the first n terms of the arithmetic sequence -/
def S (n : ℕ) : ℚ :=
  5 * n - (5 / 14) * n * (n - 1)

/-- The maximum value of S occurs when n is 7 or 8 -/
theorem max_S_at_7_or_8 :
  ∀ k : ℕ, (S k ≤ S 7 ∧ S k ≤ S 8) ∧
  (S 7 = S 8 ∨ (∀ m : ℕ, m ≠ 7 → m ≠ 8 → S m < max (S 7) (S 8))) := by
  sorry

end max_S_at_7_or_8_l1570_157095


namespace intersection_equals_interval_l1570_157061

-- Define the sets M and N
def M : Set ℝ := {x | Real.log x > 0}
def N : Set ℝ := {x | x^2 ≤ 4}

-- Define the interval (1, 2]
def interval : Set ℝ := {x | 1 < x ∧ x ≤ 2}

-- Statement to prove
theorem intersection_equals_interval : M ∩ N = interval := by sorry

end intersection_equals_interval_l1570_157061


namespace fence_pole_count_l1570_157016

/-- Calculates the number of fence poles needed for a path with a bridge --/
def fence_poles (total_length : ℕ) (bridge_length : ℕ) (pole_spacing : ℕ) : ℕ :=
  2 * ((total_length - bridge_length) / pole_spacing)

/-- Theorem statement for the fence pole problem --/
theorem fence_pole_count : 
  fence_poles 900 42 6 = 286 := by
  sorry

end fence_pole_count_l1570_157016


namespace new_profit_percentage_l1570_157019

/-- Calculate the new profit percentage given the original selling price, profit percentage, and additional profit --/
theorem new_profit_percentage
  (original_selling_price : ℝ)
  (original_profit_percentage : ℝ)
  (additional_profit : ℝ)
  (h1 : original_selling_price = 550)
  (h2 : original_profit_percentage = 0.1)
  (h3 : additional_profit = 35) :
  let original_cost_price := original_selling_price / (1 + original_profit_percentage)
  let new_cost_price := original_cost_price * 0.9
  let new_selling_price := original_selling_price + additional_profit
  let new_profit := new_selling_price - new_cost_price
  let new_profit_percentage := new_profit / new_cost_price
  new_profit_percentage = 0.3 := by
sorry


end new_profit_percentage_l1570_157019


namespace tv_price_before_tax_l1570_157075

theorem tv_price_before_tax (P : ℝ) : P + 0.15 * P = 1955 → P = 1700 := by
  sorry

end tv_price_before_tax_l1570_157075


namespace polynomial_remainder_l1570_157036

theorem polynomial_remainder (x : ℝ) : (x^11 + 2) % (x - 1) = 3 := by
  sorry

end polynomial_remainder_l1570_157036


namespace almond_butter_servings_l1570_157089

/-- Represents the number of tablespoons in the container -/
def container_amount : ℚ := 35 + 2/3

/-- Represents the number of tablespoons in one serving -/
def serving_size : ℚ := 2 + 1/2

/-- Represents the number of servings in the container -/
def number_of_servings : ℚ := container_amount / serving_size

theorem almond_butter_servings : 
  ∃ (n : ℕ) (r : ℚ), 0 ≤ r ∧ r < 1 ∧ number_of_servings = n + r ∧ n = 14 ∧ r = 4/15 :=
sorry

end almond_butter_servings_l1570_157089


namespace real_part_of_z_squared_neg_four_l1570_157069

theorem real_part_of_z_squared_neg_four (z : ℂ) : z^2 = -4 → Complex.re z = 0 := by
  sorry

end real_part_of_z_squared_neg_four_l1570_157069


namespace max_xy_value_l1570_157065

theorem max_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_sum : x + 4*y = 12) :
  xy ≤ 9 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 4*y₀ = 12 ∧ x₀*y₀ = 9 :=
by sorry

end max_xy_value_l1570_157065


namespace line_segment_lengths_l1570_157063

/-- Given points A, B, and C on a line, prove that if AB = 5 and AC = BC + 1, then AC = 3 and BC = 2 -/
theorem line_segment_lengths (A B C : ℝ) (h1 : |B - A| = 5) (h2 : |C - A| = |C - B| + 1) :
  |C - A| = 3 ∧ |C - B| = 2 := by
  sorry

end line_segment_lengths_l1570_157063


namespace student_incorrect_answer_l1570_157033

theorem student_incorrect_answer 
  (D : ℕ) -- Dividend
  (h1 : D / 36 = 42) -- Correct division
  (h2 : 63 ≠ 36) -- Student used wrong divisor
  : D / 63 = 24 := by
  sorry

end student_incorrect_answer_l1570_157033


namespace equal_derivatives_of_quadratic_functions_l1570_157093

theorem equal_derivatives_of_quadratic_functions :
  let f (x : ℝ) := 1 - 2 * x^2
  let g (x : ℝ) := -2 * x^2 + 3
  ∀ x, deriv f x = deriv g x := by
sorry

end equal_derivatives_of_quadratic_functions_l1570_157093


namespace complex_radical_expression_simplification_l1570_157000

theorem complex_radical_expression_simplification :
  3 * Real.sqrt (1/3) + Real.sqrt 2 * (Real.sqrt 3 - Real.sqrt 6) - Real.sqrt 12 / Real.sqrt 2 = - Real.sqrt 3 := by
  sorry

end complex_radical_expression_simplification_l1570_157000


namespace article_profit_l1570_157041

/-- If selling an article at 2/3 of its original price results in a 20% loss,
    then selling it at the original price results in a 20% profit. -/
theorem article_profit (original_price : ℝ) (cost_price : ℝ) 
    (h1 : original_price > 0) 
    (h2 : cost_price > 0)
    (h3 : (2/3) * original_price = 0.8 * cost_price) : 
  (original_price - cost_price) / cost_price = 0.2 := by
  sorry

#check article_profit

end article_profit_l1570_157041


namespace paths_through_B_and_C_l1570_157047

/-- Represents a point on the square grid -/
structure GridPoint where
  x : ℕ
  y : ℕ

/-- Calculates the number of paths between two points on a square grid -/
def num_paths (start finish : GridPoint) : ℕ :=
  Nat.choose (finish.x - start.x + finish.y - start.y) (finish.x - start.x)

/-- The points on the grid -/
def A : GridPoint := ⟨0, 0⟩
def B : GridPoint := ⟨2, 3⟩
def C : GridPoint := ⟨6, 4⟩
def D : GridPoint := ⟨9, 6⟩

/-- The theorem to be proved -/
theorem paths_through_B_and_C : 
  num_paths A B * num_paths B C * num_paths C D = 500 := by
  sorry

end paths_through_B_and_C_l1570_157047


namespace average_marks_math_chem_l1570_157050

theorem average_marks_math_chem (M P C : ℕ) : 
  M + P = 20 →
  C = P + 20 →
  (M + C) / 2 = 20 :=
by sorry

end average_marks_math_chem_l1570_157050


namespace no_primes_in_range_l1570_157035

theorem no_primes_in_range (n : ℕ) (h : n > 2) :
  ∀ p, Prime p → ¬(n.factorial + 2 < p ∧ p < n.factorial + n + 1) := by
  sorry

end no_primes_in_range_l1570_157035


namespace kelly_games_l1570_157002

theorem kelly_games (initial_games given_away left : ℕ) : 
  given_away = 99 → left = 22 → initial_games = given_away + left :=
by sorry

end kelly_games_l1570_157002


namespace solution_set_is_positive_reals_l1570_157064

open Set
open Function
open Real

noncomputable section

variables {f : ℝ → ℝ} (hf : Differentiable ℝ f)

def condition_1 (f : ℝ → ℝ) : Prop :=
  ∀ x, f x + deriv f x > 1

def condition_2 (f : ℝ → ℝ) : Prop :=
  f 0 = 2018

def solution_set (f : ℝ → ℝ) : Set ℝ :=
  {x | Real.exp x * f x - Real.exp x > 2017}

theorem solution_set_is_positive_reals
  (h1 : condition_1 f) (h2 : condition_2 f) :
  solution_set f = Ioi 0 :=
sorry

end

end solution_set_is_positive_reals_l1570_157064


namespace carol_peanuts_count_l1570_157020

def initial_peanuts : ℕ := 2
def additional_peanuts : ℕ := 5

theorem carol_peanuts_count : initial_peanuts + additional_peanuts = 7 := by
  sorry

end carol_peanuts_count_l1570_157020


namespace rhombus_prism_lateral_area_l1570_157001

/-- Given a rectangular quadrilateral prism with a rhombus base, this theorem calculates its lateral surface area. -/
theorem rhombus_prism_lateral_area (side_length : ℝ) (diagonal_length : ℝ) (h1 : side_length = 2) (h2 : diagonal_length = 2 * Real.sqrt 3) :
  let lateral_edge := Real.sqrt (diagonal_length^2 - side_length^2)
  let perimeter := 4 * side_length
  let lateral_area := perimeter * lateral_edge
  lateral_area = 16 * Real.sqrt 2 :=
by sorry

end rhombus_prism_lateral_area_l1570_157001


namespace direct_proportion_iff_m_eq_neg_one_l1570_157045

/-- A function f(x) is a direct proportion function if there exists a non-zero constant k such that f(x) = kx for all x. -/
def is_direct_proportion (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x, f x = k * x

/-- The function y = (m-1)x + m^2 - 1 -/
def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x + m^2 - 1

theorem direct_proportion_iff_m_eq_neg_one (m : ℝ) :
  is_direct_proportion (f m) ↔ m = -1 := by
  sorry

end direct_proportion_iff_m_eq_neg_one_l1570_157045


namespace shortest_perpendicular_best_measurement_l1570_157077

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a long jump measurement method -/
inductive LongJumpMeasurement
  | Vertical
  | ShortestLineSegment
  | TwoPointLine
  | ShortestPerpendicular

/-- Defines the accuracy of a measurement method -/
def isAccurate (method : LongJumpMeasurement) : Prop :=
  match method with
  | LongJumpMeasurement.ShortestPerpendicular => true
  | _ => false

/-- Defines the consistency of a measurement method -/
def isConsistent (method : LongJumpMeasurement) : Prop :=
  match method with
  | LongJumpMeasurement.ShortestPerpendicular => true
  | _ => false

/-- Theorem: The shortest perpendicular line segment is the most accurate and consistent method for measuring long jump performance -/
theorem shortest_perpendicular_best_measurement :
  ∀ (method : LongJumpMeasurement),
    isAccurate method ∧ isConsistent method ↔ method = LongJumpMeasurement.ShortestPerpendicular :=
by sorry

end shortest_perpendicular_best_measurement_l1570_157077


namespace y₁_not_in_third_quadrant_l1570_157010

-- Define the linear functions
def y₁ (x : ℝ) (b : ℝ) : ℝ := -x + b
def y₂ (x : ℝ) : ℝ := -x

-- State the theorem
theorem y₁_not_in_third_quadrant :
  ∃ b : ℝ, (∀ x : ℝ, y₁ x b = y₂ x + 2) →
  ∀ x y : ℝ, y = y₁ x b → (x < 0 ∧ y < 0 → False) := by
  sorry

end y₁_not_in_third_quadrant_l1570_157010


namespace lower_price_option2_l1570_157039

def initial_value : ℝ := 12000

def apply_discount (price : ℝ) (discount : ℝ) : ℝ :=
  price * (1 - discount)

def option1_final_price : ℝ :=
  apply_discount (apply_discount (apply_discount initial_value 0.3) 0.1) 0.05

def option2_final_price : ℝ :=
  apply_discount (apply_discount (apply_discount initial_value 0.3) 0.05) 0.15

theorem lower_price_option2 :
  option2_final_price < option1_final_price ∧ option2_final_price = 6783 :=
by sorry

end lower_price_option2_l1570_157039


namespace least_number_added_for_divisibility_l1570_157003

theorem least_number_added_for_divisibility (x : ℕ) : 
  (∀ y : ℕ, y < x → ¬(23 ∣ (1054 + y))) ∧ (23 ∣ (1054 + x)) → x = 4 := by
  sorry

end least_number_added_for_divisibility_l1570_157003


namespace circle_area_ratio_l1570_157046

theorem circle_area_ratio : 
  ∀ (r1 r2 : ℝ), r1 > 0 → r2 = 3 * r1 → 
  (π * r2^2) / (π * r1^2) = 9 :=
by
  sorry

end circle_area_ratio_l1570_157046


namespace square_area_26_l1570_157051

/-- The area of a square with vertices at (0, 0), (-5, -1), (-4, -6), and (1, -5) is 26 square units. -/
theorem square_area_26 : 
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (-5, -1)
  let C : ℝ × ℝ := (-4, -6)
  let D : ℝ × ℝ := (1, -5)
  let square_area := (B.1 - A.1)^2 + (B.2 - A.2)^2
  square_area = 26 := by
  sorry


end square_area_26_l1570_157051


namespace inequality_solution_l1570_157068

theorem inequality_solution (x : ℝ) :
  x ≥ -14 → (x + 2 < Real.sqrt (x + 14) ↔ -14 ≤ x ∧ x < 2) := by
  sorry

end inequality_solution_l1570_157068


namespace solution_set_min_value_min_value_achieved_l1570_157074

-- Define the function f
def f (x : ℝ) : ℝ := |2 * x - 1|

-- Part 1: Theorem for the solution set of f(2x) ≤ f(x + 1)
theorem solution_set (x : ℝ) : f (2 * x) ≤ f (x + 1) ↔ 0 ≤ x ∧ x ≤ 1 := by sorry

-- Part 2: Theorem for the minimum value of f(a²) + f(b²)
theorem min_value (a b : ℝ) (h : a + b = 2) : 
  ∃ (m : ℝ), m = 2 ∧ ∀ (a' b' : ℝ), a' + b' = 2 → f (a' ^ 2) + f (b' ^ 2) ≥ m := by sorry

-- Corollary: The minimum is achieved when a = b = 1
theorem min_value_achieved (a b : ℝ) (h : a + b = 2) : 
  f (a ^ 2) + f (b ^ 2) = 2 ↔ a = 1 ∧ b = 1 := by sorry

end solution_set_min_value_min_value_achieved_l1570_157074


namespace janice_homework_time_l1570_157026

/-- Represents the time (in minutes) it takes Janice to complete various tasks before watching a movie -/
structure JanicesTasks where
  total_time : ℝ
  homework_time : ℝ
  cleaning_time : ℝ
  dog_walking_time : ℝ
  trash_time : ℝ
  remaining_time : ℝ

/-- The theorem stating that Janice's homework time is 30 minutes given the conditions -/
theorem janice_homework_time (tasks : JanicesTasks) :
  tasks.total_time = 120 ∧
  tasks.cleaning_time = tasks.homework_time / 2 ∧
  tasks.dog_walking_time = tasks.homework_time + 5 ∧
  tasks.trash_time = tasks.homework_time / 6 ∧
  tasks.remaining_time = 35 ∧
  tasks.total_time = tasks.homework_time + tasks.cleaning_time + tasks.dog_walking_time + tasks.trash_time + tasks.remaining_time
  →
  tasks.homework_time = 30 :=
by sorry

end janice_homework_time_l1570_157026


namespace jason_initial_money_l1570_157080

theorem jason_initial_money (jason_current : ℕ) (jason_earned : ℕ) 
  (h1 : jason_current = 63) 
  (h2 : jason_earned = 60) : 
  jason_current - jason_earned = 3 := by
sorry

end jason_initial_money_l1570_157080


namespace pet_store_kittens_l1570_157097

/-- The number of kittens initially at the pet store -/
def initial_kittens : ℕ := 6

/-- The number of puppies initially at the pet store -/
def initial_puppies : ℕ := 7

/-- The number of puppies sold -/
def puppies_sold : ℕ := 2

/-- The number of kittens sold -/
def kittens_sold : ℕ := 3

/-- The number of pets remaining after the sale -/
def remaining_pets : ℕ := 8

theorem pet_store_kittens :
  initial_puppies - puppies_sold + (initial_kittens - kittens_sold) = remaining_pets :=
by sorry

end pet_store_kittens_l1570_157097


namespace inequality_proof_l1570_157091

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (3 : ℝ) / (a^3 + b^3 + c^3) ≤ 1 / (a^3 + b^3 + a*b*c) + 1 / (b^3 + c^3 + a*b*c) + 1 / (c^3 + a^3 + a*b*c) ∧
  1 / (a^3 + b^3 + a*b*c) + 1 / (b^3 + c^3 + a*b*c) + 1 / (c^3 + a^3 + a*b*c) ≤ 1 / (a*b*c) :=
by sorry

end inequality_proof_l1570_157091


namespace problem_statement_l1570_157031

theorem problem_statement : 
  let a := ((7 + 4 * Real.sqrt 3)^(1/2) - (7 - 4 * Real.sqrt 3)^(1/2)) / Real.sqrt 3
  a = 2 := by
  sorry

end problem_statement_l1570_157031


namespace transport_tax_calculation_l1570_157018

/-- Calculates the transport tax for a vehicle -/
def transportTax (horsepower : ℕ) (taxRate : ℕ) (ownershipMonths : ℕ) : ℕ :=
  horsepower * taxRate * ownershipMonths / 12

/-- Proves that the transport tax for the given conditions is 2000 rubles -/
theorem transport_tax_calculation :
  transportTax 150 20 8 = 2000 := by
  sorry

end transport_tax_calculation_l1570_157018


namespace sin_pi_six_l1570_157028

theorem sin_pi_six : Real.sin (π / 6) = 1 / 2 := by
  sorry

end sin_pi_six_l1570_157028


namespace orange_caterpillar_length_l1570_157087

theorem orange_caterpillar_length 
  (green_length : ℝ) 
  (length_difference : ℝ) 
  (h1 : green_length = 3) 
  (h2 : length_difference = 1.83) 
  (h3 : green_length = length_difference + orange_length) : 
  orange_length = 1.17 := by
  sorry

end orange_caterpillar_length_l1570_157087


namespace percentage_exceeding_speed_limit_l1570_157088

/-- Given a road where:
  * 10% of motorists receive speeding tickets
  * 60% of motorists who exceed the speed limit do not receive tickets
  Prove that 25% of motorists exceed the speed limit -/
theorem percentage_exceeding_speed_limit
  (total_motorists : ℝ)
  (h_positive : total_motorists > 0)
  (ticketed_percentage : ℝ)
  (h_ticketed : ticketed_percentage = 0.1)
  (non_ticketed_speeders_percentage : ℝ)
  (h_non_ticketed : non_ticketed_speeders_percentage = 0.6)
  : (ticketed_percentage * total_motorists) / (1 - non_ticketed_speeders_percentage) / total_motorists = 0.25 := by
  sorry

end percentage_exceeding_speed_limit_l1570_157088


namespace triangle_perimeter_range_l1570_157086

open Real

theorem triangle_perimeter_range (A B C a b c : ℝ) : 
  -- Triangle ABC is acute
  0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2 ∧
  -- Sum of angles in a triangle
  A + B + C = π ∧
  -- Given equation
  cos B^2 + cos B * cos (A - C) = sin A * sin C ∧
  -- Side length a
  a = 2 * Real.sqrt 3 ∧
  -- Derived value of B
  B = π/3 ∧
  -- Definition of sides using sine rule
  b = a * sin B / sin A ∧
  c = a * sin C / sin A
  →
  -- Perimeter range
  3 + 3 * Real.sqrt 3 < a + b + c ∧ a + b + c < 6 + 6 * Real.sqrt 3 :=
by sorry

end triangle_perimeter_range_l1570_157086


namespace four_solutions_gg_eq_3_l1570_157059

def g (x : ℝ) : ℝ := x^2 - 4*x + 3

def domain (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 5

theorem four_solutions_gg_eq_3 :
  ∃! (s : Finset ℝ), s.card = 4 ∧ 
  (∀ x ∈ s, domain x ∧ g (g x) = 3) ∧
  (∀ x, domain x → g (g x) = 3 → x ∈ s) :=
sorry

end four_solutions_gg_eq_3_l1570_157059


namespace sum_of_fifth_powers_l1570_157056

theorem sum_of_fifth_powers (ζ₁ ζ₂ ζ₃ : ℂ) 
  (h1 : ζ₁ + ζ₂ + ζ₃ = 2)
  (h2 : ζ₁^2 + ζ₂^2 + ζ₃^2 = 5)
  (h3 : ζ₁^3 + ζ₂^3 + ζ₃^3 = 14) :
  ζ₁^5 + ζ₂^5 + ζ₃^5 = 44 := by
  sorry

end sum_of_fifth_powers_l1570_157056


namespace final_sugar_amount_l1570_157044

def sugar_calculation (initial : ℕ) (used : ℕ) (bought : ℕ) : ℕ :=
  initial - used + bought

theorem final_sugar_amount :
  sugar_calculation 65 18 50 = 97 := by
  sorry

end final_sugar_amount_l1570_157044


namespace problem_solution_l1570_157058

def f (x : ℝ) : ℝ := |2*x + 3| + |x - 1|

theorem problem_solution :
  (∀ x : ℝ, f x > 4 ↔ x ∈ Set.Iio (-2) ∪ Set.Ioi 0) ∧
  (∀ m : ℝ, (∃ x₀ : ℝ, ∀ t : ℝ, f x₀ < |m + t| + |t - m|) ↔ 
    m ∈ Set.Iio (-5/4) ∪ Set.Ioi (5/4)) :=
by sorry

end problem_solution_l1570_157058


namespace intersects_x_axis_once_iff_m_range_l1570_157085

/-- The function f(x) = x³ - x² - x + m -/
def f (m : ℝ) (x : ℝ) : ℝ := x^3 - x^2 - x + m

/-- The derivative of f(x) -/
def f_derivative (x : ℝ) : ℝ := 3*x^2 - 2*x - 1

theorem intersects_x_axis_once_iff_m_range (m : ℝ) :
  (∃! x, f m x = 0) ↔ (m < -5/27 ∨ m > 0) := by sorry

end intersects_x_axis_once_iff_m_range_l1570_157085


namespace complex_power_equality_l1570_157007

theorem complex_power_equality : (((1 + Complex.I) / (1 - Complex.I)) ^ 2016 = 1) := by
  sorry

end complex_power_equality_l1570_157007


namespace students_present_l1570_157083

theorem students_present (total : ℕ) (absent_percentage : ℚ) : 
  total = 50 ∧ absent_percentage = 14/100 → 
  total - (total * absent_percentage).floor = 43 := by
  sorry

end students_present_l1570_157083


namespace train_length_calculation_l1570_157021

theorem train_length_calculation (v_fast v_slow : ℝ) (t : ℝ) (h1 : v_fast = 46) (h2 : v_slow = 36) (h3 : t = 27) : ∃ L : ℝ,
  L = 37.5 ∧ 
  2 * L = (v_fast - v_slow) * (5 / 18) * t :=
by sorry

end train_length_calculation_l1570_157021


namespace min_value_sum_of_squares_l1570_157071

theorem min_value_sum_of_squares (u v w : ℝ) 
  (h_pos_u : u > 0) (h_pos_v : v > 0) (h_pos_w : w > 0)
  (h_sum_squares : u^2 + v^2 + w^2 = 8) :
  (u^4 / 9) + (v^4 / 16) + (w^4 / 25) ≥ 32/25 := by
  sorry

end min_value_sum_of_squares_l1570_157071


namespace magic_card_price_l1570_157066

/-- The initial price of a Magic card that triples in value and results in a $200 profit when sold -/
def initial_price : ℝ := 100

/-- The value of the card after tripling -/
def tripled_value (price : ℝ) : ℝ := 3 * price

/-- The profit made from selling the card -/
def profit (initial_price : ℝ) : ℝ := tripled_value initial_price - initial_price

theorem magic_card_price :
  profit initial_price = 200 ∧ tripled_value initial_price = 3 * initial_price := by
  sorry

end magic_card_price_l1570_157066


namespace total_wool_is_82_l1570_157030

/-- The number of scarves Aaron makes -/
def aaron_scarves : ℕ := 10

/-- The number of sweaters Aaron makes -/
def aaron_sweaters : ℕ := 5

/-- The number of sweaters Enid makes -/
def enid_sweaters : ℕ := 8

/-- The number of balls of wool used for one scarf -/
def wool_per_scarf : ℕ := 3

/-- The number of balls of wool used for one sweater -/
def wool_per_sweater : ℕ := 4

/-- The total number of balls of wool used by Enid and Aaron -/
def total_wool : ℕ := aaron_scarves * wool_per_scarf + aaron_sweaters * wool_per_sweater + enid_sweaters * wool_per_sweater

theorem total_wool_is_82 : total_wool = 82 := by sorry

end total_wool_is_82_l1570_157030


namespace prob_green_ball_l1570_157055

-- Define the containers and their contents
structure Container where
  red_balls : Nat
  green_balls : Nat

-- Define the probabilities
def prob_container : Rat := 1 / 3
def prob_green (c : Container) : Rat := c.green_balls / (c.red_balls + c.green_balls)

-- Define the containers
def container_A : Container := ⟨5, 5⟩
def container_B : Container := ⟨7, 3⟩
def container_C : Container := ⟨6, 4⟩

-- State the theorem
theorem prob_green_ball : 
  prob_container * prob_green container_A +
  prob_container * prob_green container_B +
  prob_container * prob_green container_C = 2 / 5 := by
  sorry


end prob_green_ball_l1570_157055


namespace quadratic_roots_complex_l1570_157067

theorem quadratic_roots_complex (x : ℂ) : 
  x^2 + 6*x + 13 = 0 ↔ (x + 3*I) * (x - 3*I) = 0 :=
sorry

end quadratic_roots_complex_l1570_157067


namespace right_triangle_division_l1570_157042

theorem right_triangle_division (n : ℝ) (h : n > 0) :
  ∀ (rect_area rect_short rect_long small_triangle1_area : ℝ),
    rect_short > 0 →
    rect_long > 0 →
    rect_area > 0 →
    small_triangle1_area > 0 →
    rect_long = 3 * rect_short →
    rect_area = rect_short * rect_long →
    small_triangle1_area = n * rect_area →
    ∃ (small_triangle2_area : ℝ),
      small_triangle2_area > 0 ∧
      small_triangle2_area / rect_area = 1 / (4 * n) :=
by sorry

end right_triangle_division_l1570_157042


namespace binomial_10_1_l1570_157004

theorem binomial_10_1 : Nat.choose 10 1 = 10 := by
  sorry

end binomial_10_1_l1570_157004


namespace closest_to_target_l1570_157009

def target : ℕ := 100000

def numbers : List ℕ := [100260, 99830, 98900, 100320]

def distance (x : ℕ) : ℕ := Int.natAbs (x - target)

theorem closest_to_target : 
  ∀ n ∈ numbers, distance 99830 ≤ distance n :=
by sorry

end closest_to_target_l1570_157009


namespace circle_radius_is_two_l1570_157032

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 8*x + y^2 - 4*y + 16 = 0

/-- The radius of the circle -/
def circle_radius : ℝ := 2

/-- Theorem: The radius of the circle with the given equation is 2 -/
theorem circle_radius_is_two :
  ∃ (h k : ℝ), ∀ (x y : ℝ), circle_equation x y ↔ (x - h)^2 + (y - k)^2 = circle_radius^2 :=
sorry

end circle_radius_is_two_l1570_157032


namespace work_completion_days_l1570_157081

/-- Calculates the initial planned days to complete a work given the original number of workers,
    the number of absent workers, and the days taken by the remaining workers. -/
def initialPlannedDays (originalWorkers : ℕ) (absentWorkers : ℕ) (daysWithFewerWorkers : ℕ) : ℚ :=
  (originalWorkers - absentWorkers) * daysWithFewerWorkers / originalWorkers

/-- Proves that given 15 original workers, 5 absent workers, and 60 days taken by the remaining workers,
    the initial planned days to complete the work is 40. -/
theorem work_completion_days : initialPlannedDays 15 5 60 = 40 := by
  sorry

end work_completion_days_l1570_157081


namespace total_profit_equation_l1570_157099

/-- Represents the initial investment of person A in rupees -/
def initial_investment_A : ℚ := 2000

/-- Represents the initial investment of person B in rupees -/
def initial_investment_B : ℚ := 4000

/-- Represents the number of months before investment change -/
def months_before_change : ℕ := 8

/-- Represents the number of months after investment change -/
def months_after_change : ℕ := 4

/-- Represents the amount A withdrew after 8 months in rupees -/
def amount_A_withdrew : ℚ := 1000

/-- Represents the amount B added after 8 months in rupees -/
def amount_B_added : ℚ := 1000

/-- Represents A's share of the profit in rupees -/
def A_profit_share : ℚ := 175

/-- Calculates the total investment of A over the year -/
def total_investment_A : ℚ :=
  initial_investment_A * months_before_change +
  (initial_investment_A - amount_A_withdrew) * months_after_change

/-- Calculates the total investment of B over the year -/
def total_investment_B : ℚ :=
  initial_investment_B * months_before_change +
  (initial_investment_B + amount_B_added) * months_after_change

/-- Theorem stating that the total profit P satisfies the equation (5/18) * P = 175 -/
theorem total_profit_equation (P : ℚ) :
  total_investment_A / (total_investment_A + total_investment_B) * P = A_profit_share := by
  sorry

end total_profit_equation_l1570_157099


namespace prime_square_mod_360_l1570_157013

theorem prime_square_mod_360 (p : Nat) (h_prime : Prime p) (h_gt_5 : p > 5) :
  (p^2 : Nat) % 360 = 1 ∨ (p^2 : Nat) % 360 = 289 := by
  sorry

#check prime_square_mod_360

end prime_square_mod_360_l1570_157013


namespace pure_imaginary_complex_number_l1570_157082

theorem pure_imaginary_complex_number (a : ℝ) : 
  (Complex.I + 1) * (a + 2 * Complex.I) * Complex.I = Complex.I * (a + 2 : ℝ) → a = 2 := by
  sorry

end pure_imaginary_complex_number_l1570_157082


namespace alpha_plus_beta_eq_107_l1570_157096

theorem alpha_plus_beta_eq_107 :
  ∃ α β : ℝ, (∀ x : ℝ, (x - α) / (x + β) = (x^2 - 64*x + 992) / (x^2 + 72*x - 2184)) →
  α + β = 107 := by
  sorry

end alpha_plus_beta_eq_107_l1570_157096


namespace gumball_solution_l1570_157008

/-- Represents the gumball distribution problem --/
def gumball_problem (total : ℕ) (todd : ℕ) (alisha : ℕ) (bobby : ℕ) : Prop :=
  total = 45 ∧
  todd = 4 ∧
  alisha = 2 * todd ∧
  bobby = 4 * alisha - 5 ∧
  total - (todd + alisha + bobby) = 6

/-- Theorem stating that the gumball problem has a solution --/
theorem gumball_solution : ∃ (total todd alisha bobby : ℕ), gumball_problem total todd alisha bobby :=
sorry

end gumball_solution_l1570_157008


namespace simplify_nested_radicals_l1570_157062

theorem simplify_nested_radicals : 
  2 * Real.sqrt (3 + Real.sqrt (5 - Real.sqrt (13 + Real.sqrt 48))) = Real.sqrt 6 + Real.sqrt 2 := by
sorry

end simplify_nested_radicals_l1570_157062


namespace min_side_length_l1570_157011

theorem min_side_length (PQ PR SR SQ : ℝ) (h1 : PQ = 7.5) (h2 : PR = 14.5) (h3 : SR = 9.5) (h4 : SQ = 23.5) :
  ∃ (QR : ℕ), (QR : ℝ) > PR - PQ ∧ (QR : ℝ) > SQ - SR ∧ ∀ (n : ℕ), (n : ℝ) > PR - PQ ∧ (n : ℝ) > SQ - SR → n ≥ QR :=
by
  sorry

#check min_side_length

end min_side_length_l1570_157011


namespace ethans_net_income_l1570_157012

/-- Calculates Ethan's net income after deductions for a 5-week period -/
def calculate_net_income (hourly_wage : ℚ) (hours_per_day : ℕ) (days_per_week : ℕ) 
  (total_weeks : ℕ) (tax_rate : ℚ) (health_insurance_per_week : ℚ) (retirement_rate : ℚ) : ℚ :=
  let gross_income := hourly_wage * hours_per_day * days_per_week * total_weeks
  let income_tax := tax_rate * gross_income
  let health_insurance := health_insurance_per_week * total_weeks
  let retirement_contribution := retirement_rate * gross_income
  let total_deductions := income_tax + health_insurance + retirement_contribution
  gross_income - total_deductions

/-- Theorem stating that Ethan's net income after deductions for a 5-week period is $2447 -/
theorem ethans_net_income : 
  calculate_net_income 18 8 5 5 (15/100) 65 (8/100) = 2447 := by
  sorry

end ethans_net_income_l1570_157012


namespace imaginary_part_of_complex_product_l1570_157057

theorem imaginary_part_of_complex_product : 
  let i : ℂ := Complex.I
  let z : ℂ := (1 + i)^2 * (2 + i)
  Complex.im z = 4 :=
by sorry

end imaginary_part_of_complex_product_l1570_157057


namespace a_4_equals_zero_l1570_157024

def a (n : ℕ+) : ℤ := n^2 - 3*n - 4

theorem a_4_equals_zero : a 4 = 0 := by
  sorry

end a_4_equals_zero_l1570_157024


namespace divisible_by_seven_l1570_157079

theorem divisible_by_seven (k : ℕ) : ∃ m : ℤ, 2^(6*k+1) + 3^(6*k+1) + 5^(6*k) + 1 = 7*m := by
  sorry

end divisible_by_seven_l1570_157079


namespace basil_pots_l1570_157006

theorem basil_pots (rosemary_pots thyme_pots : ℕ)
  (basil_leaves rosemary_leaves thyme_leaves total_leaves : ℕ) :
  rosemary_pots = 9 →
  thyme_pots = 6 →
  basil_leaves = 4 →
  rosemary_leaves = 18 →
  thyme_leaves = 30 →
  total_leaves = 354 →
  ∃ basil_pots : ℕ,
    basil_pots * basil_leaves +
    rosemary_pots * rosemary_leaves +
    thyme_pots * thyme_leaves = total_leaves ∧
    basil_pots = 3 :=
by sorry

end basil_pots_l1570_157006


namespace expression_evaluation_l1570_157037

theorem expression_evaluation :
  let x : ℤ := -1
  let y : ℤ := 2
  let A : ℤ := 2*x + y
  let B : ℤ := 2*x - y
  (A^2 - B^2) * (x - 2*y) = 80 := by
sorry

end expression_evaluation_l1570_157037


namespace intersection_empty_union_equals_A_l1570_157025

-- Define sets A and B
def A : Set ℝ := {x : ℝ | -3 < x ∧ x < 4}
def B (a : ℝ) : Set ℝ := {x : ℝ | x^2 - 4*a*x + 3*a^2 = 0}

-- Theorem for part (1)
theorem intersection_empty (a : ℝ) : 
  A ∩ B a = ∅ ↔ a ≤ -3 ∨ a ≥ 4 :=
sorry

-- Theorem for part (2)
theorem union_equals_A (a : ℝ) :
  A ∪ B a = A ↔ -1 < a ∧ a < 4/3 :=
sorry

end intersection_empty_union_equals_A_l1570_157025


namespace russian_football_championship_l1570_157029

/-- Represents a football championship. -/
structure Championship where
  teams : ℕ
  matches_per_pair : ℕ

/-- Calculate the number of matches a single team plays. -/
def matches_per_team (c : Championship) : ℕ :=
  (c.teams - 1) * c.matches_per_pair

/-- Calculate the total number of matches in the championship. -/
def total_matches (c : Championship) : ℕ :=
  (c.teams * matches_per_team c) / 2

theorem russian_football_championship 
  (c : Championship) 
  (h1 : c.teams = 16) 
  (h2 : c.matches_per_pair = 2) : 
  matches_per_team c = 30 ∧ total_matches c = 240 := by
  sorry

#eval matches_per_team ⟨16, 2⟩
#eval total_matches ⟨16, 2⟩

end russian_football_championship_l1570_157029


namespace solve_system_l1570_157023

theorem solve_system (x y : ℝ) (h1 : 2 * x + y = 4) (h2 : (x + y) / 3 = 1) :
  x + 2 * y = 5 := by
  sorry

end solve_system_l1570_157023


namespace snake_paint_theorem_l1570_157048

/-- The amount of paint needed for a single cube -/
def paint_per_cube : ℕ := 60

/-- The number of cubes in the snake -/
def total_cubes : ℕ := 2016

/-- The number of cubes in each periodic fragment -/
def cubes_per_fragment : ℕ := 6

/-- The additional paint needed for adjustments -/
def additional_paint : ℕ := 20

/-- The total amount of paint needed for the snake -/
def total_paint_needed : ℕ :=
  (total_cubes / cubes_per_fragment) * (cubes_per_fragment * paint_per_cube) + additional_paint

theorem snake_paint_theorem :
  total_paint_needed = 120980 := by
  sorry

end snake_paint_theorem_l1570_157048


namespace gymnastics_performance_participants_l1570_157038

/-- The number of grades participating in the gymnastics performance -/
def num_grades : ℕ := 3

/-- The number of classes in each grade -/
def classes_per_grade : ℕ := 4

/-- The number of participants selected from each class -/
def participants_per_class : ℕ := 15

/-- The total number of participants in the gymnastics performance -/
def total_participants : ℕ := num_grades * classes_per_grade * participants_per_class

theorem gymnastics_performance_participants : total_participants = 180 := by
  sorry

end gymnastics_performance_participants_l1570_157038


namespace inequality_proof_l1570_157078

theorem inequality_proof (m n : ℝ) (h1 : m < n) (h2 : 1/m < 1/n) : m < 0 ∧ 0 < n := by
  sorry

end inequality_proof_l1570_157078


namespace a2_value_l1570_157040

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

def geometric_sequence (a b c : ℝ) :=
  b^2 = a * c

theorem a2_value (a : ℕ → ℝ) :
  arithmetic_sequence a 2 →
  geometric_sequence (a 1) (a 3) (a 4) →
  a 2 = -6 := by
  sorry

end a2_value_l1570_157040


namespace work_left_fraction_l1570_157084

theorem work_left_fraction (days_A days_B days_together : ℕ) (h1 : days_A = 20) (h2 : days_B = 30) (h3 : days_together = 4) :
  1 - (days_together : ℚ) * (1 / days_A + 1 / days_B) = 2/3 := by
  sorry

end work_left_fraction_l1570_157084


namespace complex_expression_equals_9980_l1570_157052

theorem complex_expression_equals_9980 : 
  3 * 995 + 4 * 996 + 5 * 997 + 6 * 998 + 7 * 999 - 4985 * 3 = 9980 := by
  sorry

end complex_expression_equals_9980_l1570_157052


namespace median_of_special_list_l1570_157034

/-- The sum of integers from 1 to n -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The list length -/
def list_length : ℕ := triangular_number 150

/-- The median position -/
def median_position : ℕ := (list_length + 1) / 2

/-- The cumulative count up to n -/
def cumulative_count (n : ℕ) : ℕ := triangular_number n

theorem median_of_special_list : ∃ (n : ℕ), n = 106 ∧ 
  cumulative_count (n - 1) < median_position ∧ 
  cumulative_count n ≥ median_position := by sorry

end median_of_special_list_l1570_157034


namespace expected_attacked_squares_l1570_157094

/-- The number of squares on a chessboard -/
def chessboardSquares : ℕ := 64

/-- The number of rooks placed on the chessboard -/
def numberOfRooks : ℕ := 3

/-- The probability that a specific square is not attacked by a single rook -/
def probNotAttacked : ℚ := (49 : ℚ) / 64

/-- The expected number of squares under attack by three randomly placed rooks on a chessboard -/
def expectedAttackedSquares : ℚ := chessboardSquares * (1 - probNotAttacked ^ numberOfRooks)

/-- Theorem stating the expected number of squares under attack -/
theorem expected_attacked_squares :
  expectedAttackedSquares = 64 * (1 - (49/64)^3) :=
sorry

end expected_attacked_squares_l1570_157094


namespace product_of_factorials_plus_one_l1570_157022

theorem product_of_factorials_plus_one : 
  (1 + 1 / 1) * 
  (1 + 1 / 2) * 
  (1 + 1 / 6) * 
  (1 + 1 / 24) * 
  (1 + 1 / 120) * 
  (1 + 1 / 720) * 
  (1 + 1 / 5040) = 5041 / 5040 := by sorry

end product_of_factorials_plus_one_l1570_157022

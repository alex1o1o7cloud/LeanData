import Mathlib

namespace ternary_121_equals_16_l3871_387181

/-- Converts a ternary (base-3) number to decimal (base-10) --/
def ternary_to_decimal (a b c : ℕ) : ℕ :=
  a * 3^2 + b * 3^1 + c * 3^0

/-- Proves that the ternary number 121₃ is equal to the decimal number 16 --/
theorem ternary_121_equals_16 : ternary_to_decimal 1 2 1 = 16 := by
  sorry

end ternary_121_equals_16_l3871_387181


namespace vanessa_record_score_l3871_387165

/-- Vanessa's record-setting basketball score --/
theorem vanessa_record_score (total_team_score : ℕ) (other_players : ℕ) (other_players_avg : ℚ)
  (h1 : total_team_score = 48)
  (h2 : other_players = 6)
  (h3 : other_players_avg = 3.5) :
  total_team_score - (other_players : ℚ) * other_players_avg = 27 := by
  sorry

end vanessa_record_score_l3871_387165


namespace solve_equation_binomial_identity_l3871_387182

-- Define A_x as the falling factorial
def A (x : ℕ) (n : ℕ) : ℕ := 
  if n ≤ x then
    (x - n + 1).factorial / (x - n).factorial
  else 0

-- Define binomial coefficient
def C (n : ℕ) (k : ℕ) : ℕ :=
  if k ≤ n then
    n.factorial / (k.factorial * (n - k).factorial)
  else 0

theorem solve_equation : ∃ x : ℕ, x > 3 ∧ 3 * A x 3 = 2 * A (x + 1) 2 + 6 * A x 2 ∧ x = 5 := by
  sorry

theorem binomial_identity (n k : ℕ) (h : k ≤ n) : k * C n k = n * C (n - 1) (k - 1) := by
  sorry

end solve_equation_binomial_identity_l3871_387182


namespace distinct_sums_l3871_387163

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  hd : d ≠ 0
  h_arithmetic : ∀ n, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * seq.a 1 + (n * (n - 1) : ℚ) / 2 * seq.d

theorem distinct_sums (seq : ArithmeticSequence) 
  (h_sum_5 : S seq 5 = 0) : 
  Finset.card (Finset.image (S seq) (Finset.range 100)) = 98 := by
  sorry

end distinct_sums_l3871_387163


namespace det_special_matrix_l3871_387108

theorem det_special_matrix (x : ℝ) : 
  Matrix.det !![x + 2, x, x; x, x + 2, x; x, x, x + 2] = 8 * x + 8 := by
  sorry

end det_special_matrix_l3871_387108


namespace trig_sum_equals_one_l3871_387162

theorem trig_sum_equals_one :
  let angle_to_real (θ : ℤ) : ℝ := (θ % 360 : ℝ) * Real.pi / 180
  Real.sin (angle_to_real (-120)) * Real.cos (angle_to_real 1290) +
  Real.cos (angle_to_real (-1020)) * Real.sin (angle_to_real (-1050)) = 1 := by
  sorry

end trig_sum_equals_one_l3871_387162


namespace baseball_hits_theorem_l3871_387135

def total_hits : ℕ := 50
def home_runs : ℕ := 3
def triples : ℕ := 2
def doubles : ℕ := 10

def singles : ℕ := total_hits - (home_runs + triples + doubles)

theorem baseball_hits_theorem :
  singles = 35 ∧ (singles : ℚ) / total_hits * 100 = 70 := by
  sorry

end baseball_hits_theorem_l3871_387135


namespace divide_600_in_ratio_1_2_l3871_387136

def divide_in_ratio (total : ℚ) (ratio1 ratio2 : ℕ) : ℚ :=
  total * ratio1 / (ratio1 + ratio2)

theorem divide_600_in_ratio_1_2 :
  divide_in_ratio 600 1 2 = 200 := by
  sorry

end divide_600_in_ratio_1_2_l3871_387136


namespace continuous_piecewise_function_sum_l3871_387122

noncomputable def g (a c : ℝ) (x : ℝ) : ℝ :=
  if x > 3 then 2 * a * x + 4
  else if -3 ≤ x ∧ x ≤ 3 then x^2 - 7
  else 3 * x - c

def IsContinuous (f : ℝ → ℝ) : Prop :=
  ∀ x₀ : ℝ, ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, |x - x₀| < δ → |f x - f x₀| < ε

theorem continuous_piecewise_function_sum (a c : ℝ) :
  IsContinuous (g a c) → a + c = -34/3 := by
  sorry

end continuous_piecewise_function_sum_l3871_387122


namespace f_seven_eq_neg_seventeen_l3871_387102

/-- Given a function f(x) = ax^7 + bx^3 + cx - 5, if f(-7) = 7, then f(7) = -17 -/
theorem f_seven_eq_neg_seventeen 
  (a b c : ℝ) 
  (f : ℝ → ℝ)
  (h1 : ∀ x, f x = a * x^7 + b * x^3 + c * x - 5)
  (h2 : f (-7) = 7) : 
  f 7 = -17 := by
  sorry

end f_seven_eq_neg_seventeen_l3871_387102


namespace division_of_fractions_l3871_387170

theorem division_of_fractions : (4 : ℚ) / (8 / 13) = 13 / 2 := by
  sorry

end division_of_fractions_l3871_387170


namespace quadratic_inequality_theorem_l3871_387106

-- Define the quadratic function
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_inequality_theorem (a b c : ℝ) :
  (∀ x : ℝ, a * x^2 + b * x + c > 0 ↔ x < -2 ∨ x > 4) →
  f a b c 2 < f a b c (-1) ∧ f a b c (-1) < f a b c 5 :=
sorry

end quadratic_inequality_theorem_l3871_387106


namespace lcm_of_4_6_10_18_l3871_387123

theorem lcm_of_4_6_10_18 : Nat.lcm 4 (Nat.lcm 6 (Nat.lcm 10 18)) = 180 := by
  sorry

end lcm_of_4_6_10_18_l3871_387123


namespace exponent_division_l3871_387156

theorem exponent_division (a : ℝ) : a^3 / a^2 = a := by sorry

end exponent_division_l3871_387156


namespace adult_tickets_sold_l3871_387175

/-- Proves that given the conditions of ticket prices, total revenue, and total tickets sold,
    the number of adult tickets sold is 22. -/
theorem adult_tickets_sold (adult_price child_price total_revenue total_tickets : ℕ) 
  (h1 : adult_price = 8)
  (h2 : child_price = 5)
  (h3 : total_revenue = 236)
  (h4 : total_tickets = 34) :
  ∃ (adult_tickets : ℕ), 
    adult_tickets * adult_price + (total_tickets - adult_tickets) * child_price = total_revenue ∧
    adult_tickets = 22 := by
  sorry


end adult_tickets_sold_l3871_387175


namespace james_hourly_rate_l3871_387161

/-- Represents the car rental scenario for James --/
structure CarRental where
  hours_per_day : ℕ
  days_per_week : ℕ
  weekly_income : ℕ

/-- Calculates the hourly rate for car rental --/
def hourly_rate (rental : CarRental) : ℚ :=
  rental.weekly_income / (rental.hours_per_day * rental.days_per_week)

/-- Theorem stating that James' hourly rate is $20 --/
theorem james_hourly_rate :
  let james_rental : CarRental := {
    hours_per_day := 8,
    days_per_week := 4,
    weekly_income := 640
  }
  hourly_rate james_rental = 20 := by sorry

end james_hourly_rate_l3871_387161


namespace tree_height_calculation_l3871_387179

/-- Given a tree and a pole with their respective shadows, calculate the height of the tree -/
theorem tree_height_calculation (tree_shadow : ℝ) (pole_height : ℝ) (pole_shadow : ℝ) :
  tree_shadow = 30 →
  pole_height = 1.5 →
  pole_shadow = 3 →
  (tree_shadow * pole_height) / pole_shadow = 15 :=
by sorry

end tree_height_calculation_l3871_387179


namespace last_digit_to_appear_is_four_l3871_387176

-- Define the Fibonacci sequence modulo 7
def fibMod7 : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => (fibMod7 n + fibMod7 (n + 1)) % 7

-- Define a function to check if a digit has appeared in the sequence up to n
def digitAppeared (d : ℕ) (n : ℕ) : Prop :=
  ∃ k, k ≤ n ∧ fibMod7 k = d

-- Define a function to check if all digits from 0 to 6 have appeared
def allDigitsAppeared (n : ℕ) : Prop :=
  ∀ d, d ≤ 6 → digitAppeared d n

-- The main theorem
theorem last_digit_to_appear_is_four :
  ∃ n, allDigitsAppeared n ∧ ¬(digitAppeared 4 (n - 1)) :=
sorry

end last_digit_to_appear_is_four_l3871_387176


namespace min_value_problem_l3871_387199

theorem min_value_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + 2*y = 1) :
  (y/x) + (1/y) ≥ 4 ∧ ((y/x) + (1/y) = 4 ↔ x = 1/3 ∧ y = 1/3) := by
  sorry

end min_value_problem_l3871_387199


namespace min_value_problem_l3871_387109

theorem min_value_problem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 2/b = 1) :
  2/(a-1) + 1/(b-2) ≥ 2 :=
by sorry

end min_value_problem_l3871_387109


namespace twins_age_problem_l3871_387129

theorem twins_age_problem (age : ℕ) : 
  (age + 1) * (age + 1) = age * age + 15 → age = 7 := by
  sorry

end twins_age_problem_l3871_387129


namespace parallel_lines_to_plane_not_always_parallel_l3871_387116

structure Plane where

structure Line where

def parallel_line_plane (l : Line) (p : Plane) : Prop := sorry

def parallel_lines (l1 l2 : Line) : Prop := sorry

theorem parallel_lines_to_plane_not_always_parallel (m n : Line) (α : Plane) : 
  m ≠ n → 
  ¬(parallel_line_plane m α → parallel_line_plane n α → parallel_lines m n) := by
  sorry

end parallel_lines_to_plane_not_always_parallel_l3871_387116


namespace boy_running_speed_l3871_387140

theorem boy_running_speed (side_length : ℝ) (time : ℝ) (speed : ℝ) : 
  side_length = 30 →
  time = 36 →
  speed = (4 * side_length / 1000) / (time / 3600) →
  speed = 12 := by
sorry

end boy_running_speed_l3871_387140


namespace rhombus_area_l3871_387184

/-- The area of a rhombus with side length 4 and an angle of 45 degrees between adjacent sides is 8√2 -/
theorem rhombus_area (side : ℝ) (angle : ℝ) : 
  side = 4 → 
  angle = Real.pi / 4 → 
  (side * side * Real.sin angle : ℝ) = 8 * Real.sqrt 2 := by
  sorry

end rhombus_area_l3871_387184


namespace matrix_equation_proof_l3871_387137

theorem matrix_equation_proof :
  let A : Matrix (Fin 2) (Fin 2) ℚ := !![2, -5; 4, -3]
  let B : Matrix (Fin 2) (Fin 2) ℚ := !![-20, -7; 9, 3]
  let N : Matrix (Fin 2) (Fin 2) ℚ := !![44/7, -57/7; -39/14, 51/14]
  N * A = B := by sorry

end matrix_equation_proof_l3871_387137


namespace triangle_area_l3871_387194

/-- The area of a triangle with vertices at (5, -2), (5, 8), and (12, 8) is 35 square units. -/
theorem triangle_area : 
  let v1 : ℝ × ℝ := (5, -2)
  let v2 : ℝ × ℝ := (5, 8)
  let v3 : ℝ × ℝ := (12, 8)
  let area := (1/2) * abs ((v2.1 - v1.1) * (v3.2 - v1.2) - (v3.1 - v1.1) * (v2.2 - v1.2))
  area = 35 := by sorry

end triangle_area_l3871_387194


namespace slope_angle_of_line_through_origin_and_unit_point_l3871_387164

/-- The slope angle of a line passing through (0,0) and (1,1) is π/4 -/
theorem slope_angle_of_line_through_origin_and_unit_point :
  let O : ℝ × ℝ := (0, 0)
  let A : ℝ × ℝ := (1, 1)
  let slope : ℝ := (A.2 - O.2) / (A.1 - O.1)
  let slope_angle : ℝ := Real.arctan slope
  slope_angle = π / 4 := by
  sorry

end slope_angle_of_line_through_origin_and_unit_point_l3871_387164


namespace function_inequality_l3871_387171

theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x : ℝ, deriv f x > f x) (a : ℝ) (ha : a > 0) : 
  f a > Real.exp a * f 0 := by
  sorry

end function_inequality_l3871_387171


namespace max_sum_on_circle_l3871_387141

theorem max_sum_on_circle (x y : ℤ) (h : x^2 + y^2 = 100) : 
  x + y ≤ 14 ∧ ∃ (a b : ℤ), a^2 + b^2 = 100 ∧ a + b = 14 := by
  sorry

end max_sum_on_circle_l3871_387141


namespace smallest_solution_absolute_value_equation_l3871_387120

theorem smallest_solution_absolute_value_equation :
  let x : ℝ := (-3 - Real.sqrt 17) / 2
  (∀ y : ℝ, y * |y| = 3 * y - 2 → x ≤ y) ∧ (x * |x| = 3 * x - 2) := by
  sorry

end smallest_solution_absolute_value_equation_l3871_387120


namespace smallest_even_natural_with_properties_l3871_387174

def is_smallest_even_natural_with_properties (a : ℕ) : Prop :=
  Even a ∧
  (∃ k₁, a + 1 = 3 * k₁) ∧
  (∃ k₂, a + 2 = 5 * k₂) ∧
  (∃ k₃, a + 3 = 7 * k₃) ∧
  (∃ k₄, a + 4 = 11 * k₄) ∧
  (∃ k₅, a + 5 = 13 * k₅) ∧
  (∀ b < a, ¬(is_smallest_even_natural_with_properties b))

theorem smallest_even_natural_with_properties : 
  is_smallest_even_natural_with_properties 788 :=
sorry

end smallest_even_natural_with_properties_l3871_387174


namespace cube_painting_cost_l3871_387173

/-- The cost to paint a cube with given dimensions and paint properties -/
theorem cube_painting_cost
  (side_length : ℝ)
  (paint_cost_per_kg : ℝ)
  (paint_coverage_per_kg : ℝ)
  (h_side : side_length = 10)
  (h_cost : paint_cost_per_kg = 60)
  (h_coverage : paint_coverage_per_kg = 20) :
  side_length ^ 2 * 6 / paint_coverage_per_kg * paint_cost_per_kg = 1800 :=
by sorry

end cube_painting_cost_l3871_387173


namespace semicircle_perimeter_approx_l3871_387148

/-- The perimeter of a semicircle with radius 12 units is approximately 61.7 units. -/
theorem semicircle_perimeter_approx : ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ 
  |π * 12 + 24 - 61.7| < ε := by
  sorry

end semicircle_perimeter_approx_l3871_387148


namespace helen_total_cookies_l3871_387105

/-- The number of cookies Helen baked yesterday -/
def cookies_yesterday : ℕ := 435

/-- The number of cookies Helen baked this morning -/
def cookies_this_morning : ℕ := 139

/-- The total number of cookies Helen baked -/
def total_cookies : ℕ := cookies_yesterday + cookies_this_morning

/-- Theorem stating that the total number of cookies Helen baked is 574 -/
theorem helen_total_cookies : total_cookies = 574 := by
  sorry

end helen_total_cookies_l3871_387105


namespace megans_acorns_l3871_387178

/-- The initial number of acorns Megan had -/
def T : ℝ := 20

/-- Theorem stating the conditions and the correct answer for Megan's acorn problem -/
theorem megans_acorns :
  (0.35 * T = 7) ∧ (0.45 * T = 9) ∧ (T = 20) := by
  sorry

#check megans_acorns

end megans_acorns_l3871_387178


namespace valid_selections_count_l3871_387130

/-- Represents a 6x6 grid of blocks -/
def Grid := Fin 6 → Fin 6 → Bool

/-- Represents a selection of 4 blocks on the grid -/
def Selection := Fin 4 → (Fin 6 × Fin 6)

/-- Checks if a selection forms an L shape -/
def is_L_shape (s : Selection) : Prop := sorry

/-- Checks if no two blocks in the selection share a row or column -/
def no_shared_row_col (s : Selection) : Prop := sorry

/-- The number of valid selections -/
def num_valid_selections : ℕ := sorry

theorem valid_selections_count :
  num_valid_selections = 1800 := by sorry

end valid_selections_count_l3871_387130


namespace specific_pyramid_height_l3871_387145

/-- Represents a right pyramid with a rectangular base -/
structure RightPyramid where
  basePerimeter : ℝ
  baseLength : ℝ
  baseBreadth : ℝ
  apexToVertexDistance : ℝ

/-- Calculates the height of a right pyramid -/
def pyramidHeight (p : RightPyramid) : ℝ :=
  sorry

/-- Theorem stating the height of the specific pyramid -/
theorem specific_pyramid_height :
  let p : RightPyramid := {
    basePerimeter := 40,
    baseLength := 40 / 3,
    baseBreadth := 20 / 3,
    apexToVertexDistance := 15
  }
  pyramidHeight p = 10 * Real.sqrt 19 / 3 := by
  sorry

end specific_pyramid_height_l3871_387145


namespace smallest_five_digit_multiple_l3871_387139

theorem smallest_five_digit_multiple : ∃ n : ℕ,
  (n ≥ 10000 ∧ n < 100000) ∧  -- 5-digit number
  (15 ∣ n) ∧ (32 ∣ n) ∧ (9 ∣ n) ∧ (5 ∣ n) ∧ (54 ∣ n) ∧
  (∀ m : ℕ, m ≥ 10000 ∧ m < 100000 ∧ 
    (15 ∣ m) ∧ (32 ∣ m) ∧ (9 ∣ m) ∧ (5 ∣ m) ∧ (54 ∣ m) → n ≤ m) ∧
  n = 17280 :=
by sorry

end smallest_five_digit_multiple_l3871_387139


namespace specific_participants_match_probability_l3871_387144

/-- The number of participants in the tournament -/
def n : ℕ := 26

/-- The probability that two specific participants will play against each other -/
def probability : ℚ := 1 / 13

/-- Theorem stating the probability of two specific participants playing against each other -/
theorem specific_participants_match_probability :
  (n - 1 : ℚ) / (n * (n - 1) / 2) = probability := by sorry

end specific_participants_match_probability_l3871_387144


namespace five_card_selection_with_constraints_l3871_387112

/-- The number of cards in a standard deck -/
def standard_deck_size : ℕ := 52

/-- The number of suits in a standard deck -/
def number_of_suits : ℕ := 4

/-- The number of cards in each suit -/
def cards_per_suit : ℕ := 13

/-- The number of cards to be chosen -/
def cards_to_choose : ℕ := 5

/-- The number of cards that must share a suit -/
def cards_sharing_suit : ℕ := 2

/-- 
  The number of ways to choose 5 cards from a standard deck of 52 cards, 
  where exactly two cards share a suit and the remaining three are of different suits.
-/
theorem five_card_selection_with_constraints : 
  (number_of_suits) * 
  (Nat.choose cards_per_suit cards_sharing_suit) * 
  (Nat.choose (number_of_suits - 1) (cards_to_choose - cards_sharing_suit)) * 
  (cards_per_suit ^ (cards_to_choose - cards_sharing_suit)) = 684684 := by
  sorry

end five_card_selection_with_constraints_l3871_387112


namespace magnitude_of_z_l3871_387167

theorem magnitude_of_z (z : ℂ) (h : z^2 = 3 - 4*I) : Complex.abs z = Real.sqrt 5 := by
  sorry

end magnitude_of_z_l3871_387167


namespace upper_limit_of_b_l3871_387150

theorem upper_limit_of_b (a b : ℤ) (h1 : 6 < a) (h2 : a < 17) (h3 : 3 < b) 
  (h4 : (a : ℚ) / b ≤ 3.75) (h5 : 3.75 ≤ (a : ℚ) / b) : b ≤ 4 := by
  sorry

end upper_limit_of_b_l3871_387150


namespace smallest_k_for_inequality_l3871_387134

theorem smallest_k_for_inequality : 
  ∃ (k : ℕ), k > 0 ∧ 
  (∀ (x : ℝ) (n : ℕ), x ∈ Set.Icc 0 1 → n > 0 → x^k * (1-x)^n < 1 / (1+n:ℝ)^3) ∧
  (∀ (k' : ℕ), k' > 0 → k' < k → 
    ∃ (x : ℝ) (n : ℕ), x ∈ Set.Icc 0 1 ∧ n > 0 ∧ x^k' * (1-x)^n ≥ 1 / (1+n:ℝ)^3) ∧
  k = 4 := by
sorry

end smallest_k_for_inequality_l3871_387134


namespace no_negative_roots_l3871_387160

theorem no_negative_roots : ∀ x : ℝ, x < 0 → 4 * x^4 - 7 * x^3 - 20 * x^2 - 13 * x + 25 ≠ 0 := by
  sorry

end no_negative_roots_l3871_387160


namespace salary_decrease_percentage_l3871_387187

/-- Calculates the percentage decrease in salary after an initial increase -/
theorem salary_decrease_percentage 
  (original_salary : ℝ) 
  (initial_increase_percentage : ℝ) 
  (final_salary : ℝ) 
  (h1 : original_salary = 1000.0000000000001)
  (h2 : initial_increase_percentage = 10)
  (h3 : final_salary = 1045) :
  let increased_salary := original_salary * (1 + initial_increase_percentage / 100)
  let decrease_percentage := (1 - final_salary / increased_salary) * 100
  decrease_percentage = 5 := by
sorry

end salary_decrease_percentage_l3871_387187


namespace odd_numbers_equality_l3871_387125

theorem odd_numbers_equality (a b c d k m : ℕ) : 
  Odd a → Odd b → Odd c → Odd d →
  0 < a → a < b → b < c → c < d →
  a * d = b * c →
  a + d = 2^k →
  b + c = 2^m →
  a = 1 := by
sorry

end odd_numbers_equality_l3871_387125


namespace smallest_root_of_g_l3871_387157

-- Define the function g(x)
def g (x : ℝ) : ℝ := 10 * x^4 - 17 * x^2 + 7

-- State the theorem
theorem smallest_root_of_g :
  ∃ (r : ℝ), r = -Real.sqrt (7/5) ∧
  (∀ x : ℝ, g x = 0 → r ≤ x) ∧
  g r = 0 :=
sorry

end smallest_root_of_g_l3871_387157


namespace sum_equals_negative_six_l3871_387113

theorem sum_equals_negative_six (a b c d : ℤ) 
  (h : a + 2 = b + 3 ∧ b + 3 = c + 4 ∧ c + 4 = d + 5 ∧ d + 5 = a + b + c + d + 8) : 
  a + b + c + d = -6 := by
sorry

end sum_equals_negative_six_l3871_387113


namespace probability_of_matching_pair_l3871_387114

def blue_socks : ℕ := 12
def red_socks : ℕ := 10

def total_socks : ℕ := blue_socks + red_socks

def ways_to_pick_two (n : ℕ) : ℕ := n * (n - 1) / 2

def matching_pairs : ℕ := ways_to_pick_two blue_socks + ways_to_pick_two red_socks

def total_combinations : ℕ := ways_to_pick_two total_socks

theorem probability_of_matching_pair :
  (matching_pairs : ℚ) / total_combinations = 111 / 231 := by sorry

end probability_of_matching_pair_l3871_387114


namespace no_positive_integer_solutions_l3871_387151

theorem no_positive_integer_solutions :
  ¬ ∃ (x y z : ℕ+), x^4004 + y^4004 = z^2002 :=
sorry

end no_positive_integer_solutions_l3871_387151


namespace count_adjacent_arrangements_l3871_387180

/-- The number of distinct arrangements of the letters in "КАРАКАТИЦА" where 'Р' and 'Ц' are adjacent -/
def adjacent_arrangements : ℕ := 15120

/-- The word from which we are forming arrangements -/
def word : String := "КАРАКАТИЦА"

/-- The length of the word -/
def word_length : ℕ := word.length

/-- The number of 'А's in the word -/
def count_A : ℕ := (word.toList.filter (· = 'А')).length

/-- The number of 'К's in the word -/
def count_K : ℕ := (word.toList.filter (· = 'К')).length

/-- Theorem stating that the number of distinct arrangements of the letters in "КАРАКАТИЦА" 
    where 'Р' and 'Ц' are adjacent is equal to adjacent_arrangements -/
theorem count_adjacent_arrangements :
  adjacent_arrangements = 
    2 * (Nat.factorial (word_length - 1)) / 
    (Nat.factorial count_A * Nat.factorial count_K) :=
by sorry

end count_adjacent_arrangements_l3871_387180


namespace chess_tournament_games_l3871_387159

/-- The number of games played in a chess tournament. -/
def num_games (n : ℕ) (games_per_pair : ℕ) : ℕ :=
  n * (n - 1) / 2 * games_per_pair

/-- Theorem: In a chess tournament with 25 players, where every player plays
    three times with each of their opponents, the total number of games is 900. -/
theorem chess_tournament_games :
  num_games 25 3 = 900 := by
  sorry

end chess_tournament_games_l3871_387159


namespace part_one_part_two_l3871_387147

-- Define the propositions p and q
def p (a x : ℝ) : Prop := a < x ∧ x < 3 * a

def q (x : ℝ) : Prop := 2 < x ∧ x < 3

-- Part 1
theorem part_one (a x : ℝ) (h1 : a > 0) (h2 : a = 1) (h3 : p a x ∧ q x) :
  2 < x ∧ x < 3 := by sorry

-- Part 2
theorem part_two (a : ℝ) (h1 : a > 0) 
  (h2 : ∀ x, q x → p a x) 
  (h3 : ∃ x, p a x ∧ ¬q x) :
  1 ≤ a ∧ a ≤ 2 := by sorry

end part_one_part_two_l3871_387147


namespace probability_even_sum_l3871_387128

def set_a : Finset ℕ := {11, 44, 55}
def set_b : Finset ℕ := {1}

def is_sum_even (x : ℕ) (y : ℕ) : Bool :=
  (x + y) % 2 = 0

def count_even_sums : ℕ :=
  (set_a.filter (λ x => is_sum_even x 1)).card

theorem probability_even_sum :
  (count_even_sums : ℚ) / (set_a.card : ℚ) = 1 / 3 := by
  sorry

end probability_even_sum_l3871_387128


namespace set_difference_M_N_range_of_a_l3871_387190

-- Define set difference
def set_difference (A B : Set ℝ) : Set ℝ := {x | x ∈ A ∧ x ∉ B}

-- Define sets M and N
def M : Set ℝ := {x | ∃ y, y = Real.sqrt (2*x - 1)}
def N : Set ℝ := {y | ∃ x, y = 1 - x^2}

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | 0 < a*x - 1 ∧ a*x - 1 ≤ 5}
def B : Set ℝ := {y | -1/2 < y ∧ y ≤ 2}

-- Theorem 1
theorem set_difference_M_N : set_difference M N = {x | x > 1} := by sorry

-- Theorem 2
theorem range_of_a (a : ℝ) : set_difference (A a) B = ∅ → a < -12 ∨ a ≥ 3 := by sorry

end set_difference_M_N_range_of_a_l3871_387190


namespace harry_fish_count_harry_fish_count_proof_l3871_387104

/-- Given three friends with fish, prove Harry has 224 fish -/
theorem harry_fish_count : ℕ → ℕ → ℕ → Prop :=
  fun sam_fish joe_fish harry_fish =>
    sam_fish = 7 ∧
    joe_fish = 8 * sam_fish ∧
    harry_fish = 4 * joe_fish →
    harry_fish = 224

/-- Proof of the theorem -/
theorem harry_fish_count_proof : ∃ (sam_fish joe_fish harry_fish : ℕ),
  harry_fish_count sam_fish joe_fish harry_fish :=
by
  sorry

end harry_fish_count_harry_fish_count_proof_l3871_387104


namespace bob_marathon_preparation_l3871_387169

/-- The total miles Bob runs in 3 days -/
def total_miles : ℝ := 70

/-- Miles run on day one -/
def day_one_miles : ℝ := 0.2 * total_miles

/-- Miles run on day two -/
def day_two_miles : ℝ := 0.5 * (total_miles - day_one_miles)

/-- Miles run on day three -/
def day_three_miles : ℝ := 28

theorem bob_marathon_preparation :
  day_one_miles + day_two_miles + day_three_miles = total_miles :=
by sorry

end bob_marathon_preparation_l3871_387169


namespace complex_square_second_quadrant_l3871_387152

/-- Given that (1+2i)^2 = a+bi where a and b are real numbers,
    prove that the point P(a,b) lies in the second quadrant. -/
theorem complex_square_second_quadrant :
  ∃ (a b : ℝ), (1 + 2 * Complex.I) ^ 2 = a + b * Complex.I ∧
  a < 0 ∧ b > 0 := by
  sorry

end complex_square_second_quadrant_l3871_387152


namespace max_p_value_l3871_387154

theorem max_p_value (p q r : ℝ) (sum_eq : p + q + r = 10) (prod_sum_eq : p*q + p*r + q*r = 25) :
  p ≤ 20/3 ∧ ∃ q r : ℝ, p = 20/3 ∧ p + q + r = 10 ∧ p*q + p*r + q*r = 25 := by
  sorry

end max_p_value_l3871_387154


namespace pyramid_volume_integer_heights_l3871_387158

theorem pyramid_volume_integer_heights (base_side : ℕ) (height : ℕ) :
  base_side = 640 →
  height = 1024 →
  (∃ (n : ℕ), n = 85 ∧
    (∀ h : ℕ, h < height →
      (25 * (height - h)^3) % 192 = 0 ↔ h ∈ Finset.range (n + 1))) :=
by sorry

end pyramid_volume_integer_heights_l3871_387158


namespace polynomials_common_factor_l3871_387193

def p1 (x : ℝ) : ℝ := 16 * x^5 - x
def p2 (x : ℝ) : ℝ := (x - 1)^2 - 4 * (x - 1) + 4
def p3 (x : ℝ) : ℝ := (x + 1)^2 - 4 * x * (x + 1) + 4 * x^2
def p4 (x : ℝ) : ℝ := -4 * x^2 - 1 + 4 * x

theorem polynomials_common_factor :
  ∃ (f : ℝ → ℝ) (g1 g4 : ℝ → ℝ),
    (∀ x, p1 x = f x * g1 x) ∧
    (∀ x, p4 x = f x * g4 x) ∧
    (∀ x, f x ≠ 0) ∧
    (∀ x, f x ≠ 1) ∧
    (∀ x, f x ≠ -1) ∧
    (∀ (h2 h3 : ℝ → ℝ),
      (∀ x, p2 x ≠ f x * h2 x) ∧
      (∀ x, p3 x ≠ f x * h3 x)) :=
by sorry

end polynomials_common_factor_l3871_387193


namespace smallest_n_congruence_l3871_387168

theorem smallest_n_congruence (n : ℕ+) : 
  (∀ m : ℕ+, m < n → ¬(5 * m : ℤ) ≡ 1978 [ZMOD 26]) ∧ 
  (5 * n : ℤ) ≡ 1978 [ZMOD 26] ↔ 
  n = 16 := by sorry

end smallest_n_congruence_l3871_387168


namespace x_fourth_coefficient_equals_a_9_l3871_387103

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The sequence a_n = 2n + 2 -/
def a (n : ℕ) : ℕ := 2 * n + 2

/-- The theorem to prove -/
theorem x_fourth_coefficient_equals_a_9 :
  binomial 5 4 + binomial 6 4 = a 9 := by sorry

end x_fourth_coefficient_equals_a_9_l3871_387103


namespace water_amount_in_new_recipe_l3871_387131

/-- Represents the ratio of ingredients in a recipe --/
structure RecipeRatio :=
  (flour : ℚ)
  (water : ℚ)
  (sugar : ℚ)

/-- The original recipe ratio --/
def original_ratio : RecipeRatio :=
  { flour := 11, water := 8, sugar := 1 }

/-- The new recipe ratio --/
def new_ratio : RecipeRatio :=
  { flour := 22, water := 8, sugar := 4 }

/-- The amount of sugar in the new recipe --/
def new_sugar_amount : ℚ := 2

theorem water_amount_in_new_recipe :
  let water_amount := (new_ratio.water / new_ratio.sugar) * new_sugar_amount
  water_amount = 4 := by
  sorry

#check water_amount_in_new_recipe

end water_amount_in_new_recipe_l3871_387131


namespace indicator_light_signals_l3871_387107

/-- The number of indicator lights in a row -/
def num_lights : ℕ := 8

/-- The number of lights displayed at a time -/
def lights_displayed : ℕ := 4

/-- The number of adjacent lights among those displayed -/
def adjacent_lights : ℕ := 3

/-- The number of colors each light can display -/
def colors_per_light : ℕ := 2

/-- The total number of different signals that can be displayed -/
def total_signals : ℕ := 320

theorem indicator_light_signals :
  (num_lights = 8) →
  (lights_displayed = 4) →
  (adjacent_lights = 3) →
  (colors_per_light = 2) →
  total_signals = 320 := by sorry

end indicator_light_signals_l3871_387107


namespace target_probability_value_l3871_387138

/-- The probability of hitting the target on a single shot -/
def hit_probability : ℝ := 0.85

/-- The probability of missing the target on a single shot -/
def miss_probability : ℝ := 1 - hit_probability

/-- The probability of missing the first two shots and hitting the third shot -/
def target_probability : ℝ := miss_probability * miss_probability * hit_probability

theorem target_probability_value : target_probability = 0.019125 := by
  sorry

end target_probability_value_l3871_387138


namespace polynomial_inequality_l3871_387177

theorem polynomial_inequality (a b c : ℝ) 
  (h : ∀ x : ℝ, |x| ≤ 1 → |a * x^2 + b * x + c| ≤ 1) :
  ∀ x : ℝ, |x| ≤ 1 → |c * x^2 + b * x + a| ≤ 2 := by
  sorry

end polynomial_inequality_l3871_387177


namespace initial_bench_press_weight_l3871_387186

/-- The initial bench press weight before injury -/
def W : ℝ := 500

/-- The bench press weight after injury -/
def after_injury : ℝ := 0.2 * W

/-- The bench press weight after training -/
def after_training : ℝ := 3 * after_injury

/-- The final bench press weight -/
def final_weight : ℝ := 300

theorem initial_bench_press_weight :
  W = 500 ∧ after_injury = 0.2 * W ∧ after_training = 3 * after_injury ∧ after_training = final_weight := by
  sorry

end initial_bench_press_weight_l3871_387186


namespace pizza_toppings_l3871_387132

theorem pizza_toppings (total_slices : ℕ) (pepperoni_slices : ℕ) (mushroom_slices : ℕ)
  (h_total : total_slices = 16)
  (h_pepperoni : pepperoni_slices = 8)
  (h_mushroom : mushroom_slices = 14)
  (h_at_least_one : ∀ slice, slice ≤ total_slices → (slice ≤ pepperoni_slices ∨ slice ≤ mushroom_slices)) :
  ∃ both_toppings : ℕ, 
    both_toppings = pepperoni_slices + mushroom_slices - total_slices ∧
    both_toppings = 6 :=
by sorry

end pizza_toppings_l3871_387132


namespace flour_info_doesnt_determine_sugar_l3871_387195

/-- Represents a cake recipe --/
structure Recipe where
  flour : ℕ
  sugar : ℕ

/-- Represents the state of Jessica's baking process --/
structure BakingProcess where
  flour_added : ℕ
  flour_needed : ℕ

/-- Given information about flour doesn't determine sugar amount --/
theorem flour_info_doesnt_determine_sugar 
  (recipe : Recipe) 
  (baking : BakingProcess) 
  (h1 : recipe.flour = 8)
  (h2 : baking.flour_added = 4)
  (h3 : baking.flour_needed = 4)
  (h4 : baking.flour_added + baking.flour_needed = recipe.flour) :
  ∃ (r1 r2 : Recipe), r1.flour = r2.flour ∧ r1.sugar ≠ r2.sugar :=
sorry

end flour_info_doesnt_determine_sugar_l3871_387195


namespace odd_function_sum_l3871_387111

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def periodic_4 (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 4) = f x

theorem odd_function_sum (f : ℝ → ℝ) 
  (h_odd : is_odd_function f)
  (h_periodic : ∀ x, f x + f (4 - x) = 0)
  (h_f_1 : f 1 = 8) :
  f 2010 + f 2011 + f 2012 = -8 := by
sorry

end odd_function_sum_l3871_387111


namespace solve_exponential_equation_l3871_387146

theorem solve_exponential_equation :
  ∃ x : ℝ, (8 : ℝ) ^ (4 * x - 6) = (1 / 2 : ℝ) ^ (x + 5) ∧ x = 1 :=
by sorry

end solve_exponential_equation_l3871_387146


namespace a_2n_is_perfect_square_l3871_387101

/-- Definition of a_n: number of natural numbers with digit sum n and digits in {1,3,4} -/
def a (n : ℕ) : ℕ := sorry

/-- Theorem: a_{2n} is a perfect square for all natural numbers n -/
theorem a_2n_is_perfect_square (n : ℕ) : ∃ k : ℕ, a (2 * n) = k ^ 2 := by sorry

end a_2n_is_perfect_square_l3871_387101


namespace arrangement_count_is_150_l3871_387124

/-- The number of ways to arrange volunteers among events --/
def arrange_volunteers (n : ℕ) (k : ℕ) : ℕ :=
  k^n - k * (k-1)^n + (k * (k-1) / 2) * (k-2)^n

/-- The number of arrangements for 5 volunteers and 3 events --/
def arrangement_count : ℕ := arrange_volunteers 5 3

/-- Theorem: The number of arrangements for 5 volunteers and 3 events,
    such that each event has at least one participant, is 150 --/
theorem arrangement_count_is_150 : arrangement_count = 150 := by
  sorry

end arrangement_count_is_150_l3871_387124


namespace correct_arrangement_count_l3871_387198

def arrangement_count : ℕ := 
  (Finset.range 3).sum (λ i =>
    Nat.choose 4 i * Nat.choose 6 (i + 2) * Nat.choose 5 i)

theorem correct_arrangement_count : arrangement_count = 1315 := by
  sorry

end correct_arrangement_count_l3871_387198


namespace minimum_nickels_needed_l3871_387155

/-- The price of the book in cents -/
def book_price : ℕ := 4250

/-- The number of $10 bills Jane has -/
def ten_dollar_bills : ℕ := 4

/-- The number of quarters Jane has -/
def quarters : ℕ := 5

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The minimum number of nickels Jane needs to afford the book -/
def min_nickels : ℕ := 25

theorem minimum_nickels_needed :
  ∀ n : ℕ,
  (ten_dollar_bills * 1000 + quarters * 25 + n * nickel_value ≥ book_price) →
  (n ≥ min_nickels) :=
sorry

end minimum_nickels_needed_l3871_387155


namespace basketball_team_sales_l3871_387143

/-- The number of cupcakes sold -/
def cupcakes : ℕ := 50

/-- The price of each cupcake in dollars -/
def cupcake_price : ℚ := 2

/-- The price of each cookie in dollars -/
def cookie_price : ℚ := 1/2

/-- The number of basketballs bought -/
def basketballs : ℕ := 2

/-- The price of each basketball in dollars -/
def basketball_price : ℚ := 40

/-- The number of energy drinks bought -/
def energy_drinks : ℕ := 20

/-- The price of each energy drink in dollars -/
def energy_drink_price : ℚ := 2

/-- The number of cookies sold -/
def cookies_sold : ℕ := 40

theorem basketball_team_sales :
  cookies_sold * cookie_price = 
    basketballs * basketball_price + 
    energy_drinks * energy_drink_price - 
    cupcakes * cupcake_price :=
by sorry

end basketball_team_sales_l3871_387143


namespace tiling_8x2_equals_fib_9_l3871_387110

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Number of ways to tile a 2 × n rectangle with 1 × 2 dominoes -/
def tiling_ways (n : ℕ) : ℕ := fib (n + 1)

/-- Theorem: The number of ways to tile an 8 × 2 rectangle with 1 × 2 dominoes
    is equal to the 9th Fibonacci number -/
theorem tiling_8x2_equals_fib_9 :
  tiling_ways 8 = fib 9 := by
  sorry

#eval tiling_ways 8  -- Expected output: 34

end tiling_8x2_equals_fib_9_l3871_387110


namespace increase_data_effect_l3871_387191

/-- Represents a data set with its average and variance -/
structure DataSet where
  average : ℝ
  variance : ℝ

/-- Represents the operation of increasing each data point by a fixed value -/
def increase_data (d : DataSet) (inc : ℝ) : DataSet :=
  { average := d.average + inc, variance := d.variance }

/-- Theorem stating the effect of increasing each data point on the average and variance -/
theorem increase_data_effect (d : DataSet) (inc : ℝ) :
  d.average = 2 ∧ d.variance = 3 ∧ inc = 60 →
  (increase_data d inc).average = 62 ∧ (increase_data d inc).variance = 3 := by
  sorry

end increase_data_effect_l3871_387191


namespace negation_equivalence_l3871_387100

-- Define the original proposition
def original_proposition (a : ℝ) : Prop :=
  ∀ x : ℝ, x > 0 → x^2 + a*x + 3 ≥ 0

-- Define the negation of the proposition
def negation_proposition (a : ℝ) : Prop :=
  ∃ x : ℝ, x > 0 ∧ x^2 + a*x + 3 < 0

-- Theorem stating that the negation is correct
theorem negation_equivalence (a : ℝ) :
  ¬(original_proposition a) ↔ negation_proposition a :=
by sorry

end negation_equivalence_l3871_387100


namespace product_equals_two_l3871_387142

theorem product_equals_two : 10 * (1/5) * 4 * (1/16) * (1/2) * 8 = 2 := by
  sorry

end product_equals_two_l3871_387142


namespace car_distance_traveled_l3871_387188

/-- Calculates the distance traveled by a car given its speed and time -/
def distanceTraveled (speed : ℚ) (time : ℚ) : ℚ :=
  speed * time

/-- The actual speed of the car in km/h -/
def actualSpeed : ℚ := 35

/-- The fraction of the actual speed at which the car is traveling -/
def speedFraction : ℚ := 5 / 7

/-- The time the car travels in hours -/
def travelTime : ℚ := 126 / 75

/-- The theorem stating the distance traveled by the car -/
theorem car_distance_traveled :
  distanceTraveled (speedFraction * actualSpeed) travelTime = 42 := by
  sorry

end car_distance_traveled_l3871_387188


namespace median_in_group_two_l3871_387115

/-- Represents the labor time groups --/
inductive LaborGroup
  | One
  | Two
  | Three
  | Four

/-- The frequency of each labor group --/
def frequency : LaborGroup → Nat
  | LaborGroup.One => 10
  | LaborGroup.Two => 20
  | LaborGroup.Three => 12
  | LaborGroup.Four => 8

/-- The total number of surveyed students --/
def totalStudents : Nat := 50

/-- The cumulative frequency up to and including a given group --/
def cumulativeFrequency (g : LaborGroup) : Nat :=
  match g with
  | LaborGroup.One => frequency LaborGroup.One
  | LaborGroup.Two => frequency LaborGroup.One + frequency LaborGroup.Two
  | LaborGroup.Three => frequency LaborGroup.One + frequency LaborGroup.Two + frequency LaborGroup.Three
  | LaborGroup.Four => totalStudents

/-- The median position --/
def medianPosition : Nat := totalStudents / 2

theorem median_in_group_two :
  cumulativeFrequency LaborGroup.One < medianPosition ∧
  medianPosition ≤ cumulativeFrequency LaborGroup.Two :=
sorry

end median_in_group_two_l3871_387115


namespace quadratic_root_l3871_387172

/- Given a quadratic equation x^2 - (m+n)x + mn - p = 0 with roots α and β -/
theorem quadratic_root (m n p : ℤ) (α β : ℝ) 
  (h_distinct : m ≠ n ∧ n ≠ p ∧ m ≠ p)
  (h_roots : ∀ x : ℝ, x^2 - (m+n)*x + mn - p = 0 ↔ x = α ∨ x = β)
  (h_alpha : α = 3) :
  β = m + n - 3 := by
sorry

end quadratic_root_l3871_387172


namespace pure_imaginary_sixth_power_l3871_387119

theorem pure_imaginary_sixth_power (a : ℝ) (z : ℂ) :
  z = a + (a + 1) * Complex.I →
  z.im ≠ 0 →
  z.re = 0 →
  z^6 = -1 := by sorry

end pure_imaginary_sixth_power_l3871_387119


namespace prob_two_defective_consignment_l3871_387121

/-- Represents a consignment of picture tubes -/
structure Consignment where
  total : ℕ
  defective : ℕ
  h_defective_le_total : defective ≤ total

/-- Calculates the probability of selecting two defective tubes without replacement -/
def prob_two_defective (c : Consignment) : ℚ :=
  (c.defective : ℚ) / (c.total : ℚ) * ((c.defective - 1) : ℚ) / ((c.total - 1) : ℚ)

theorem prob_two_defective_consignment :
  let c : Consignment := ⟨20, 5, by norm_num⟩
  prob_two_defective c = 1 / 19 := by sorry

end prob_two_defective_consignment_l3871_387121


namespace intersection_M_N_l3871_387197

-- Define the sets M and N
def M : Set ℝ := {s | |s| < 4}
def N : Set ℝ := {x | 3 * x ≥ -1}

-- State the theorem
theorem intersection_M_N : M ∩ N = {x | -1/3 ≤ x ∧ x < 4} := by
  sorry

end intersection_M_N_l3871_387197


namespace exists_number_with_specific_digit_sum_l3871_387149

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Theorem stating the existence of a number with specific digit sum properties -/
theorem exists_number_with_specific_digit_sum : 
  ∃ m : ℕ, sumOfDigits m = 1990 ∧ sumOfDigits (m^2) = 1990^2 := by sorry

end exists_number_with_specific_digit_sum_l3871_387149


namespace power_of_64_l3871_387126

theorem power_of_64 : (64 : ℝ) ^ (5/6) = 32 := by
  have h1 : (64 : ℝ) = 2^6 := by sorry
  sorry

end power_of_64_l3871_387126


namespace quadratic_inequality_solution_l3871_387133

theorem quadratic_inequality_solution (x : ℝ) : x^2 - 8*x + 12 < 0 ↔ 2 < x ∧ x < 6 := by
  sorry

end quadratic_inequality_solution_l3871_387133


namespace definite_integral_result_l3871_387183

theorem definite_integral_result : 
  ∫ x in -Real.arcsin (2 / Real.sqrt 5)..π/4, (2 - Real.tan x) / (Real.sin x + 3 * Real.cos x)^2 = 15/4 - Real.log 4 := by
  sorry

end definite_integral_result_l3871_387183


namespace amulet_price_is_40_l3871_387117

/-- Calculates the selling price of amulets given the following conditions:
  * Dirk sells amulets for 2 days
  * Each day he sells 25 amulets
  * Each amulet costs him $30 to make
  * He gives 10% of his revenue to the faire
  * He made a profit of $300
-/
def amulet_price (days : ℕ) (amulets_per_day : ℕ) (cost_per_amulet : ℕ) 
                 (faire_percentage : ℚ) (profit : ℕ) : ℚ :=
  let total_amulets := days * amulets_per_day
  let total_cost := total_amulets * cost_per_amulet
  let x := (profit + total_cost) / (total_amulets * (1 - faire_percentage))
  x

theorem amulet_price_is_40 :
  amulet_price 2 25 30 (1/10) 300 = 40 := by
  sorry

end amulet_price_is_40_l3871_387117


namespace matrix_equation_solution_l3871_387118

-- Define the matrix evaluation rule
def matrix_value (a b c d : ℝ) : ℝ := a * b - c * d

-- Define our specific matrix as a function of x
def M (x : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![3*x+1, 2; 2*x, x+1]

-- State the theorem
theorem matrix_equation_solution :
  ∀ x : ℝ, matrix_value (M x 0 0) (M x 1 1) (M x 0 1) (M x 1 0) = 5 ↔ 
  x = 2 * Real.sqrt 3 / 3 ∨ x = -2 * Real.sqrt 3 / 3 := by
  sorry

end matrix_equation_solution_l3871_387118


namespace no_integer_solution_l3871_387189

theorem no_integer_solution : ¬ ∃ (x y : ℤ), 2 * x + 6 * y = 91 := by
  sorry

end no_integer_solution_l3871_387189


namespace unique_four_digit_palindromic_square_l3871_387127

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

theorem unique_four_digit_palindromic_square : 
  ∃! n : ℕ, is_four_digit n ∧ is_perfect_square n ∧ is_palindrome n :=
sorry

end unique_four_digit_palindromic_square_l3871_387127


namespace completing_square_quadratic_l3871_387192

theorem completing_square_quadratic (x : ℝ) : 
  (x^2 - 6*x + 8 = 0) ↔ ((x - 3)^2 = 1) := by
  sorry

end completing_square_quadratic_l3871_387192


namespace negation_equivalence_l3871_387196

theorem negation_equivalence :
  (¬ ∀ x : ℝ, x^2 - 2*x - 3 ≤ 0) ↔ (∃ x : ℝ, x^2 - 2*x - 3 > 0) :=
by sorry

end negation_equivalence_l3871_387196


namespace tangent_square_area_l3871_387153

/-- A square with two vertices on a circle and two on its tangent -/
structure TangentSquare where
  /-- The radius of the circle -/
  R : ℝ
  /-- The side length of the square -/
  x : ℝ
  /-- Two vertices of the square lie on the circle -/
  vertices_on_circle : x ≤ 2 * R
  /-- Two vertices of the square lie on the tangent -/
  vertices_on_tangent : x^2 / 4 = R^2 - (x - R)^2

/-- The area of a TangentSquare with radius 5 is 64 -/
theorem tangent_square_area (s : TangentSquare) (h : s.R = 5) : s.x^2 = 64 := by
  sorry

end tangent_square_area_l3871_387153


namespace solution_set_f_geq_4_range_of_a_l3871_387166

-- Define the function f
def f (x : ℝ) : ℝ := |1 - 2*x| - |1 + x|

-- Theorem for the solution set of f(x) ≥ 4
theorem solution_set_f_geq_4 :
  {x : ℝ | f x ≥ 4} = {x : ℝ | x ≤ -2 ∨ x ≥ 6} := by sorry

-- Theorem for the range of a
theorem range_of_a :
  {a : ℝ | ∀ x, a^2 + 2*a + |1 + x| > f x} = {a : ℝ | a < -3 ∨ a > 1} := by sorry

end solution_set_f_geq_4_range_of_a_l3871_387166


namespace polynomial_divisibility_condition_l3871_387185

/-- A polynomial with integer coefficients -/
def IntPolynomial := Polynomial ℤ

/-- Definition of divisibility for integers -/
def divides (a b : ℤ) : Prop := ∃ k : ℤ, b = a * k

/-- Definition of an odd prime number -/
def is_odd_prime (p : ℕ) : Prop := Nat.Prime p ∧ p % 2 = 1

/-- The main theorem -/
theorem polynomial_divisibility_condition (f : IntPolynomial) :
  (∀ p : ℕ, is_odd_prime p → divides (f.eval p) ((p - 3).factorial + (p + 1) / 2)) →
  (f = Polynomial.X) ∨ (f = -Polynomial.X) ∨ (f = Polynomial.C 1) := by
  sorry

end polynomial_divisibility_condition_l3871_387185

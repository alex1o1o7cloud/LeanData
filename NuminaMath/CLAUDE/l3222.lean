import Mathlib

namespace NUMINAMATH_CALUDE_waste_after_ten_years_l3222_322206

/-- Calculates the amount of waste after n years given an initial amount and growth rate -/
def wasteAmount (a : ℝ) (b : ℝ) (n : ℕ) : ℝ :=
  a * (1 + b) ^ n

/-- Theorem: The amount of waste after 10 years is a(1+b)^10 -/
theorem waste_after_ten_years (a b : ℝ) :
  wasteAmount a b 10 = a * (1 + b) ^ 10 := by
  sorry

#check waste_after_ten_years

end NUMINAMATH_CALUDE_waste_after_ten_years_l3222_322206


namespace NUMINAMATH_CALUDE_queen_diamond_probability_l3222_322293

/-- Represents a standard deck of 52 playing cards -/
structure Deck :=
  (cards : Finset (Nat × Nat))
  (size : cards.card = 52)
  (valid : ∀ c ∈ cards, c.1 ∈ Finset.range 13 ∧ c.2 ∈ Finset.range 4)

/-- Represents the event of drawing a Queen as the first card -/
def isFirstCardQueen (d : Deck) : Finset (Nat × Nat) :=
  d.cards.filter (λ c => c.1 = 11)

/-- Represents the event of drawing a diamond as the second card -/
def isSecondCardDiamond (d : Deck) : Finset (Nat × Nat) :=
  d.cards.filter (λ c => c.2 = 1)

/-- The main theorem stating the probability of drawing a Queen first and a diamond second -/
theorem queen_diamond_probability (d : Deck) :
  (isFirstCardQueen d).card / d.cards.card *
  (isSecondCardDiamond d).card / (d.cards.card - 1) = 18 / 221 :=
sorry

end NUMINAMATH_CALUDE_queen_diamond_probability_l3222_322293


namespace NUMINAMATH_CALUDE_tic_tac_toe_rounds_l3222_322256

theorem tic_tac_toe_rounds (total_rounds : ℕ) (difference : ℕ) (william_wins harry_wins : ℕ) : 
  total_rounds = 15 → 
  difference = 5 → 
  william_wins = harry_wins + difference → 
  william_wins + harry_wins = total_rounds → 
  william_wins = 10 := by
sorry

end NUMINAMATH_CALUDE_tic_tac_toe_rounds_l3222_322256


namespace NUMINAMATH_CALUDE_divisibility_condition_l3222_322221

theorem divisibility_condition (a b : ℕ) (ha : a ≥ 3) (hb : b ≥ 3) :
  (a * b^2 + b + 7 ∣ a^2 * b + a + b) →
  ∃ k : ℕ, k ≥ 1 ∧ a = 7 * k^2 ∧ b = 7 * k :=
by sorry

end NUMINAMATH_CALUDE_divisibility_condition_l3222_322221


namespace NUMINAMATH_CALUDE_worker_task_completion_time_l3222_322243

/-- Given two workers who can complete a task together in 35 days,
    and one of them can complete the task alone in 84 days,
    prove that the other worker can complete the task alone in 70 days. -/
theorem worker_task_completion_time 
  (total_time : ℝ) 
  (worker1_time : ℝ) 
  (worker2_time : ℝ) 
  (h1 : total_time = 35) 
  (h2 : worker1_time = 84) 
  (h3 : 1 / total_time = 1 / worker1_time + 1 / worker2_time) : 
  worker2_time = 70 := by
  sorry

end NUMINAMATH_CALUDE_worker_task_completion_time_l3222_322243


namespace NUMINAMATH_CALUDE_total_winter_clothing_l3222_322215

/-- The number of boxes containing winter clothing -/
def num_boxes : ℕ := 3

/-- The number of scarves in each box -/
def scarves_per_box : ℕ := 3

/-- The number of mittens in each box -/
def mittens_per_box : ℕ := 4

/-- Theorem: The total number of pieces of winter clothing is 21 -/
theorem total_winter_clothing : 
  num_boxes * (scarves_per_box + mittens_per_box) = 21 := by
  sorry

end NUMINAMATH_CALUDE_total_winter_clothing_l3222_322215


namespace NUMINAMATH_CALUDE_cafe_meal_cost_l3222_322238

theorem cafe_meal_cost (s c k : ℝ) : 
  (2 * s + 5 * c + 2 * k = 6.50) → 
  (3 * s + 8 * c + 3 * k = 10.20) → 
  (s + c + k = 1.90) :=
by
  sorry

end NUMINAMATH_CALUDE_cafe_meal_cost_l3222_322238


namespace NUMINAMATH_CALUDE_prob_zero_or_one_white_is_four_fifths_l3222_322242

def total_balls : ℕ := 6
def red_balls : ℕ := 4
def white_balls : ℕ := 2
def selected_balls : ℕ := 3

def prob_zero_or_one_white (total : ℕ) (red : ℕ) (white : ℕ) (selected : ℕ) : ℚ :=
  (Nat.choose red selected + Nat.choose white 1 * Nat.choose red (selected - 1)) /
  Nat.choose total selected

theorem prob_zero_or_one_white_is_four_fifths :
  prob_zero_or_one_white total_balls red_balls white_balls selected_balls = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_prob_zero_or_one_white_is_four_fifths_l3222_322242


namespace NUMINAMATH_CALUDE_inverse_proportion_k_negative_l3222_322201

theorem inverse_proportion_k_negative
  (k : ℝ) (y₁ y₂ : ℝ)
  (h1 : k ≠ 0)
  (h2 : y₁ = k / (-2))
  (h3 : y₂ = k / 5)
  (h4 : y₁ > y₂) :
  k < 0 := by
sorry

end NUMINAMATH_CALUDE_inverse_proportion_k_negative_l3222_322201


namespace NUMINAMATH_CALUDE_intersection_equality_implies_x_values_l3222_322264

theorem intersection_equality_implies_x_values (x : ℝ) : 
  let A : Set ℝ := {1, 4, x}
  let B : Set ℝ := {1, x^2}
  (A ∩ B = B) → (x = -2 ∨ x = 2 ∨ x = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_equality_implies_x_values_l3222_322264


namespace NUMINAMATH_CALUDE_first_problem_answer_l3222_322268

/-- Given three math problems with the following properties:
    1. The second problem's answer is twice the first problem's answer.
    2. The third problem's answer is 400 less than the sum of the first two problems' answers.
    3. The total of all three answers is 3200.
    Prove that the answer to the first math problem is 600. -/
theorem first_problem_answer (a : ℕ) : 
  (∃ b c : ℕ, 
    b = 2 * a ∧ 
    c = a + b - 400 ∧ 
    a + b + c = 3200) → 
  a = 600 :=
by sorry

end NUMINAMATH_CALUDE_first_problem_answer_l3222_322268


namespace NUMINAMATH_CALUDE_weight_difference_l3222_322234

theorem weight_difference (rachel jimmy adam : ℝ) 
  (h1 : rachel = 75)
  (h2 : rachel < jimmy)
  (h3 : rachel = adam + 15)
  (h4 : (rachel + jimmy + adam) / 3 = 72) :
  jimmy - rachel = 6 := by
sorry

end NUMINAMATH_CALUDE_weight_difference_l3222_322234


namespace NUMINAMATH_CALUDE_james_brothers_count_l3222_322295

def market_value : ℝ := 500000
def selling_price : ℝ := market_value * 1.2
def revenue_after_taxes : ℝ := selling_price * 0.9
def share_per_person : ℝ := 135000

theorem james_brothers_count :
  ∃ (n : ℕ), (revenue_after_taxes / (n + 1 : ℝ) = share_per_person) ∧ n = 3 :=
by sorry

end NUMINAMATH_CALUDE_james_brothers_count_l3222_322295


namespace NUMINAMATH_CALUDE_cross_section_area_formula_l3222_322279

/-- Regular tetrahedron with edge length a -/
structure RegularTetrahedron (a : ℝ) :=
  (edge_length : a > 0)

/-- Plane passing through the midpoint of an edge and perpendicular to an adjacent edge -/
structure CrossSectionPlane (t : RegularTetrahedron a) :=
  (passes_through_midpoint : Bool)
  (perpendicular_to_adjacent : Bool)

/-- The area of the cross-section formed by the plane -/
def cross_section_area (t : RegularTetrahedron a) (p : CrossSectionPlane t) : ℝ :=
  sorry

/-- Theorem stating the area of the cross-section -/
theorem cross_section_area_formula (a : ℝ) (t : RegularTetrahedron a) (p : CrossSectionPlane t) :
  p.passes_through_midpoint ∧ p.perpendicular_to_adjacent →
  cross_section_area t p = (a^2 * Real.sqrt 2) / 16 :=
sorry

end NUMINAMATH_CALUDE_cross_section_area_formula_l3222_322279


namespace NUMINAMATH_CALUDE_line_tangent_to_circle_l3222_322252

/-- The line x + y = 2k is tangent to the circle x^2 + y^2 = 4k if and only if k = 2 -/
theorem line_tangent_to_circle (k : ℝ) : 
  (∀ x y : ℝ, x + y = 2 * k → x^2 + y^2 = 4 * k) ↔ k = 2 := by
  sorry

end NUMINAMATH_CALUDE_line_tangent_to_circle_l3222_322252


namespace NUMINAMATH_CALUDE_tan_2y_value_l3222_322205

theorem tan_2y_value (x y : ℝ) 
  (h : Real.sin (x - y) * Real.cos x - Real.cos (x - y) * Real.sin x = 3/5) : 
  Real.tan (2 * y) = 24/7 ∨ Real.tan (2 * y) = -24/7 := by
  sorry

end NUMINAMATH_CALUDE_tan_2y_value_l3222_322205


namespace NUMINAMATH_CALUDE_like_terms_imply_value_l3222_322246

theorem like_terms_imply_value (a b : ℤ) : 
  (1 = a - 1) → (b + 1 = 4) → (a - b)^2023 = -1 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_imply_value_l3222_322246


namespace NUMINAMATH_CALUDE_absolute_value_ratio_l3222_322273

theorem absolute_value_ratio (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a^2 + b^2 = 18*a*b) :
  |((a+b)/(a-b))| = Real.sqrt 5 / 2 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_ratio_l3222_322273


namespace NUMINAMATH_CALUDE_xiaoyue_speed_l3222_322213

/-- Prove that Xiaoyue's average speed is 50 km/h given the conditions of the problem -/
theorem xiaoyue_speed (x : ℝ) 
  (h1 : x > 0)  -- Xiaoyue's speed is positive
  (h2 : 20 / x - 18 / (1.2 * x) = 1 / 10) : x = 50 := by
  sorry

#check xiaoyue_speed

end NUMINAMATH_CALUDE_xiaoyue_speed_l3222_322213


namespace NUMINAMATH_CALUDE_expression_value_l3222_322207

theorem expression_value : 
  let a := 2021
  (a^3 - 3*a^2*(a+1) + 4*a*(a+1)^2 - (a+1)^3 + 2) / (a*(a+1)) = 1 + 1/a := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3222_322207


namespace NUMINAMATH_CALUDE_multiple_choice_questions_l3222_322270

theorem multiple_choice_questions (total : ℕ) (problem_solving_percent : ℚ) 
  (h1 : total = 50)
  (h2 : problem_solving_percent = 80 / 100) :
  (total : ℚ) * (1 - problem_solving_percent) = 10 := by
  sorry

end NUMINAMATH_CALUDE_multiple_choice_questions_l3222_322270


namespace NUMINAMATH_CALUDE_residue_14_power_2046_mod_17_l3222_322266

theorem residue_14_power_2046_mod_17 : 14^2046 % 17 = 12 := by
  sorry

end NUMINAMATH_CALUDE_residue_14_power_2046_mod_17_l3222_322266


namespace NUMINAMATH_CALUDE_ellipse_equation_l3222_322283

/-- An ellipse with one focus at (1, 0) and eccentricity √2/2 has the equation x^2/2 + y^2 = 1 -/
theorem ellipse_equation (e : ℝ × ℝ → Prop) :
  (∃ (a b c : ℝ), 
    -- One focus is at (1, 0)
    c = 1 ∧
    -- Eccentricity is √2/2
    c / a = Real.sqrt 2 / 2 ∧
    -- Standard form of ellipse equation
    b^2 = a^2 - c^2 ∧
    (∀ (x y : ℝ), e (x, y) ↔ x^2 / a^2 + y^2 / b^2 = 1)) →
  (∀ (x y : ℝ), e (x, y) ↔ x^2 / 2 + y^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l3222_322283


namespace NUMINAMATH_CALUDE_union_of_sets_l3222_322282

theorem union_of_sets : 
  let A : Set ℕ := {1, 3}
  let B : Set ℕ := {1, 2, 4, 5}
  A ∪ B = {1, 2, 3, 4, 5} := by
sorry

end NUMINAMATH_CALUDE_union_of_sets_l3222_322282


namespace NUMINAMATH_CALUDE_remainder_eight_n_mod_seven_l3222_322239

theorem remainder_eight_n_mod_seven (n : ℤ) (h : n % 4 = 3) : (8 * n) % 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_eight_n_mod_seven_l3222_322239


namespace NUMINAMATH_CALUDE_two_digit_number_property_l3222_322204

theorem two_digit_number_property : ∃! n : ℕ, 
  10 ≤ n ∧ n < 100 ∧ 
  n = 3 * ((n / 10) + (n % 10)) :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_two_digit_number_property_l3222_322204


namespace NUMINAMATH_CALUDE_cost_per_patch_is_correct_l3222_322250

/-- The cost per patch for Sean's business -/
def cost_per_patch : ℝ := 1.25

/-- The order quantity of patches -/
def order_quantity : ℕ := 100

/-- The selling price per patch -/
def selling_price : ℝ := 12

/-- The net profit for selling one order -/
def net_profit : ℝ := 1075

/-- Theorem: The cost per patch is $1.25 given the order quantity, selling price, and net profit -/
theorem cost_per_patch_is_correct : 
  selling_price * order_quantity - cost_per_patch * order_quantity = net_profit :=
by sorry

end NUMINAMATH_CALUDE_cost_per_patch_is_correct_l3222_322250


namespace NUMINAMATH_CALUDE_vector_sum_parallel_l3222_322245

theorem vector_sum_parallel (y : ℝ) : 
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![2, y]
  (∃ (k : ℝ), a = k • b) →
  (a + 2 • b) = ![5, 10] := by
sorry

end NUMINAMATH_CALUDE_vector_sum_parallel_l3222_322245


namespace NUMINAMATH_CALUDE_inequality_solution_l3222_322269

open Real

theorem inequality_solution (x y : ℝ) : 
  (Real.sqrt 3 * Real.tan x - (Real.sin y) ^ (1/4) - 
   Real.sqrt ((3 / (Real.cos x)^2) + (Real.sin y)^(1/2) - 6) ≥ Real.sqrt 3) ↔ 
  (∃ (n k : ℤ), x = π/4 + n*π ∧ y = k*π) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l3222_322269


namespace NUMINAMATH_CALUDE_max_ab_value_max_ab_value_achieved_l3222_322299

theorem max_ab_value (a b : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc 0 1 → |a * x + b| ≤ 1) → 
  a * b ≤ (1/4 : ℝ) := by
  sorry

theorem max_ab_value_achieved (a b : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc 0 1 → |a * x + b| ≤ 1) → 
  (∃ a' b' : ℝ, (∀ x : ℝ, x ∈ Set.Icc 0 1 → |a' * x + b'| ≤ 1) ∧ a' * b' = (1/4 : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_max_ab_value_max_ab_value_achieved_l3222_322299


namespace NUMINAMATH_CALUDE_equation_solution_l3222_322286

theorem equation_solution (x : ℝ) : (x + 5)^2 = 16 ↔ x = -1 ∨ x = -9 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3222_322286


namespace NUMINAMATH_CALUDE_correct_factorization_l3222_322227

theorem correct_factorization (x : ℝ) : x^2 - x + (1/4) = (x - 1/2)^2 := by
  sorry

end NUMINAMATH_CALUDE_correct_factorization_l3222_322227


namespace NUMINAMATH_CALUDE_region_area_theorem_l3222_322222

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Calculates the area of the region bounded by two circles and the x-axis -/
def areaRegion (c1 c2 : Circle) : ℝ :=
  sorry

theorem region_area_theorem (c1 c2 : Circle) 
  (h1 : c1.center = (3, 5) ∧ c1.radius = 5)
  (h2 : c2.center = (13, 5) ∧ c2.radius = 5) : 
  areaRegion c1 c2 = 50 - 12.5 * Real.pi :=
sorry

end NUMINAMATH_CALUDE_region_area_theorem_l3222_322222


namespace NUMINAMATH_CALUDE_range_of_a_l3222_322274

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := 1 + a * x
def g (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * x + a

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x ∈ Set.Ioo 0 2, f a x ≥ 0
def q (a : ℝ) : Prop := ∃ x > 0, g a x = 0

-- State the theorem
theorem range_of_a :
  {a : ℝ | (p a ∨ q a) ∧ ¬(p a ∧ q a)} =
  Set.Icc (-1) (-1/2) ∪ Set.Ioi 0 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3222_322274


namespace NUMINAMATH_CALUDE_vector_equality_transitivity_l3222_322290

variable {V : Type*} [AddCommGroup V]

theorem vector_equality_transitivity (a b c : V) : a = b → b = c → a = c := by
  sorry

end NUMINAMATH_CALUDE_vector_equality_transitivity_l3222_322290


namespace NUMINAMATH_CALUDE_sum_of_digits_10_pow_93_minus_937_l3222_322208

def digit_sum (n : ℕ) : ℕ := sorry

theorem sum_of_digits_10_pow_93_minus_937 :
  digit_sum (10^93 - 937) = 819 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_10_pow_93_minus_937_l3222_322208


namespace NUMINAMATH_CALUDE_badge_ratio_l3222_322203

/-- Proves that the ratio of delegates who made their own badges to delegates without pre-printed badges is 1:2 -/
theorem badge_ratio (total : ℕ) (pre_printed : ℕ) (no_badge : ℕ) 
  (h1 : total = 36)
  (h2 : pre_printed = 16)
  (h3 : no_badge = 10) : 
  (total - pre_printed - no_badge) / (total - pre_printed) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_badge_ratio_l3222_322203


namespace NUMINAMATH_CALUDE_perpendicular_unit_vector_l3222_322294

def a : ℝ × ℝ := (2, 1)

theorem perpendicular_unit_vector :
  let v : ℝ × ℝ := (Real.sqrt 5 / 5, -2 * Real.sqrt 5 / 5)
  (v.1 * v.1 + v.2 * v.2 = 1) ∧ (a.1 * v.1 + a.2 * v.2 = 0) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_unit_vector_l3222_322294


namespace NUMINAMATH_CALUDE_complex_arithmetic_l3222_322219

theorem complex_arithmetic (A M S : ℂ) (P : ℝ) 
  (hA : A = 5 - 2*I) 
  (hM : M = -3 + 2*I) 
  (hS : S = 2*I) 
  (hP : P = 3) : 
  A - M + S - P = 5 - 2*I := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_l3222_322219


namespace NUMINAMATH_CALUDE_phone_number_revenue_l3222_322292

theorem phone_number_revenue (X Y : ℕ) : 
  125 * X - 64 * Y = 5 ∧ X < 250 ∧ Y < 250 → 
  (X = 41 ∧ Y = 80) ∨ (X = 105 ∧ Y = 205) :=
sorry

end NUMINAMATH_CALUDE_phone_number_revenue_l3222_322292


namespace NUMINAMATH_CALUDE_class_size_l3222_322289

theorem class_size (error_increase : ℝ) (average_increase : ℝ) (n : ℕ) : 
  error_increase = 20 →
  average_increase = 1/2 →
  error_increase = n * average_increase →
  n = 40 := by
  sorry

end NUMINAMATH_CALUDE_class_size_l3222_322289


namespace NUMINAMATH_CALUDE_james_annual_training_hours_l3222_322271

/-- Calculates the annual training hours for an athlete with a specific schedule -/
def annualTrainingHours (sessionsPerDay : ℕ) (hoursPerSession : ℕ) (daysPerWeek : ℕ) : ℕ :=
  sessionsPerDay * hoursPerSession * daysPerWeek * 52

/-- Proves that James' annual training hours equal 2080 -/
theorem james_annual_training_hours :
  annualTrainingHours 2 4 5 = 2080 := by
  sorry

#eval annualTrainingHours 2 4 5

end NUMINAMATH_CALUDE_james_annual_training_hours_l3222_322271


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l3222_322217

theorem absolute_value_equation_solution : 
  {x : ℝ | |x - 5| = 3*x + 6} = {-11/2, -1/4} := by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l3222_322217


namespace NUMINAMATH_CALUDE_factorial_250_trailing_zeros_l3222_322236

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

/-- Theorem: 250! ends with 62 zeros -/
theorem factorial_250_trailing_zeros :
  trailingZeros 250 = 62 := by
  sorry

end NUMINAMATH_CALUDE_factorial_250_trailing_zeros_l3222_322236


namespace NUMINAMATH_CALUDE_fractional_part_inequality_l3222_322253

theorem fractional_part_inequality (α : ℝ) (h_α : 0 < α ∧ α < 1) :
  ∃ x : ℝ, 0 < x ∧ x < 1 ∧ ∀ n : ℕ+, α^(n : ℝ) < (n : ℝ) * x - ⌊(n : ℝ) * x⌋ := by
  sorry

end NUMINAMATH_CALUDE_fractional_part_inequality_l3222_322253


namespace NUMINAMATH_CALUDE_factorial_equation_solutions_l3222_322223

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem factorial_equation_solutions :
  ∀ a b c : ℕ+,
    (factorial a.val + factorial b.val = 2^(factorial c.val)) ↔
    ((a, b, c) = (1, 1, 1) ∨ (a, b, c) = (2, 2, 2)) :=
by sorry

end NUMINAMATH_CALUDE_factorial_equation_solutions_l3222_322223


namespace NUMINAMATH_CALUDE_tan_theta_range_l3222_322220

theorem tan_theta_range (θ : Real) (a : Real) 
  (h1 : -π/2 < θ ∧ θ < π/2) 
  (h2 : Real.sin θ + Real.cos θ = a) 
  (h3 : 0 < a ∧ a < 1) : 
  -1 < Real.tan θ ∧ Real.tan θ < 0 := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_range_l3222_322220


namespace NUMINAMATH_CALUDE_partner_investment_time_l3222_322275

/-- Given two partners p and q with investment and profit ratios, prove q's investment time -/
theorem partner_investment_time
  (investment_ratio_p investment_ratio_q : ℚ)
  (profit_ratio_p profit_ratio_q : ℚ)
  (investment_time_p : ℚ) :
  investment_ratio_p = 7 →
  investment_ratio_q = 5 →
  profit_ratio_p = 7 →
  profit_ratio_q = 10 →
  investment_time_p = 2 →
  ∃ (investment_time_q : ℚ),
    investment_time_q = 4 ∧
    (profit_ratio_p / profit_ratio_q) =
    ((investment_ratio_p * investment_time_p) /
     (investment_ratio_q * investment_time_q)) :=
by sorry


end NUMINAMATH_CALUDE_partner_investment_time_l3222_322275


namespace NUMINAMATH_CALUDE_sum_of_ages_l3222_322209

/-- Given that Ben is 3 years younger than Dan and Ben is 25 years old, 
    prove that the sum of their ages is 53. -/
theorem sum_of_ages (ben_age dan_age : ℕ) 
  (h1 : ben_age = 25) 
  (h2 : dan_age = ben_age + 3) : 
  ben_age + dan_age = 53 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ages_l3222_322209


namespace NUMINAMATH_CALUDE_water_level_change_notation_l3222_322247

/-- Represents the change in water level -/
def WaterLevelChange : ℤ → ℝ
  | 2 => 2
  | -2 => -2
  | _ => 0  -- Default case, not relevant for this problem

/-- The water level rise notation -/
def WaterLevelRiseNotation : ℝ := 2

/-- The water level drop notation -/
def WaterLevelDropNotation : ℝ := -2

theorem water_level_change_notation :
  WaterLevelChange 2 = WaterLevelRiseNotation ∧
  WaterLevelChange (-2) = WaterLevelDropNotation :=
by sorry

end NUMINAMATH_CALUDE_water_level_change_notation_l3222_322247


namespace NUMINAMATH_CALUDE_candies_remaining_l3222_322261

-- Define the number of candies for each color
def red_candies : ℕ := 50
def yellow_candies : ℕ := 3 * red_candies - 35
def blue_candies : ℕ := (2 * yellow_candies) / 3
def green_candies : ℕ := 20
def purple_candies : ℕ := green_candies / 2
def silver_candies : ℕ := 10

-- Define the number of candies Carlos ate
def carlos_ate : ℕ := yellow_candies + green_candies / 2

-- Define the total number of candies
def total_candies : ℕ := red_candies + yellow_candies + blue_candies + green_candies + purple_candies + silver_candies

-- Theorem statement
theorem candies_remaining : total_candies - carlos_ate = 156 := by
  sorry

end NUMINAMATH_CALUDE_candies_remaining_l3222_322261


namespace NUMINAMATH_CALUDE_sphere_surface_area_l3222_322278

theorem sphere_surface_area (d : ℝ) (h : d = 12) : 
  4 * Real.pi * (d / 2)^2 = 144 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l3222_322278


namespace NUMINAMATH_CALUDE_triangle_side_length_l3222_322233

theorem triangle_side_length (a b c : ℝ) (R : ℝ) :
  a > 0 → b > 0 → c > 0 → R > 0 →
  (a^2 / (b * c)) - (c / b) - (b / c) = Real.sqrt 3 →
  R = 3 →
  a = 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3222_322233


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l3222_322226

theorem point_in_fourth_quadrant (P : ℝ × ℝ) :
  P.1 = Real.tan (2011 * π / 180) →
  P.2 = Real.cos (2011 * π / 180) →
  Real.tan (2011 * π / 180) > 0 →
  Real.cos (2011 * π / 180) < 0 →
  P.1 > 0 ∧ P.2 < 0 := by
sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l3222_322226


namespace NUMINAMATH_CALUDE_max_value_x_cubed_over_polynomial_l3222_322272

theorem max_value_x_cubed_over_polynomial (x : ℝ) :
  x^3 / (x^6 + x^4 + x^3 - 3*x^2 + 9) ≤ 1/7 ∧
  ∃ y : ℝ, y^3 / (y^6 + y^4 + y^3 - 3*y^2 + 9) = 1/7 :=
by sorry

end NUMINAMATH_CALUDE_max_value_x_cubed_over_polynomial_l3222_322272


namespace NUMINAMATH_CALUDE_john_weight_on_bar_l3222_322228

/-- The weight John can put on the bar given the weight bench capacity, safety margin, and his own weight -/
def weight_on_bar (bench_capacity : ℝ) (safety_margin : ℝ) (john_weight : ℝ) : ℝ :=
  bench_capacity * (1 - safety_margin) - john_weight

/-- Theorem stating the weight John can put on the bar -/
theorem john_weight_on_bar :
  weight_on_bar 1000 0.2 250 = 550 := by
  sorry

end NUMINAMATH_CALUDE_john_weight_on_bar_l3222_322228


namespace NUMINAMATH_CALUDE_correct_locus_definition_l3222_322277

-- Define a type for points in a geometric space
variable {Point : Type}

-- Define a predicate for the locus condition
variable (locus_condition : Point → Prop)

-- Define the locus as a set of points
def locus (locus_condition : Point → Prop) : Set Point :=
  {p : Point | locus_condition p}

-- State the theorem
theorem correct_locus_definition (p : Point) :
  p ∈ locus locus_condition ↔ locus_condition p :=
sorry

end NUMINAMATH_CALUDE_correct_locus_definition_l3222_322277


namespace NUMINAMATH_CALUDE_both_vegan_and_kosher_l3222_322232

/-- Represents the meal delivery scenario -/
structure MealDelivery where
  total : ℕ
  vegan : ℕ
  kosher : ℕ
  neither : ℕ

/-- Theorem stating the number of clients needing both vegan and kosher meals -/
theorem both_vegan_and_kosher (m : MealDelivery) 
  (h_total : m.total = 30)
  (h_vegan : m.vegan = 7)
  (h_kosher : m.kosher = 8)
  (h_neither : m.neither = 18) :
  m.total - m.neither - (m.vegan + m.kosher - (m.total - m.neither)) = 3 := by
  sorry

#check both_vegan_and_kosher

end NUMINAMATH_CALUDE_both_vegan_and_kosher_l3222_322232


namespace NUMINAMATH_CALUDE_complex_sum_argument_l3222_322211

theorem complex_sum_argument : ∃ (r : ℝ), 
  Complex.exp (11 * Real.pi * Complex.I / 60) + 
  Complex.exp (23 * Real.pi * Complex.I / 60) + 
  Complex.exp (35 * Real.pi * Complex.I / 60) + 
  Complex.exp (47 * Real.pi * Complex.I / 60) + 
  Complex.exp (59 * Real.pi * Complex.I / 60) + 
  Complex.exp (Real.pi * Complex.I / 60) = 
  r * Complex.exp (7 * Real.pi * Complex.I / 12) :=
by sorry

end NUMINAMATH_CALUDE_complex_sum_argument_l3222_322211


namespace NUMINAMATH_CALUDE_frog_jump_distance_l3222_322254

/-- The jumping contest between a grasshopper and a frog -/
theorem frog_jump_distance (grasshopper_jump : ℕ) (additional_distance : ℕ) 
  (h1 : grasshopper_jump = 25)
  (h2 : additional_distance = 15) : 
  grasshopper_jump + additional_distance = 40 := by
  sorry

#check frog_jump_distance

end NUMINAMATH_CALUDE_frog_jump_distance_l3222_322254


namespace NUMINAMATH_CALUDE_quadratic_root_in_interval_l3222_322235

theorem quadratic_root_in_interval (a b c : ℝ) (h : 2*a + 3*b + 6*c = 0) :
  ∃ x : ℝ, 0 < x ∧ x < 1 ∧ a*x^2 + b*x + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_in_interval_l3222_322235


namespace NUMINAMATH_CALUDE_people_in_line_l3222_322214

theorem people_in_line (people_in_front : ℕ) (people_behind : ℕ) : 
  people_in_front = 11 → people_behind = 12 → people_in_front + people_behind + 1 = 24 := by
  sorry

end NUMINAMATH_CALUDE_people_in_line_l3222_322214


namespace NUMINAMATH_CALUDE_distinct_sums_largest_value_l3222_322202

theorem distinct_sums_largest_value (A B C D : ℕ) : 
  (A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) →
  (A + C ≠ B + C ∧ A + C ≠ B + D ∧ A + C ≠ D + A ∧
   B + C ≠ B + D ∧ B + C ≠ D + A ∧
   B + D ≠ D + A) →
  ({A, B, C, D, A + C, B + C, B + D, D + A} : Finset ℕ) = {1, 2, 3, 4, 5, 6, 7, 8} →
  A > B ∧ A > C ∧ A > D →
  A = 12 := by
sorry

end NUMINAMATH_CALUDE_distinct_sums_largest_value_l3222_322202


namespace NUMINAMATH_CALUDE_trapezoid_perimeter_is_220_l3222_322216

/-- A trapezoid with the given properties -/
structure Trapezoid where
  AB : ℝ
  CD : ℝ
  BC : ℝ
  angle_BCD : ℝ

/-- The perimeter of the trapezoid -/
def perimeter (t : Trapezoid) : ℝ := t.AB + 2 * t.BC + t.CD

/-- Theorem stating that the perimeter of the given trapezoid is 220 -/
theorem trapezoid_perimeter_is_220 (t : Trapezoid) 
  (h1 : t.AB = 60)
  (h2 : t.CD = 40)
  (h3 : t.angle_BCD = 120 * π / 180)
  (h4 : t.BC = Real.sqrt (t.CD^2 + (t.AB - t.CD)^2 - 2 * t.CD * (t.AB - t.CD) * Real.cos t.angle_BCD)) :
  perimeter t = 220 := by
  sorry

#check trapezoid_perimeter_is_220

end NUMINAMATH_CALUDE_trapezoid_perimeter_is_220_l3222_322216


namespace NUMINAMATH_CALUDE_fireflies_that_flew_away_l3222_322287

def initial_fireflies : ℕ := 3
def additional_fireflies : ℕ := 12 - 4
def remaining_fireflies : ℕ := 9

theorem fireflies_that_flew_away :
  initial_fireflies + additional_fireflies - remaining_fireflies = 2 := by
  sorry

end NUMINAMATH_CALUDE_fireflies_that_flew_away_l3222_322287


namespace NUMINAMATH_CALUDE_quadratic_solution_difference_squared_l3222_322257

theorem quadratic_solution_difference_squared :
  ∀ d e : ℝ,
  (5 * d^2 + 20 * d - 55 = 0) →
  (5 * e^2 + 20 * e - 55 = 0) →
  (d - e)^2 = 600 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_difference_squared_l3222_322257


namespace NUMINAMATH_CALUDE_notebook_price_l3222_322262

theorem notebook_price :
  ∀ (s n c : ℕ),
  s > 18 →
  s ≤ 36 →
  c > n →
  s * n * c = 990 →
  c = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_notebook_price_l3222_322262


namespace NUMINAMATH_CALUDE_puppies_per_cage_l3222_322265

theorem puppies_per_cage 
  (initial_puppies : ℕ) 
  (sold_puppies : ℕ) 
  (num_cages : ℕ) 
  (h1 : initial_puppies = 78) 
  (h2 : sold_puppies = 30) 
  (h3 : num_cages = 6) 
  (h4 : initial_puppies > sold_puppies) :
  (initial_puppies - sold_puppies) / num_cages = 8 := by
  sorry

end NUMINAMATH_CALUDE_puppies_per_cage_l3222_322265


namespace NUMINAMATH_CALUDE_martins_walk_l3222_322218

/-- The distance between Martin's house and Lawrence's house -/
def distance : ℝ := 12

/-- The time Martin spent walking -/
def time : ℝ := 6

/-- Martin's walking speed -/
def speed : ℝ := 2

/-- Theorem: The distance between Martin's house and Lawrence's house is 12 miles -/
theorem martins_walk : distance = speed * time := by
  sorry

end NUMINAMATH_CALUDE_martins_walk_l3222_322218


namespace NUMINAMATH_CALUDE_cases_in_1990_l3222_322251

-- Define the initial and final years
def initialYear : ℕ := 1970
def finalYear : ℕ := 2000
def midYear : ℕ := 1990

-- Define the initial and final number of cases
def initialCases : ℕ := 600000
def finalCases : ℕ := 2000

-- Function to calculate cases at a given year assuming linear decrease
def casesAtYear (year : ℕ) : ℚ :=
  initialCases - (initialCases - finalCases : ℚ) * (year - initialYear : ℚ) / (finalYear - initialYear : ℚ)

-- Theorem statement
theorem cases_in_1990 : 
  ⌊casesAtYear midYear⌋ = 201333 :=
sorry

end NUMINAMATH_CALUDE_cases_in_1990_l3222_322251


namespace NUMINAMATH_CALUDE_first_expression_equality_second_expression_equality_l3222_322291

-- First expression
theorem first_expression_equality (a : ℝ) :
  (-2 * a)^6 * (-3 * a^3) + (2 * a)^2 * 3 = -192 * a^9 + 12 * a^2 := by sorry

-- Second expression
theorem second_expression_equality :
  |(-1/8)| + π^3 + (-1/2)^3 - (1/3)^2 = π^3 - 1/9 := by sorry

end NUMINAMATH_CALUDE_first_expression_equality_second_expression_equality_l3222_322291


namespace NUMINAMATH_CALUDE_highest_score_l3222_322284

theorem highest_score (total_innings : ℕ) (overall_average : ℚ) (score_difference : ℕ) (average_without_extremes : ℚ) :
  total_innings = 46 →
  overall_average = 63 →
  score_difference = 150 →
  average_without_extremes = 58 →
  ∃ (highest_score lowest_score : ℕ),
    highest_score - lowest_score = score_difference ∧
    (total_innings : ℚ) * overall_average = (total_innings - 2 : ℚ) * average_without_extremes + highest_score + lowest_score ∧
    highest_score = 248 := by
  sorry

end NUMINAMATH_CALUDE_highest_score_l3222_322284


namespace NUMINAMATH_CALUDE_perpendicular_vectors_implies_m_half_l3222_322296

/-- Given two vectors a and b in R², if a is perpendicular to b,
    then the second component of a is equal to 1/2. -/
theorem perpendicular_vectors_implies_m_half (a b : ℝ × ℝ) :
  a.1 = 1 →
  a.2 = m →
  b.1 = -1 →
  b.2 = 2 →
  a.1 * b.1 + a.2 * b.2 = 0 →
  m = 1/2 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_implies_m_half_l3222_322296


namespace NUMINAMATH_CALUDE_num_segments_collinear_points_l3222_322298

/-- The number of distinct segments formed by n collinear points -/
def num_segments (n : ℕ) : ℕ := n.choose 2

/-- Theorem: For n distinct collinear points, the number of distinct segments is n choose 2 -/
theorem num_segments_collinear_points (n : ℕ) (h : n ≥ 2) :
  num_segments n = n.choose 2 := by sorry

end NUMINAMATH_CALUDE_num_segments_collinear_points_l3222_322298


namespace NUMINAMATH_CALUDE_largest_number_with_same_quotient_and_remainder_l3222_322237

theorem largest_number_with_same_quotient_and_remainder : ∃ (n : ℕ), n = 90 ∧
  (∀ m : ℕ, m > n →
    ¬(∃ (q r : ℕ), m = 13 * q + r ∧ m = 15 * q + r ∧ r < 13 ∧ r < 15)) ∧
  (∃ (q r : ℕ), n = 13 * q + r ∧ n = 15 * q + r ∧ r < 13 ∧ r < 15) :=
by sorry

end NUMINAMATH_CALUDE_largest_number_with_same_quotient_and_remainder_l3222_322237


namespace NUMINAMATH_CALUDE_equal_squares_of_equal_products_l3222_322297

theorem equal_squares_of_equal_products (a b c d : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
  (h1 : a * (b + c + d) = b * (a + c + d))
  (h2 : a * (b + c + d) = c * (a + b + d))
  (h3 : a * (b + c + d) = d * (a + b + c)) :
  a^2 = b^2 ∧ a^2 = c^2 ∧ a^2 = d^2 := by
sorry

end NUMINAMATH_CALUDE_equal_squares_of_equal_products_l3222_322297


namespace NUMINAMATH_CALUDE_simplify_complex_expression_l3222_322230

theorem simplify_complex_expression (x : ℝ) (hx : x > 0) : 
  Real.sqrt (2 * (1 + Real.sqrt (1 + ((x^4 - 1) / (2 * x^2))^2))) = (x^2 + 1) / x := by
  sorry

end NUMINAMATH_CALUDE_simplify_complex_expression_l3222_322230


namespace NUMINAMATH_CALUDE_brick_wall_theorem_l3222_322231

/-- Calculates the total number of bricks in a wall with a given number of rows,
    where each row has one less brick than the row below it. -/
def totalBricks (rows : ℕ) (bottomRowBricks : ℕ) : ℕ :=
  (rows * (2 * bottomRowBricks - rows + 1)) / 2

/-- Theorem: A brick wall with 5 rows, where the bottom row has 38 bricks
    and each subsequent row has one less brick than the row below it,
    contains a total of 180 bricks. -/
theorem brick_wall_theorem :
  totalBricks 5 38 = 180 := by
  sorry

end NUMINAMATH_CALUDE_brick_wall_theorem_l3222_322231


namespace NUMINAMATH_CALUDE_no_representation_2023_l3222_322280

theorem no_representation_2023 : ¬∃ (a b c : ℕ), 
  (a + b + c = 2023) ∧ 
  (∃ k : ℕ, a = k * (b + c)) ∧ 
  (∃ m : ℕ, b + c = m * (b - c + 1)) := by
sorry

end NUMINAMATH_CALUDE_no_representation_2023_l3222_322280


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l3222_322225

theorem quadratic_expression_value (a : ℝ) (h : 2 * a^2 - a - 3 = 0) :
  (2 * a + 3) * (2 * a - 3) + (2 * a - 1)^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l3222_322225


namespace NUMINAMATH_CALUDE_least_five_digit_congruent_to_8_mod_17_l3222_322241

theorem least_five_digit_congruent_to_8_mod_17 : 
  ∀ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ n ≡ 8 [ZMOD 17] → n ≥ 10009 :=
by sorry

end NUMINAMATH_CALUDE_least_five_digit_congruent_to_8_mod_17_l3222_322241


namespace NUMINAMATH_CALUDE_cabbage_production_increase_cabbage_production_increase_holds_l3222_322248

theorem cabbage_production_increase : ℕ → Prop :=
  fun n =>
    (∃ a : ℕ, a * a = 11236) ∧
    (∀ b : ℕ, b * b < 11236 → b * b ≤ (n - 1) * (n - 1)) ∧
    (n * n < 11236) →
    11236 - (n - 1) * (n - 1) = 211

theorem cabbage_production_increase_holds : cabbage_production_increase 106 := by
  sorry

end NUMINAMATH_CALUDE_cabbage_production_increase_cabbage_production_increase_holds_l3222_322248


namespace NUMINAMATH_CALUDE_popsicle_consumption_l3222_322244

/-- The number of Popsicles eaten in a given time period -/
def popsicles_eaten (rate : ℚ) (time : ℚ) : ℚ :=
  time / rate

/-- Proves that eating 1 Popsicle every 20 minutes for 6 hours results in 18 Popsicles -/
theorem popsicle_consumption : popsicles_eaten (20 / 60) 6 = 18 := by
  sorry

#eval popsicles_eaten (20 / 60) 6

end NUMINAMATH_CALUDE_popsicle_consumption_l3222_322244


namespace NUMINAMATH_CALUDE_sin_cos_difference_equals_neg_half_l3222_322255

theorem sin_cos_difference_equals_neg_half : 
  Real.sin (119 * π / 180) * Real.cos (91 * π / 180) - 
  Real.sin (91 * π / 180) * Real.sin (29 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_difference_equals_neg_half_l3222_322255


namespace NUMINAMATH_CALUDE_percentage_problem_l3222_322281

theorem percentage_problem : ∃ P : ℝ, P * 600 = 50 / 100 * 900 ∧ P = 75 / 100 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3222_322281


namespace NUMINAMATH_CALUDE_inverse_proportion_l3222_322260

/-- Given that x is inversely proportional to y, prove that if x = 5 when y = 15, 
    then x = -5/2 when y = -30 -/
theorem inverse_proportion (x y : ℝ) (h : ∃ k : ℝ, ∀ x y, x * y = k) 
    (h1 : 5 * 15 = x * y) : 
  x * (-30) = 5 * 15 → x = -5/2 := by sorry

end NUMINAMATH_CALUDE_inverse_proportion_l3222_322260


namespace NUMINAMATH_CALUDE_continued_proportionality_and_linear_combination_l3222_322240

theorem continued_proportionality_and_linear_combination :
  -- Part (1)
  (∀ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 →
    x / (2*y + z) = y / (2*z + x) ∧ y / (2*z + x) = z / (2*x + y) →
    x / (2*y + z) = 1/3) ∧
  -- Part (2)
  (∀ (a b c : ℝ), a ≠ b ∧ b ≠ c ∧ c ≠ a →
    (a + b) / (a - b) = (b + c) / (2*(b - c)) ∧
    (b + c) / (2*(b - c)) = (c + a) / (3*(c - a)) →
    8*a + 9*b + 5*c = 0) := by
  sorry

end NUMINAMATH_CALUDE_continued_proportionality_and_linear_combination_l3222_322240


namespace NUMINAMATH_CALUDE_shop_owner_profit_l3222_322200

/-- Represents the profit calculation for a shop owner using false weights -/
theorem shop_owner_profit (buying_cheat : ℝ) (selling_cheat : ℝ) : 
  buying_cheat = 0.12 →
  selling_cheat = 0.30 →
  let actual_buy_amount := 1 + buying_cheat
  let actual_sell_amount := 1 - selling_cheat
  let sell_portions := actual_buy_amount / actual_sell_amount
  let revenue := sell_portions * 100
  let profit := revenue - 100
  let percentage_profit := (profit / 100) * 100
  percentage_profit = 60 := by
sorry


end NUMINAMATH_CALUDE_shop_owner_profit_l3222_322200


namespace NUMINAMATH_CALUDE_even_decreasing_implies_increasing_l3222_322288

-- Define the properties of the function
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def IsDecreasingOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

def IsIncreasingOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

-- State the theorem
theorem even_decreasing_implies_increasing 
  (f : ℝ → ℝ) (a b : ℝ) 
  (h_pos : 0 < a ∧ a < b) 
  (h_even : IsEven f) 
  (h_decreasing : IsDecreasingOn f a b) : 
  IsIncreasingOn f (-b) (-a) :=
by
  sorry

end NUMINAMATH_CALUDE_even_decreasing_implies_increasing_l3222_322288


namespace NUMINAMATH_CALUDE_inequality_and_equality_conditions_l3222_322259

theorem inequality_and_equality_conditions (a b : ℝ) 
  (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : a + b = 1) :
  (1/2 ≤ (a^3 + b^3) / (a^2 + b^2)) ∧ 
  ((a^3 + b^3) / (a^2 + b^2) ≤ 1) ∧
  ((a^3 + b^3) / (a^2 + b^2) = 1/2 ↔ a = 1/2 ∧ b = 1/2) ∧
  ((a^3 + b^3) / (a^2 + b^2) = 1 ↔ (a = 0 ∧ b = 1) ∨ (a = 1 ∧ b = 0)) :=
sorry

end NUMINAMATH_CALUDE_inequality_and_equality_conditions_l3222_322259


namespace NUMINAMATH_CALUDE_adult_ticket_price_l3222_322224

theorem adult_ticket_price 
  (total_amount : ℕ)
  (child_price : ℕ)
  (total_tickets : ℕ)
  (child_tickets : ℕ)
  (h1 : total_amount = 104)
  (h2 : child_price = 4)
  (h3 : total_tickets = 21)
  (h4 : child_tickets = 11) :
  ∃ (adult_price : ℕ), 
    adult_price * (total_tickets - child_tickets) + child_price * child_tickets = total_amount ∧ 
    adult_price = 6 :=
by sorry

end NUMINAMATH_CALUDE_adult_ticket_price_l3222_322224


namespace NUMINAMATH_CALUDE_find_a_l3222_322285

-- Define the universal set U
def U (a : ℤ) : Set ℤ := {2, 4, a^2 - a + 1}

-- Define set A
def A (a : ℤ) : Set ℤ := {a + 4, 4}

-- Define the complement of A relative to U
def complement_A (a : ℤ) : Set ℤ := U a \ A a

-- Theorem statement
theorem find_a : ∃ (a : ℤ), 
  (U a = {2, 4, a^2 - a + 1}) ∧ 
  (A a = {a + 4, 4}) ∧ 
  (complement_A a = {7}) ∧ 
  (a = -2) :=
sorry

end NUMINAMATH_CALUDE_find_a_l3222_322285


namespace NUMINAMATH_CALUDE_count_symmetric_scanning_codes_l3222_322276

/-- A symmetric scanning code is a 5x5 grid of squares that remains unchanged
    when rotated by multiples of 90° or reflected horizontally or vertically. -/
def SymmetricScanningCode : Type := Unit

/-- The number of distinct regions in a symmetric 5x5 scanning code -/
def NumDistinctRegions : Nat := 5

/-- The function that counts the number of valid symmetric scanning codes -/
def countValidSymmetricScanningCodes : Nat :=
  2^NumDistinctRegions - 2

/-- Theorem stating that the number of valid symmetric 5x5 scanning codes is 30 -/
theorem count_symmetric_scanning_codes :
  countValidSymmetricScanningCodes = 30 := by
  sorry

#check count_symmetric_scanning_codes

end NUMINAMATH_CALUDE_count_symmetric_scanning_codes_l3222_322276


namespace NUMINAMATH_CALUDE_composite_19_8n_17_l3222_322249

theorem composite_19_8n_17 (n : ℕ) (hn : n > 0) : 
  ∃ (k : ℕ), k > 1 ∧ k < 19 * 8^n + 17 ∧ (19 * 8^n + 17) % k = 0 := by
  sorry

end NUMINAMATH_CALUDE_composite_19_8n_17_l3222_322249


namespace NUMINAMATH_CALUDE_cube_edge_length_l3222_322229

theorem cube_edge_length 
  (paint_cost : ℝ) 
  (coverage_per_quart : ℝ) 
  (total_cost : ℝ) 
  (h1 : paint_cost = 3.20)
  (h2 : coverage_per_quart = 120)
  (h3 : total_cost = 16) : 
  ∃ (edge_length : ℝ), edge_length = 10 ∧ 
  6 * edge_length^2 = (total_cost / paint_cost) * coverage_per_quart :=
by
  sorry

end NUMINAMATH_CALUDE_cube_edge_length_l3222_322229


namespace NUMINAMATH_CALUDE_decimal_sum_to_fraction_l3222_322258

theorem decimal_sum_to_fraction :
  (0.1 : ℚ) + 0.03 + 0.004 + 0.0006 + 0.00007 = 13467 / 100000 := by
  sorry

end NUMINAMATH_CALUDE_decimal_sum_to_fraction_l3222_322258


namespace NUMINAMATH_CALUDE_isosceles_triangle_l3222_322267

theorem isosceles_triangle (a b c : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_equation : a^2 - b^2 + a*c - b*c = 0) : 
  a = b ∨ b = c ∨ c = a := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_l3222_322267


namespace NUMINAMATH_CALUDE_twelve_disks_on_circle_l3222_322210

theorem twelve_disks_on_circle (n : ℕ) (r : ℝ) :
  n = 12 ∧ r = 1 →
  ∃ (disk_radius : ℝ),
    disk_radius = 2 - Real.sqrt 3 ∧
    n * (π * disk_radius^2) = π * (84 - 48 * Real.sqrt 3) ∧
    ∃ (a b c : ℕ), a = 84 ∧ b = 48 ∧ c = 3 ∧ a + b + c = 135 :=
by sorry

end NUMINAMATH_CALUDE_twelve_disks_on_circle_l3222_322210


namespace NUMINAMATH_CALUDE_book_price_percentage_l3222_322212

theorem book_price_percentage (suggested_retail_price : ℝ) : 
  suggested_retail_price > 0 →
  let marked_price := 0.7 * suggested_retail_price
  let purchase_price := 0.5 * marked_price
  purchase_price / suggested_retail_price = 0.35 := by sorry

end NUMINAMATH_CALUDE_book_price_percentage_l3222_322212


namespace NUMINAMATH_CALUDE_stationery_ratio_is_three_to_one_l3222_322263

/-- The number of pieces of stationery Georgia has -/
def georgia_stationery : ℕ := 25

/-- The number of pieces of stationery Lorene has -/
def lorene_stationery : ℕ := georgia_stationery + 50

/-- The ratio of Lorene's stationery to Georgia's stationery -/
def stationery_ratio : ℚ := lorene_stationery / georgia_stationery

theorem stationery_ratio_is_three_to_one :
  stationery_ratio = 3 / 1 := by
  sorry

end NUMINAMATH_CALUDE_stationery_ratio_is_three_to_one_l3222_322263

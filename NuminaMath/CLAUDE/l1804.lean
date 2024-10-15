import Mathlib

namespace NUMINAMATH_CALUDE_present_age_of_b_l1804_180443

theorem present_age_of_b (a b : ℕ) : 
  (a + 10 = 2 * (b - 10)) →  -- In 10 years, A will be twice as old as B was 10 years ago
  (a = b + 9) →              -- A is currently 9 years older than B
  b = 39                     -- B's present age is 39 years
:= by sorry

end NUMINAMATH_CALUDE_present_age_of_b_l1804_180443


namespace NUMINAMATH_CALUDE_counterexample_exists_l1804_180459

def is_in_set (n : ℕ) : Prop := n = 14 ∨ n = 18 ∨ n = 20 ∨ n = 24 ∨ n = 30

theorem counterexample_exists : 
  ∃ n : ℕ, ¬(Nat.Prime n) ∧ ¬(Nat.Prime (n + 2)) ∧ is_in_set n :=
sorry

end NUMINAMATH_CALUDE_counterexample_exists_l1804_180459


namespace NUMINAMATH_CALUDE_inequality_proof_l1804_180489

theorem inequality_proof (a b x y : ℝ) 
  (ha : a > 0) (hb : b > 0) (hx : x > 0) (hy : y > 0) 
  (hab : a + b = 1) : 
  (a*x + b*y) * (b*x + a*y) ≥ x*y := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1804_180489


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_factorial_squared_gt_power_of_two_l1804_180490

theorem negation_of_existence (P : ℕ → Prop) : 
  (¬ ∃ n, P n) ↔ (∀ n, ¬ P n) :=
by sorry

theorem negation_of_factorial_squared_gt_power_of_two : 
  (¬ ∃ n : ℕ, (n.factorial ^ 2 : ℝ) > 2^n) ↔ 
  (∀ n : ℕ, (n.factorial ^ 2 : ℝ) ≤ 2^n) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_factorial_squared_gt_power_of_two_l1804_180490


namespace NUMINAMATH_CALUDE_book_cost_l1804_180407

/-- The cost of a book given partial payment and a condition on the remaining amount -/
theorem book_cost (paid : ℝ) (total_cost : ℝ) : 
  paid = 100 →
  (total_cost - paid) = (total_cost - (total_cost - paid)) →
  total_cost = 200 := by
  sorry

end NUMINAMATH_CALUDE_book_cost_l1804_180407


namespace NUMINAMATH_CALUDE_gasoline_reduction_percentage_l1804_180483

theorem gasoline_reduction_percentage
  (original_price : ℝ)
  (original_quantity : ℝ)
  (price_increase_percentage : ℝ)
  (spending_increase_percentage : ℝ)
  (h1 : price_increase_percentage = 0.25)
  (h2 : spending_increase_percentage = 0.05)
  (h3 : original_price > 0)
  (h4 : original_quantity > 0) :
  let new_price := original_price * (1 + price_increase_percentage)
  let new_total_cost := original_price * original_quantity * (1 + spending_increase_percentage)
  let new_quantity := new_total_cost / new_price
  (1 - new_quantity / original_quantity) * 100 = 16 := by
sorry

end NUMINAMATH_CALUDE_gasoline_reduction_percentage_l1804_180483


namespace NUMINAMATH_CALUDE_mary_max_earnings_l1804_180472

/-- Calculates the maximum weekly earnings for a worker with given parameters. -/
def maxWeeklyEarnings (maxHours : ℕ) (regularHours : ℕ) (regularRate : ℚ) (overtimeRateIncrease : ℚ) : ℚ :=
  let overtimeRate := regularRate * (1 + overtimeRateIncrease)
  let regularEarnings := (regularHours.min maxHours : ℚ) * regularRate
  let overtimeHours := maxHours - regularHours
  let overtimeEarnings := (overtimeHours.max 0 : ℚ) * overtimeRate
  regularEarnings + overtimeEarnings

/-- Theorem stating Mary's maximum weekly earnings -/
theorem mary_max_earnings :
  maxWeeklyEarnings 80 20 8 (1/4) = 760 := by
  sorry

end NUMINAMATH_CALUDE_mary_max_earnings_l1804_180472


namespace NUMINAMATH_CALUDE_lcm_gcf_product_24_60_l1804_180456

theorem lcm_gcf_product_24_60 : Nat.lcm 24 60 * Nat.gcd 24 60 = 1440 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcf_product_24_60_l1804_180456


namespace NUMINAMATH_CALUDE_rotation_equivalence_l1804_180450

/-- 
Given a point that is rotated 450 degrees clockwise and x degrees counterclockwise 
about the same center to reach the same final position, prove that x = 270 degrees,
assuming x < 360.
-/
theorem rotation_equivalence (x : ℝ) : 
  (450 % 360 : ℝ) = (360 - x) % 360 → x < 360 → x = 270 := by
  sorry

end NUMINAMATH_CALUDE_rotation_equivalence_l1804_180450


namespace NUMINAMATH_CALUDE_asymptote_sum_l1804_180447

/-- Given a rational function y = x / (x^3 + Ax^2 + Bx + C) with integer coefficients A, B, C,
    if the graph has vertical asymptotes at x = -3, 0, 3, then A + B + C = -9 -/
theorem asymptote_sum (A B C : ℤ) :
  (∀ x : ℝ, x ≠ -3 ∧ x ≠ 0 ∧ x ≠ 3 →
    ∃ y : ℝ, y = x / (x^3 + A * x^2 + B * x + C)) →
  (A + B + C = -9) := by
  sorry

end NUMINAMATH_CALUDE_asymptote_sum_l1804_180447


namespace NUMINAMATH_CALUDE_bubble_sort_probability_main_result_l1804_180496

def n : ℕ := 50

/-- The probability that r₂₅ ends up in the 35th position after one bubble pass -/
def probability : ℚ := 1 / 1190

theorem bubble_sort_probability (r : Fin n → ℕ) (h : Function.Injective r) :
  probability = (Nat.factorial 33) / (Nat.factorial 35) :=
sorry

theorem main_result : probability.num + probability.den = 1191 :=
sorry

end NUMINAMATH_CALUDE_bubble_sort_probability_main_result_l1804_180496


namespace NUMINAMATH_CALUDE_infinite_sqrt_two_plus_l1804_180431

theorem infinite_sqrt_two_plus (x : ℝ) : x > 0 ∧ x^2 = 2 + x → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_infinite_sqrt_two_plus_l1804_180431


namespace NUMINAMATH_CALUDE_average_rounds_is_four_l1804_180428

/-- Represents the distribution of golf rounds played by members -/
structure GolfRoundsDistribution where
  rounds : Fin 6 → ℕ
  members : Fin 6 → ℕ

/-- Calculates the average number of rounds played, rounded to the nearest whole number -/
def averageRoundsRounded (dist : GolfRoundsDistribution) : ℕ :=
  let totalRounds := (Finset.range 6).sum (λ i => dist.rounds i * dist.members i)
  let totalMembers := (Finset.range 6).sum (λ i => dist.members i)
  (totalRounds + totalMembers / 2) / totalMembers

/-- The specific distribution given in the problem -/
def givenDistribution : GolfRoundsDistribution where
  rounds := λ i => i.val + 1
  members := ![4, 3, 5, 6, 2, 7]

theorem average_rounds_is_four :
  averageRoundsRounded givenDistribution = 4 := by
  sorry

end NUMINAMATH_CALUDE_average_rounds_is_four_l1804_180428


namespace NUMINAMATH_CALUDE_total_money_l1804_180446

theorem total_money (a b c : ℕ) : 
  a + c = 200 → b + c = 310 → c = 10 → a + b + c = 500 := by
  sorry

end NUMINAMATH_CALUDE_total_money_l1804_180446


namespace NUMINAMATH_CALUDE_difference_of_squares_fifty_thirty_l1804_180452

theorem difference_of_squares_fifty_thirty : 50^2 - 30^2 = 1600 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_fifty_thirty_l1804_180452


namespace NUMINAMATH_CALUDE_product_of_repeating_decimal_and_eight_l1804_180436

/-- Represents the repeating decimal 0.456̄ -/
def repeating_decimal : ℚ := 456 / 999

theorem product_of_repeating_decimal_and_eight :
  repeating_decimal * 8 = 1216 / 333 := by
  sorry

end NUMINAMATH_CALUDE_product_of_repeating_decimal_and_eight_l1804_180436


namespace NUMINAMATH_CALUDE_salaries_sum_l1804_180487

theorem salaries_sum (A_salary B_salary : ℝ) : 
  A_salary = 5250 →
  A_salary * 0.05 = B_salary * 0.15 →
  A_salary + B_salary = 7000 :=
by
  sorry

end NUMINAMATH_CALUDE_salaries_sum_l1804_180487


namespace NUMINAMATH_CALUDE_tan_150_degrees_l1804_180430

theorem tan_150_degrees : Real.tan (150 * π / 180) = -1 / Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_150_degrees_l1804_180430


namespace NUMINAMATH_CALUDE_dartboard_section_angle_l1804_180491

/-- Represents a circular dartboard divided into sections -/
structure Dartboard where
  /-- The probability of a dart landing in a particular section -/
  section_probability : ℝ
  /-- The central angle of the section in degrees -/
  section_angle : ℝ

/-- 
Theorem: For a circular dartboard divided into sections by radius lines, 
if the probability of a dart landing in a particular section is 1/4, 
then the central angle of that section is 90 degrees.
-/
theorem dartboard_section_angle (d : Dartboard) 
  (h_prob : d.section_probability = 1/4) : 
  d.section_angle = 90 := by
  sorry

end NUMINAMATH_CALUDE_dartboard_section_angle_l1804_180491


namespace NUMINAMATH_CALUDE_cube_surface_area_l1804_180406

/-- The surface area of a cube with edge length 6a is 216a² -/
theorem cube_surface_area (a : ℝ) : 
  let edge_length : ℝ := 6 * a
  let surface_area : ℝ := 6 * (edge_length ^ 2)
  surface_area = 216 * (a ^ 2) := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_l1804_180406


namespace NUMINAMATH_CALUDE_discrete_probability_distribution_l1804_180480

theorem discrete_probability_distribution (p₁ p₃ : ℝ) : 
  p₃ = 4 * p₁ →
  p₁ + 0.15 + p₃ + 0.25 + 0.35 = 1 →
  p₁ = 0.05 ∧ p₃ = 0.20 := by
sorry

end NUMINAMATH_CALUDE_discrete_probability_distribution_l1804_180480


namespace NUMINAMATH_CALUDE_overtime_hours_calculation_l1804_180424

theorem overtime_hours_calculation 
  (regular_rate : ℝ) 
  (regular_hours : ℝ) 
  (total_pay : ℝ) 
  (h1 : regular_rate = 3)
  (h2 : regular_hours = 40)
  (h3 : total_pay = 192) :
  let overtime_rate := 2 * regular_rate
  let regular_pay := regular_rate * regular_hours
  let overtime_pay := total_pay - regular_pay
  overtime_pay / overtime_rate = 12 := by
sorry

end NUMINAMATH_CALUDE_overtime_hours_calculation_l1804_180424


namespace NUMINAMATH_CALUDE_fifth_pattern_white_tiles_l1804_180499

/-- The number of white tiles in the n-th pattern of a hexagonal tile sequence -/
def white_tiles (n : ℕ) : ℕ := 4 * n + 2

/-- Theorem: The number of white tiles in the fifth pattern is 22 -/
theorem fifth_pattern_white_tiles : white_tiles 5 = 22 := by
  sorry

end NUMINAMATH_CALUDE_fifth_pattern_white_tiles_l1804_180499


namespace NUMINAMATH_CALUDE_negative_eight_to_four_thirds_equals_sixteen_l1804_180494

theorem negative_eight_to_four_thirds_equals_sixteen :
  (-8 : ℝ) ^ (4/3) = 16 := by
  sorry

end NUMINAMATH_CALUDE_negative_eight_to_four_thirds_equals_sixteen_l1804_180494


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l1804_180432

-- Define variables
variable (a b x y : ℝ)

-- Theorem 1
theorem simplify_expression_1 : 
  6*a + 7*b^2 - 9 + 4*a - b^2 + 6 = 6*b^2 + 10*a - 3 := by sorry

-- Theorem 2
theorem simplify_expression_2 :
  5*x - 2*(4*x + 5*y) + 3*(3*x - 4*y) = 6*x - 22*y := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l1804_180432


namespace NUMINAMATH_CALUDE_last_colored_square_l1804_180479

/-- Represents a position in the rectangle --/
structure Position where
  row : Nat
  col : Nat

/-- Represents the dimensions of the rectangle --/
structure Dimensions where
  width : Nat
  height : Nat

/-- Represents the spiral coloring process --/
def spiralColor (dims : Dimensions) : Position :=
  sorry

/-- Theorem stating the last colored square in a 200x100 rectangle --/
theorem last_colored_square :
  spiralColor ⟨200, 100⟩ = ⟨51, 50⟩ := by
  sorry

end NUMINAMATH_CALUDE_last_colored_square_l1804_180479


namespace NUMINAMATH_CALUDE_magnitude_a_plus_2b_l1804_180455

/-- Given two vectors a and b in ℝ², prove that |a + 2b| = √17 under certain conditions. -/
theorem magnitude_a_plus_2b (a b : ℝ × ℝ) 
  (h1 : ‖a‖ = 1)
  (h2 : ‖b‖ = 2)
  (h3 : a - b = (Real.sqrt 2, Real.sqrt 3)) :
  ‖a + 2 • b‖ = Real.sqrt 17 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_a_plus_2b_l1804_180455


namespace NUMINAMATH_CALUDE_ratio_sum_equation_solver_l1804_180404

theorem ratio_sum_equation_solver (x y z a : ℚ) : 
  (∃ k : ℚ, x = 3 * k ∧ y = 4 * k ∧ z = 7 * k) →
  y = 15 * a - 5 →
  x + y + z = 70 →
  a = 5 / 3 := by
sorry

end NUMINAMATH_CALUDE_ratio_sum_equation_solver_l1804_180404


namespace NUMINAMATH_CALUDE_cos_96_cos_24_minus_sin_96_sin_24_l1804_180476

theorem cos_96_cos_24_minus_sin_96_sin_24 :
  Real.cos (96 * π / 180) * Real.cos (24 * π / 180) - 
  Real.sin (96 * π / 180) * Real.sin (24 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_96_cos_24_minus_sin_96_sin_24_l1804_180476


namespace NUMINAMATH_CALUDE_range_of_a_proof_l1804_180401

/-- Proposition p: there exists a real x₀ such that x₀² + 2ax₀ - 2a = 0 -/
def p (a : ℝ) : Prop := ∃ x₀ : ℝ, x₀^2 + 2*a*x₀ - 2*a = 0

/-- Proposition q: for all real x, ax² + 4x + a > -2x² + 1 -/
def q (a : ℝ) : Prop := ∀ x : ℝ, a*x^2 + 4*x + a > -2*x^2 + 1

/-- The range of a given the conditions -/
def range_of_a : Set ℝ := {a : ℝ | a ≤ -2}

theorem range_of_a_proof (h1 : ∀ a : ℝ, p a ∨ q a) (h2 : ∀ a : ℝ, ¬(p a ∧ q a)) :
  ∀ a : ℝ, a ∈ range_of_a ↔ (p a ∧ ¬q a) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_proof_l1804_180401


namespace NUMINAMATH_CALUDE_isosceles_triangle_vertex_angle_l1804_180429

-- Define an isosceles triangle
structure IsoscelesTriangle where
  -- We don't need to define all properties of an isosceles triangle,
  -- just the ones relevant to our problem
  vertex_angle : ℝ
  base_angle : ℝ
  is_valid : vertex_angle + 2 * base_angle = 180

-- Define our theorem
theorem isosceles_triangle_vertex_angle 
  (triangle : IsoscelesTriangle) 
  (h : triangle.vertex_angle = 70 ∨ triangle.base_angle = 70) : 
  triangle.vertex_angle = 40 ∨ triangle.vertex_angle = 70 := by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_vertex_angle_l1804_180429


namespace NUMINAMATH_CALUDE_expression_range_l1804_180478

def expression_value (parenthesization : List (List Nat)) : ℚ :=
  sorry

theorem expression_range :
  ∀ p : List (List Nat),
    (∀ n, n ∈ p.join → n ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9]) →
    (∀ n, n ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9] → n ∈ p.join) →
    1 / 362880 ≤ expression_value p ∧ expression_value p ≤ 181440 :=
  sorry

end NUMINAMATH_CALUDE_expression_range_l1804_180478


namespace NUMINAMATH_CALUDE_cube_root_simplification_l1804_180463

theorem cube_root_simplification :
  (20^3 + 30^3 + 40^3 + 60^3 : ℝ)^(1/3) = 10 * 315^(1/3) :=
by sorry

end NUMINAMATH_CALUDE_cube_root_simplification_l1804_180463


namespace NUMINAMATH_CALUDE_even_function_implies_a_equals_one_l1804_180466

/-- A function f is even if f(x) = f(-x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

/-- The function y = (x+1)(x-a) -/
def f (a : ℝ) (x : ℝ) : ℝ := (x + 1) * (x - a)

theorem even_function_implies_a_equals_one :
  ∃ a : ℝ, IsEven (f a) → a = 1 := by sorry

end NUMINAMATH_CALUDE_even_function_implies_a_equals_one_l1804_180466


namespace NUMINAMATH_CALUDE_shaded_area_theorem_l1804_180449

/-- Represents a rectangle with diagonals divided into 12 equal segments -/
structure DividedRectangle where
  blank_area : ℝ
  total_area : ℝ

/-- The theorem stating the relationship between blank and shaded areas -/
theorem shaded_area_theorem (rect : DividedRectangle) 
  (h1 : rect.blank_area = 10) 
  (h2 : rect.total_area = rect.blank_area + 14) : 
  rect.total_area - rect.blank_area = 14 := by
  sorry

#check shaded_area_theorem

end NUMINAMATH_CALUDE_shaded_area_theorem_l1804_180449


namespace NUMINAMATH_CALUDE_root_sum_transformation_l1804_180420

theorem root_sum_transformation (α β γ : ℂ) : 
  (x^3 - x + 1 = 0 ↔ x = α ∨ x = β ∨ x = γ) →
  (1 - α) / (1 + α) + (1 - β) / (1 + β) + (1 - γ) / (1 + γ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_transformation_l1804_180420


namespace NUMINAMATH_CALUDE_alpha_square_greater_beta_square_l1804_180437

theorem alpha_square_greater_beta_square 
  (α β : ℝ) 
  (h1 : α ∈ Set.Icc (-Real.pi/2) (Real.pi/2))
  (h2 : β ∈ Set.Icc (-Real.pi/2) (Real.pi/2))
  (h3 : α * Real.sin α - β * Real.sin β > 0) : 
  α^2 > β^2 := by
sorry

end NUMINAMATH_CALUDE_alpha_square_greater_beta_square_l1804_180437


namespace NUMINAMATH_CALUDE_bowling_ball_weight_is_correct_l1804_180495

/-- The weight of one bowling ball in pounds -/
def bowling_ball_weight : ℝ := 18.75

/-- The weight of one canoe in pounds -/
def canoe_weight : ℝ := 30

/-- Theorem stating that the weight of one bowling ball is 18.75 pounds -/
theorem bowling_ball_weight_is_correct : bowling_ball_weight = 18.75 := by
  -- Define the relationship between bowling balls and canoes
  have h1 : 8 * bowling_ball_weight = 5 * canoe_weight := by sorry
  
  -- Define the relationship between canoes and their total weight
  have h2 : 4 * canoe_weight = 120 := by sorry
  
  -- Prove that the bowling ball weight is correct
  sorry

#eval bowling_ball_weight

end NUMINAMATH_CALUDE_bowling_ball_weight_is_correct_l1804_180495


namespace NUMINAMATH_CALUDE_opposite_unit_vector_l1804_180467

/-- Given a vector a = (-3, 4), prove that the unit vector a₀ in the opposite direction of a has coordinates (3/5, -4/5). -/
theorem opposite_unit_vector (a : ℝ × ℝ) (h : a = (-3, 4)) :
  let a₀ := (-(a.1) / Real.sqrt ((a.1)^2 + (a.2)^2), -(a.2) / Real.sqrt ((a.1)^2 + (a.2)^2))
  a₀ = (3/5, -4/5) := by
sorry


end NUMINAMATH_CALUDE_opposite_unit_vector_l1804_180467


namespace NUMINAMATH_CALUDE_field_trip_total_cost_l1804_180445

/-- Calculates the total cost of a field trip for multiple classes --/
def field_trip_cost (num_classes : ℕ) (students_per_class : ℕ) (adults_per_class : ℕ) 
                    (student_fee : ℚ) (adult_fee : ℚ) : ℚ :=
  let total_students := num_classes * students_per_class
  let total_adults := num_classes * adults_per_class
  (total_students : ℚ) * student_fee + (total_adults : ℚ) * adult_fee

/-- Theorem stating the total cost of the field trip --/
theorem field_trip_total_cost : 
  field_trip_cost 4 40 5 (11/2) (13/2) = 1010 := by
  sorry

#eval field_trip_cost 4 40 5 (11/2) (13/2)

end NUMINAMATH_CALUDE_field_trip_total_cost_l1804_180445


namespace NUMINAMATH_CALUDE_f_properties_l1804_180441

def f (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x

theorem f_properties :
  (∃ (x : ℝ), -2 < x ∧ x < 2 ∧ f x = 5) ∧
  (∀ (y : ℝ), -2 < y ∧ y < 2 → f y ≤ 5) ∧
  (¬ ∃ (z : ℝ), -2 < z ∧ z < 2 ∧ ∀ (w : ℝ), -2 < w ∧ w < 2 → f z ≤ f w) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1804_180441


namespace NUMINAMATH_CALUDE_mikes_house_payments_l1804_180462

theorem mikes_house_payments (lower_rate higher_rate total_payments num_lower_payments num_higher_payments : ℚ) :
  higher_rate = 310 →
  total_payments = 3615 →
  num_lower_payments = 5 →
  num_higher_payments = 7 →
  num_lower_payments + num_higher_payments = 12 →
  num_lower_payments * lower_rate + num_higher_payments * higher_rate = total_payments →
  lower_rate = 289 := by
sorry

end NUMINAMATH_CALUDE_mikes_house_payments_l1804_180462


namespace NUMINAMATH_CALUDE_sequence_gcd_theorem_l1804_180482

theorem sequence_gcd_theorem (d m : ℕ) (hd : d > 1) :
  ∃ k l : ℕ, k ≠ l ∧ Nat.gcd (2^(2^k) + d) (2^(2^l) + d) > m := by
  sorry

end NUMINAMATH_CALUDE_sequence_gcd_theorem_l1804_180482


namespace NUMINAMATH_CALUDE_number_of_topics_six_students_three_groups_ninety_arrangements_l1804_180471

theorem number_of_topics (num_students : Nat) (num_groups : Nat) (num_arrangements : Nat) : Nat :=
  let students_per_group := num_students / num_groups
  let ways_to_divide := num_arrangements / (num_groups^students_per_group)
  ways_to_divide

theorem six_students_three_groups_ninety_arrangements :
  number_of_topics 6 3 90 = 1 := by sorry

end NUMINAMATH_CALUDE_number_of_topics_six_students_three_groups_ninety_arrangements_l1804_180471


namespace NUMINAMATH_CALUDE_at_least_one_less_than_or_equal_to_one_l1804_180473

theorem at_least_one_less_than_or_equal_to_one 
  (x y z : ℝ) 
  (pos_x : 0 < x) 
  (pos_y : 0 < y) 
  (pos_z : 0 < z) 
  (sum_eq_three : x + y + z = 3) :
  (x * (x + y - z) ≤ 1) ∨ (y * (y + z - x) ≤ 1) ∨ (z * (z + x - y) ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_less_than_or_equal_to_one_l1804_180473


namespace NUMINAMATH_CALUDE_range_of_m_l1804_180414

theorem range_of_m : ∀ m : ℝ,
  (∃ x y : ℝ, x < 0 ∧ y < 0 ∧ x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0 ↔ m > 2) →
  (¬∃ x : ℝ, 4*x^2 + 4*(m - 2)*x + 1 = 0 ↔ 1 < m ∧ m < 3) →
  ((m > 2 ∨ (1 < m ∧ m < 3)) ∧ ¬(m > 2 ∧ 1 < m ∧ m < 3)) →
  m ≥ 3 ∨ (1 < m ∧ m ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l1804_180414


namespace NUMINAMATH_CALUDE_problem_solution_l1804_180413

theorem problem_solution (a b m : ℝ) 
  (h1 : 2^a = m) 
  (h2 : 5^b = m) 
  (h3 : 1/a + 1/b = 2) : 
  m = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1804_180413


namespace NUMINAMATH_CALUDE_basketball_team_starters_count_l1804_180451

def total_players : ℕ := 18
def num_triplets : ℕ := 3
def num_twins : ℕ := 2
def num_starters : ℕ := 7
def triplets_in_lineup : ℕ := 2
def twins_in_lineup : ℕ := 1

def remaining_players : ℕ := total_players - num_triplets - num_twins

theorem basketball_team_starters_count :
  (Nat.choose num_triplets triplets_in_lineup) *
  (Nat.choose num_twins twins_in_lineup) *
  (Nat.choose remaining_players (num_starters - triplets_in_lineup - twins_in_lineup)) = 4290 := by
  sorry

end NUMINAMATH_CALUDE_basketball_team_starters_count_l1804_180451


namespace NUMINAMATH_CALUDE_lineup_calculation_1_lineup_calculation_2_l1804_180497

/-- Represents a basketball team -/
structure BasketballTeam where
  veterans : Nat
  newPlayers : Nat

/-- Represents the conditions for lineup selection -/
structure LineupConditions where
  specificVeteranMustPlay : Bool
  specificNewPlayersCannotPlay : Nat
  forwardPlayers : Nat
  guardPlayers : Nat
  versatilePlayers : Nat

/-- Calculates the number of different lineups under given conditions -/
def calculateLineups (team : BasketballTeam) (conditions : LineupConditions) : Nat :=
  sorry

/-- Theorem for the first lineup calculation -/
theorem lineup_calculation_1 (team : BasketballTeam) (conditions : LineupConditions) :
  team.veterans = 7 ∧ team.newPlayers = 5 ∧
  conditions.specificVeteranMustPlay = true ∧
  conditions.specificNewPlayersCannotPlay = 2 →
  calculateLineups team conditions = 126 :=
sorry

/-- Theorem for the second lineup calculation -/
theorem lineup_calculation_2 (team : BasketballTeam) (conditions : LineupConditions) :
  team.veterans + team.newPlayers = 12 ∧
  conditions.forwardPlayers = 6 ∧
  conditions.guardPlayers = 4 ∧
  conditions.versatilePlayers = 2 →
  calculateLineups team conditions = 636 :=
sorry

end NUMINAMATH_CALUDE_lineup_calculation_1_lineup_calculation_2_l1804_180497


namespace NUMINAMATH_CALUDE_difference_calculation_l1804_180474

theorem difference_calculation : 
  (1 / 10 : ℚ) * 8000 - (1 / 20 : ℚ) / 100 * 8000 = 796 := by
  sorry

end NUMINAMATH_CALUDE_difference_calculation_l1804_180474


namespace NUMINAMATH_CALUDE_missing_number_is_eight_l1804_180400

/-- Represents a 1 × 12 table filled with numbers -/
def Table := Fin 12 → ℝ

/-- The sum of any four adjacent cells in the table is 11 -/
def SumAdjacent (t : Table) : Prop :=
  ∀ i : Fin 9, t i + t (i + 1) + t (i + 2) + t (i + 3) = 11

/-- The table contains the known numbers 4, 1, and 2 -/
def ContainsKnownNumbers (t : Table) : Prop :=
  ∃ (i j k : Fin 12), t i = 4 ∧ t j = 1 ∧ t k = 2

/-- The theorem to be proved -/
theorem missing_number_is_eight
  (t : Table)
  (h1 : SumAdjacent t)
  (h2 : ContainsKnownNumbers t) :
  ∃ (l : Fin 12), t l = 8 :=
sorry

end NUMINAMATH_CALUDE_missing_number_is_eight_l1804_180400


namespace NUMINAMATH_CALUDE_inequality_proof_l1804_180458

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (1 / (a^3 * (b + c))) + (1 / (b^3 * (a + c))) + (1 / (c^3 * (a + b))) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1804_180458


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l1804_180410

/-- Two 2D vectors are parallel if their corresponding components are proportional -/
def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem parallel_vectors_m_value (m : ℝ) :
  let a : ℝ × ℝ := (-1, 1)
  let b : ℝ × ℝ := (3, m)
  parallel a b → m = -3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l1804_180410


namespace NUMINAMATH_CALUDE_truth_telling_probability_l1804_180470

theorem truth_telling_probability (prob_A prob_B : ℝ) 
  (h_A : prob_A = 0.85) 
  (h_B : prob_B = 0.60) : 
  prob_A * prob_B = 0.51 := by
  sorry

end NUMINAMATH_CALUDE_truth_telling_probability_l1804_180470


namespace NUMINAMATH_CALUDE_canoe_upstream_speed_l1804_180411

/-- Proves that the upstream speed of a canoe is 9 km/hr given its downstream speed and the stream speed -/
theorem canoe_upstream_speed 
  (downstream_speed : ℝ) 
  (stream_speed : ℝ) 
  (h1 : downstream_speed = 12) 
  (h2 : stream_speed = 1.5) : 
  downstream_speed - 2 * stream_speed = 9 := by
  sorry

end NUMINAMATH_CALUDE_canoe_upstream_speed_l1804_180411


namespace NUMINAMATH_CALUDE_scientific_notation_218_million_l1804_180427

theorem scientific_notation_218_million :
  ∃ (a : ℝ) (n : ℤ), 
    1 ≤ a ∧ a < 10 ∧ 
    218000000 = a * (10 : ℝ) ^ n ∧
    a = 2.18 ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_218_million_l1804_180427


namespace NUMINAMATH_CALUDE_jake_snake_revenue_l1804_180403

/-- Calculates the total revenue from selling baby snakes --/
def total_revenue (num_snakes : ℕ) (eggs_per_snake : ℕ) (regular_price : ℕ) (rare_price_multiplier : ℕ) : ℕ :=
  let total_babies := num_snakes * eggs_per_snake
  let num_regular_babies := total_babies - 1
  let rare_price := regular_price * rare_price_multiplier
  num_regular_babies * regular_price + rare_price

/-- The revenue from Jake's snake business --/
theorem jake_snake_revenue :
  total_revenue 3 2 250 4 = 2250 := by
  sorry

end NUMINAMATH_CALUDE_jake_snake_revenue_l1804_180403


namespace NUMINAMATH_CALUDE_min_value_ab_l1804_180434

/-- Given that ab > 0 and points A(a,0), B(0,b), and C(-2,-2) are collinear, 
    the minimum value of ab is 16 -/
theorem min_value_ab (a b : ℝ) (hab : a * b > 0) 
    (hcollinear : (0 - a) * (b + 2) = (b - 0) * (0 + 2)) : 
  ∀ x y : ℝ, x * y > 0 ∧ 
    (0 - x) * (y + 2) = (y - 0) * (0 + 2) → 
    a * b ≤ x * y ∧ a * b = 16 := by
  sorry

end NUMINAMATH_CALUDE_min_value_ab_l1804_180434


namespace NUMINAMATH_CALUDE_car_travel_distance_l1804_180405

/-- Proves that a car traveling at 70 kmh for a certain time covers a distance of 105 km,
    given that if it had traveled 35 kmh faster, the trip would have lasted 30 minutes less. -/
theorem car_travel_distance :
  ∀ (time : ℝ),
  time > 0 →
  let distance := 70 * time
  let faster_time := time - 0.5
  let faster_speed := 70 + 35
  distance = faster_speed * faster_time →
  distance = 105 :=
by
  sorry

end NUMINAMATH_CALUDE_car_travel_distance_l1804_180405


namespace NUMINAMATH_CALUDE_pet_store_birds_l1804_180444

/-- The number of bird cages in the pet store -/
def num_cages : ℕ := 6

/-- The number of parrots in each cage -/
def parrots_per_cage : ℕ := 6

/-- The number of parakeets in each cage -/
def parakeets_per_cage : ℕ := 2

/-- The total number of birds in the pet store -/
def total_birds : ℕ := num_cages * (parrots_per_cage + parakeets_per_cage)

theorem pet_store_birds : total_birds = 48 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_birds_l1804_180444


namespace NUMINAMATH_CALUDE_simplify_expression_1_l1804_180492

theorem simplify_expression_1 (x : ℝ) : 
  5*x^2 + x + 3 + 4*x - 8*x^2 - 2 = -3*x^2 + 5*x + 1 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_l1804_180492


namespace NUMINAMATH_CALUDE_range_of_dot_product_trajectory_of_P_l1804_180421

noncomputable section

-- Define the hyperbola C
def C (x y : ℝ) : Prop := x^2 / 2 - y^2 / 3 = 1

-- Define the foci F₁ and F₂
def F₁ : ℝ × ℝ := (-Real.sqrt 5, 0)
def F₂ : ℝ × ℝ := (Real.sqrt 5, 0)

-- Define a point M on the right branch of C
def M (x y : ℝ) : Prop := C x y ∧ x ≥ Real.sqrt 2

-- Define the dot product of vectors
def dot_product (v₁ v₂ : ℝ × ℝ) : ℝ :=
  v₁.1 * v₂.1 + v₁.2 * v₂.2

-- Theorem 1: Range of OM · F₁M
theorem range_of_dot_product (x y : ℝ) :
  M x y → dot_product (x, y) (x + F₁.1, y + F₁.2) ≥ 2 + Real.sqrt 10 := by sorry

-- Define a point P with constant sum of distances from F₁ and F₂
def P (x y : ℝ) : Prop :=
  ∃ (k : ℝ), Real.sqrt ((x - F₁.1)^2 + (y - F₁.2)^2) +
             Real.sqrt ((x - F₂.1)^2 + (y - F₂.2)^2) = k

-- Define the cosine of angle F₁PF₂
def cos_F₁PF₂ (x y : ℝ) : ℝ :=
  let d₁ := Real.sqrt ((x - F₁.1)^2 + (y - F₁.2)^2)
  let d₂ := Real.sqrt ((x - F₂.1)^2 + (y - F₂.2)^2)
  ((x - F₁.1) * (x - F₂.1) + (y - F₁.2) * (y - F₂.2)) / (d₁ * d₂)

-- Theorem 2: Trajectory of P
theorem trajectory_of_P (x y : ℝ) :
  P x y ∧ (∀ (u v : ℝ), P u v → cos_F₁PF₂ x y ≤ cos_F₁PF₂ u v) ∧ cos_F₁PF₂ x y = -1/9
  → x^2/9 + y^2/4 = 1 := by sorry

end NUMINAMATH_CALUDE_range_of_dot_product_trajectory_of_P_l1804_180421


namespace NUMINAMATH_CALUDE_plane_relation_l1804_180422

-- Define the concept of a plane
class Plane

-- Define the concept of a line
class Line

-- Define the parallelism relation between planes
def parallel (α β : Plane) : Prop := sorry

-- Define the intersection relation between planes
def intersects (α β : Plane) : Prop := sorry

-- Define the relation of a line being parallel to a plane
def line_parallel_to_plane (l : Line) (β : Plane) : Prop := sorry

-- Define the property of having infinitely many parallel lines
def has_infinitely_many_parallel_lines (α β : Plane) : Prop :=
  ∃ (S : Set Line), Set.Infinite S ∧ ∀ l ∈ S, line_parallel_to_plane l β

-- State the theorem
theorem plane_relation (α β : Plane) :
  has_infinitely_many_parallel_lines α β → parallel α β ∨ intersects α β :=
sorry

end NUMINAMATH_CALUDE_plane_relation_l1804_180422


namespace NUMINAMATH_CALUDE_some_dragons_not_breathe_fire_negates_all_dragons_breathe_fire_l1804_180415

-- Define the universe of discourse
def Dragon : Type := sorry

-- Define the property of breathing fire
def breathes_fire : Dragon → Prop := sorry

-- Theorem: "Some dragons do not breathe fire" is equivalent to 
-- the negation of "All dragons breathe fire"
theorem some_dragons_not_breathe_fire_negates_all_dragons_breathe_fire :
  (∃ d : Dragon, ¬(breathes_fire d)) ↔ ¬(∀ d : Dragon, breathes_fire d) := by
  sorry

end NUMINAMATH_CALUDE_some_dragons_not_breathe_fire_negates_all_dragons_breathe_fire_l1804_180415


namespace NUMINAMATH_CALUDE_blue_balls_count_l1804_180488

theorem blue_balls_count (B : ℕ) : 
  (5 : ℚ) * 4 / (2 * ((7 + B : ℚ) * (6 + B))) = 0.1282051282051282 → B = 6 := by
  sorry

end NUMINAMATH_CALUDE_blue_balls_count_l1804_180488


namespace NUMINAMATH_CALUDE_complement_A_union_B_eq_univ_A_inter_B_ne_B_l1804_180439

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 2}
def B (a : ℝ) : Set ℝ := {x : ℝ | a ≤ x ∧ x ≤ 3 - 2*a}

-- Statement for part 1
theorem complement_A_union_B_eq_univ (a : ℝ) :
  (Set.univ \ A) ∪ B a = Set.univ ↔ a ≤ 0 :=
sorry

-- Statement for part 2
theorem A_inter_B_ne_B (a : ℝ) :
  A ∩ B a ≠ B a ↔ a < 1/2 :=
sorry

end NUMINAMATH_CALUDE_complement_A_union_B_eq_univ_A_inter_B_ne_B_l1804_180439


namespace NUMINAMATH_CALUDE_greatest_valid_sequence_length_l1804_180440

/-- A sequence of distinct positive integers satisfying the given condition -/
def ValidSequence (s : Nat → Nat) (n : Nat) : Prop :=
  (∀ i j, i < n → j < n → i ≠ j → s i ≠ s j) ∧ 
  (∀ i, i < n - 1 → (s i) ^ (s (i + 1)) = (s (i + 1)) ^ (s (i + 2)))

/-- The theorem stating that 5 is the greatest positive integer satisfying the condition -/
theorem greatest_valid_sequence_length : 
  (∃ s : Nat → Nat, ValidSequence s 5) ∧ 
  (∀ n : Nat, n > 5 → ¬∃ s : Nat → Nat, ValidSequence s n) :=
sorry

end NUMINAMATH_CALUDE_greatest_valid_sequence_length_l1804_180440


namespace NUMINAMATH_CALUDE_IMO_2002_problem_l1804_180408

theorem IMO_2002_problem (A B C : ℕ) : 
  A > 0 → B > 0 → C > 0 → 
  A ≠ B → B ≠ C → A ≠ C →
  A * B * C = 2310 →
  (∀ X Y Z : ℕ, X > 0 → Y > 0 → Z > 0 → X ≠ Y → Y ≠ Z → X ≠ Z → X * Y * Z = 2310 → 
    A + B + C ≤ X + Y + Z) →
  (∀ X Y Z : ℕ, X > 0 → Y > 0 → Z > 0 → X ≠ Y → Y ≠ Z → X ≠ Z → X * Y * Z = 2310 → 
    A + B + C ≥ X + Y + Z) →
  A + B + C = 52 ∧ A + B + C = 390 :=
by sorry

end NUMINAMATH_CALUDE_IMO_2002_problem_l1804_180408


namespace NUMINAMATH_CALUDE_largest_number_l1804_180498

theorem largest_number (a b c d : ℝ) (h1 : a = 3) (h2 : b = -7) (h3 : c = 0) (h4 : d = 1/9) :
  a = max a (max b (max c d)) :=
sorry

end NUMINAMATH_CALUDE_largest_number_l1804_180498


namespace NUMINAMATH_CALUDE_systematic_sampling_l1804_180425

theorem systematic_sampling (total_students : Nat) (num_groups : Nat) (first_group_number : Nat) (target_group : Nat) :
  total_students = 480 →
  num_groups = 30 →
  first_group_number = 5 →
  target_group = 8 →
  (target_group - 1) * (total_students / num_groups) + first_group_number = 117 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_l1804_180425


namespace NUMINAMATH_CALUDE_ice_cube_distribution_l1804_180477

theorem ice_cube_distribution (total_cubes : ℕ) (num_chests : ℕ) (cubes_per_chest : ℕ) 
  (h1 : total_cubes = 294)
  (h2 : num_chests = 7)
  (h3 : total_cubes = num_chests * cubes_per_chest) :
  cubes_per_chest = 42 := by
  sorry

end NUMINAMATH_CALUDE_ice_cube_distribution_l1804_180477


namespace NUMINAMATH_CALUDE_square_inequality_for_negatives_l1804_180464

theorem square_inequality_for_negatives (a b : ℝ) (h1 : a < b) (h2 : b < 0) : a^2 > b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_inequality_for_negatives_l1804_180464


namespace NUMINAMATH_CALUDE_number_of_appliances_l1804_180409

/-- Proves that the number of appliances in a batch is 34, given the purchase price,
    selling price, and total profit. -/
theorem number_of_appliances (purchase_price selling_price total_profit : ℕ) : 
  purchase_price = 230 →
  selling_price = 250 →
  total_profit = 680 →
  (total_profit / (selling_price - purchase_price) : ℕ) = 34 :=
by
  sorry

end NUMINAMATH_CALUDE_number_of_appliances_l1804_180409


namespace NUMINAMATH_CALUDE_slope_product_constant_l1804_180442

/-- The trajectory C -/
def C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / 4 + p.2^2 = 1 ∧ p.2 ≠ 0}

/-- The line y = kx -/
def Line (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = k * p.1}

theorem slope_product_constant
  (M : ℝ × ℝ) (h_M : M ∈ C)
  (k : ℝ)
  (A B : ℝ × ℝ) (h_A : A ∈ C ∩ Line k) (h_B : B ∈ C ∩ Line k)
  (h_AB : A.1 = -B.1 ∧ A.2 = -B.2)
  (h_MA : M.1 ≠ A.1) (h_MB : M.1 ≠ B.1) :
  let K_MA := (M.2 - A.2) / (M.1 - A.1)
  let K_MB := (M.2 - B.2) / (M.1 - B.1)
  K_MA * K_MB = -1/4 :=
sorry

end NUMINAMATH_CALUDE_slope_product_constant_l1804_180442


namespace NUMINAMATH_CALUDE_work_completion_time_l1804_180481

-- Define the work rates for a, b, and c
def work_rate_a : ℚ := 1 / 24
def work_rate_b : ℚ := 1 / 30
def work_rate_c : ℚ := 1 / 40

-- Define the combined work rate of a, b, and c
def combined_rate : ℚ := work_rate_a + work_rate_b + work_rate_c

-- Define the combined work rate of a and b
def combined_rate_ab : ℚ := work_rate_a + work_rate_b

-- Define the total days to complete the work
def total_days : ℚ := 11

-- Theorem statement
theorem work_completion_time :
  (total_days - 4) * combined_rate + 4 * combined_rate_ab = 1 :=
sorry

end NUMINAMATH_CALUDE_work_completion_time_l1804_180481


namespace NUMINAMATH_CALUDE_exists_composite_evaluation_l1804_180402

/-- A polynomial with integer coefficients -/
def IntPolynomial := List Int

/-- Evaluate a polynomial at a given integer -/
def evalPoly (p : IntPolynomial) (x : Int) : Int :=
  p.foldr (fun a b => a + x * b) 0

/-- A number is composite if it has a factor other than 1 and itself -/
def isComposite (n : Int) : Prop :=
  ∃ m, 1 < m ∧ m < n.natAbs ∧ n % m = 0

theorem exists_composite_evaluation (polys : List IntPolynomial) :
  ∃ a : Int, ∀ p ∈ polys, isComposite (evalPoly p a) := by
  sorry

#check exists_composite_evaluation

end NUMINAMATH_CALUDE_exists_composite_evaluation_l1804_180402


namespace NUMINAMATH_CALUDE_rachel_homework_l1804_180475

theorem rachel_homework (math_pages reading_pages : ℕ) : 
  math_pages = 3 → reading_pages = math_pages + 1 → reading_pages = 4 := by
  sorry

end NUMINAMATH_CALUDE_rachel_homework_l1804_180475


namespace NUMINAMATH_CALUDE_milk_mixture_problem_l1804_180485

/-- Proves that the butterfat percentage of the added milk is 10% given the conditions of the problem -/
theorem milk_mixture_problem (final_percentage : ℝ) (initial_volume : ℝ) (initial_percentage : ℝ) (added_volume : ℝ) :
  final_percentage = 20 →
  initial_volume = 8 →
  initial_percentage = 30 →
  added_volume = 8 →
  (initial_volume * initial_percentage + added_volume * (100 * final_percentage - initial_volume * initial_percentage) / added_volume) / (initial_volume + added_volume) = 10 := by
sorry

end NUMINAMATH_CALUDE_milk_mixture_problem_l1804_180485


namespace NUMINAMATH_CALUDE_custom_op_result_l1804_180426

/-- Custom operation * for non-zero integers -/
def custom_op (a b : ℤ) : ℚ := (a : ℚ)⁻¹ + (b : ℚ)⁻¹

/-- Theorem stating the result of the custom operation given specific conditions -/
theorem custom_op_result (a b : ℤ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a + b = 11) (h4 : a * b = 24) :
  custom_op a b = 11 / 24 := by
  sorry

#check custom_op_result

end NUMINAMATH_CALUDE_custom_op_result_l1804_180426


namespace NUMINAMATH_CALUDE_greatest_multiple_of_four_cubed_less_than_2000_l1804_180412

theorem greatest_multiple_of_four_cubed_less_than_2000 :
  ∃ (x : ℕ), x % 4 = 0 ∧ x^3 < 2000 ∧ ∀ (y : ℕ), y % 4 = 0 ∧ y^3 < 2000 → y ≤ x :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_four_cubed_less_than_2000_l1804_180412


namespace NUMINAMATH_CALUDE_f_derivative_at_2_l1804_180465

def f (x : ℝ) : ℝ := (x + 3) * (x + 2) * (x + 1) * x * (x - 1) * (x - 2) * (x - 3)

theorem f_derivative_at_2 : 
  (deriv f) 2 = -120 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_2_l1804_180465


namespace NUMINAMATH_CALUDE_pyramid_ball_count_l1804_180433

/-- The number of layers in the pyramid -/
def n : ℕ := 13

/-- The number of balls in the top layer -/
def first_term : ℕ := 4

/-- The number of balls in the bottom layer -/
def last_term : ℕ := 40

/-- The sum of the arithmetic sequence representing the number of balls in each layer -/
def sum_of_sequence : ℕ := n * (first_term + last_term) / 2

theorem pyramid_ball_count :
  sum_of_sequence = 286 := by sorry

end NUMINAMATH_CALUDE_pyramid_ball_count_l1804_180433


namespace NUMINAMATH_CALUDE_complement_intersection_A_B_l1804_180453

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 3, 4}
def B : Set Nat := {3, 5}

theorem complement_intersection_A_B :
  (A ∩ B)ᶜ = {1, 2, 4, 5} :=
by sorry

end NUMINAMATH_CALUDE_complement_intersection_A_B_l1804_180453


namespace NUMINAMATH_CALUDE_isosceles_60_similar_l1804_180435

/-- An isosceles triangle with a 60° interior angle -/
structure IsoscelesTriangle60 where
  -- We represent the triangle by its three angles
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  -- The triangle is isosceles
  isIsosceles : (angle1 = angle2) ∨ (angle1 = angle3) ∨ (angle2 = angle3)
  -- One of the angles is 60°
  has60Degree : angle1 = 60 ∨ angle2 = 60 ∨ angle3 = 60
  -- The sum of angles in a triangle is 180°
  sumIs180 : angle1 + angle2 + angle3 = 180

/-- Two isosceles triangles with a 60° interior angle are similar -/
theorem isosceles_60_similar (t1 t2 : IsoscelesTriangle60) : 
  t1.angle1 = t2.angle1 ∧ t1.angle2 = t2.angle2 ∧ t1.angle3 = t2.angle3 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_60_similar_l1804_180435


namespace NUMINAMATH_CALUDE_steve_socks_count_l1804_180460

/-- The number of pairs of matching socks Steve has -/
def matching_pairs : ℕ := 4

/-- The number of mismatching socks Steve has -/
def mismatching_socks : ℕ := 17

/-- The total number of socks Steve has -/
def total_socks : ℕ := 2 * matching_pairs + mismatching_socks

theorem steve_socks_count : total_socks = 25 := by
  sorry

end NUMINAMATH_CALUDE_steve_socks_count_l1804_180460


namespace NUMINAMATH_CALUDE_pen_pencil_ratio_l1804_180438

/-- Given 54 pencils and 9 more pencils than pens, prove that the ratio of pens to pencils is 5:6 -/
theorem pen_pencil_ratio : 
  ∀ (num_pens num_pencils : ℕ), 
  num_pencils = 54 → 
  num_pencils = num_pens + 9 → 
  (num_pens : ℚ) / (num_pencils : ℚ) = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_pen_pencil_ratio_l1804_180438


namespace NUMINAMATH_CALUDE_simplify_expression_1_expand_expression_2_l1804_180416

-- First expression
theorem simplify_expression_1 (x y : ℝ) (h : y ≠ 0) :
  (3 * x^2 * y - 6 * x * y) / (3 * x * y) = x - 2 := by sorry

-- Second expression
theorem expand_expression_2 (a b : ℝ) :
  (a + b + 2) * (a + b - 2) = a^2 + 2*a*b + b^2 - 4 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_expand_expression_2_l1804_180416


namespace NUMINAMATH_CALUDE_product_is_term_iff_first_term_is_power_of_ratio_l1804_180423

/-- A geometric progression is defined by its first term and common ratio -/
structure GeometricProgression where
  a : ℝ  -- First term
  q : ℝ  -- Common ratio

/-- The nth term of a geometric progression -/
def GeometricProgression.nthTerm (gp : GeometricProgression) (n : ℕ) : ℝ :=
  gp.a * gp.q ^ n

/-- Condition for product of terms to be another term -/
def productIsTermCondition (gp : GeometricProgression) : Prop :=
  ∃ m : ℤ, gp.a = gp.q ^ m

theorem product_is_term_iff_first_term_is_power_of_ratio (gp : GeometricProgression) :
  (∀ n p k : ℕ, ∃ k : ℕ, gp.nthTerm n * gp.nthTerm p = gp.nthTerm k) ↔
  productIsTermCondition gp :=
sorry

end NUMINAMATH_CALUDE_product_is_term_iff_first_term_is_power_of_ratio_l1804_180423


namespace NUMINAMATH_CALUDE_smallest_side_difference_l1804_180484

theorem smallest_side_difference (P Q R : ℕ) : 
  P + Q + R = 2021 →  -- Perimeter condition
  P < Q →             -- PQ < PR
  Q ≤ R →             -- PR ≤ QR
  P + R > Q →         -- Triangle inequality
  P + Q > R →         -- Triangle inequality
  Q + R > P →         -- Triangle inequality
  (∀ P' Q' R' : ℕ, 
    P' + Q' + R' = 2021 → 
    P' < Q' → 
    Q' ≤ R' → 
    P' + R' > Q' → 
    P' + Q' > R' → 
    Q' + R' > P' → 
    Q' - P' ≥ Q - P) →
  Q - P = 1 := by
sorry

end NUMINAMATH_CALUDE_smallest_side_difference_l1804_180484


namespace NUMINAMATH_CALUDE_ln_sufficient_not_necessary_l1804_180457

-- Define the statement that ln a > ln b implies e^a > e^b
def ln_implies_exp (a b : ℝ) : Prop :=
  (∀ a b : ℝ, Real.log a > Real.log b → Real.exp a > Real.exp b)

-- Define the statement that e^a > e^b does not always imply ln a > ln b
def exp_not_always_implies_ln (a b : ℝ) : Prop :=
  (∃ a b : ℝ, Real.exp a > Real.exp b ∧ ¬(Real.log a > Real.log b))

-- Theorem stating that ln a > ln b is sufficient but not necessary for e^a > e^b
theorem ln_sufficient_not_necessary :
  (∀ a b : ℝ, ln_implies_exp a b) ∧ (∃ a b : ℝ, exp_not_always_implies_ln a b) :=
sorry

end NUMINAMATH_CALUDE_ln_sufficient_not_necessary_l1804_180457


namespace NUMINAMATH_CALUDE_sequence_convergence_l1804_180417

def is_smallest_prime_divisor (p n : ℕ) : Prop :=
  Nat.Prime p ∧ p ∣ n ∧ ∀ q, Nat.Prime q → q ∣ n → p ≤ q

def sequence_condition (a p : ℕ → ℕ) : Prop :=
  (∀ n, a n > 0 ∧ p n > 0) ∧
  a 1 ≥ 2 ∧
  (∀ n, is_smallest_prime_divisor (p n) (a n)) ∧
  (∀ n, a (n + 1) = a n + a n / p n)

theorem sequence_convergence (a p : ℕ → ℕ) (h : sequence_condition a p) :
  ∃ N, ∀ n > N, a (n + 3) = 3 * a n := by
  sorry

end NUMINAMATH_CALUDE_sequence_convergence_l1804_180417


namespace NUMINAMATH_CALUDE_sanya_towels_per_wash_l1804_180419

/-- The number of bath towels Sanya can wash in one wash -/
def towels_per_wash : ℕ := sorry

/-- The number of hours Sanya has per day for washing -/
def hours_per_day : ℕ := 2

/-- The total number of bath towels Sanya has -/
def total_towels : ℕ := 98

/-- The number of days it takes to wash all towels -/
def days_to_wash_all : ℕ := 7

/-- Theorem stating that Sanya can wash 7 towels in one wash -/
theorem sanya_towels_per_wash :
  towels_per_wash = 7 := by sorry

end NUMINAMATH_CALUDE_sanya_towels_per_wash_l1804_180419


namespace NUMINAMATH_CALUDE_celebrity_baby_matching_probability_l1804_180418

theorem celebrity_baby_matching_probability :
  let n : ℕ := 4
  let total_arrangements := n.factorial
  let correct_arrangements : ℕ := 1
  (correct_arrangements : ℚ) / total_arrangements = 1 / 24 :=
by sorry

end NUMINAMATH_CALUDE_celebrity_baby_matching_probability_l1804_180418


namespace NUMINAMATH_CALUDE_sine_of_angle_l1804_180454

theorem sine_of_angle (α : Real) (m : Real) (h1 : m ≠ 0) 
  (h2 : Real.sqrt 3 / Real.sqrt (3 + m^2) = m / 6) 
  (h3 : Real.cos α = m / 6) : 
  Real.sin α = Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_sine_of_angle_l1804_180454


namespace NUMINAMATH_CALUDE_average_first_15_even_numbers_l1804_180486

theorem average_first_15_even_numbers : 
  let first_15_even : List ℕ := List.range 15 |>.map (fun n => 2 * (n + 1))
  (first_15_even.sum / first_15_even.length : ℚ) = 16 := by
  sorry

end NUMINAMATH_CALUDE_average_first_15_even_numbers_l1804_180486


namespace NUMINAMATH_CALUDE_initial_investment_interest_rate_l1804_180469

/-- Proves that the interest rate of the initial investment is 5% given the problem conditions --/
theorem initial_investment_interest_rate
  (initial_investment : ℝ)
  (additional_investment : ℝ)
  (additional_rate : ℝ)
  (total_rate : ℝ)
  (h1 : initial_investment = 2000)
  (h2 : additional_investment = 1000)
  (h3 : additional_rate = 0.08)
  (h4 : total_rate = 0.06)
  (h5 : ∃ r : ℝ, r * initial_investment + additional_rate * additional_investment = 
        total_rate * (initial_investment + additional_investment)) :
  ∃ r : ℝ, r = 0.05 :=
sorry

end NUMINAMATH_CALUDE_initial_investment_interest_rate_l1804_180469


namespace NUMINAMATH_CALUDE_odd_function_negative_l1804_180448

/-- An odd function f with a specific definition for non-negative x -/
def odd_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ (∀ x ≥ 0, f x = x * (1 - x))

/-- Theorem stating the form of f(x) for non-positive x -/
theorem odd_function_negative (f : ℝ → ℝ) (h : odd_function f) :
  ∀ x ≤ 0, f x = x * (1 + x) := by
  sorry

end NUMINAMATH_CALUDE_odd_function_negative_l1804_180448


namespace NUMINAMATH_CALUDE_parts_from_64_blanks_l1804_180468

/-- Calculates the total number of parts that can be produced from a given number of initial blanks,
    where shavings from a certain number of parts can be remelted into one new blank. -/
def total_parts (initial_blanks : ℕ) (parts_per_remelted_blank : ℕ) : ℕ :=
  let first_batch := initial_blanks
  let second_batch := initial_blanks / parts_per_remelted_blank
  let third_batch := second_batch / parts_per_remelted_blank
  first_batch + second_batch + third_batch

/-- Theorem stating that given 64 initial blanks and the ability to remelt shavings 
    from 8 parts into one new blank, the total number of parts that can be produced is 73. -/
theorem parts_from_64_blanks : total_parts 64 8 = 73 := by
  sorry

end NUMINAMATH_CALUDE_parts_from_64_blanks_l1804_180468


namespace NUMINAMATH_CALUDE_y_greater_than_one_l1804_180461

theorem y_greater_than_one (x y : ℝ) (h1 : x^3 > y^2) (h2 : y^3 > x^2) : y > 1 := by
  sorry

end NUMINAMATH_CALUDE_y_greater_than_one_l1804_180461


namespace NUMINAMATH_CALUDE_largest_common_divisor_of_difference_of_squares_l1804_180493

theorem largest_common_divisor_of_difference_of_squares (m n : ℤ) 
  (h_m_even : Even m) (h_n_odd : Odd n) (h_n_lt_m : n < m) :
  (∀ k : ℤ, k ∣ (m^2 - n^2) → k ≤ 2) ∧ 2 ∣ (m^2 - n^2) := by
  sorry

#check largest_common_divisor_of_difference_of_squares

end NUMINAMATH_CALUDE_largest_common_divisor_of_difference_of_squares_l1804_180493

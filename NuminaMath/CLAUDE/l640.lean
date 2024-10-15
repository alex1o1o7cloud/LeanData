import Mathlib

namespace NUMINAMATH_CALUDE_proposition_uses_or_l640_64051

-- Define the equation
def equation (x : ℝ) : Prop := x^2 = 4

-- Define the solution set
def solution_set : Set ℝ := {2, -2}

-- Define the proposition
def proposition : Prop := ∀ x, equation x ↔ x ∈ solution_set

-- Theorem: The proposition uses the "or" conjunction
theorem proposition_uses_or : 
  (∀ x, equation x ↔ (x = 2 ∨ x = -2)) ↔ proposition := by sorry

end NUMINAMATH_CALUDE_proposition_uses_or_l640_64051


namespace NUMINAMATH_CALUDE_intersection_when_a_is_4_subset_condition_l640_64097

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | a + 1 ≤ x ∧ x ≤ 2 * a - 1}
def B : Set ℝ := {x | x ≤ 3 ∨ x > 5}

-- Theorem 1: When a = 4, A ∩ B = {6, 7}
theorem intersection_when_a_is_4 : A 4 ∩ B = {6, 7} := by sorry

-- Theorem 2: A ⊆ B if and only if a < 2 or a > 4
theorem subset_condition (a : ℝ) : A a ⊆ B ↔ a < 2 ∨ a > 4 := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_is_4_subset_condition_l640_64097


namespace NUMINAMATH_CALUDE_bobby_candy_problem_l640_64025

theorem bobby_candy_problem (initial : ℕ) :
  initial + 17 = 43 → initial = 26 := by
  sorry

end NUMINAMATH_CALUDE_bobby_candy_problem_l640_64025


namespace NUMINAMATH_CALUDE_worksheets_turned_in_l640_64001

/-- 
Given:
- initial_worksheets: The initial number of worksheets to grade
- graded_worksheets: The number of worksheets graded
- final_worksheets: The final number of worksheets to grade

Prove that the number of worksheets turned in after grading is 36.
-/
theorem worksheets_turned_in 
  (initial_worksheets : ℕ) 
  (graded_worksheets : ℕ) 
  (final_worksheets : ℕ) 
  (h1 : initial_worksheets = 34)
  (h2 : graded_worksheets = 7)
  (h3 : final_worksheets = 63) :
  final_worksheets - (initial_worksheets - graded_worksheets) = 36 := by
  sorry

end NUMINAMATH_CALUDE_worksheets_turned_in_l640_64001


namespace NUMINAMATH_CALUDE_glass_bowl_selling_price_l640_64092

theorem glass_bowl_selling_price 
  (total_bowls : ℕ) 
  (cost_per_bowl : ℚ) 
  (bowls_sold : ℕ) 
  (percentage_gain : ℚ) 
  (h1 : total_bowls = 118) 
  (h2 : cost_per_bowl = 12) 
  (h3 : bowls_sold = 102) 
  (h4 : percentage_gain = 8050847457627118 / 100000000000000000) : 
  ∃ (selling_price : ℚ), selling_price = 15 ∧ 
  (total_bowls * cost_per_bowl * (1 + percentage_gain) / bowls_sold).floor = selling_price := by
  sorry

#check glass_bowl_selling_price

end NUMINAMATH_CALUDE_glass_bowl_selling_price_l640_64092


namespace NUMINAMATH_CALUDE_sum_of_digits_of_large_number_l640_64080

/-- The sum of the digits of 10^85 - 85 -/
def sumOfDigits : ℕ := 753

/-- The number represented by 10^85 - 85 -/
def largeNumber : ℕ := 10^85 - 85

theorem sum_of_digits_of_large_number :
  (largeNumber.digits 10).sum = sumOfDigits := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_large_number_l640_64080


namespace NUMINAMATH_CALUDE_number_divided_by_three_equals_number_minus_five_l640_64081

theorem number_divided_by_three_equals_number_minus_five : 
  ∃ x : ℚ, x / 3 = x - 5 ∧ x = 15 / 2 := by
  sorry

end NUMINAMATH_CALUDE_number_divided_by_three_equals_number_minus_five_l640_64081


namespace NUMINAMATH_CALUDE_shaded_triangle_area_l640_64056

/-- Given a rectangle with sides 12 units long and a square with sides 4 units long
    placed in one corner of the rectangle, the area of the triangle formed by
    the diagonal of the rectangle and two sides of the rectangle is 54 square units. -/
theorem shaded_triangle_area (rectangle_side : ℝ) (square_side : ℝ) : 
  rectangle_side = 12 →
  square_side = 4 →
  let triangle_base := rectangle_side - (rectangle_side - square_side) * (square_side / rectangle_side)
  let triangle_height := rectangle_side
  (1/2) * triangle_base * triangle_height = 54 := by
sorry

end NUMINAMATH_CALUDE_shaded_triangle_area_l640_64056


namespace NUMINAMATH_CALUDE_inverse_square_relation_l640_64071

/-- Given that x varies inversely as the square of y, and y = 2 when x = 1,
    prove that x = 1/9 when y = 6 -/
theorem inverse_square_relation (x y : ℝ) (k : ℝ) (h1 : x = k / y^2) 
    (h2 : 1 = k / 2^2) : 
    (y = 6) → (x = 1/9) := by
  sorry

end NUMINAMATH_CALUDE_inverse_square_relation_l640_64071


namespace NUMINAMATH_CALUDE_laura_payment_l640_64019

/-- The amount Laura gave to the cashier --/
def amount_given_to_cashier (pants_price : ℕ) (shirts_price : ℕ) (pants_quantity : ℕ) (shirts_quantity : ℕ) (change : ℕ) : ℕ :=
  pants_price * pants_quantity + shirts_price * shirts_quantity + change

/-- Theorem stating that Laura gave $250 to the cashier --/
theorem laura_payment : amount_given_to_cashier 54 33 2 4 10 = 250 := by
  sorry

end NUMINAMATH_CALUDE_laura_payment_l640_64019


namespace NUMINAMATH_CALUDE_vector_magnitude_problem_l640_64046

/-- Given two vectors a and b in a real inner product space such that 
    |a| = |b| = |a - 2b| = 1, prove that |a + 2b| = 3. -/
theorem vector_magnitude_problem (a b : EuclideanSpace ℝ (Fin n)) 
  (h1 : ‖a‖ = 1) 
  (h2 : ‖b‖ = 1) 
  (h3 : ‖a - 2 • b‖ = 1) : 
  ‖a + 2 • b‖ = 3 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_problem_l640_64046


namespace NUMINAMATH_CALUDE_special_triangle_sides_l640_64043

/-- A triangle with specific properties -/
structure SpecialTriangle where
  -- The sides of the triangle
  a : ℝ
  b : ℝ
  c : ℝ
  -- The altitudes (heights) of the triangle
  ha : ℕ
  hb : ℕ
  hc : ℕ
  -- The radius of the inscribed circle
  r : ℝ
  -- Conditions
  radius_condition : r = 4/3
  altitudes_sum : ha + hb + hc = 13
  altitude_relation : 1/ha + 1/hb + 1/hc = 3/4

/-- Theorem about the side lengths of the special triangle -/
theorem special_triangle_sides (t : SpecialTriangle) : 
  t.a = 32 / Real.sqrt 15 ∧ 
  t.b = 24 / Real.sqrt 15 ∧ 
  t.c = 16 / Real.sqrt 15 :=
by sorry

end NUMINAMATH_CALUDE_special_triangle_sides_l640_64043


namespace NUMINAMATH_CALUDE_male_attendees_fraction_l640_64095

theorem male_attendees_fraction (M F : ℚ) : 
  M + F = 1 →
  (3/4 : ℚ) * M + (5/6 : ℚ) * F = 7/9 →
  M = 2/3 := by
sorry

end NUMINAMATH_CALUDE_male_attendees_fraction_l640_64095


namespace NUMINAMATH_CALUDE_ages_problem_l640_64045

/-- The present ages of individuals A, B, C, and D satisfy the given conditions. -/
theorem ages_problem (A B C D : ℕ) : 
  (C + 10 = 3 * (A + 10)) →  -- In 10 years, C will be 3 times as old as A
  (A = 2 * (B - 10)) →       -- A will be twice as old as B was 10 years ago
  (A = B + 12) →             -- A is now 12 years older than B
  (B = D + 5) →              -- B is 5 years older than D
  (D = C / 2) →              -- D is half the age of C
  (A = 88 ∧ B = 76 ∧ C = 142 ∧ D = 71) :=
by sorry

end NUMINAMATH_CALUDE_ages_problem_l640_64045


namespace NUMINAMATH_CALUDE_max_product_of_fractions_l640_64011

def is_digit (n : ℕ) : Prop := n ≥ 1 ∧ n ≤ 9

theorem max_product_of_fractions (A B C D : ℕ) 
  (hA : is_digit A) (hB : is_digit B) (hC : is_digit C) (hD : is_digit D)
  (hAB : A ≠ B) (hAC : A ≠ C) (hAD : A ≠ D) (hBC : B ≠ C) (hBD : B ≠ D) (hCD : C ≠ D) :
  (∀ (W X Y Z : ℕ), is_digit W → is_digit X → is_digit Y → is_digit Z →
    W ≠ X → W ≠ Y → W ≠ Z → X ≠ Y → X ≠ Z → Y ≠ Z →
    (A : ℚ) / B * (C : ℚ) / D ≥ (W : ℚ) / X * (Y : ℚ) / Z) →
  (A : ℚ) / B * (C : ℚ) / D = 36 := by
sorry

end NUMINAMATH_CALUDE_max_product_of_fractions_l640_64011


namespace NUMINAMATH_CALUDE_polynomial_equality_l640_64099

/-- Given that 2x^5 + 4x^3 + 3x + 4 + g(x) = x^4 - 2x^3 + 3,
    prove that g(x) = -2x^5 + x^4 - 6x^3 - 3x - 1 -/
theorem polynomial_equality (x : ℝ) (g : ℝ → ℝ) 
    (h : ∀ x, 2 * x^5 + 4 * x^3 + 3 * x + 4 + g x = x^4 - 2 * x^3 + 3) :
  g x = -2 * x^5 + x^4 - 6 * x^3 - 3 * x - 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l640_64099


namespace NUMINAMATH_CALUDE_regression_analysis_l640_64085

/-- Unit prices -/
def unit_prices : List ℝ := [4, 5, 6, 7, 8, 9]

/-- Sales volumes -/
def sales_volumes : List ℝ := [90, 84, 83, 80, 75, 68]

/-- Empirical regression equation -/
def regression_equation (x : ℝ) (a : ℝ) : ℝ := -4 * x + a

theorem regression_analysis :
  let avg_sales := (sales_volumes.sum) / (sales_volumes.length : ℝ)
  let slope := -4
  let a := avg_sales + 4 * ((unit_prices.sum) / (unit_prices.length : ℝ))
  (avg_sales = 80) ∧ 
  (slope = -4) ∧
  (regression_equation 10 a = 66) := by sorry

end NUMINAMATH_CALUDE_regression_analysis_l640_64085


namespace NUMINAMATH_CALUDE_jons_toaster_cost_l640_64067

/-- Calculates the total cost of a toaster purchase with given parameters. -/
def toaster_total_cost (msrp : ℝ) (standard_insurance_rate : ℝ) (premium_insurance_additional : ℝ) 
                       (tax_rate : ℝ) (recycling_fee : ℝ) : ℝ :=
  let standard_insurance := msrp * standard_insurance_rate
  let premium_insurance := standard_insurance + premium_insurance_additional
  let subtotal := msrp + premium_insurance
  let tax := subtotal * tax_rate
  subtotal + tax + recycling_fee

/-- Theorem stating that the total cost for Jon's toaster purchase is $69.50 -/
theorem jons_toaster_cost : 
  toaster_total_cost 30 0.2 7 0.5 5 = 69.5 := by
  sorry

end NUMINAMATH_CALUDE_jons_toaster_cost_l640_64067


namespace NUMINAMATH_CALUDE_part1_part2_l640_64096

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 4*x - 12 ≤ 0}
def B (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < 3*a + 2}

-- Part 1
theorem part1 : A ∩ (Set.univ \ (B 1)) = {x | -2 ≤ x ∧ x ≤ 0 ∨ 5 ≤ x ∧ x ≤ 6} := by sorry

-- Part 2
theorem part2 : ∀ a : ℝ, A ∩ (B a) = B a ↔ a ∈ Set.Iic (-3/2) ∪ Set.Icc (-1) (4/3) := by sorry

end NUMINAMATH_CALUDE_part1_part2_l640_64096


namespace NUMINAMATH_CALUDE_fraction_inequality_l640_64012

theorem fraction_inequality (x₁ x₂ : ℝ) (h1 : 0 < x₁) (h2 : x₁ < x₂) :
  (x₁ + 1) / (x₂ + 1) > x₁ / x₂ := by sorry

end NUMINAMATH_CALUDE_fraction_inequality_l640_64012


namespace NUMINAMATH_CALUDE_sum_is_three_or_seven_l640_64086

theorem sum_is_three_or_seven (x y z : ℝ) 
  (eq1 : x + y / z = 2)
  (eq2 : y + z / x = 2)
  (eq3 : z + x / y = 2) :
  let S := x + y + z
  S = 3 ∨ S = 7 := by
sorry

end NUMINAMATH_CALUDE_sum_is_three_or_seven_l640_64086


namespace NUMINAMATH_CALUDE_number_of_students_l640_64076

/-- Given a class where:
    1. The initial average marks of students is 100.
    2. A student's mark is wrongly noted as 50 instead of 10.
    3. The correct average marks is 96.
    Prove that the number of students in the class is 10. -/
theorem number_of_students (n : ℕ) 
    (h1 : (100 * n) / n = 100)  -- Initial average is 100
    (h2 : (100 * n - 40) / n = 96)  -- Correct average is 96
    : n = 10 := by
  sorry


end NUMINAMATH_CALUDE_number_of_students_l640_64076


namespace NUMINAMATH_CALUDE_tan_value_fourth_quadrant_l640_64014

theorem tan_value_fourth_quadrant (α : Real) 
  (h1 : α ∈ Set.Icc (3 * Real.pi / 2) (2 * Real.pi)) 
  (h2 : Real.sin α + Real.cos α = 1/5) : 
  Real.tan α = -3/4 := by
sorry

end NUMINAMATH_CALUDE_tan_value_fourth_quadrant_l640_64014


namespace NUMINAMATH_CALUDE_q_satisfies_conditions_l640_64083

/-- The quadratic polynomial q(x) that satisfies specific conditions -/
def q (x : ℚ) : ℚ := -25/11 * x^2 + 75/11 * x + 450/11

/-- Theorem stating that q(x) satisfies the given conditions -/
theorem q_satisfies_conditions :
  q (-3) = 0 ∧ q 6 = 0 ∧ q 8 = -50 := by
  sorry

end NUMINAMATH_CALUDE_q_satisfies_conditions_l640_64083


namespace NUMINAMATH_CALUDE_circle_center_distance_l640_64017

/-- Given a circle with equation x^2 + y^2 - 6x + 8y + 4 = 0 and a point (19, 11),
    the distance between the center of the circle and the point is √481. -/
theorem circle_center_distance (x y : ℝ) : 
  (x^2 + y^2 - 6*x + 8*y + 4 = 0) → 
  Real.sqrt ((19 - x)^2 + (11 - y)^2) = Real.sqrt 481 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_distance_l640_64017


namespace NUMINAMATH_CALUDE_reservoir_water_ratio_l640_64008

/-- Proof of the ratio of water in a reservoir --/
theorem reservoir_water_ratio :
  ∀ (total_capacity current_amount normal_level : ℝ),
  current_amount = 14000000 →
  current_amount = 0.7 * total_capacity →
  normal_level = total_capacity - 10000000 →
  current_amount / normal_level = 1.4 := by
  sorry

end NUMINAMATH_CALUDE_reservoir_water_ratio_l640_64008


namespace NUMINAMATH_CALUDE_roots_of_equation_l640_64090

def equation (x : ℝ) : ℝ := x * (2*x - 5)^2 * (x + 3) * (7 - x)

theorem roots_of_equation :
  {x : ℝ | equation x = 0} = {0, 2.5, -3, 7} := by sorry

end NUMINAMATH_CALUDE_roots_of_equation_l640_64090


namespace NUMINAMATH_CALUDE_rachel_setup_time_l640_64037

/-- Represents the time in hours for Rachel's speed painting process. -/
structure PaintingTime where
  setup : ℝ
  paintingPerVideo : ℝ
  cleanup : ℝ
  editAndPostPerVideo : ℝ
  totalPerVideo : ℝ
  batchSize : ℕ

/-- The setup time for Rachel's speed painting process is 1 hour. -/
theorem rachel_setup_time (t : PaintingTime) : t.setup = 1 :=
  by
  have h1 : t.paintingPerVideo = 1 := by sorry
  have h2 : t.cleanup = 1 := by sorry
  have h3 : t.editAndPostPerVideo = 1.5 := by sorry
  have h4 : t.totalPerVideo = 3 := by sorry
  have h5 : t.batchSize = 4 := by sorry
  
  have total_batch_time : t.setup + t.batchSize * (t.paintingPerVideo + t.editAndPostPerVideo) + t.cleanup = t.batchSize * t.totalPerVideo :=
    by sorry
  
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_rachel_setup_time_l640_64037


namespace NUMINAMATH_CALUDE_game_download_proof_l640_64074

/-- Proves that the amount downloaded before the connection slowed down is 310 MB -/
theorem game_download_proof (total_size : ℕ) (current_speed : ℕ) (remaining_time : ℕ) 
  (h1 : total_size = 880)
  (h2 : current_speed = 3)
  (h3 : remaining_time = 190) :
  total_size - current_speed * remaining_time = 310 := by
  sorry

end NUMINAMATH_CALUDE_game_download_proof_l640_64074


namespace NUMINAMATH_CALUDE_system_solution_l640_64018

theorem system_solution : ∃ (x y : ℝ), x - y = 3 ∧ x + y = 1 ∧ x = 2 ∧ y = -1 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l640_64018


namespace NUMINAMATH_CALUDE_independence_day_banana_distribution_l640_64078

theorem independence_day_banana_distribution :
  ∀ (total_children : ℕ) (total_bananas : ℕ),
    (2 * total_children = total_bananas) →
    (4 * (total_children - 390) = total_bananas) →
    total_children = 780 := by
  sorry

end NUMINAMATH_CALUDE_independence_day_banana_distribution_l640_64078


namespace NUMINAMATH_CALUDE_magic_8_ball_probability_l640_64030

/-- The probability of getting a positive answer from the Magic 8 Ball -/
def p : ℚ := 1/3

/-- The number of questions asked -/
def n : ℕ := 7

/-- The number of positive answers we're interested in -/
def k : ℕ := 3

/-- The probability of getting exactly k positive answers out of n questions -/
def probability_k_successes (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k) * p^k * (1-p)^(n-k)

theorem magic_8_ball_probability :
  probability_k_successes n k p = 560/2187 := by
  sorry

end NUMINAMATH_CALUDE_magic_8_ball_probability_l640_64030


namespace NUMINAMATH_CALUDE_unique_swap_pair_l640_64039

/-- A quadratic polynomial function -/
def QuadraticPolynomial (α : Type) [Ring α] := α → α

theorem unique_swap_pair
  (f : QuadraticPolynomial ℝ)
  (a b : ℝ)
  (h_distinct : a ≠ b)
  (h_swap : f a = b ∧ f b = a) :
  ¬∃ c d, c ≠ d ∧ (c, d) ≠ (a, b) ∧ f c = d ∧ f d = c :=
sorry

end NUMINAMATH_CALUDE_unique_swap_pair_l640_64039


namespace NUMINAMATH_CALUDE_lion_path_angles_l640_64087

theorem lion_path_angles (r : ℝ) (path_length : ℝ) (turn_angles : List ℝ) : 
  r = 10 →
  path_length = 30000 →
  path_length ≤ 2 * r + r * (turn_angles.sum) →
  turn_angles.sum ≥ 2998 := by
sorry

end NUMINAMATH_CALUDE_lion_path_angles_l640_64087


namespace NUMINAMATH_CALUDE_expression_evaluation_l640_64023

theorem expression_evaluation : (-1)^10 * 2 + (-2)^3 / 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l640_64023


namespace NUMINAMATH_CALUDE_triangle_properties_l640_64062

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : 2 * Real.cos t.C * (t.a * Real.cos t.B + t.b * Real.cos t.A) = t.c)
  (h2 : t.c = Real.sqrt 7)
  (h3 : (1/2) * t.a * t.b * Real.sin t.C = (3 * Real.sqrt 3) / 2) :
  t.C = π/3 ∧ t.a + t.b + t.c = 5 + Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l640_64062


namespace NUMINAMATH_CALUDE_birthday_candles_sharing_l640_64068

/-- 
Given that Ambika has 4 birthday candles and Aniyah has 6 times as many,
this theorem proves that when they put their candles together and share them equally,
each will have 14 candles.
-/
theorem birthday_candles_sharing (ambika_candles : ℕ) (aniyah_multiplier : ℕ) :
  ambika_candles = 4 →
  aniyah_multiplier = 6 →
  let aniyah_candles := ambika_candles * aniyah_multiplier
  let total_candles := ambika_candles + aniyah_candles
  total_candles / 2 = 14 := by
  sorry


end NUMINAMATH_CALUDE_birthday_candles_sharing_l640_64068


namespace NUMINAMATH_CALUDE_parabola_c_value_l640_64069

-- Define the parabola equation
def parabola (a b c : ℝ) (x y : ℝ) : Prop := x = a * y^2 + b * y + c

-- Define the vertex of the parabola
def vertex (x y : ℝ) : Prop := x = 5 ∧ y = 3

-- Define a point on the parabola
def point_on_parabola (x y : ℝ) : Prop := x = 3 ∧ y = 5

-- Theorem statement
theorem parabola_c_value :
  ∀ (a b c : ℝ),
  (∀ x y, vertex x y → parabola a b c x y) →
  (∀ x y, point_on_parabola x y → parabola a b c x y) →
  a = -1 →
  c = -4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_c_value_l640_64069


namespace NUMINAMATH_CALUDE_isosceles_triangles_in_square_l640_64049

theorem isosceles_triangles_in_square (s : ℝ) (h : s = 2) :
  let square_area := s^2
  let triangle_area := square_area / 4
  let half_base := s / 2
  let height := triangle_area / half_base
  let side_length := Real.sqrt (half_base^2 + height^2)
  side_length = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_isosceles_triangles_in_square_l640_64049


namespace NUMINAMATH_CALUDE_seven_mile_taxi_cost_l640_64044

/-- Calculates the total cost of a taxi ride given the fixed cost, cost per mile, and distance traveled. -/
def taxi_cost (fixed_cost : ℝ) (cost_per_mile : ℝ) (distance : ℝ) : ℝ :=
  fixed_cost + cost_per_mile * distance

/-- Theorem stating that a 7-mile taxi ride with $2.00 fixed cost and $0.30 per mile costs $4.10 -/
theorem seven_mile_taxi_cost :
  taxi_cost 2.00 0.30 7 = 4.10 := by
  sorry

end NUMINAMATH_CALUDE_seven_mile_taxi_cost_l640_64044


namespace NUMINAMATH_CALUDE_statement_1_false_statement_2_true_statement_3_true_statement_4_false_statement_5_true_l640_64060

-- Define the heartsuit operation
def heartsuit (x y : ℝ) : ℝ := |x - y|

-- Statement 1
theorem statement_1_false : ∃ x y z : ℝ, (heartsuit (heartsuit x y) z) ≠ (heartsuit x (heartsuit y z)) := by sorry

-- Statement 2
theorem statement_2_true : ∀ x y : ℝ, 3 * (heartsuit x y) = heartsuit (3 * x) (3 * y) := by sorry

-- Statement 3
theorem statement_3_true : ∀ x y : ℝ, heartsuit x (-y) = heartsuit (-x) y := by sorry

-- Statement 4
theorem statement_4_false : ∃ x : ℝ, heartsuit x x ≠ x := by sorry

-- Statement 5
theorem statement_5_true : ∀ x y : ℝ, heartsuit x y ≥ 0 := by sorry

end NUMINAMATH_CALUDE_statement_1_false_statement_2_true_statement_3_true_statement_4_false_statement_5_true_l640_64060


namespace NUMINAMATH_CALUDE_calculate_rates_l640_64057

/-- Represents the rates and quantities in the problem -/
structure Rates where
  b : ℕ  -- number of bananas Charles cooked
  d : ℕ  -- number of dishes Sandrine washed
  r1 : ℚ  -- rate at which Charles picks pears (pears per hour)
  r2 : ℚ  -- rate at which Charles cooks bananas (bananas per hour)
  r3 : ℚ  -- rate at which Sandrine washes dishes (dishes per hour)

/-- The main theorem representing the problem -/
theorem calculate_rates (rates : Rates) : 
  rates.d = rates.b + 10 ∧ 
  rates.b = 3 * 50 ∧ 
  rates.r1 = 50 / 4 ∧ 
  rates.r2 = rates.b / 2 ∧ 
  rates.r3 = rates.d / 5 → 
  rates.r1 = 12.5 ∧ rates.r2 = 75 ∧ rates.r3 = 32 := by
  sorry

end NUMINAMATH_CALUDE_calculate_rates_l640_64057


namespace NUMINAMATH_CALUDE_distribute_and_combine_l640_64026

theorem distribute_and_combine (a b : ℝ) : 2 * (a - b) + 3 * b = 2 * a + b := by
  sorry

end NUMINAMATH_CALUDE_distribute_and_combine_l640_64026


namespace NUMINAMATH_CALUDE_train_length_calculation_l640_64059

/-- Calculates the length of a train given its speed, the platform length, and the time taken to cross the platform. -/
theorem train_length_calculation (train_speed : ℝ) (platform_length : ℝ) (crossing_time : ℝ) : 
  train_speed = 55 → 
  platform_length = 620 → 
  crossing_time = 71.99424046076314 → 
  ∃ (train_length : ℝ), (train_length ≥ 479.9 ∧ train_length ≤ 480.1) := by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_l640_64059


namespace NUMINAMATH_CALUDE_cube_face_sum_l640_64089

theorem cube_face_sum (a b c d e f : ℕ+) :
  (a * b * c + a * e * c + a * b * f + a * e * f +
   d * b * c + d * e * c + d * b * f + d * e * f) = 1491 →
  (a + b + c + d + e + f : ℕ) = 41 := by
sorry

end NUMINAMATH_CALUDE_cube_face_sum_l640_64089


namespace NUMINAMATH_CALUDE_triangular_area_l640_64007

/-- The area of the triangular part of a piece of land -/
theorem triangular_area (total_length total_width rect_length rect_width : ℝ) 
  (h1 : total_length = 20)
  (h2 : total_width = 6)
  (h3 : rect_length = 15)
  (h4 : rect_width = 6) :
  total_length * total_width - rect_length * rect_width = 30 := by
  sorry

end NUMINAMATH_CALUDE_triangular_area_l640_64007


namespace NUMINAMATH_CALUDE_abs_diff_roots_quadratic_l640_64016

theorem abs_diff_roots_quadratic : 
  let a : ℝ := 1
  let b : ℝ := -7
  let c : ℝ := 10
  let r₁ : ℝ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ : ℝ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  |r₁ - r₂| = 3 := by
sorry


end NUMINAMATH_CALUDE_abs_diff_roots_quadratic_l640_64016


namespace NUMINAMATH_CALUDE_file_app_difference_l640_64077

/-- Given initial and final counts of apps and files on a phone, 
    prove the difference between final files and apps --/
theorem file_app_difference 
  (initial_apps : ℕ) 
  (initial_files : ℕ) 
  (final_apps : ℕ) 
  (final_files : ℕ) 
  (h1 : initial_apps = 11) 
  (h2 : initial_files = 3) 
  (h3 : final_apps = 2) 
  (h4 : final_files = 24) : 
  final_files - final_apps = 22 := by
  sorry

end NUMINAMATH_CALUDE_file_app_difference_l640_64077


namespace NUMINAMATH_CALUDE_quadruple_solution_l640_64040

-- Define the condition function
def condition (a b c d : ℝ) : Prop :=
  a + b * c * d = b + c * d * a ∧
  a + b * c * d = c + d * a * b ∧
  a + b * c * d = d + a * b * c

-- Define the solution set
def solution_set (a b c d : ℝ) : Prop :=
  (a = b ∧ b = c ∧ c = d) ∨
  (a = b ∧ c = d ∧ c = 1 / a ∧ a ≠ 0) ∨
  (a = 1 ∧ b = 1 ∧ c = 1) ∨
  (a = -1 ∧ b = -1 ∧ c = -1)

-- Theorem statement
theorem quadruple_solution (a b c d : ℝ) :
  condition a b c d → solution_set a b c d :=
sorry

end NUMINAMATH_CALUDE_quadruple_solution_l640_64040


namespace NUMINAMATH_CALUDE_polynomial_expansion_value_l640_64048

/-- The value of a in the expansion of (x+y)^7 -/
theorem polynomial_expansion_value (a b : ℝ) : 
  a > 0 ∧ b > 0 ∧ a + b = 1 ∧ 
  (21 * a^5 * b^2 = 35 * a^4 * b^3) →
  a = 5/8 := by
sorry

end NUMINAMATH_CALUDE_polynomial_expansion_value_l640_64048


namespace NUMINAMATH_CALUDE_problem_statement_l640_64072

theorem problem_statement (x y : ℝ) : 
  x = (Real.sqrt 7 + Real.sqrt 3) / (Real.sqrt 7 - Real.sqrt 3) →
  y = (Real.sqrt 7 - Real.sqrt 3) / (Real.sqrt 7 + Real.sqrt 3) →
  x^4 + y^4 + (x + y)^4 = 1152 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l640_64072


namespace NUMINAMATH_CALUDE_correct_calculation_l640_64088

-- Define variables
variable (x y : ℝ)

-- Theorem statement
theorem correct_calculation : -4 * x * y + 3 * x * y = -x * y := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l640_64088


namespace NUMINAMATH_CALUDE_total_migration_l640_64082

/-- The number of bird families that flew away for the winter -/
def total_migrated : ℕ := 118

/-- The number of bird families that flew to Africa -/
def africa_migrated : ℕ := 38

/-- The number of bird families that flew to Asia -/
def asia_migrated : ℕ := 80

/-- Theorem: The total number of bird families that migrated is equal to the sum of those that flew to Africa and Asia -/
theorem total_migration :
  total_migrated = africa_migrated + asia_migrated := by
sorry

end NUMINAMATH_CALUDE_total_migration_l640_64082


namespace NUMINAMATH_CALUDE_sector_area_l640_64027

theorem sector_area (θ : Real) (r : Real) (h1 : θ = 72 * π / 180) (h2 : r = 20) :
  (1 / 2) * θ * r^2 = 80 * π := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l640_64027


namespace NUMINAMATH_CALUDE_ferris_wheel_seats_l640_64058

/-- The number of people each seat can hold -/
def seat_capacity : ℕ := 6

/-- The total number of people the Ferris wheel can hold -/
def total_capacity : ℕ := 84

/-- The number of seats on the Ferris wheel -/
def num_seats : ℕ := total_capacity / seat_capacity

theorem ferris_wheel_seats : num_seats = 14 := by
  sorry

end NUMINAMATH_CALUDE_ferris_wheel_seats_l640_64058


namespace NUMINAMATH_CALUDE_perimeter_of_parallelogram_l640_64032

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)
  (AB AC BC : ℝ)
  (angleBAC : ℝ)

-- Define the parallelogram ADEF
structure Parallelogram :=
  (A D E F : ℝ × ℝ)

-- Define the problem statement
theorem perimeter_of_parallelogram (t : Triangle) (p : Parallelogram) : 
  t.AB = 20 →
  t.AC = 24 →
  t.BC = 18 →
  t.angleBAC = 60 * π / 180 →
  (p.D.1 - t.A.1) / (t.B.1 - t.A.1) = (p.D.2 - t.A.2) / (t.B.2 - t.A.2) →
  (p.E.1 - t.B.1) / (t.C.1 - t.B.1) = (p.E.2 - t.B.2) / (t.C.2 - t.B.2) →
  (p.F.1 - t.A.1) / (t.C.1 - t.A.1) = (p.F.2 - t.A.2) / (t.C.2 - t.A.2) →
  (p.E.1 - p.D.1) / (t.C.1 - t.A.1) = (p.E.2 - p.D.2) / (t.C.2 - t.A.2) →
  (p.F.1 - p.E.1) / (t.B.1 - t.A.1) = (p.F.2 - p.E.2) / (t.B.2 - t.A.2) →
  Real.sqrt ((p.A.1 - p.D.1)^2 + (p.A.2 - p.D.2)^2) +
  Real.sqrt ((p.D.1 - p.E.1)^2 + (p.D.2 - p.E.2)^2) +
  Real.sqrt ((p.E.1 - p.F.1)^2 + (p.E.2 - p.F.2)^2) +
  Real.sqrt ((p.F.1 - p.A.1)^2 + (p.F.2 - p.A.2)^2) = 44 :=
by sorry


end NUMINAMATH_CALUDE_perimeter_of_parallelogram_l640_64032


namespace NUMINAMATH_CALUDE_stock_sale_total_amount_l640_64079

/-- Calculates the total amount including brokerage for a stock sale -/
theorem stock_sale_total_amount 
  (cash_realized : ℝ) 
  (brokerage_rate : ℝ) 
  (h1 : cash_realized = 104.25)
  (h2 : brokerage_rate = 1/4) : 
  ∃ (total_amount : ℝ), total_amount = 104.51 ∧ 
  total_amount = cash_realized + (brokerage_rate / 100) * cash_realized := by
  sorry

end NUMINAMATH_CALUDE_stock_sale_total_amount_l640_64079


namespace NUMINAMATH_CALUDE_total_basketball_cost_l640_64055

/-- Represents a basketball team -/
structure Team where
  players : Nat
  basketballs_per_player : Nat
  price_per_basketball : Nat

/-- Calculates the total cost of basketballs for a team -/
def team_cost (t : Team) : Nat :=
  t.players * t.basketballs_per_player * t.price_per_basketball

/-- The Spurs basketball team -/
def spurs : Team :=
  { players := 22
    basketballs_per_player := 11
    price_per_basketball := 15 }

/-- The Dynamos basketball team -/
def dynamos : Team :=
  { players := 18
    basketballs_per_player := 9
    price_per_basketball := 20 }

/-- The Lions basketball team -/
def lions : Team :=
  { players := 26
    basketballs_per_player := 7
    price_per_basketball := 12 }

/-- Theorem stating the total cost of basketballs for all three teams -/
theorem total_basketball_cost :
  team_cost spurs + team_cost dynamos + team_cost lions = 9054 := by
  sorry

end NUMINAMATH_CALUDE_total_basketball_cost_l640_64055


namespace NUMINAMATH_CALUDE_product_of_roots_quadratic_l640_64061

theorem product_of_roots_quadratic (x₁ x₂ : ℝ) :
  x₁^2 - 3*x₁ + 2 = 0 → x₂^2 - 3*x₂ + 2 = 0 → x₁ * x₂ = 2 := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_quadratic_l640_64061


namespace NUMINAMATH_CALUDE_concert_revenue_proof_l640_64022

/-- Calculates the total revenue of a concert given ticket prices and attendance numbers. -/
def concertRevenue (adultPrice : ℕ) (adultAttendance : ℕ) (childAttendance : ℕ) : ℕ :=
  adultPrice * adultAttendance + (adultPrice / 2) * childAttendance

/-- Proves that the total revenue of the concert is $5122 given the specified conditions. -/
theorem concert_revenue_proof :
  concertRevenue 26 183 28 = 5122 := by
  sorry

#eval concertRevenue 26 183 28

end NUMINAMATH_CALUDE_concert_revenue_proof_l640_64022


namespace NUMINAMATH_CALUDE_vector_operation_l640_64015

/-- Given planar vectors a and b, prove that 1/2a - 3/2b equals (-1,2) -/
theorem vector_operation (a b : ℝ × ℝ) :
  a = (1, 1) →
  b = (1, -1) →
  (1/2 : ℝ) • a - (3/2 : ℝ) • b = (-1, 2) := by
sorry

end NUMINAMATH_CALUDE_vector_operation_l640_64015


namespace NUMINAMATH_CALUDE_triangle_shape_l640_64028

theorem triangle_shape (A B C : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) 
  (hSum : A + B + C = π) (hSine : 2 * Real.sin B * Real.cos C = Real.sin A) :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  a / Real.sin A = b / Real.sin B ∧
  b / Real.sin B = c / Real.sin C ∧
  b = c :=
sorry

end NUMINAMATH_CALUDE_triangle_shape_l640_64028


namespace NUMINAMATH_CALUDE_calculation_proof_l640_64050

theorem calculation_proof : 0.25^2005 * 4^2006 - 8^100 * 0.5^300 = 3 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l640_64050


namespace NUMINAMATH_CALUDE_parabola_arc_projection_difference_l640_64093

/-- 
Given a parabola y = x^2 + px + q and two rays y = x and y = 2x for x ≥ 0,
prove that the difference between the projection of the right arc and 
the projection of the left arc on the x-axis is equal to 1.
-/
theorem parabola_arc_projection_difference 
  (p q : ℝ) : 
  ∃ (x₁ x₂ x₃ x₄ : ℝ), 
    (x₁ < x₂) ∧ (x₃ < x₄) ∧
    (x₁^2 + p*x₁ + q = x₁) ∧ 
    (x₂^2 + p*x₂ + q = x₂) ∧
    (x₃^2 + p*x₃ + q = 2*x₃) ∧ 
    (x₄^2 + p*x₄ + q = 2*x₄) ∧
    (x₄ - x₂) - (x₁ - x₃) = 1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_arc_projection_difference_l640_64093


namespace NUMINAMATH_CALUDE_car_wash_soap_cost_l640_64000

/-- The cost of each bottle of car wash soap -/
def bottle_cost (washes_per_bottle : ℕ) (total_washes : ℕ) (total_cost : ℕ) : ℚ :=
  total_cost / (total_washes / washes_per_bottle)

/-- Theorem stating that the cost of each bottle is $4 -/
theorem car_wash_soap_cost :
  bottle_cost 4 20 20 = 4 := by
  sorry

end NUMINAMATH_CALUDE_car_wash_soap_cost_l640_64000


namespace NUMINAMATH_CALUDE_expression_evaluation_l640_64063

theorem expression_evaluation : 
  |Real.sqrt 2 - Real.sqrt 3| + 2 * Real.cos (π / 4) - Real.sqrt 2 * Real.sqrt 6 = -Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l640_64063


namespace NUMINAMATH_CALUDE_curve_W_properties_l640_64094

-- Define the curve W
def W (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + (y + 1)^2) * Real.sqrt (x^2 + (y - 1)^2) = 3

-- Theorem stating the properties of curve W
theorem curve_W_properties :
  -- 1. x = 0 is an axis of symmetry
  (∀ y : ℝ, W 0 y ↔ W 0 (-y)) ∧
  -- 2. (0, 2) and (0, -2) are points on W
  W 0 2 ∧ W 0 (-2) ∧
  -- 3. The range of y-coordinates is [-2, 2]
  (∀ x y : ℝ, W x y → -2 ≤ y ∧ y ≤ 2) ∧
  (∀ y : ℝ, -2 ≤ y ∧ y ≤ 2 → ∃ x : ℝ, W x y) :=
by sorry


end NUMINAMATH_CALUDE_curve_W_properties_l640_64094


namespace NUMINAMATH_CALUDE_inverse_81_mod_101_l640_64052

theorem inverse_81_mod_101 (h : (9 : ZMod 101)⁻¹ = 90) : (81 : ZMod 101)⁻¹ = 20 := by
  sorry

end NUMINAMATH_CALUDE_inverse_81_mod_101_l640_64052


namespace NUMINAMATH_CALUDE_jordan_machine_input_l640_64002

theorem jordan_machine_input (x : ℝ) : 2 * x + 3 - 5 = 27 → x = 14.5 := by
  sorry

end NUMINAMATH_CALUDE_jordan_machine_input_l640_64002


namespace NUMINAMATH_CALUDE_inessa_is_cleverest_l640_64042

-- Define the foxes
inductive Fox : Type
  | Alisa : Fox
  | Larisa : Fox
  | Inessa : Fox

-- Define a relation for "is cleverer than"
def is_cleverer_than : Fox → Fox → Prop := sorry

-- Define a property for being the cleverest
def is_cleverest : Fox → Prop := sorry

-- Define a function to check if a fox is telling the truth
def tells_truth : Fox → Prop := sorry

-- State the theorem
theorem inessa_is_cleverest :
  -- The cleverest fox lies, others tell the truth
  (∀ f : Fox, is_cleverest f ↔ ¬(tells_truth f)) →
  -- Larisa's statement
  (tells_truth Fox.Larisa ↔ ¬(is_cleverest Fox.Alisa)) →
  -- Alisa's statement
  (tells_truth Fox.Alisa ↔ is_cleverer_than Fox.Alisa Fox.Larisa) →
  -- Inessa's statement
  (tells_truth Fox.Inessa ↔ is_cleverer_than Fox.Alisa Fox.Inessa) →
  -- There is exactly one cleverest fox
  (∃! f : Fox, is_cleverest f) →
  -- Conclusion: Inessa is the cleverest
  is_cleverest Fox.Inessa :=
by
  sorry

end NUMINAMATH_CALUDE_inessa_is_cleverest_l640_64042


namespace NUMINAMATH_CALUDE_abc_positive_l640_64005

-- Define the quadratic function
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem abc_positive 
  (a b c : ℝ) 
  (ha : a ≠ 0)
  (h0 : quadratic a b c 0 = -2)
  (h1 : quadratic a b c 1 = -2)
  (hneg_half : quadratic a b c (-1/2) > 0) :
  a * b * c > 0 := by
  sorry

end NUMINAMATH_CALUDE_abc_positive_l640_64005


namespace NUMINAMATH_CALUDE_total_students_present_l640_64003

/-- Represents a kindergarten session with registered and absent students -/
structure Session where
  registered : ℕ
  absent : ℕ

/-- Calculates the number of present students in a session -/
def presentStudents (s : Session) : ℕ := s.registered - s.absent

/-- Represents the kindergarten school data -/
structure KindergartenSchool where
  morning : Session
  earlyAfternoon : Session
  lateAfternoon : Session
  earlyEvening : Session
  lateEvening : Session
  transferredOut : ℕ
  newRegistrations : ℕ
  newAttending : ℕ

/-- The main theorem to prove -/
theorem total_students_present (school : KindergartenSchool)
  (h1 : school.morning = { registered := 75, absent := 9 })
  (h2 : school.earlyAfternoon = { registered := 72, absent := 12 })
  (h3 : school.lateAfternoon = { registered := 90, absent := 15 })
  (h4 : school.earlyEvening = { registered := 50, absent := 6 })
  (h5 : school.lateEvening = { registered := 60, absent := 10 })
  (h6 : school.transferredOut = 3)
  (h7 : school.newRegistrations = 3)
  (h8 : school.newAttending = 1) :
  presentStudents school.morning +
  presentStudents school.earlyAfternoon +
  presentStudents school.lateAfternoon +
  presentStudents school.earlyEvening +
  presentStudents school.lateEvening -
  school.transferredOut +
  school.newAttending = 293 := by
  sorry

end NUMINAMATH_CALUDE_total_students_present_l640_64003


namespace NUMINAMATH_CALUDE_number_categorization_l640_64020

/-- Define the set of numbers we're working with -/
def numbers : Set ℚ := {-3.14, 22/7, 0, 2023}

/-- Define the set of negative rational numbers -/
def negative_rationals : Set ℚ := {x : ℚ | x < 0}

/-- Define the set of positive fractions -/
def positive_fractions : Set ℚ := {x : ℚ | x > 0 ∧ x ≠ ⌊x⌋}

/-- Define the set of non-negative integers -/
def non_negative_integers : Set ℤ := {x : ℤ | x ≥ 0}

/-- Define the set of natural numbers (including 0) -/
def natural_numbers : Set ℕ := Set.univ

/-- Theorem stating the categorization of the given numbers -/
theorem number_categorization :
  (-3.14 ∈ negative_rationals) ∧
  (22/7 ∈ positive_fractions) ∧
  (0 ∈ non_negative_integers) ∧
  (2023 ∈ non_negative_integers) ∧
  (0 ∈ natural_numbers) ∧
  (2023 ∈ natural_numbers) :=
by sorry

end NUMINAMATH_CALUDE_number_categorization_l640_64020


namespace NUMINAMATH_CALUDE_gcd_18_30_l640_64024

theorem gcd_18_30 : Nat.gcd 18 30 = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_18_30_l640_64024


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l640_64038

theorem simplify_and_rationalize :
  (Real.sqrt 3 / Real.sqrt 4) * (Real.sqrt 5 / Real.sqrt 6) * (Real.sqrt 8 / Real.sqrt 9) = Real.sqrt 15 / 9 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l640_64038


namespace NUMINAMATH_CALUDE_middle_integer_is_five_l640_64036

/-- Given three consecutive one-digit, positive, odd integers where their sum is
    one-seventh of their product, the middle integer is 5. -/
theorem middle_integer_is_five : 
  ∀ n : ℕ, 
    (n > 0 ∧ n < 10) →  -- one-digit positive integer
    (n % 2 = 1) →  -- odd integer
    (∃ (a b : ℕ), a = n - 2 ∧ b = n + 2 ∧  -- consecutive odd integers
      a > 0 ∧ b < 10 ∧  -- all are one-digit positive
      a % 2 = 1 ∧ b % 2 = 1 ∧  -- all are odd
      (a + n + b) = (a * n * b) / 7) →  -- sum is one-seventh of product
    n = 5 :=
by sorry

end NUMINAMATH_CALUDE_middle_integer_is_five_l640_64036


namespace NUMINAMATH_CALUDE_compare_base_6_and_base_2_l640_64098

def base_6_to_decimal (n : ℕ) : ℕ := 
  (n / 10) * 6 + (n % 10)

def base_2_to_decimal (n : ℕ) : ℕ :=
  (n / 100) * 4 + ((n / 10) % 10) * 2 + (n % 10)

theorem compare_base_6_and_base_2 : 
  base_6_to_decimal 12 > base_2_to_decimal 101 := by
  sorry

end NUMINAMATH_CALUDE_compare_base_6_and_base_2_l640_64098


namespace NUMINAMATH_CALUDE_sqrt_four_fifths_simplification_l640_64029

theorem sqrt_four_fifths_simplification :
  Real.sqrt (4 / 5) = (2 * Real.sqrt 5) / 5 := by sorry

end NUMINAMATH_CALUDE_sqrt_four_fifths_simplification_l640_64029


namespace NUMINAMATH_CALUDE_eighth_term_of_specific_arithmetic_sequence_l640_64064

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem eighth_term_of_specific_arithmetic_sequence
  (a : ℕ → ℤ)
  (h_arith : arithmetic_sequence a)
  (h_first : a 1 = -1)
  (h_diff : ∃ d : ℤ, d = -3 ∧ ∀ n : ℕ, a (n + 1) = a n + d) :
  a 8 = -22 :=
sorry

end NUMINAMATH_CALUDE_eighth_term_of_specific_arithmetic_sequence_l640_64064


namespace NUMINAMATH_CALUDE_probability_play_one_instrument_l640_64031

/-- Given a population with the following properties:
  * The total population is 10000
  * One-third of the population plays at least one instrument
  * 450 people play two or more instruments
  This theorem states that the probability of a randomly selected person
  playing exactly one instrument is 0.2883 -/
theorem probability_play_one_instrument (total_population : ℕ)
  (plays_at_least_one : ℕ) (plays_two_or_more : ℕ) :
  total_population = 10000 →
  plays_at_least_one = total_population / 3 →
  plays_two_or_more = 450 →
  (plays_at_least_one - plays_two_or_more : ℚ) / total_population = 2883 / 10000 := by
  sorry

end NUMINAMATH_CALUDE_probability_play_one_instrument_l640_64031


namespace NUMINAMATH_CALUDE_parallel_lines_x_value_l640_64013

/-- A line in a 2D plane --/
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- Check if two lines are parallel --/
def parallel (l1 l2 : Line) : Prop :=
  (l1.point1.1 = l1.point2.1) = (l2.point1.1 = l2.point2.1)

theorem parallel_lines_x_value (l1 l2 : Line) (x : ℝ) :
  l1.point1 = (-1, -2) →
  l1.point2 = (-1, 4) →
  l2.point1 = (2, 1) →
  l2.point2 = (x, 6) →
  parallel l1 l2 →
  x = 2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_x_value_l640_64013


namespace NUMINAMATH_CALUDE_abc_inequality_l640_64053

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : Real.sqrt a ^ 3 + Real.sqrt b ^ 3 + Real.sqrt c ^ 3 = 1) :
  a * b * c ≤ 1 / 9 ∧
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (a * b * c)) := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l640_64053


namespace NUMINAMATH_CALUDE_A_union_complement_B_eq_l640_64073

def U : Set Nat := {1,2,3,4,5}
def A : Set Nat := {1,2,3}
def B : Set Nat := {2,4}

theorem A_union_complement_B_eq : A ∪ (U \ B) = {1,2,3,5} := by sorry

end NUMINAMATH_CALUDE_A_union_complement_B_eq_l640_64073


namespace NUMINAMATH_CALUDE_polynomial_simplification_l640_64047

theorem polynomial_simplification (w : ℝ) : 
  2 * w^2 + 3 - 4 * w^2 + 2 * w - 6 * w + 4 = -2 * w^2 - 4 * w + 7 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l640_64047


namespace NUMINAMATH_CALUDE_greatest_value_quadratic_inequality_l640_64084

theorem greatest_value_quadratic_inequality :
  ∃ (x_max : ℝ), x_max = 9 ∧
  (∀ x : ℝ, x^2 - 12*x + 27 ≤ 0 → x ≤ x_max) ∧
  (x_max^2 - 12*x_max + 27 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_greatest_value_quadratic_inequality_l640_64084


namespace NUMINAMATH_CALUDE_organize_toys_time_l640_64035

/-- The time in minutes it takes to organize all toys given the following conditions:
  * There are 50 toys to organize
  * 4 toys are put into the box every 45 seconds
  * 3 toys are taken out immediately after each 45-second interval
-/
def organizeToys (totalToys : ℕ) (putIn : ℕ) (takeOut : ℕ) (cycleTime : ℚ) : ℚ :=
  let netIncrease : ℕ := putIn - takeOut
  let almostFullCycles : ℕ := (totalToys - putIn) / netIncrease
  let almostFullTime : ℚ := (almostFullCycles : ℚ) * cycleTime
  let finalCycleTime : ℚ := cycleTime
  (almostFullTime + finalCycleTime) / 60

theorem organize_toys_time :
  organizeToys 50 4 3 (45 / 60) = 35.25 := by
  sorry

end NUMINAMATH_CALUDE_organize_toys_time_l640_64035


namespace NUMINAMATH_CALUDE_probability_white_ball_specific_l640_64091

/-- The probability of drawing a white ball from a bag -/
def probability_white_ball (black white red : ℕ) : ℚ :=
  white / (black + white + red)

/-- Theorem: The probability of drawing a white ball from a bag with 3 black, 2 white, and 1 red ball is 1/3 -/
theorem probability_white_ball_specific : probability_white_ball 3 2 1 = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_probability_white_ball_specific_l640_64091


namespace NUMINAMATH_CALUDE_rectangular_prism_sum_l640_64054

theorem rectangular_prism_sum (a b c : ℕ+) : 
  a * b * c = 21 → a ≠ b → b ≠ c → a ≠ c → a + b + c = 11 := by
sorry

end NUMINAMATH_CALUDE_rectangular_prism_sum_l640_64054


namespace NUMINAMATH_CALUDE_complement_of_M_in_U_l640_64075

def U : Set Nat := {1, 2, 3, 4, 5}
def M : Set Nat := {4, 5}

theorem complement_of_M_in_U :
  (U \ M) = {1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_in_U_l640_64075


namespace NUMINAMATH_CALUDE_intersection_P_Q_l640_64010

def P : Set ℝ := {x | |x - 1| < 4}
def Q : Set ℝ := {x | ∃ y, y = Real.log (x + 2)}

theorem intersection_P_Q : P ∩ Q = Set.Ioo (-2 : ℝ) 5 := by sorry

end NUMINAMATH_CALUDE_intersection_P_Q_l640_64010


namespace NUMINAMATH_CALUDE_greatest_integer_2pi_minus_6_l640_64021

theorem greatest_integer_2pi_minus_6 :
  Int.floor (2 * Real.pi - 6) = 0 :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_2pi_minus_6_l640_64021


namespace NUMINAMATH_CALUDE_negation_equivalence_l640_64034

theorem negation_equivalence (a b : ℝ) :
  ¬(a ≤ 2 ∧ b ≤ 2) ↔ (a > 2 ∨ b > 2) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l640_64034


namespace NUMINAMATH_CALUDE_sum_with_radical_conjugate_l640_64065

theorem sum_with_radical_conjugate : 
  let x : ℝ := 15 - Real.sqrt 5005
  let y : ℝ := 15 + Real.sqrt 5005
  x + y = 30 := by
  sorry

end NUMINAMATH_CALUDE_sum_with_radical_conjugate_l640_64065


namespace NUMINAMATH_CALUDE_opposite_faces_in_cube_l640_64009

structure Cube where
  faces : Fin 6 → Char
  top : Fin 6
  front : Fin 6
  right : Fin 6
  back : Fin 6
  left : Fin 6
  bottom : Fin 6
  unique_faces : ∀ i j, i ≠ j → faces i ≠ faces j

def is_opposite (c : Cube) (f1 f2 : Fin 6) : Prop :=
  f1 ≠ f2 ∧ f1 ≠ c.top ∧ f1 ≠ c.bottom ∧ 
  f2 ≠ c.top ∧ f2 ≠ c.bottom ∧
  (f1 = c.front ∧ f2 = c.back ∨
   f1 = c.back ∧ f2 = c.front ∨
   f1 = c.left ∧ f2 = c.right ∨
   f1 = c.right ∧ f2 = c.left)

theorem opposite_faces_in_cube (c : Cube) 
  (h1 : c.faces c.top = 'A')
  (h2 : c.faces c.front = 'B')
  (h3 : c.faces c.right = 'C')
  (h4 : c.faces c.back = 'D')
  (h5 : c.faces c.left = 'E') :
  is_opposite c c.front c.back :=
by sorry

end NUMINAMATH_CALUDE_opposite_faces_in_cube_l640_64009


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l640_64033

theorem complex_fraction_equality : 2 * (1 + 1 / (1 - 1 / (2 + 2))) = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l640_64033


namespace NUMINAMATH_CALUDE_median_and_mode_of_scores_l640_64066

def scores : List Nat := [74, 84, 84, 84, 87, 92, 92]

def median (l : List Nat) : Nat := sorry

def mode (l : List Nat) : Nat := sorry

theorem median_and_mode_of_scores :
  median scores = 84 ∧ mode scores = 84 := by sorry

end NUMINAMATH_CALUDE_median_and_mode_of_scores_l640_64066


namespace NUMINAMATH_CALUDE_binomial_coefficient_identity_l640_64006

theorem binomial_coefficient_identity (n k : ℕ) (h1 : k ≤ n) (h2 : n ≥ 1) :
  Nat.choose n k = Nat.choose (n - 1) (k - 1) + Nat.choose (n - 1) k := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_identity_l640_64006


namespace NUMINAMATH_CALUDE_number_percentage_problem_l640_64004

theorem number_percentage_problem (N : ℚ) : 
  (4/5 : ℚ) * (3/8 : ℚ) * N = 24 → (5/2 : ℚ) * N = 200 := by
  sorry

end NUMINAMATH_CALUDE_number_percentage_problem_l640_64004


namespace NUMINAMATH_CALUDE_apple_sale_percentage_l640_64041

/-- The percentage of apples sold by a fruit seller -/
theorem apple_sale_percentage (original : Real) (remaining : Real) 
  (h1 : original = 2499.9987500006246)
  (h2 : remaining = 500) :
  let sold := original - remaining
  let percentage := (sold / original) * 100
  ∃ ε > 0, abs (percentage - 80) < ε :=
by sorry

end NUMINAMATH_CALUDE_apple_sale_percentage_l640_64041


namespace NUMINAMATH_CALUDE_square_root_sequence_l640_64070

theorem square_root_sequence (n : ℕ) : 
  (∀ k ∈ Finset.range 35, Int.floor (Real.sqrt (n^2 + k : ℝ)) = n) ↔ n = 17 := by
sorry

end NUMINAMATH_CALUDE_square_root_sequence_l640_64070

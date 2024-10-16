import Mathlib

namespace NUMINAMATH_CALUDE_egg_groups_l3282_328218

/-- Given 16 eggs split into groups of 2, prove that the number of groups is 8 -/
theorem egg_groups (total_eggs : ℕ) (eggs_per_group : ℕ) (num_groups : ℕ) : 
  total_eggs = 16 → eggs_per_group = 2 → num_groups = total_eggs / eggs_per_group → num_groups = 8 := by
  sorry

end NUMINAMATH_CALUDE_egg_groups_l3282_328218


namespace NUMINAMATH_CALUDE_perfect_cube_factors_of_8820_l3282_328285

def prime_factorization (n : ℕ) : List (ℕ × ℕ) := sorry

def is_perfect_cube (n : ℕ) : Prop := sorry

def count_perfect_cube_factors (n : ℕ) : ℕ := sorry

theorem perfect_cube_factors_of_8820 :
  let factorization := prime_factorization 8820
  (factorization = [(2, 2), (3, 2), (5, 1), (7, 2)]) →
  count_perfect_cube_factors 8820 = 1 := by sorry

end NUMINAMATH_CALUDE_perfect_cube_factors_of_8820_l3282_328285


namespace NUMINAMATH_CALUDE_remainder_sum_l3282_328254

theorem remainder_sum (n : ℤ) (h : n % 18 = 11) : (n % 2 + n % 9 = 3) := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_l3282_328254


namespace NUMINAMATH_CALUDE_arithmetic_computation_l3282_328201

theorem arithmetic_computation : 2 + 8 * 3 - 4 + 7 * 6 / 3 = 36 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l3282_328201


namespace NUMINAMATH_CALUDE_function_derivative_inequality_l3282_328271

theorem function_derivative_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
    (hf' : ∀ x, deriv f x < 1) :
  ∀ m : ℝ, f (1 - m) - f m > 1 - 2*m → m > 1/2 := by
  sorry

end NUMINAMATH_CALUDE_function_derivative_inequality_l3282_328271


namespace NUMINAMATH_CALUDE_paint_usage_l3282_328289

theorem paint_usage (total_paint : ℚ) (first_week_fraction : ℚ) (total_used : ℚ) : 
  total_paint = 360 →
  first_week_fraction = 1/6 →
  total_used = 120 →
  (total_used - first_week_fraction * total_paint) / (total_paint - first_week_fraction * total_paint) = 1/5 :=
by sorry

end NUMINAMATH_CALUDE_paint_usage_l3282_328289


namespace NUMINAMATH_CALUDE_group_size_calculation_l3282_328247

theorem group_size_calculation (iceland : ℕ) (norway : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : iceland = 55)
  (h2 : norway = 33)
  (h3 : both = 51)
  (h4 : neither = 53) :
  iceland + norway - both + neither = 90 := by
  sorry

end NUMINAMATH_CALUDE_group_size_calculation_l3282_328247


namespace NUMINAMATH_CALUDE_smallest_two_digit_multiple_of_five_ending_in_five_plus_smallest_three_digit_multiple_of_seven_above_150_l3282_328262

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def ends_in_five (n : ℕ) : Prop := n % 10 = 5

theorem smallest_two_digit_multiple_of_five_ending_in_five_plus_smallest_three_digit_multiple_of_seven_above_150
  (a b : ℕ)
  (ha1 : is_two_digit a)
  (ha2 : a % 5 = 0)
  (ha3 : ends_in_five a)
  (ha4 : ∀ n, is_two_digit n → n % 5 = 0 → ends_in_five n → a ≤ n)
  (hb1 : is_three_digit b)
  (hb2 : b % 7 = 0)
  (hb3 : b > 150)
  (hb4 : ∀ n, is_three_digit n → n % 7 = 0 → n > 150 → b ≤ n) :
  a + b = 176 := by
  sorry

end NUMINAMATH_CALUDE_smallest_two_digit_multiple_of_five_ending_in_five_plus_smallest_three_digit_multiple_of_seven_above_150_l3282_328262


namespace NUMINAMATH_CALUDE_f_geq_f0_range_of_a_l3282_328249

-- Define the function f
def f (x : ℝ) : ℝ := |x + 2| + |x - 1|

-- Theorem 1: f(x) ≥ f(0) for all x
theorem f_geq_f0 : ∀ x : ℝ, f x ≥ f 0 := by sorry

-- Theorem 2: Given 2f(x) ≥ f(a+1) for all x, the range of a is [-4.5, 1.5]
theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, 2 * f x ≥ f (a + 1)) → -4.5 ≤ a ∧ a ≤ 1.5 := by sorry

end NUMINAMATH_CALUDE_f_geq_f0_range_of_a_l3282_328249


namespace NUMINAMATH_CALUDE_simplify_expression_l3282_328279

theorem simplify_expression (x : ℝ) : 
  3 * x^3 + 4 * x^2 + 2 - (7 - 3 * x^3 - 4 * x^2) = 6 * x^3 + 8 * x^2 - 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3282_328279


namespace NUMINAMATH_CALUDE_wooden_block_length_is_3070_l3282_328268

/-- The length of a wooden block in centimeters, given that it is 30 cm shorter than 31 meters -/
def wooden_block_length : ℕ :=
  let meters_to_cm : ℕ → ℕ := (· * 100)
  meters_to_cm 31 - 30

theorem wooden_block_length_is_3070 : wooden_block_length = 3070 := by
  sorry

end NUMINAMATH_CALUDE_wooden_block_length_is_3070_l3282_328268


namespace NUMINAMATH_CALUDE_local_minimum_value_inequality_condition_l3282_328296

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a * x^2 - 3 * x

-- Theorem for part (1)
theorem local_minimum_value (a : ℝ) :
  (∃ k, ∀ x, x ≠ 1 → (f a x - f a 1) / (x - 1) = k) →
  (∃ x₀, ∀ x, x ≠ x₀ → f a x > f a x₀) →
  f a x₀ = -Real.log 2 - 5/4 := by sorry

-- Theorem for part (2)
theorem inequality_condition (x₁ x₂ m : ℝ) :
  1 ≤ x₁ → x₁ < x₂ → x₂ ≤ 2 →
  (∀ x₁ x₂, 1 ≤ x₁ → x₁ < x₂ → x₂ ≤ 2 →
    f 1 x₁ - f 1 x₂ > m * (x₂ - x₁) / (x₁ * x₂)) ↔
  m ≤ -6 := by sorry

end NUMINAMATH_CALUDE_local_minimum_value_inequality_condition_l3282_328296


namespace NUMINAMATH_CALUDE_simplify_expression_l3282_328261

theorem simplify_expression (h : Real.pi < 4) : 
  Real.sqrt ((Real.pi - 4)^2) + Real.pi = 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3282_328261


namespace NUMINAMATH_CALUDE_cupcakes_per_child_third_group_l3282_328234

/- Define the total number of cupcakes -/
def total_cupcakes : ℕ := 144

/- Define the number of children -/
def num_children : ℕ := 12

/- Define the ratio for distribution -/
def ratio : List ℕ := [3, 2, 1]

/- Define the number of groups -/
def num_groups : ℕ := ratio.length

/- Assume the number of children is divisible by the number of groups -/
axiom children_divisible : num_children % num_groups = 0

/- Define the number of children per group -/
def children_per_group : ℕ := num_children / num_groups

/- Define the total parts in the ratio -/
def total_ratio_parts : ℕ := ratio.sum

/- Define cupcakes per ratio part -/
def cupcakes_per_part : ℕ := total_cupcakes / total_ratio_parts

/- Theorem: The number of cupcakes each child gets in the third group is 6 -/
theorem cupcakes_per_child_third_group :
  (cupcakes_per_part * ratio.getLast!) / children_per_group = 6 := by
  sorry

end NUMINAMATH_CALUDE_cupcakes_per_child_third_group_l3282_328234


namespace NUMINAMATH_CALUDE_share_ratio_l3282_328299

/-- Given a total amount of $500 divided among three people a, b, and c,
    where a's share is $200, a gets a fraction of b and c's combined share,
    and b gets 6/9 of a and c's combined share, prove that the ratio of
    a's share to the combined share of b and c is 2:3. -/
theorem share_ratio (total : ℚ) (a b c : ℚ) :
  total = 500 →
  a = 200 →
  ∃ x : ℚ, a = x * (b + c) →
  b = (6/9) * (a + c) →
  a + b + c = total →
  a / (b + c) = 2/3 :=
by sorry

end NUMINAMATH_CALUDE_share_ratio_l3282_328299


namespace NUMINAMATH_CALUDE_speed_limit_representation_l3282_328209

-- Define the speed limit
def speed_limit : ℝ := 70

-- Define a vehicle's speed
variable (v : ℝ)

-- Theorem stating that v ≤ speed_limit correctly represents the speed limit instruction
theorem speed_limit_representation : 
  (v ≤ speed_limit) ↔ (v ≤ speed_limit ∧ ¬(v > speed_limit)) :=
by sorry

end NUMINAMATH_CALUDE_speed_limit_representation_l3282_328209


namespace NUMINAMATH_CALUDE_largest_power_of_two_dividing_n_l3282_328292

def n : ℕ := 15^4 - 9^4

theorem largest_power_of_two_dividing_n : 
  ∃ k : ℕ, k = 5 ∧ 2^k ∣ n ∧ ∀ m : ℕ, 2^m ∣ n → m ≤ k :=
sorry

end NUMINAMATH_CALUDE_largest_power_of_two_dividing_n_l3282_328292


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3282_328205

/-- Given a right triangle with shorter leg x and longer leg (3x - 2), 
    prove that if the area of the triangle is 90 square feet, 
    then the length of the hypotenuse is approximately 23.65 feet. -/
theorem right_triangle_hypotenuse (x : ℝ) : 
  x > 0 →
  (1/2) * x * (3*x - 2) = 90 →
  ∃ h : ℝ, h^2 = x^2 + (3*x - 2)^2 ∧ abs (h - 23.65) < 0.01 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3282_328205


namespace NUMINAMATH_CALUDE_courtyard_width_l3282_328253

/-- The width of a courtyard given its length, brick dimensions, and total number of bricks --/
theorem courtyard_width (length : ℝ) (brick_length brick_width : ℝ) (total_bricks : ℝ) :
  length = 28 →
  brick_length = 0.22 →
  brick_width = 0.12 →
  total_bricks = 13787.878787878788 →
  ∃ width : ℝ, abs (width - 13.012) < 0.001 ∧ 
    length * width * 100 * 100 = total_bricks * brick_length * brick_width * 100 * 100 :=
by sorry

end NUMINAMATH_CALUDE_courtyard_width_l3282_328253


namespace NUMINAMATH_CALUDE_midpoint_after_translation_l3282_328282

-- Define the points B and G
def B : ℝ × ℝ := (2, 3)
def G : ℝ × ℝ := (6, 3)

-- Define the translation
def translate (p : ℝ × ℝ) : ℝ × ℝ := (p.1 - 7, p.2 - 3)

-- Theorem statement
theorem midpoint_after_translation :
  let B' := translate B
  let G' := translate G
  (B'.1 + G'.1) / 2 = -3 ∧ (B'.2 + G'.2) / 2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_midpoint_after_translation_l3282_328282


namespace NUMINAMATH_CALUDE_optimal_selling_price_l3282_328244

/-- Represents the profit optimization problem for a product -/
def ProfitOptimization (initialCost initialPrice initialSales : ℝ) 
                       (priceIncrease salesDecrease : ℝ) : Prop :=
  let profitFunction := fun x : ℝ => 
    (initialPrice + priceIncrease * x - initialCost) * (initialSales - salesDecrease * x)
  ∃ (optimalX : ℝ), 
    (∀ x : ℝ, profitFunction x ≤ profitFunction optimalX) ∧
    initialPrice + priceIncrease * optimalX = 14

/-- The main theorem stating the optimal selling price -/
theorem optimal_selling_price :
  ProfitOptimization 8 10 200 0.5 10 := by
  sorry

#check optimal_selling_price

end NUMINAMATH_CALUDE_optimal_selling_price_l3282_328244


namespace NUMINAMATH_CALUDE_cameron_questions_total_l3282_328203

/-- Represents a tour group with a number of regular tourists and inquisitive tourists -/
structure TourGroup where
  regular_tourists : ℕ
  inquisitive_tourists : ℕ

/-- Calculates the number of questions answered for a tour group -/
def questions_answered (group : TourGroup) (questions_per_tourist : ℕ) : ℕ :=
  group.regular_tourists * questions_per_tourist + 
  group.inquisitive_tourists * (questions_per_tourist * 3)

theorem cameron_questions_total : 
  let questions_per_tourist := 2
  let tour1 := TourGroup.mk 6 0
  let tour2 := TourGroup.mk 11 0
  let tour3 := TourGroup.mk 7 1
  let tour4 := TourGroup.mk 7 0
  questions_answered tour1 questions_per_tourist +
  questions_answered tour2 questions_per_tourist +
  questions_answered tour3 questions_per_tourist +
  questions_answered tour4 questions_per_tourist = 68 := by
  sorry

end NUMINAMATH_CALUDE_cameron_questions_total_l3282_328203


namespace NUMINAMATH_CALUDE_tangent_line_to_quartic_l3282_328221

/-- The value of b for which y = x^4 is tangent to y = 4x + b is -3 -/
theorem tangent_line_to_quartic (x : ℝ) : 
  ∃ (m n : ℝ), 
    n = m^4 ∧ 
    n = 4*m + (-3) ∧ 
    (4:ℝ) = 4*m^3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_to_quartic_l3282_328221


namespace NUMINAMATH_CALUDE_divisor_count_fourth_power_l3282_328260

theorem divisor_count_fourth_power (x d : ℕ) : 
  (∃ n : ℕ, x = n^4) → 
  (d = (Finset.filter (· ∣ x) (Finset.range (x + 1))).card) →
  d ≡ 1 [MOD 4] := by
sorry

end NUMINAMATH_CALUDE_divisor_count_fourth_power_l3282_328260


namespace NUMINAMATH_CALUDE_river_road_cars_l3282_328287

theorem river_road_cars (buses cars : ℕ) : 
  buses * 10 = cars ∧ cars - buses = 90 → cars = 100 := by
  sorry

end NUMINAMATH_CALUDE_river_road_cars_l3282_328287


namespace NUMINAMATH_CALUDE_factorization_1_factorization_2_factorization_3_l3282_328216

-- Define variables
variable (a b m x y : ℝ)

-- Theorem 1
theorem factorization_1 : 3*m - 3*y + a*m - a*y = (m - y) * (3 + a) := by sorry

-- Theorem 2
theorem factorization_2 : a^2*x + a^2*y + b^2*x + b^2*y = (x + y) * (a^2 + b^2) := by sorry

-- Theorem 3
theorem factorization_3 : a^2 + 2*a*b + b^2 - 1 = (a + b + 1) * (a + b - 1) := by sorry

end NUMINAMATH_CALUDE_factorization_1_factorization_2_factorization_3_l3282_328216


namespace NUMINAMATH_CALUDE_smallest_sum_of_sequence_l3282_328258

theorem smallest_sum_of_sequence (P Q R S : ℤ) : 
  P > 0 → Q > 0 → R > 0 →  -- P, Q, R are positive integers
  (R - Q = Q - P) →  -- P, Q, R form an arithmetic sequence
  (R * R = Q * S) →  -- Q, R, S form a geometric sequence
  (R = (4 * Q) / 3) →  -- R/Q = 4/3
  (∀ P' Q' R' S' : ℤ, 
    P' > 0 → Q' > 0 → R' > 0 → 
    (R' - Q' = Q' - P') → 
    (R' * R' = Q' * S') → 
    (R' = (4 * Q') / 3) → 
    P + Q + R + S ≤ P' + Q' + R' + S') →
  P + Q + R + S = 171 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_sequence_l3282_328258


namespace NUMINAMATH_CALUDE_walker_children_puzzle_l3282_328291

def is_aabb (n : ℕ) : Prop :=
  ∃ a b : ℕ, n = a * 1000 + a * 100 + b * 10 + b ∧ 0 < a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9

def divisible_by_nine_out_of_ten (n : ℕ) : Prop :=
  ∃ k : ℕ, k ∈ (Finset.range 10).filter (λ i => i ≠ 0) ∧
    ∀ i ∈ (Finset.range 10).filter (λ i => i ≠ 0), i ≠ k → n % i = 0

theorem walker_children_puzzle :
  ∀ n : ℕ, is_aabb n → divisible_by_nine_out_of_ten n →
    ∃ (x y : ℕ), x + y = n → 
      ∃ k : ℕ, k ∈ (Finset.range 10).filter (λ i => i ≠ 0) ∧ n % k ≠ 0 ∧ k = 9 :=
sorry

end NUMINAMATH_CALUDE_walker_children_puzzle_l3282_328291


namespace NUMINAMATH_CALUDE_sequence_correct_l3282_328231

def sequence_term (n : ℕ) : ℤ := (-1)^n * (2^n - 1)

theorem sequence_correct : 
  sequence_term 1 = -1 ∧ 
  sequence_term 2 = 3 ∧ 
  sequence_term 3 = -7 ∧ 
  sequence_term 4 = 15 := by
sorry

end NUMINAMATH_CALUDE_sequence_correct_l3282_328231


namespace NUMINAMATH_CALUDE_common_tangent_range_l3282_328240

/-- The range of parameter a for which the curves y = ln x + 1 and y = x² + x + 3a have a common tangent line -/
theorem common_tangent_range :
  ∀ a : ℝ, (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧
    (1 / x₁ = 2 * x₂ + 1) ∧
    (Real.log x₁ + 1 = x₂^2 + x₂ + 3 * a) ∧
    (Real.log x₁ + x₂^2 = 3 * a)) ↔
  a ≥ (1 - 4 * Real.log 2) / 12 :=
by sorry

end NUMINAMATH_CALUDE_common_tangent_range_l3282_328240


namespace NUMINAMATH_CALUDE_wire_cutting_problem_l3282_328251

theorem wire_cutting_problem (total_length : ℝ) (ratio : ℝ) :
  total_length = 120 →
  ratio = 7 / 13 →
  ∃ (shorter_piece longer_piece : ℝ),
    shorter_piece + longer_piece = total_length ∧
    longer_piece = ratio * shorter_piece ∧
    shorter_piece = 78 := by
  sorry

end NUMINAMATH_CALUDE_wire_cutting_problem_l3282_328251


namespace NUMINAMATH_CALUDE_perfume_dilution_l3282_328270

/-- Proves that adding 7.2 ounces of water to 12 ounces of a 40% alcohol solution
    results in a 25% alcohol solution -/
theorem perfume_dilution (initial_volume : ℝ) (initial_concentration : ℝ)
                         (target_concentration : ℝ) (water_added : ℝ) :
  initial_volume = 12 →
  initial_concentration = 0.4 →
  target_concentration = 0.25 →
  water_added = 7.2 →
  (initial_volume * initial_concentration) / (initial_volume + water_added) = target_concentration :=
by
  sorry

#check perfume_dilution

end NUMINAMATH_CALUDE_perfume_dilution_l3282_328270


namespace NUMINAMATH_CALUDE_vector_problem_l3282_328257

/-- Given two non-collinear vectors e₁ and e₂ in a real vector space,
    prove that if CB = e₁ + 3e₂, CD = 2e₁ - e₂, BF = 3e₁ - ke₂,
    and points B, D, and F are collinear, then k = 12. -/
theorem vector_problem (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V]
  (e₁ e₂ : V) (hne : ¬ ∃ (r : ℝ), e₁ = r • e₂) 
  (CB CD BF : V)
  (hCB : CB = e₁ + 3 • e₂)
  (hCD : CD = 2 • e₁ - e₂)
  (k : ℝ)
  (hBF : BF = 3 • e₁ - k • e₂)
  (hcollinear : ∃ (t : ℝ), BF = t • (CD - CB)) :
  k = 12 := by sorry

end NUMINAMATH_CALUDE_vector_problem_l3282_328257


namespace NUMINAMATH_CALUDE_purely_imaginary_product_l3282_328297

theorem purely_imaginary_product (x : ℝ) : 
  (Complex.I : ℂ).im * ((x + 2 * Complex.I) * ((x + 3) + 2 * Complex.I) * ((x + 5) + 2 * Complex.I)).re = 0 ↔ 
  x = -5 ∨ x = -4 ∨ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_product_l3282_328297


namespace NUMINAMATH_CALUDE_empty_boxes_count_l3282_328243

theorem empty_boxes_count (total boxes_with_markers boxes_with_crayons boxes_with_both : ℕ) 
  (h1 : total = 15)
  (h2 : boxes_with_markers = 9)
  (h3 : boxes_with_crayons = 4)
  (h4 : boxes_with_both = 5) :
  total - (boxes_with_markers + boxes_with_crayons - boxes_with_both) = 7 := by
  sorry

end NUMINAMATH_CALUDE_empty_boxes_count_l3282_328243


namespace NUMINAMATH_CALUDE_max_value_fourth_root_sum_l3282_328272

theorem max_value_fourth_root_sum (a b c d : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) (hd : 0 ≤ d ∧ d ≤ 1) : 
  (abcd : ℝ) ^ (1/4) + ((1-a)*(1-b)*(1-c)*(1-d) : ℝ) ^ (1/4) ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_fourth_root_sum_l3282_328272


namespace NUMINAMATH_CALUDE_response_rate_percentage_l3282_328226

/-- Given that 300 responses are needed and the minimum number of questionnaires
    to be mailed is 375, prove that the response rate percentage is 80%. -/
theorem response_rate_percentage
  (responses_needed : ℕ)
  (questionnaires_mailed : ℕ)
  (h1 : responses_needed = 300)
  (h2 : questionnaires_mailed = 375) :
  (responses_needed : ℝ) / questionnaires_mailed * 100 = 80 := by
  sorry

end NUMINAMATH_CALUDE_response_rate_percentage_l3282_328226


namespace NUMINAMATH_CALUDE_max_guaranteed_pastries_l3282_328217

/-- Represents a game with circular arrangement of plates and pastries. -/
structure PastryGame where
  num_plates : Nat
  max_move : Nat

/-- Represents the result of the game. -/
inductive GameResult
  | CanGuarantee
  | CannotGuarantee

/-- Determines if a certain number of pastries can be guaranteed on a single plate. -/
def can_guarantee (game : PastryGame) (k : Nat) : GameResult :=
  sorry

/-- The main theorem stating the maximum number of pastries that can be guaranteed. -/
theorem max_guaranteed_pastries (game : PastryGame) : 
  game.num_plates = 2019 → game.max_move = 16 → can_guarantee game 32 = GameResult.CanGuarantee ∧ 
  can_guarantee game 33 = GameResult.CannotGuarantee :=
  sorry

end NUMINAMATH_CALUDE_max_guaranteed_pastries_l3282_328217


namespace NUMINAMATH_CALUDE_triangle_angle_weighted_average_bounds_l3282_328264

theorem triangle_angle_weighted_average_bounds 
  (A B C a b c : ℝ) 
  (h_triangle : A + B + C = π) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) : 
  π / 3 ≤ (a * A + b * B + c * C) / (a + b + c) ∧ 
  (a * A + b * B + c * C) / (a + b + c) < π / 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_weighted_average_bounds_l3282_328264


namespace NUMINAMATH_CALUDE_multiples_of_five_relation_l3282_328207

theorem multiples_of_five_relation (a b c : ℤ) : 
  (∃ k l m : ℤ, a = 5 * k ∧ b = 5 * l ∧ c = 5 * m) →  -- a, b, c are multiples of 5
  a < b →                                            -- a < b
  b < c →                                            -- b < c
  c = a + 10 →                                       -- c = a + 10
  (a - b) * (a - c) / (b - c) = -10 := by             -- Prove that the expression equals -10
sorry


end NUMINAMATH_CALUDE_multiples_of_five_relation_l3282_328207


namespace NUMINAMATH_CALUDE_alternating_series_sum_l3282_328239

def S (n : ℕ) : ℤ :=
  if n % 2 = 0 then -n / 2 else (n + 1) / 2

theorem alternating_series_sum (a b : ℕ) :
  (S a + S b + S (a + b) = 1) ↔ (Odd a ∧ Odd b) :=
sorry

end NUMINAMATH_CALUDE_alternating_series_sum_l3282_328239


namespace NUMINAMATH_CALUDE_rsa_congruence_l3282_328237

theorem rsa_congruence (p q e d M : ℕ) : 
  Nat.Prime p → 
  Nat.Prime q → 
  p ≠ q → 
  (e * d) % ((p - 1) * (q - 1)) = 1 → 
  ((M ^ e) ^ d) % (p * q) = M % (p * q) := by
sorry

end NUMINAMATH_CALUDE_rsa_congruence_l3282_328237


namespace NUMINAMATH_CALUDE_carpet_shaded_area_l3282_328283

theorem carpet_shaded_area (S T : ℝ) : 
  12 / S = 4 →
  S / T = 2 →
  S > 0 →
  T > 0 →
  S^2 + 8 * T^2 = 27 := by
sorry

end NUMINAMATH_CALUDE_carpet_shaded_area_l3282_328283


namespace NUMINAMATH_CALUDE_f_of_4_eq_17_g_of_2_eq_29_l3282_328235

-- Define the function f
def f (x : ℝ) : ℝ := 5 * x - 3

-- Define the function g
def g (t : ℝ) : ℝ := 4 * t^3 + 2 * t - 7

-- Theorem for f(4) = 17
theorem f_of_4_eq_17 : f 4 = 17 := by sorry

-- Theorem for g(2) = 29
theorem g_of_2_eq_29 : g 2 = 29 := by sorry

end NUMINAMATH_CALUDE_f_of_4_eq_17_g_of_2_eq_29_l3282_328235


namespace NUMINAMATH_CALUDE_womens_doubles_handshakes_l3282_328274

/-- The number of handshakes in a women's doubles tennis tournament -/
theorem womens_doubles_handshakes : 
  let total_players : ℕ := 8
  let players_per_team : ℕ := 2
  let total_teams : ℕ := total_players / players_per_team
  let handshakes_per_player : ℕ := total_players - players_per_team
  total_players * handshakes_per_player / 2 = 24 := by
  sorry

end NUMINAMATH_CALUDE_womens_doubles_handshakes_l3282_328274


namespace NUMINAMATH_CALUDE_external_tangent_intersections_collinear_l3282_328245

-- Define a circle in 2D space
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the concept of disjoint circles
def disjoint (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x1 - x2)^2 + (y1 - y2)^2 > (c1.radius + c2.radius)^2

-- Define the intersection point of external tangents
def external_tangent_intersection (c1 c2 : Circle) : ℝ × ℝ :=
  sorry -- The actual computation is not needed for the statement

-- Define collinearity of three points
def collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (y2 - y1) * (x3 - x1) = (y3 - y1) * (x2 - x1)

-- The main theorem
theorem external_tangent_intersections_collinear (C1 C2 C3 : Circle)
  (h12 : disjoint C1 C2) (h23 : disjoint C2 C3) (h31 : disjoint C3 C1) :
  let T12 := external_tangent_intersection C1 C2
  let T23 := external_tangent_intersection C2 C3
  let T31 := external_tangent_intersection C3 C1
  collinear T12 T23 T31 :=
sorry

end NUMINAMATH_CALUDE_external_tangent_intersections_collinear_l3282_328245


namespace NUMINAMATH_CALUDE_suraj_innings_l3282_328248

/-- The number of innings Suraj played before the last one -/
def n : ℕ := sorry

/-- Suraj's average before the last innings -/
def A : ℚ := sorry

/-- Suraj's new average after the last innings -/
def new_average : ℚ := 28

/-- The increase in Suraj's average after the last innings -/
def average_increase : ℚ := 8

/-- The runs Suraj scored in the last innings -/
def last_innings_runs : ℕ := 140

theorem suraj_innings : 
  (n : ℚ) * A + last_innings_runs = (n + 1) * new_average ∧
  new_average = A + average_increase ∧
  n = 14 := by sorry

end NUMINAMATH_CALUDE_suraj_innings_l3282_328248


namespace NUMINAMATH_CALUDE_average_monthly_income_l3282_328281

/-- Given a person's expenses and savings over a year, calculate their average monthly income. -/
theorem average_monthly_income
  (expense_first_3_months : ℕ)
  (expense_next_4_months : ℕ)
  (expense_last_5_months : ℕ)
  (yearly_savings : ℕ)
  (h1 : expense_first_3_months = 1700)
  (h2 : expense_next_4_months = 1550)
  (h3 : expense_last_5_months = 1800)
  (h4 : yearly_savings = 5200) :
  (3 * expense_first_3_months + 4 * expense_next_4_months + 5 * expense_last_5_months + yearly_savings) / 12 = 2125 := by
  sorry

end NUMINAMATH_CALUDE_average_monthly_income_l3282_328281


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_range_l3282_328276

/-- A point in the fourth quadrant has positive x-coordinate and negative y-coordinate -/
def fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

/-- The theorem states that if a point P(a, a-2) is in the fourth quadrant, then 0 < a < 2 -/
theorem point_in_fourth_quadrant_range (a : ℝ) :
  fourth_quadrant a (a - 2) → 0 < a ∧ a < 2 := by
  sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_range_l3282_328276


namespace NUMINAMATH_CALUDE_sixth_sum_is_189_l3282_328220

/-- A sequence and its partial sums satisfying the given condition -/
def SequenceWithSum (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → S n = 2 * a n - 3

/-- The sixth partial sum of the sequence is 189 -/
theorem sixth_sum_is_189 (a : ℕ → ℝ) (S : ℕ → ℝ) 
    (h : SequenceWithSum a S) : S 6 = 189 := by
  sorry

end NUMINAMATH_CALUDE_sixth_sum_is_189_l3282_328220


namespace NUMINAMATH_CALUDE_passengers_remaining_approx_40_l3282_328250

/-- Calculates the number of passengers remaining after three stops -/
def passengers_after_stops (initial : ℕ) : ℚ :=
  let after_first := initial - (initial / 3)
  let after_second := after_first - (after_first / 4)
  let after_third := after_second - (after_second / 5)
  after_third

/-- Theorem: Given 100 initial passengers and three stops with specified fractions of passengers getting off, 
    the number of remaining passengers is approximately 40 -/
theorem passengers_remaining_approx_40 :
  ∃ ε > 0, ε < 1 ∧ |passengers_after_stops 100 - 40| < ε :=
sorry

end NUMINAMATH_CALUDE_passengers_remaining_approx_40_l3282_328250


namespace NUMINAMATH_CALUDE_fraction_decomposition_l3282_328298

theorem fraction_decomposition (A B : ℚ) :
  (∀ x : ℚ, x ≠ -4 ∧ x ≠ 2/3 →
    (7 * x - 15) / (3 * x^2 + 2 * x - 8) = A / (x + 4) + B / (3 * x - 2)) →
  A = 43 / 14 ∧ B = -31 / 14 := by
sorry

end NUMINAMATH_CALUDE_fraction_decomposition_l3282_328298


namespace NUMINAMATH_CALUDE_peas_soybean_mixture_ratio_l3282_328256

/-- Proves that the ratio of peas to soybean in a mixture costing Rs. 19/kg is 2:1,
    given that peas cost Rs. 16/kg and soybean costs Rs. 25/kg. -/
theorem peas_soybean_mixture_ratio : 
  ∀ (x y : ℝ), 
    x > 0 → y > 0 →
    16 * x + 25 * y = 19 * (x + y) →
    x / y = 2 := by
  sorry

end NUMINAMATH_CALUDE_peas_soybean_mixture_ratio_l3282_328256


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l3282_328284

theorem polynomial_division_remainder :
  ∃ q : Polynomial ℝ, (X^4 + 3•X^2 - 4) = (X^2 + 2) * q + (X^2 - 4) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l3282_328284


namespace NUMINAMATH_CALUDE_sector_area_l3282_328215

/-- The area of a circular sector with radius 12 meters and central angle 42 degrees -/
theorem sector_area : 
  let r : ℝ := 12
  let θ : ℝ := 42
  let sector_area := (θ / 360) * Real.pi * r^2
  sector_area = (42 / 360) * Real.pi * 12^2 := by sorry

end NUMINAMATH_CALUDE_sector_area_l3282_328215


namespace NUMINAMATH_CALUDE_product_of_repeating_decimals_division_of_repeating_decimals_l3282_328241

-- Define the repeating decimals
def repeating_decimal_18 : ℚ := 2 / 11
def repeating_decimal_36 : ℚ := 4 / 11

-- Theorem for the product
theorem product_of_repeating_decimals :
  repeating_decimal_18 * repeating_decimal_36 = 8 / 121 := by
  sorry

-- Theorem for the division
theorem division_of_repeating_decimals :
  repeating_decimal_18 / repeating_decimal_36 = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_product_of_repeating_decimals_division_of_repeating_decimals_l3282_328241


namespace NUMINAMATH_CALUDE_negation_equivalence_l3282_328263

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 + x + 1 ≤ 0) ↔ (∀ x : ℝ, x^2 + x + 1 > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3282_328263


namespace NUMINAMATH_CALUDE_longest_side_of_equal_area_rectangles_l3282_328214

/-- Represents a rectangle with integer sides -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- The area of a rectangle -/
def area (r : Rectangle) : ℕ := r.width * r.height

theorem longest_side_of_equal_area_rectangles 
  (r1 r2 r3 : Rectangle) 
  (h_equal_areas : area r1 = area r2 ∧ area r2 = area r3)
  (h_one_side_19 : r1.width = 19 ∨ r1.height = 19 ∨ 
                   r2.width = 19 ∨ r2.height = 19 ∨ 
                   r3.width = 19 ∨ r3.height = 19) :
  ∃ (r : Rectangle), (r = r1 ∨ r = r2 ∨ r = r3) ∧ 
    (r.width = 380 ∨ r.height = 380) :=
sorry

end NUMINAMATH_CALUDE_longest_side_of_equal_area_rectangles_l3282_328214


namespace NUMINAMATH_CALUDE_larry_expression_equality_l3282_328228

theorem larry_expression_equality (e : ℝ) : 
  let a : ℝ := 2
  let b : ℝ := 1
  let c : ℝ := 4
  let d : ℝ := 5
  (a + (b - (c + (d + e)))) = (a + b - c - d - e) :=
by sorry

end NUMINAMATH_CALUDE_larry_expression_equality_l3282_328228


namespace NUMINAMATH_CALUDE_right_triangle_7_24_25_l3282_328290

theorem right_triangle_7_24_25 (a b c : ℝ) (h1 : a = 7) (h2 : b = 24) (h3 : c = 25) :
  a^2 + b^2 = c^2 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_7_24_25_l3282_328290


namespace NUMINAMATH_CALUDE_dime_probability_l3282_328230

/-- Represents the types of coins in the jar -/
inductive Coin
  | Quarter
  | Dime
  | Penny

/-- The value of each coin type in cents -/
def coinValue : Coin → ℕ
  | Coin.Quarter => 25
  | Coin.Dime => 10
  | Coin.Penny => 1

/-- The total value of each coin type in the jar in cents -/
def totalValue : Coin → ℕ
  | Coin.Quarter => 1250
  | Coin.Dime => 500
  | Coin.Penny => 250

/-- The number of coins of each type in the jar -/
def coinCount (c : Coin) : ℕ := totalValue c / coinValue c

/-- The total number of coins in the jar -/
def totalCoins : ℕ := coinCount Coin.Quarter + coinCount Coin.Dime + coinCount Coin.Penny

/-- The probability of randomly choosing a dime from the jar -/
def probDime : ℚ := coinCount Coin.Dime / totalCoins

theorem dime_probability : probDime = 1 / 7 := by
  sorry

#eval probDime

end NUMINAMATH_CALUDE_dime_probability_l3282_328230


namespace NUMINAMATH_CALUDE_problem_statement_l3282_328236

theorem problem_statement (x : ℝ) (h : x + 1/x = 7) : 
  (x - 3)^2 + 49 / (x - 3)^2 = 23 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l3282_328236


namespace NUMINAMATH_CALUDE_boxes_with_neither_pens_nor_pencils_l3282_328202

theorem boxes_with_neither_pens_nor_pencils 
  (total_boxes : ℕ) 
  (boxes_with_pencils : ℕ) 
  (boxes_with_pens : ℕ) 
  (boxes_with_both : ℕ) 
  (h1 : total_boxes = 15)
  (h2 : boxes_with_pencils = 7)
  (h3 : boxes_with_pens = 4)
  (h4 : boxes_with_both = 3) :
  total_boxes - (boxes_with_pencils + boxes_with_pens - boxes_with_both) = 7 :=
by sorry

end NUMINAMATH_CALUDE_boxes_with_neither_pens_nor_pencils_l3282_328202


namespace NUMINAMATH_CALUDE_zach_cookies_theorem_l3282_328269

/-- The number of cookies Zach baked over three days --/
def total_cookies (monday_cookies : ℕ) : ℕ :=
  let tuesday_cookies := monday_cookies / 2
  let wednesday_cookies := tuesday_cookies * 3
  monday_cookies + tuesday_cookies + wednesday_cookies - 4

/-- Theorem stating that Zach had 92 cookies at the end of three days --/
theorem zach_cookies_theorem :
  total_cookies 32 = 92 :=
by sorry

end NUMINAMATH_CALUDE_zach_cookies_theorem_l3282_328269


namespace NUMINAMATH_CALUDE_restaurant_glasses_count_l3282_328238

theorem restaurant_glasses_count :
  ∀ (x y : ℕ),
  -- x is the number of 12-glass boxes, y is the number of 16-glass boxes
  y = x + 16 →
  -- The average number of glasses per box is 15
  (12 * x + 16 * y) / (x + y) = 15 →
  -- The total number of glasses is 480
  12 * x + 16 * y = 480 :=
by
  sorry

end NUMINAMATH_CALUDE_restaurant_glasses_count_l3282_328238


namespace NUMINAMATH_CALUDE_food_price_calculation_l3282_328224

/-- The original food price before tax and tip -/
def original_price : ℝ := 160

/-- The sales tax rate -/
def sales_tax_rate : ℝ := 0.1

/-- The tip rate -/
def tip_rate : ℝ := 0.2

/-- The total bill amount -/
def total_bill : ℝ := 211.20

theorem food_price_calculation :
  original_price * (1 + sales_tax_rate) * (1 + tip_rate) = total_bill := by
  sorry


end NUMINAMATH_CALUDE_food_price_calculation_l3282_328224


namespace NUMINAMATH_CALUDE_replacement_cost_theorem_l3282_328223

/-- Calculate the total cost of replacing cardio machines in multiple gyms -/
def total_replacement_cost (num_gyms : ℕ) (bikes_per_gym : ℕ) (treadmills_per_gym : ℕ) (ellipticals_per_gym : ℕ) (bike_cost : ℝ) : ℝ :=
  let total_bikes := num_gyms * bikes_per_gym
  let total_treadmills := num_gyms * treadmills_per_gym
  let total_ellipticals := num_gyms * ellipticals_per_gym
  let treadmill_cost := bike_cost * 1.5
  let elliptical_cost := treadmill_cost * 2
  total_bikes * bike_cost + total_treadmills * treadmill_cost + total_ellipticals * elliptical_cost

/-- Theorem: The total cost to replace all cardio machines in 20 gyms is $455,000 -/
theorem replacement_cost_theorem :
  total_replacement_cost 20 10 5 5 700 = 455000 := by
  sorry

end NUMINAMATH_CALUDE_replacement_cost_theorem_l3282_328223


namespace NUMINAMATH_CALUDE_binomial_15_4_l3282_328255

theorem binomial_15_4 : Nat.choose 15 4 = 1365 := by
  sorry

end NUMINAMATH_CALUDE_binomial_15_4_l3282_328255


namespace NUMINAMATH_CALUDE_tripled_minus_six_l3282_328259

theorem tripled_minus_six (x : ℝ) : 3 * x - 6 = 15 → x = 7 := by
  sorry

end NUMINAMATH_CALUDE_tripled_minus_six_l3282_328259


namespace NUMINAMATH_CALUDE_tangent_point_coordinates_l3282_328267

/-- The function representing the curve y = x^3 + x^2 -/
def f (x : ℝ) : ℝ := x^3 + x^2

/-- The derivative of f -/
def f' (x : ℝ) : ℝ := 3*x^2 + 2*x

theorem tangent_point_coordinates :
  ∀ a : ℝ, f' a = 4 → (a = 1 ∧ f a = 2) ∨ (a = -1 ∧ f a = -2) :=
by sorry

end NUMINAMATH_CALUDE_tangent_point_coordinates_l3282_328267


namespace NUMINAMATH_CALUDE_marcel_total_cost_l3282_328219

def calculate_total_cost (pen_price : ℝ) : ℝ :=
  let briefcase_price := 5 * pen_price
  let notebook_price := 2 * pen_price
  let calculator_price := 3 * notebook_price
  let briefcase_discount := 0.15 * briefcase_price
  let discounted_briefcase_price := briefcase_price - briefcase_discount
  let total_before_tax := pen_price + discounted_briefcase_price + notebook_price + calculator_price
  let tax := 0.10 * total_before_tax
  total_before_tax + tax

theorem marcel_total_cost :
  calculate_total_cost 4 = 58.30 := by sorry

end NUMINAMATH_CALUDE_marcel_total_cost_l3282_328219


namespace NUMINAMATH_CALUDE_expression_evaluation_l3282_328200

theorem expression_evaluation :
  let x : ℝ := Real.sqrt 2 - 1
  let expr := ((x / (x - 1)) - (x / (x^2 - 1))) / ((x^2 - x) / (x^2 - 2*x + 1))
  expr = 1 - Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3282_328200


namespace NUMINAMATH_CALUDE_equality_of_fractions_l3282_328288

theorem equality_of_fractions (x y z k : ℝ) :
  (9 / (x + y) = k / (x + z)) ∧ (k / (x + z) = 15 / (z - y)) → k = 24 := by
  sorry

end NUMINAMATH_CALUDE_equality_of_fractions_l3282_328288


namespace NUMINAMATH_CALUDE_expression_equality_l3282_328294

theorem expression_equality : 
  Real.sqrt 27 / (Real.sqrt 3 / 2) * (2 * Real.sqrt 2) - 6 * Real.sqrt 2 = 6 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l3282_328294


namespace NUMINAMATH_CALUDE_population_changes_l3282_328280

/-- Enumeration of possible population number changes --/
inductive PopulationChange
  | Increase
  | Decrease
  | Fluctuation
  | Extinction

/-- Theorem stating that population changes can be increase, decrease, fluctuation, or extinction --/
theorem population_changes : 
  ∀ (change : PopulationChange), 
    change = PopulationChange.Increase ∨
    change = PopulationChange.Decrease ∨
    change = PopulationChange.Fluctuation ∨
    change = PopulationChange.Extinction :=
by
  sorry

#check population_changes

end NUMINAMATH_CALUDE_population_changes_l3282_328280


namespace NUMINAMATH_CALUDE_baseball_cards_distribution_l3282_328286

theorem baseball_cards_distribution (total_cards : ℕ) (num_friends : ℕ) (cards_per_friend : ℕ) :
  total_cards = 24 →
  num_friends = 4 →
  total_cards = num_friends * cards_per_friend →
  cards_per_friend = 6 := by
  sorry

end NUMINAMATH_CALUDE_baseball_cards_distribution_l3282_328286


namespace NUMINAMATH_CALUDE_mat_length_approximation_l3282_328275

/-- Represents the setup of a circular table with place mats -/
structure TableSetup where
  tableRadius : ℝ
  numMats : ℕ
  matWidth : ℝ

/-- Calculates the length of place mats given a table setup -/
def calculateMatLength (setup : TableSetup) : ℝ :=
  sorry

/-- Theorem stating that for the given setup, the mat length is approximately 3.9308 meters -/
theorem mat_length_approximation (setup : TableSetup) 
  (h1 : setup.tableRadius = 6)
  (h2 : setup.numMats = 8)
  (h3 : setup.matWidth = 1.5) :
  abs (calculateMatLength setup - 3.9308) < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_mat_length_approximation_l3282_328275


namespace NUMINAMATH_CALUDE_board_game_cost_l3282_328212

theorem board_game_cost (jump_rope_cost playground_ball_cost dalton_savings uncle_gift additional_needed : ℕ) 
  (h1 : jump_rope_cost = 7)
  (h2 : playground_ball_cost = 4)
  (h3 : dalton_savings = 6)
  (h4 : uncle_gift = 13)
  (h5 : additional_needed = 4) :
  jump_rope_cost + playground_ball_cost + (dalton_savings + uncle_gift + additional_needed) - (dalton_savings + uncle_gift) = 12 := by
  sorry

end NUMINAMATH_CALUDE_board_game_cost_l3282_328212


namespace NUMINAMATH_CALUDE_sum_of_five_reals_l3282_328233

theorem sum_of_five_reals (a b c d e : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d) (pos_e : 0 < e)
  (eq1 : a + b = c)
  (eq2 : a + b + c = d)
  (eq3 : a + b + c + d = e)
  (c_val : c = 5) : 
  a + b + c + d + e = 40 := by
sorry

end NUMINAMATH_CALUDE_sum_of_five_reals_l3282_328233


namespace NUMINAMATH_CALUDE_football_team_progress_l3282_328273

/-- Calculates the net progress of a football team given yards lost and gained -/
def netProgress (yardsLost : Int) (yardsGained : Int) : Int :=
  yardsGained - yardsLost

/-- Proves that when a team loses 5 yards and gains 10 yards, their net progress is 5 yards -/
theorem football_team_progress : netProgress 5 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_football_team_progress_l3282_328273


namespace NUMINAMATH_CALUDE_cable_car_travel_time_l3282_328222

/-- Represents the time in minutes to travel half the circular route -/
def travel_time : ℝ := 22.5

/-- Represents the number of cable cars on the circular route -/
def num_cars : ℕ := 80

/-- Represents the time interval in seconds between encounters with opposing cars -/
def encounter_interval : ℝ := 15

/-- Theorem stating that given the conditions, the travel time from A to B is 22.5 minutes -/
theorem cable_car_travel_time :
  ∀ (cars : ℕ) (interval : ℝ),
  cars = num_cars →
  interval = encounter_interval →
  travel_time = (cars : ℝ) * interval / (2 * 60) :=
by sorry

end NUMINAMATH_CALUDE_cable_car_travel_time_l3282_328222


namespace NUMINAMATH_CALUDE_equation_solution_l3282_328277

theorem equation_solution (a : ℝ) (h1 : a ≠ -2) (h2 : a ≠ -3) (h3 : a ≠ 1/2) :
  let x : ℝ := (2*a - 1) / (a + 3)
  (2 : ℝ) ^ ((a + 3) / (a + 2)) * (32 : ℝ) ^ (1 / (x * (a + 2))) = (4 : ℝ) ^ (1 / x) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3282_328277


namespace NUMINAMATH_CALUDE_compound_composition_l3282_328242

/-- Represents the number of atoms of each element in the compound -/
structure Compound where
  h : ℕ
  c : ℕ
  o : ℕ

/-- Atomic weights of elements in g/mol -/
def atomic_weight (element : String) : ℝ :=
  match element with
  | "H" => 1
  | "C" => 12
  | "O" => 16
  | _ => 0

/-- Calculate the molecular weight of a compound -/
def molecular_weight (comp : Compound) : ℝ :=
  comp.h * atomic_weight "H" + comp.c * atomic_weight "C" + comp.o * atomic_weight "O"

/-- The main theorem to prove -/
theorem compound_composition :
  ∃ (comp : Compound), comp.h = 2 ∧ comp.o = 3 ∧ molecular_weight comp = 62 ∧ comp.c = 1 :=
by sorry

end NUMINAMATH_CALUDE_compound_composition_l3282_328242


namespace NUMINAMATH_CALUDE_max_n_with_2013_trailing_zeros_l3282_328232

/-- Count the number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125) + (n / 625) + (n / 3125)

/-- The maximum value of N such that N! has exactly 2013 trailing zeros -/
theorem max_n_with_2013_trailing_zeros :
  ∀ n : ℕ, n > 8069 → trailingZeros n > 2013 ∧
  trailingZeros 8069 = 2013 :=
by sorry

end NUMINAMATH_CALUDE_max_n_with_2013_trailing_zeros_l3282_328232


namespace NUMINAMATH_CALUDE_wy_equals_uv_l3282_328246

-- Define the variables
variable (u v w y : ℝ)
variable (α β : ℝ)

-- Define the conditions
axiom sin_roots : (Real.sin α)^2 - u * (Real.sin α) + v = 0 ∧ (Real.sin β)^2 - u * (Real.sin β) + v = 0
axiom cos_roots : (Real.cos α)^2 - w * (Real.cos α) + y = 0 ∧ (Real.cos β)^2 - w * (Real.cos β) + y = 0
axiom right_triangle : Real.sin α = Real.cos β ∧ Real.sin β = Real.cos α

-- State the theorem
theorem wy_equals_uv : wy = uv := by sorry

end NUMINAMATH_CALUDE_wy_equals_uv_l3282_328246


namespace NUMINAMATH_CALUDE_calculate_e_l3282_328295

/-- Given the relationships between variables j, p, t, b, a, and e, prove that e = 21.5 -/
theorem calculate_e (j p t b a e : ℝ) 
  (h1 : j = 0.75 * p)
  (h2 : j = 0.80 * t)
  (h3 : t = p - (e / 100) * p)
  (h4 : b = 1.40 * j)
  (h5 : a = 0.85 * b)
  (h6 : e = 2 * ((p - a) / p) * 100) :
  e = 21.5 := by
  sorry

end NUMINAMATH_CALUDE_calculate_e_l3282_328295


namespace NUMINAMATH_CALUDE_old_workers_in_sample_l3282_328208

/-- Represents the composition of workers in a unit -/
structure WorkerComposition where
  total : ℕ
  young : ℕ
  old : ℕ
  middleAged : ℕ
  young_count : young ≤ total
  middleAged_relation : middleAged = 2 * old
  total_sum : total = young + old + middleAged

/-- Represents a stratified sample of workers -/
structure StratifiedSample where
  composition : WorkerComposition
  young_sample : ℕ
  young_sample_valid : young_sample ≤ composition.young

/-- Theorem stating the number of old workers in the stratified sample -/
theorem old_workers_in_sample (unit : WorkerComposition) (sample : StratifiedSample)
    (h_unit : unit.total = 430 ∧ unit.young = 160)
    (h_sample : sample.composition = unit ∧ sample.young_sample = 32) :
    (sample.young_sample : ℚ) / unit.young * unit.old = 18 := by
  sorry

end NUMINAMATH_CALUDE_old_workers_in_sample_l3282_328208


namespace NUMINAMATH_CALUDE_log_equation_solution_l3282_328227

theorem log_equation_solution (y : ℝ) (h : y > 0) : 
  Real.log y / Real.log 3 + Real.log y / Real.log 9 = 5 → y = 3^(10/3) := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l3282_328227


namespace NUMINAMATH_CALUDE_a_will_eat_hat_l3282_328293

-- Define the types of people
inductive Person : Type
| Knight : Person
| Liar : Person

-- Define the statement made by A about B
def statement_about_B (a b : Person) : Prop :=
  match a with
  | Person.Knight => b = Person.Knight
  | Person.Liar => True

-- Define A's statement about eating the hat
def statement_about_hat (a : Person) : Prop :=
  match a with
  | Person.Knight => True  -- Will eat the hat
  | Person.Liar => False   -- Won't eat the hat

-- Theorem statement
theorem a_will_eat_hat (a b : Person) :
  (statement_about_B a b = True) →
  (statement_about_hat a = True) := by
  sorry


end NUMINAMATH_CALUDE_a_will_eat_hat_l3282_328293


namespace NUMINAMATH_CALUDE_derivative_at_one_l3282_328211

-- Define the function
def f (x : ℝ) : ℝ := x^2 + 2*x + 1

-- State the theorem
theorem derivative_at_one : 
  deriv f 1 = 4 := by sorry

end NUMINAMATH_CALUDE_derivative_at_one_l3282_328211


namespace NUMINAMATH_CALUDE_inequality_system_solution_l3282_328229

theorem inequality_system_solution (a b : ℝ) :
  (∀ x : ℝ, -1 < x ∧ x < 1 ↔ x - a > 2 ∧ 2*x - b < 0) →
  a^(-b) = 1/9 := by
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l3282_328229


namespace NUMINAMATH_CALUDE_min_value_of_expression_l3282_328204

theorem min_value_of_expression :
  (∀ x y : ℝ, x^2 + 2*x*y + y^2 ≥ 0) ∧
  (∃ x y : ℝ, x^2 + 2*x*y + y^2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l3282_328204


namespace NUMINAMATH_CALUDE_not_P_necessary_not_sufficient_for_not_Q_l3282_328210

-- Define the propositions P and Q as functions from ℝ to Prop
def P (x : ℝ) : Prop := |2*x - 3| > 1
def Q (x : ℝ) : Prop := x^2 - 3*x + 2 ≥ 0

-- Define the relationship between ¬P and ¬Q
theorem not_P_necessary_not_sufficient_for_not_Q :
  (∀ x, ¬(Q x) → ¬(P x)) ∧ 
  ¬(∀ x, ¬(P x) → ¬(Q x)) :=
sorry

end NUMINAMATH_CALUDE_not_P_necessary_not_sufficient_for_not_Q_l3282_328210


namespace NUMINAMATH_CALUDE_prob_adjacent_vertices_decagon_l3282_328278

/-- A decagon is a polygon with 10 vertices -/
def Decagon := { n : ℕ // n = 10 }

/-- The number of vertices in a decagon -/
def num_vertices (d : Decagon) : ℕ := d.val

/-- The number of adjacent vertices for any vertex in a decagon -/
def num_adjacent_vertices (d : Decagon) : ℕ := 2

/-- The probability of choosing two distinct adjacent vertices in a decagon -/
def prob_adjacent_vertices (d : Decagon) : ℚ :=
  (num_adjacent_vertices d : ℚ) / ((num_vertices d - 1) : ℚ)

/-- Theorem: The probability of choosing two distinct adjacent vertices in a decagon is 2/9 -/
theorem prob_adjacent_vertices_decagon (d : Decagon) :
  prob_adjacent_vertices d = 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_prob_adjacent_vertices_decagon_l3282_328278


namespace NUMINAMATH_CALUDE_travel_distance_ratio_l3282_328225

theorem travel_distance_ratio (total_distance train_distance : ℝ)
  (h1 : total_distance = 500)
  (h2 : train_distance = 300)
  (h3 : ∃ bus_distance cab_distance : ℝ,
    total_distance = train_distance + bus_distance + cab_distance ∧
    cab_distance = (1/3) * bus_distance) :
  ∃ bus_distance : ℝ, bus_distance / train_distance = 1/2 :=
sorry

end NUMINAMATH_CALUDE_travel_distance_ratio_l3282_328225


namespace NUMINAMATH_CALUDE_length_PQ_value_l3282_328265

/-- Triangle ABC with given side lengths and angle bisectors --/
structure TriangleABC where
  -- Side lengths
  AB : ℝ
  BC : ℝ
  CA : ℝ
  -- AH is altitude
  AH : ℝ
  -- Q and P are intersection points of angle bisectors with altitude
  AQ : ℝ
  AP : ℝ
  -- Conditions
  side_lengths : AB = 6 ∧ BC = 10 ∧ CA = 8
  altitude : AH = 4.8
  angle_bisector_intersections : AQ = 20/3 ∧ AP = 3

/-- The length of PQ in the given triangle configuration --/
def length_PQ (t : TriangleABC) : ℝ := t.AQ - t.AP

/-- Theorem stating that the length of PQ is 3.67 --/
theorem length_PQ_value (t : TriangleABC) : length_PQ t = 3.67 := by
  sorry

end NUMINAMATH_CALUDE_length_PQ_value_l3282_328265


namespace NUMINAMATH_CALUDE_unique_function_property_l3282_328213

theorem unique_function_property (f : ℕ → ℕ) :
  (∀ n : ℕ, f n + f (f n) = 2 * n) ↔ (∀ n : ℕ, f n = n) := by
  sorry

end NUMINAMATH_CALUDE_unique_function_property_l3282_328213


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3282_328252

theorem inequality_solution_set (x : ℝ) : 
  x^2 - 2*|x| - 15 > 0 ↔ x < -5 ∨ x > 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3282_328252


namespace NUMINAMATH_CALUDE_smallest_root_property_l3282_328266

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop := x^2 - 9*x - 10 = 0

-- Define a as the smallest root
def a : ℝ := sorry

-- State the properties of a
axiom a_is_root : quadratic_equation a
axiom a_is_smallest : ∀ x, quadratic_equation x → a ≤ x

-- Theorem to prove
theorem smallest_root_property : a^4 - 909*a = 910 := by sorry

end NUMINAMATH_CALUDE_smallest_root_property_l3282_328266


namespace NUMINAMATH_CALUDE_star_two_neg_three_l3282_328206

-- Define the new operation
def star (a b : ℤ) : ℤ := a * b - (a + b)

-- State the theorem
theorem star_two_neg_three : star 2 (-3) = -5 := by
  sorry

end NUMINAMATH_CALUDE_star_two_neg_three_l3282_328206

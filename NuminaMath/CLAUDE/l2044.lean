import Mathlib

namespace NUMINAMATH_CALUDE_ellipse_perpendicular_chord_bounds_l2044_204432

/-- Given an ellipse (x²/a²) + (y²/b²) = 1 with a > b > 0, for any two points A and B on the ellipse
    such that OA ⊥ OB, the distance |AB| satisfies (ab / √(a² + b²)) ≤ |AB| ≤ √(a² + b²) -/
theorem ellipse_perpendicular_chord_bounds (a b : ℝ) (ha : 0 < b) (hab : b < a) :
  ∀ (A B : ℝ × ℝ),
    (A.1^2 / a^2 + A.2^2 / b^2 = 1) →
    (B.1^2 / a^2 + B.2^2 / b^2 = 1) →
    (A.1 * B.1 + A.2 * B.2 = 0) →
    (a * b / Real.sqrt (a^2 + b^2) ≤ Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)) ∧
    (Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) ≤ Real.sqrt (a^2 + b^2)) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_perpendicular_chord_bounds_l2044_204432


namespace NUMINAMATH_CALUDE_digit_150_of_3_over_11_l2044_204490

theorem digit_150_of_3_over_11 : ∃ (d : ℕ), d = 7 ∧ 
  (∀ (n : ℕ), n ≥ 1 → n ≤ 150 → 
    (3 * 10^n) % 11 = (d * 10^(150 - n)) % 11) := by
  sorry

end NUMINAMATH_CALUDE_digit_150_of_3_over_11_l2044_204490


namespace NUMINAMATH_CALUDE_max_value_of_a_l2044_204455

theorem max_value_of_a (x y a : ℝ) 
  (h1 : x > 1/3) 
  (h2 : y > 1) 
  (h3 : ∀ (x y : ℝ), x > 1/3 → y > 1 → 
    (9 * x^2) / (a^2 * (y-1)) + (y^2) / (a^2 * (3*x-1)) ≥ 1) : 
  a ≤ 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_max_value_of_a_l2044_204455


namespace NUMINAMATH_CALUDE_fixed_point_coordinates_l2044_204464

theorem fixed_point_coordinates : ∃! (A : ℝ × ℝ), ∀ (k : ℝ),
  (3 + k) * A.1 + (1 - 2*k) * A.2 + 1 + 5*k = 0 ∧ A = (-1, 2) := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_coordinates_l2044_204464


namespace NUMINAMATH_CALUDE_interest_difference_proof_l2044_204449

/-- Simple interest calculation -/
def simple_interest (principal rate time : ℚ) : ℚ :=
  principal * rate * time / 100

theorem interest_difference_proof : 
  let principal : ℚ := 2500
  let rate : ℚ := 8
  let time : ℚ := 8
  let interest := simple_interest principal rate time
  principal - interest = 900 := by
  sorry

end NUMINAMATH_CALUDE_interest_difference_proof_l2044_204449


namespace NUMINAMATH_CALUDE_monotone_decreasing_implies_a_leq_neg_four_l2044_204477

/-- A quadratic function f(x) = x^2 + 2ax + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x + 2

/-- The property of f being monotonically decreasing on (-∞, 4] -/
def is_monotone_decreasing_on_interval (a : ℝ) : Prop :=
  ∀ x y, x < y → y ≤ 4 → f a x > f a y

/-- The main theorem: if f is monotonically decreasing on (-∞, 4], then a ≤ -4 -/
theorem monotone_decreasing_implies_a_leq_neg_four (a : ℝ) :
  is_monotone_decreasing_on_interval a → a ≤ -4 :=
by
  sorry

end NUMINAMATH_CALUDE_monotone_decreasing_implies_a_leq_neg_four_l2044_204477


namespace NUMINAMATH_CALUDE_square_area_with_point_l2044_204457

/-- A square with a point inside satisfying certain distance conditions -/
structure SquareWithPoint where
  -- The side length of the square
  a : ℝ
  -- Coordinates of point P
  x : ℝ
  y : ℝ
  -- Conditions
  square_positive : 0 < a
  inside_square : 0 ≤ x ∧ x ≤ a ∧ 0 ≤ y ∧ y ≤ a
  distance_to_A : x^2 + y^2 = 4
  distance_to_B : (a - x)^2 + y^2 = 9
  distance_to_C : (a - x)^2 + (a - y)^2 = 16

/-- The area of a square with a point inside satisfying certain distance conditions is 10 + √63 -/
theorem square_area_with_point (s : SquareWithPoint) : s.a^2 = 10 + Real.sqrt 63 := by
  sorry

end NUMINAMATH_CALUDE_square_area_with_point_l2044_204457


namespace NUMINAMATH_CALUDE_congruence_solution_l2044_204404

theorem congruence_solution : ∃! n : ℕ, n < 47 ∧ (13 * n) % 47 = 15 % 47 := by
  sorry

end NUMINAMATH_CALUDE_congruence_solution_l2044_204404


namespace NUMINAMATH_CALUDE_larger_number_proof_l2044_204468

theorem larger_number_proof (x y : ℝ) 
  (h1 : x - y = 3) 
  (h2 : x + y = 29) 
  (h3 : x * y > 200) : 
  max x y = 16 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l2044_204468


namespace NUMINAMATH_CALUDE_arithmetic_sequence_k_equals_4_l2044_204410

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_k_equals_4
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_first : a 1 = 1)
  (h_diff : ∃ d : ℝ, d = 2 ∧ ∀ n : ℕ, a (n + 1) = a n + d)
  (h_k : ∃ k : ℕ, a k = 7) :
  ∃ k : ℕ, a k = 7 ∧ k = 4 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_k_equals_4_l2044_204410


namespace NUMINAMATH_CALUDE_ball_game_attendance_l2044_204411

/-- The number of children at a ball game -/
def num_children : ℕ :=
  let num_adults : ℕ := 10
  let adult_ticket_price : ℕ := 8
  let child_ticket_price : ℕ := 4
  let total_bill : ℕ := 124
  let adult_cost : ℕ := num_adults * adult_ticket_price
  let child_cost : ℕ := total_bill - adult_cost
  child_cost / child_ticket_price

theorem ball_game_attendance : num_children = 11 := by
  sorry

end NUMINAMATH_CALUDE_ball_game_attendance_l2044_204411


namespace NUMINAMATH_CALUDE_larger_solution_of_quadratic_l2044_204448

theorem larger_solution_of_quadratic (x : ℝ) : 
  x^2 - 13*x + 36 = 0 → x = 9 ∨ x = 4 → 9 > 4 := by
  sorry

end NUMINAMATH_CALUDE_larger_solution_of_quadratic_l2044_204448


namespace NUMINAMATH_CALUDE_bertha_family_without_daughters_l2044_204480

/-- Represents a family tree starting from Bertha -/
structure BerthaFamily where
  daughters : Nat
  daughters_with_children : Nat
  total_descendants : Nat

/-- The conditions of Bertha's family -/
def bertha_family : BerthaFamily := {
  daughters := 6,
  daughters_with_children := 4,
  total_descendants := 30
}

/-- Theorem: The number of Bertha's daughters and granddaughters who have no daughters is 26 -/
theorem bertha_family_without_daughters : 
  (bertha_family.total_descendants - bertha_family.daughters_with_children * bertha_family.daughters) + 
  (bertha_family.daughters - bertha_family.daughters_with_children) = 26 := by
  sorry

#check bertha_family_without_daughters

end NUMINAMATH_CALUDE_bertha_family_without_daughters_l2044_204480


namespace NUMINAMATH_CALUDE_sum_reciprocals_and_range_l2044_204430

theorem sum_reciprocals_and_range (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1 / a + 1 / b ≥ 4) ∧
  ({x : ℝ | ∀ (a b : ℝ), a > 0 → b > 0 → a + b = 1 → |x - 2| + |2*x - 1| ≤ 1 / a + 1 / b} = Set.Icc (-1/3) (7/3)) := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocals_and_range_l2044_204430


namespace NUMINAMATH_CALUDE_total_cases_california_l2044_204450

/-- Calculates the total number of positive Coronavirus cases after three days,
    given the initial number of cases and daily changes. -/
def totalCasesAfterThreeDays (initialCases : ℕ) (newCasesDay2 : ℕ) (recoveriesDay2 : ℕ)
                              (newCasesDay3 : ℕ) (recoveriesDay3 : ℕ) : ℕ :=
  initialCases + (newCasesDay2 - recoveriesDay2) + (newCasesDay3 - recoveriesDay3)

/-- Theorem stating that given the specific numbers from the problem,
    the total number of positive cases after the third day is 3750. -/
theorem total_cases_california : totalCasesAfterThreeDays 2000 500 50 1500 200 = 3750 := by
  sorry

end NUMINAMATH_CALUDE_total_cases_california_l2044_204450


namespace NUMINAMATH_CALUDE_unfair_coin_probability_l2044_204462

def coin_flips (n : ℕ) (p : ℚ) (k : ℕ) : ℚ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

theorem unfair_coin_probability :
  let n : ℕ := 8
  let p_tails : ℚ := 3/4
  let k : ℕ := 3
  coin_flips n p_tails k = 189/128 := by
sorry

end NUMINAMATH_CALUDE_unfair_coin_probability_l2044_204462


namespace NUMINAMATH_CALUDE_determinant_max_value_l2044_204445

theorem determinant_max_value (θ : ℝ) :
  (∀ θ', -Real.sin (4 * θ') / 2 ≤ -Real.sin (4 * θ) / 2) →
  -Real.sin (4 * θ) / 2 = 1/2 := by
sorry

end NUMINAMATH_CALUDE_determinant_max_value_l2044_204445


namespace NUMINAMATH_CALUDE_least_3digit_base8_divisible_by_7_l2044_204429

/-- Converts a base 8 number to decimal --/
def base8ToDecimal (n : ℕ) : ℕ := sorry

/-- Converts a decimal number to base 8 --/
def decimalToBase8 (n : ℕ) : ℕ := sorry

/-- Checks if a number is a 3-digit base 8 number --/
def isThreeDigitBase8 (n : ℕ) : Prop := 
  100 ≤ n ∧ n ≤ 777

theorem least_3digit_base8_divisible_by_7 :
  let n := 106
  isThreeDigitBase8 n ∧ 
  base8ToDecimal n % 7 = 0 ∧
  ∀ m : ℕ, isThreeDigitBase8 m ∧ base8ToDecimal m % 7 = 0 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_least_3digit_base8_divisible_by_7_l2044_204429


namespace NUMINAMATH_CALUDE_expression_evaluation_l2044_204453

theorem expression_evaluation : (3^2 - 3) - 2 * (4^2 - 4) + (5^2 - 5) = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2044_204453


namespace NUMINAMATH_CALUDE_complex_magnitude_l2044_204479

theorem complex_magnitude (z : ℂ) (i : ℂ) (h1 : i * i = -1) (h2 : (1 - z) * i = 2) : Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l2044_204479


namespace NUMINAMATH_CALUDE_room_length_l2044_204481

/-- The length of a room given its width, total paving cost, and paving rate per square meter. -/
theorem room_length (width : ℝ) (total_cost : ℝ) (rate_per_sqm : ℝ) 
  (h_width : width = 3.75)
  (h_total_cost : total_cost = 12375)
  (h_rate : rate_per_sqm = 600) : 
  total_cost / rate_per_sqm / width = 5.5 := by
sorry

#eval (12375 / 600 / 3.75 : Float)

end NUMINAMATH_CALUDE_room_length_l2044_204481


namespace NUMINAMATH_CALUDE_triangle_angle_less_than_right_angle_l2044_204403

theorem triangle_angle_less_than_right_angle 
  (A B C : ℝ) (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : 2/b = 1/a + 1/c) : B < π/2 :=
sorry

end NUMINAMATH_CALUDE_triangle_angle_less_than_right_angle_l2044_204403


namespace NUMINAMATH_CALUDE_wire_length_problem_l2044_204407

theorem wire_length_problem (shorter_piece longer_piece total_length : ℝ) : 
  shorter_piece = 20 →
  longer_piece = 2/4 * shorter_piece →
  total_length = shorter_piece + longer_piece →
  total_length = 60 := by
sorry

end NUMINAMATH_CALUDE_wire_length_problem_l2044_204407


namespace NUMINAMATH_CALUDE_sum_with_twenty_equals_thirty_l2044_204419

theorem sum_with_twenty_equals_thirty (x : ℝ) : 20 + x = 30 → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_with_twenty_equals_thirty_l2044_204419


namespace NUMINAMATH_CALUDE_fraction_equals_zero_l2044_204418

theorem fraction_equals_zero (x : ℝ) : 
  (x - 5) / (4 * x^2 - 1) = 0 ↔ x = 5 := by sorry

end NUMINAMATH_CALUDE_fraction_equals_zero_l2044_204418


namespace NUMINAMATH_CALUDE_product_of_ten_proper_fractions_equals_one_tenth_l2044_204408

theorem product_of_ten_proper_fractions_equals_one_tenth :
  ∃ (a b c d e f g h i j : ℚ),
    (0 < a ∧ a < 1) ∧
    (0 < b ∧ b < 1) ∧
    (0 < c ∧ c < 1) ∧
    (0 < d ∧ d < 1) ∧
    (0 < e ∧ e < 1) ∧
    (0 < f ∧ f < 1) ∧
    (0 < g ∧ g < 1) ∧
    (0 < h ∧ h < 1) ∧
    (0 < i ∧ i < 1) ∧
    (0 < j ∧ j < 1) ∧
    a * b * c * d * e * f * g * h * i * j = 1/10 :=
by sorry

end NUMINAMATH_CALUDE_product_of_ten_proper_fractions_equals_one_tenth_l2044_204408


namespace NUMINAMATH_CALUDE_initial_height_proof_l2044_204484

/-- Calculates the initial height of a person before a growth spurt -/
def initial_height (growth_rate : ℕ) (growth_period : ℕ) (final_height_feet : ℕ) : ℕ :=
  let final_height_inches := final_height_feet * 12
  let total_growth := growth_rate * growth_period
  final_height_inches - total_growth

/-- Theorem stating that given the specific growth conditions, 
    the initial height was 66 inches -/
theorem initial_height_proof : 
  initial_height 2 3 6 = 66 := by
  sorry

end NUMINAMATH_CALUDE_initial_height_proof_l2044_204484


namespace NUMINAMATH_CALUDE_apples_given_to_neighbor_l2044_204431

theorem apples_given_to_neighbor (initial_apples remaining_apples : ℕ) 
  (h1 : initial_apples = 127)
  (h2 : remaining_apples = 39) :
  initial_apples - remaining_apples = 88 := by
  sorry

end NUMINAMATH_CALUDE_apples_given_to_neighbor_l2044_204431


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l2044_204488

/-- An isosceles triangle with two sides measuring 4 and 9 has a perimeter of 22. -/
theorem isosceles_triangle_perimeter : ℝ → ℝ → ℝ → Prop :=
  fun a b c =>
    (a = 4 ∧ b = 9 ∧ c = 9) ∨ (a = 9 ∧ b = 4 ∧ c = 9) ∨ (a = 9 ∧ b = 9 ∧ c = 4) →
    a + b + c = 22

/-- Proof of the theorem -/
theorem isosceles_triangle_perimeter_proof :
  ∃ (a b c : ℝ), isosceles_triangle_perimeter a b c :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l2044_204488


namespace NUMINAMATH_CALUDE_jake_has_nine_peaches_l2044_204444

/-- The number of peaches Steven has -/
def steven_peaches : ℕ := 16

/-- The number of peaches Jake has fewer than Steven -/
def jake_fewer_than_steven : ℕ := 7

/-- The number of peaches Jake has -/
def jake_peaches : ℕ := steven_peaches - jake_fewer_than_steven

/-- Theorem: Jake has 9 peaches -/
theorem jake_has_nine_peaches : jake_peaches = 9 := by
  sorry

end NUMINAMATH_CALUDE_jake_has_nine_peaches_l2044_204444


namespace NUMINAMATH_CALUDE_supermarket_distribution_l2044_204422

/-- Proves that given a total of 420 supermarkets divided between two countries,
    with one country having 56 more supermarkets than the other,
    the country with more supermarkets has 238 supermarkets. -/
theorem supermarket_distribution (total : ℕ) (difference : ℕ) (more : ℕ) (less : ℕ) :
  total = 420 →
  difference = 56 →
  more = less + difference →
  total = more + less →
  more = 238 := by
  sorry

end NUMINAMATH_CALUDE_supermarket_distribution_l2044_204422


namespace NUMINAMATH_CALUDE_quadratic_equation_solutions_l2044_204451

theorem quadratic_equation_solutions : 
  ∀ x : ℝ, x^2 = 6*x ↔ x = 0 ∨ x = 6 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solutions_l2044_204451


namespace NUMINAMATH_CALUDE_complex_operations_l2044_204467

theorem complex_operations (z₁ z₂ : ℂ) 
  (h₁ : z₁ = 2 + 3*I) (h₂ : z₂ = 5 - 7*I) : 
  (z₁ + z₂ = 7 - 4*I) ∧ 
  (z₁ - z₂ = -3 + 10*I) ∧ 
  (z₁ * z₂ = 31 + I) := by
  sorry

#check complex_operations

end NUMINAMATH_CALUDE_complex_operations_l2044_204467


namespace NUMINAMATH_CALUDE_gravel_cost_calculation_l2044_204437

/-- The cost of gravel in dollars per cubic foot -/
def gravel_cost_per_cubic_foot : ℝ := 8

/-- The number of cubic feet in one cubic yard -/
def cubic_feet_per_cubic_yard : ℝ := 27

/-- The volume of gravel in cubic yards -/
def gravel_volume_cubic_yards : ℝ := 3

/-- The cost of the given volume of gravel in dollars -/
def total_cost : ℝ := gravel_volume_cubic_yards * cubic_feet_per_cubic_yard * gravel_cost_per_cubic_foot

theorem gravel_cost_calculation : total_cost = 648 := by
  sorry

end NUMINAMATH_CALUDE_gravel_cost_calculation_l2044_204437


namespace NUMINAMATH_CALUDE_colored_area_half_l2044_204428

/-- Triangle ABC with side AB divided into n parts and AC into n+1 parts -/
structure DividedTriangle where
  ABC : Triangle
  n : ℕ

/-- The ratio of the sum of areas of colored triangles to the area of ABC -/
def coloredAreaRatio (dt : DividedTriangle) : ℚ :=
  sorry

/-- Theorem: The colored area ratio is always 1/2 -/
theorem colored_area_half (dt : DividedTriangle) : coloredAreaRatio dt = 1/2 :=
  sorry

end NUMINAMATH_CALUDE_colored_area_half_l2044_204428


namespace NUMINAMATH_CALUDE_little_john_sweets_expenditure_l2044_204421

theorem little_john_sweets_expenditure
  (initial_amount : ℚ)
  (final_amount : ℚ)
  (amount_per_friend : ℚ)
  (num_friends : ℕ)
  (h1 : initial_amount = 8.5)
  (h2 : final_amount = 4.85)
  (h3 : amount_per_friend = 1.2)
  (h4 : num_friends = 2) :
  initial_amount - final_amount - (↑num_friends * amount_per_friend) = 1.25 :=
by sorry

end NUMINAMATH_CALUDE_little_john_sweets_expenditure_l2044_204421


namespace NUMINAMATH_CALUDE_allocation_theorem_l2044_204417

/-- The number of ways to allocate doctors and nurses to schools -/
def allocation_methods (num_doctors num_nurses num_schools : ℕ) : ℕ :=
  num_doctors * (num_nurses.choose (num_nurses / num_schools))

/-- Theorem: There are 12 ways to allocate 2 doctors and 4 nurses to 2 schools -/
theorem allocation_theorem :
  allocation_methods 2 4 2 = 12 := by
  sorry

#eval allocation_methods 2 4 2

end NUMINAMATH_CALUDE_allocation_theorem_l2044_204417


namespace NUMINAMATH_CALUDE_george_borrowing_weeks_l2044_204459

def loan_amount : ℝ := 100
def initial_fee_rate : ℝ := 0.05
def total_fee : ℝ := 15

def fee_after_weeks (weeks : ℕ) : ℝ :=
  loan_amount * initial_fee_rate * (2 ^ weeks - 1)

theorem george_borrowing_weeks :
  ∃ (weeks : ℕ), weeks > 0 ∧ fee_after_weeks weeks ≤ total_fee ∧ fee_after_weeks (weeks + 1) > total_fee :=
by sorry

end NUMINAMATH_CALUDE_george_borrowing_weeks_l2044_204459


namespace NUMINAMATH_CALUDE_total_leaves_on_ferns_l2044_204426

theorem total_leaves_on_ferns : 
  let total_ferns : ℕ := 12
  let type_a_ferns : ℕ := 4
  let type_b_ferns : ℕ := 5
  let type_c_ferns : ℕ := 3
  let type_a_fronds : ℕ := 15
  let type_a_leaves_per_frond : ℕ := 45
  let type_b_fronds : ℕ := 20
  let type_b_leaves_per_frond : ℕ := 30
  let type_c_fronds : ℕ := 25
  let type_c_leaves_per_frond : ℕ := 40

  total_ferns = type_a_ferns + type_b_ferns + type_c_ferns →
  (type_a_ferns * type_a_fronds * type_a_leaves_per_frond +
   type_b_ferns * type_b_fronds * type_b_leaves_per_frond +
   type_c_ferns * type_c_fronds * type_c_leaves_per_frond) = 8700 :=
by
  sorry

#check total_leaves_on_ferns

end NUMINAMATH_CALUDE_total_leaves_on_ferns_l2044_204426


namespace NUMINAMATH_CALUDE_no_function_satisfies_condition_l2044_204440

/-- The type of positive natural numbers -/
def PositiveNat := {n : ℕ // n > 0}

/-- n-th iterate of a function -/
def iterate (f : PositiveNat → PositiveNat) : ℕ → (PositiveNat → PositiveNat)
  | 0 => id
  | n + 1 => f ∘ (iterate f n)

/-- The main theorem stating that no function satisfies the given condition -/
theorem no_function_satisfies_condition :
  ¬ ∃ (f : PositiveNat → PositiveNat),
    ∀ (n : ℕ), (iterate f n) ⟨n + 1, Nat.succ_pos n⟩ = ⟨n + 2, Nat.succ_pos (n + 1)⟩ :=
by sorry

end NUMINAMATH_CALUDE_no_function_satisfies_condition_l2044_204440


namespace NUMINAMATH_CALUDE_rectangle_area_l2044_204414

theorem rectangle_area (x : ℝ) (w : ℝ) (h : w > 0) : 
  (3 * w)^2 + w^2 = x^2 → 3 * w^2 = 3 * x^2 / 10 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_l2044_204414


namespace NUMINAMATH_CALUDE_bicycle_stock_decrease_l2044_204486

/-- The monthly decrease in bicycle stock -/
def monthly_decrease : ℕ := sorry

/-- The number of months between January 1 and October 1 -/
def months : ℕ := 9

/-- The total decrease in bicycle stock from January 1 to October 1 -/
def total_decrease : ℕ := 36

/-- Theorem stating that the monthly decrease in bicycle stock is 4 -/
theorem bicycle_stock_decrease : monthly_decrease = 4 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_stock_decrease_l2044_204486


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l2044_204424

/-- Given a square tile with side length 10 dm, containing four identical rectangles and a small square,
    where the perimeter of the small square is five times smaller than the perimeter of the entire square,
    prove that the dimensions of the rectangles are 4 dm × 6 dm. -/
theorem rectangle_dimensions (tile_side : ℝ) (small_square_side : ℝ) (rect_short_side : ℝ) (rect_long_side : ℝ) :
  tile_side = 10 →
  small_square_side * 4 = tile_side * 4 / 5 →
  tile_side = small_square_side + 2 * rect_short_side →
  tile_side = rect_short_side + rect_long_side →
  rect_short_side = 4 ∧ rect_long_side = 6 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l2044_204424


namespace NUMINAMATH_CALUDE_smallest_prime_factors_difference_l2044_204491

theorem smallest_prime_factors_difference (n : Nat) (h : n = 296045) :
  ∃ (p q : Nat), Prime p ∧ Prime q ∧ p < q ∧ p ∣ n ∧ q ∣ n ∧
  (∀ r : Nat, Prime r → r ∣ n → p ≤ r) ∧
  (∀ r : Nat, Prime r → r ∣ n → r ≠ p → q ≤ r) ∧
  q - p = 4 :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_factors_difference_l2044_204491


namespace NUMINAMATH_CALUDE_square_difference_formula_l2044_204434

theorem square_difference_formula (x y : ℚ) 
  (h1 : x + y = 8 / 15) (h2 : x - y = 2 / 15) : 
  x^2 - y^2 = 16 / 225 := by
sorry

end NUMINAMATH_CALUDE_square_difference_formula_l2044_204434


namespace NUMINAMATH_CALUDE_square_difference_l2044_204478

theorem square_difference : 100^2 - 2 * 100 * 99 + 99^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l2044_204478


namespace NUMINAMATH_CALUDE_min_x_over_y_for_system_l2044_204412

/-- Given a system of equations, this theorem states that the minimum value of x/y
    for all solutions (x, y) is equal to (-1 - √217) / 12. -/
theorem min_x_over_y_for_system (x y : ℝ) :
  x^3 + 3*y^3 = 11 →
  x^2*y + x*y^2 = 6 →
  ∃ (min_val : ℝ), (∀ (x' y' : ℝ), x'^3 + 3*y'^3 = 11 → x'^2*y' + x'*y'^2 = 6 → x' / y' ≥ min_val) ∧
                   min_val = (-1 - Real.sqrt 217) / 12 :=
sorry

end NUMINAMATH_CALUDE_min_x_over_y_for_system_l2044_204412


namespace NUMINAMATH_CALUDE_abc_inequality_l2044_204463

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a * b * c ≥ (a + b - c) * (b + c - a) * (c + a - b) := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l2044_204463


namespace NUMINAMATH_CALUDE_oliver_money_theorem_l2044_204487

def oliver_money_problem (initial savings frisbee puzzle gift : ℕ) : Prop :=
  initial + savings - frisbee - puzzle + gift = 15

theorem oliver_money_theorem :
  oliver_money_problem 9 5 4 3 8 := by
  sorry

end NUMINAMATH_CALUDE_oliver_money_theorem_l2044_204487


namespace NUMINAMATH_CALUDE_girls_in_club_l2044_204482

theorem girls_in_club (total : Nat) (girls : Nat) (boys : Nat) : 
  total = 36 →
  girls + boys = total →
  (∀ (group : Nat), group = 33 → girls > group / 2) →
  (∃ (group : Nat), group = 31 ∧ boys ≥ group / 2) →
  girls = 20 :=
by sorry

end NUMINAMATH_CALUDE_girls_in_club_l2044_204482


namespace NUMINAMATH_CALUDE_cos_double_angle_special_point_l2044_204409

/-- Given that the terminal side of angle α passes through point (1,2), prove that cos 2α = -3/5 -/
theorem cos_double_angle_special_point (α : ℝ) :
  (∃ r : ℝ, r > 0 ∧ r * Real.cos α = 1 ∧ r * Real.sin α = 2) →
  Real.cos (2 * α) = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_cos_double_angle_special_point_l2044_204409


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l2044_204447

open Real

theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) :
  c * cos (π - B) = (b - 2 * a) * sin (π / 2 - C) →
  c = sqrt 13 →
  b = 3 →
  C = π / 3 ∧ 
  (1 / 2) * a * b * sin C = 3 * sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l2044_204447


namespace NUMINAMATH_CALUDE_percentage_with_neither_is_twenty_percent_l2044_204460

/-- Represents the study of adults in a neighborhood -/
structure NeighborhoodStudy where
  total : ℕ
  insomnia : ℕ
  migraines : ℕ
  both : ℕ

/-- Calculates the percentage of adults with neither insomnia nor migraines -/
def percentageWithNeither (study : NeighborhoodStudy) : ℚ :=
  let withNeither := study.total - (study.insomnia + study.migraines - study.both)
  (withNeither : ℚ) / study.total * 100

/-- The main theorem stating that the percentage of adults with neither condition is 20% -/
theorem percentage_with_neither_is_twenty_percent (study : NeighborhoodStudy)
  (h_total : study.total = 150)
  (h_insomnia : study.insomnia = 90)
  (h_migraines : study.migraines = 60)
  (h_both : study.both = 30) :
  percentageWithNeither study = 20 := by
  sorry

#eval percentageWithNeither { total := 150, insomnia := 90, migraines := 60, both := 30 }

end NUMINAMATH_CALUDE_percentage_with_neither_is_twenty_percent_l2044_204460


namespace NUMINAMATH_CALUDE_reciprocal_of_eight_l2044_204442

-- Define the reciprocal function
def reciprocal (x : ℚ) : ℚ := 1 / x

-- Theorem statement
theorem reciprocal_of_eight : reciprocal 8 = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_eight_l2044_204442


namespace NUMINAMATH_CALUDE_base_conversion_equivalence_l2044_204441

def base_three_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ (digits.length - 1 - i))) 0

def decimal_to_base_five (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc
      else aux (m / 5) ((m % 5) :: acc)
    aux n []

def base_three_number : List Nat := [2, 0, 1, 2, 1]
def base_five_number : List Nat := [1, 2, 0, 3]

theorem base_conversion_equivalence :
  decimal_to_base_five (base_three_to_decimal base_three_number) = base_five_number := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_equivalence_l2044_204441


namespace NUMINAMATH_CALUDE_no_xy_term_l2044_204492

-- Define the expression as a function of x, y, and a
def expression (x y a : ℝ) : ℝ :=
  2 * (x^2 - x*y + y^2) - (3*x^2 - a*x*y + y^2)

-- Theorem statement
theorem no_xy_term (a : ℝ) :
  (∀ x y : ℝ, ∃ k : ℝ, expression x y a = -x^2 + k + y^2) ↔ a = 2 := by
  sorry

end NUMINAMATH_CALUDE_no_xy_term_l2044_204492


namespace NUMINAMATH_CALUDE_right_triangle_distance_theorem_l2044_204405

theorem right_triangle_distance_theorem (a b : ℝ) (ha : a = 9) (hb : b = 12) :
  let c := Real.sqrt (a^2 + b^2)
  let s := (a + b + c) / 2
  let area := a * b / 2
  let r := area / s
  let centroid_dist := 2 * c / 3
  1 = centroid_dist - r :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_distance_theorem_l2044_204405


namespace NUMINAMATH_CALUDE_jills_salary_l2044_204456

/-- Proves that given the conditions of Jill's income allocation, her net monthly salary is $3600 -/
theorem jills_salary (salary : ℝ) 
  (h1 : salary / 5 * 0.15 = 108) : salary = 3600 := by
  sorry

end NUMINAMATH_CALUDE_jills_salary_l2044_204456


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2044_204474

/-- A geometric sequence with real terms -/
def GeometricSequence := ℕ → ℝ

/-- Sum of the first n terms of a geometric sequence -/
def SumGeometric (a : GeometricSequence) (n : ℕ) : ℝ := sorry

theorem geometric_sequence_sum 
  (a : GeometricSequence) 
  (h1 : SumGeometric a 10 = 10) 
  (h2 : SumGeometric a 30 = 70) : 
  SumGeometric a 40 = 150 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2044_204474


namespace NUMINAMATH_CALUDE_alexis_alyssa_age_multiple_l2044_204497

theorem alexis_alyssa_age_multiple :
  ∀ (alexis_age alyssa_age : ℝ),
    alexis_age = 45 →
    alyssa_age = 45 →
    ∃ k : ℝ, alexis_age = k * alyssa_age - 162 →
    k = 4.6 :=
by
  sorry

end NUMINAMATH_CALUDE_alexis_alyssa_age_multiple_l2044_204497


namespace NUMINAMATH_CALUDE_black_and_white_drawing_cost_l2044_204470

/-- The cost of a black and white drawing -/
def black_and_white_cost : ℝ := 160

/-- The cost of a color drawing -/
def color_cost : ℝ := 240

/-- The size of the drawing -/
def drawing_size : ℕ × ℕ := (9, 13)

theorem black_and_white_drawing_cost :
  black_and_white_cost = 160 ∧
  color_cost = black_and_white_cost * 1.5 ∧
  color_cost = 240 := by
  sorry

end NUMINAMATH_CALUDE_black_and_white_drawing_cost_l2044_204470


namespace NUMINAMATH_CALUDE_ship_power_at_6_knots_l2044_204400

-- Define the quadratic function
def H (a b c : ℝ) (v : ℝ) : ℝ := a * v^2 + b * v + c

-- State the theorem
theorem ship_power_at_6_knots 
  (a b c : ℝ) 
  (h1 : H a b c 5 = 300)
  (h2 : H a b c 7 = 780)
  (h3 : H a b c 9 = 1420) :
  H a b c 6 = 520 := by
  sorry

end NUMINAMATH_CALUDE_ship_power_at_6_knots_l2044_204400


namespace NUMINAMATH_CALUDE_thirtieth_triangular_and_difference_l2044_204466

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

theorem thirtieth_triangular_and_difference :
  (triangular_number 30 = 465) ∧
  (triangular_number 30 - triangular_number 29 = 30) := by
  sorry

end NUMINAMATH_CALUDE_thirtieth_triangular_and_difference_l2044_204466


namespace NUMINAMATH_CALUDE_codes_lost_l2044_204435

/-- The number of digits in each code -/
def code_length : Nat := 5

/-- The number of possible digits (0 to 9) -/
def digit_options : Nat := 10

/-- The number of non-zero digits (1 to 9) -/
def nonzero_digit_options : Nat := 9

/-- The number of codes with leading zeros allowed -/
def codes_with_leading_zeros : Nat := digit_options ^ code_length

/-- The number of codes without leading zeros -/
def codes_without_leading_zeros : Nat := nonzero_digit_options * (digit_options ^ (code_length - 1))

theorem codes_lost (code_length : Nat) (digit_options : Nat) (nonzero_digit_options : Nat) 
  (codes_with_leading_zeros : Nat) (codes_without_leading_zeros : Nat) :
  codes_with_leading_zeros - codes_without_leading_zeros = 10000 := by
  sorry

end NUMINAMATH_CALUDE_codes_lost_l2044_204435


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l2044_204465

/-- Two 2D vectors are parallel if their cross product is zero -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem parallel_vectors_x_value :
  let m : ℝ × ℝ := (1, 2)
  let n : ℝ → ℝ × ℝ := λ x ↦ (x, 2 - 2*x)
  ∀ x : ℝ, are_parallel m (n x) → x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l2044_204465


namespace NUMINAMATH_CALUDE_a_is_editor_l2044_204416

-- Define the professions
inductive Profession
| Doctor
| Teacher
| Editor

-- Define the volunteers
structure Volunteer where
  name : String
  profession : Profession
  age : Nat

-- Define the fair
structure Fair where
  volunteers : List Volunteer

-- Define the proposition
theorem a_is_editor (f : Fair) : 
  (∃ a b c : Volunteer, 
    a ∈ f.volunteers ∧ b ∈ f.volunteers ∧ c ∈ f.volunteers ∧
    a.name = "A" ∧ b.name = "B" ∧ c.name = "C" ∧
    a.profession ≠ b.profession ∧ b.profession ≠ c.profession ∧ c.profession ≠ a.profession ∧
    (∃ d : Volunteer, d ∈ f.volunteers ∧ d.profession = Profession.Doctor ∧ d.age ≠ a.age) ∧
    (∃ e : Volunteer, e ∈ f.volunteers ∧ e.profession = Profession.Editor ∧ e.age > c.age) ∧
    (∃ d : Volunteer, d ∈ f.volunteers ∧ d.profession = Profession.Doctor ∧ d.age > b.age)) →
  (∃ a : Volunteer, a ∈ f.volunteers ∧ a.name = "A" ∧ a.profession = Profession.Editor) :=
by sorry

end NUMINAMATH_CALUDE_a_is_editor_l2044_204416


namespace NUMINAMATH_CALUDE_base6_210_equals_base4_1032_l2044_204439

-- Define a function to convert a base 6 number to base 10
def base6ToBase10 (n : ℕ) : ℕ :=
  (n / 100) * 36 + ((n / 10) % 10) * 6 + (n % 10)

-- Define a function to convert a base 10 number to base 4
def base10ToBase4 (n : ℕ) : ℕ :=
  (n / 64) * 1000 + ((n / 16) % 4) * 100 + ((n / 4) % 4) * 10 + (n % 4)

-- Theorem statement
theorem base6_210_equals_base4_1032 :
  base10ToBase4 (base6ToBase10 210) = 1032 :=
sorry

end NUMINAMATH_CALUDE_base6_210_equals_base4_1032_l2044_204439


namespace NUMINAMATH_CALUDE_systematic_sampling_result_l2044_204458

/-- Represents a systematic sampling setup -/
structure SystematicSampling where
  totalPopulation : Nat
  sampleSize : Nat
  interval : Nat
  startingNumber : Nat

/-- Calculates the selected number within a given range for a systematic sampling -/
def selectedNumber (s : SystematicSampling) (rangeStart rangeEnd : Nat) : Nat :=
  let adjustedStart := (rangeStart - s.startingNumber) / s.interval * s.interval + s.startingNumber
  if adjustedStart < rangeStart then
    adjustedStart + s.interval
  else
    adjustedStart

/-- Theorem stating that for the given systematic sampling, the selected number in the range 033 to 048 is 039 -/
theorem systematic_sampling_result :
  let s : SystematicSampling := {
    totalPopulation := 800,
    sampleSize := 50,
    interval := 16,
    startingNumber := 7
  }
  selectedNumber s 33 48 = 39 := by
  sorry


end NUMINAMATH_CALUDE_systematic_sampling_result_l2044_204458


namespace NUMINAMATH_CALUDE_set_inclusion_equivalence_l2044_204475

theorem set_inclusion_equivalence (a : ℝ) : 
  let A := {x : ℝ | 2*a + 1 ≤ x ∧ x ≤ 3*a - 5}
  let B := {x : ℝ | 3 ≤ x ∧ x ≤ 22}
  (∃ x, x ∈ A) → (A ⊆ A ∩ B ↔ 6 ≤ a ∧ a ≤ 9) := by
sorry

end NUMINAMATH_CALUDE_set_inclusion_equivalence_l2044_204475


namespace NUMINAMATH_CALUDE_stating_clock_hands_overlap_at_316_l2044_204472

/-- Represents the number of degrees the hour hand moves in one minute -/
def hourHandDegPerMin : ℝ := 0.5

/-- Represents the number of degrees the minute hand moves in one minute -/
def minuteHandDegPerMin : ℝ := 6

/-- Represents the number of degrees between the hour and minute hands at 3:00 -/
def initialAngle : ℝ := 90

/-- 
Theorem stating that the hour and minute hands of a clock overlap 16 minutes after 3:00
-/
theorem clock_hands_overlap_at_316 :
  ∃ (x : ℝ), x > 0 ∧ x < 60 ∧ 
  minuteHandDegPerMin * x - hourHandDegPerMin * x = initialAngle ∧
  x = 16 := by
  sorry

end NUMINAMATH_CALUDE_stating_clock_hands_overlap_at_316_l2044_204472


namespace NUMINAMATH_CALUDE_division_of_decimals_l2044_204454

theorem division_of_decimals : (0.45 : ℚ) / (0.005 : ℚ) = 90 := by sorry

end NUMINAMATH_CALUDE_division_of_decimals_l2044_204454


namespace NUMINAMATH_CALUDE_range_of_a_l2044_204425

/-- Given functions f and g, prove the range of a -/
theorem range_of_a (a : ℝ) : 
  (∀ x₁ : ℝ, ∃ x₂ : ℝ, |2 * x₁ - a| + |2 * x₁ + 3| = |x₂ - 1| + 2) →
  (a ≥ -1 ∨ a ≤ -5) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2044_204425


namespace NUMINAMATH_CALUDE_trig_identity_l2044_204471

theorem trig_identity :
  1 / Real.cos (70 * π / 180) - Real.sqrt 3 / Real.sin (70 * π / 180) =
  4 * Real.sin (10 * π / 180) / Real.sin (40 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l2044_204471


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2044_204469

/-- Given a geometric sequence {a_n} with a₁ = 2 and q = 2,
    prove that if the sum of the first n terms Sn = 126, then n = 6 -/
theorem geometric_sequence_sum (n : ℕ) (a : ℕ → ℝ) (S : ℕ → ℝ) :
  a 1 = 2 →
  (∀ k, a (k + 1) = 2 * a k) →
  S n = (a 1) * (1 - 2^n) / (1 - 2) →
  S n = 126 →
  n = 6 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2044_204469


namespace NUMINAMATH_CALUDE_largest_n_is_correct_l2044_204427

/-- The largest value of n for which 3x^2 + nx + 90 can be factored as the product of two linear factors with integer coefficients -/
def largest_n : ℕ := 271

/-- A function representing the quadratic expression 3x^2 + nx + 90 -/
def quadratic (n : ℕ) (x : ℚ) : ℚ := 3 * x^2 + n * x + 90

/-- A predicate that checks if a quadratic expression can be factored into two linear factors with integer coefficients -/
def has_integer_linear_factors (n : ℕ) : Prop :=
  ∃ (a b c d : ℤ), ∀ (x : ℚ), quadratic n x = (a * x + b) * (c * x + d)

theorem largest_n_is_correct :
  (∀ n : ℕ, n > largest_n → ¬(has_integer_linear_factors n)) ∧
  (has_integer_linear_factors largest_n) :=
by sorry


end NUMINAMATH_CALUDE_largest_n_is_correct_l2044_204427


namespace NUMINAMATH_CALUDE_expression_value_l2044_204438

theorem expression_value : (100 - (3010 - 301)) + (3010 - (301 - 100)) = 200 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2044_204438


namespace NUMINAMATH_CALUDE_derivative_lg_over_x_l2044_204443

open Real

noncomputable def lg (x : ℝ) : ℝ := log x / log 10

theorem derivative_lg_over_x (x : ℝ) (h : x > 0) :
  deriv (λ x => lg x / x) x = (1 - log 10 * lg x) / (x^2 * log 10) :=
by sorry

end NUMINAMATH_CALUDE_derivative_lg_over_x_l2044_204443


namespace NUMINAMATH_CALUDE_common_root_equations_l2044_204406

theorem common_root_equations (p : ℝ) (h_p : p > 0) : 
  (∃ x : ℝ, 3 * x^2 - 4 * p * x + 9 = 0 ∧ x^2 - 2 * p * x + 5 = 0) ↔ p = 3 :=
by sorry

end NUMINAMATH_CALUDE_common_root_equations_l2044_204406


namespace NUMINAMATH_CALUDE_a_perpendicular_to_a_minus_b_l2044_204423

def a : ℝ × ℝ := (-1, 2)
def b : ℝ × ℝ := (-3, 1)

theorem a_perpendicular_to_a_minus_b : a • (a - b) = 0 := by
  sorry

end NUMINAMATH_CALUDE_a_perpendicular_to_a_minus_b_l2044_204423


namespace NUMINAMATH_CALUDE_ten_row_triangle_pieces_l2044_204420

/-- Calculates the number of unit rods in an n-row triangle -/
def unitRods (n : ℕ) : ℕ := n * (3 + 3 * n) / 2

/-- Calculates the number of connectors in an n-row triangle -/
def connectors (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Calculates the total number of pieces in an n-row triangle -/
def totalPieces (n : ℕ) : ℕ := unitRods n + connectors (n + 1)

theorem ten_row_triangle_pieces :
  totalPieces 10 = 231 ∧ unitRods 2 = 9 ∧ connectors 3 = 6 := by sorry

end NUMINAMATH_CALUDE_ten_row_triangle_pieces_l2044_204420


namespace NUMINAMATH_CALUDE_unique_solution_to_equation_l2044_204476

theorem unique_solution_to_equation (x : ℝ) :
  x + 2 ≠ 0 →
  ((16 * x - x^2) / (x + 2) * (x + (16 - x) / (x + 2)) = 60) ↔ x = 4 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_to_equation_l2044_204476


namespace NUMINAMATH_CALUDE_second_class_average_l2044_204446

/-- Proves that the average mark of the second class is 69.83 given the conditions of the problem -/
theorem second_class_average (students_class1 : ℕ) (students_class2 : ℕ) 
  (avg_class1 : ℝ) (total_avg : ℝ) :
  students_class1 = 39 →
  students_class2 = 35 →
  avg_class1 = 45 →
  total_avg = 56.75 →
  let total_students := students_class1 + students_class2
  let avg_class2 := (total_avg * total_students - avg_class1 * students_class1) / students_class2
  avg_class2 = 69.83 := by
sorry

end NUMINAMATH_CALUDE_second_class_average_l2044_204446


namespace NUMINAMATH_CALUDE_expression_evaluation_l2044_204402

theorem expression_evaluation (a : ℚ) (h : a = 4/3) :
  (6 * a^2 - 8 * a + 3) * (3 * a - 4) = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2044_204402


namespace NUMINAMATH_CALUDE_cubic_inequality_solution_l2044_204496

theorem cubic_inequality_solution (x : ℝ) :
  x^3 + x^2 - 4*x - 4 < 0 ↔ x < -2 ∨ (-1 < x ∧ x < 2) :=
sorry

end NUMINAMATH_CALUDE_cubic_inequality_solution_l2044_204496


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l2044_204413

theorem absolute_value_inequality (k : ℝ) :
  (∀ x : ℝ, |x + 1| - |x - 2| > k) → k ≥ -3 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l2044_204413


namespace NUMINAMATH_CALUDE_simplify_expression_l2044_204433

theorem simplify_expression : (625 : ℝ) ^ (1/4) * (225 : ℝ) ^ (1/2) = 75 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2044_204433


namespace NUMINAMATH_CALUDE_bottom_sphere_radius_l2044_204483

/-- Represents a right circular cone -/
structure Cone where
  baseRadius : ℝ
  height : ℝ

/-- Represents a sphere -/
structure Sphere where
  radius : ℝ

/-- Represents three stacked spheres in a cone -/
structure StackedSpheres where
  cone : Cone
  bottomSphere : Sphere
  middleSphere : Sphere
  topSphere : Sphere

/-- The condition for the spheres to fit in the cone -/
def spheresFitInCone (s : StackedSpheres) : Prop :=
  s.bottomSphere.radius + s.middleSphere.radius + s.topSphere.radius ≤ s.cone.height

/-- The theorem stating the radius of the bottom sphere -/
theorem bottom_sphere_radius (s : StackedSpheres) 
  (h1 : s.cone.baseRadius = 8)
  (h2 : s.cone.height = 18)
  (h3 : s.middleSphere.radius = 2 * s.bottomSphere.radius)
  (h4 : s.topSphere.radius = 3 * s.bottomSphere.radius)
  (h5 : spheresFitInCone s) :
  s.bottomSphere.radius = 3 := by
  sorry

end NUMINAMATH_CALUDE_bottom_sphere_radius_l2044_204483


namespace NUMINAMATH_CALUDE_range_of_x_l2044_204473

theorem range_of_x (x y : ℝ) (h : 4 * x * y + 4 * y^2 + x + 6 = 0) :
  x ≤ -2 ∨ x ≥ 3 := by
sorry

end NUMINAMATH_CALUDE_range_of_x_l2044_204473


namespace NUMINAMATH_CALUDE_symmetric_circle_equation_l2044_204401

/-- The equation of a circle -/
structure Circle where
  equation : ℝ → ℝ → Prop

/-- Symmetry of a circle with respect to the origin -/
def symmetricCircle (c : Circle) : Circle :=
  sorry

/-- The original circle -/
def originalCircle : Circle :=
  { equation := fun x y ↦ x^2 + y^2 + 4*x - 1 = 0 }

theorem symmetric_circle_equation :
  (symmetricCircle originalCircle).equation = fun x y ↦ (x-2)^2 + y^2 = 5 :=
sorry

end NUMINAMATH_CALUDE_symmetric_circle_equation_l2044_204401


namespace NUMINAMATH_CALUDE_no_extreme_value_at_negative_one_increasing_function_p_range_l2044_204489

def f (p : ℝ) (x : ℝ) : ℝ := x^3 + 3*p*x^2 + 3*p*x + 1

theorem no_extreme_value_at_negative_one (p : ℝ) :
  ¬∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), 0 < |x + 1| ∧ |x + 1| < ε → f p x ≤ f p (-1) ∨ f p x ≥ f p (-1) :=
sorry

theorem increasing_function_p_range :
  ∀ (p : ℝ), (∀ (x y : ℝ), -1 < x ∧ x < y → f p x < f p y) ↔ 0 ≤ p ∧ p ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_no_extreme_value_at_negative_one_increasing_function_p_range_l2044_204489


namespace NUMINAMATH_CALUDE_equation_to_lines_l2044_204498

theorem equation_to_lines :
  ∀ x y : ℝ,
  (2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2) ↔
  (y = -x - 2 ∨ y = -2 * x + 1) :=
by sorry

end NUMINAMATH_CALUDE_equation_to_lines_l2044_204498


namespace NUMINAMATH_CALUDE_min_socks_for_given_problem_l2044_204499

/-- The minimum number of socks to pull out to guarantee at least one of each color -/
def min_socks_to_pull (red blue green khaki : ℕ) : ℕ :=
  (red + blue + green + khaki) - min red (min blue (min green khaki)) + 1

/-- Theorem stating the minimum number of socks to pull out for the given problem -/
theorem min_socks_for_given_problem :
  min_socks_to_pull 10 20 30 40 = 91 := by
  sorry

end NUMINAMATH_CALUDE_min_socks_for_given_problem_l2044_204499


namespace NUMINAMATH_CALUDE_max_distance_with_tire_switch_l2044_204461

/-- Given a car with front tires lasting 24000 km and rear tires lasting 36000 km,
    the maximum distance the car can travel by switching tires once is 48000 km. -/
theorem max_distance_with_tire_switch (front_tire_life rear_tire_life : ℕ) 
  (h1 : front_tire_life = 24000)
  (h2 : rear_tire_life = 36000) :
  ∃ (switch_point : ℕ), 
    switch_point ≤ front_tire_life ∧
    switch_point ≤ rear_tire_life ∧
    switch_point + min (front_tire_life - switch_point) (rear_tire_life - switch_point) = 48000 :=
by sorry

end NUMINAMATH_CALUDE_max_distance_with_tire_switch_l2044_204461


namespace NUMINAMATH_CALUDE_max_value_of_f_l2044_204452

def S : Set ℕ := {2, 3, 4, 5, 6, 7, 8}

def f (a b : ℕ) : ℚ := (a : ℚ) / (10 * b + a) + (b : ℚ) / (10 * a + b)

theorem max_value_of_f :
  ∃ (a b : ℕ), a ∈ S ∧ b ∈ S ∧
  (∀ (x y : ℕ), x ∈ S → y ∈ S → f x y ≤ f a b) ∧
  f a b = 89 / 287 := by sorry

end NUMINAMATH_CALUDE_max_value_of_f_l2044_204452


namespace NUMINAMATH_CALUDE_max_correct_answers_l2044_204494

/-- Represents the result of a math contest. -/
structure ContestResult where
  correct : ℕ
  blank : ℕ
  incorrect : ℕ
  deriving Repr

/-- Calculates the score for a given contest result. -/
def calculateScore (result : ContestResult) : ℤ :=
  5 * result.correct - 2 * result.incorrect

/-- Checks if a contest result is valid (total questions = 60). -/
def isValidResult (result : ContestResult) : Prop :=
  result.correct + result.blank + result.incorrect = 60

/-- Theorem stating the maximum number of correct answers Evelyn could have. -/
theorem max_correct_answers (result : ContestResult) 
  (h1 : isValidResult result) 
  (h2 : calculateScore result = 150) : 
  result.correct ≤ 38 := by
  sorry

end NUMINAMATH_CALUDE_max_correct_answers_l2044_204494


namespace NUMINAMATH_CALUDE_centroid_sum_l2044_204415

def vertex1 : Fin 3 → ℚ := ![9, 2, -1]
def vertex2 : Fin 3 → ℚ := ![5, -2, 3]
def vertex3 : Fin 3 → ℚ := ![1, 6, 5]

def centroid (v1 v2 v3 : Fin 3 → ℚ) : Fin 3 → ℚ :=
  fun i => (v1 i + v2 i + v3 i) / 3

theorem centroid_sum :
  (centroid vertex1 vertex2 vertex3 0 +
   centroid vertex1 vertex2 vertex3 1 +
   centroid vertex1 vertex2 vertex3 2) = 28 / 3 := by
  sorry

end NUMINAMATH_CALUDE_centroid_sum_l2044_204415


namespace NUMINAMATH_CALUDE_book_arrangement_theorem_l2044_204493

def arrange_books (geom_copies : ℕ) (alg_copies : ℕ) : ℕ :=
  let total_slots := geom_copies + alg_copies - 1
  let remaining_geom := geom_copies - 2
  (total_slots.choose remaining_geom) * 2

theorem book_arrangement_theorem :
  arrange_books 4 5 = 112 :=
sorry

end NUMINAMATH_CALUDE_book_arrangement_theorem_l2044_204493


namespace NUMINAMATH_CALUDE_monotonic_range_k_negative_range_k_l2044_204495

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 1

-- Define the function g
def g (a b k : ℝ) (x : ℝ) : ℝ := f a b x - k * x

-- Theorem for part (1)
theorem monotonic_range_k (a b : ℝ) (h1 : a > 0) (h2 : f a b (-1) = 0) 
  (h3 : ∀ x : ℝ, f a b x ≥ 0) :
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, Monotone (g a b)) ↔ (k ≤ -2 ∨ k ≥ 6) :=
sorry

-- Theorem for part (2)
theorem negative_range_k (a b : ℝ) (h1 : a > 0) (h2 : f a b (-1) = 0) 
  (h3 : ∀ x : ℝ, f a b x ≥ 0) :
  (∀ x ∈ Set.Icc 1 2, g a b k x < 0) ↔ k > 9/2 :=
sorry

end NUMINAMATH_CALUDE_monotonic_range_k_negative_range_k_l2044_204495


namespace NUMINAMATH_CALUDE_matrix_equation_solution_l2044_204485

theorem matrix_equation_solution : 
  let N : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; 1, 3]
  N^3 - 3 • N^2 + 2 • N = !![6, 12; 3, 6] := by sorry

end NUMINAMATH_CALUDE_matrix_equation_solution_l2044_204485


namespace NUMINAMATH_CALUDE_weekly_toy_production_l2044_204436

/-- A factory produces toys with the following conditions:
  * Workers work 5 days a week
  * Workers produce the same number of toys every day
  * Workers produce 1100 toys each day
-/
def toy_factory (days_per_week : ℕ) (toys_per_day : ℕ) : Prop :=
  days_per_week = 5 ∧ toys_per_day = 1100

/-- The number of toys produced in a week -/
def weekly_production (days_per_week : ℕ) (toys_per_day : ℕ) : ℕ :=
  days_per_week * toys_per_day

/-- Theorem: Under the given conditions, the factory produces 5500 toys in a week -/
theorem weekly_toy_production :
  ∀ (days_per_week toys_per_day : ℕ),
    toy_factory days_per_week toys_per_day →
    weekly_production days_per_week toys_per_day = 5500 :=
by
  sorry

end NUMINAMATH_CALUDE_weekly_toy_production_l2044_204436

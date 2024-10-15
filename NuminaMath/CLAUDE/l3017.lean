import Mathlib

namespace NUMINAMATH_CALUDE_incorrect_expression_l3017_301741

/-- Represents a repeating decimal with a 3-digit non-repeating part and a 2-digit repeating part -/
structure RepeatingDecimal where
  nonRepeating : ℕ  -- Represents P (3-digit non-repeating part)
  repeating : ℕ     -- Represents Q (2-digit repeating part)
  nonRepeating_three_digits : nonRepeating < 1000
  repeating_two_digits : repeating < 100

/-- Converts a RepeatingDecimal to its decimal representation -/
def toDecimal (d : RepeatingDecimal) : ℚ :=
  (d.nonRepeating : ℚ) / 1000 + (d.repeating : ℚ) / 99900

/-- The statement that the given expression is incorrect -/
theorem incorrect_expression (d : RepeatingDecimal) :
  ¬(10^3 * (10^2 - 1) * toDecimal d = (d.repeating : ℚ) * (100 * d.nonRepeating - 1)) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_expression_l3017_301741


namespace NUMINAMATH_CALUDE_percentage_problem_l3017_301799

theorem percentage_problem (x : ℝ) : (350 / 100) * x = 140 → x = 40 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3017_301799


namespace NUMINAMATH_CALUDE_polynomial_value_equals_one_l3017_301755

theorem polynomial_value_equals_one (x₀ : ℂ) (h : x₀^2 + x₀ + 2 = 0) :
  x₀^4 + 2*x₀^3 + 3*x₀^2 + 2*x₀ + 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_equals_one_l3017_301755


namespace NUMINAMATH_CALUDE_parallel_vectors_k_value_l3017_301786

/-- Two vectors in ℝ² are parallel if their components are proportional -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_k_value (k : ℝ) :
  let a : ℝ × ℝ := (2, k)
  let b : ℝ × ℝ := (1, 2)
  parallel a b → k = 4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_k_value_l3017_301786


namespace NUMINAMATH_CALUDE_conic_section_type_l3017_301712

theorem conic_section_type (x y : ℝ) : 
  (9 * x^2 - 16 * y^2 = 0) → 
  ∃ (m₁ m₂ : ℝ), (∀ x y, (y = m₁ * x ∨ y = m₂ * x) ↔ 9 * x^2 - 16 * y^2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_conic_section_type_l3017_301712


namespace NUMINAMATH_CALUDE_not_divisible_by_power_of_two_l3017_301762

theorem not_divisible_by_power_of_two (p : ℕ) (hp : p > 1) :
  ¬(2^p ∣ 3^p + 1) := by
sorry

end NUMINAMATH_CALUDE_not_divisible_by_power_of_two_l3017_301762


namespace NUMINAMATH_CALUDE_no_fraction_value_l3017_301742

-- Define the No operator
def No : ℕ → ℕ
  | 0 => 1
  | n + 1 => (n + 1) * No n

-- State the theorem
theorem no_fraction_value :
  (No 2022) / (No 2023) = 1 / 2023 := by sorry

end NUMINAMATH_CALUDE_no_fraction_value_l3017_301742


namespace NUMINAMATH_CALUDE_two_numbers_with_given_means_l3017_301748

theorem two_numbers_with_given_means (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a * b = 5) →
  (2 * a * b / (a + b) = 5/3) →
  ((a = 1 ∧ b = 5) ∨ (a = 5 ∧ b = 1)) :=
by sorry

end NUMINAMATH_CALUDE_two_numbers_with_given_means_l3017_301748


namespace NUMINAMATH_CALUDE_soccer_substitutions_mod_1000_l3017_301719

/-- Number of ways to make n substitutions -/
def num_substitutions (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | k + 1 => 11 * (13 - k) * num_substitutions k

/-- Total number of ways to make up to 4 substitutions -/
def total_substitutions : ℕ :=
  (List.range 5).map num_substitutions |> List.sum

theorem soccer_substitutions_mod_1000 :
  total_substitutions % 1000 = 25 := by
  sorry

end NUMINAMATH_CALUDE_soccer_substitutions_mod_1000_l3017_301719


namespace NUMINAMATH_CALUDE_vector_decomposition_l3017_301771

/-- Given vectors in ℝ³ -/
def x : Fin 3 → ℝ := ![6, 5, -14]
def p : Fin 3 → ℝ := ![1, 1, 4]
def q : Fin 3 → ℝ := ![0, -3, 2]
def r : Fin 3 → ℝ := ![2, 1, -1]

/-- Theorem: x can be expressed as a linear combination of p, q, and r -/
theorem vector_decomposition :
  x = (-2 : ℝ) • p + (-1 : ℝ) • q + (4 : ℝ) • r := by
  sorry

end NUMINAMATH_CALUDE_vector_decomposition_l3017_301771


namespace NUMINAMATH_CALUDE_function_order_l3017_301718

/-- A quadratic function f(x) = x^2 + bx + c that satisfies f(x-1) = f(3-x) for all x ∈ ℝ -/
def f (b c : ℝ) (x : ℝ) : ℝ := x^2 + b*x + c

/-- The symmetry condition of the function -/
axiom symmetry (b c : ℝ) : ∀ x, f b c (x - 1) = f b c (3 - x)

/-- Theorem stating the order of f(0), f(-2), and f(5) -/
theorem function_order (b c : ℝ) : f b c 0 < f b c (-2) ∧ f b c (-2) < f b c 5 := by
  sorry

end NUMINAMATH_CALUDE_function_order_l3017_301718


namespace NUMINAMATH_CALUDE_value_calculation_l3017_301774

theorem value_calculation (x : ℝ) (y : ℝ) (h1 : x = 50.0) (h2 : y = 0.20 * x - 4) : y = 6.0 := by
  sorry

end NUMINAMATH_CALUDE_value_calculation_l3017_301774


namespace NUMINAMATH_CALUDE_integral_cos_sin_l3017_301708

theorem integral_cos_sin : ∫ x in (0)..(π/2), (1 + Real.cos x) / (1 + Real.sin x + Real.cos x) = Real.log 2 + π/2 := by
  sorry

end NUMINAMATH_CALUDE_integral_cos_sin_l3017_301708


namespace NUMINAMATH_CALUDE_intersection_when_m_is_3_range_of_m_when_union_equals_B_l3017_301784

-- Define the sets A and B
def A : Set ℝ := {x | x < 0 ∨ x > 3}
def B (m : ℝ) : Set ℝ := {x | x < m - 1 ∨ x > 2 * m}

-- Part 1: Prove that when m = 3, A ∩ B = {x | x < 0 ∨ x > 6}
theorem intersection_when_m_is_3 : A ∩ B 3 = {x | x < 0 ∨ x > 6} := by sorry

-- Part 2: Prove that when B ∪ A = B, the range of m is [1, 3/2]
theorem range_of_m_when_union_equals_B :
  (∀ m : ℝ, B m ∪ A = B m) ↔ (∀ m : ℝ, 1 ≤ m ∧ m ≤ 3/2) := by sorry

end NUMINAMATH_CALUDE_intersection_when_m_is_3_range_of_m_when_union_equals_B_l3017_301784


namespace NUMINAMATH_CALUDE_sine_sum_problem_l3017_301736

theorem sine_sum_problem (α : Real) (h1 : α ∈ Set.Ioo 0 π) (h2 : Real.tan (α - π/4) = 1/3) :
  Real.sin (π/4 + α) = (3 * Real.sqrt 10) / 10 := by
  sorry

end NUMINAMATH_CALUDE_sine_sum_problem_l3017_301736


namespace NUMINAMATH_CALUDE_fraction_equivalence_l3017_301795

theorem fraction_equivalence : (15 : ℚ) / 25 = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l3017_301795


namespace NUMINAMATH_CALUDE_not_square_for_prime_l3017_301752

theorem not_square_for_prime (p : ℕ) (h_prime : Nat.Prime p) : ¬∃ (a : ℤ), (7 * p + 3^p - 4 : ℤ) = a^2 := by
  sorry

end NUMINAMATH_CALUDE_not_square_for_prime_l3017_301752


namespace NUMINAMATH_CALUDE_least_bananas_l3017_301757

theorem least_bananas (b₁ b₂ b₃ : ℕ) : 
  (∃ (A B C : ℕ), 
    A = b₁ / 2 + b₂ / 3 + 5 * b₃ / 12 ∧
    B = b₁ / 4 + 2 * b₂ / 3 + 5 * b₃ / 12 ∧
    C = b₁ / 4 + b₂ / 3 + b₃ / 6 ∧
    A = 4 * k ∧ B = 3 * k ∧ C = 2 * k ∧
    (∀ m, m < b₁ + b₂ + b₃ → 
      ¬(∃ (A' B' C' : ℕ), 
        A' = m / 2 + (b₁ + b₂ + b₃ - m) / 3 + 5 * (b₁ + b₂ + b₃ - m) / 12 ∧
        B' = m / 4 + 2 * (b₁ + b₂ + b₃ - m) / 3 + 5 * (b₁ + b₂ + b₃ - m) / 12 ∧
        C' = m / 4 + (b₁ + b₂ + b₃ - m) / 3 + (b₁ + b₂ + b₃ - m) / 6 ∧
        A' = 4 * k' ∧ B' = 3 * k' ∧ C' = 2 * k'))) →
  b₁ + b₂ + b₃ = 276 :=
by sorry

end NUMINAMATH_CALUDE_least_bananas_l3017_301757


namespace NUMINAMATH_CALUDE_largest_inscribed_equilateral_triangle_area_l3017_301766

/-- The area of the largest equilateral triangle inscribed in a circle of radius 10 -/
theorem largest_inscribed_equilateral_triangle_area (r : ℝ) (h : r = 10) :
  let s := r * (3 / Real.sqrt 3)
  let area := (s^2 * Real.sqrt 3) / 4
  area = 75 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_largest_inscribed_equilateral_triangle_area_l3017_301766


namespace NUMINAMATH_CALUDE_common_root_condition_l3017_301704

theorem common_root_condition (m : ℝ) : 
  (∃ x : ℝ, m * x - 1000 = 1001 ∧ 1001 * x = m - 1000 * x) ↔ (m = 2001 ∨ m = -2001) := by
  sorry

end NUMINAMATH_CALUDE_common_root_condition_l3017_301704


namespace NUMINAMATH_CALUDE_truck_loading_capacity_correct_bag_count_l3017_301710

theorem truck_loading_capacity (truck_capacity : ℕ) 
                                (box_count box_weight : ℕ) 
                                (crate_count crate_weight : ℕ) 
                                (sack_count sack_weight : ℕ) 
                                (bag_weight : ℕ) : ℕ :=
  let total_loaded := box_count * box_weight + crate_count * crate_weight + sack_count * sack_weight
  let remaining_capacity := truck_capacity - total_loaded
  remaining_capacity / bag_weight

theorem correct_bag_count : 
  truck_loading_capacity 13500 100 100 10 60 50 50 40 = 10 := by
  sorry

end NUMINAMATH_CALUDE_truck_loading_capacity_correct_bag_count_l3017_301710


namespace NUMINAMATH_CALUDE_simplify_sqrt_a_squared_b_over_two_l3017_301701

theorem simplify_sqrt_a_squared_b_over_two
  (a b : ℝ) (ha : a < 0) :
  Real.sqrt ((a^2 * b) / 2) = -a / 2 * Real.sqrt (2 * b) :=
by sorry

end NUMINAMATH_CALUDE_simplify_sqrt_a_squared_b_over_two_l3017_301701


namespace NUMINAMATH_CALUDE_coefficient_sum_l3017_301772

theorem coefficient_sum (b₀ b₁ b₂ b₃ b₄ b₅ b₆ : ℝ) :
  (∀ x : ℝ, (5*x - 2)^6 = b₆*x^6 + b₅*x^5 + b₄*x^4 + b₃*x^3 + b₂*x^2 + b₁*x + b₀) →
  b₆ + b₅ + b₄ + b₃ + b₂ + b₁ + b₀ = 729 := by
sorry

end NUMINAMATH_CALUDE_coefficient_sum_l3017_301772


namespace NUMINAMATH_CALUDE_least_value_f_1998_l3017_301703

/-- A function from positive integers to positive integers satisfying the given condition -/
def SpecialFunction (f : ℕ+ → ℕ+) : Prop :=
  ∀ s t : ℕ+, f (t^2 * f s) = s * (f t)^2

/-- The theorem stating the least possible value of f(1998) -/
theorem least_value_f_1998 :
  ∃ (f : ℕ+ → ℕ+), SpecialFunction f ∧
    (∀ g : ℕ+ → ℕ+, SpecialFunction g → f 1998 ≤ g 1998) ∧
    f 1998 = 120 :=
sorry

end NUMINAMATH_CALUDE_least_value_f_1998_l3017_301703


namespace NUMINAMATH_CALUDE_infinite_series_sum_l3017_301760

theorem infinite_series_sum : 
  ∑' n : ℕ, (1 / ((2*n+1)^2 - (2*n-1)^2)) * (1 / (2*n-1)^2 - 1 / (2*n+1)^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l3017_301760


namespace NUMINAMATH_CALUDE_total_earnings_proof_l3017_301778

/-- Calculates the total earnings for a three-day fundraiser car wash activity. -/
def total_earnings (friday_earnings : ℕ) : ℕ :=
  let saturday_earnings := 2 * friday_earnings + 7
  let sunday_earnings := friday_earnings + 78
  friday_earnings + saturday_earnings + sunday_earnings

/-- Proves that the total earnings over three days is 673, given the specified conditions. -/
theorem total_earnings_proof :
  total_earnings 147 = 673 := by
  sorry

#eval total_earnings 147

end NUMINAMATH_CALUDE_total_earnings_proof_l3017_301778


namespace NUMINAMATH_CALUDE_unreachable_target_l3017_301761

/-- A permutation of the first 100 natural numbers -/
def Permutation := Fin 100 → Fin 100

/-- The initial sequence 1, 2, 3, ..., 99, 100 -/
def initial : Permutation := fun i => i + 1

/-- The target sequence 100, 99, 98, ..., 2, 1 -/
def target : Permutation := fun i => 100 - i

/-- A valid swap in the sequence -/
def validSwap (p : Permutation) (i j : Fin 100) : Prop :=
  ∃ k, i < k ∧ k < j ∧ j = i + 2 ∧
    (∀ m, m ≠ i ∧ m ≠ j → p m = p m) ∧
    p i = p j ∧ p j = p i

/-- A sequence that can be obtained from the initial sequence using valid swaps -/
inductive reachable : Permutation → Prop
  | init : reachable initial
  | swap : ∀ {p q : Permutation}, reachable p → validSwap p i j → q = p ∘ (Equiv.swap i j) → reachable q

theorem unreachable_target : ¬ reachable target := by sorry

end NUMINAMATH_CALUDE_unreachable_target_l3017_301761


namespace NUMINAMATH_CALUDE_area_difference_circle_square_l3017_301709

/-- The difference between the area of a circle with diameter 8 inches and 
    the area of a square with diagonal 8 inches is approximately 18.3 square inches. -/
theorem area_difference_circle_square : 
  let circle_diameter : ℝ := 8
  let square_diagonal : ℝ := 8
  let circle_area : ℝ := π * (circle_diameter / 2)^2
  let square_area : ℝ := (square_diagonal^2) / 2
  let area_difference : ℝ := circle_area - square_area
  ∃ ε > 0, abs (area_difference - 18.3) < ε ∧ ε < 0.1 :=
by sorry

end NUMINAMATH_CALUDE_area_difference_circle_square_l3017_301709


namespace NUMINAMATH_CALUDE_percentage_problem_l3017_301750

theorem percentage_problem (N : ℝ) (P : ℝ) : 
  N = 800 →
  0.4 * N = (P / 100) * 650 + 190 →
  P = 20 := by sorry

end NUMINAMATH_CALUDE_percentage_problem_l3017_301750


namespace NUMINAMATH_CALUDE_courtyard_length_l3017_301706

/-- Proves that the length of a courtyard is 25 meters given specific conditions -/
theorem courtyard_length : 
  ∀ (width : ℝ) (brick_length brick_width : ℝ) (total_bricks : ℕ),
  width = 15 →
  brick_length = 0.2 →
  brick_width = 0.1 →
  total_bricks = 18750 →
  (width * (total_bricks : ℝ) * brick_length * brick_width) / width = 25 := by
sorry

end NUMINAMATH_CALUDE_courtyard_length_l3017_301706


namespace NUMINAMATH_CALUDE_complement_of_B_l3017_301746

-- Define the set B
def B : Set ℝ := {x | x^2 - 3*x + 2 < 0}

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- State the theorem
theorem complement_of_B : 
  Set.compl B = {x : ℝ | x ≤ 1 ∨ x ≥ 2} := by sorry

end NUMINAMATH_CALUDE_complement_of_B_l3017_301746


namespace NUMINAMATH_CALUDE_product_of_diff_squares_l3017_301730

theorem product_of_diff_squares (a b c d : ℕ+) 
  (ha : ∃ (x y : ℕ+), a = x^2 - y^2)
  (hb : ∃ (z w : ℕ+), b = z^2 - w^2)
  (hc : ∃ (p q : ℕ+), c = p^2 - q^2)
  (hd : ∃ (r s : ℕ+), d = r^2 - s^2) :
  ∃ (u v : ℕ+), (a * b * c * d : ℕ) = u^2 - v^2 :=
sorry

end NUMINAMATH_CALUDE_product_of_diff_squares_l3017_301730


namespace NUMINAMATH_CALUDE_saturday_attendance_l3017_301773

theorem saturday_attendance (price : ℝ) (total_earnings : ℝ) : 
  price = 10 →
  total_earnings = 300 →
  ∃ (saturday : ℕ),
    saturday * price + (saturday / 2) * price = total_earnings ∧
    saturday = 20 := by
  sorry

end NUMINAMATH_CALUDE_saturday_attendance_l3017_301773


namespace NUMINAMATH_CALUDE_arithmetic_sequence_5_to_119_l3017_301731

/-- The number of terms in an arithmetic sequence -/
def arithmeticSequenceLength (a₁ : ℕ) (aₙ : ℕ) (d : ℕ) : ℕ :=
  (aₙ - a₁) / d + 1

/-- Proof that the arithmetic sequence from 5 to 119 with common difference 3 has 39 terms -/
theorem arithmetic_sequence_5_to_119 :
  arithmeticSequenceLength 5 119 3 = 39 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_5_to_119_l3017_301731


namespace NUMINAMATH_CALUDE_two_six_minus_one_prime_divisors_l3017_301739

theorem two_six_minus_one_prime_divisors :
  ∃ (p q : Nat), Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧
  (∀ (r : Nat), Nat.Prime r → r ∣ (2^6 - 1) → r = p ∨ r = q) ∧
  p + q = 10 := by
  sorry

end NUMINAMATH_CALUDE_two_six_minus_one_prime_divisors_l3017_301739


namespace NUMINAMATH_CALUDE_parabola_tangent_intersection_l3017_301785

noncomputable def parabola (x : ℝ) : ℝ := x^2

def point_A : ℝ × ℝ := (1, 1)

noncomputable def point_B (x2 : ℝ) : ℝ × ℝ := (x2, x2^2)

noncomputable def tangent_slope (x : ℝ) : ℝ := 2 * x

noncomputable def tangent_line_A (x : ℝ) : ℝ := 2 * (x - 1) + 1

noncomputable def tangent_line_B (x2 x : ℝ) : ℝ := 2 * x2 * (x - x2) + x2^2

noncomputable def intersection_point (x2 : ℝ) : ℝ × ℝ :=
  let x_c := (x2^2 - 1) / (2 - 2*x2)
  let y_c := 2 * x_c - 1
  (x_c, y_c)

noncomputable def vector_AC (x2 : ℝ) : ℝ × ℝ :=
  let C := intersection_point x2
  (C.1 - point_A.1, C.2 - point_A.2)

noncomputable def vector_BC (x2 : ℝ) : ℝ × ℝ :=
  let C := intersection_point x2
  let B := point_B x2
  (C.1 - B.1, C.2 - B.2)

noncomputable def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem parabola_tangent_intersection (x2 : ℝ) :
  dot_product (vector_AC x2) (vector_BC x2) = 0 → x2 = -1/4 :=
by sorry

end NUMINAMATH_CALUDE_parabola_tangent_intersection_l3017_301785


namespace NUMINAMATH_CALUDE_exists_k_for_A_l3017_301724

theorem exists_k_for_A (n m : ℕ) (hn : n ≥ 2) (hm : m ≥ 2) :
  ∃ k : ℕ, ((n + Real.sqrt (n^2 - 4)) / 2)^m = (k + Real.sqrt (k^2 - 4)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_exists_k_for_A_l3017_301724


namespace NUMINAMATH_CALUDE_wire_length_ratio_l3017_301732

theorem wire_length_ratio :
  let edge_length : ℕ := 8
  let large_cube_wire_length : ℕ := 12 * edge_length
  let large_cube_volume : ℕ := edge_length ^ 3
  let unit_cube_wire_length : ℕ := 12
  let total_unit_cubes : ℕ := large_cube_volume
  let total_unit_cube_wire_length : ℕ := total_unit_cubes * unit_cube_wire_length
  (large_cube_wire_length : ℚ) / total_unit_cube_wire_length = 1 / 64 := by
  sorry

end NUMINAMATH_CALUDE_wire_length_ratio_l3017_301732


namespace NUMINAMATH_CALUDE_height_difference_ruby_xavier_l3017_301758

-- Constants and conversion factors
def inch_to_cm : ℝ := 2.54
def m_to_cm : ℝ := 100

-- Given heights and relationships
def janet_height_inch : ℝ := 62.75
def charlene_height_factor : ℝ := 1.5
def pablo_charlene_diff_m : ℝ := 1.85
def ruby_pablo_diff_cm : ℝ := 0.5
def xavier_charlene_diff_m : ℝ := 2.13
def paul_xavier_diff_cm : ℝ := 97.75
def paul_ruby_diff_m : ℝ := 0.5

-- Theorem statement
theorem height_difference_ruby_xavier :
  let janet_height_cm := janet_height_inch * inch_to_cm
  let charlene_height_cm := charlene_height_factor * janet_height_cm
  let pablo_height_cm := charlene_height_cm + pablo_charlene_diff_m * m_to_cm
  let ruby_height_cm := pablo_height_cm - ruby_pablo_diff_cm
  let xavier_height_cm := charlene_height_cm + xavier_charlene_diff_m * m_to_cm
  let paul_height_cm := ruby_height_cm + paul_ruby_diff_m * m_to_cm
  let height_diff_cm := xavier_height_cm - ruby_height_cm
  let height_diff_inch := height_diff_cm / inch_to_cm
  ∃ ε > 0, |height_diff_inch - 18.78| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_height_difference_ruby_xavier_l3017_301758


namespace NUMINAMATH_CALUDE_matching_shoes_probability_l3017_301754

theorem matching_shoes_probability (n : ℕ) (h : n = 12) :
  let total_shoes := 2 * n
  let total_combinations := (total_shoes.choose 2 : ℚ)
  let matching_pairs := n
  matching_pairs / total_combinations = 1 / 46 :=
by sorry

end NUMINAMATH_CALUDE_matching_shoes_probability_l3017_301754


namespace NUMINAMATH_CALUDE_raisin_mixture_problem_l3017_301781

/-- The number of scoops of natural seedless raisins needed to create a mixture with
    a specific cost per scoop, given the costs and quantities of two types of raisins. -/
theorem raisin_mixture_problem (cost_natural : ℚ) (cost_golden : ℚ) (scoops_golden : ℕ) (cost_mixture : ℚ) :
  cost_natural = 345/100 →
  cost_golden = 255/100 →
  scoops_golden = 20 →
  cost_mixture = 3 →
  ∃ scoops_natural : ℕ,
    scoops_natural = 20 ∧
    (cost_natural * scoops_natural + cost_golden * scoops_golden) / (scoops_natural + scoops_golden) = cost_mixture :=
by sorry

end NUMINAMATH_CALUDE_raisin_mixture_problem_l3017_301781


namespace NUMINAMATH_CALUDE_cosine_sine_graph_shift_l3017_301789

theorem cosine_sine_graph_shift (x : ℝ) :
  3 * Real.cos (2 * x) = 3 * Real.sin (2 * (x + π / 6) + π / 6) := by
  sorry

end NUMINAMATH_CALUDE_cosine_sine_graph_shift_l3017_301789


namespace NUMINAMATH_CALUDE_farm_chicken_count_l3017_301743

/-- The number of chicken coops on the farm -/
def num_coops : ℕ := 9

/-- The number of chickens in each coop -/
def chickens_per_coop : ℕ := 60

/-- The total number of chickens on the farm -/
def total_chickens : ℕ := num_coops * chickens_per_coop

theorem farm_chicken_count : total_chickens = 540 := by
  sorry

end NUMINAMATH_CALUDE_farm_chicken_count_l3017_301743


namespace NUMINAMATH_CALUDE_two_digit_number_representation_l3017_301700

/-- Represents a two-digit number -/
def two_digit_number (a b : ℕ) : ℕ := 10 * b + a

/-- Theorem stating that a two-digit number with digit a in the units place
    and digit b in the tens place is represented as 10b + a -/
theorem two_digit_number_representation (a b : ℕ) (h1 : a < 10) (h2 : b < 10) (h3 : b ≠ 0) :
  two_digit_number a b = 10 * b + a := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_representation_l3017_301700


namespace NUMINAMATH_CALUDE_middle_term_of_arithmetic_sequence_l3017_301720

def arithmetic_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) - a n = a 1 - a 0

theorem middle_term_of_arithmetic_sequence 
  (a : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a) 
  (h_first : a 0 = 12) 
  (h_last : a 6 = 54) :
  a 3 = 33 := by
sorry

end NUMINAMATH_CALUDE_middle_term_of_arithmetic_sequence_l3017_301720


namespace NUMINAMATH_CALUDE_stating_magical_stack_size_magical_stack_n_l3017_301769

/-- Represents a stack of cards with the described properties. -/
structure CardStack :=
  (n : ℕ)  -- Half the total number of cards
  (is_magical : Bool)  -- Whether the stack is magical after restacking

/-- 
  Theorem stating that a magical stack where card 101 retains its position
  must have 302 cards in total.
-/
theorem magical_stack_size 
  (stack : CardStack) 
  (h_magical : stack.is_magical = true) 
  (h_101_position : ∃ (pos : ℕ), pos ≤ 2 * stack.n ∧ pos = 101) :
  2 * stack.n = 302 := by
  sorry

/-- 
  Corollary: The value of n in a magical stack where card 101 
  retains its position is 151.
-/
theorem magical_stack_n 
  (stack : CardStack) 
  (h_magical : stack.is_magical = true) 
  (h_101_position : ∃ (pos : ℕ), pos ≤ 2 * stack.n ∧ pos = 101) :
  stack.n = 151 := by
  sorry

end NUMINAMATH_CALUDE_stating_magical_stack_size_magical_stack_n_l3017_301769


namespace NUMINAMATH_CALUDE_smallest_n_for_non_simplest_fraction_l3017_301707

theorem smallest_n_for_non_simplest_fraction : ∃ (d : ℕ), d > 1 ∧ d ∣ (17 + 2) ∧ d ∣ (3 * 17^2 + 7) ∧
  ∀ (n : ℕ), n > 0 ∧ n < 17 → ∀ (k : ℕ), k > 1 → ¬(k ∣ (n + 2) ∧ k ∣ (3 * n^2 + 7)) :=
by sorry

#check smallest_n_for_non_simplest_fraction

end NUMINAMATH_CALUDE_smallest_n_for_non_simplest_fraction_l3017_301707


namespace NUMINAMATH_CALUDE_product_evaluation_l3017_301779

theorem product_evaluation :
  (5 + 7) * (5^2 + 7^2) * (5^4 + 7^4) * (5^8 + 7^8) * (5^16 + 7^16) * (5^32 + 7^32) * (5^64 + 7^64) = 7^128 - 5^128 := by
  sorry

end NUMINAMATH_CALUDE_product_evaluation_l3017_301779


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3017_301738

theorem polynomial_simplification (x : ℝ) :
  (x^3 + 4*x^2 - 7*x + 11) + (-4*x^4 - x^3 + x^2 + 7*x + 3) = -4*x^4 + 5*x^2 + 14 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3017_301738


namespace NUMINAMATH_CALUDE_reflected_ray_equation_l3017_301782

-- Define the points
def start_point : ℝ × ℝ := (-1, 3)
def end_point : ℝ × ℝ := (4, 6)

-- Define the reflection surface (x-axis)
def reflection_axis : Set (ℝ × ℝ) := {p | p.2 = 0}

-- Define the reflected ray
def reflected_ray : Set (ℝ × ℝ) := {p | ∃ t : ℝ, p = (1 - t) • start_point + t • end_point}

-- Theorem statement
theorem reflected_ray_equation :
  ∀ p ∈ reflected_ray, 9 * p.1 - 5 * p.2 - 6 = 0 :=
sorry

end NUMINAMATH_CALUDE_reflected_ray_equation_l3017_301782


namespace NUMINAMATH_CALUDE_wills_jogging_time_l3017_301770

/-- Calculates the jogging time given initial calories, burn rate, and final calories -/
def joggingTime (initialCalories : ℕ) (burnRate : ℕ) (finalCalories : ℕ) : ℕ :=
  (initialCalories - finalCalories) / burnRate

/-- Theorem stating that Will's jogging time is 30 minutes -/
theorem wills_jogging_time :
  let initialCalories : ℕ := 900
  let burnRate : ℕ := 10
  let finalCalories : ℕ := 600
  joggingTime initialCalories burnRate finalCalories = 30 := by
  sorry

end NUMINAMATH_CALUDE_wills_jogging_time_l3017_301770


namespace NUMINAMATH_CALUDE_max_value_inequality_l3017_301745

theorem max_value_inequality (x : ℝ) : 
  (∀ y, y > x → (6 + 5*y + y^2) * Real.sqrt (2*y^2 - y^3 - y) > 0) → 
  ((6 + 5*x + x^2) * Real.sqrt (2*x^2 - x^3 - x) ≤ 0) → 
  x ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_max_value_inequality_l3017_301745


namespace NUMINAMATH_CALUDE_weight_loss_days_l3017_301796

/-- Calculates the number of days required to lose a target weight given daily calorie intake, burn rate, and calories per pound of weight loss. -/
def daysToLoseWeight (caloriesEaten : ℕ) (caloriesBurned : ℕ) (caloriesPerPound : ℕ) (targetPounds : ℕ) : ℕ :=
  (caloriesPerPound * targetPounds) / (caloriesBurned - caloriesEaten)

theorem weight_loss_days :
  daysToLoseWeight 1800 2300 4000 10 = 80 := by
  sorry

end NUMINAMATH_CALUDE_weight_loss_days_l3017_301796


namespace NUMINAMATH_CALUDE_astrophysics_degrees_l3017_301715

def microphotonics : Real := 12
def home_electronics : Real := 24
def food_additives : Real := 15
def genetically_modified_microorganisms : Real := 29
def industrial_lubricants : Real := 8
def total_degrees : Real := 360

def other_sectors_total : Real :=
  microphotonics + home_electronics + food_additives + 
  genetically_modified_microorganisms + industrial_lubricants

def astrophysics_percentage : Real := 100 - other_sectors_total

theorem astrophysics_degrees : 
  (astrophysics_percentage / 100) * total_degrees = 43.2 := by
  sorry

end NUMINAMATH_CALUDE_astrophysics_degrees_l3017_301715


namespace NUMINAMATH_CALUDE_prob_three_red_prob_same_color_prob_not_same_color_l3017_301702

-- Define the probability of drawing a red ball
def prob_red : ℚ := 1 / 2

-- Define the probability of drawing a yellow ball
def prob_yellow : ℚ := 1 - prob_red

-- Define the number of draws
def num_draws : ℕ := 3

-- Theorem for the probability of drawing three red balls
theorem prob_three_red :
  prob_red ^ num_draws = 1 / 8 := by sorry

-- Theorem for the probability of drawing three balls of the same color
theorem prob_same_color :
  prob_red ^ num_draws + prob_yellow ^ num_draws = 1 / 4 := by sorry

-- Theorem for the probability of not drawing all three balls of the same color
theorem prob_not_same_color :
  1 - (prob_red ^ num_draws + prob_yellow ^ num_draws) = 3 / 4 := by sorry

end NUMINAMATH_CALUDE_prob_three_red_prob_same_color_prob_not_same_color_l3017_301702


namespace NUMINAMATH_CALUDE_intersection_sum_l3017_301759

theorem intersection_sum (a b : ℝ) : 
  (3 = (1/3) * 6 + a) ∧ (6 = (1/3) * 3 + b) → a + b = 6 := by
  sorry

end NUMINAMATH_CALUDE_intersection_sum_l3017_301759


namespace NUMINAMATH_CALUDE_paulas_walking_distance_l3017_301792

/-- Represents a pedometer with a maximum step count before reset --/
structure Pedometer where
  max_steps : ℕ
  steps_per_km : ℕ

/-- Represents the yearly walking data --/
structure YearlyWalkingData where
  pedometer : Pedometer
  resets : ℕ
  final_reading : ℕ

def calculate_total_steps (data : YearlyWalkingData) : ℕ :=
  data.resets * (data.pedometer.max_steps + 1) + data.final_reading

def calculate_kilometers (data : YearlyWalkingData) : ℚ :=
  (calculate_total_steps data : ℚ) / data.pedometer.steps_per_km

theorem paulas_walking_distance (data : YearlyWalkingData) 
  (h1 : data.pedometer.max_steps = 49999)
  (h2 : data.pedometer.steps_per_km = 1200)
  (h3 : data.resets = 76)
  (h4 : data.final_reading = 25000) :
  ∃ (k : ℕ), k ≥ 3187 ∧ k ≤ 3200 ∧ calculate_kilometers data = k := by
  sorry

#eval calculate_kilometers {
  pedometer := { max_steps := 49999, steps_per_km := 1200 },
  resets := 76,
  final_reading := 25000
}

end NUMINAMATH_CALUDE_paulas_walking_distance_l3017_301792


namespace NUMINAMATH_CALUDE_convex_n_gon_interior_angles_ratio_l3017_301780

theorem convex_n_gon_interior_angles_ratio (n : ℕ) : 
  n ≥ 3 →
  ∃ x : ℝ, x > 0 ∧
    (∀ k : ℕ, k ≤ n → k * x < 180) ∧
    n * (n + 1) / 2 * x = (n - 2) * 180 →
  n = 3 ∨ n = 4 :=
sorry

end NUMINAMATH_CALUDE_convex_n_gon_interior_angles_ratio_l3017_301780


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l3017_301788

theorem sum_of_roots_quadratic : ∀ (a b c : ℝ), a ≠ 0 → 
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0) →
  (∃ s : ℝ, s = -(b / a) ∧ s = 7) := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l3017_301788


namespace NUMINAMATH_CALUDE_spinner_probability_l3017_301721

theorem spinner_probability (p_A p_B p_C p_D p_E : ℚ) : 
  p_A = 1/5 →
  p_B = 1/3 →
  p_C = p_D →
  p_E = 2 * p_C →
  p_A + p_B + p_C + p_D + p_E = 1 →
  p_C = 7/60 := by
sorry

end NUMINAMATH_CALUDE_spinner_probability_l3017_301721


namespace NUMINAMATH_CALUDE_w_squared_value_l3017_301737

theorem w_squared_value (w : ℝ) (h : (w + 10)^2 = (4*w + 6)*(w + 5)) : w^2 = 70/3 := by
  sorry

end NUMINAMATH_CALUDE_w_squared_value_l3017_301737


namespace NUMINAMATH_CALUDE_rotten_oranges_percentage_l3017_301793

/-- Proves that the percentage of rotten oranges is 15% given the problem conditions -/
theorem rotten_oranges_percentage
  (total_oranges : ℕ)
  (total_bananas : ℕ)
  (rotten_bananas_percentage : ℚ)
  (good_fruits_percentage : ℚ)
  (h1 : total_oranges = 600)
  (h2 : total_bananas = 400)
  (h3 : rotten_bananas_percentage = 4 / 100)
  (h4 : good_fruits_percentage = 894 / 1000)
  : (90 : ℚ) / total_oranges = 15 / 100 := by
  sorry

#check rotten_oranges_percentage

end NUMINAMATH_CALUDE_rotten_oranges_percentage_l3017_301793


namespace NUMINAMATH_CALUDE_shooting_score_proof_l3017_301797

theorem shooting_score_proof (total_shots : ℕ) (total_score : ℕ) (ten_point_shots : ℕ) (remaining_shots : ℕ) :
  total_shots = 10 →
  total_score = 90 →
  ten_point_shots = 4 →
  remaining_shots = total_shots - ten_point_shots →
  (∃ (seven_point_shots eight_point_shots nine_point_shots : ℕ),
    seven_point_shots + eight_point_shots + nine_point_shots = remaining_shots ∧
    7 * seven_point_shots + 8 * eight_point_shots + 9 * nine_point_shots = total_score - 10 * ten_point_shots) →
  (∃ (nine_point_shots : ℕ), nine_point_shots = 3) :=
by sorry

end NUMINAMATH_CALUDE_shooting_score_proof_l3017_301797


namespace NUMINAMATH_CALUDE_geometric_sequence_general_term_l3017_301711

/-- Geometric sequence with first three terms summing to 14 and common ratio 2 -/
def GeometricSequence (a : ℕ+ → ℝ) : Prop :=
  (∀ n : ℕ+, a (n + 1) = 2 * a n) ∧ 
  (a 1 + a 2 + a 3 = 14)

/-- The general term of the geometric sequence is 2^n -/
theorem geometric_sequence_general_term (a : ℕ+ → ℝ) 
  (h : GeometricSequence a) : 
  ∀ n : ℕ+, a n = 2^(n : ℝ) := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_general_term_l3017_301711


namespace NUMINAMATH_CALUDE_smallest_positive_period_dependence_l3017_301713

noncomputable def f (a b x : ℝ) : ℝ := a * (Real.cos x)^2 + b * Real.sin x + Real.tan x

theorem smallest_positive_period_dependence (a b : ℝ) :
  ∃ (p : ℝ), p > 0 ∧ 
  (∀ (x : ℝ), f a b (x + p) = f a b x) ∧
  (∀ (q : ℝ), 0 < q ∧ q < p → ∃ (x : ℝ), f a b (x + q) ≠ f a b x) ∧
  (∀ (a' : ℝ), ∃ (p' : ℝ), p' > 0 ∧ 
    (∀ (x : ℝ), f a' b (x + p') = f a' b x) ∧
    (∀ (q : ℝ), 0 < q ∧ q < p' → ∃ (x : ℝ), f a' b (x + q) ≠ f a' b x) ∧
    p' = p) ∧
  (∃ (b' : ℝ), b' ≠ b → 
    ∀ (p' : ℝ), (∀ (x : ℝ), f a b' (x + p') = f a b' x) →
    (∀ (q : ℝ), 0 < q ∧ q < p' → ∃ (x : ℝ), f a b' (x + q) ≠ f a b' x) →
    p' ≠ p) :=
by sorry


end NUMINAMATH_CALUDE_smallest_positive_period_dependence_l3017_301713


namespace NUMINAMATH_CALUDE_equation_solution_l3017_301747

theorem equation_solution : ∃ x : ℝ, 
  (1 / (x + 9) + 1 / (x + 7) = 1 / (x + 10) + 1 / (x + 6)) ∧ 
  x = -8 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l3017_301747


namespace NUMINAMATH_CALUDE_root_of_f_equals_two_max_m_for_inequality_ab_equals_one_l3017_301768

noncomputable section

-- Define the function f
def f (a b x : ℝ) : ℝ := a^x + b^x

-- Define the conditions on a and b
class PositiveNotOne (r : ℝ) : Prop where
  pos : r > 0
  not_one : r ≠ 1

-- Theorem 1a
theorem root_of_f_equals_two 
  (h₁ : PositiveNotOne 2) 
  (h₂ : PositiveNotOne (1/2)) :
  ∃ x : ℝ, f 2 (1/2) x = 2 ∧ x = 0 := by sorry

-- Theorem 1b
theorem max_m_for_inequality 
  (h₁ : PositiveNotOne 2) 
  (h₂ : PositiveNotOne (1/2)) :
  ∃ m : ℝ, (∀ x : ℝ, f 2 (1/2) (2*x) ≥ m * f 2 (1/2) x - 6) ∧ 
  (∀ m' : ℝ, (∀ x : ℝ, f 2 (1/2) (2*x) ≥ m' * f 2 (1/2) x - 6) → m' ≤ m) ∧
  m = 4 := by sorry

-- Define function g
def g (a b x : ℝ) : ℝ := f a b x - 2

-- Theorem 2
theorem ab_equals_one 
  (ha : 0 < a ∧ a < 1) 
  (hb : b > 1) 
  (h : PositiveNotOne a) 
  (h' : PositiveNotOne b) 
  (hg : ∃! x : ℝ, g a b x = 0) :
  a * b = 1 := by sorry

end NUMINAMATH_CALUDE_root_of_f_equals_two_max_m_for_inequality_ab_equals_one_l3017_301768


namespace NUMINAMATH_CALUDE_range_of_inequality_l3017_301764

-- Define an even function that is monotonically increasing on [0, +∞)
def f (x : ℝ) : ℝ := sorry

-- Define the properties of f
axiom f_even : ∀ x, f (-x) = f x
axiom f_monotone : ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y

-- Theorem statement
theorem range_of_inequality :
  ∀ x : ℝ, f (2 * x - 1) ≤ f 3 ↔ -1 ≤ x ∧ x ≤ 2 := by sorry

end NUMINAMATH_CALUDE_range_of_inequality_l3017_301764


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l3017_301728

open Real

theorem triangle_abc_properties (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  a > 0 ∧ b > 0 ∧ c > 0 →
  sin A / a = sin B / b →
  sin B / b = sin C / c →
  2 * sin A - sin B = 2 * sin C * cos B →
  c = 2 →
  C = π / 3 ∧ ∀ x, (2 * a - b = x) → -2 < x ∧ x < 4 :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l3017_301728


namespace NUMINAMATH_CALUDE_x_power_243_minus_inverse_l3017_301722

theorem x_power_243_minus_inverse (x : ℝ) (h : x - 1/x = Real.sqrt 3) : 
  x^243 - 1/x^243 = 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_x_power_243_minus_inverse_l3017_301722


namespace NUMINAMATH_CALUDE_largest_multiple_of_12_less_than_neg_95_l3017_301777

theorem largest_multiple_of_12_less_than_neg_95 : 
  ∀ n : ℤ, n * 12 < -95 → n * 12 ≤ -96 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_12_less_than_neg_95_l3017_301777


namespace NUMINAMATH_CALUDE_max_value_of_g_l3017_301751

def g (x : ℝ) : ℝ := 5 * x - x^5

theorem max_value_of_g :
  ∃ (max : ℝ), max = 4 ∧
  ∀ x : ℝ, 0 ≤ x → x ≤ Real.sqrt 5 → g x ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_of_g_l3017_301751


namespace NUMINAMATH_CALUDE_garden_perimeter_garden_perimeter_proof_l3017_301723

/-- The perimeter of a rectangular garden with width 24 meters and the same area as a rectangular playground of length 16 meters and width 12 meters is equal to 64 meters. -/
theorem garden_perimeter : ℝ → ℝ → ℝ → ℝ → Prop :=
  fun (garden_width playground_length playground_width garden_perimeter : ℝ) =>
    garden_width = 24 ∧
    playground_length = 16 ∧
    playground_width = 12 ∧
    garden_width * (playground_length * playground_width / garden_width) = playground_length * playground_width →
    garden_perimeter = 2 * (garden_width + (playground_length * playground_width / garden_width)) →
    garden_perimeter = 64

/-- Proof of the garden_perimeter theorem -/
theorem garden_perimeter_proof : garden_perimeter 24 16 12 64 := by
  sorry

end NUMINAMATH_CALUDE_garden_perimeter_garden_perimeter_proof_l3017_301723


namespace NUMINAMATH_CALUDE_equation_solutions_l3017_301794

def equation (x : ℝ) : Prop :=
  1 / (x^2 + 14*x - 36) + 1 / (x^2 + 5*x - 14) + 1 / (x^2 - 16*x - 36) = 0

theorem equation_solutions :
  {x : ℝ | equation x} = {9, -4, 12, 3} := by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3017_301794


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l3017_301798

theorem least_subtraction_for_divisibility (n m k : ℕ) (h : n - k ≡ 0 [MOD m]) : 
  ∀ j < k, ¬(n - j ≡ 0 [MOD m]) → k = n % m :=
sorry

-- The specific problem instance
def original_number : ℕ := 1852745
def divisor : ℕ := 251
def subtrahend : ℕ := 130

theorem problem_solution :
  (original_number - subtrahend) % divisor = 0 ∧
  ∀ j < subtrahend, (original_number - j) % divisor ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l3017_301798


namespace NUMINAMATH_CALUDE_ams_sequence_results_in_14_l3017_301783

/-- Milly's operation: multiply by 3 -/
def milly (x : ℤ) : ℤ := 3 * x

/-- Abby's operation: add 2 -/
def abby (x : ℤ) : ℤ := x + 2

/-- Sam's operation: subtract 1 -/
def sam (x : ℤ) : ℤ := x - 1

/-- The theorem stating that applying Abby's, Milly's, and Sam's operations in order to 3 results in 14 -/
theorem ams_sequence_results_in_14 : sam (milly (abby 3)) = 14 := by
  sorry

end NUMINAMATH_CALUDE_ams_sequence_results_in_14_l3017_301783


namespace NUMINAMATH_CALUDE_smallest_digit_for_divisibility_by_9_l3017_301791

theorem smallest_digit_for_divisibility_by_9 :
  ∃ (d : Nat), d < 10 ∧ (562000 + d * 100 + 48) % 9 = 0 ∧
  ∀ (k : Nat), k < d → k < 10 → (562000 + k * 100 + 48) % 9 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_digit_for_divisibility_by_9_l3017_301791


namespace NUMINAMATH_CALUDE_modern_growth_pattern_l3017_301744

/-- Represents the different types of population growth patterns --/
inductive PopulationGrowthPattern
  | Traditional
  | Modern
  | Primitive
  | TransitionPrimitiveToTraditional

/-- Represents the level of a demographic rate --/
inductive RateLevel
  | Low
  | Medium
  | High

/-- Represents a country --/
structure Country where
  birthRate : RateLevel
  deathRate : RateLevel
  naturalGrowthRate : RateLevel

/-- Determines the population growth pattern of a country --/
def determineGrowthPattern (c : Country) : PopulationGrowthPattern :=
  sorry

theorem modern_growth_pattern (ourCountry : Country) 
  (h1 : ourCountry.birthRate = RateLevel.Low)
  (h2 : ourCountry.deathRate = RateLevel.Low)
  (h3 : ourCountry.naturalGrowthRate = RateLevel.Low) :
  determineGrowthPattern ourCountry = PopulationGrowthPattern.Modern :=
sorry

end NUMINAMATH_CALUDE_modern_growth_pattern_l3017_301744


namespace NUMINAMATH_CALUDE_shirts_not_all_on_sale_l3017_301740

-- Define the universe of discourse
variable (Shirt : Type)
-- Define the property of being on sale
variable (on_sale : Shirt → Prop)
-- Define the property of being in the store
variable (in_store : Shirt → Prop)

-- Theorem statement
theorem shirts_not_all_on_sale 
  (h : ¬ (∀ s : Shirt, in_store s → on_sale s)) : 
  (∃ s : Shirt, in_store s ∧ ¬ on_sale s) ∧ 
  (¬ (∀ s : Shirt, in_store s → on_sale s)) := by
  sorry


end NUMINAMATH_CALUDE_shirts_not_all_on_sale_l3017_301740


namespace NUMINAMATH_CALUDE_tangent_line_range_l3017_301733

/-- Given a circle and a line, if there exists a point on the line such that
    the tangents from this point to the circle form a 60° angle,
    then the parameter k in the line equation is between -2√2 and 2√2. -/
theorem tangent_line_range (k : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 = 1 ∧ x + y + k = 0 ∧ 
   ∃ (p : ℝ × ℝ), p.1 + p.2 + k = 0 ∧ 
   ∃ (a b : ℝ × ℝ), a.1^2 + a.2^2 = 1 ∧ b.1^2 + b.2^2 = 1 ∧ 
   ((p.1 - a.1)*(b.1 - a.1) + (p.2 - a.2)*(b.2 - a.2))^2 = 
   ((p.1 - a.1)^2 + (p.2 - a.2)^2) * ((b.1 - a.1)^2 + (b.2 - a.2)^2) / 4) →
  -2 * Real.sqrt 2 ≤ k ∧ k ≤ 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_range_l3017_301733


namespace NUMINAMATH_CALUDE_lattice_triangle_circumcircle_diameter_bound_l3017_301776

/-- A triangle with vertices on lattice points -/
structure LatticeTriangle where
  vertices : Fin 3 → ℤ × ℤ

/-- The side lengths of a LatticeTriangle -/
def side_lengths (t : LatticeTriangle) : Fin 3 → ℝ := sorry

/-- The diameter of the circumcircle of a LatticeTriangle -/
def circumcircle_diameter (t : LatticeTriangle) : ℝ := sorry

/-- Theorem: The diameter of the circumcircle of a triangle with lattice point vertices
    does not exceed the product of its side lengths -/
theorem lattice_triangle_circumcircle_diameter_bound (t : LatticeTriangle) :
  circumcircle_diameter t ≤ (side_lengths t 0) * (side_lengths t 1) * (side_lengths t 2) := by
  sorry

end NUMINAMATH_CALUDE_lattice_triangle_circumcircle_diameter_bound_l3017_301776


namespace NUMINAMATH_CALUDE_negative_two_less_than_negative_one_l3017_301735

theorem negative_two_less_than_negative_one : -2 < -1 := by
  sorry

end NUMINAMATH_CALUDE_negative_two_less_than_negative_one_l3017_301735


namespace NUMINAMATH_CALUDE_min_horizontal_distance_l3017_301787

def f (x : ℝ) := x^3 - x^2 - x - 6

theorem min_horizontal_distance :
  ∃ (x1 x2 : ℝ),
    f x1 = 8 ∧
    f x2 = -8 ∧
    ∀ (y1 y2 : ℝ),
      f y1 = 8 → f y2 = -8 →
      |x1 - x2| ≤ |y1 - y2| ∧
      |x1 - x2| = 1 :=
sorry

end NUMINAMATH_CALUDE_min_horizontal_distance_l3017_301787


namespace NUMINAMATH_CALUDE_tamika_always_wins_l3017_301714

theorem tamika_always_wins : ∀ a b : ℕ, 
  a ∈ ({11, 12, 13} : Set ℕ) → 
  b ∈ ({11, 12, 13} : Set ℕ) → 
  a ≠ b → 
  a * b > (2 + 3 + 4) := by
sorry

end NUMINAMATH_CALUDE_tamika_always_wins_l3017_301714


namespace NUMINAMATH_CALUDE_perimeter_of_triangle_MNO_l3017_301734

/-- A right prism with equilateral triangular bases -/
structure RightPrism :=
  (height : ℝ)
  (base_side : ℝ)

/-- Points on the edges of the prism -/
structure PrismPoints (prism : RightPrism) :=
  (M : ℝ × ℝ × ℝ)
  (N : ℝ × ℝ × ℝ)
  (O : ℝ × ℝ × ℝ)

/-- The perimeter of triangle MNO in the prism -/
def triangle_perimeter (prism : RightPrism) (points : PrismPoints prism) : ℝ :=
  sorry

/-- Theorem stating the perimeter of triangle MNO -/
theorem perimeter_of_triangle_MNO (prism : RightPrism) (points : PrismPoints prism) 
  (h1 : prism.height = 20)
  (h2 : prism.base_side = 10)
  (h3 : points.M = (5, 0, 0))
  (h4 : points.N = (5, 5*Real.sqrt 3, 0))
  (h5 : points.O = (5, 0, 10)) :
  triangle_perimeter prism points = 5 + 10 * Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_perimeter_of_triangle_MNO_l3017_301734


namespace NUMINAMATH_CALUDE_mooncake_packing_l3017_301767

theorem mooncake_packing :
  ∃ (x y : ℕ), 
    9 * x + 4 * y = 35 ∧ 
    (∀ (a b : ℕ), 9 * a + 4 * b = 35 → x + y ≤ a + b) ∧
    x + y = 5 := by
  sorry

end NUMINAMATH_CALUDE_mooncake_packing_l3017_301767


namespace NUMINAMATH_CALUDE_regions_less_than_199_with_99_lines_l3017_301756

/-- The number of regions created by dividing a plane with lines -/
def num_regions (num_lines : ℕ) (all_parallel : Bool) (all_concurrent : Bool) : ℕ :=
  if all_parallel then
    num_lines + 1
  else if all_concurrent then
    2 * num_lines - 1
  else
    1 + num_lines + (num_lines.choose 2)

/-- Theorem stating the possible number of regions less than 199 when 99 lines divide a plane -/
theorem regions_less_than_199_with_99_lines :
  let possible_regions := {n : ℕ | n < 199 ∧ ∃ (parallel concurrent : Bool), 
    num_regions 99 parallel concurrent = n}
  possible_regions = {100, 198} := by
  sorry

end NUMINAMATH_CALUDE_regions_less_than_199_with_99_lines_l3017_301756


namespace NUMINAMATH_CALUDE_ratio_solution_set_l3017_301727

-- Define the real-valued functions f and g
variable (f g : ℝ → ℝ)

-- Define the solution set of f(x) ≥ 0
def solution_set_f (f : ℝ → ℝ) : Set ℝ := {x | f x ≥ 0}

-- Define the solution set of g(x) ≥ 0
def solution_set_g (g : ℝ → ℝ) : Set ℝ := {x | g x ≥ 0}

-- Define the solution set of f(x)/g(x) > 0
def solution_set_ratio (f g : ℝ → ℝ) : Set ℝ := {x | f x / g x > 0}

-- State the theorem
theorem ratio_solution_set 
  (h1 : solution_set_f f = Set.Icc 1 2) 
  (h2 : solution_set_g g = ∅) : 
  solution_set_ratio f g = Set.Ioi 2 ∪ Set.Iio 1 := by
  sorry

end NUMINAMATH_CALUDE_ratio_solution_set_l3017_301727


namespace NUMINAMATH_CALUDE_emily_new_salary_l3017_301763

def emily_initial_salary : ℕ := 1000000
def employee_salaries : List ℕ := [30000, 30000, 25000, 35000, 20000]
def min_salary : ℕ := 35000
def tax_rate : ℚ := 15 / 100

def calculate_new_salary (initial_salary : ℕ) (employee_salaries : List ℕ) (min_salary : ℕ) (tax_rate : ℚ) : ℕ :=
  sorry

theorem emily_new_salary :
  calculate_new_salary emily_initial_salary employee_salaries min_salary tax_rate = 959750 :=
sorry

end NUMINAMATH_CALUDE_emily_new_salary_l3017_301763


namespace NUMINAMATH_CALUDE_cos_96_cos_24_minus_sin_96_cos_66_l3017_301725

theorem cos_96_cos_24_minus_sin_96_cos_66 : 
  Real.cos (96 * π / 180) * Real.cos (24 * π / 180) - 
  Real.sin (96 * π / 180) * Real.cos (66 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_96_cos_24_minus_sin_96_cos_66_l3017_301725


namespace NUMINAMATH_CALUDE_length_of_lm_l3017_301717

/-- An isosceles triangle with given properties -/
structure IsoscelesTriangle where
  area : ℝ
  altitude : ℝ
  base : ℝ

/-- A line segment parallel to the base of the triangle -/
structure ParallelLine where
  length : ℝ

/-- The resulting trapezoid after cutting the triangle -/
structure Trapezoid where
  area : ℝ

/-- Theorem: Length of LM in the given isosceles triangle scenario -/
theorem length_of_lm (triangle : IsoscelesTriangle) (trapezoid : Trapezoid) 
    (h1 : triangle.area = 200)
    (h2 : triangle.altitude = 40)
    (h3 : trapezoid.area = 150)
    (h4 : triangle.base = 2 * triangle.area / triangle.altitude) :
  ∃ (lm : ParallelLine), lm.length = 5 := by
  sorry

end NUMINAMATH_CALUDE_length_of_lm_l3017_301717


namespace NUMINAMATH_CALUDE_magnitude_of_perpendicular_vector_l3017_301765

/-- Given two planar vectors a and b, where a is perpendicular to b,
    prove that the magnitude of b is √5 --/
theorem magnitude_of_perpendicular_vector
  (a b : ℝ × ℝ)
  (h1 : a = (1, 2))
  (h2 : b.1 = -2)
  (h3 : a.1 * b.1 + a.2 * b.2 = 0) :
  Real.sqrt (b.1^2 + b.2^2) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_perpendicular_vector_l3017_301765


namespace NUMINAMATH_CALUDE_amandas_flowers_l3017_301749

theorem amandas_flowers (amanda_flowers : ℕ) (peter_flowers : ℕ) : 
  peter_flowers = 3 * amanda_flowers →
  peter_flowers - 15 = 45 →
  amanda_flowers = 20 := by
sorry

end NUMINAMATH_CALUDE_amandas_flowers_l3017_301749


namespace NUMINAMATH_CALUDE_sixth_candy_to_pete_l3017_301726

/-- Represents the recipients of candy wrappers -/
inductive Recipient : Type
  | Pete : Recipient
  | Vasey : Recipient

/-- Represents the sequence of candy wrapper distributions -/
def CandySequence : Fin 6 → Recipient
  | ⟨0, _⟩ => Recipient.Pete
  | ⟨1, _⟩ => Recipient.Pete
  | ⟨2, _⟩ => Recipient.Pete
  | ⟨3, _⟩ => Recipient.Vasey
  | ⟨4, _⟩ => Recipient.Vasey
  | ⟨5, _⟩ => Recipient.Pete

theorem sixth_candy_to_pete :
  CandySequence ⟨5, by norm_num⟩ = Recipient.Pete := by sorry

end NUMINAMATH_CALUDE_sixth_candy_to_pete_l3017_301726


namespace NUMINAMATH_CALUDE_triangle_midpoints_sum_l3017_301705

theorem triangle_midpoints_sum (a b c : ℝ) : 
  a + b + c = 15 → 
  a - b = 3 → 
  (a + b) / 2 + (b + c) / 2 + (c + a) / 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_triangle_midpoints_sum_l3017_301705


namespace NUMINAMATH_CALUDE_tree_spacing_l3017_301729

theorem tree_spacing (total_length : ℕ) (num_trees : ℕ) (tree_space : ℕ) 
  (h1 : total_length = 157)
  (h2 : num_trees = 13)
  (h3 : tree_space = 1) :
  (total_length - num_trees * tree_space) / (num_trees - 1) = 12 :=
sorry

end NUMINAMATH_CALUDE_tree_spacing_l3017_301729


namespace NUMINAMATH_CALUDE_positive_combination_l3017_301753

theorem positive_combination (x y : ℝ) (h1 : x + y > 0) (h2 : 4 * x + y > 0) : 
  8 * x + 5 * y > 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_combination_l3017_301753


namespace NUMINAMATH_CALUDE_range_of_m_l3017_301790

-- Define the propositions p and q
def p (x : ℝ) : Prop := (x + 1) * (x - 3) < 0
def q (x m : ℝ) : Prop := 3 * x - 4 < m

-- Define the necessary but not sufficient condition
def necessary_but_not_sufficient (p q : ℝ → Prop) : Prop :=
  (∀ x, q x → p x) ∧ ∃ x, p x ∧ ¬q x

-- Theorem statement
theorem range_of_m (m : ℝ) :
  (∃ x, q x m) ∧ necessary_but_not_sufficient (p · ) (q · m) ↔ m ≥ 5 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l3017_301790


namespace NUMINAMATH_CALUDE_problem_statement_l3017_301775

theorem problem_statement (x y : ℝ) (h1 : y > x) (h2 : x > 0) (h3 : x / y + y / x = 4) :
  (x + y) / (x - y) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3017_301775


namespace NUMINAMATH_CALUDE_third_tea_price_l3017_301716

/-- The price of the first variety of tea in Rs per kg -/
def price1 : ℝ := 126

/-- The price of the second variety of tea in Rs per kg -/
def price2 : ℝ := 135

/-- The price of the mixture in Rs per kg -/
def mixPrice : ℝ := 153

/-- The ratio of the first variety in the mixture -/
def ratio1 : ℝ := 1

/-- The ratio of the second variety in the mixture -/
def ratio2 : ℝ := 1

/-- The ratio of the third variety in the mixture -/
def ratio3 : ℝ := 2

/-- The theorem stating the price of the third variety of tea -/
theorem third_tea_price : 
  ∃ (price3 : ℝ), 
    (ratio1 * price1 + ratio2 * price2 + ratio3 * price3) / (ratio1 + ratio2 + ratio3) = mixPrice ∧ 
    price3 = 175.5 := by
  sorry

end NUMINAMATH_CALUDE_third_tea_price_l3017_301716

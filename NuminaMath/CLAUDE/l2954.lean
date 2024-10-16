import Mathlib

namespace NUMINAMATH_CALUDE_math_test_score_difference_l2954_295495

theorem math_test_score_difference :
  ∀ (grant_score john_score hunter_score : ℕ),
    grant_score = 100 →
    john_score = 2 * hunter_score →
    hunter_score = 45 →
    grant_score - john_score = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_math_test_score_difference_l2954_295495


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l2954_295432

-- Problem 1
theorem problem_1 (x : ℝ) (h : x^2 - 2*x = 5) : 2*x^2 - 4*x + 2023 = 2033 := by
  sorry

-- Problem 2
theorem problem_2 (m n : ℝ) (h : m - n = -3) : 2*(m-n) - m + n + 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l2954_295432


namespace NUMINAMATH_CALUDE_base_conversion_theorem_l2954_295433

theorem base_conversion_theorem (n : ℕ) (C D : ℕ) : 
  n > 0 ∧ 
  C < 8 ∧ 
  D < 5 ∧ 
  n = 8 * C + D ∧ 
  n = 5 * D + C → 
  n = 0 := by sorry

end NUMINAMATH_CALUDE_base_conversion_theorem_l2954_295433


namespace NUMINAMATH_CALUDE_abc_negative_root_at_four_y1_greater_y2_l2954_295475

/-- Represents a parabola y = ax² + bx + c with vertex at (1, n) and 4a - 2b + c = 0 -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  n : ℝ
  vertex_x : a * 1 + b = 0
  vertex_y : a * 1^2 + b * 1 + c = n
  condition : 4 * a - 2 * b + c = 0

/-- If n > 0, then abc < 0 -/
theorem abc_negative (p : Parabola) (h : p.n > 0) : p.a * p.b * p.c < 0 := by sorry

/-- The equation ax² + bx + c = 0 has a root at x = 4 -/
theorem root_at_four (p : Parabola) : p.a * 4^2 + p.b * 4 + p.c = 0 := by sorry

/-- For any two points A(x₁, y₁) and B(x₂, y₂) on the parabola with x₁ < x₂, 
    if a(x₁ + x₂ - 2) < 0, then y₁ > y₂ -/
theorem y1_greater_y2 (p : Parabola) (x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : y₁ = p.a * x₁^2 + p.b * x₁ + p.c)
  (h2 : y₂ = p.a * x₂^2 + p.b * x₂ + p.c)
  (h3 : x₁ < x₂)
  (h4 : p.a * (x₁ + x₂ - 2) < 0) : 
  y₁ > y₂ := by sorry

end NUMINAMATH_CALUDE_abc_negative_root_at_four_y1_greater_y2_l2954_295475


namespace NUMINAMATH_CALUDE_cookie_earnings_proof_l2954_295407

/-- The amount earned by girl scouts from selling cookies -/
def cookie_earnings : ℝ := 30

/-- The cost per person to go to the pool -/
def pool_cost_per_person : ℝ := 2.5

/-- The number of people going to the pool -/
def number_of_people : ℕ := 10

/-- The amount left after paying for the pool -/
def amount_left : ℝ := 5

/-- Theorem stating that the cookie earnings equal $30 -/
theorem cookie_earnings_proof :
  cookie_earnings = pool_cost_per_person * number_of_people + amount_left :=
by sorry

end NUMINAMATH_CALUDE_cookie_earnings_proof_l2954_295407


namespace NUMINAMATH_CALUDE_intersection_area_is_sqrt_k_l2954_295431

/-- Regular tetrahedron with edge length 5 -/
structure RegularTetrahedron where
  edge_length : ℝ
  edge_length_eq : edge_length = 5

/-- Plane passing through specific points of a regular tetrahedron -/
structure IntersectionPlane (t : RegularTetrahedron) where
  -- Midpoint of edge VA
  point_R : ℝ × ℝ × ℝ
  -- Midpoint of edge AB
  point_S : ℝ × ℝ × ℝ
  -- Point one-third from C to B
  point_T : ℝ × ℝ × ℝ

/-- Area of the intersection between the tetrahedron and the plane -/
def intersection_area (t : RegularTetrahedron) (p : IntersectionPlane t) : ℝ := sorry

/-- The theorem to be proved -/
theorem intersection_area_is_sqrt_k (t : RegularTetrahedron) (p : IntersectionPlane t) :
  ∃ k : ℝ, k > 0 ∧ intersection_area t p = Real.sqrt k := by sorry

end NUMINAMATH_CALUDE_intersection_area_is_sqrt_k_l2954_295431


namespace NUMINAMATH_CALUDE_isosceles_triangles_height_ratio_l2954_295477

/-- Two isosceles triangles with equal vertical angles and area ratio 16:49 have height ratio 4:7 -/
theorem isosceles_triangles_height_ratio (b₁ h₁ b₂ h₂ : ℝ) : 
  b₁ > 0 → h₁ > 0 → b₂ > 0 → h₂ > 0 →  -- Positive dimensions
  (1/2 * b₁ * h₁) / (1/2 * b₂ * h₂) = 16/49 →  -- Area ratio
  b₁ / b₂ = h₁ / h₂ →  -- Similar triangles condition
  h₁ / h₂ = 4/7 := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangles_height_ratio_l2954_295477


namespace NUMINAMATH_CALUDE_power_nine_2023_mod_50_l2954_295482

theorem power_nine_2023_mod_50 : 9^2023 % 50 = 29 := by
  sorry

end NUMINAMATH_CALUDE_power_nine_2023_mod_50_l2954_295482


namespace NUMINAMATH_CALUDE_joes_lift_weight_l2954_295447

theorem joes_lift_weight (first_lift second_lift : ℕ) : 
  first_lift + second_lift = 600 →
  2 * first_lift = second_lift + 300 →
  first_lift = 300 := by
  sorry

end NUMINAMATH_CALUDE_joes_lift_weight_l2954_295447


namespace NUMINAMATH_CALUDE_infinitely_many_divisors_of_2_pow_n_plus_1_l2954_295445

theorem infinitely_many_divisors_of_2_pow_n_plus_1 (m : ℕ) :
  (3 ^ m) ∣ (2 ^ (3 ^ m) + 1) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_divisors_of_2_pow_n_plus_1_l2954_295445


namespace NUMINAMATH_CALUDE_even_sum_of_even_l2954_295440

theorem even_sum_of_even (a b : ℤ) : Even a ∧ Even b → Even (a + b) := by sorry

end NUMINAMATH_CALUDE_even_sum_of_even_l2954_295440


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l2954_295425

theorem fraction_equation_solution :
  ∀ x : ℚ, (1 : ℚ) / 3 + (1 : ℚ) / 4 = 1 / x → x = 12 / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l2954_295425


namespace NUMINAMATH_CALUDE_tea_blend_gain_percent_l2954_295467

/-- Represents the cost and quantity of a tea variety -/
structure TeaVariety where
  cost : ℚ
  quantity : ℚ

/-- Calculates the gain percent for a tea blend -/
def gainPercent (tea1 : TeaVariety) (tea2 : TeaVariety) (sellingPrice : ℚ) : ℚ :=
  let totalCost := tea1.cost * tea1.quantity + tea2.cost * tea2.quantity
  let totalQuantity := tea1.quantity + tea2.quantity
  let costPrice := totalCost / totalQuantity
  ((sellingPrice - costPrice) / costPrice) * 100

/-- Theorem stating that the gain percent for the given tea blend is 12% -/
theorem tea_blend_gain_percent :
  let tea1 := TeaVariety.mk 18 5
  let tea2 := TeaVariety.mk 20 3
  let sellingPrice := 21
  gainPercent tea1 tea2 sellingPrice = 12 := by
  sorry

#eval gainPercent (TeaVariety.mk 18 5) (TeaVariety.mk 20 3) 21

end NUMINAMATH_CALUDE_tea_blend_gain_percent_l2954_295467


namespace NUMINAMATH_CALUDE_circle_locus_l2954_295458

/-- The locus of the center of a circle passing through (-2, 0) and tangent to x = 2 -/
theorem circle_locus (x₀ y₀ : ℝ) : 
  (∃ (r : ℝ), r > 0 ∧ 
    (x₀ + 2)^2 + y₀^2 = r^2 ∧ 
    |x₀ - 2| = r) →
  y₀^2 = -8 * x₀ :=
by sorry

end NUMINAMATH_CALUDE_circle_locus_l2954_295458


namespace NUMINAMATH_CALUDE_sin_y_in_terms_of_c_and_d_l2954_295486

theorem sin_y_in_terms_of_c_and_d (c d y : ℝ) 
  (h1 : c > d) (h2 : d > 0) (h3 : 0 < y) (h4 : y < π / 2)
  (h5 : Real.tan y = (3 * c * d) / (c^2 - d^2)) :
  Real.sin y = (3 * c * d) / Real.sqrt (c^4 + 7 * c^2 * d^2 + d^4) := by
  sorry

end NUMINAMATH_CALUDE_sin_y_in_terms_of_c_and_d_l2954_295486


namespace NUMINAMATH_CALUDE_pitcher_juice_distribution_l2954_295434

theorem pitcher_juice_distribution (C : ℝ) (h : C > 0) : 
  let juice_volume := (2 / 3) * C
  let cups := 6
  let juice_per_cup := juice_volume / cups
  (juice_per_cup / C) * 100 = 11.11 := by
  sorry

end NUMINAMATH_CALUDE_pitcher_juice_distribution_l2954_295434


namespace NUMINAMATH_CALUDE_min_value_fraction_l2954_295471

theorem min_value_fraction (x : ℝ) (h : x > 6) : 
  (∀ y > 6, x^2 / (x - 6) ≤ y^2 / (y - 6)) → x^2 / (x - 6) = 24 := by
  sorry

end NUMINAMATH_CALUDE_min_value_fraction_l2954_295471


namespace NUMINAMATH_CALUDE_evaluate_expression_l2954_295411

theorem evaluate_expression : -25 - 7 * (4 + 2) = -67 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2954_295411


namespace NUMINAMATH_CALUDE_right_triangle_with_incircle_l2954_295462

theorem right_triangle_with_incircle (r c a b : ℝ) : 
  r = 15 →  -- radius of incircle
  c = 73 →  -- hypotenuse
  r = (a + b - c) / 2 →  -- incircle radius formula
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  ((a = 55 ∧ b = 48) ∨ (a = 48 ∧ b = 55)) := by sorry

end NUMINAMATH_CALUDE_right_triangle_with_incircle_l2954_295462


namespace NUMINAMATH_CALUDE_triangle_area_l2954_295485

theorem triangle_area (a b c A B C : Real) : 
  a = 7 →
  2 * Real.sin A = Real.sqrt 3 →
  Real.sin B + Real.sin C = 13 * Real.sqrt 3 / 14 →
  (1/2) * a * b * Real.sin C = 10 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l2954_295485


namespace NUMINAMATH_CALUDE_sequence_median_l2954_295423

def sequence_sum (n : ℕ) : ℕ := n * (n + 1) / 2

theorem sequence_median : 
  let total_elements := sequence_sum 100
  let median_position := total_elements / 2
  ∃ k : ℕ, 
    k ≤ 100 ∧ 
    sequence_sum (k - 1) < median_position ∧ 
    median_position ≤ sequence_sum k ∧
    k = 71 := by sorry

end NUMINAMATH_CALUDE_sequence_median_l2954_295423


namespace NUMINAMATH_CALUDE_lines_perpendicular_to_plane_are_parallel_l2954_295468

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- Define the theorem
theorem lines_perpendicular_to_plane_are_parallel
  (m n : Line) (α : Plane)
  (h1 : m ≠ n)
  (h2 : perpendicular m α)
  (h3 : perpendicular n α) :
  parallel m n :=
sorry

end NUMINAMATH_CALUDE_lines_perpendicular_to_plane_are_parallel_l2954_295468


namespace NUMINAMATH_CALUDE_extreme_point_range_l2954_295416

theorem extreme_point_range (m : ℝ) : 
  (∃! x₀ : ℝ, x₀ > 0 ∧ 1/2 ≤ x₀ ∧ x₀ ≤ 3 ∧
    (∀ x : ℝ, x > 0 → (x₀ + 1/x₀ + m = 0 ∧
      ∀ y : ℝ, y > 0 → y ≠ x₀ → y + 1/y + m ≠ 0))) →
  -10/3 ≤ m ∧ m < -5/2 :=
sorry

end NUMINAMATH_CALUDE_extreme_point_range_l2954_295416


namespace NUMINAMATH_CALUDE_max_digit_sum_is_24_l2954_295430

/-- Represents a time in 24-hour format -/
structure Time24 where
  hours : Nat
  minutes : Nat
  hours_valid : hours < 24
  minutes_valid : minutes < 60

/-- Calculates the sum of digits for a given natural number -/
def sumOfDigits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- Calculates the sum of digits for a given time in 24-hour format -/
def timeDigitSum (t : Time24) : Nat :=
  sumOfDigits t.hours + sumOfDigits t.minutes

/-- The maximum possible sum of digits in a 24-hour format display -/
def maxDigitSum : Nat := 24

/-- Theorem stating that the maximum sum of digits in a 24-hour format display is 24 -/
theorem max_digit_sum_is_24 :
  ∀ t : Time24, timeDigitSum t ≤ maxDigitSum :=
by
  sorry

#check max_digit_sum_is_24

end NUMINAMATH_CALUDE_max_digit_sum_is_24_l2954_295430


namespace NUMINAMATH_CALUDE_complex_trajectory_l2954_295444

theorem complex_trajectory (x y : ℝ) (z : ℂ) (h1 : x ≥ 1/2) (h2 : z = x + y * I) (h3 : Complex.abs (z - 1) = x) :
  y^2 = 2*x - 1 :=
sorry

end NUMINAMATH_CALUDE_complex_trajectory_l2954_295444


namespace NUMINAMATH_CALUDE_arc_length_specific_sector_l2954_295429

/-- Arc length formula for a sector -/
def arc_length (r : ℝ) (θ : ℝ) : ℝ := r * θ

/-- Theorem: The length of an arc in a sector with radius 2 and central angle π/3 is 2π/3 -/
theorem arc_length_specific_sector :
  let r : ℝ := 2
  let θ : ℝ := π / 3
  arc_length r θ = 2 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_arc_length_specific_sector_l2954_295429


namespace NUMINAMATH_CALUDE_intersection_A_B_l2954_295428

open Set Real

-- Define sets A and B
def A : Set ℝ := {x | x^2 - x - 2 > 0}
def B : Set ℝ := Ioo 0 3

-- State the theorem
theorem intersection_A_B : A ∩ B = Ioo 2 3 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l2954_295428


namespace NUMINAMATH_CALUDE_roots_sum_properties_l2954_295413

theorem roots_sum_properties (a : ℤ) (x₁ x₂ : ℝ) (h_odd : Odd a) (h_roots : x₁^2 + a*x₁ - 1 = 0 ∧ x₂^2 + a*x₂ - 1 = 0) :
  ∀ n : ℕ, 
    (∃ k : ℤ, x₁^n + x₂^n = k) ∧ 
    (∃ m : ℤ, x₁^(n+1) + x₂^(n+1) = m) ∧ 
    (Int.gcd (↑⌊x₁^n + x₂^n⌋) (↑⌊x₁^(n+1) + x₂^(n+1)⌋) = 1) :=
by sorry

end NUMINAMATH_CALUDE_roots_sum_properties_l2954_295413


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2954_295449

/-- An arithmetic sequence with 10 terms where the sum of odd-numbered terms is 15
    and the sum of even-numbered terms is 30 has a common difference of 3. -/
theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℚ) -- The arithmetic sequence
  (h1 : a 1 + a 3 + a 5 + a 7 + a 9 = 15) -- Sum of odd-numbered terms
  (h2 : a 2 + a 4 + a 6 + a 8 + a 10 = 30) -- Sum of even-numbered terms
  (h3 : ∀ n : ℕ, n < 10 → a (n + 1) - a n = a 2 - a 1) -- Definition of arithmetic sequence
  : a 2 - a 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2954_295449


namespace NUMINAMATH_CALUDE_polynomial_ratio_equals_infinite_sum_l2954_295435

theorem polynomial_ratio_equals_infinite_sum (x : ℝ) (h : x ∈ Set.Ioo 0 1) :
  x / (1 - x) = ∑' n, x^(2^n) / (1 - x^(2^n + 1)) :=
sorry

end NUMINAMATH_CALUDE_polynomial_ratio_equals_infinite_sum_l2954_295435


namespace NUMINAMATH_CALUDE_dannys_travel_time_l2954_295478

theorem dannys_travel_time (danny_time steve_time halfway_danny halfway_steve : ℝ) 
  (h1 : steve_time = 2 * danny_time)
  (h2 : halfway_danny = danny_time / 2)
  (h3 : halfway_steve = steve_time / 2)
  (h4 : halfway_steve - halfway_danny = 12.5) :
  danny_time = 25 := by sorry

end NUMINAMATH_CALUDE_dannys_travel_time_l2954_295478


namespace NUMINAMATH_CALUDE_complement_of_A_in_S_l2954_295406

def S : Set ℝ := Set.univ

def A : Set ℝ := {x : ℝ | x^2 - 2*x - 3 ≤ 0}

theorem complement_of_A_in_S :
  (S \ A) = {x : ℝ | x < -1 ∨ x > 3} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_S_l2954_295406


namespace NUMINAMATH_CALUDE_transportation_cost_optimization_l2954_295409

/-- Transportation cost optimization problem -/
theorem transportation_cost_optimization 
  (distance : ℝ) 
  (max_speed : ℝ) 
  (fixed_cost : ℝ) 
  (variable_cost_factor : ℝ) :
  distance = 1000 →
  max_speed = 80 →
  fixed_cost = 400 →
  variable_cost_factor = 1/4 →
  ∃ (optimal_speed : ℝ),
    optimal_speed > 0 ∧ 
    optimal_speed ≤ max_speed ∧
    optimal_speed = 40 ∧
    ∀ (speed : ℝ), 
      speed > 0 → 
      speed ≤ max_speed → 
      distance * (variable_cost_factor * speed + fixed_cost / speed) ≥ 
      distance * (variable_cost_factor * optimal_speed + fixed_cost / optimal_speed) :=
by sorry


end NUMINAMATH_CALUDE_transportation_cost_optimization_l2954_295409


namespace NUMINAMATH_CALUDE_min_seating_circular_table_l2954_295494

/-- Represents a circular table with chairs and seated people. -/
structure CircularTable where
  chairs : ℕ
  seated : ℕ

/-- Predicate to check if a seating arrangement is valid. -/
def validSeating (table : CircularTable) : Prop :=
  table.seated ≤ table.chairs ∧
  ∀ k : ℕ, k < table.seated → ∃ j : ℕ, j < table.seated ∧ j ≠ k ∧
    (((k + 1) % table.chairs = j) ∨ ((k + table.chairs - 1) % table.chairs = j))

/-- The theorem to be proved. -/
theorem min_seating_circular_table :
  ∃ (n : ℕ), n = 20 ∧
  validSeating ⟨60, n⟩ ∧
  ∀ m : ℕ, m < n → ¬validSeating ⟨60, m⟩ := by
  sorry

end NUMINAMATH_CALUDE_min_seating_circular_table_l2954_295494


namespace NUMINAMATH_CALUDE_forty_seventh_digit_is_six_l2954_295453

def sequence_digit (n : ℕ) : ℕ :=
  let start := 90
  let digit_pos := n - 1
  let num_index := digit_pos / 2
  let in_num_pos := digit_pos % 2
  let current_num := start - num_index
  (current_num / 10^(1 - in_num_pos)) % 10

theorem forty_seventh_digit_is_six :
  sequence_digit 47 = 6 := by
  sorry

end NUMINAMATH_CALUDE_forty_seventh_digit_is_six_l2954_295453


namespace NUMINAMATH_CALUDE_hex_numeric_count_2023_l2954_295490

/-- Converts a positive integer to its hexadecimal representation --/
def to_hex (n : ℕ+) : List (Fin 16) :=
  sorry

/-- Checks if a hexadecimal representation contains only numeric digits (0-9) --/
def hex_only_numeric (l : List (Fin 16)) : Bool :=
  sorry

/-- Counts numbers up to n whose hexadecimal representation contains only numeric digits --/
def count_hex_numeric (n : ℕ+) : ℕ :=
  sorry

/-- Sums the digits of a natural number --/
def sum_digits (n : ℕ) : ℕ :=
  sorry

/-- Theorem statement --/
theorem hex_numeric_count_2023 :
  sum_digits (count_hex_numeric 2023) = 25 :=
sorry

end NUMINAMATH_CALUDE_hex_numeric_count_2023_l2954_295490


namespace NUMINAMATH_CALUDE_find_number_l2954_295481

theorem find_number : ∃ x : ℤ, 27 * (x + 143) = 9693 ∧ x = 216 := by sorry

end NUMINAMATH_CALUDE_find_number_l2954_295481


namespace NUMINAMATH_CALUDE_geometric_sum_first_10_terms_l2954_295460

def geometric_sequence (a₁ : ℚ) (r : ℚ) (n : ℕ) : ℚ := a₁ * r^(n - 1)

def geometric_sum (a₁ : ℚ) (r : ℚ) (n : ℕ) : ℚ := a₁ * (1 - r^n) / (1 - r)

theorem geometric_sum_first_10_terms :
  let a₁ : ℚ := 12
  let r : ℚ := 1/3
  let n : ℕ := 10
  geometric_sum a₁ r n = 1062864/59049 := by sorry

end NUMINAMATH_CALUDE_geometric_sum_first_10_terms_l2954_295460


namespace NUMINAMATH_CALUDE_quadratic_sets_l2954_295496

/-- A quadratic function with a minimum value -/
structure QuadraticWithMinimum where
  a : ℝ
  f : ℝ → ℝ
  hf : f = fun x ↦ a * x^2 + x
  ha : a > 0

/-- The set A where f(x) < 0 -/
def setA (q : QuadraticWithMinimum) : Set ℝ :=
  {x | q.f x < 0}

/-- The set B defined by |x+4| < a -/
def setB (a : ℝ) : Set ℝ :=
  {x | |x + 4| < a}

/-- The main theorem -/
theorem quadratic_sets (q : QuadraticWithMinimum) :
  (setA q = Set.Ioo (-1 / q.a) 0) ∧
  (setB q.a ⊆ setA q ↔ 0 < q.a ∧ q.a ≤ Real.sqrt 5 - 2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sets_l2954_295496


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l2954_295443

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 1 < 0}
def B : Set ℝ := {x | ∃ y, y = Real.log x}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x | x > -1} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l2954_295443


namespace NUMINAMATH_CALUDE_ice_cube_distribution_l2954_295469

/-- Given a total of 30 ice cubes and 6 cups, prove that each cup should contain 5 ice cubes when divided equally. -/
theorem ice_cube_distribution (total_ice_cubes : ℕ) (num_cups : ℕ) (ice_per_cup : ℕ) :
  total_ice_cubes = 30 →
  num_cups = 6 →
  ice_per_cup = total_ice_cubes / num_cups →
  ice_per_cup = 5 := by
  sorry

end NUMINAMATH_CALUDE_ice_cube_distribution_l2954_295469


namespace NUMINAMATH_CALUDE_reflect_d_twice_l2954_295420

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflects a point over the y-axis -/
def reflectOverYAxis (p : Point) : Point :=
  { x := -p.x, y := p.y }

/-- Reflects a point over the line y = -x -/
def reflectOverYEqualNegX (p : Point) : Point :=
  { x := -p.y, y := -p.x }

/-- The main theorem stating that reflecting point D(5,1) over y-axis and then over y=-x results in D''(-1,5) -/
theorem reflect_d_twice :
  let d : Point := { x := 5, y := 1 }
  let d' := reflectOverYAxis d
  let d'' := reflectOverYEqualNegX d'
  d''.x = -1 ∧ d''.y = 5 := by sorry

end NUMINAMATH_CALUDE_reflect_d_twice_l2954_295420


namespace NUMINAMATH_CALUDE_apple_theorem_l2954_295415

/-- Represents the number of apples each person has -/
structure Apples where
  A : ℕ
  B : ℕ
  C : ℕ

/-- The conditions of the apple distribution problem -/
def apple_distribution (a : Apples) : Prop :=
  a.A + a.B + a.C < 100 ∧
  a.A - a.A / 6 - a.A / 4 = a.B + a.A / 6 ∧
  a.B + a.A / 6 = a.C + a.A / 4

theorem apple_theorem (a : Apples) (h : apple_distribution a) :
  a.A ≤ 48 ∧ a.B = a.C + 4 := by
  sorry

#check apple_theorem

end NUMINAMATH_CALUDE_apple_theorem_l2954_295415


namespace NUMINAMATH_CALUDE_sequence_increasing_iff_l2954_295412

theorem sequence_increasing_iff (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) = 2^n - 3 * a n) →
  (∀ n : ℕ, a (n + 1) > a n) ↔
  a 0 = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_sequence_increasing_iff_l2954_295412


namespace NUMINAMATH_CALUDE_age_difference_l2954_295487

/-- Given the ages of four people with specific relationships, prove that Jack's age is 5 years more than twice Shannen's age. -/
theorem age_difference (beckett_age olaf_age shannen_age jack_age : ℕ) : 
  beckett_age = 12 →
  olaf_age = beckett_age + 3 →
  shannen_age = olaf_age - 2 →
  jack_age > 2 * shannen_age →
  beckett_age + olaf_age + shannen_age + jack_age = 71 →
  jack_age - 2 * shannen_age = 5 := by
sorry

end NUMINAMATH_CALUDE_age_difference_l2954_295487


namespace NUMINAMATH_CALUDE_registration_methods_count_l2954_295491

/-- The number of subjects available for registration -/
def num_subjects : ℕ := 4

/-- The number of students registering -/
def num_students : ℕ := 3

/-- The number of different registration methods -/
def registration_methods : ℕ := num_subjects ^ num_students

/-- Theorem stating that the number of registration methods is 64 -/
theorem registration_methods_count : registration_methods = 64 := by sorry

end NUMINAMATH_CALUDE_registration_methods_count_l2954_295491


namespace NUMINAMATH_CALUDE_range_of_a_l2954_295448

theorem range_of_a (x y a : ℝ) (h1 : x > y) (h2 : (a + 3) * x < (a + 3) * y) : a < -3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2954_295448


namespace NUMINAMATH_CALUDE_rain_probability_l2954_295465

theorem rain_probability (umbrellas : ℕ) (take_umbrella_prob : ℝ) :
  umbrellas = 2 →
  take_umbrella_prob = 0.2 →
  ∃ (rain_prob : ℝ),
    rain_prob + (rain_prob / (rain_prob + 1)) - (rain_prob^2 / (rain_prob + 1)) = take_umbrella_prob ∧
    rain_prob = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_rain_probability_l2954_295465


namespace NUMINAMATH_CALUDE_handkerchiefs_per_day_l2954_295497

-- Define the handkerchief dimensions
def handkerchief_side : ℝ := 25

-- Define the total fabric used in square meters
def total_fabric : ℝ := 3

-- Define the number of days
def days : ℕ := 8

-- Define the conversion factor from m² to cm²
def m2_to_cm2 : ℝ := 10000

-- Theorem statement
theorem handkerchiefs_per_day :
  let handkerchief_area : ℝ := handkerchief_side * handkerchief_side
  let total_fabric_cm2 : ℝ := total_fabric * m2_to_cm2
  let total_handkerchiefs : ℝ := total_fabric_cm2 / handkerchief_area
  total_handkerchiefs / days = 6 := by sorry

end NUMINAMATH_CALUDE_handkerchiefs_per_day_l2954_295497


namespace NUMINAMATH_CALUDE_count_lines_with_integer_chord_l2954_295480

/-- Represents a line in the form kx - y - 4k + 1 = 0 --/
structure Line where
  k : ℝ

/-- Represents the circle x^2 + (y + 1)^2 = 25 --/
def Circle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + (p.2 + 1)^2 = 25}

/-- Returns true if the line intersects the circle with a chord of integer length --/
def hasIntegerChord (l : Line) : Prop :=
  ∃ n : ℕ, ∃ p q : ℝ × ℝ,
    p ∈ Circle ∧ q ∈ Circle ∧
    l.k * p.1 - p.2 - 4 * l.k + 1 = 0 ∧
    l.k * q.1 - q.2 - 4 * l.k + 1 = 0 ∧
    (p.1 - q.1)^2 + (p.2 - q.2)^2 = n^2

/-- The theorem to be proved --/
theorem count_lines_with_integer_chord :
  ∃! (s : Finset Line), s.card = 10 ∧ ∀ l ∈ s, hasIntegerChord l :=
sorry

end NUMINAMATH_CALUDE_count_lines_with_integer_chord_l2954_295480


namespace NUMINAMATH_CALUDE_oil_drop_probability_l2954_295417

theorem oil_drop_probability (c : ℝ) (h : c > 0) : 
  (0.5 * c)^2 / (π * (c/2)^2) = 0.25 / π := by
  sorry

end NUMINAMATH_CALUDE_oil_drop_probability_l2954_295417


namespace NUMINAMATH_CALUDE_savings_over_three_years_l2954_295426

def multi_tariff_meter_cost : ℕ := 3500
def installation_cost : ℕ := 1100
def monthly_consumption : ℕ := 300
def night_consumption : ℕ := 230
def day_consumption : ℕ := monthly_consumption - night_consumption
def multi_tariff_day_rate : ℚ := 52/10
def multi_tariff_night_rate : ℚ := 34/10
def standard_rate : ℚ := 46/10

def monthly_cost_multi_tariff : ℚ :=
  (night_consumption : ℚ) * multi_tariff_night_rate + (day_consumption : ℚ) * multi_tariff_day_rate

def monthly_cost_standard : ℚ :=
  (monthly_consumption : ℚ) * standard_rate

def total_cost_multi_tariff (months : ℕ) : ℚ :=
  (multi_tariff_meter_cost : ℚ) + (installation_cost : ℚ) + monthly_cost_multi_tariff * (months : ℚ)

def total_cost_standard (months : ℕ) : ℚ :=
  monthly_cost_standard * (months : ℚ)

theorem savings_over_three_years :
  total_cost_standard 36 - total_cost_multi_tariff 36 = 3824 := by sorry

end NUMINAMATH_CALUDE_savings_over_three_years_l2954_295426


namespace NUMINAMATH_CALUDE_earliest_meeting_time_l2954_295473

theorem earliest_meeting_time (david_lap_time maria_lap_time leo_lap_time : ℕ) 
  (h1 : david_lap_time = 5)
  (h2 : maria_lap_time = 8)
  (h3 : leo_lap_time = 10) :
  Nat.lcm (Nat.lcm david_lap_time maria_lap_time) leo_lap_time = 40 := by
sorry

end NUMINAMATH_CALUDE_earliest_meeting_time_l2954_295473


namespace NUMINAMATH_CALUDE_symmetric_circle_equation_l2954_295455

/-- Given a circle and a line of symmetry, this theorem proves the equation of the symmetric circle. -/
theorem symmetric_circle_equation (x y : ℝ) :
  (x^2 + y^2 + 2*x - 2*y + 1 = 0) →  -- Given circle equation
  (x - y = 0) →                      -- Line of symmetry
  (x^2 + y^2 - 2*x + 2*y + 1 = 0)    -- Symmetric circle equation
:= by sorry

end NUMINAMATH_CALUDE_symmetric_circle_equation_l2954_295455


namespace NUMINAMATH_CALUDE_video_votes_l2954_295419

theorem video_votes (score : ℤ) (like_percentage : ℚ) : 
  score = 130 ∧ like_percentage = 70 / 100 → 
  ∃ total_votes : ℕ, 
    (like_percentage * total_votes : ℚ) - ((1 - like_percentage) * total_votes : ℚ) = score ∧
    total_votes = 325 := by
  sorry

end NUMINAMATH_CALUDE_video_votes_l2954_295419


namespace NUMINAMATH_CALUDE_range_of_a_l2954_295437

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | 2 * x^2 - 3 * x + 1 ≤ 0}
def B (a : ℝ) : Set ℝ := {x : ℝ | x^2 - (2 * a + 1) * x + a * (a + 1) ≤ 0}

-- State the theorem
theorem range_of_a (a : ℝ) : (A ∩ (Set.compl (B a)) = ∅) → (0 ≤ a ∧ a ≤ 1/2) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2954_295437


namespace NUMINAMATH_CALUDE_natasha_hill_climbing_l2954_295405

/-- Natasha's hill climbing problem -/
theorem natasha_hill_climbing
  (time_up : ℝ)
  (time_down : ℝ)
  (avg_speed_total : ℝ)
  (h_time_up : time_up = 4)
  (h_time_down : time_down = 2)
  (h_avg_speed_total : avg_speed_total = 1.5) :
  let distance := avg_speed_total * (time_up + time_down) / 2
  let avg_speed_up := distance / time_up
  avg_speed_up = 1.125 := by
sorry

end NUMINAMATH_CALUDE_natasha_hill_climbing_l2954_295405


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2954_295403

def M : Set ℕ := {1, 2, 5}
def N : Set ℕ := {x | x ≤ 2}

theorem intersection_of_M_and_N : M ∩ N = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2954_295403


namespace NUMINAMATH_CALUDE_complex_magnitude_equation_l2954_295470

theorem complex_magnitude_equation : 
  ∃ (t : ℝ), t > 0 ∧ Complex.abs (8 + 3 * t * Complex.I) = 13 ↔ t = Real.sqrt (105 / 3) :=
by sorry

end NUMINAMATH_CALUDE_complex_magnitude_equation_l2954_295470


namespace NUMINAMATH_CALUDE_lcm_lower_bound_l2954_295466

theorem lcm_lower_bound (a : Fin 10 → ℕ) (h_order : ∀ i j, i < j → a i < a j) :
  Nat.lcm (a 0) (Nat.lcm (a 1) (Nat.lcm (a 2) (Nat.lcm (a 3) (Nat.lcm (a 4) (Nat.lcm (a 5) (Nat.lcm (a 6) (Nat.lcm (a 7) (Nat.lcm (a 8) (a 9))))))))) ≥ 10 * a 0 := by
  sorry

end NUMINAMATH_CALUDE_lcm_lower_bound_l2954_295466


namespace NUMINAMATH_CALUDE_set_S_satisfies_conditions_l2954_295463

def S : Finset Nat := {2, 3, 11, 23, 31}

def P : Nat := S.prod id

theorem set_S_satisfies_conditions :
  (∀ x ∈ S, x > 1) ∧
  (∀ x ∈ S, x ∣ (P / x + 1) ∧ x ≠ (P / x + 1)) := by
  sorry

end NUMINAMATH_CALUDE_set_S_satisfies_conditions_l2954_295463


namespace NUMINAMATH_CALUDE_adam_students_in_ten_years_l2954_295492

theorem adam_students_in_ten_years : 
  let students_per_year : ℕ := 50
  let first_year_students : ℕ := 40
  let total_years : ℕ := 10
  (total_years - 1) * students_per_year + first_year_students = 490 := by
  sorry

end NUMINAMATH_CALUDE_adam_students_in_ten_years_l2954_295492


namespace NUMINAMATH_CALUDE_plane_equation_l2954_295474

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A line in 3D space represented by parametric equations -/
structure Line3D where
  x : ℝ → ℝ
  y : ℝ → ℝ
  z : ℝ → ℝ

/-- A plane in 3D space represented by Ax + By + Cz + D = 0 -/
structure Plane where
  A : ℤ
  B : ℤ
  C : ℤ
  D : ℤ

def point_on_plane (p : Point3D) (plane : Plane) : Prop :=
  plane.A * p.x + plane.B * p.y + plane.C * p.z + plane.D = 0

def line_on_plane (l : Line3D) (plane : Plane) : Prop :=
  ∀ t : ℝ, point_on_plane ⟨l.x t, l.y t, l.z t⟩ plane

def is_solution (plane : Plane) : Prop :=
  let p1 : Point3D := ⟨1, 4, -3⟩
  let p2 : Point3D := ⟨0, -3, 0⟩
  let l : Line3D := { x := λ t => 4 * t + 2, y := λ t => -t - 2, z := λ t => 5 * t + 1 }
  point_on_plane p1 plane ∧
  point_on_plane p2 plane ∧
  line_on_plane l plane ∧
  plane.A > 0 ∧
  Nat.gcd (Int.natAbs plane.A) (Nat.gcd (Int.natAbs plane.B) (Nat.gcd (Int.natAbs plane.C) (Int.natAbs plane.D))) = 1

theorem plane_equation : is_solution ⟨10, 9, -13, 27⟩ := by
  sorry

end NUMINAMATH_CALUDE_plane_equation_l2954_295474


namespace NUMINAMATH_CALUDE_survey_respondents_l2954_295451

theorem survey_respondents (x y : ℕ) : 
  x = 60 → -- 60 people preferred brand X
  x = 3 * y → -- The ratio of preference for X to Y is 3:1
  x + y = 80 -- Total number of respondents
  :=
by
  sorry

end NUMINAMATH_CALUDE_survey_respondents_l2954_295451


namespace NUMINAMATH_CALUDE_find_m_l2954_295488

theorem find_m : ∃ m : ℝ, ∀ x : ℝ, (x + 2) * (x + 3) = x^2 + m*x + 6 → m = 5 := by
  sorry

end NUMINAMATH_CALUDE_find_m_l2954_295488


namespace NUMINAMATH_CALUDE_min_value_of_sum_of_squares_l2954_295456

theorem min_value_of_sum_of_squares (x y z : ℝ) (h : 2*x + 3*y + 4*z = 1) :
  ∃ (m : ℝ), (∀ a b c : ℝ, 2*a + 3*b + 4*c = 1 → a^2 + b^2 + c^2 ≥ m) ∧
             (x^2 + y^2 + z^2 = m) ∧
             (m = 1/29) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_sum_of_squares_l2954_295456


namespace NUMINAMATH_CALUDE_cubic_sum_of_quadratic_roots_l2954_295479

theorem cubic_sum_of_quadratic_roots : ∀ r s : ℝ,
  r^2 - 5*r + 6 = 0 →
  s^2 - 5*s + 6 = 0 →
  r^3 + s^3 = 35 := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_of_quadratic_roots_l2954_295479


namespace NUMINAMATH_CALUDE_rectangle_existence_l2954_295436

/-- A point in a 2D plane -/
structure Point := (x : ℝ) (y : ℝ)

/-- A line in a 2D plane -/
structure Line := (a : ℝ) (b : ℝ) (c : ℝ)

/-- A triangle defined by three points -/
structure Triangle := (K : Point) (L : Point) (M : Point)

/-- A rectangle defined by four points -/
structure Rectangle := (A : Point) (B : Point) (C : Point) (D : Point)

/-- Check if a point lies on a line -/
def Point.on_line (P : Point) (l : Line) : Prop :=
  l.a * P.x + l.b * P.y + l.c = 0

/-- Check if a point lies on the extension of a line segment -/
def Point.on_extension (P : Point) (A : Point) (B : Point) : Prop :=
  ∃ (t : ℝ), t > 1 ∧ P.x = A.x + t * (B.x - A.x) ∧ P.y = A.y + t * (B.y - A.y)

/-- Theorem: Given a triangle and a point on the extension of one side, 
    there exists a rectangle with vertices on the triangle's sides -/
theorem rectangle_existence (T : Triangle) (A : Point) :
  A.on_extension T.L T.K →
  ∃ (R : Rectangle),
    R.A = A ∧
    R.B.on_line (Line.mk (T.M.y - T.K.y) (T.K.x - T.M.x) (T.M.x * T.K.y - T.K.x * T.M.y)) ∧
    R.C.on_line (Line.mk (T.L.y - T.K.y) (T.K.x - T.L.x) (T.L.x * T.K.y - T.K.x * T.L.y)) ∧
    R.D.on_line (Line.mk (T.M.y - T.L.y) (T.L.x - T.M.x) (T.M.x * T.L.y - T.L.x * T.M.y)) :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_existence_l2954_295436


namespace NUMINAMATH_CALUDE_pumpkin_contest_theorem_l2954_295472

def pumpkin_contest (brad jessica betty carlos emily dave : ℝ) : Prop :=
  brad = 54 ∧
  jessica = brad / 2 ∧
  betty = 4 * jessica ∧
  carlos = 2.5 * (brad + jessica) ∧
  emily = 1.5 * (betty - brad) ∧
  dave = (jessica + betty) / 2 + 20 ∧
  max brad (max jessica (max betty (max carlos (max emily dave)))) -
  min brad (min jessica (min betty (min carlos (min emily dave)))) = 175.5

theorem pumpkin_contest_theorem :
  ∃ brad jessica betty carlos emily dave : ℝ,
    pumpkin_contest brad jessica betty carlos emily dave :=
by
  sorry

end NUMINAMATH_CALUDE_pumpkin_contest_theorem_l2954_295472


namespace NUMINAMATH_CALUDE_parabola_point_distance_l2954_295441

theorem parabola_point_distance (x y : ℝ) : 
  x^2 = 4*y →                             -- P is on the parabola x^2 = 4y
  (x^2 + (y - 1)^2 = 4) →                 -- Distance from P to A(0,1) is 2
  y = 1 :=                                -- Distance from P to x-axis is 1
by sorry

end NUMINAMATH_CALUDE_parabola_point_distance_l2954_295441


namespace NUMINAMATH_CALUDE_total_tiles_count_l2954_295421

def room_length : ℕ := 18
def room_width : ℕ := 15
def border_tile_size : ℕ := 2
def border_width : ℕ := 2
def inner_tile_size : ℕ := 3

def border_tiles : ℕ := 
  2 * (room_length / border_tile_size + room_width / border_tile_size) + 4

def inner_area : ℕ := (room_length - 2 * border_width) * (room_width - 2 * border_width)
def inner_tiles : ℕ := inner_area / (inner_tile_size * inner_tile_size)

theorem total_tiles_count :
  border_tiles + inner_tiles = 45 := by sorry

end NUMINAMATH_CALUDE_total_tiles_count_l2954_295421


namespace NUMINAMATH_CALUDE_tetrahedron_faces_tetrahedron_has_four_faces_l2954_295459

/-- A tetrahedron is a three-dimensional geometric shape with four triangular faces. -/
structure Tetrahedron where
  -- We don't need to define the internal structure for this problem

/-- The number of faces in a tetrahedron is 4. -/
theorem tetrahedron_faces (t : Tetrahedron) : Nat :=
  4

#check tetrahedron_faces

/-- Proof that a tetrahedron has 4 faces. -/
theorem tetrahedron_has_four_faces (t : Tetrahedron) : tetrahedron_faces t = 4 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_faces_tetrahedron_has_four_faces_l2954_295459


namespace NUMINAMATH_CALUDE_total_differential_arcctg_l2954_295418

noncomputable def z (x y : ℝ) : ℝ := Real.arctan (y / x)

theorem total_differential_arcctg (x y dx dy : ℝ) (hx : x = 1) (hy : y = 3) (hdx : dx = 0.01) (hdy : dy = -0.05) :
  let dz := -(y / (x^2 + y^2)) * dx + (x / (x^2 + y^2)) * dy
  dz = -0.008 := by
  sorry

end NUMINAMATH_CALUDE_total_differential_arcctg_l2954_295418


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2954_295476

def geometric_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) / a n = a 2 / a 1

theorem geometric_sequence_sum (a : ℕ → ℝ) (n : ℕ) :
  (∀ k, a k > 0) →
  geometric_sequence a →
  a 2 = 3 →
  a 1 + a 3 = 10 →
  (∃ S_n : ℝ, S_n = (27/2) - (1/2) * 3^(n-3) ∨ S_n = (3^n - 1) / 2) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2954_295476


namespace NUMINAMATH_CALUDE_linear_function_quadrants_l2954_295489

/-- A linear function passing through the first quadrant -/
def passes_through_first_quadrant (k b : ℝ) : Prop :=
  ∃ x y : ℝ, y = k * x + b ∧ x > 0 ∧ y > 0

/-- A linear function passing through the fourth quadrant -/
def passes_through_fourth_quadrant (k b : ℝ) : Prop :=
  ∃ x y : ℝ, y = k * x + b ∧ x > 0 ∧ y < 0

/-- Theorem stating that a linear function y = kx + b with kb < 0 passes through both
    the first and fourth quadrants -/
theorem linear_function_quadrants (k b : ℝ) (h : k * b < 0) :
  passes_through_first_quadrant k b ∧ passes_through_fourth_quadrant k b :=
sorry

end NUMINAMATH_CALUDE_linear_function_quadrants_l2954_295489


namespace NUMINAMATH_CALUDE_coefficient_x3_in_2x_plus_1_power_5_l2954_295442

-- Define the binomial coefficient function
def binomial (n k : ℕ) : ℕ := sorry

-- Define the coefficient of x^3 in the expansion of (2x+1)^5
def coefficient_x3 : ℕ := binomial 5 2 * 2^3

-- Theorem statement
theorem coefficient_x3_in_2x_plus_1_power_5 : coefficient_x3 = 80 := by sorry

end NUMINAMATH_CALUDE_coefficient_x3_in_2x_plus_1_power_5_l2954_295442


namespace NUMINAMATH_CALUDE_all_parameterizations_valid_l2954_295401

/-- The slope of the line -/
def m : ℝ := -3

/-- The y-intercept of the line -/
def b : ℝ := 4

/-- The line equation: y = mx + b -/
def on_line (x y : ℝ) : Prop := y = m * x + b

/-- A parameterization is valid if it satisfies the line equation for all t -/
def valid_parameterization (p : ℝ × ℝ) (v : ℝ × ℝ) : Prop :=
  ∀ t : ℝ, on_line (p.1 + t * v.1) (p.2 + t * v.2)

/-- Theorem: All given parameterizations are valid -/
theorem all_parameterizations_valid :
  valid_parameterization (0, 4) (1, -3) ∧
  valid_parameterization (-2/3, 0) (3, -9) ∧
  valid_parameterization (-4/3, 8) (2, -6) ∧
  valid_parameterization (-2, 10) (1/2, -1) ∧
  valid_parameterization (1, 1) (4, -12) :=
sorry

end NUMINAMATH_CALUDE_all_parameterizations_valid_l2954_295401


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l2954_295402

theorem complex_modulus_problem (z : ℂ) (h : z * (2 - Complex.I) = 3 + Complex.I) :
  Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l2954_295402


namespace NUMINAMATH_CALUDE_book_purchase_equation_l2954_295446

theorem book_purchase_equation (x : ℝ) : x > 0 →
  (∀ y : ℝ, y = x + 8 → y > 0) →
  (15000 : ℝ) / (x + 8) = (12000 : ℝ) / x :=
by
  sorry

end NUMINAMATH_CALUDE_book_purchase_equation_l2954_295446


namespace NUMINAMATH_CALUDE_log_problem_l2954_295483

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_problem (x : ℝ) (h : log 3 (5 * x) = 3) : log x 125 = 3/2 :=
by
  sorry

end NUMINAMATH_CALUDE_log_problem_l2954_295483


namespace NUMINAMATH_CALUDE_hyperbola_intersection_line_l2954_295422

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 / 3 = 1

-- Define the foci
def F₁ : ℝ × ℝ := (-2, 0)
def F₂ : ℝ × ℝ := (2, 0)

-- Define eccentricity
def e : ℝ := 2

-- Define point M
def M : ℝ × ℝ := (1, 3)

-- Define the line l
def line_l (x y : ℝ) : Prop := y = x + 2

-- Theorem statement
theorem hyperbola_intersection_line :
  ∀ A B : ℝ × ℝ,
  hyperbola A.1 A.2 ∧ hyperbola B.1 B.2 →  -- A and B are on the hyperbola
  M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →  -- M is midpoint of AB
  line_l A.1 A.2 ∧ line_l B.1 B.2 →  -- A and B are on line l
  ∀ x y : ℝ, line_l x y ↔ y = x + 2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_intersection_line_l2954_295422


namespace NUMINAMATH_CALUDE_plant_initial_length_l2954_295498

def plant_problem (initial_length : ℝ) : Prop :=
  let daily_growth : ℝ := 0.6875
  let length_day_4 : ℝ := initial_length + 4 * daily_growth
  let length_day_10 : ℝ := initial_length + 10 * daily_growth
  length_day_10 = 1.3 * length_day_4

theorem plant_initial_length : 
  ∃ (initial_length : ℝ), plant_problem initial_length ∧ initial_length = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_plant_initial_length_l2954_295498


namespace NUMINAMATH_CALUDE_seated_students_count_l2954_295457

/-- Given a school meeting with teachers and students, calculate the number of seated students. -/
theorem seated_students_count 
  (total_attendees : ℕ) 
  (seated_teachers : ℕ) 
  (standing_students : ℕ) 
  (h1 : total_attendees = 355) 
  (h2 : seated_teachers = 30) 
  (h3 : standing_students = 25) : 
  total_attendees = seated_teachers + standing_students + 300 :=
by sorry

end NUMINAMATH_CALUDE_seated_students_count_l2954_295457


namespace NUMINAMATH_CALUDE_smallest_2000_digit_product_l2954_295424

/-- The smallest positive integer whose digits have a product of 2000 -/
def N : ℕ := sorry

/-- Predicate to check if a natural number's digits have a product of 2000 -/
def digits_product_2000 (n : ℕ) : Prop := sorry

/-- Function to calculate the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

theorem smallest_2000_digit_product :
  (∀ m : ℕ, m < N → ¬(digits_product_2000 m)) ∧
  digits_product_2000 N ∧
  sum_of_digits N = 25 := by sorry

end NUMINAMATH_CALUDE_smallest_2000_digit_product_l2954_295424


namespace NUMINAMATH_CALUDE_men_in_club_l2954_295499

/-- Proves the number of men in a club given certain conditions -/
theorem men_in_club (total : ℕ) (participants : ℕ) (h1 : total = 30) (h2 : participants = 18) : 
  ∃ (men women : ℕ), men + women = total ∧ men + (women / 3) = participants ∧ men = 12 := by
  sorry

end NUMINAMATH_CALUDE_men_in_club_l2954_295499


namespace NUMINAMATH_CALUDE_problem_2023_l2954_295400

theorem problem_2023 : (2023^2 - 2023) / 2023 = 2022 := by
  sorry

end NUMINAMATH_CALUDE_problem_2023_l2954_295400


namespace NUMINAMATH_CALUDE_divisible_by_35_l2954_295404

theorem divisible_by_35 (n : ℕ) : ∃ k : ℤ, (3 : ℤ)^(6*n) - (2 : ℤ)^(6*n) = 35 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_35_l2954_295404


namespace NUMINAMATH_CALUDE_floor_equation_solution_l2954_295452

theorem floor_equation_solution (x : ℝ) : 
  (⌊⌊3 * x⌋ + 1/2⌋ = ⌊x + 4⌋) ↔ (5/3 ≤ x ∧ x < 7/3) :=
sorry

end NUMINAMATH_CALUDE_floor_equation_solution_l2954_295452


namespace NUMINAMATH_CALUDE_min_value_exponential_sum_l2954_295461

theorem min_value_exponential_sum (x y : ℝ) (h : x + y = 5) :
  3^x + 3^y ≥ 18 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_exponential_sum_l2954_295461


namespace NUMINAMATH_CALUDE_wrapping_paper_area_formula_l2954_295408

/-- The area of square wrapping paper required to wrap a rectangular box -/
def wrapping_paper_area (l w h : ℝ) : ℝ :=
  (l + 4 + 2 * h) ^ 2

/-- Theorem stating the formula for the area of wrapping paper -/
theorem wrapping_paper_area_formula (l w h : ℝ) (hl : l > 0) (hw : w > 0) (hh : h > 0) :
  wrapping_paper_area l w h = l^2 + 8*l + 16 + 4*l*h + 16*h + 4*h^2 := by
  sorry

#check wrapping_paper_area_formula

end NUMINAMATH_CALUDE_wrapping_paper_area_formula_l2954_295408


namespace NUMINAMATH_CALUDE_gas_cost_per_gallon_l2954_295439

/-- Proves that the cost of gas per gallon is $4 given the specified conditions --/
theorem gas_cost_per_gallon 
  (pay_rate : ℝ) 
  (truck_efficiency : ℝ) 
  (profit : ℝ) 
  (trip_distance : ℝ) 
  (h1 : pay_rate = 0.50)
  (h2 : truck_efficiency = 20)
  (h3 : profit = 180)
  (h4 : trip_distance = 600) :
  (trip_distance * pay_rate - profit) / (trip_distance / truck_efficiency) = 4 := by
  sorry

end NUMINAMATH_CALUDE_gas_cost_per_gallon_l2954_295439


namespace NUMINAMATH_CALUDE_complement_of_B_l2954_295493

-- Define the sets A and B
def A (x : ℝ) : Set ℝ := {1, 3, x}
def B (x : ℝ) : Set ℝ := {1, x^2}

-- Define the universal set U
def U (x : ℝ) : Set ℝ := A x ∪ B x

-- State the theorem
theorem complement_of_B (x : ℝ) :
  (B x ∪ (U x \ B x) = A x) →
  ((x = 0 ∧ U x \ B x = {3}) ∨
   (x = Real.sqrt 3 ∧ U x \ B x = {Real.sqrt 3}) ∨
   (x = -Real.sqrt 3 ∧ U x \ B x = {-Real.sqrt 3})) :=
by sorry

end NUMINAMATH_CALUDE_complement_of_B_l2954_295493


namespace NUMINAMATH_CALUDE_triangle_max_area_l2954_295454

theorem triangle_max_area (A B C : ℝ) (a b c : ℝ) :
  a = 2 →
  Real.sin B = Real.sqrt 3 * Real.sin C →
  (∃ (S : ℝ), S = (1/2) * a * c * Real.sin B ∧
    ∀ (S' : ℝ), S' = (1/2) * a * c * Real.sin B → S' ≤ S) →
  (∃ (S_max : ℝ), S_max = Real.sqrt 3 ∧
    ∀ (S : ℝ), S = (1/2) * a * c * Real.sin B → S ≤ S_max) :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_area_l2954_295454


namespace NUMINAMATH_CALUDE_probability_sum_30_l2954_295450

/-- Represents a 20-faced die with specific numbering --/
structure Die :=
  (faces : Finset ℕ)
  (blank_face : Bool)
  (fair : Bool)
  (face_count : faces.card + (if blank_face then 1 else 0) = 20)

/-- Die 1 with faces numbered 1-18 and one blank face --/
def die1 : Die :=
  { faces := Finset.range 19 \ {0},
    blank_face := true,
    fair := true,
    face_count := sorry }

/-- Die 2 with faces numbered 1-9 and 11-20 and one blank face --/
def die2 : Die :=
  { faces := (Finset.range 21 \ {0, 10}),
    blank_face := true,
    fair := true,
    face_count := sorry }

/-- The probability of an event given the number of favorable outcomes and total outcomes --/
def probability (favorable : ℕ) (total : ℕ) : ℚ :=
  favorable / total

/-- The number of ways to roll a sum of 30 with the given dice --/
def favorable_outcomes : ℕ := 8

/-- The total number of possible outcomes when rolling two 20-faced dice --/
def total_outcomes : ℕ := 400

/-- The main theorem: probability of rolling a sum of 30 is 1/50 --/
theorem probability_sum_30 :
  probability favorable_outcomes total_outcomes = 1 / 50 :=
sorry

end NUMINAMATH_CALUDE_probability_sum_30_l2954_295450


namespace NUMINAMATH_CALUDE_music_tool_cost_l2954_295438

/-- Calculates the cost of a music tool given the total spent and costs of other items --/
theorem music_tool_cost (total_spent flute_cost songbook_cost : ℚ) :
  total_spent = 158.35 ∧ flute_cost = 142.46 ∧ songbook_cost = 7 →
  total_spent - (flute_cost + songbook_cost) = 8.89 := by
sorry

end NUMINAMATH_CALUDE_music_tool_cost_l2954_295438


namespace NUMINAMATH_CALUDE_total_savings_after_three_months_l2954_295427

def savings (n : ℕ) : ℕ := 10 + 30 * n

theorem total_savings_after_three_months : 
  savings 0 + savings 1 + savings 2 = 70 := by
  sorry

end NUMINAMATH_CALUDE_total_savings_after_three_months_l2954_295427


namespace NUMINAMATH_CALUDE_problem_statement_l2954_295410

theorem problem_statement (a b : ℕ+) :
  (18 ^ a.val) * (9 ^ (3 * a.val - 1)) = (2 ^ 6) * (3 ^ b.val) →
  a.val = 6 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l2954_295410


namespace NUMINAMATH_CALUDE_cyclic_quadrilateral_inequality_l2954_295414

/-- A cyclic quadrilateral is a quadrilateral whose vertices all lie on a single circle. -/
def CyclicQuadrilateral (A B C D : ℝ × ℝ) : Prop :=
  ∃ (O : ℝ × ℝ) (r : ℝ), 
    dist O A = r ∧ dist O B = r ∧ dist O C = r ∧ dist O D = r

/-- The theorem states that for a cyclic quadrilateral ABCD, 
    the sum of the absolute differences between opposite sides 
    is greater than or equal to twice the absolute difference between the diagonals. -/
theorem cyclic_quadrilateral_inequality 
  (A B C D : ℝ × ℝ) 
  (h : CyclicQuadrilateral A B C D) : 
  |dist A B - dist C D| + |dist A D - dist B C| ≥ 2 * |dist A C - dist B D| :=
sorry

end NUMINAMATH_CALUDE_cyclic_quadrilateral_inequality_l2954_295414


namespace NUMINAMATH_CALUDE_max_gift_sets_l2954_295484

theorem max_gift_sets (total_chocolates total_candies left_chocolates left_candies : ℕ)
  (h1 : total_chocolates = 69)
  (h2 : total_candies = 86)
  (h3 : left_chocolates = 5)
  (h4 : left_candies = 6) :
  Nat.gcd (total_chocolates - left_chocolates) (total_candies - left_candies) = 16 :=
by sorry

end NUMINAMATH_CALUDE_max_gift_sets_l2954_295484


namespace NUMINAMATH_CALUDE_birth_ticket_cost_l2954_295464

/-- The cost of a ticket to Mars at a given time -/
def ticket_cost (years_since_birth : ℕ) : ℚ := sorry

/-- The cost is halved every 10 years -/
axiom cost_halves (y : ℕ) : ticket_cost (y + 10) = ticket_cost y / 2

/-- When Matty is 30, a ticket costs $125,000 -/
axiom cost_at_30 : ticket_cost 30 = 125000

/-- The cost of a ticket to Mars when Matty was born was $1,000,000 -/
theorem birth_ticket_cost : ticket_cost 0 = 1000000 := by sorry

end NUMINAMATH_CALUDE_birth_ticket_cost_l2954_295464

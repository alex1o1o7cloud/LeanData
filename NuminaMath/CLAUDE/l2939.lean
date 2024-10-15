import Mathlib

namespace NUMINAMATH_CALUDE_triangle_properties_l2939_293921

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.A ≠ π / 2)
  (h2 : 3 * Real.sin t.A * Real.cos t.B + (1/2) * t.b * Real.sin (2 * t.A) = 3 * Real.sin t.C) :
  (t.a = 3) ∧ 
  (t.A = 2 * π / 3 → 
    ∃ (max_perimeter : Real), max_perimeter = 3 + 2 * Real.sqrt 3 ∧
    ∀ (perimeter : Real), perimeter = t.a + t.b + t.c → perimeter ≤ max_perimeter) :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l2939_293921


namespace NUMINAMATH_CALUDE_base_number_problem_l2939_293994

theorem base_number_problem (x y a : ℝ) (h1 : x * y = 1) 
  (h2 : a ^ ((x + y)^2) / a ^ ((x - y)^2) = 625) : a = 5 := by
  sorry

end NUMINAMATH_CALUDE_base_number_problem_l2939_293994


namespace NUMINAMATH_CALUDE_max_chord_length_l2939_293975

-- Define the family of curves
def family_of_curves (θ : ℝ) (x y : ℝ) : Prop :=
  2 * (2 * Real.sin θ - Real.cos θ + 3) * x^2 - (8 * Real.sin θ + Real.cos θ + 1) * y = 0

-- Define the line y = 2x
def line (x y : ℝ) : Prop := y = 2 * x

-- Theorem statement
theorem max_chord_length :
  ∃ (max_length : ℝ),
    (∀ θ x₁ y₁ x₂ y₂ : ℝ,
      family_of_curves θ x₁ y₁ ∧
      family_of_curves θ x₂ y₂ ∧
      line x₁ y₁ ∧
      line x₂ y₂ →
      Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) ≤ max_length) ∧
    max_length = 8 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_max_chord_length_l2939_293975


namespace NUMINAMATH_CALUDE_at_least_one_multiple_of_three_l2939_293929

theorem at_least_one_multiple_of_three (a b : ℤ) : 
  (a + b) % 3 = 0 ∨ (a * b) % 3 = 0 ∨ (a - b) % 3 = 0 := by
sorry

end NUMINAMATH_CALUDE_at_least_one_multiple_of_three_l2939_293929


namespace NUMINAMATH_CALUDE_wage_increase_percentage_l2939_293915

theorem wage_increase_percentage (original_wage new_wage : ℝ) 
  (h1 : original_wage = 28)
  (h2 : new_wage = 42) :
  (new_wage - original_wage) / original_wage * 100 = 50 := by
sorry

end NUMINAMATH_CALUDE_wage_increase_percentage_l2939_293915


namespace NUMINAMATH_CALUDE_jason_retirement_age_l2939_293992

def military_career (join_age time_to_chief : ℕ) : Prop :=
  let time_to_senior_chief : ℕ := time_to_chief + (time_to_chief / 4)
  let time_to_master_chief : ℕ := time_to_senior_chief - (time_to_senior_chief / 10)
  let time_to_command_master_chief : ℕ := time_to_master_chief + (time_to_master_chief / 2)
  let additional_time : ℕ := 5
  let total_service_time : ℕ := time_to_chief + time_to_senior_chief + time_to_master_chief + 
                                 time_to_command_master_chief + additional_time
  join_age + total_service_time = 63

theorem jason_retirement_age : 
  military_career 18 8 := by sorry

end NUMINAMATH_CALUDE_jason_retirement_age_l2939_293992


namespace NUMINAMATH_CALUDE_lomonosov_digit_mapping_l2939_293996

theorem lomonosov_digit_mapping :
  ∃ (L O M N S V H C B : ℕ),
    (L < 10) ∧ (O < 10) ∧ (M < 10) ∧ (N < 10) ∧ (S < 10) ∧
    (V < 10) ∧ (H < 10) ∧ (C < 10) ∧ (B < 10) ∧
    (L ≠ O) ∧ (L ≠ M) ∧ (L ≠ N) ∧ (L ≠ S) ∧ (L ≠ V) ∧ (L ≠ H) ∧ (L ≠ C) ∧ (L ≠ B) ∧
    (O ≠ M) ∧ (O ≠ N) ∧ (O ≠ S) ∧ (O ≠ V) ∧ (O ≠ H) ∧ (O ≠ C) ∧ (O ≠ B) ∧
    (M ≠ N) ∧ (M ≠ S) ∧ (M ≠ V) ∧ (M ≠ H) ∧ (M ≠ C) ∧ (M ≠ B) ∧
    (N ≠ S) ∧ (N ≠ V) ∧ (N ≠ H) ∧ (N ≠ C) ∧ (N ≠ B) ∧
    (S ≠ V) ∧ (S ≠ H) ∧ (S ≠ C) ∧ (S ≠ B) ∧
    (V ≠ H) ∧ (V ≠ C) ∧ (V ≠ B) ∧
    (H ≠ C) ∧ (H ≠ B) ∧
    (C ≠ B) ∧
    (L + O / M + O + H + O / C = O * 10 + B) ∧
    (O < M) ∧ (O < C) := by
  sorry

end NUMINAMATH_CALUDE_lomonosov_digit_mapping_l2939_293996


namespace NUMINAMATH_CALUDE_fair_spending_remainder_l2939_293919

/-- Calculates the remaining amount after spending on snacks and games at a fair. -/
theorem fair_spending_remainder (initial_amount snack_cost : ℕ) : 
  initial_amount = 80 →
  snack_cost = 18 →
  initial_amount - (snack_cost + 3 * snack_cost) = 8 := by
  sorry

#check fair_spending_remainder

end NUMINAMATH_CALUDE_fair_spending_remainder_l2939_293919


namespace NUMINAMATH_CALUDE_parabola_equation_l2939_293935

/-- Represents a parabola with specific properties -/
structure Parabola where
  -- Equation coefficients
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  e : ℤ
  f : ℤ
  -- c is positive
  c_pos : c > 0
  -- GCD of absolute values of coefficients is 1
  gcd_one : Nat.gcd (Int.natAbs a) (Nat.gcd (Int.natAbs b) (Nat.gcd (Int.natAbs c) (Nat.gcd (Int.natAbs d) (Nat.gcd (Int.natAbs e) (Int.natAbs f))))) = 1
  -- Passes through (2,6)
  passes_through : a * 2^2 + b * 2 * 6 + c * 6^2 + d * 2 + e * 6 + f = 0
  -- Focus y-coordinate is 2
  focus_y : ∃ (x : ℚ), a * x^2 + b * x * 2 + c * 2^2 + d * x + e * 2 + f = 0
  -- Axis of symmetry parallel to x-axis
  sym_axis_parallel : b = 0
  -- Vertex on y-axis
  vertex_on_y : ∃ (y : ℚ), a * 0^2 + b * 0 * y + c * y^2 + d * 0 + e * y + f = 0

/-- The parabola equation matches the given form -/
theorem parabola_equation (p : Parabola) : p.a = 0 ∧ p.b = 0 ∧ p.c = 1 ∧ p.d = -8 ∧ p.e = -4 ∧ p.f = 4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_l2939_293935


namespace NUMINAMATH_CALUDE_logarithm_simplification_l2939_293923

open Real

theorem logarithm_simplification (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (ha1 : a ≠ 1) :
  let log_a := fun x => log x / log a
  let log_ab := fun x => log x / log (a * b)
  (log_a b + log_a (b^(1/(2*log b / log (a^2)))))/(log_a b - log_ab b) *
  (log_ab b * log_a b)/(b^(2*log b * log_a b) - 1) = 1 / (log_a b - 1) := by
  sorry

end NUMINAMATH_CALUDE_logarithm_simplification_l2939_293923


namespace NUMINAMATH_CALUDE_dot_product_OA_OB_line_l_equations_l2939_293905

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define point M
def M : ℝ × ℝ := (6, 0)

-- Define line l passing through M and intersecting the parabola
def line_l (m : ℝ) (x y : ℝ) : Prop := x = m*y + 6

-- Define points A and B as intersections of line l and the parabola
def intersect_points (m : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) := sorry

-- Theorem for the dot product of OA and OB
theorem dot_product_OA_OB (m : ℝ) :
  let ((x₁, y₁), (x₂, y₂)) := intersect_points m
  (x₁ * x₂ + y₁ * y₂ : ℝ) = 12 := sorry

-- Theorem for the equations of line l given the area of triangle OAB
theorem line_l_equations :
  (∃ m : ℝ, let ((x₁, y₁), (x₂, y₂)) := intersect_points m
   (1/2 : ℝ) * 6 * |y₁ - y₂| = 12 * Real.sqrt 10) →
  (∃ l₁ l₂ : ℝ → ℝ → Prop,
    (∀ x y, l₁ x y ↔ x + 2*y - 6 = 0) ∧
    (∀ x y, l₂ x y ↔ x - 2*y - 6 = 0) ∧
    (∀ x y, line_l 2 x y ↔ l₁ x y) ∧
    (∀ x y, line_l (-2) x y ↔ l₂ x y)) := sorry

end NUMINAMATH_CALUDE_dot_product_OA_OB_line_l_equations_l2939_293905


namespace NUMINAMATH_CALUDE_decreasing_function_positive_l2939_293933

/-- A function f is decreasing on ℝ if for all x₁ < x₂, f(x₁) > f(x₂) -/
def DecreasingOn (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ → f x₁ > f x₂

/-- The condition that f'(x) satisfies f(x) / f''(x) < 1 - x -/
def DerivativeCondition (f : ℝ → ℝ) : Prop :=
  ∀ x, DifferentiableAt ℝ f x ∧ DifferentiableAt ℝ (deriv f) x ∧
    f x / (deriv (deriv f) x) < 1 - x

theorem decreasing_function_positive
  (f : ℝ → ℝ)
  (h_decreasing : DecreasingOn f)
  (h_condition : DerivativeCondition f) :
  ∀ x, f x > 0 := by
  sorry

end NUMINAMATH_CALUDE_decreasing_function_positive_l2939_293933


namespace NUMINAMATH_CALUDE_set_operations_and_subset_l2939_293903

-- Define the sets A, B, and C
def A : Set ℝ := {x | 3 ≤ x ∧ x < 6}
def B : Set ℝ := {x | 2 < x ∧ x < 9}
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 1}

-- State the theorem
theorem set_operations_and_subset :
  (A ∩ B = {x | 3 ≤ x ∧ x < 6}) ∧
  (A ∪ B = {x | 2 < x ∧ x < 9}) ∧
  ({a : ℝ | C a ⊆ B} = {a | 2 ≤ a ∧ a ≤ 8}) :=
by sorry

end NUMINAMATH_CALUDE_set_operations_and_subset_l2939_293903


namespace NUMINAMATH_CALUDE_least_n_satisfying_inequality_l2939_293976

theorem least_n_satisfying_inequality : 
  ∃ (n : ℕ), n > 0 ∧ (∀ (k : ℕ), k > 0 → (1 : ℚ) / k - (1 : ℚ) / (k + 1) < (1 : ℚ) / 15 → k ≥ n) ∧ 
  ((1 : ℚ) / n - (1 : ℚ) / (n + 1) < (1 : ℚ) / 15) ∧ n = 4 :=
sorry

end NUMINAMATH_CALUDE_least_n_satisfying_inequality_l2939_293976


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l2939_293902

theorem other_root_of_quadratic (k : ℝ) : 
  (∃ x : ℝ, 3 * x^2 + k * x = -6) ∧ (3 * 3^2 + k * 3 = -6) → 
  (∃ r : ℝ, r ≠ 3 ∧ 3 * r^2 + k * r = -6 ∧ r = 2/3) :=
by sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l2939_293902


namespace NUMINAMATH_CALUDE_triangular_array_coins_l2939_293901

theorem triangular_array_coins (N : ℕ) : 
  (N * (N + 1)) / 2 = 2010 → N = 63 ∧ (N / 10 + N % 10 = 9) := by
  sorry

end NUMINAMATH_CALUDE_triangular_array_coins_l2939_293901


namespace NUMINAMATH_CALUDE_complex_in_second_quadrant_l2939_293900

theorem complex_in_second_quadrant :
  let z : ℂ := Complex.mk (Real.cos 2) (Real.sin 3)
  z.re < 0 ∧ z.im > 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_in_second_quadrant_l2939_293900


namespace NUMINAMATH_CALUDE_quadrilateral_angle_measure_l2939_293962

theorem quadrilateral_angle_measure :
  ∀ (a b c d : ℝ),
  a = 50 →
  b = 180 - 30 →
  d = 180 - 40 →
  a + b + c + d = 360 →
  c = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_angle_measure_l2939_293962


namespace NUMINAMATH_CALUDE_arrangement_theorem_l2939_293999

/-- The number of permutations of n elements taken r at a time -/
def permutations (n : ℕ) (r : ℕ) : ℕ := sorry

/-- The number of ways to arrange 8 people in a row with 3 specific people not adjacent -/
def arrangements_with_non_adjacent (total : ℕ) (non_adjacent : ℕ) : ℕ :=
  permutations (total - non_adjacent + 1) non_adjacent * permutations (total - non_adjacent) (total - non_adjacent)

theorem arrangement_theorem :
  arrangements_with_non_adjacent 8 3 = permutations 6 3 * permutations 5 5 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_theorem_l2939_293999


namespace NUMINAMATH_CALUDE_divisibility_by_ten_l2939_293988

theorem divisibility_by_ten (a : ℤ) : 
  (10 ∣ (a^10 + 1)) ↔ (a % 10 = 3 ∨ a % 10 = 7 ∨ a % 10 = -3 ∨ a % 10 = -7) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_ten_l2939_293988


namespace NUMINAMATH_CALUDE_intersection_M_N_l2939_293981

-- Define set M
def M : Set ℤ := {x | x^2 - x ≤ 0}

-- Define set N
def N : Set ℤ := {x | ∃ n : ℕ, x = 2 * n}

-- Theorem statement
theorem intersection_M_N : M ∩ N = {0} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2939_293981


namespace NUMINAMATH_CALUDE_power_function_through_point_l2939_293967

theorem power_function_through_point (a : ℝ) :
  (2 : ℝ) ^ a = Real.sqrt 2 → a = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_power_function_through_point_l2939_293967


namespace NUMINAMATH_CALUDE_ten_person_handshake_count_l2939_293934

/-- Represents a group of people with distinct heights -/
structure HeightGroup where
  n : ℕ
  heights : Fin n → ℕ
  distinct_heights : ∀ i j, i ≠ j → heights i ≠ heights j

/-- The number of handshakes in a height group -/
def handshake_count (group : HeightGroup) : ℕ :=
  (group.n * (group.n - 1)) / 2

/-- Theorem: In a group of 10 people with distinct heights, where each person
    only shakes hands with those taller than themselves, the total number of
    handshakes is 45. -/
theorem ten_person_handshake_count :
  ∀ (group : HeightGroup), group.n = 10 → handshake_count group = 45 := by
  sorry

end NUMINAMATH_CALUDE_ten_person_handshake_count_l2939_293934


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l2939_293952

/-- A geometric sequence {a_n} -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

/-- The main theorem -/
theorem geometric_sequence_ratio
  (a : ℕ → ℝ)
  (h_geo : geometric_sequence a)
  (h_prod : a 7 * a 11 = 6)
  (h_sum : a 4 + a 14 = 5) :
  a 20 / a 10 = 3/2 ∨ a 20 / a 10 = 2/3 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l2939_293952


namespace NUMINAMATH_CALUDE_polygon_diagonals_sides_l2939_293989

theorem polygon_diagonals_sides (n : ℕ) (h : n > 2) : 
  n * (n - 3) / 2 = 2 * n → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_polygon_diagonals_sides_l2939_293989


namespace NUMINAMATH_CALUDE_twentieth_century_power_diff_l2939_293965

def is_20th_century (year : ℕ) : Prop := 1900 ≤ year ∧ year ≤ 1999

def is_power_diff (year : ℕ) : Prop :=
  ∃ (n k : ℕ), year = 2^n - 2^k

theorem twentieth_century_power_diff :
  {year : ℕ | is_20th_century year ∧ is_power_diff year} = {1984, 1920} := by
  sorry

end NUMINAMATH_CALUDE_twentieth_century_power_diff_l2939_293965


namespace NUMINAMATH_CALUDE_rug_inner_length_is_four_l2939_293969

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.length * r.width

/-- Represents the rug with three colored regions -/
structure Rug where
  inner : Rectangle
  middle : Rectangle
  outer : Rectangle

/-- Checks if three real numbers form an arithmetic progression -/
def isArithmeticProgression (a b c : ℝ) : Prop :=
  b - a = c - b

theorem rug_inner_length_is_four (rug : Rug) : 
  rug.inner.width = 2 ∧ 
  rug.middle.length = rug.inner.length + 4 ∧ 
  rug.middle.width = rug.inner.width + 4 ∧
  rug.outer.length = rug.middle.length + 4 ∧
  rug.outer.width = rug.middle.width + 4 ∧
  isArithmeticProgression (area rug.inner) (area rug.middle - area rug.inner) (area rug.outer - area rug.middle) →
  rug.inner.length = 4 := by
sorry

end NUMINAMATH_CALUDE_rug_inner_length_is_four_l2939_293969


namespace NUMINAMATH_CALUDE_k_value_when_A_is_quadratic_binomial_C_value_when_k_is_negative_one_l2939_293932

-- Define the polynomials A and B
def A (k : ℝ) (x : ℝ) : ℝ := -2 * x^2 - (k - 1) * x + 1
def B (x : ℝ) : ℝ := -2 * (x^2 - x + 2)

-- Define what it means for a polynomial to be a quadratic binomial
def is_quadratic_binomial (p : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ ∀ x, p x = a * x^2 + b * x + c

-- Theorem 1: When A is a quadratic binomial, k = 1
theorem k_value_when_A_is_quadratic_binomial :
  ∀ k : ℝ, is_quadratic_binomial (A k) → k = 1 :=
sorry

-- Theorem 2: When k = -1 and C + 2A = B, then C = 2x^2 - 2x - 6
theorem C_value_when_k_is_negative_one :
  ∀ C : ℝ → ℝ, (∀ x, C x + 2 * A (-1) x = B x) →
  ∀ x, C x = 2 * x^2 - 2 * x - 6 :=
sorry

end NUMINAMATH_CALUDE_k_value_when_A_is_quadratic_binomial_C_value_when_k_is_negative_one_l2939_293932


namespace NUMINAMATH_CALUDE_subtraction_and_simplification_l2939_293998

theorem subtraction_and_simplification :
  (12 : ℚ) / 25 - (3 : ℚ) / 75 = (11 : ℚ) / 25 := by sorry

end NUMINAMATH_CALUDE_subtraction_and_simplification_l2939_293998


namespace NUMINAMATH_CALUDE_shale_mix_cost_per_pound_l2939_293927

/-- Prove that the cost of the shale mix per pound is $5 -/
theorem shale_mix_cost_per_pound
  (limestone_cost : ℝ)
  (total_weight : ℝ)
  (total_cost_per_pound : ℝ)
  (limestone_weight : ℝ)
  (h1 : limestone_cost = 3)
  (h2 : total_weight = 100)
  (h3 : total_cost_per_pound = 4.25)
  (h4 : limestone_weight = 37.5) :
  let shale_weight := total_weight - limestone_weight
  let total_cost := total_weight * total_cost_per_pound
  let limestone_total_cost := limestone_weight * limestone_cost
  let shale_total_cost := total_cost - limestone_total_cost
  shale_total_cost / shale_weight = 5 := by
sorry

end NUMINAMATH_CALUDE_shale_mix_cost_per_pound_l2939_293927


namespace NUMINAMATH_CALUDE_unique_intersection_l2939_293904

/-- The first function f(x) = x^2 - 7x + 3 -/
def f (x : ℝ) : ℝ := x^2 - 7*x + 3

/-- The second function g(x) = -3x^2 + 5x - 6 -/
def g (x : ℝ) : ℝ := -3*x^2 + 5*x - 6

/-- The theorem stating that f and g intersect at exactly one point (3/2, -21/4) -/
theorem unique_intersection :
  ∃! p : ℝ × ℝ, 
    p.1 = 3/2 ∧ 
    p.2 = -21/4 ∧ 
    f p.1 = g p.1 ∧
    ∀ x : ℝ, f x = g x → x = p.1 := by
  sorry

#check unique_intersection

end NUMINAMATH_CALUDE_unique_intersection_l2939_293904


namespace NUMINAMATH_CALUDE_play_role_assignment_l2939_293957

def number_of_assignments (men women : ℕ) (male_roles female_roles either_roles : ℕ) : ℕ :=
  men * women * (Nat.choose (men + women - male_roles - female_roles) either_roles)

theorem play_role_assignment :
  number_of_assignments 4 7 1 1 4 = 3528 := by
  sorry

end NUMINAMATH_CALUDE_play_role_assignment_l2939_293957


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l2939_293966

theorem partial_fraction_decomposition (x : ℝ) (A B C : ℝ) :
  (1 : ℝ) / (x^3 - 7*x^2 + 10*x + 24) = A / (x - 2) + B / (x - 6) + C / (x - 6)^2 →
  x^3 - 7*x^2 + 10*x + 24 = (x - 2) * (x - 6)^2 →
  A = 1/16 := by sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l2939_293966


namespace NUMINAMATH_CALUDE_total_distance_is_200_l2939_293997

/-- Represents the cycling journey of Jack and Peter -/
structure CyclingJourney where
  speed : ℝ
  timeHomeToStore : ℝ
  timeStoreToPeter : ℝ
  distanceStoreToPeter : ℝ

/-- Calculates the total distance cycled by Jack and Peter -/
def totalDistanceCycled (journey : CyclingJourney) : ℝ :=
  let distanceHomeToStore := journey.speed * journey.timeHomeToStore
  let distanceStoreToPeter := journey.distanceStoreToPeter
  distanceHomeToStore + 2 * distanceStoreToPeter

/-- Theorem stating the total distance cycled is 200 miles -/
theorem total_distance_is_200 (journey : CyclingJourney) 
  (h1 : journey.timeHomeToStore = 2 * journey.timeStoreToPeter)
  (h2 : journey.speed > 0)
  (h3 : journey.distanceStoreToPeter = 50) :
  totalDistanceCycled journey = 200 := by
  sorry


end NUMINAMATH_CALUDE_total_distance_is_200_l2939_293997


namespace NUMINAMATH_CALUDE_eddie_dump_rate_l2939_293993

/-- Given that Sam dumps tea for 6 hours at 60 crates per hour,
    and Eddie takes 4 hours to dump the same amount,
    prove that Eddie's rate is 90 crates per hour. -/
theorem eddie_dump_rate 
  (sam_hours : ℕ) 
  (sam_rate : ℕ) 
  (eddie_hours : ℕ) 
  (h1 : sam_hours = 6)
  (h2 : sam_rate = 60)
  (h3 : eddie_hours = 4)
  (h4 : sam_hours * sam_rate = eddie_hours * eddie_rate) :
  eddie_rate = 90 :=
by
  sorry

#check eddie_dump_rate

end NUMINAMATH_CALUDE_eddie_dump_rate_l2939_293993


namespace NUMINAMATH_CALUDE_least_positive_integer_multiple_of_53_l2939_293931

theorem least_positive_integer_multiple_of_53 : 
  ∃ (x : ℕ+), (x = 4) ∧ 
  (∀ (y : ℕ+), y < x → ¬(53 ∣ (3*y)^2 + 2*41*3*y + 41^2)) ∧ 
  (53 ∣ (3*x)^2 + 2*41*3*x + 41^2) := by
sorry

end NUMINAMATH_CALUDE_least_positive_integer_multiple_of_53_l2939_293931


namespace NUMINAMATH_CALUDE_rectangle_area_modification_l2939_293974

/-- Given a rectangle with initial dimensions 5 × 7 inches, if shortening one side by 2 inches
    results in an area of 21 square inches, then doubling the length of the other side
    will result in an area of 70 square inches. -/
theorem rectangle_area_modification (length width : ℝ) : 
  length = 5 ∧ width = 7 ∧ 
  ((length - 2) * width = 21 ∨ length * (width - 2) = 21) →
  length * (2 * width) = 70 :=
sorry

end NUMINAMATH_CALUDE_rectangle_area_modification_l2939_293974


namespace NUMINAMATH_CALUDE_quadratic_equation_equivalence_l2939_293943

theorem quadratic_equation_equivalence (x : ℝ) :
  let k : ℝ := 0.32653061224489793
  (2 * k * x^2 + 7 * k * x + 2 = 0) ↔ 
  (0.65306122448979586 * x^2 + 2.2857142857142865 * x + 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_equivalence_l2939_293943


namespace NUMINAMATH_CALUDE_characterization_of_n_l2939_293928

def invalid_n : Set ℕ := {2, 3, 5, 6, 7, 8, 13, 14, 15, 17, 19, 21, 23, 26, 27, 30, 47, 51, 53, 55, 61}

def satisfies_condition (n : ℕ) : Prop :=
  ∃ (m : ℕ) (a : Fin (m-1) → ℕ), 
    (∀ i : Fin (m-1), 1 ≤ a i ∧ a i ≤ m - 1) ∧
    (∀ i j : Fin (m-1), i ≠ j → a i ≠ a j) ∧
    n = (Finset.univ.sum fun i => a i * (m - a i))

theorem characterization_of_n (n : ℕ) :
  n > 0 → (satisfies_condition n ↔ n ∉ invalid_n) := by sorry

end NUMINAMATH_CALUDE_characterization_of_n_l2939_293928


namespace NUMINAMATH_CALUDE_smallest_n_with_partial_divisibility_l2939_293941

theorem smallest_n_with_partial_divisibility : ∃ (n : ℕ), n > 0 ∧
  (∀ (m : ℕ), m > 0 → m < n →
    (∃ (k : ℕ), k > 0 ∧ k ≤ m ∧ (m * (m + 1)) % k = 0) ∧
    (∀ (k : ℕ), k > 0 ∧ k ≤ m → (m * (m + 1)) % k = 0)) ∧
  (∃ (k : ℕ), k > 0 ∧ k ≤ n ∧ (n * (n + 1)) % k = 0) ∧
  (∃ (k : ℕ), k > 0 ∧ k ≤ n ∧ (n * (n + 1)) % k ≠ 0) ∧
  n = 4 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_with_partial_divisibility_l2939_293941


namespace NUMINAMATH_CALUDE_bouncing_ball_height_l2939_293912

/-- Represents the height of a bouncing ball -/
def BouncingBall (h : ℝ) : Prop :=
  -- The ball rebounds to 50% of its previous height
  let h₁ := h / 2
  let h₂ := h₁ / 2
  -- The total travel distance when it touches the floor for the third time is 200 cm
  h + 2 * h₁ + 2 * h₂ = 200

/-- Theorem stating that the original height of the ball is 80 cm -/
theorem bouncing_ball_height :
  ∃ h : ℝ, BouncingBall h ∧ h = 80 :=
sorry

end NUMINAMATH_CALUDE_bouncing_ball_height_l2939_293912


namespace NUMINAMATH_CALUDE_length_of_ae_l2939_293972

/-- Given 5 consecutive points on a straight line, prove the length of ae -/
theorem length_of_ae (a b c d e : ℝ) : 
  (b - a) = 5 →
  (c - a) = 11 →
  (c - b) = 2 * (d - c) →
  (e - d) = 4 →
  (e - a) = 18 := by
  sorry

end NUMINAMATH_CALUDE_length_of_ae_l2939_293972


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l2939_293906

theorem inequality_and_equality_condition (a b c : ℝ) (h : a * b * c = 1 / 8) :
  a^2 + b^2 + c^2 + a^2 * b^2 + b^2 * c^2 + c^2 * a^2 ≥ 15 / 16 ∧
  (a^2 + b^2 + c^2 + a^2 * b^2 + b^2 * c^2 + c^2 * a^2 = 15 / 16 ↔ a = 1 / 2 ∧ b = 1 / 2 ∧ c = 1 / 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l2939_293906


namespace NUMINAMATH_CALUDE_square_border_pieces_l2939_293995

/-- The number of pieces on one side of the square arrangement -/
def side_length : ℕ := 12

/-- The total number of pieces in the border of a square arrangement -/
def border_pieces (n : ℕ) : ℕ := 2 * n + 2 * (n - 2)

/-- Theorem stating that in a 12x12 square arrangement, there are 44 pieces in the border -/
theorem square_border_pieces :
  border_pieces side_length = 44 := by
  sorry

#eval border_pieces side_length

end NUMINAMATH_CALUDE_square_border_pieces_l2939_293995


namespace NUMINAMATH_CALUDE_parallel_vectors_tan_theta_l2939_293922

theorem parallel_vectors_tan_theta (θ : Real) 
  (h_acute : 0 < θ ∧ θ < π / 2)
  (a : Fin 2 → Real)
  (b : Fin 2 → Real)
  (h_a : a = ![1 - Real.sin θ, 1])
  (h_b : b = ![1/2, 1 + Real.sin θ])
  (h_parallel : ∃ (k : Real), a = k • b) :
  Real.tan θ = 1 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_tan_theta_l2939_293922


namespace NUMINAMATH_CALUDE_system_of_equations_solutions_l2939_293925

theorem system_of_equations_solutions :
  (∃ x y : ℝ, y = 2*x - 3 ∧ 2*x + y = 5 ∧ x = 2 ∧ y = 1) ∧
  (∃ x y : ℝ, 3*x + 4*y = 5 ∧ 5*x - 2*y = 17 ∧ x = 3 ∧ y = -1) := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solutions_l2939_293925


namespace NUMINAMATH_CALUDE_intersection_max_k_l2939_293936

theorem intersection_max_k : 
  let f : ℝ → ℝ := fun x => Real.log x / x
  ∃ k_max : ℝ, k_max = 1 / Real.exp 1 ∧ 
    (∀ k : ℝ, (∃ x : ℝ, x > 0 ∧ k * x = Real.log x) → k ≤ k_max) :=
by sorry

end NUMINAMATH_CALUDE_intersection_max_k_l2939_293936


namespace NUMINAMATH_CALUDE_households_without_car_or_bike_l2939_293948

/-- Prove that the number of households without either a car or a bike is 11 -/
theorem households_without_car_or_bike (total : ℕ) (car_and_bike : ℕ) (car : ℕ) (bike_only : ℕ)
  (h_total : total = 90)
  (h_car_and_bike : car_and_bike = 18)
  (h_car : car = 44)
  (h_bike_only : bike_only = 35) :
  total - (car + bike_only) = 11 := by
  sorry

end NUMINAMATH_CALUDE_households_without_car_or_bike_l2939_293948


namespace NUMINAMATH_CALUDE_max_consecutive_integers_sum_45_l2939_293982

theorem max_consecutive_integers_sum_45 (n : ℕ) 
  (h : ∃ a : ℤ, (Finset.range n).sum (λ i => a + i) = 45) : n ≤ 90 := by
  sorry

end NUMINAMATH_CALUDE_max_consecutive_integers_sum_45_l2939_293982


namespace NUMINAMATH_CALUDE_candidate_vote_percentage_l2939_293950

theorem candidate_vote_percentage
  (total_votes : ℕ)
  (invalid_percentage : ℚ)
  (candidate_valid_votes : ℕ)
  (h1 : total_votes = 560000)
  (h2 : invalid_percentage = 15 / 100)
  (h3 : candidate_valid_votes = 309400) :
  (candidate_valid_votes : ℚ) / ((1 - invalid_percentage) * total_votes) * 100 = 65 := by
  sorry

end NUMINAMATH_CALUDE_candidate_vote_percentage_l2939_293950


namespace NUMINAMATH_CALUDE_inequality_always_true_l2939_293908

theorem inequality_always_true (a : ℝ) : 
  (∀ x : ℝ, 2 * x^2 - a * x + 1 > 0) ↔ (-2 * Real.sqrt 2 < a ∧ a < 2 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_always_true_l2939_293908


namespace NUMINAMATH_CALUDE_extreme_value_condition_l2939_293973

/-- A function f has an extreme value at x₀ -/
def has_extreme_value (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∃ ε > 0, ∀ x, |x - x₀| < ε → f x₀ ≤ f x ∨ f x ≤ f x₀

theorem extreme_value_condition (f : ℝ → ℝ) (hf : Differentiable ℝ f) :
  ¬(∀ x₀ : ℝ, has_extreme_value f x₀ ↔ f x₀ = 0) :=
by sorry

end NUMINAMATH_CALUDE_extreme_value_condition_l2939_293973


namespace NUMINAMATH_CALUDE_prob_two_twos_in_five_rolls_l2939_293907

/-- The probability of rolling a specific number on a fair six-sided die -/
def prob_single_roll : ℚ := 1 / 6

/-- The number of rolls -/
def num_rolls : ℕ := 5

/-- The number of times we want to roll the specific number -/
def target_rolls : ℕ := 2

/-- The probability of rolling a specific number exactly k times in n rolls of a fair six-sided die -/
def prob_specific_rolls (n k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) * (prob_single_roll ^ k) * ((1 - prob_single_roll) ^ (n - k))

theorem prob_two_twos_in_five_rolls :
  prob_specific_rolls num_rolls target_rolls = 625 / 3888 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_twos_in_five_rolls_l2939_293907


namespace NUMINAMATH_CALUDE_combined_new_wattage_l2939_293946

def original_wattages : List ℝ := [60, 80, 100, 120]

def increase_percentage : ℝ := 0.25

def increased_wattage (w : ℝ) : ℝ := w * (1 + increase_percentage)

theorem combined_new_wattage :
  (original_wattages.map increased_wattage).sum = 450 := by
  sorry

end NUMINAMATH_CALUDE_combined_new_wattage_l2939_293946


namespace NUMINAMATH_CALUDE_tom_average_increase_l2939_293953

def tom_scores : List ℝ := [92, 89, 91, 93]

theorem tom_average_increase :
  let first_three := tom_scores.take 3
  let all_four := tom_scores
  let avg_first_three := first_three.sum / first_three.length
  let avg_all_four := all_four.sum / all_four.length
  avg_all_four - avg_first_three = 0.58 := by
  sorry

end NUMINAMATH_CALUDE_tom_average_increase_l2939_293953


namespace NUMINAMATH_CALUDE_probability_not_adjacent_seats_l2939_293930

-- Define the number of seats
def num_seats : ℕ := 10

-- Define the function to calculate the number of ways to choose k items from n items
def choose (n k : ℕ) : ℕ := (n.factorial) / (k.factorial * (n - k).factorial)

-- Define the number of ways two people can sit next to each other in a row of seats
def adjacent_seats (n : ℕ) : ℕ := n - 1

-- Theorem statement
theorem probability_not_adjacent_seats :
  let total_ways := choose num_seats 2
  let adjacent_ways := adjacent_seats num_seats
  (total_ways - adjacent_ways) / total_ways = 4 / 5 := by sorry

end NUMINAMATH_CALUDE_probability_not_adjacent_seats_l2939_293930


namespace NUMINAMATH_CALUDE_hyperbola_line_intersection_l2939_293956

/-- Hyperbola C: x²/a² - y²/b² = 1 -/
def Hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

/-- Line l: y = kx + m -/
def Line (k m : ℝ) (x y : ℝ) : Prop :=
  y = k * x + m

/-- Point on a line -/
def PointOnLine (k m x y : ℝ) : Prop :=
  Line k m x y

/-- Point on the hyperbola -/
def PointOnHyperbola (a b x y : ℝ) : Prop :=
  Hyperbola a b x y

/-- Midpoint of two points -/
def Midpoint (x1 y1 x2 y2 xm ym : ℝ) : Prop :=
  xm = (x1 + x2) / 2 ∧ ym = (y1 + y2) / 2

/-- kAB · kOM = 3/4 condition -/
def SlopeProduct (xa ya xb yb xm ym : ℝ) : Prop :=
  ((yb - ya) / (xb - xa)) * (ym / xm) = 3/4

/-- Circle passing through three points -/
def CircleThroughPoints (x1 y1 x2 y2 x3 y3 : ℝ) : Prop :=
  ∃ (xc yc r : ℝ), (x1 - xc)^2 + (y1 - yc)^2 = r^2 ∧
                   (x2 - xc)^2 + (y2 - yc)^2 = r^2 ∧
                   (x3 - xc)^2 + (y3 - yc)^2 = r^2

theorem hyperbola_line_intersection
  (a b : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : a^2 / b^2 = 4/3)  -- Derived from eccentricity √7/2
  (k m : ℝ)
  (xa ya xb yb : ℝ)
  (h4 : PointOnHyperbola a b xa ya)
  (h5 : PointOnHyperbola a b xb yb)
  (h6 : PointOnLine k m xa ya)
  (h7 : PointOnLine k m xb yb)
  (xm ym : ℝ)
  (h8 : Midpoint xa ya xb yb xm ym)
  (h9 : SlopeProduct xa ya xb yb xm ym)
  (h10 : ¬(PointOnLine k m 2 0))
  (h11 : CircleThroughPoints xa ya xb yb 2 0) :
  PointOnLine k m 14 0 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_line_intersection_l2939_293956


namespace NUMINAMATH_CALUDE_profit_percentage_is_fifty_percent_l2939_293985

def purchase_price : ℕ := 14000
def repair_cost : ℕ := 5000
def transportation_charges : ℕ := 1000
def selling_price : ℕ := 30000

def total_cost : ℕ := purchase_price + repair_cost + transportation_charges
def profit : ℕ := selling_price - total_cost

theorem profit_percentage_is_fifty_percent :
  (profit : ℚ) / (total_cost : ℚ) * 100 = 50 := by sorry

end NUMINAMATH_CALUDE_profit_percentage_is_fifty_percent_l2939_293985


namespace NUMINAMATH_CALUDE_cube_paint_theorem_l2939_293990

/-- 
Given a cube with side length n, prove that if exactly one-third of the total number of faces 
of the n³ unit cubes (after cutting) are blue, then n = 3.
-/
theorem cube_paint_theorem (n : ℕ) (h : n > 0) : 
  (6 * n^2 : ℚ) / (6 * n^3 : ℚ) = 1/3 → n = 3 := by
  sorry

#check cube_paint_theorem

end NUMINAMATH_CALUDE_cube_paint_theorem_l2939_293990


namespace NUMINAMATH_CALUDE_coupon_difference_l2939_293942

/-- Represents the savings from a coupon given a price -/
def coupon_savings (price : ℝ) : (ℝ → ℝ) → ℝ := fun f => f price

/-- Coupon A: 20% off the listed price -/
def coupon_a (price : ℝ) : ℝ := 0.2 * price

/-- Coupon B: flat $40 off -/
def coupon_b (_ : ℝ) : ℝ := 40

/-- Coupon C: 30% off the amount by which the listed price exceeds $120 plus an additional $20 -/
def coupon_c (price : ℝ) : ℝ := 0.3 * (price - 120) + 20

/-- The proposition that Coupon A is at least as good as Coupon B or C for a given price -/
def coupon_a_best (price : ℝ) : Prop :=
  coupon_savings price coupon_a ≥ max (coupon_savings price coupon_b) (coupon_savings price coupon_c)

theorem coupon_difference :
  ∃ (x y : ℝ),
    x > 120 ∧
    y > 120 ∧
    x ≤ y ∧
    coupon_a_best x ∧
    coupon_a_best y ∧
    (∀ p, x < p ∧ p < y → coupon_a_best p) ∧
    (∀ p, p < x → ¬coupon_a_best p) ∧
    (∀ p, p > y → ¬coupon_a_best p) ∧
    y - x = 100 :=
  sorry

end NUMINAMATH_CALUDE_coupon_difference_l2939_293942


namespace NUMINAMATH_CALUDE_valid_rearrangements_count_valid_rearrangements_count_is_360_l2939_293940

/-- Represents the word to be rearranged -/
def word : String := "REPRESENT"

/-- Counts the occurrences of a character in a string -/
def count_char (s : String) (c : Char) : Nat :=
  s.toList.filter (· = c) |>.length

/-- The number of vowels in the word -/
def num_vowels : Nat :=
  count_char word 'E'

/-- The number of consonants in the word -/
def num_consonants : Nat :=
  word.length - num_vowels

/-- The number of unique consonants in the word -/
def num_unique_consonants : Nat :=
  (word.toList.filter (λ c => c ≠ 'E') |>.eraseDups).length

/-- The main theorem stating the number of valid rearrangements -/
theorem valid_rearrangements_count : Nat :=
  (Nat.factorial num_consonants) / (Nat.factorial (num_consonants - num_unique_consonants + 1))

/-- The proof of the main theorem -/
theorem valid_rearrangements_count_is_360 : valid_rearrangements_count = 360 := by
  sorry

end NUMINAMATH_CALUDE_valid_rearrangements_count_valid_rearrangements_count_is_360_l2939_293940


namespace NUMINAMATH_CALUDE_watch_price_proof_l2939_293959

/-- The sticker price of the watch in dollars -/
def stickerPrice : ℝ := 250

/-- The price at store X after discounts -/
def priceX (price : ℝ) : ℝ := 0.8 * price - 50

/-- The price at store Y after discount -/
def priceY (price : ℝ) : ℝ := 0.9 * price

theorem watch_price_proof :
  priceY stickerPrice - priceX stickerPrice = 25 :=
sorry

end NUMINAMATH_CALUDE_watch_price_proof_l2939_293959


namespace NUMINAMATH_CALUDE_parabola_sum_l2939_293963

/-- Parabola in the first quadrant -/
def parabola (x y : ℝ) : Prop := x^2 = (1/2) * y ∧ x > 0 ∧ y > 0

/-- Point on the parabola -/
def point_on_parabola (a : ℕ → ℝ) (i : ℕ) : Prop :=
  parabola (a i) (2 * (a i)^2)

/-- Tangent line intersection property -/
def tangent_intersection (a : ℕ → ℝ) : Prop :=
  ∀ i : ℕ, i > 0 → point_on_parabola a i →
    ∃ m b : ℝ, (m * (a (i+1)) + b = 0) ∧
              (∀ x y : ℝ, y - 2*(a i)^2 = m*(x - a i) → parabola x y)

/-- The main theorem -/
theorem parabola_sum (a : ℕ → ℝ) :
  (∀ i : ℕ, i > 0 → point_on_parabola a i) →
  tangent_intersection a →
  a 2 = 32 →
  a 2 + a 4 + a 6 = 42 := by sorry

end NUMINAMATH_CALUDE_parabola_sum_l2939_293963


namespace NUMINAMATH_CALUDE_checkerboard_square_selection_l2939_293961

theorem checkerboard_square_selection (b : ℕ) : 
  let n := 2 * b + 1
  (n^2 * (n - 1)) / 2 = n * (n - 1) * n / 2 - n * (n - 1) / 2 :=
by sorry

end NUMINAMATH_CALUDE_checkerboard_square_selection_l2939_293961


namespace NUMINAMATH_CALUDE_angle_between_vectors_l2939_293917

/-- Given two vectors a and b in ℝ², where a is perpendicular to b,
    prove that the angle between (a - b) and b is 150°. -/
theorem angle_between_vectors (a b : ℝ × ℝ) (h_a : a = (Real.sqrt 3, 1))
    (h_b : b.2 = -3) (h_perp : a.1 * b.1 + a.2 * b.2 = 0) :
    let diff := (a.1 - b.1, a.2 - b.2)
    Real.arccos ((diff.1 * b.1 + diff.2 * b.2) / 
      (Real.sqrt (diff.1^2 + diff.2^2) * Real.sqrt (b.1^2 + b.2^2))) =
    150 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_angle_between_vectors_l2939_293917


namespace NUMINAMATH_CALUDE_solution_interval_l2939_293910

theorem solution_interval (x : ℝ) : (x / 4 ≤ 3 + x ∧ 3 + x < -3 * (1 + x)) ↔ x ∈ Set.Ici (-4) ∩ Set.Iio (-3/2) := by
  sorry

end NUMINAMATH_CALUDE_solution_interval_l2939_293910


namespace NUMINAMATH_CALUDE_cos_195_plus_i_sin_195_to_60_l2939_293954

-- Define DeMoivre's Theorem
axiom deMoivre (θ : ℝ) (n : ℕ) : 
  (Complex.exp (Complex.I * θ)) ^ n = Complex.exp (Complex.I * (n * θ))

-- Define the problem
theorem cos_195_plus_i_sin_195_to_60 :
  (Complex.exp (Complex.I * (195 * π / 180))) ^ 60 = -1 := by sorry

end NUMINAMATH_CALUDE_cos_195_plus_i_sin_195_to_60_l2939_293954


namespace NUMINAMATH_CALUDE_triangle_side_length_l2939_293978

theorem triangle_side_length (a b c : ℝ) (A : ℝ) :
  a = Real.sqrt 5 →
  c = 2 →
  Real.cos A = 2 / 3 →
  b = 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2939_293978


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2939_293977

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x : ℝ, (1 - 2*x)^7 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = -2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2939_293977


namespace NUMINAMATH_CALUDE_reinforcement_size_l2939_293984

/-- Calculates the size of a reinforcement given initial garrison size, provision duration, and new provision duration after reinforcement arrival. -/
def calculate_reinforcement (initial_garrison : ℕ) (initial_duration : ℕ) (days_before_reinforcement : ℕ) (new_duration : ℕ) : ℕ :=
  let remaining_provisions := initial_garrison * (initial_duration - days_before_reinforcement)
  (remaining_provisions / new_duration) - initial_garrison

/-- Proves that the reinforcement size is 1900 given the problem conditions. -/
theorem reinforcement_size :
  calculate_reinforcement 2000 54 15 20 = 1900 := by
  sorry

end NUMINAMATH_CALUDE_reinforcement_size_l2939_293984


namespace NUMINAMATH_CALUDE_hyperbola_properties_l2939_293971

-- Define the hyperbola equation
def hyperbola_eq (x y : ℝ) : Prop :=
  17 * x^2 - 16 * x * y + 4 * y^2 - 34 * x + 16 * y + 13 = 0

-- Define the points A and B
def point_A : ℝ × ℝ := (1, 1)
def point_B : ℝ × ℝ := (1, -1)

-- Define the center
def center : ℝ × ℝ := (1, 0)

-- Define the conjugate axis equations
def conjugate_axis_eq (x y : ℝ) : Prop :=
  y = (13 + 5 * Real.sqrt 17) / 16 * (x - 1) ∨
  y = (13 - 5 * Real.sqrt 17) / 16 * (x - 1)

theorem hyperbola_properties :
  (hyperbola_eq point_A.1 point_A.2) ∧
  (hyperbola_eq point_B.1 point_B.2) →
  (∃ (x y : ℝ), hyperbola_eq x y ∧ conjugate_axis_eq x y) ∧
  (center.1 = 1 ∧ center.2 = 0) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l2939_293971


namespace NUMINAMATH_CALUDE_unique_positive_solution_arctan_equation_l2939_293991

theorem unique_positive_solution_arctan_equation :
  ∃! y : ℝ, y > 0 ∧ Real.arctan (1 / y) + Real.arctan (1 / y^2) = π / 4 :=
by sorry

end NUMINAMATH_CALUDE_unique_positive_solution_arctan_equation_l2939_293991


namespace NUMINAMATH_CALUDE_min_fleet_size_10x10_l2939_293916

/-- A ship is a figure made up of unit squares connected by common edges -/
def Ship : Type := Unit

/-- A fleet is a set of ships where no two ships contain squares that share a common vertex -/
def Fleet : Type := Set Ship

/-- The size of the grid -/
def gridSize : ℕ := 10

/-- The minimum number of squares in a fleet to which no new ship can be added -/
def minFleetSize (n : ℕ) : ℕ :=
  if n % 3 = 0 then (n / 3) ^ 2
  else (n / 3 + 1) ^ 2

theorem min_fleet_size_10x10 :
  minFleetSize gridSize = 16 := by sorry

end NUMINAMATH_CALUDE_min_fleet_size_10x10_l2939_293916


namespace NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l2939_293983

/-- Given an arithmetic sequence with first term 3^2 and third term 3^4, 
    the middle term y is equal to 45. -/
theorem arithmetic_sequence_middle_term : 
  ∀ (a : ℕ → ℤ), 
    (∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence condition
    a 0 = 3^2 →                                       -- first term
    a 2 = 3^4 →                                       -- third term
    a 1 = 45 :=                                       -- middle term (y)
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l2939_293983


namespace NUMINAMATH_CALUDE_arithmetic_sequence_equidistant_sum_l2939_293970

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_equidistant_sum
  (a : ℕ → ℝ) (h : arithmetic_sequence a) :
  a 4 + a 8 = 16 → a 2 + a 10 = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_equidistant_sum_l2939_293970


namespace NUMINAMATH_CALUDE_min_value_expression_l2939_293939

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a^4 + b^4 + 16 / (a^2 + b^2)^2 ≥ 4 ∧
  (a^4 + b^4 + 16 / (a^2 + b^2)^2 = 4 ↔ a = b ∧ a = 2^(1/4)) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2939_293939


namespace NUMINAMATH_CALUDE_sum_of_integers_between_1_and_10_l2939_293955

theorem sum_of_integers_between_1_and_10 : 
  (Finset.range 8).sum (fun i => i + 2) = 44 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_between_1_and_10_l2939_293955


namespace NUMINAMATH_CALUDE_sum_abcd_equals_negative_ten_thirds_l2939_293918

theorem sum_abcd_equals_negative_ten_thirds
  (a b c d : ℚ)
  (h : a + 1 = b + 2 ∧ b + 2 = c + 3 ∧ c + 3 = d + 4 ∧ d + 4 = a + b + c + d + 5) :
  a + b + c + d = -10/3 :=
by sorry

end NUMINAMATH_CALUDE_sum_abcd_equals_negative_ten_thirds_l2939_293918


namespace NUMINAMATH_CALUDE_point_symmetry_false_l2939_293938

/-- Two points in a 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Definition of symmetry about x-axis -/
def symmetricAboutXAxis (p1 p2 : Point2D) : Prop :=
  p1.x = p2.x ∧ p1.y = -p2.y

/-- The main theorem -/
theorem point_symmetry_false : 
  ¬ symmetricAboutXAxis ⟨-3, -4⟩ ⟨3, -4⟩ := by
  sorry

end NUMINAMATH_CALUDE_point_symmetry_false_l2939_293938


namespace NUMINAMATH_CALUDE_line_equation_l2939_293987

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/6 + y^2/3 = 1

-- Define the line l
def line_l (x y : ℝ) : Prop := x + Real.sqrt 2 * y - 2 * Real.sqrt 2 = 0

-- Define points A, B, M, and N
def point_A : ℝ × ℝ := sorry
def point_B : ℝ × ℝ := sorry
def point_M : ℝ × ℝ := sorry
def point_N : ℝ × ℝ := sorry

-- Define the conditions
def conditions : Prop :=
  ellipse point_A.1 point_A.2 ∧
  ellipse point_B.1 point_B.2 ∧
  point_A.1 > 0 ∧ point_A.2 > 0 ∧
  point_B.1 > 0 ∧ point_B.2 > 0 ∧
  point_M.2 = 0 ∧
  point_N.1 = 0 ∧
  (point_M.1 - point_A.1)^2 + (point_M.2 - point_A.2)^2 =
    (point_N.1 - point_B.1)^2 + (point_N.2 - point_B.2)^2 ∧
  (point_M.1 - point_N.1)^2 + (point_M.2 - point_N.2)^2 = 12

-- Theorem statement
theorem line_equation (h : conditions) : 
  ∀ x y, line_l x y ↔ (x, y) ∈ {p | ∃ t, p = (1-t) • point_M + t • point_N} :=
sorry

end NUMINAMATH_CALUDE_line_equation_l2939_293987


namespace NUMINAMATH_CALUDE_xingyou_age_l2939_293911

theorem xingyou_age : ℕ :=
  let current_age : ℕ := sorry
  let current_height : ℕ := sorry
  have h1 : current_age = current_height := by sorry
  have h2 : current_age + 3 = 2 * current_height := by sorry
  have h3 : current_age = 3 := by sorry
  3

#check xingyou_age

end NUMINAMATH_CALUDE_xingyou_age_l2939_293911


namespace NUMINAMATH_CALUDE_waiter_tip_calculation_l2939_293937

theorem waiter_tip_calculation (total_customers : ℕ) (non_tipping_customers : ℕ) (total_tip : ℚ) :
  total_customers = 7 →
  non_tipping_customers = 5 →
  total_tip = 6 →
  (total_tip / (total_customers - non_tipping_customers) : ℚ) = 3 :=
by sorry

end NUMINAMATH_CALUDE_waiter_tip_calculation_l2939_293937


namespace NUMINAMATH_CALUDE_gauss_family_mean_age_l2939_293945

def gauss_family_ages : List ℕ := [7, 7, 7, 14, 15]

theorem gauss_family_mean_age :
  (gauss_family_ages.sum / gauss_family_ages.length : ℚ) = 10 := by
  sorry

end NUMINAMATH_CALUDE_gauss_family_mean_age_l2939_293945


namespace NUMINAMATH_CALUDE_complex_sum_equality_l2939_293944

theorem complex_sum_equality : 
  let A : ℂ := 3 + 2*I
  let O : ℂ := -3 + I
  let P : ℂ := 1 - 2*I
  let S : ℂ := 4 + 5*I
  let T : ℂ := -1
  A - O + P + S + T = 10 + 4*I :=
by sorry

end NUMINAMATH_CALUDE_complex_sum_equality_l2939_293944


namespace NUMINAMATH_CALUDE_systematic_sampling_probability_l2939_293913

/-- Represents a sampling method -/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified

/-- Represents a population with a given size -/
structure Population where
  size : ℕ

/-- Represents a sample drawn from a population -/
structure Sample where
  size : ℕ
  population : Population
  method : SamplingMethod

/-- The probability of an individual being selected in a given sample -/
def selectionProbability (s : Sample) : ℚ :=
  s.size / s.population.size

/-- Theorem: In a population of 2008 parts, after eliminating 8 parts randomly
    and then selecting 20 parts using systematic sampling, the probability
    of each part being selected is 20/2008 -/
theorem systematic_sampling_probability :
  let initialPopulation : Population := ⟨2008⟩
  let eliminatedPopulation : Population := ⟨2000⟩
  let sample : Sample := ⟨20, eliminatedPopulation, SamplingMethod.Systematic⟩
  selectionProbability sample = 20 / 2008 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_probability_l2939_293913


namespace NUMINAMATH_CALUDE_expression_simplification_l2939_293960

theorem expression_simplification (a : ℝ) (h : a = Real.sqrt 3 - 3) : 
  (a^2 - 4*a + 4) / (a^2 - 4) / ((a - 2) / (a^2 + 2*a)) + 3 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2939_293960


namespace NUMINAMATH_CALUDE_cos_210_degrees_l2939_293980

theorem cos_210_degrees : Real.cos (210 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_210_degrees_l2939_293980


namespace NUMINAMATH_CALUDE_video_game_lives_l2939_293947

/-- 
Given a video game scenario where:
- x is the initial number of lives
- y is the number of power-ups collected
- Each power-up gives 5 extra lives
- The player lost 13 lives
- After these events, the player ended up with 70 lives

Prove that the initial number of lives (x) is equal to 83 minus 5 times 
the number of power-ups collected (y).
-/
theorem video_game_lives (x y : ℤ) : 
  (x - 13 + 5 * y = 70) → (x = 83 - 5 * y) := by
  sorry

end NUMINAMATH_CALUDE_video_game_lives_l2939_293947


namespace NUMINAMATH_CALUDE_age_difference_l2939_293914

/-- Given three people a, b, and c, prove that a is 2 years older than b -/
theorem age_difference (a b c : ℕ) : 
  (∃ k, a = b + k) →  -- a is some years older than b
  (b = 2 * c) →       -- b is twice as old as c
  (a + b + c = 27) →  -- The total of the ages of a, b, and c is 27
  (b = 10) →          -- b is 10 years old
  (a = b + 2)         -- a is 2 years older than b
  := by sorry

end NUMINAMATH_CALUDE_age_difference_l2939_293914


namespace NUMINAMATH_CALUDE_hiker_catches_cyclist_l2939_293958

/-- Proves that a hiker catches up to a cyclist in 30 minutes under specific conditions -/
theorem hiker_catches_cyclist (hiker_speed cyclist_speed : ℝ) (cyclist_travel_time : ℝ) : 
  hiker_speed = 4 →
  cyclist_speed = 24 →
  cyclist_travel_time = 5 / 60 →
  let cyclist_distance := cyclist_speed * cyclist_travel_time
  let catchup_time := cyclist_distance / hiker_speed
  catchup_time * 60 = 30 := by sorry

end NUMINAMATH_CALUDE_hiker_catches_cyclist_l2939_293958


namespace NUMINAMATH_CALUDE_jason_money_last_week_l2939_293964

/-- Given information about Fred and Jason's money before and after washing cars,
    prove how much money Jason had last week. -/
theorem jason_money_last_week
  (fred_money_last_week : ℕ)
  (fred_money_now : ℕ)
  (jason_money_now : ℕ)
  (fred_earned : ℕ)
  (h1 : fred_money_last_week = 19)
  (h2 : fred_money_now = 40)
  (h3 : jason_money_now = 69)
  (h4 : fred_earned = 21)
  (h5 : fred_money_now = fred_money_last_week + fred_earned) :
  jason_money_now - fred_earned = 48 :=
by sorry

end NUMINAMATH_CALUDE_jason_money_last_week_l2939_293964


namespace NUMINAMATH_CALUDE_g_expression_and_minimum_l2939_293986

noncomputable def f (a x : ℝ) : ℝ := a * x^2 - 2 * x + 1

noncomputable def M (a : ℝ) : ℝ := max (f a 1) (f a 3)

noncomputable def N (a : ℝ) : ℝ := 1 - 1/a

noncomputable def g (a : ℝ) : ℝ := M a - N a

theorem g_expression_and_minimum (a : ℝ) (h : 1/3 ≤ a ∧ a ≤ 1) :
  ((1/3 ≤ a ∧ a ≤ 1/2 → g a = a - 2 + 1/a) ∧
   (1/2 < a ∧ a ≤ 1 → g a = 9*a - 6 + 1/a)) ∧
  (∀ b, 1/3 ≤ b ∧ b ≤ 1 → g b ≥ 1/2) ∧
  g (1/2) = 1/2 := by sorry

end NUMINAMATH_CALUDE_g_expression_and_minimum_l2939_293986


namespace NUMINAMATH_CALUDE_kitten_weight_l2939_293979

theorem kitten_weight (k d1 d2 : ℝ) 
  (total_weight : k + d1 + d2 = 30)
  (larger_dog_relation : k + d2 = 3 * d1)
  (smaller_dog_relation : k + d1 = d2 + 10) :
  k = 25 / 2 := by
sorry

end NUMINAMATH_CALUDE_kitten_weight_l2939_293979


namespace NUMINAMATH_CALUDE_second_meeting_time_l2939_293926

-- Define the number of rounds Charging Bull completes in an hour
def charging_bull_rounds_per_hour : ℕ := 40

-- Define the time Racing Magic takes to complete one round (in seconds)
def racing_magic_time_per_round : ℕ := 150

-- Define the number of seconds in an hour
def seconds_per_hour : ℕ := 3600

-- Define the function to calculate the meeting time in minutes
def meeting_time : ℕ :=
  let racing_magic_rounds_per_hour := seconds_per_hour / racing_magic_time_per_round
  let lcm_rounds := Nat.lcm racing_magic_rounds_per_hour charging_bull_rounds_per_hour
  let hours_to_meet := lcm_rounds / racing_magic_rounds_per_hour
  hours_to_meet * 60

-- Theorem statement
theorem second_meeting_time :
  meeting_time = 300 := by sorry

end NUMINAMATH_CALUDE_second_meeting_time_l2939_293926


namespace NUMINAMATH_CALUDE_abc_value_l2939_293920

theorem abc_value (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a * b = 24 * Real.rpow 3 (1/4))
  (hac : a * c = 40 * Real.rpow 3 (1/4))
  (hbc : b * c = 15 * Real.rpow 3 (1/4)) :
  a * b * c = 120 * Real.rpow 3 (3/8) := by
sorry

end NUMINAMATH_CALUDE_abc_value_l2939_293920


namespace NUMINAMATH_CALUDE_game_probabilities_and_earnings_l2939_293951

/-- Represents the outcome of drawing balls -/
inductive DrawOutcome
  | AllSameColor
  | DifferentColors

/-- Represents the game setup -/
structure GameSetup :=
  (total_balls : Nat)
  (white_balls : Nat)
  (yellow_balls : Nat)
  (same_color_payout : Int)
  (diff_color_payment : Int)
  (draws_per_day : Nat)
  (days_per_month : Nat)

/-- Calculates the probability of drawing 3 white balls -/
def prob_three_white (setup : GameSetup) : Rat :=
  sorry

/-- Calculates the probability of drawing 2 yellow and 1 white ball -/
def prob_two_yellow_one_white (setup : GameSetup) : Rat :=
  sorry

/-- Calculates the expected monthly earnings -/
def expected_monthly_earnings (setup : GameSetup) : Int :=
  sorry

/-- Main theorem stating the probabilities and expected earnings -/
theorem game_probabilities_and_earnings (setup : GameSetup)
  (h1 : setup.total_balls = 6)
  (h2 : setup.white_balls = 3)
  (h3 : setup.yellow_balls = 3)
  (h4 : setup.same_color_payout = -5)
  (h5 : setup.diff_color_payment = 1)
  (h6 : setup.draws_per_day = 100)
  (h7 : setup.days_per_month = 30) :
  prob_three_white setup = 1/20 ∧
  prob_two_yellow_one_white setup = 1/10 ∧
  expected_monthly_earnings setup = 1200 :=
sorry

end NUMINAMATH_CALUDE_game_probabilities_and_earnings_l2939_293951


namespace NUMINAMATH_CALUDE_julia_watch_collection_l2939_293949

/-- Proves that the percentage of gold watches in Julia's collection is 9.09% -/
theorem julia_watch_collection (silver : ℕ) (bronze : ℕ) (gold : ℕ) (total : ℕ) : 
  silver = 20 →
  bronze = 3 * silver →
  total = silver + bronze + gold →
  total = 88 →
  (gold : ℝ) / (total : ℝ) * 100 = 9.09 := by
  sorry

end NUMINAMATH_CALUDE_julia_watch_collection_l2939_293949


namespace NUMINAMATH_CALUDE_interest_rate_problem_l2939_293909

/-- Simple interest calculation -/
def simple_interest (principal rate time : ℚ) : ℚ :=
  principal * rate * time / 100

/-- Problem statement -/
theorem interest_rate_problem (principal interest time : ℚ) 
  (h1 : principal = 4000)
  (h2 : interest = 640)
  (h3 : time = 2)
  (h4 : simple_interest principal (8 : ℚ) time = interest) :
  8 = (interest * 100) / (principal * time) :=
by sorry

end NUMINAMATH_CALUDE_interest_rate_problem_l2939_293909


namespace NUMINAMATH_CALUDE_remainder_of_M_div_500_l2939_293924

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def product_of_factorials : ℕ := (List.range 50).foldl (fun acc n => acc * factorial (n + 1)) 1

def trailing_zeros (n : ℕ) : ℕ := 
  if n = 0 then 0 else (n.digits 10).takeWhile (· = 0) |>.length

def M : ℕ := trailing_zeros product_of_factorials

theorem remainder_of_M_div_500 : M % 500 = 21 := by sorry

end NUMINAMATH_CALUDE_remainder_of_M_div_500_l2939_293924


namespace NUMINAMATH_CALUDE_total_learning_time_is_19_l2939_293968

/-- Represents the learning time for each vowel -/
def vowel_time : Fin 5 → ℕ
  | 0 => 4  -- A
  | 1 => 6  -- E
  | 2 => 5  -- I
  | 3 => 3  -- O
  | 4 => 4  -- U

/-- The break time between learning pairs -/
def break_time : ℕ := 2

/-- Calculates the total learning time for all vowels -/
def total_learning_time : ℕ :=
  let pair1 := max (vowel_time 1) (vowel_time 3)  -- E and O
  let pair2 := max (vowel_time 2) (vowel_time 4)  -- I and U
  let single := vowel_time 0  -- A
  pair1 + break_time + pair2 + break_time + single

/-- Theorem stating that the total learning time is 19 days -/
theorem total_learning_time_is_19 : total_learning_time = 19 := by
  sorry

#eval total_learning_time

end NUMINAMATH_CALUDE_total_learning_time_is_19_l2939_293968

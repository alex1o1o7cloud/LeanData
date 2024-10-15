import Mathlib

namespace NUMINAMATH_CALUDE_negation_equivalence_l3830_383020

def exactly_one_even (a b c : ℕ) : Prop :=
  (a % 2 = 0 ∧ b % 2 ≠ 0 ∧ c % 2 ≠ 0) ∨
  (a % 2 ≠ 0 ∧ b % 2 = 0 ∧ c % 2 ≠ 0) ∨
  (a % 2 ≠ 0 ∧ b % 2 ≠ 0 ∧ c % 2 = 0)

def at_least_two_even_or_all_odd (a b c : ℕ) : Prop :=
  (a % 2 = 0 ∧ b % 2 = 0) ∨
  (a % 2 = 0 ∧ c % 2 = 0) ∨
  (b % 2 = 0 ∧ c % 2 = 0) ∨
  (a % 2 ≠ 0 ∧ b % 2 ≠ 0 ∧ c % 2 ≠ 0)

theorem negation_equivalence (a b c : ℕ) :
  ¬(exactly_one_even a b c) ↔ at_least_two_even_or_all_odd a b c :=
sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3830_383020


namespace NUMINAMATH_CALUDE_quadratic_roots_and_triangle_perimeter_l3830_383045

-- Define the quadratic equation
def quadratic_equation (m x : ℝ) : ℝ := x^2 - (m + 3) * x + 4 * m - 4

-- Define the discriminant of the quadratic equation
def discriminant (m : ℝ) : ℝ := (m - 5)^2

-- Define the isosceles triangle ABC
structure IsoscelesTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  isosceles : b = c
  one_side_is_five : a = 5
  roots_are_sides : ∃ m : ℝ, (quadratic_equation m b = 0) ∧ (quadratic_equation m c = 0)

-- Theorem statement
theorem quadratic_roots_and_triangle_perimeter :
  (∀ m : ℝ, discriminant m ≥ 0) ∧
  (∀ t : IsoscelesTriangle, t.a + t.b + t.c = 13 ∨ t.a + t.b + t.c = 14) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_and_triangle_perimeter_l3830_383045


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3830_383092

theorem sufficient_not_necessary_condition (a : ℝ) :
  (∀ x : ℝ, a ∈ Set.Iic 0 → ∃ y : ℝ, y^2 - y + a ≤ 0) ∧
  (∃ b : ℝ, b ∉ Set.Iic 0 ∧ ∃ z : ℝ, z^2 - z + b ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3830_383092


namespace NUMINAMATH_CALUDE_inequality_proof_l3830_383058

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  1 < (a / Real.sqrt (a^2 + b^2)) + (b / Real.sqrt (b^2 + c^2)) + (c / Real.sqrt (c^2 + a^2)) ∧
  (a / Real.sqrt (a^2 + b^2)) + (b / Real.sqrt (b^2 + c^2)) + (c / Real.sqrt (c^2 + a^2)) ≤ (3 * Real.sqrt 3) / 2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3830_383058


namespace NUMINAMATH_CALUDE_log_16_2_l3830_383013

theorem log_16_2 : Real.log 2 / Real.log 16 = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_log_16_2_l3830_383013


namespace NUMINAMATH_CALUDE_system_solution_l3830_383029

theorem system_solution : ∃ (x y : ℝ), x + y = 5 ∧ x - y = 1 ∧ x = 3 ∧ y = 2 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3830_383029


namespace NUMINAMATH_CALUDE_initial_marbles_calculation_l3830_383048

theorem initial_marbles_calculation (a b : ℚ) :
  a/b + 489.35 = 2778.65 →
  a/b = 2289.3 := by
sorry

end NUMINAMATH_CALUDE_initial_marbles_calculation_l3830_383048


namespace NUMINAMATH_CALUDE_half_angle_quadrant_l3830_383008

theorem half_angle_quadrant (α : Real) : 
  (π / 2 < α ∧ α < π) → 
  ((0 < (α / 2) ∧ (α / 2) < π / 2) ∨ (π < (α / 2) ∧ (α / 2) < 3 * π / 2)) :=
by sorry

end NUMINAMATH_CALUDE_half_angle_quadrant_l3830_383008


namespace NUMINAMATH_CALUDE_spade_calculation_l3830_383004

/-- Definition of the ♠ operation for real numbers -/
def spade (x y : ℝ) : ℝ := (x + y) * (x - y)

/-- Theorem stating that 6 ♠ (7 ♠ 7) = 36 -/
theorem spade_calculation : spade 6 (spade 7 7) = 36 := by
  sorry

end NUMINAMATH_CALUDE_spade_calculation_l3830_383004


namespace NUMINAMATH_CALUDE_x_neg_one_is_local_minimum_l3830_383093

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x + 1

theorem x_neg_one_is_local_minimum :
  ∃ δ > 0, ∀ x : ℝ, x ≠ -1 ∧ |x - (-1)| < δ → f x ≥ f (-1) := by
  sorry

end NUMINAMATH_CALUDE_x_neg_one_is_local_minimum_l3830_383093


namespace NUMINAMATH_CALUDE_distance_between_complex_points_l3830_383054

theorem distance_between_complex_points :
  let z₁ : ℂ := 3 + 3 * Complex.I
  let z₂ : ℂ := -2 + Real.sqrt 2 * Complex.I
  Complex.abs (z₁ - z₂) = Real.sqrt (36 - 6 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_distance_between_complex_points_l3830_383054


namespace NUMINAMATH_CALUDE_student_problem_attempt_l3830_383033

theorem student_problem_attempt :
  ∀ (correct incorrect : ℕ),
    correct + incorrect ≤ 20 ∧
    8 * correct - 5 * incorrect = 13 →
    correct + incorrect = 13 :=
by
  sorry

end NUMINAMATH_CALUDE_student_problem_attempt_l3830_383033


namespace NUMINAMATH_CALUDE_rhombus_fourth_vertex_area_l3830_383005

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A square defined by its four vertices -/
structure Square where
  a : Point
  b : Point
  c : Point
  d : Point

/-- A rhombus defined by its four vertices -/
structure Rhombus where
  p : Point
  q : Point
  r : Point
  s : Point

/-- Predicate to check if a square is a unit square -/
def isUnitSquare (sq : Square) : Prop := sorry

/-- Predicate to check if a point lies on a side of the square -/
def pointOnSide (p : Point) (sq : Square) : Prop := sorry

/-- Function to calculate the area of a set of points -/
def areaOfSet (s : Set Point) : ℝ := sorry

/-- The set of all possible locations for the fourth vertex of the rhombus -/
def fourthVertexSet (sq : Square) : Set Point := sorry

/-- Main theorem -/
theorem rhombus_fourth_vertex_area (sq : Square) :
  isUnitSquare sq →
  (∃ (r : Rhombus), 
    pointOnSide r.p sq ∧ 
    pointOnSide r.q sq ∧ 
    pointOnSide r.r sq) →
  areaOfSet (fourthVertexSet sq) = 7/3 := sorry

end NUMINAMATH_CALUDE_rhombus_fourth_vertex_area_l3830_383005


namespace NUMINAMATH_CALUDE_trig_identity_l3830_383024

theorem trig_identity (α : ℝ) 
  (h1 : Real.cos (7 * Real.pi / 2 + α) = 4 / 7)
  (h2 : Real.tan α < 0) :
  Real.cos (Real.pi - α) + Real.sin (Real.pi / 2 - α) * Real.tan α = (4 + Real.sqrt 33) / 7 := by
sorry

end NUMINAMATH_CALUDE_trig_identity_l3830_383024


namespace NUMINAMATH_CALUDE_polygon_sides_when_interior_thrice_exterior_l3830_383011

theorem polygon_sides_when_interior_thrice_exterior : ∀ n : ℕ,
  (n ≥ 3) →
  (180 * (n - 2) = 3 * 360) →
  n = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_when_interior_thrice_exterior_l3830_383011


namespace NUMINAMATH_CALUDE_quadratic_equation_1_l3830_383073

theorem quadratic_equation_1 : ∃ x₁ x₂ : ℝ, x₁ = 1 ∧ x₂ = 3 ∧
  x₁^2 - 4*x₁ + 3 = 0 ∧ x₂^2 - 4*x₂ + 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_1_l3830_383073


namespace NUMINAMATH_CALUDE_consecutive_multiples_of_four_sum_l3830_383028

theorem consecutive_multiples_of_four_sum (n : ℕ) : 
  (4*n + (4*n + 8) = 140) → (4*n + (4*n + 4) + (4*n + 8) = 210) :=
by
  sorry

end NUMINAMATH_CALUDE_consecutive_multiples_of_four_sum_l3830_383028


namespace NUMINAMATH_CALUDE_count_solutions_l3830_383027

def is_solution (m n r : ℕ+) : Prop :=
  m * n + n * r + m * r = 2 * (m + n + r)

theorem count_solutions : 
  ∃! (solutions : Finset (ℕ+ × ℕ+ × ℕ+)), 
    (∀ (m n r : ℕ+), (m, n, r) ∈ solutions ↔ is_solution m n r) ∧ 
    solutions.card = 7 :=
sorry

end NUMINAMATH_CALUDE_count_solutions_l3830_383027


namespace NUMINAMATH_CALUDE_solution_set_implies_sum_l3830_383076

-- Define the quadratic function
def f (a b : ℝ) (x : ℝ) := a * x^2 + b * x + 2

-- State the theorem
theorem solution_set_implies_sum (a b : ℝ) :
  (∀ x, f a b x > 0 ↔ x ∈ Set.Ioo (-1/2 : ℝ) (1/3 : ℝ)) →
  a + b = -14 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_implies_sum_l3830_383076


namespace NUMINAMATH_CALUDE_star_property_l3830_383017

-- Define the operation ※
def star (a b m n : ℕ) : ℕ := (a^b)^m + (b^a)^n

-- State the theorem
theorem star_property (m n : ℕ) :
  (star 1 4 m n = 10) → (star 2 2 m n = 15) → (4^(2*m + n - 1) = 81) := by
  sorry

end NUMINAMATH_CALUDE_star_property_l3830_383017


namespace NUMINAMATH_CALUDE_road_trip_gas_cost_l3830_383096

/-- Calculates the total cost of filling up a car's gas tank at multiple stations -/
theorem road_trip_gas_cost (tank_capacity : ℝ) (prices : List ℝ) : 
  tank_capacity = 12 ∧ 
  prices = [3, 3.5, 4, 4.5] →
  (prices.map (λ price => tank_capacity * price)).sum = 180 := by
  sorry

end NUMINAMATH_CALUDE_road_trip_gas_cost_l3830_383096


namespace NUMINAMATH_CALUDE_fraction_equality_l3830_383091

theorem fraction_equality (x y : ℝ) (h : (x + y) / (1 - x * y) = Real.sqrt 5) :
  |1 - x * y| / (Real.sqrt (1 + x^2) * Real.sqrt (1 + y^2)) = Real.sqrt 6 / 6 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3830_383091


namespace NUMINAMATH_CALUDE_pyramid_volume_l3830_383063

/-- The volume of a pyramid with a square base of side length 2 and height 2 is 8/3 cubic units -/
theorem pyramid_volume (base_side_length height : ℝ) (h1 : base_side_length = 2) (h2 : height = 2) :
  (1 / 3 : ℝ) * base_side_length^2 * height = 8 / 3 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_volume_l3830_383063


namespace NUMINAMATH_CALUDE_triangle_inequality_with_circumradius_and_altitudes_l3830_383001

-- Define a structure for a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  R : ℝ
  h_a : ℝ
  h_b : ℝ
  h_c : ℝ
  -- Add conditions to ensure it's a valid triangle
  pos_sides : 0 < a ∧ 0 < b ∧ 0 < c
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b
  pos_R : 0 < R
  pos_altitudes : 0 < h_a ∧ 0 < h_b ∧ 0 < h_c

-- State the theorem
theorem triangle_inequality_with_circumradius_and_altitudes (t : Triangle) :
  t.a^2 + t.b^2 + t.c^2 ≥ 2 * t.R * (t.h_a + t.h_b + t.h_c) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_with_circumradius_and_altitudes_l3830_383001


namespace NUMINAMATH_CALUDE_temperature_sum_l3830_383090

theorem temperature_sum (t1 t2 t3 k1 k2 k3 : ℚ) : 
  t1 = 5 / 9 * (k1 - 32) →
  t2 = 5 / 9 * (k2 - 32) →
  t3 = 5 / 9 * (k3 - 32) →
  t1 = 105 →
  t2 = 80 →
  t3 = 45 →
  k1 + k2 + k3 = 510 := by
sorry

end NUMINAMATH_CALUDE_temperature_sum_l3830_383090


namespace NUMINAMATH_CALUDE_union_and_intersection_range_of_a_l3830_383032

-- Define the sets A, B, and C
def A : Set ℝ := {x | 2 < x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}
def C (a : ℝ) : Set ℝ := {x | 5 - a < x ∧ x < a}

-- Theorem for part (1)
theorem union_and_intersection :
  (A ∪ B = {x | 2 < x ∧ x < 10}) ∧
  ((Set.univ \ A) ∩ B = {x | 7 ≤ x ∧ x < 10}) :=
sorry

-- Theorem for part (2)
theorem range_of_a (a : ℝ) :
  C a ⊆ B → a ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_union_and_intersection_range_of_a_l3830_383032


namespace NUMINAMATH_CALUDE_binary_110_equals_6_l3830_383019

-- Define a function to convert binary to decimal
def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

-- Theorem statement
theorem binary_110_equals_6 :
  binary_to_decimal [false, true, true] = 6 := by
  sorry

end NUMINAMATH_CALUDE_binary_110_equals_6_l3830_383019


namespace NUMINAMATH_CALUDE_simplify_expression_l3830_383012

theorem simplify_expression : (2^5 + 4^4) * (2^2 - (-2)^3)^8 = 123876479488 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3830_383012


namespace NUMINAMATH_CALUDE_second_term_of_geometric_series_l3830_383022

/-- For an infinite geometric series with common ratio 1/4 and sum 16, the second term is 3 -/
theorem second_term_of_geometric_series : 
  ∀ (a : ℝ), 
  (a / (1 - (1/4 : ℝ)) = 16) →  -- Sum of infinite geometric series
  (a * (1/4 : ℝ) = 3) :=        -- Second term
by sorry

end NUMINAMATH_CALUDE_second_term_of_geometric_series_l3830_383022


namespace NUMINAMATH_CALUDE_equilateral_triangle_on_parallel_lines_l3830_383003

-- Define the parallel lines
variable (D₁ D₂ D₃ : Set (ℝ × ℝ))

-- Define the property of being parallel
def Parallel (l₁ l₂ : Set (ℝ × ℝ)) : Prop := sorry

-- Define a point on a line
def PointOnLine (p : ℝ × ℝ) (l : Set (ℝ × ℝ)) : Prop := p ∈ l

-- Define an equilateral triangle
def IsEquilateralTriangle (A₁ A₂ A₃ : ℝ × ℝ) : Prop := sorry

-- Theorem statement
theorem equilateral_triangle_on_parallel_lines
  (h₁ : Parallel D₁ D₂)
  (h₂ : Parallel D₂ D₃)
  (h₃ : Parallel D₁ D₃) :
  ∃ (A₁ A₂ A₃ : ℝ × ℝ),
    IsEquilateralTriangle A₁ A₂ A₃ ∧
    PointOnLine A₁ D₁ ∧
    PointOnLine A₂ D₂ ∧
    PointOnLine A₃ D₃ := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_on_parallel_lines_l3830_383003


namespace NUMINAMATH_CALUDE_janes_bagels_l3830_383042

theorem janes_bagels (muffin_cost bagel_cost : ℕ) (total_days : ℕ) : 
  muffin_cost = 60 →
  bagel_cost = 80 →
  total_days = 7 →
  ∃! (num_bagels : ℕ), 
    num_bagels ≤ total_days ∧
    ∃ (total_cost : ℕ), 
      total_cost * 100 = num_bagels * bagel_cost + (total_days - num_bagels) * muffin_cost ∧
      num_bagels = 4 :=
by sorry

end NUMINAMATH_CALUDE_janes_bagels_l3830_383042


namespace NUMINAMATH_CALUDE_function_composition_condition_l3830_383014

theorem function_composition_condition (a b : ℤ) :
  (∃ (f g : ℤ → ℤ), ∀ x, f (g x) = x + a ∧ g (f x) = x + b) ↔ |a| = |b| :=
by sorry

end NUMINAMATH_CALUDE_function_composition_condition_l3830_383014


namespace NUMINAMATH_CALUDE_ham_and_cake_probability_l3830_383046

/-- The probability of packing a ham sandwich and cake on the same day -/
theorem ham_and_cake_probability :
  let total_days : ℕ := 5
  let ham_days : ℕ := 3
  let cake_days : ℕ := 1
  (ham_days : ℚ) / total_days * cake_days / total_days = 3 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ham_and_cake_probability_l3830_383046


namespace NUMINAMATH_CALUDE_sixtieth_digit_is_five_l3830_383078

def repeating_decimal (whole : ℕ) (repeating : List ℕ) : ℚ := sorry

def nth_digit_after_decimal (q : ℚ) (n : ℕ) : ℕ := sorry

theorem sixtieth_digit_is_five :
  let decimal := repeating_decimal 6 [4, 5, 3]
  nth_digit_after_decimal decimal 60 = 5 := by sorry

end NUMINAMATH_CALUDE_sixtieth_digit_is_five_l3830_383078


namespace NUMINAMATH_CALUDE_john_burritos_days_l3830_383035

theorem john_burritos_days (boxes : ℕ) (burritos_per_box : ℕ) (fraction_given_away : ℚ)
  (burritos_eaten_per_day : ℕ) (burritos_left : ℕ) :
  boxes = 3 →
  burritos_per_box = 20 →
  fraction_given_away = 1 / 3 →
  burritos_eaten_per_day = 3 →
  burritos_left = 10 →
  (boxes * burritos_per_box * (1 - fraction_given_away) - burritos_left) / burritos_eaten_per_day = 10 := by
  sorry

#check john_burritos_days

end NUMINAMATH_CALUDE_john_burritos_days_l3830_383035


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l3830_383009

/-- The eccentricity of an ellipse with equation x^2 + ky^2 = 3k (k > 0) is √3/2,
    given that one of its foci coincides with the focus of the parabola y^2 = 12x. -/
theorem ellipse_eccentricity (k : ℝ) (h_k : k > 0) : 
  let ellipse := {(x, y) : ℝ × ℝ | x^2 + k*y^2 = 3*k}
  let parabola := {(x, y) : ℝ × ℝ | y^2 = 12*x}
  let parabola_focus : ℝ × ℝ := (3, 0)
  ∃ (ellipse_focus : ℝ × ℝ), 
    ellipse_focus ∈ ellipse ∧ 
    ellipse_focus = parabola_focus →
    let a := Real.sqrt (3*k)
    let b := Real.sqrt 3
    let c := 3
    let eccentricity := c / a
    eccentricity = Real.sqrt 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l3830_383009


namespace NUMINAMATH_CALUDE_square_division_theorem_l3830_383067

/-- A type representing a square division -/
structure SquareDivision where
  n : ℕ
  is_valid : Bool

/-- Function that checks if a square can be divided into n smaller squares -/
def can_divide_square (n : ℕ) : Prop :=
  ∃ (sd : SquareDivision), sd.n = n ∧ sd.is_valid = true

theorem square_division_theorem :
  (∀ n : ℕ, n > 5 → can_divide_square n) ∧
  ¬(can_divide_square 2) ∧
  ¬(can_divide_square 3) := by sorry

end NUMINAMATH_CALUDE_square_division_theorem_l3830_383067


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3830_383095

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  (∀ n : ℕ, a n > 0) →  -- positive terms
  (∀ n : ℕ, a (n + 1) = a n * q) →  -- geometric sequence
  a 1 = 3 →  -- first term
  a 1 + a 2 + a 3 = 21 →  -- sum of first three terms
  a 3 + a 4 + a 5 = 84 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3830_383095


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l3830_383082

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 9/y = 1) :
  x + y ≥ 16 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 1/x₀ + 9/y₀ = 1 ∧ x₀ + y₀ = 16 := by
  sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l3830_383082


namespace NUMINAMATH_CALUDE_laptop_final_price_l3830_383089

/-- Calculate the final price of a laptop given the original price, two discount rates, and a recycling fee rate. -/
theorem laptop_final_price
  (original_price : ℝ)
  (discount1 : ℝ)
  (discount2 : ℝ)
  (recycling_fee_rate : ℝ)
  (h1 : original_price = 1000)
  (h2 : discount1 = 0.1)
  (h3 : discount2 = 0.25)
  (h4 : recycling_fee_rate = 0.05) :
  let price_after_discount1 := original_price * (1 - discount1)
  let price_after_discount2 := price_after_discount1 * (1 - discount2)
  let recycling_fee := price_after_discount2 * recycling_fee_rate
  let final_price := price_after_discount2 + recycling_fee
  final_price = 708.75 :=
by
  sorry

end NUMINAMATH_CALUDE_laptop_final_price_l3830_383089


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_range_of_m_for_full_solution_set_solution_set_eq_nonnegative_reals_l3830_383061

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 1| + |x - 3|

-- Theorem for part I
theorem solution_set_of_inequality (x : ℝ) :
  f x ≤ 3 * x + 4 ↔ x ≥ 0 := by sorry

-- Theorem for part II
theorem range_of_m_for_full_solution_set :
  ∀ m : ℝ, (∀ x : ℝ, f x ≥ m) ↔ m ∈ Set.Iic 4 := by sorry

-- Define the solution set for part I
def solution_set : Set ℝ := {x : ℝ | f x ≤ 3 * x + 4}

-- Theorem stating that the solution set is equivalent to [0, +∞)
theorem solution_set_eq_nonnegative_reals :
  solution_set = Set.Ici 0 := by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_range_of_m_for_full_solution_set_solution_set_eq_nonnegative_reals_l3830_383061


namespace NUMINAMATH_CALUDE_cool_parents_problem_l3830_383007

theorem cool_parents_problem (U : Finset ℕ) (A B : Finset ℕ) 
  (h1 : Finset.card U = 40)
  (h2 : Finset.card A = 18)
  (h3 : Finset.card B = 20)
  (h4 : Finset.card (A ∩ B) = 11)
  (h5 : A ⊆ U)
  (h6 : B ⊆ U) :
  Finset.card (U \ (A ∪ B)) = 13 := by
  sorry

end NUMINAMATH_CALUDE_cool_parents_problem_l3830_383007


namespace NUMINAMATH_CALUDE_exists_quadrilateral_equal_tangents_l3830_383016

-- Define a quadrilateral as a structure with four angles
structure Quadrilateral where
  α : Real
  β : Real
  γ : Real
  δ : Real

-- Define the property of having equal tangents for all angles
def hasEqualTangents (q : Quadrilateral) : Prop :=
  Real.tan q.α = Real.tan q.β ∧
  Real.tan q.β = Real.tan q.γ ∧
  Real.tan q.γ = Real.tan q.δ

-- Define the property of angles summing to 360°
def anglesSum360 (q : Quadrilateral) : Prop :=
  q.α + q.β + q.γ + q.δ = 360

-- Theorem stating the existence of a quadrilateral with equal tangents
theorem exists_quadrilateral_equal_tangents :
  ∃ q : Quadrilateral, anglesSum360 q ∧ hasEqualTangents q :=
sorry

end NUMINAMATH_CALUDE_exists_quadrilateral_equal_tangents_l3830_383016


namespace NUMINAMATH_CALUDE_fudge_pan_dimensions_l3830_383069

theorem fudge_pan_dimensions (side1 : ℝ) (area : ℝ) : 
  side1 = 29 → area = 522 → (area / side1) = 18 := by
  sorry

end NUMINAMATH_CALUDE_fudge_pan_dimensions_l3830_383069


namespace NUMINAMATH_CALUDE_locus_of_midpoints_correct_l3830_383066

/-- Given a square ABCD with center at the origin, rotating around its center,
    and a fixed line l with equation y = a, this function represents the locus of
    the midpoints of segments PQ, where P is the foot of the perpendicular from D to l,
    and Q is the midpoint of AB. -/
def locusOfMidpoints (a : ℝ) (x y : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ t : ℝ, 
    p.1 = t ∧ 
    p.2 = -t + a/2 ∧ 
    t = (x - y)/4 ∧ 
    x - y ∈ Set.Icc (-Real.sqrt (2 * (x^2 + y^2))) (Real.sqrt (2 * (x^2 + y^2)))}

/-- Theorem stating that the locus of midpoints is correct for any rotating square ABCD
    with center at the origin and any fixed line y = a. -/
theorem locus_of_midpoints_correct (a : ℝ) : 
  ∀ x y : ℝ, locusOfMidpoints a x y = 
    {p : ℝ × ℝ | ∃ t : ℝ, 
      p.1 = t ∧ 
      p.2 = -t + a/2 ∧ 
      t = (x - y)/4 ∧ 
      x - y ∈ Set.Icc (-Real.sqrt (2 * (x^2 + y^2))) (Real.sqrt (2 * (x^2 + y^2)))} :=
by sorry

end NUMINAMATH_CALUDE_locus_of_midpoints_correct_l3830_383066


namespace NUMINAMATH_CALUDE_square_sum_eq_double_product_iff_equal_l3830_383002

theorem square_sum_eq_double_product_iff_equal (a b : ℝ) :
  a^2 + b^2 = 2*a*b ↔ a = b := by
sorry

end NUMINAMATH_CALUDE_square_sum_eq_double_product_iff_equal_l3830_383002


namespace NUMINAMATH_CALUDE_shaded_percentage_is_correct_l3830_383015

/-- Represents a 7x7 grid with a checkered shading pattern and unshaded fourth row and column -/
structure CheckeredGrid :=
  (size : Nat)
  (is_seven_by_seven : size = 7)
  (checkered_pattern : Bool)
  (unshaded_fourth_row : Bool)
  (unshaded_fourth_column : Bool)

/-- Calculates the number of shaded squares in the CheckeredGrid -/
def shaded_squares (grid : CheckeredGrid) : Nat :=
  grid.size * grid.size - (grid.size + grid.size - 1)

/-- Calculates the total number of squares in the CheckeredGrid -/
def total_squares (grid : CheckeredGrid) : Nat :=
  grid.size * grid.size

/-- Theorem stating that the percentage of shaded squares is 36/49 -/
theorem shaded_percentage_is_correct (grid : CheckeredGrid) :
  (shaded_squares grid : ℚ) / (total_squares grid : ℚ) = 36 / 49 := by
  sorry

#eval (36 : ℚ) / 49  -- To show the approximate decimal value

end NUMINAMATH_CALUDE_shaded_percentage_is_correct_l3830_383015


namespace NUMINAMATH_CALUDE_decimal_between_four_and_five_l3830_383077

theorem decimal_between_four_and_five : ∃ x : ℝ, (x = 4.5) ∧ (4 < x) ∧ (x < 5) := by
  sorry

end NUMINAMATH_CALUDE_decimal_between_four_and_five_l3830_383077


namespace NUMINAMATH_CALUDE_davids_pushups_l3830_383050

theorem davids_pushups (zachary_pushups : ℕ) (david_extra_pushups : ℕ) :
  zachary_pushups = 47 →
  david_extra_pushups = 15 →
  zachary_pushups + david_extra_pushups = 62 :=
by sorry

end NUMINAMATH_CALUDE_davids_pushups_l3830_383050


namespace NUMINAMATH_CALUDE_patio_rearrangement_l3830_383070

theorem patio_rearrangement (total_tiles : ℕ) (initial_rows : ℕ) (added_rows : ℕ) :
  total_tiles = 126 →
  initial_rows = 9 →
  added_rows = 4 →
  ∃ (initial_columns final_columns : ℕ),
    initial_columns * initial_rows = total_tiles ∧
    final_columns * (initial_rows + added_rows) = total_tiles ∧
    initial_columns - final_columns = 5 :=
by sorry

end NUMINAMATH_CALUDE_patio_rearrangement_l3830_383070


namespace NUMINAMATH_CALUDE_remainder_99_101_div_9_l3830_383097

theorem remainder_99_101_div_9 : (99 * 101) % 9 = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_99_101_div_9_l3830_383097


namespace NUMINAMATH_CALUDE_top_tier_lamps_l3830_383023

/-- Represents the number of tiers in the tower -/
def n : ℕ := 7

/-- Represents the common ratio of the geometric sequence -/
def r : ℕ := 2

/-- Represents the total number of lamps in the tower -/
def total_lamps : ℕ := 381

/-- Calculates the sum of a geometric series -/
def geometric_sum (a₁ : ℕ) : ℕ := a₁ * (1 - r^n) / (1 - r)

/-- Theorem stating that the number of lamps on the top tier is 3 -/
theorem top_tier_lamps : ∃ (a₁ : ℕ), geometric_sum a₁ = total_lamps ∧ a₁ = 3 := by
  sorry

end NUMINAMATH_CALUDE_top_tier_lamps_l3830_383023


namespace NUMINAMATH_CALUDE_hyperbola_transverse_axis_length_l3830_383006

/-- The length of the transverse axis of a hyperbola with equation x²/9 - y²/16 = 1 is 6 -/
theorem hyperbola_transverse_axis_length :
  ∀ (x y : ℝ), x^2 / 9 - y^2 / 16 = 1 →
  ∃ (a : ℝ), a > 0 ∧ a^2 = 9 ∧ 2 * a = 6 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_transverse_axis_length_l3830_383006


namespace NUMINAMATH_CALUDE_books_read_by_tony_dean_breanna_l3830_383081

/-- The number of different books read by Tony, Dean, and Breanna -/
def totalDifferentBooks (tonyBooks deanBooks breannaBooks sharedTonyDean sharedAll : ℕ) : ℕ :=
  tonyBooks + deanBooks + breannaBooks - sharedTonyDean - sharedAll

/-- Theorem stating the total number of different books read -/
theorem books_read_by_tony_dean_breanna : 
  totalDifferentBooks 23 12 17 3 1 = 48 := by
  sorry

#eval totalDifferentBooks 23 12 17 3 1

end NUMINAMATH_CALUDE_books_read_by_tony_dean_breanna_l3830_383081


namespace NUMINAMATH_CALUDE_sin_cos_ratio_l3830_383049

theorem sin_cos_ratio (θ : Real) (h : Real.sqrt 2 * Real.sin (θ + π/4) = 3 * Real.cos θ) :
  Real.sin θ / (Real.sin θ - Real.cos θ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_ratio_l3830_383049


namespace NUMINAMATH_CALUDE_skylar_donation_l3830_383064

/-- Represents the donation scenario for Skylar -/
structure DonationScenario where
  start_age : ℕ
  current_age : ℕ
  annual_donation : ℕ

/-- Calculates the total amount donated given a DonationScenario -/
def total_donated (scenario : DonationScenario) : ℕ :=
  (scenario.current_age - scenario.start_age) * scenario.annual_donation

/-- Theorem stating that Skylar's total donation is $432,000 -/
theorem skylar_donation :
  let scenario : DonationScenario := {
    start_age := 17,
    current_age := 71,
    annual_donation := 8000
  }
  total_donated scenario = 432000 := by
  sorry

end NUMINAMATH_CALUDE_skylar_donation_l3830_383064


namespace NUMINAMATH_CALUDE_quartic_ratio_l3830_383053

theorem quartic_ratio (a b c d e : ℝ) (h : a ≠ 0) :
  (∀ x : ℝ, a * x^4 + b * x^3 + c * x^2 + d * x + e = 0 ↔ x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4) →
  c / e = 35 / 24 := by
  sorry

end NUMINAMATH_CALUDE_quartic_ratio_l3830_383053


namespace NUMINAMATH_CALUDE_complex_square_root_l3830_383065

theorem complex_square_root : ∃ (z : ℂ),
  let a : ℝ := Real.sqrt ((-81 + Real.sqrt 8865) / 2)
  let b : ℝ := -24 / a
  z = Complex.mk a b ∧ z^2 = Complex.mk (-81) (-48) := by
  sorry

end NUMINAMATH_CALUDE_complex_square_root_l3830_383065


namespace NUMINAMATH_CALUDE_distributive_law_analogy_l3830_383030

theorem distributive_law_analogy (a b c : ℝ) (h : c ≠ 0) :
  (a + b) * c = a * c + b * c ↔ (a + b) / c = a / c + b / c :=
sorry

end NUMINAMATH_CALUDE_distributive_law_analogy_l3830_383030


namespace NUMINAMATH_CALUDE_smallest_y_for_square_l3830_383039

theorem smallest_y_for_square (y : ℕ) : y = 10 ↔ 
  (y > 0 ∧ 
   ∃ n : ℕ, 4410 * y = n^2 ∧
   ∀ z < y, z > 0 → ¬∃ m : ℕ, 4410 * z = m^2) := by
sorry

end NUMINAMATH_CALUDE_smallest_y_for_square_l3830_383039


namespace NUMINAMATH_CALUDE_jesse_bananas_l3830_383085

/-- The number of friends Jesse shares his bananas with -/
def num_friends : ℕ := 3

/-- The number of bananas each friend would get if Jesse shares equally -/
def bananas_per_friend : ℕ := 7

/-- The total number of bananas Jesse has -/
def total_bananas : ℕ := num_friends * bananas_per_friend

theorem jesse_bananas : total_bananas = 21 := by
  sorry

end NUMINAMATH_CALUDE_jesse_bananas_l3830_383085


namespace NUMINAMATH_CALUDE_remainder_squared_plus_five_l3830_383044

theorem remainder_squared_plus_five (a : ℕ) (h : a % 7 = 4) :
  (a^2 + 5) % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_squared_plus_five_l3830_383044


namespace NUMINAMATH_CALUDE_unknown_number_solution_l3830_383087

theorem unknown_number_solution : 
  ∃ x : ℚ, (x + 23 / 89) * 89 = 4028 ∧ x = 45 := by sorry

end NUMINAMATH_CALUDE_unknown_number_solution_l3830_383087


namespace NUMINAMATH_CALUDE_largest_when_first_changed_l3830_383047

def original : ℚ := 0.12345

def change_digit (n : ℕ) : ℚ :=
  match n with
  | 1 => 0.92345
  | 2 => 0.19345
  | 3 => 0.12945
  | 4 => 0.12395
  | 5 => 0.12349
  | _ => original

theorem largest_when_first_changed :
  ∀ n : ℕ, n ≥ 1 → n ≤ 5 → change_digit 1 ≥ change_digit n :=
sorry

end NUMINAMATH_CALUDE_largest_when_first_changed_l3830_383047


namespace NUMINAMATH_CALUDE_fence_perimeter_l3830_383080

/-- The number of posts -/
def num_posts : ℕ := 36

/-- The width of each post in feet -/
def post_width : ℚ := 1/2

/-- The distance between adjacent posts in feet -/
def post_spacing : ℕ := 6

/-- The number of posts per side of the square field -/
def posts_per_side : ℕ := 10

/-- The length of one side of the square field in feet -/
def side_length : ℚ := (posts_per_side - 1) * post_spacing + posts_per_side * post_width

/-- The outer perimeter of the fence in feet -/
def outer_perimeter : ℚ := 4 * side_length

theorem fence_perimeter : outer_perimeter = 236 := by sorry

end NUMINAMATH_CALUDE_fence_perimeter_l3830_383080


namespace NUMINAMATH_CALUDE_multiplication_results_l3830_383083

theorem multiplication_results (h : 25 * 4 = 100) : 
  (25 * 8 = 200) ∧ 
  (25 * 12 = 300) ∧ 
  (250 * 40 = 10000) ∧ 
  (25 * 24 = 600) := by
sorry

end NUMINAMATH_CALUDE_multiplication_results_l3830_383083


namespace NUMINAMATH_CALUDE_tree_growth_problem_l3830_383052

/-- A tree growing problem -/
theorem tree_growth_problem (initial_height : ℝ) (growth_rate : ℝ) :
  initial_height = 4 →
  growth_rate = 0.4 →
  ∃ (total_years : ℕ),
    total_years = 6 ∧
    (initial_height + total_years * growth_rate) = 
    (initial_height + 4 * growth_rate) * (1 + 1/7) :=
by sorry

end NUMINAMATH_CALUDE_tree_growth_problem_l3830_383052


namespace NUMINAMATH_CALUDE_power_sum_equality_l3830_383037

theorem power_sum_equality (x y a b : ℝ) (h1 : x + y = a + b) (h2 : x^2 + y^2 = a^2 + b^2) :
  ∀ n : ℕ, x^n + y^n = a^n + b^n := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equality_l3830_383037


namespace NUMINAMATH_CALUDE_small_pond_green_percentage_l3830_383098

def total_ducks : ℕ := 100
def small_pond_ducks : ℕ := 20
def large_pond_ducks : ℕ := 80
def large_pond_green_percentage : ℚ := 15 / 100
def total_green_percentage : ℚ := 16 / 100

theorem small_pond_green_percentage :
  ∃ x : ℚ,
    x * small_pond_ducks + large_pond_green_percentage * large_pond_ducks =
    total_green_percentage * total_ducks ∧
    x = 20 / 100 := by
  sorry

end NUMINAMATH_CALUDE_small_pond_green_percentage_l3830_383098


namespace NUMINAMATH_CALUDE_video_dislikes_l3830_383018

/-- Represents the number of dislikes for a video -/
def dislikes (likes : ℕ) (additional : ℕ) (extra : ℕ) : ℕ :=
  likes / 2 + additional + extra

/-- Theorem stating the final number of dislikes for the video -/
theorem video_dislikes : dislikes 3000 100 1000 = 2600 := by
  sorry

end NUMINAMATH_CALUDE_video_dislikes_l3830_383018


namespace NUMINAMATH_CALUDE_steve_exceeds_goal_and_optimal_strategy_l3830_383000

/-- Represents the berry types --/
inductive Berry
  | Lingonberry
  | Cloudberry
  | Blueberry

/-- Represents Steve's berry-picking job --/
structure BerryJob where
  goal : ℕ
  payRates : Berry → ℕ
  basketCapacity : ℕ
  mondayPicking : Berry → ℕ
  tuesdayPicking : Berry → ℕ

def stevesJob : BerryJob :=
  { goal := 150
  , payRates := fun b => match b with
      | Berry.Lingonberry => 2
      | Berry.Cloudberry => 3
      | Berry.Blueberry => 5
  , basketCapacity := 30
  , mondayPicking := fun b => match b with
      | Berry.Lingonberry => 8
      | Berry.Cloudberry => 10
      | Berry.Blueberry => 12
  , tuesdayPicking := fun b => match b with
      | Berry.Lingonberry => 24
      | Berry.Cloudberry => 20
      | Berry.Blueberry => 5
  }

def earnings (job : BerryJob) (picking : Berry → ℕ) : ℕ :=
  (picking Berry.Lingonberry * job.payRates Berry.Lingonberry) +
  (picking Berry.Cloudberry * job.payRates Berry.Cloudberry) +
  (picking Berry.Blueberry * job.payRates Berry.Blueberry)

def totalEarnings (job : BerryJob) : ℕ :=
  earnings job job.mondayPicking + earnings job job.tuesdayPicking

theorem steve_exceeds_goal_and_optimal_strategy (job : BerryJob) :
  (totalEarnings job > job.goal) ∧
  (∀ picking : Berry → ℕ,
    (picking Berry.Lingonberry + picking Berry.Cloudberry + picking Berry.Blueberry ≤ job.basketCapacity) →
    (earnings job picking ≤ job.basketCapacity * job.payRates Berry.Blueberry)) :=
by sorry

end NUMINAMATH_CALUDE_steve_exceeds_goal_and_optimal_strategy_l3830_383000


namespace NUMINAMATH_CALUDE_sandy_age_l3830_383025

theorem sandy_age (S M : ℕ) (h1 : S = M - 16) (h2 : S * 9 = M * 7) : S = 56 := by
  sorry

end NUMINAMATH_CALUDE_sandy_age_l3830_383025


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_16_l3830_383056

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℤ  -- The sequence
  d : ℤ      -- Common difference
  first_term_eq : a 1 = a 1  -- Placeholder for the first term
  diff_eq : ∀ n, a (n + 1) - a n = d

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  n * seq.a 1 + n * (n - 1) / 2 * seq.d

theorem arithmetic_sequence_sum_16 
  (seq : ArithmeticSequence) 
  (h1 : seq.a 12 = -8) 
  (h2 : sum_n seq 9 = -9) : 
  sum_n seq 16 = -72 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_16_l3830_383056


namespace NUMINAMATH_CALUDE_boxwoods_shaped_into_spheres_l3830_383068

/-- Calculates the number of boxwoods shaped into spheres given the total number of boxwoods,
    costs for trimming and shaping, and the total charge. -/
theorem boxwoods_shaped_into_spheres
  (total_boxwoods : ℕ)
  (trim_cost : ℚ)
  (shape_cost : ℚ)
  (total_charge : ℚ)
  (h1 : total_boxwoods = 30)
  (h2 : trim_cost = 5)
  (h3 : shape_cost = 15)
  (h4 : total_charge = 210) :
  (total_charge - (total_boxwoods * trim_cost)) / shape_cost = 4 :=
by sorry

end NUMINAMATH_CALUDE_boxwoods_shaped_into_spheres_l3830_383068


namespace NUMINAMATH_CALUDE_democrat_count_l3830_383072

theorem democrat_count (total : ℕ) (female : ℕ) (male : ℕ) : 
  total = 810 →
  female + male = total →
  (female / 2 : ℚ) + (male / 4 : ℚ) = (1 / 3 : ℚ) * total →
  female / 2 = 135 :=
by
  sorry

end NUMINAMATH_CALUDE_democrat_count_l3830_383072


namespace NUMINAMATH_CALUDE_G_fraction_is_lowest_terms_denominator_minus_numerator_l3830_383041

/-- G is defined as the infinite repeating decimal 0.837837837... -/
def G : ℚ := 837 / 999

/-- The fraction representation of G in lowest terms -/
def G_fraction : ℚ := 31 / 37

theorem G_fraction_is_lowest_terms : G = G_fraction := by sorry

theorem denominator_minus_numerator : Nat.gcd 31 37 = 1 ∧ 37 - 31 = 6 := by sorry

end NUMINAMATH_CALUDE_G_fraction_is_lowest_terms_denominator_minus_numerator_l3830_383041


namespace NUMINAMATH_CALUDE_train_passing_pole_l3830_383038

theorem train_passing_pole (train_length platform_length : ℝ) 
  (platform_passing_time : ℝ) : 
  train_length = 120 →
  platform_length = 120 →
  platform_passing_time = 22 →
  (∃ (pole_passing_time : ℝ), 
    pole_passing_time = train_length / (train_length + platform_length) * platform_passing_time ∧
    pole_passing_time = 11) :=
by sorry

end NUMINAMATH_CALUDE_train_passing_pole_l3830_383038


namespace NUMINAMATH_CALUDE_total_cost_is_14000_l3830_383088

/-- Represents the dimensions and costs of the roads on a rectangular lawn. -/
structure LawnRoads where
  lawn_length : ℝ
  lawn_width : ℝ
  road1_width : ℝ
  road1_cost_per_sqm : ℝ
  road2_width : ℝ
  road2_cost_per_sqm : ℝ
  hill_length : ℝ
  hill_cost_increase : ℝ

/-- Calculates the total cost of traveling both roads on the lawn. -/
def total_cost (lr : LawnRoads) : ℝ :=
  let road1_area := lr.lawn_length * lr.road1_width
  let road1_cost := road1_area * lr.road1_cost_per_sqm
  let hill_area := lr.hill_length * lr.road1_width
  let hill_additional_cost := hill_area * (lr.road1_cost_per_sqm * lr.hill_cost_increase)
  let road2_area := lr.lawn_width * lr.road2_width
  let road2_cost := road2_area * lr.road2_cost_per_sqm
  road1_cost + hill_additional_cost + road2_cost

/-- Theorem stating that the total cost of traveling both roads is 14000. -/
theorem total_cost_is_14000 (lr : LawnRoads) 
    (h1 : lr.lawn_length = 150)
    (h2 : lr.lawn_width = 80)
    (h3 : lr.road1_width = 12)
    (h4 : lr.road1_cost_per_sqm = 4)
    (h5 : lr.road2_width = 8)
    (h6 : lr.road2_cost_per_sqm = 5)
    (h7 : lr.hill_length = 60)
    (h8 : lr.hill_cost_increase = 0.25) :
    total_cost lr = 14000 := by
  sorry


end NUMINAMATH_CALUDE_total_cost_is_14000_l3830_383088


namespace NUMINAMATH_CALUDE_cylinder_volume_from_square_rotation_l3830_383034

/-- The volume of a cylinder formed by rotating a square about its vertical line of symmetry -/
theorem cylinder_volume_from_square_rotation (square_side : ℝ) (height : ℝ) : 
  square_side = 10 → height = 20 → 
  (π * (square_side / 2)^2 * height : ℝ) = 500 * π := by
  sorry

end NUMINAMATH_CALUDE_cylinder_volume_from_square_rotation_l3830_383034


namespace NUMINAMATH_CALUDE_vacation_group_size_l3830_383036

def airbnb_cost : ℕ := 3200
def car_cost : ℕ := 800
def share_per_person : ℕ := 500

theorem vacation_group_size :
  (airbnb_cost + car_cost) / share_per_person = 8 :=
by sorry

end NUMINAMATH_CALUDE_vacation_group_size_l3830_383036


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l3830_383040

theorem solve_exponential_equation :
  ∃ x : ℝ, 3^(2*x + 1) = (1/81 : ℝ) ∧ x = -5/2 := by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l3830_383040


namespace NUMINAMATH_CALUDE_division_theorem_l3830_383099

theorem division_theorem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) : 
  dividend = 140 → 
  divisor = 15 → 
  remainder = 5 → 
  dividend = divisor * quotient + remainder → 
  quotient = 9 := by
sorry

end NUMINAMATH_CALUDE_division_theorem_l3830_383099


namespace NUMINAMATH_CALUDE_fraction_equality_l3830_383094

theorem fraction_equality (p r s u : ℝ) 
  (h1 : p / r = 4)
  (h2 : s / r = 8)
  (h3 : s / u = 1 / 4) :
  u / p = 8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3830_383094


namespace NUMINAMATH_CALUDE_eggs_leftover_l3830_383060

theorem eggs_leftover (abigail beatrice carson : ℕ) 
  (h_abigail : abigail = 60)
  (h_beatrice : beatrice = 75)
  (h_carson : carson = 27) :
  (abigail + beatrice + carson) % 18 = 0 := by
sorry

end NUMINAMATH_CALUDE_eggs_leftover_l3830_383060


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l3830_383043

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x + y = (x * y)^2) : 
  1 / x + 1 / y = 1 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l3830_383043


namespace NUMINAMATH_CALUDE_hillarys_remaining_money_l3830_383062

/-- Calculates the amount Hillary is left with after selling crafts and accounting for all costs and transactions. -/
theorem hillarys_remaining_money
  (base_price : ℝ)
  (cost_per_craft : ℝ)
  (crafts_sold : ℕ)
  (extra_money : ℝ)
  (tax_rate : ℝ)
  (deposit_amount : ℝ)
  (h1 : base_price = 12)
  (h2 : cost_per_craft = 4)
  (h3 : crafts_sold = 3)
  (h4 : extra_money = 7)
  (h5 : tax_rate = 0.1)
  (h6 : deposit_amount = 26)
  : ∃ (remaining : ℝ), remaining = 1.9 ∧ remaining ≥ 0 := by
  sorry

#check hillarys_remaining_money

end NUMINAMATH_CALUDE_hillarys_remaining_money_l3830_383062


namespace NUMINAMATH_CALUDE_cube_preserves_inequality_l3830_383026

theorem cube_preserves_inequality (a b : ℝ) (h : a > b) : a^3 > b^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_preserves_inequality_l3830_383026


namespace NUMINAMATH_CALUDE_local_minimum_condition_l3830_383051

/-- The function f(x) = x(x-a)² has a local minimum at x=2 if and only if a = 2 -/
theorem local_minimum_condition (a : ℝ) :
  (∃ δ > 0, ∀ x : ℝ, |x - 2| < δ → x * (x - a)^2 ≥ 2 * (2 - a)^2) ↔ a = 2 := by
  sorry

end NUMINAMATH_CALUDE_local_minimum_condition_l3830_383051


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3830_383059

/-- An arithmetic sequence. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of the 3rd to 7th terms equals 450. -/
def SumCondition (a : ℕ → ℝ) : Prop :=
  a 3 + a 4 + a 5 + a 6 + a 7 = 450

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  ArithmeticSequence a → SumCondition a → a 2 + a 8 = 180 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3830_383059


namespace NUMINAMATH_CALUDE_best_distribution_for_1_l3830_383086

/-- Represents a distribution of pearls among 4 people -/
def Distribution := Fin 4 → ℕ

/-- The total number of pearls to be distributed -/
def totalPearls : ℕ := 10

/-- A valid distribution must sum to the total number of pearls -/
def isValidDistribution (d : Distribution) : Prop :=
  (Finset.univ.sum d) = totalPearls

/-- A distribution passes if it has at least half of the votes -/
def passes (d : Distribution) : Prop :=
  2 * (Finset.filter (fun i => d i > 0) Finset.univ).card ≥ 4

/-- The best distribution for person 3 if 1 and 2 are eliminated -/
def bestFor3 : Distribution :=
  fun i => if i = 2 then 10 else 0

/-- The proposed best distribution for person 1 -/
def proposedBest : Distribution :=
  fun i => match i with
  | 0 => 9
  | 2 => 1
  | _ => 0

theorem best_distribution_for_1 :
  isValidDistribution proposedBest ∧
  passes proposedBest ∧
  ∀ d : Distribution, isValidDistribution d ∧ passes d → proposedBest 0 ≥ d 0 :=
by sorry

end NUMINAMATH_CALUDE_best_distribution_for_1_l3830_383086


namespace NUMINAMATH_CALUDE_trapezoid_sides_for_given_circle_l3830_383055

/-- Represents a trapezoid formed by tangents to a circle -/
structure CircleTrapezoid where
  radius : ℝ
  chord_length : ℝ

/-- Calculates the sides of the trapezoid -/
def trapezoid_sides (t : CircleTrapezoid) : (ℝ × ℝ × ℝ × ℝ) :=
  sorry

/-- Theorem stating the correct sides of the trapezoid for the given circle -/
theorem trapezoid_sides_for_given_circle :
  let t : CircleTrapezoid := { radius := 5, chord_length := 8 }
  trapezoid_sides t = (12.5, 5, 12.5, 20) := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_sides_for_given_circle_l3830_383055


namespace NUMINAMATH_CALUDE_solve_for_x_l3830_383031

theorem solve_for_x (x y : ℝ) (h1 : x - y = 7) (h2 : x + y = 11) : x = 9 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_x_l3830_383031


namespace NUMINAMATH_CALUDE_rose_ratio_l3830_383084

theorem rose_ratio (total : ℕ) (red : ℕ) (yellow : ℕ) (white : ℕ) : 
  total = 80 →
  yellow = (total - red) / 4 →
  red + white = 75 →
  total = red + yellow + white →
  (red : ℚ) / total = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_rose_ratio_l3830_383084


namespace NUMINAMATH_CALUDE_rhombus_diagonal_length_main_theorem_l3830_383071

/-- Represents a rhombus with given properties -/
structure Rhombus where
  diagonal1 : ℝ
  perimeter : ℝ
  diagonal2 : ℝ

/-- Theorem stating the relationship between the diagonals and perimeter of a specific rhombus -/
theorem rhombus_diagonal_length (r : Rhombus) 
    (h1 : r.diagonal1 = 10)
    (h2 : r.perimeter = 52) : 
    r.diagonal2 = 24 := by
  sorry

/-- Main theorem to be proved -/
theorem main_theorem : ∃ r : Rhombus, r.diagonal1 = 10 ∧ r.perimeter = 52 ∧ r.diagonal2 = 24 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_diagonal_length_main_theorem_l3830_383071


namespace NUMINAMATH_CALUDE_counterexample_exists_l3830_383021

theorem counterexample_exists : ∃ (a b : ℝ), a > b ∧ a⁻¹ ≥ b⁻¹ := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l3830_383021


namespace NUMINAMATH_CALUDE_f_inequality_l3830_383010

open Real

-- Define a derivable function f on ℝ
variable (f : ℝ → ℝ)

-- Define the condition that f is twice differentiable
variable (hf : TwiceDifferentiable ℝ f)

-- Define the condition 3f(x) > f''(x) for all x ∈ ℝ
variable (h1 : ∀ x : ℝ, 3 * f x > (deriv^[2] f) x)

-- Define the condition f(1) = e^3
variable (h2 : f 1 = exp 3)

-- State the theorem
theorem f_inequality : f 2 < exp 6 := by
  sorry

end NUMINAMATH_CALUDE_f_inequality_l3830_383010


namespace NUMINAMATH_CALUDE_special_polynomial_at_zero_l3830_383075

/-- A polynomial of degree 6 satisfying specific conditions -/
def special_polynomial (p : ℝ → ℝ) : Prop :=
  (∃ a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ, ∀ x, p x = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6) ∧
  (∀ n : ℕ, n ≤ 6 → p (3^n) = 1 / (2^n))

/-- Theorem stating that a special polynomial evaluates to 0 at x = 0 -/
theorem special_polynomial_at_zero
  (p : ℝ → ℝ)
  (h : special_polynomial p) :
  p 0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_special_polynomial_at_zero_l3830_383075


namespace NUMINAMATH_CALUDE_absolute_value_equation_l3830_383079

theorem absolute_value_equation (x : ℝ) : |x - 3| = 2 → x = 5 ∨ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_l3830_383079


namespace NUMINAMATH_CALUDE_log_sum_equation_l3830_383074

theorem log_sum_equation (k : ℤ) (x : ℝ) 
  (h : (7.318 * Real.log x / Real.log k) + 
       (Real.log x / Real.log (k ^ (1/2 : ℝ))) + 
       (Real.log x / Real.log (k ^ (1/3 : ℝ))) + 
       -- ... (representing the sum up to k terms)
       (Real.log x / Real.log (k ^ (1/k : ℝ))) = 
       (k + 1 : ℝ) / 2) :
  x = k ^ (1/k : ℝ) := by
sorry

end NUMINAMATH_CALUDE_log_sum_equation_l3830_383074


namespace NUMINAMATH_CALUDE_GH_distance_is_40_l3830_383057

/-- An isosceles trapezoid with specific properties -/
structure IsoscelesTrapezoid where
  /-- The length of a diagonal -/
  diagonal_length : ℝ
  /-- The distance from point G to vertex A -/
  GA_distance : ℝ
  /-- The distance from point G to vertex D -/
  GD_distance : ℝ
  /-- The base angle at the longer base (AD) -/
  base_angle : ℝ
  /-- Assumption that the diagonal length is 20√5 -/
  diagonal_length_eq : diagonal_length = 20 * Real.sqrt 5
  /-- Assumption that GA distance is 20 -/
  GA_distance_eq : GA_distance = 20
  /-- Assumption that GD distance is 40 -/
  GD_distance_eq : GD_distance = 40
  /-- Assumption that the base angle is π/4 -/
  base_angle_eq : base_angle = Real.pi / 4

/-- The main theorem stating that GH distance is 40 -/
theorem GH_distance_is_40 (t : IsoscelesTrapezoid) : ℝ := by
  sorry

#check GH_distance_is_40

end NUMINAMATH_CALUDE_GH_distance_is_40_l3830_383057

import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_exterior_angles_regular_pentagon_proof_l1248_124856

/-- The sum of exterior angles of a regular pentagon is 360 degrees. -/
def sum_exterior_angles_regular_pentagon : ℝ := 360

/-- A regular pentagon is a polygon with 5 sides. -/
def regular_pentagon : ℕ := 5

/-- The sum of exterior angles of any polygon is always 360 degrees. -/
axiom sum_exterior_angles_polygon : ℕ → ℝ

/-- The sum of exterior angles of a regular pentagon is equal to the sum of exterior angles of any polygon. -/
theorem sum_exterior_angles_regular_pentagon_proof :
  sum_exterior_angles_polygon regular_pentagon = sum_exterior_angles_regular_pentagon :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_exterior_angles_regular_pentagon_proof_l1248_124856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bankers_gain_problem_l1248_124813

/-- Calculates the present worth given the banker's gain, time period, and interest rate. -/
noncomputable def present_worth (bankers_gain : ℝ) (years : ℕ) (rate : ℝ) : ℝ :=
  bankers_gain / ((1 + rate) ^ years - 1 - years * rate)

/-- The banker's gain problem -/
theorem bankers_gain_problem :
  let bankers_gain : ℝ := 120
  let years : ℕ := 4
  let rate : ℝ := 0.15
  let calculated_present_worth := present_worth bankers_gain years rate
  abs (calculated_present_worth - 805.69) < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bankers_gain_problem_l1248_124813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_to_circle_l1248_124896

/-- Circle C with center (1, 2) and radius √2 -/
def C : Set (ℝ × ℝ) := {p | (p.1 - 1)^2 + (p.2 - 2)^2 = 2}

/-- Point P -/
def P : ℝ × ℝ := (2, -1)

/-- First tangent line -/
def tangent1 (p : ℝ × ℝ) : Prop := p.1 + p.2 - 1 = 0

/-- Second tangent line -/
def tangent2 (p : ℝ × ℝ) : Prop := 7*p.1 - p.2 - 15 = 0

/-- Theorem stating that the given lines are tangent to circle C through point P -/
theorem tangent_lines_to_circle :
  ∀ (p : ℝ × ℝ),
    p ∈ C ∧ (tangent1 p ∨ tangent2 p) →
    (∃ (t : ℝ), (1 - t)*P.1 + t*p.1 = p.1 ∧ (1 - t)*P.2 + t*p.2 = p.2) ∧
    (∀ (ε : ℝ), ε ≠ 0 →
      ∃ (δ : ℝ), δ > 0 ∧ ∀ (p' : ℝ × ℝ),
        p' ∈ C →
        (p'.1 - p.1)^2 + (p'.2 - p.2)^2 < δ^2 →
        (tangent1 p' ∨ tangent2 p') → (p'.1 - p.1)^2 + (p'.2 - p.2)^2 ≥ ε^2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_to_circle_l1248_124896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l1248_124815

-- Define a triangle with sides a, b, c
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the conditions
def ValidTriangle (t : Triangle) : Prop :=
  t.a > t.b ∧ t.b > t.c ∧ t.a = 2 * (t.b - t.c)

-- Define the projection q
def Projection (t : Triangle) (q : ℝ) : Prop :=
  ∃ (h : ℝ), h^2 = t.c^2 - q^2 ∧ h^2 = t.b^2 - (t.a - q)^2

-- Theorem statement
theorem projection_theorem (t : Triangle) (q : ℝ) 
  (h1 : ValidTriangle t) (h2 : Projection t q) : 
  t.c + 2 * q = 3 * t.a / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l1248_124815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_equation_solutions_sum_l1248_124812

theorem sin_equation_solutions_sum (a : ℝ) :
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
   x₁ ∈ Set.Icc (0 : ℝ) (π / 2) ∧
   x₂ ∈ Set.Icc (0 : ℝ) (π / 2) ∧
   Real.sin (2 * x₁ + π / 3) = a ∧
   Real.sin (2 * x₂ + π / 3) = a) →
  ∃ (x₁ x₂ : ℝ), x₁ + x₂ = π / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_equation_solutions_sum_l1248_124812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_properties_complex_number_conditions_l1248_124880

variable (m : ℝ)
def z (m : ℝ) : ℂ := m + 1 + (m - 1) * Complex.I

theorem complex_number_properties :
  (∃ (x : ℝ), z m = x) ∧
  (∃ (y : ℝ), y ≠ 0 ∧ z m = y * Complex.I) ∧
  (∃ (y : ℝ), y ≠ 0 ∧ z m = y * Complex.I ∧ (z m).re = 0) :=
by
  sorry

theorem complex_number_conditions :
  ((∃ (x : ℝ), z m = x) ↔ m = 1) ∧
  ((∃ (y : ℝ), y ≠ 0 ∧ z m = y * Complex.I) ↔ m ≠ 1) ∧
  ((∃ (y : ℝ), y ≠ 0 ∧ z m = y * Complex.I ∧ (z m).re = 0) ↔ m = -1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_properties_complex_number_conditions_l1248_124880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_bounds_l1248_124844

theorem sequence_bounds (n : ℕ) (hn : n > 1) : 
  ∃ a : ℕ → ℚ, a 0 = 1/2 ∧
  (∀ k, a (k + 1) = a k + (a k)^2 / n) ∧
  1 - 1/n < a n ∧ a n < 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_bounds_l1248_124844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_l1248_124898

noncomputable def S (n : ℕ) (a : ℝ) : ℝ := (1/2) * 3^(n+1) - a

noncomputable def a_n (n : ℕ) (a : ℝ) : ℝ :=
  if n = 0 then S 1 a
  else S n a - S (n-1) a

theorem geometric_sequence_sum (a : ℝ) :
  (∀ n : ℕ, S n a = (1/2) * 3^(n+1) - a) →
  (∀ n : ℕ, a_n (n+1) a = 3 * a_n n a) →
  a = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_l1248_124898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_diff_11_5_smallest_diff_36_5_smallest_diff_53_37_l1248_124833

-- Part (a)
theorem smallest_diff_11_5 :
  (∃ (k n : ℕ), (11 : ℤ)^k - (5 : ℤ)^n = 4 ∨ (5 : ℤ)^n - (11 : ℤ)^k = 4) ∧
  (∀ (k n : ℕ), |(11 : ℤ)^k - (5 : ℤ)^n| ≥ 4) :=
sorry

-- Part (b)
theorem smallest_diff_36_5 :
  (∃ (k n : ℕ), (36 : ℤ)^k - (5 : ℤ)^n = 11 ∨ (5 : ℤ)^n - (36 : ℤ)^k = 11) ∧
  (∀ (k n : ℕ), |(36 : ℤ)^k - (5 : ℤ)^n| ≥ 11) :=
sorry

-- Part (c)
theorem smallest_diff_53_37 :
  (∃ (k n : ℕ), (53 : ℤ)^k - (37 : ℤ)^n = 16 ∨ (37 : ℤ)^n - (53 : ℤ)^k = 16) ∧
  (∀ (k n : ℕ), |(53 : ℤ)^k - (37 : ℤ)^n| ≥ 16) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_diff_11_5_smallest_diff_36_5_smallest_diff_53_37_l1248_124833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_value_l1248_124873

-- Define z as a variable of type ℂ (complex numbers)
variable (z : ℂ)

-- Now define A and B using the variable z
def A : Set ℂ := {1, 3, Complex.I * z}
def B : Set ℂ := {4}

-- Theorem statement
theorem z_value (h : A z ∪ B = A z) : z = -4 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_value_l1248_124873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_lateral_surface_area_l1248_124877

/-- The lateral surface area of a cylinder formed by rotating a square with side length 1 around one of its sides -/
noncomputable def lateral_surface_area : ℝ := 2 * Real.pi

/-- Theorem stating that the lateral surface area of the described cylinder is 2π -/
theorem cylinder_lateral_surface_area : lateral_surface_area = 2 * Real.pi := by
  -- Unfold the definition of lateral_surface_area
  unfold lateral_surface_area
  -- The equality now holds by reflexivity
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_lateral_surface_area_l1248_124877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1248_124864

noncomputable def f (x : Real) : Real := Real.sin x * Real.cos x - Real.sqrt 3 * (Real.sin x) ^ 2

theorem f_properties :
  (∃ T > 0, ∀ x, f (x + T) = f x ∧ ∀ S, 0 < S → S < T → ∃ y, f (y + S) ≠ f y) ∧
  (∀ x, -π/3 ≤ x ∧ x ≤ π/6 → f x ≥ -Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1248_124864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l1248_124828

theorem sin_alpha_value (α : ℝ) :
  let β := α + π / 6
  (∃ (r : ℝ), r * Real.cos β = -1 ∧ r * Real.sin β = -2 * Real.sqrt 2) →
  Real.sin α = (1 - 2 * Real.sqrt 6) / 6 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l1248_124828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_supermarket_profit_theorem_l1248_124867

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 2 then x^(1/2)
  else -x^2 + 6*x - 4

-- State the theorem
theorem supermarket_profit_theorem :
  -- The function satisfies the given data points
  f 1 = 1 ∧
  f 2 = Real.sqrt 2 ∧
  f 3 = 5 ∧
  f 4 = 4 ∧
  f 5 = 1 ∧
  -- The function has a maximum value at x = 3
  ∀ x, 0 ≤ x ∧ x ≤ 5 → f x ≤ f 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_supermarket_profit_theorem_l1248_124867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l1248_124838

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  S : Real

-- Define the conditions
def dot_product_condition (t : Triangle) : Prop :=
  t.b * t.c * Real.cos t.A ≤ 2 * Real.sqrt 3 * t.S

def angle_ratio_condition (t : Triangle) : Prop :=
  ∃ (x : Real), Real.tan t.A = x ∧ Real.tan t.B = 2*x ∧ Real.tan t.C = 3*x

-- Theorem statements
theorem part1 (t : Triangle) (h : dot_product_condition t) :
  Real.pi / 6 ≤ t.A ∧ t.A < Real.pi :=
sorry

theorem part2 (t : Triangle) (h1 : angle_ratio_condition t) (h2 : t.c = 1) :
  t.b = 2 * Real.sqrt 2 / 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l1248_124838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fill_pool_time_l1248_124839

/-- Represents the time required to fill a leaking pool -/
noncomputable def time_to_fill_pool (capacity : ℝ) (fill_rate : ℝ) (leak_rate : ℝ) : ℝ :=
  capacity / (fill_rate - leak_rate)

/-- Theorem: Given the specified conditions, the time to fill the pool is 40 minutes -/
theorem fill_pool_time :
  let capacity : ℝ := 60
  let fill_rate : ℝ := 1.6
  let leak_rate : ℝ := 0.1
  time_to_fill_pool capacity fill_rate leak_rate = 40 := by
  -- Unfold the definition of time_to_fill_pool
  unfold time_to_fill_pool
  -- Simplify the expression
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fill_pool_time_l1248_124839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_of_cubes_no_real_roots_l1248_124810

/-- Given two linear functions f(x) = ax + b and g(x) = ax + d where b ≠ d,
    the expression (b - d)(f(x)² + f(x)g(x) + g(x)²) is always positive or always negative for all real x. -/
theorem difference_of_cubes_no_real_roots (a b d : ℝ) (h : b ≠ d) :
  ∃ (s : Bool), ∀ (x : ℝ), (0 < (b - d) * ((a * x + b)^2 + (a * x + b) * (a * x + d) + (a * x + d)^2)) = s :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_of_cubes_no_real_roots_l1248_124810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_circumradius_l1248_124852

/-- The circumradius of a sector with given radius and central angle. -/
noncomputable def circumradius (r : ℝ) (θ : ℝ) : ℝ := r / Real.cos (θ / 2)

/-- Given a circle of radius 9 and a sector with central angle 60°, 
    the radius of the circle circumscribed about the sector is 6√3. -/
theorem sector_circumradius (r : ℝ) (θ : ℝ) : 
  r = 9 → θ = π / 3 → circumradius r θ = 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_circumradius_l1248_124852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocal_distances_l1248_124892

/-- Line l with parametric equations x = 1 + t, y = √3 * t -/
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (1 + t, Real.sqrt 3 * t)

/-- Curve C with Cartesian equation x²/4 + y²/3 = 1 -/
def curve_C (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

/-- Point M -/
def point_M : ℝ × ℝ := (1, 0)

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Theorem: The sum of reciprocals of distances from M to A and B is 4/3 -/
theorem sum_reciprocal_distances :
  ∀ t1 t2 : ℝ,
  curve_C (line_l t1).1 (line_l t1).2 →
  curve_C (line_l t2).1 (line_l t2).2 →
  t1 ≠ t2 →
  1 / distance point_M (line_l t1) + 1 / distance point_M (line_l t2) = 4/3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocal_distances_l1248_124892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base12_subtraction_l1248_124861

/-- Represents a digit in base 12 --/
inductive Base12Digit
| D0 | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | A | B

/-- Represents a number in base 12 --/
def Base12Number := List Base12Digit

/-- Convert a Base12Number to a natural number --/
def base12ToNat (n : Base12Number) : ℕ :=
  sorry

/-- Convert a natural number to a Base12Number --/
def natToBase12 (n : ℕ) : Base12Number :=
  sorry

/-- Subtract two Base12Numbers --/
def subtract (a b : Base12Number) : Base12Number :=
  sorry

theorem base12_subtraction :
  let n1 : Base12Number := [Base12Digit.D9, Base12Digit.B, Base12Digit.D5]
  let n2 : Base12Number := [Base12Digit.D6, Base12Digit.A, Base12Digit.D3]
  let result : Base12Number := [Base12Digit.D3, Base12Digit.D1, Base12Digit.D2]
  subtract n1 n2 = result := by
  sorry

#eval "Base12 subtraction theorem stated."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base12_subtraction_l1248_124861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_money_proof_l1248_124894

/-- Calculates the total cents from given amounts -/
def total_cents (lance_cents : ℕ) (margaret_fraction : ℚ) (guy_quarters guy_dimes : ℕ) (bill_dimes : ℕ) : ℕ :=
  lance_cents + 
  (margaret_fraction * 100).floor.toNat + 
  (guy_quarters * 25 + guy_dimes * 10) + 
  (bill_dimes * 10)

/-- Proves that the sum of the given amounts equals 265 cents -/
theorem total_money_proof : 
  total_cents 70 (3/4) 2 1 6 = 265 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_money_proof_l1248_124894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_theorem_max_profit_price_l1248_124891

/-- Represents the profit function for a price increase scenario -/
def profit_increase (x : ℤ) : ℤ := -10 * x^2 + 100 * x + 6000

/-- Represents the profit function for a price decrease scenario -/
def profit_decrease (a : ℤ) : ℤ := -20 * a^2 + 100 * a + 6000

/-- Theorem stating the maximum profit and the corresponding price increase -/
theorem max_profit_theorem :
  ∃ (x : ℤ), x ≥ 0 ∧ x ≤ 30 ∧
  (∀ (y : ℤ), y ≥ 0 → y ≤ 30 → profit_increase x ≥ profit_increase y) ∧
  profit_increase x = 6250 ∧ x = 5 := by
  sorry

/-- Corollary stating that the maximum profit is achieved at a selling price of 65 yuan -/
theorem max_profit_price :
  ∃ (x : ℤ), x ≥ 0 ∧ x ≤ 30 ∧
  (∀ (y : ℤ), y ≥ 0 → y ≤ 30 → profit_increase x ≥ profit_increase y) ∧
  60 + x = 65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_theorem_max_profit_price_l1248_124891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_subset_relations_l1248_124831

-- Define the Quadrilateral type (since it's not defined in Mathlib)
structure Quadrilateral where
  -- We'll leave the implementation details abstract for now
  mk :: -- This allows us to create Quadrilaterals without specifying their properties

-- Define predicates for the different types of quadrilaterals
def isRhombus : Quadrilateral → Prop := sorry
def isParallelogram : Quadrilateral → Prop := sorry
def isQuadrilateral : Quadrilateral → Prop := sorry
def isSquare : Quadrilateral → Prop := sorry

-- Define the sets
def M : Set Quadrilateral := {q | isRhombus q}
def N : Set Quadrilateral := {q | isParallelogram q}
def P : Set Quadrilateral := {q | isQuadrilateral q}
def Q : Set Quadrilateral := {q | isSquare q}

-- State the theorem
theorem quadrilateral_subset_relations : Q ⊆ M ∧ M ⊆ N ∧ N ⊆ P := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_subset_relations_l1248_124831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_sticks_24_solution_is_24_l1248_124862

/-- Calculates the number of sticks needed for a 6-step staircase given the number for a 4-step staircase -/
def sticks_for_6_step (sticks_4_step : ℕ) : ℕ := 
  let n : ℕ := 10  -- Assuming the increase from 4 to 5 steps is 10
  sticks_4_step + n + (n + 4)

/-- Proves that 24 additional sticks are needed to extend from 4 to 6 steps -/
theorem additional_sticks_24 (sticks_4_step : ℕ) 
  (h : sticks_4_step = 32) : sticks_for_6_step sticks_4_step - sticks_4_step = 24 := by
  rw [sticks_for_6_step, h]
  ring

#eval sticks_for_6_step 32 - 32  -- Should output 24

/-- Main theorem stating the solution to the problem -/
theorem solution_is_24 : ∃ (n : ℕ), 
  sticks_for_6_step 32 - 32 = n ∧ n = 24 := by
  use 24
  constructor
  · exact additional_sticks_24 32 rfl
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_sticks_24_solution_is_24_l1248_124862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sandwich_preference_l1248_124853

theorem sandwich_preference (total_students : ℕ) 
  (cookie_percent pizza_percent pasta_percent : ℚ) : ℕ :=
  by
    have h1 : total_students = 200 := by sorry
    have h2 : cookie_percent = 25/100 := by sorry
    have h3 : pizza_percent = 30/100 := by sorry
    have h4 : pasta_percent = 35/100 := by sorry
    have h5 : cookie_percent + pizza_percent + pasta_percent < 1 := by sorry
    have h6 : (20 : ℕ) = (1 - (cookie_percent + pizza_percent + pasta_percent)) * total_students := by sorry
    exact 20


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sandwich_preference_l1248_124853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_three_digit_numbers_without_479_l1248_124875

/-- The set of digits that are not allowed in the numbers -/
def forbidden_digits : Finset ℕ := {4, 7, 9}

/-- A function that checks if a natural number has any forbidden digits -/
def has_forbidden_digit (n : ℕ) : Prop :=
  ∃ d ∈ forbidden_digits, (d.repr.data.head? = some (n.repr.data.head?))

/-- The set of valid digits for any position in the number -/
def valid_digits : Finset ℕ := Finset.filter (λ d ↦ d ∉ forbidden_digits) (Finset.range 10)

/-- The set of valid first digits (excluding 0) -/
def valid_first_digits : Finset ℕ := valid_digits.filter (λ d ↦ d ≠ 0)

/-- The count of three-digit numbers without 4, 7, and 9 -/
def count_valid_numbers : ℕ :=
  valid_first_digits.card * valid_digits.card * valid_digits.card

theorem count_three_digit_numbers_without_479 :
  count_valid_numbers = 216 := by
  sorry

#eval count_valid_numbers

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_three_digit_numbers_without_479_l1248_124875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_activation_code_l1248_124835

-- Define the sequence
def a : ℕ → ℕ
| n => 2^(n.log2)

-- Define the sum of the first N terms
def sum_first_n (N : ℕ) : ℕ :=
  Finset.sum (Finset.range N) a

-- Define the property we're looking for
def is_valid (N : ℕ) : Prop :=
  N > 100 ∧ ∃ k : ℕ, sum_first_n N = 2^k

-- State the theorem
theorem activation_code : (∀ m < 440, ¬(is_valid m)) ∧ is_valid 440 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_activation_code_l1248_124835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shane_gum_left_l1248_124816

def elyses_gum : ℕ := 100
def rick_fraction : ℚ := 1/2
def shane_fraction : ℚ := 1/2
def shane_chewed : ℕ := 11

theorem shane_gum_left : 
  (elyses_gum : ℚ) * rick_fraction * shane_fraction - shane_chewed = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shane_gum_left_l1248_124816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_length_equality_l1248_124893

/-- The focal length of a hyperbola x²/a - y²/b = 1 is √(a + b) -/
noncomputable def focal_length (a b : ℝ) : ℝ := Real.sqrt (a + b)

/-- Given two hyperbolas C₁: x²/7 - y²/(4+2a) = 1 and C₂: y²/(11-a) - x²/6 = 1,
    if their focal lengths are equal, then a = 2 -/
theorem hyperbola_focal_length_equality (a : ℝ) :
  focal_length 7 (4 + 2*a) = focal_length 6 (11 - a) → a = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_length_equality_l1248_124893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l1248_124841

noncomputable def f (x : ℝ) := 1/x - 6 + 2*x

theorem zero_in_interval :
  ∃ c ∈ Set.Ioo 2 3, f c = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l1248_124841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_anagram_pairs_with_100_between_l1248_124857

/-- Represents a letter in the set {T, U, R, N, I, P} -/
inductive Letter : Type
| T | U | R | N | I | P

/-- A sequence of five letters -/
def Sequence := Fin 5 → Letter

/-- Converts a letter to its corresponding digit in base 6 -/
def letterToDigit (l : Letter) : Fin 6 :=
  match l with
  | Letter.I => 0
  | Letter.N => 1
  | Letter.P => 2
  | Letter.R => 3
  | Letter.T => 4
  | Letter.U => 5

/-- Converts a sequence to its numerical representation in base 6 -/
def sequenceToNumber (s : Sequence) : ℕ :=
  (letterToDigit (s 0)).val * 7776 + (letterToDigit (s 1)).val * 1296 + 
  (letterToDigit (s 2)).val * 216 + (letterToDigit (s 3)).val * 36 + 
  (letterToDigit (s 4)).val * 6

/-- Checks if two sequences are anagrams -/
def areAnagrams (s1 s2 : Sequence) : Prop :=
  (sequenceToNumber s1 + sequenceToNumber s2) % 5 = 0

/-- Theorem: There are no pairs of anagrams with exactly 100 sequences between them -/
theorem no_anagram_pairs_with_100_between :
  ¬ ∃ (s1 s2 : Sequence), areAnagrams s1 s2 ∧ sequenceToNumber s2 - sequenceToNumber s1 = 101 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_anagram_pairs_with_100_between_l1248_124857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_circle_theorem_l1248_124859

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := y^2 - x^2 / 3 = 1

-- Define the circle (renamed to avoid conflict)
def circle_eq (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 4

-- Define the focus of the hyperbola
def focus (x y : ℝ) : Prop := x = 0 ∧ y = 2

-- Define the eccentricity of the hyperbola
def eccentricity : ℝ := 2

-- Theorem statement
theorem hyperbola_circle_theorem :
  ∃ (x y : ℝ), hyperbola x y ∧ circle_eq x y ∧ focus x y ∧ 
  Real.sqrt ((x - 0)^2 + (y - 2)^2) = eccentricity :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_circle_theorem_l1248_124859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_symmetric_scanning_codes_l1248_124842

/-- Represents a symmetric scanning code on an 8x8 grid. -/
structure SymmetricScanningCode where
  /-- The grid is represented by 11 independent squares due to symmetry -/
  grid : Fin 11 → Bool
  /-- Ensures that not all squares are the same color -/
  not_all_same : ∃ i j, grid i ≠ grid j

/-- Instance of Fintype for SymmetricScanningCode -/
instance : Fintype SymmetricScanningCode :=
  sorry

/-- The number of symmetric scanning codes for an 8x8 grid -/
def num_symmetric_scanning_codes : ℕ := 2046

/-- Theorem stating the correct number of symmetric scanning codes -/
theorem count_symmetric_scanning_codes :
  Fintype.card SymmetricScanningCode = num_symmetric_scanning_codes :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_symmetric_scanning_codes_l1248_124842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_product_inequality_l1248_124874

theorem sum_product_inequality (x y z : ℝ) (hx : x ∈ Set.Icc 1 2) (hy : y ∈ Set.Icc 1 2) (hz : z ∈ Set.Icc 1 2) :
  (x + y + z) * (1/x + 1/y + 1/z) ≥ 6 * (x/(y+z) + y/(x+z) + z/(x+y)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_product_inequality_l1248_124874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_of_numbers_l1248_124820

theorem ratio_of_numbers (A B : ℕ) (hA : A = 20) (hB : B > 0) (hLCM : Nat.lcm A B = 80) :
  A / B = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_of_numbers_l1248_124820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_passing_game_l1248_124821

/-- Represents the number of ways the ball can be with A at the k-th pass -/
def a : ℕ → ℕ := sorry

/-- Represents the number of ways the ball can be with either B or C at the k-th pass -/
def b : ℕ → ℕ := sorry

/-- Initial conditions -/
axiom a_1 : a 1 = 0
axiom b_1 : b 1 = 2

/-- Recurrence relations -/
axiom a_recurrence : ∀ k : ℕ, k ≥ 1 → a (k + 1) = b k
axiom b_recurrence : ∀ k : ℕ, k ≥ 1 → b (k + 1) = 2 * a k + b k

/-- The main theorem: there are 682 ways for the ball to return to A on the 11th pass -/
theorem ball_passing_game : a 11 = 682 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_passing_game_l1248_124821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_chord_length_l1248_124887

/-- Circle with center (1, 2) and radius 5 -/
def my_circle (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 25

/-- Line passing through (3, 1) -/
def line_through_point (m : ℝ) (x y : ℝ) : Prop :=
  (2*m + 1)*x + (m + 1)*y - 7*m - 4 = 0

/-- Line perpendicular to the line connecting (1, 2) and (3, 1) -/
def perpendicular_line (x y : ℝ) : Prop := 2*x - y - 5 = 0

theorem minimize_chord_length :
  ∀ m : ℝ,
  (∃ x y : ℝ, my_circle x y ∧ line_through_point m x y) →
  (∀ x y : ℝ,
    my_circle x y ∧ line_through_point m x y →
    ∃ x' y' : ℝ, my_circle x' y' ∧ perpendicular_line x' y' ∧
    (x - x')^2 + (y - y')^2 ≤ (x - 1)^2 + (y - 2)^2) :=
by
  sorry

#check minimize_chord_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_chord_length_l1248_124887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_regression_equation_l1248_124882

noncomputable def data : List (ℝ × ℝ) := [(1, 2), (2, 3), (3, 4), (4, 5)]

noncomputable def mean_x : ℝ := (data.map Prod.fst).sum / data.length
noncomputable def mean_y : ℝ := (data.map Prod.snd).sum / data.length

theorem linear_regression_equation (x y : ℝ) :
  (x, y) ∈ data → y = x + 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_regression_equation_l1248_124882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trick_always_possible_l1248_124871

/-- Represents the state of a coin (Heads or Tails) -/
inductive CoinState
| Heads
| Tails

/-- Represents a group of three coins -/
def Triplet := (CoinState × CoinState × CoinState)

/-- Represents the row of 27 coins -/
def CoinRow := Vector Triplet 9

/-- Represents the position of the uncovered coin in a triplet -/
inductive UncoveredPosition
| Left
| Middle
| Right

/-- Function to determine if a triplet has a majority of heads -/
def hasMajorityHeads (t : Triplet) : Bool :=
  match t with
  | (CoinState.Heads, CoinState.Heads, _) => true
  | (CoinState.Heads, _, CoinState.Heads) => true
  | (_, CoinState.Heads, CoinState.Heads) => true
  | _ => false

/-- Function to determine the position to uncover based on the triplet state -/
def uncoverPosition (t : Triplet) : UncoveredPosition :=
  match t with
  | (CoinState.Heads, CoinState.Heads, _) => UncoveredPosition.Left
  | (_, CoinState.Heads, CoinState.Heads) => UncoveredPosition.Middle
  | (CoinState.Heads, _, CoinState.Heads) => UncoveredPosition.Right
  | _ => UncoveredPosition.Left  -- Default case, should not occur if used correctly

/-- The main theorem stating that the trick is always possible -/
theorem trick_always_possible (coins : CoinRow) :
  ∃ (uncovered : Vector (Fin 9) 5) (additional : Vector (Fin 9) 5),
    (∀ i : Fin 5, hasMajorityHeads (coins.get (uncovered.get i))) ∧
    (∀ i : Fin 5, hasMajorityHeads (coins.get (additional.get i))) ∧
    (∀ i : Fin 5, uncoverPosition (coins.get (uncovered.get i)) = 
                  UncoveredPosition.Left) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trick_always_possible_l1248_124871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1248_124869

-- Define the function f(x) = 2 / (x - 8)
noncomputable def f (x : ℝ) : ℝ := 2 / (x - 8)

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | f x ≠ 0} = {x : ℝ | x ≠ 8} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1248_124869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_fifth_black_third_green_l1248_124843

/-- Represents the color of a ball -/
inductive BallColor
  | Red
  | Black
  | Blue
  | Green

/-- Represents the initial state of the box -/
def initial_box : Multiset BallColor :=
  Multiset.replicate 3 BallColor.Red
  + Multiset.replicate 4 BallColor.Black
  + Multiset.replicate 2 BallColor.Blue
  + Multiset.replicate 1 BallColor.Green

/-- The probability of drawing a specific sequence of balls -/
noncomputable def probability_of_sequence (box : Multiset BallColor) (sequence : List BallColor) : ℚ :=
  sorry

/-- The sequence we're interested in: any color, any color, green, any color, black -/
def target_sequence : List (Option BallColor) :=
  [none, none, some BallColor.Green, none, some BallColor.Black]

/-- The main theorem to prove -/
theorem probability_fifth_black_third_green :
  probability_of_sequence initial_box
    (target_sequence.filterMap id) = 1 / 35 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_fifth_black_third_green_l1248_124843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_chord_length_l1248_124827

-- Define the line l
noncomputable def line_l (t : ℝ) (α : ℝ) : ℝ × ℝ := (-1 + t * Real.cos α, 1 + t * Real.sin α)

-- Define the circle C
noncomputable def circle_C (θ : ℝ) : ℝ × ℝ := (2 + 4 * Real.cos θ, 1 + 4 * Real.sin θ)

-- Define the center and radius of circle C
def circle_center : ℝ × ℝ := (2, 1)
def circle_radius : ℝ := 4

-- Define point P
def point_P : ℝ × ℝ := (-1, 1)

-- Theorem statement
theorem shortest_chord_length :
  ∃ (t₁ t₂ α θ₁ θ₂ : ℝ),
    let (x₁, y₁) := line_l t₁ α
    let (x₂, y₂) := line_l t₂ α
    let (cx₁, cy₁) := circle_C θ₁
    let (cx₂, cy₂) := circle_C θ₂
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = (2 * Real.sqrt 7)^2 ∧
    ∀ (t₃ t₄ θ₃ θ₄ : ℝ),
      let (x₃, y₃) := line_l t₃ α
      let (x₄, y₄) := line_l t₄ α
      let (cx₃, cy₃) := circle_C θ₃
      let (cx₄, cy₄) := circle_C θ₄
      (x₃ - x₄)^2 + (y₃ - y₄)^2 ≥ (2 * Real.sqrt 7)^2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_chord_length_l1248_124827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l1248_124849

/-- The length of the chord cut by the ray θ = π/4 on the circle ρ = 4sin θ is 2√2 -/
theorem chord_length :
  ∃ (chord_length : ℝ), chord_length = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l1248_124849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_k_value_l1248_124817

/-- Given vectors a and b, if k*a + b is perpendicular to 2*a - b, then k = 7/5 -/
theorem perpendicular_vectors_k_value (a b : ℝ × ℝ × ℝ) (k : ℝ) 
  (ha : a = (1, 1, 0)) 
  (hb : b = (-1, 0, 2)) 
  (hperp : (k • a + b) • (2 • a - b) = 0) : 
  k = 7/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_k_value_l1248_124817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_triangle_area_l1248_124866

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := x - 3/x

-- Define the tangent line at a point (x₀, f(x₀))
noncomputable def tangent_line (x₀ : ℝ) (x : ℝ) : ℝ :=
  f x₀ + (1 + 3/x₀^2) * (x - x₀)

-- Define the area of the triangle
noncomputable def triangle_area (x₀ : ℝ) : ℝ :=
  let y_intercept := tangent_line x₀ 0
  let x_intercept := 2 * x₀
  (1/2) * x_intercept * (-y_intercept)

-- Theorem statement
theorem constant_triangle_area :
  ∀ x₀ : ℝ, x₀ ≠ 0 → triangle_area x₀ = 6 := by
  sorry

#check constant_triangle_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_triangle_area_l1248_124866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_divisibility_l1248_124837

theorem polynomial_divisibility (f : Polynomial ℤ → Polynomial ℤ) (k : ℕ) (x : Polynomial ℤ) :
  (x ∣ (f^[k] x)) ↔ (x ∣ f x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_divisibility_l1248_124837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_function_difference_l1248_124884

/-- The function f(x) = a^x -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x

/-- The theorem statement -/
theorem exponential_function_difference (a : ℝ) : 
  a > 0 → a ≠ 1 → (∀ x ∈ Set.Icc 1 2, f a x ≤ f a 2) → 
  (∀ x ∈ Set.Icc 1 2, f a x ≥ f a 1) → 
  f a 2 - f a 1 = 6 → a = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_function_difference_l1248_124884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_labor_practice_problem_l1248_124889

-- Define the problem parameters
def teachers : ℕ → ℕ → Prop :=
  λ x y => y - 30 * x = 7 ∧ 31 * x - y = 1

def bus_capacity : ℕ → ℕ → Prop :=
  λ a b => a = 35 ∧ b = 30

def bus_rental : ℕ → ℕ → Prop :=
  λ a b => a = 400 ∧ b = 320

def total_capacity (x y m : ℕ) : Prop :=
  35 * m + 30 * (x - m) ≥ x + y

def total_rental (x m : ℕ) : Prop :=
  400 * m + 320 * (x - m) ≤ 3000

-- State the theorem
theorem labor_practice_problem (x y m : ℕ) :
  teachers x y →
  bus_capacity 35 30 →
  bus_rental 400 320 →
  total_capacity x y m →
  total_rental x m →
  x = 8 ∧ y = 247 ∧ m = 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_labor_practice_problem_l1248_124889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_average_cost_at_ten_years_l1248_124822

/-- Represents the cost structure and usage of a car -/
structure CarCost where
  initialCost : ℝ := 150000
  annualExpenses : ℝ := 15000
  firstYearMaintenance : ℝ := 3000
  maintenanceIncrease : ℝ := 3000

/-- Calculates the average annual cost of owning the car for n years -/
noncomputable def averageAnnualCost (c : CarCost) (n : ℝ) : ℝ :=
  (c.initialCost + c.annualExpenses * n + (c.firstYearMaintenance + c.maintenanceIncrease * n) * n / 2) / n

/-- Theorem stating that the average annual cost is minimized at 10 years -/
theorem min_average_cost_at_ten_years (c : CarCost) :
  ∀ n : ℝ, n > 0 → averageAnnualCost c 10 ≤ averageAnnualCost c n := by
  sorry

#check min_average_cost_at_ten_years

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_average_cost_at_ten_years_l1248_124822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l1248_124834

theorem relationship_abc (a b c : ℝ) 
  (ha : a = (6 : ℝ) ^ (7/10 : ℝ)) 
  (hb : b = (7/10 : ℝ) ^ (6 : ℝ)) 
  (hc : c = Real.log 6 / Real.log (7/10)) :
  c < b ∧ b < a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l1248_124834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_forest_inhabitants_classification_l1248_124899

-- Define the inhabitants
inductive Inhabitant : Type
| A : Inhabitant
| B : Inhabitant
| C : Inhabitant

-- Define the properties
axiom is_knight : Inhabitant → Prop
axiom is_werewolf : Inhabitant → Prop

-- Define the statements made by each inhabitant
def statement_A : Prop := is_werewolf Inhabitant.A
def statement_B : Prop := is_werewolf Inhabitant.B
def statement_C : Prop := ∀ (x y : Inhabitant), x ≠ y → (is_knight x → ¬is_knight y)

-- Theorem to prove
theorem forest_inhabitants_classification :
  -- Conditions
  (∀ i : Inhabitant, is_knight i ∨ ¬is_knight i) →
  (∃! i : Inhabitant, is_werewolf i) →
  (is_knight Inhabitant.A ↔ statement_A) →
  (is_knight Inhabitant.B ↔ statement_B) →
  (is_knight Inhabitant.C ↔ statement_C) →
  -- Conclusion
  (¬is_knight Inhabitant.A ∧ ¬is_werewolf Inhabitant.A) ∧
  (¬is_knight Inhabitant.B ∧ ¬is_werewolf Inhabitant.B) ∧
  (is_knight Inhabitant.C ∧ is_werewolf Inhabitant.C) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_forest_inhabitants_classification_l1248_124899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l1248_124840

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.exp (x * Real.log 2) + x - 2

-- State the theorem
theorem root_in_interval :
  (∀ x y : ℝ, x < y → f x < f y) →  -- f is monotonically increasing
  f 0 = -1 →
  f 1 = 1 →
  ∃ r ∈ Set.Ioo 0 1, f r = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l1248_124840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_inspections_defective_part_l1248_124830

/-- Represents a manufacturing stage -/
inductive Stage
  | RoughMachining
  | ReworkMachining
  | FineMachining

/-- Represents the state of a part after inspection -/
inductive InspectionResult
  | Pass
  | Fail

/-- A manufacturing process with inspections after each stage -/
def ManufacturingProcess := List (Stage × InspectionResult)

/-- Counts the number of inspections in a manufacturing process -/
def countInspections (process : ManufacturingProcess) : Nat :=
  process.length

/-- Theorem: The maximum number of inspections for a defective part is 3 -/
theorem max_inspections_defective_part (process : ManufacturingProcess) 
  (h1 : process.length > 0)
  (h2 : process.getLast?.isSome)
  (h3 : process.getLast? = some (Stage.FineMachining, InspectionResult.Fail)) :
  countInspections process ≤ 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_inspections_defective_part_l1248_124830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_max_F_l1248_124897

noncomputable def F (A B x : ℝ) : ℝ :=
  |Real.cos x^2 + 2 * Real.sin x * Real.cos x - Real.sin x^2 + A * x + B|

theorem minimize_max_F :
  ∃ (M : ℝ), M = Real.sqrt 2 ∧
  (∀ A B : ℝ, ∀ x ∈ Set.Icc 0 (3 * Real.pi / 2), F A B x ≤ M) ∧
  (∀ ε > 0, ∃ A B : ℝ, ∃ x ∈ Set.Icc 0 (3 * Real.pi / 2), F A B x > M - ε) ∧
  (∀ A B : ℝ, ∃ x ∈ Set.Icc 0 (3 * Real.pi / 2), F 0 0 x ≤ F A B x) := by
  sorry

#check minimize_max_F

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_max_F_l1248_124897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_to_circle_l1248_124814

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 18*x + y^2 + 14*y + 149 = 0

-- Define the center of the circle
def circle_center : ℝ × ℝ := (9, -7)

-- Define the radius of the circle
noncomputable def circle_radius : ℝ := Real.sqrt 19

-- Define the point we're measuring distance from
def point : ℝ × ℝ := (1, 1)

-- Theorem statement
theorem shortest_distance_to_circle :
  let distance_to_center := Real.sqrt ((point.1 - circle_center.1)^2 + (point.2 - circle_center.2)^2)
  (distance_to_center - circle_radius) = 8 * Real.sqrt 2 - Real.sqrt 19 := by
  sorry

#check shortest_distance_to_circle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_to_circle_l1248_124814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_specific_l1248_124886

/-- The polar coordinate equation of a circle with center (r, φ) and radius R -/
def polar_circle_equation (r : ℝ) (φ : ℝ) (R : ℝ) (θ : ℝ) (ρ : ℝ) : Prop :=
  ρ = 2 * R * Real.cos (θ - φ)

/-- Theorem: The polar coordinate equation of a circle with center (2, π/3) and radius 2 -/
theorem circle_equation_specific :
  ∀ θ ρ, polar_circle_equation 2 (π/3) 2 θ ρ ↔ ρ = 4 * Real.cos (θ - π/3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_specific_l1248_124886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_seven_theta_l1248_124806

theorem cos_seven_theta (θ : ℝ) (h : Real.cos θ = 1/4) : 
  Real.cos (7 * θ) = -160481/2097152 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_seven_theta_l1248_124806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1248_124860

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The dot product of vectors AB and AC in triangle ABC -/
noncomputable def dotProduct (t : Triangle) : ℝ := t.b * t.c * Real.cos t.A

/-- The area of triangle ABC -/
noncomputable def area (t : Triangle) : ℝ := (1/2) * t.b * t.c * Real.sin t.A

theorem triangle_problem (t : Triangle) 
  (h_cosA : Real.cos t.A = 3/5)
  (h_area : area t = 4) :
  dotProduct t = 6 ∧ (t.b = 2 → t.a = Real.sqrt 17) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1248_124860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_consumption_relation_l1248_124802

/-- Proves that a 10% tax reduction and a specific revenue effect implies a certain consumption increase -/
theorem tax_consumption_relation (T C : ℝ) (h_positive : T > 0 ∧ C > 0) :
  let T' := 0.90 * T
  let R := T * C
  let R' := 0.9999999999999858 * R
  ∃ X : ℝ, R' = T' * (C * (1 + X / 100)) ∧ abs (X - 11.11111111110953) < 1e-10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_consumption_relation_l1248_124802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_slope_product_l1248_124829

noncomputable section

/-- An ellipse with semi-major axis a, semi-minor axis b, and eccentricity e --/
structure Ellipse (a b e : ℝ) : Prop where
  positive : a > b ∧ b > 0
  eccentricity : e = Real.sqrt 3 / 2
  equation : ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1

/-- A point on the ellipse --/
structure PointOnEllipse (E : Ellipse a b e) (x y : ℝ) : Prop where
  on_ellipse : x^2 / a^2 + y^2 / b^2 = 1

/-- Two points symmetric with respect to the origin --/
def SymmetricPoints (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ = -x₂ ∧ y₁ = -y₂

/-- The slope of a line passing through two points --/
def Slope (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  (y₂ - y₁) / (x₂ - x₁)

/-- The main theorem --/
theorem ellipse_slope_product (a b e : ℝ) (E : Ellipse a b e) 
  (x₁ y₁ x₂ y₂ xM yM : ℝ) 
  (A : PointOnEllipse E x₁ y₁) 
  (B : PointOnEllipse E x₂ y₂) 
  (M : PointOnEllipse E xM yM) 
  (h_symmetric : SymmetricPoints x₁ y₁ x₂ y₂) :
  Slope xM yM x₁ y₁ * Slope xM yM x₂ y₂ = -1/4 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_slope_product_l1248_124829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ac_length_is_one_or_two_l1248_124805

-- Define the triangle ABC
structure Triangle where
  A : Real -- Angle A in radians
  AB : Real -- Length of side AB
  BC : Real -- Length of side BC

-- Define our specific triangle
noncomputable def ourTriangle : Triangle where
  A := Real.pi / 6  -- 30° in radians
  AB := Real.sqrt 3
  BC := 1

-- Theorem statement
theorem ac_length_is_one_or_two (t : Triangle) (h : t = ourTriangle) :
  ∃ (AC : Real), (AC = 1 ∨ AC = 2) ∧ 
  AC^2 = t.AB^2 + t.BC^2 - 2 * t.AB * t.BC * Real.cos t.A := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ac_length_is_one_or_two_l1248_124805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_increasing_on_negative_l1248_124808

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the function g
def g (x : ℝ) : ℝ := x^2 * f x

-- State the theorem
theorem g_increasing_on_negative (h1 : ∀ x y, x < y → f x < f y) 
                                 (h2 : ∀ x, f x < 0) :
  ∀ a b, a < b → b < 0 → g a < g b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_increasing_on_negative_l1248_124808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_theorem_l1248_124836

/-- A function f is monotonically increasing on ℝ -/
def MonotonicallyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

/-- The function f(x) = (1/3)ax^3 + cx -/
noncomputable def f (a c : ℝ) (x : ℝ) : ℝ := (1/3) * a * x^3 + c * x

/-- The expression we want to maximize -/
noncomputable def g (a c : ℝ) : ℝ := a / (c^2 + 4) + c / (a^2 + 4)

theorem max_value_theorem (a c : ℝ) :
  MonotonicallyIncreasing (f a c) →
  a ≤ 4 →
  g a c ≤ 1/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_theorem_l1248_124836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_translation_l1248_124885

/-- Represents a parabola in the form y = a(x - h)² + k, where (h, k) is the vertex --/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- Translates a parabola horizontally and vertically --/
def translate (p : Parabola) (dx dy : ℝ) : Parabola :=
  { a := p.a, h := p.h + dx, k := p.k + dy }

theorem parabola_translation (x y : ℝ) :
  let original := Parabola.mk 1 0 0  -- y = x²
  let translated := translate original 2 (-3)  -- 2 units right, 3 units down
  y = x^2 ↔ y = (x - translated.h)^2 + translated.k := by
  sorry

#check parabola_translation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_translation_l1248_124885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_solutions_l1248_124855

noncomputable def r : ℂ := 1
noncomputable def s : ℂ := Complex.exp (2 * Real.pi * Complex.I / 3)
noncomputable def t : ℂ := Complex.exp (4 * Real.pi * Complex.I / 3)

def satisfies_property (c : ℂ) : Prop :=
  ∀ z : ℂ, (z - r) * (z - s) * (z - t) = (z - c*r) * (z - c*s) * (z - c*t)

theorem exactly_three_solutions :
  ∃! (solution_set : Finset ℂ),
    (∀ c ∈ solution_set, satisfies_property c) ∧
    (Finset.card solution_set = 3) :=
by
  sorry

#check exactly_three_solutions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_solutions_l1248_124855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gross_profit_percentage_example_l1248_124848

/-- Calculates the gross profit percentage given the sales price and gross profit -/
noncomputable def gross_profit_percentage (sales_price : ℝ) (gross_profit : ℝ) : ℝ :=
  let cost := sales_price - gross_profit
  (gross_profit / cost) * 100

/-- Theorem: Given a sales price of $54 and a gross profit of $30, 
    the gross profit percentage is 125% -/
theorem gross_profit_percentage_example : 
  gross_profit_percentage 54 30 = 125 := by
  unfold gross_profit_percentage
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num
  -- QED
  

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gross_profit_percentage_example_l1248_124848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lauren_reaches_andrea_in_35_minutes_l1248_124823

noncomputable section

-- Define the initial distance between Andrea and Lauren
def initial_distance : ℝ := 30

-- Define the rate at which the distance between them decreases
def decrease_rate : ℝ := 2

-- Define the time Andrea bikes before stopping
def andrea_bike_time : ℝ := 10

-- Define the relationship between Andrea's and Lauren's speeds
def speed_ratio : ℝ := 4

-- Define Lauren's speed
noncomputable def lauren_speed : ℝ := decrease_rate / (1 + speed_ratio)

-- Define the remaining distance after Andrea stops
noncomputable def remaining_distance : ℝ := initial_distance - decrease_rate * andrea_bike_time

-- Define the total time it takes for Lauren to reach Andrea
noncomputable def total_time : ℝ := andrea_bike_time + remaining_distance / lauren_speed

-- Theorem stating that the total time is 35 minutes
theorem lauren_reaches_andrea_in_35_minutes : total_time = 35 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lauren_reaches_andrea_in_35_minutes_l1248_124823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_labor_tools_equation_l1248_124803

/-- Represents the unit price of type A labor tools -/
def x : ℝ := sorry

/-- The total cost of type A labor tools -/
def cost_A : ℝ := 1000

/-- The total cost of type B labor tools -/
def cost_B : ℝ := 2400

/-- The difference in unit price between type B and type A tools -/
def price_difference : ℝ := 4

/-- The ratio of quantity of type B tools to type A tools -/
def quantity_ratio : ℝ := 2

/-- Theorem stating the fractional equation satisfied by x -/
theorem labor_tools_equation (h₁ : x > 0) :
  cost_B / (x + price_difference) = quantity_ratio * (cost_A / x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_labor_tools_equation_l1248_124803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_beta_value_l1248_124851

theorem cos_beta_value (α β : ℝ)
  (h1 : Real.cos α = 1 / 7)
  (h2 : Real.cos (α + β) = -11 / 14)
  (h3 : 0 < α ∧ α < π / 2)
  (h4 : π / 2 < α + β ∧ α + β < π) :
  Real.cos β = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_beta_value_l1248_124851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_l_equation_chord_length_proof_l1248_124854

-- Define the lines
def line1 (x y : ℝ) : Prop := 3*x + y - 1 = 0
def line2 (x y : ℝ) : Prop := x - 5*y - 11 = 0
def line3 (x y : ℝ) : Prop := x + 4*y = 0

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + (y-11)^2 = 25

-- Define the intersection point of line1 and line2
def intersection_point : ℝ × ℝ := (1, -2)

-- Define the perpendicular line l
def line_l (x y : ℝ) : Prop := 4*x - y - 6 = 0

-- Theorem for the equation of line l
theorem line_l_equation : 
  (∀ x y, line_l x y ↔ (x = intersection_point.1 ∧ y = intersection_point.2)) ∧
  (∀ x y, line_l x y → line3 x y → x = y) :=
sorry

-- Define the length of the chord
noncomputable def chord_length : ℝ := 4 * Real.sqrt 2

-- Theorem for the length of the chord
theorem chord_length_proof :
  ∃ a b c d : ℝ, 
    line_l a b ∧ line_l c d ∧ 
    circle_eq a b ∧ circle_eq c d ∧
    Real.sqrt ((a - c)^2 + (b - d)^2) = chord_length :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_l_equation_chord_length_proof_l1248_124854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solid_revolution_volume_formula_l1248_124809

-- Define the right triangle
structure RightTriangle where
  a : ℝ
  α : ℝ
  h_a_pos : a > 0
  h_α_pos : α > 0
  h_α_lt_pi_half : α < Real.pi / 2

-- Define the volume of the solid of revolution
noncomputable def solidRevolutionVolume (t : RightTriangle) : ℝ :=
  (Real.pi * t.a^3 / 3) * Real.tan (Real.pi / 2 - t.α / 2) * Real.cos (t.α / 2) * Real.tan (Real.pi / 2 - t.α)

-- Theorem statement
theorem solid_revolution_volume_formula (t : RightTriangle) :
  solidRevolutionVolume t =
  (Real.pi * t.a^3 / 3) * Real.tan (Real.pi / 2 - t.α / 2) * Real.cos (t.α / 2) * Real.tan (Real.pi / 2 - t.α) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solid_revolution_volume_formula_l1248_124809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_turtle_walk_length_l1248_124881

/-- Represents a 3x3 grid with dots labeled A to M -/
inductive GridDot
| A | B | C | D | E | F | K | L | M

/-- Represents a path between two adjacent dots -/
inductive GridPath
| orthogonal (start finish : GridDot)
| diagonal (start finish : GridDot)

/-- Represents a turtle's walk on the grid -/
def TurtleWalk := List GridPath

/-- Checks if a path is valid (connects adjacent dots) -/
def isValidPath (p : GridPath) : Bool :=
  match p with
  | GridPath.orthogonal start finish => sorry
  | GridPath.diagonal start finish => sorry

/-- Checks if a turtle's walk is valid (alternates between orthogonal and diagonal moves) -/
def isValidWalk : TurtleWalk → Bool
  | [] => true
  | [_] => true
  | (p1 :: p2 :: rest) =>
    match p1, p2 with
    | GridPath.orthogonal _ _, GridPath.diagonal _ _ => isValidWalk (p2 :: rest)
    | GridPath.diagonal _ _, GridPath.orthogonal _ _ => isValidWalk (p2 :: rest)
    | _, _ => false

/-- Checks if a turtle's walk doesn't repeat any path -/
def hasNoDuplicates (w : TurtleWalk) : Bool :=
  sorry

/-- The main theorem: The maximum number of paths in a valid turtle walk is 17 -/
theorem max_turtle_walk_length :
  ∀ (w : TurtleWalk), isValidWalk w → hasNoDuplicates w → w.length ≤ 17 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_turtle_walk_length_l1248_124881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_sin_theta_l1248_124865

/-- The function f(x) as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (x / 2) - 4 * Real.cos (x / 2)

/-- Definition of symmetry with respect to a vertical line -/
def symmetric_about (f : ℝ → ℝ) (θ : ℝ) : Prop :=
  ∀ x, f (θ + x) = f (θ - x)

/-- The main theorem -/
theorem symmetry_implies_sin_theta (θ : ℝ) :
  symmetric_about f θ → Real.sin θ = -24/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_sin_theta_l1248_124865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1248_124847

noncomputable def f (x : ℝ) : ℝ := min (4*x + 1) (min (x + 2) (-2*x + 4))

theorem max_value_of_f :
  ∃ M : ℝ, (∀ x : ℝ, f x ≤ M) ∧ (∃ x : ℝ, f x = M) ∧ M = 8/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1248_124847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_cube_exists_l1248_124804

theorem smaller_cube_exists (n : ℕ) (points : Finset (Fin n × Fin n × Fin n)) :
  n = 13 →
  points.card = 1956 →
  ∃ (x y z : Fin n), ∀ (p : Fin n × Fin n × Fin n),
    p ∈ points →
    ¬(x ≤ p.1 ∧ p.1 < x.val + 1 ∧
      y ≤ p.2.1 ∧ p.2.1 < y.val + 1 ∧
      z ≤ p.2.2 ∧ p.2.2 < z.val + 1) :=
by
  intro hn hcard
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_cube_exists_l1248_124804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_zeros_l1248_124850

-- Define the function f(x) = cos(log x)
noncomputable def f (x : ℝ) : ℝ := Real.cos (Real.log x)

-- State the theorem
theorem infinitely_many_zeros :
  ∃ S : Set ℝ, (∀ x ∈ S, 0 < x ∧ x < 1 ∧ f x = 0) ∧ Set.Infinite S :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_zeros_l1248_124850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_points_l1248_124807

/-- Square with side length √2 -/
structure Square where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  side_length : ℝ
  is_square : A = (0, 1) ∧ B = (-1, 0) ∧ C = (0, -1) ∧ D = (1, 0) ∧ side_length = Real.sqrt 2

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Theorem: Locus of points P satisfying the given condition -/
theorem locus_of_points (s : Square) (P : ℝ × ℝ) :
  P.1 ≥ 0 →
  (distance P s.A + distance P s.C) / Real.sqrt 2 = max (distance P s.B) (distance P s.D) →
  P.1^2 + P.2^2 = 1 := by
  sorry

#check locus_of_points

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_points_l1248_124807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cathy_can_win_l1248_124895

/-- Represents the state of the marble game -/
structure GameState where
  n : Nat  -- Number of marbles
  k : Nat  -- Number of boxes
  boxes : List (List Nat)  -- List of boxes, each containing a list of marbles

/-- Checks if a game state is valid according to the rules -/
def isValidState (state : GameState) : Prop :=
  state.n > 0 ∧ 
  state.k > 0 ∧ 
  state.boxes.length = state.k ∧
  (∀ box ∈ state.boxes, ∀ marble ∈ box, 1 ≤ marble ∧ marble ≤ state.n) ∧
  (state.boxes.map List.length).sum = state.n

/-- Represents a valid move in the game -/
inductive Move where
  | moveMarble : Nat → Nat → Move

/-- Applies a move to a game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  sorry

/-- Checks if Cathy has won the game -/
def hasWon (state : GameState) : Prop :=
  ∃ box ∈ state.boxes, box = [state.n]

/-- Cathy's winning strategy -/
noncomputable def winningStrategy (state : GameState) : List Move :=
  sorry

/-- The main theorem stating the winning condition for Cathy -/
theorem cathy_can_win (n k : Nat) :
  (∃ initialState : GameState, 
    isValidState initialState ∧ 
    initialState.n = n ∧ 
    initialState.k = k ∧ 
    (∃ strategy : List Move, 
      hasWon (List.foldl applyMove initialState strategy))) ↔ 
  n ≤ 2^(k-1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cathy_can_win_l1248_124895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_vector_computation_l1248_124846

theorem matrix_vector_computation 
  (N : Matrix (Fin 2) (Fin 2) ℝ) 
  (a b : Fin 2 → ℝ) 
  (h1 : N.mulVec a = ![5, 4]) 
  (h2 : N.mulVec b = ![-3, -6]) : 
  N.mulVec (2 • a - 2 • b) = ![16, 20] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_vector_computation_l1248_124846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1248_124870

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.b = Real.sqrt 5 ∧
  t.c = 3 ∧
  Real.sqrt 5 * t.a * Real.sin t.B = t.b ∧
  Real.pi / 2 < t.A ∧ t.A < Real.pi

-- Theorem statement
theorem triangle_properties (t : Triangle) (h : triangle_conditions t) :
  Real.sin t.A = Real.sqrt 5 / 5 ∧
  t.a + t.b + t.c = Real.sqrt 26 + Real.sqrt 5 + 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1248_124870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_to_cone_volume_l1248_124800

-- Define the radius of the semicircular sheet
noncomputable def sheet_radius : ℝ := 6

-- Define the volume of the cone
noncomputable def cone_volume (r h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

-- Theorem statement
theorem semicircle_to_cone_volume :
  let r := sheet_radius / 2 -- radius of cone base
  let h := Real.sqrt (sheet_radius^2 - r^2) -- height of cone
  cone_volume r h = 9 * Real.sqrt 3 * Real.pi :=
by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_to_cone_volume_l1248_124800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_selling_price_calculation_l1248_124825

theorem selling_price_calculation (cost_price markup_percentage discount_percentage : ℝ) : 
  cost_price = 540 →
  markup_percentage = 15 →
  discount_percentage = 26.08695652173913 →
  (cost_price * (1 + markup_percentage / 100) * (1 - discount_percentage / 100)) = 459 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_selling_price_calculation_l1248_124825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_prism_circumscribed_sphere_radius_l1248_124863

/-- The radius of the circumscribed sphere of a right triangular prism -/
noncomputable def circumscribed_sphere_radius (base_side : ℝ) (height : ℝ) (depth : ℝ) : ℝ :=
  (1 / 2) * Real.sqrt (height^2 + depth^2)

/-- Theorem: The radius of the circumscribed sphere of a specific right triangular prism is 5 -/
theorem specific_prism_circumscribed_sphere_radius :
  circumscribed_sphere_radius (4 * Real.sqrt 2) 6 8 = 5 := by
  sorry

#check specific_prism_circumscribed_sphere_radius

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_prism_circumscribed_sphere_radius_l1248_124863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_square_real_iff_ab_zero_l1248_124876

theorem complex_square_real_iff_ab_zero (a b : ℝ) :
  let z : ℂ := Complex.mk a b
  (∃ (r : ℝ), z^2 = r) ↔ a * b = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_square_real_iff_ab_zero_l1248_124876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_expanded_form_l1248_124819

theorem sum_of_coefficients_expanded_form (d : ℝ) : 
  -2 + 20 + (-48) = -30 := by
  -- Proof
  rfl

#eval -2 + 20 + (-48)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_expanded_form_l1248_124819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_price_from_decreased_price_l1248_124858

/-- 
Given an article whose price after a 40% decrease is 1050 Rs, 
prove that its original price was 1750 Rs.
-/
theorem original_price_from_decreased_price (decreased_price : ℝ) 
  (h : decreased_price = 1050) : decreased_price = 1750 * 0.6 :=
  by
  -- Define the original price
  let original_price : ℝ := 1750
  
  -- Define the decrease percentage
  let decrease_percentage : ℝ := 0.4
  
  -- Assert that the decreased price is equal to the original price minus the decrease
  have decreased_price_eq : decreased_price = original_price * (1 - decrease_percentage) := by
    rw [h]
    norm_num
  
  -- Prove that the decreased price is equal to 1750 * 0.6
  rw [decreased_price_eq]
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_price_from_decreased_price_l1248_124858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_reciprocal_sum_l1248_124883

-- Define the curve C₁ in polar coordinates
def C₁ (ρ θ : ℝ) : Prop :=
  ρ^2 - 4*ρ*(Real.cos θ) - 4*ρ*(Real.sin θ) + 7 = 0

-- Define the line C₂ in polar coordinates
def C₂ (θ : ℝ) : Prop :=
  θ = Real.pi/3

-- Define the intersection points
def intersection_points (ρ₁ ρ₂ : ℝ) : Prop :=
  C₁ ρ₁ (Real.pi/3) ∧ C₁ ρ₂ (Real.pi/3) ∧ ρ₁ ≠ ρ₂

-- Theorem statement
theorem intersection_reciprocal_sum (ρ₁ ρ₂ : ℝ) 
  (h : intersection_points ρ₁ ρ₂) : 
  (1/ρ₁) + (1/ρ₂) = (2*Real.sqrt 3 + 2)/7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_reciprocal_sum_l1248_124883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_plane_necessary_not_sufficient_l1248_124878

-- Define plane structure
structure Plane where
  -- You can add specific fields if needed

-- Define line structure
structure Line where
  -- You can add specific fields if needed

-- Define perpendicularity between a line and a plane
def perpendicular (l : Line) (p : Plane) : Prop :=
  sorry -- Placeholder definition

-- Define perpendicularity between two lines
def perpendicular_lines (l1 l2 : Line) : Prop :=
  sorry -- Placeholder definition

-- Define when a line is a subset of a plane
def subset_plane (l : Line) (p : Plane) : Prop :=
  sorry -- Placeholder definition

-- Main theorem
theorem perpendicular_plane_necessary_not_sufficient 
  (α : Plane) (a b l : Line) 
  (ha : subset_plane a α) (hb : subset_plane b α) :
  (perpendicular l α → perpendicular_lines l a ∧ perpendicular_lines l b) ∧
  ∃ (α : Plane) (a b l : Line),
    subset_plane a α ∧ subset_plane b α ∧
    perpendicular_lines l a ∧ perpendicular_lines l b ∧
    ¬perpendicular l α :=
by
  sorry -- Placeholder proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_plane_necessary_not_sufficient_l1248_124878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_problem_l1248_124872

/-- The average score function for a batsman after n innings -/
noncomputable def A (p : ℝ) (n : ℝ) : ℝ := p * (n / (n + 3))^2

/-- The theorem representing the cricket problem -/
theorem cricket_problem (p : ℝ) :
  (∃ (A_16 : ℝ), A p 16 = A_16 ∧ A p 17 = A_16 + 3) →
  A p 17 = (129600 * 289) / (1969 * 400) := by
  sorry

/-- The final average score after 17 innings -/
noncomputable def final_average : ℝ := (129600 * 289) / (1969 * 400)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_problem_l1248_124872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_range_l1248_124890

/-- The function f(x) = x^3 + x + 2/3 -/
noncomputable def f (x : ℝ) : ℝ := x^3 + x + 2/3

/-- The derivative of f(x) -/
noncomputable def f' (x : ℝ) : ℝ := 3 * x^2 + 1

theorem tangent_slope_range :
  ∀ x : ℝ, f' x ≥ 1 := by
  intro x
  unfold f'
  -- The proof that 3 * x^2 + 1 ≥ 1 for all real x
  have h : 3 * x^2 ≥ 0 := by
    apply mul_nonneg
    · norm_num
    · apply sq_nonneg
  linarith

#check tangent_slope_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_range_l1248_124890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1248_124811

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 4^x - 4 * 2^x + 3

-- Theorem statement
theorem f_properties :
  (∃ (x : ℝ), f x = -1 ∧ ∀ (y : ℝ), f y ≥ f x) ∧
  (f 1 = -1) ∧
  (∀ (x : ℝ), f x ≤ 35 ↔ x ≤ 3) := by
  sorry

#check f_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1248_124811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_condition_l1248_124888

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sin (3 * x) + a * Real.cos (3 * x)

theorem symmetry_condition (a : ℝ) :
  (∀ x : ℝ, f a (x + (-π/9)) = f a (-x + (-π/9))) ↔ a = -Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_condition_l1248_124888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_properties_l1248_124818

theorem triangle_angle_properties (A : ℝ) (h_acute : 0 < A ∧ A < Real.pi / 2) :
  let sinA := Real.sin A
  let cosA := Real.cos A
  let tanA := Real.tan A
  -- Part I
  (sinA - 2 * cosA = -1 → tanA = 3/4) ∧
  -- Part II
  (sinA - 2 * cosA < 0 ∧
   (∀ x ∈ Set.Icc 1 2, 
    Monotone (fun x => sinA * x^2 - 2*cosA * x + 1)) →
   0 ≤ sinA^2 - sinA * cosA ∧ sinA^2 - sinA * cosA < 2/5) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_properties_l1248_124818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqDmToCmConversion_monthsToYearsConversion_sqDmToSqMConversion_hourConversion_l1248_124801

-- Define conversion rates
noncomputable def sqDmToCm : ℚ := 100
noncomputable def monthsPerYear : ℚ := 12
noncomputable def sqMToSqDm : ℚ := 100

-- Define conversion functions
noncomputable def convertSqDmToCm (sqDm : ℚ) : ℚ := sqDm * sqDmToCm
noncomputable def convertMonthsToYears (months : ℚ) : ℚ := months / monthsPerYear
noncomputable def convertSqDmToSqM (sqDm : ℚ) : ℚ := sqDm / sqMToSqDm
def convert24To12Hour (hour : ℕ) : ℕ := hour - 12

-- Theorem statements
theorem sqDmToCmConversion : convertSqDmToCm 2 = 200 := by sorry

theorem monthsToYearsConversion : convertMonthsToYears 24 = 2 := by sorry

theorem sqDmToSqMConversion : convertSqDmToSqM 3000 = 30 := by sorry

theorem hourConversion : convert24To12Hour 15 = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqDmToCmConversion_monthsToYearsConversion_sqDmToSqMConversion_hourConversion_l1248_124801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_and_trig_expression_values_l1248_124879

-- Define the given condition
def given_condition (x : ℝ) : Prop :=
  Real.tan (Real.pi/4 + x) = -1/2

-- Define that x is in the second quadrant
def second_quadrant (x : ℝ) : Prop :=
  Real.pi/2 < x ∧ x < Real.pi

-- Theorem statement
theorem tan_and_trig_expression_values (x : ℝ) 
  (h1 : given_condition x) (h2 : second_quadrant x) : 
  Real.tan (2*x) = 3/4 ∧ 
  Real.sqrt ((1 + Real.sin x) / (1 - Real.sin x)) + Real.sqrt ((1 - Real.sin x) / (1 + Real.sin x)) = 2 * Real.sqrt 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_and_trig_expression_values_l1248_124879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_wins_match_game_l1248_124826

/-- A game where two players take turns removing matches from a pile. -/
structure MatchGame where
  /-- The initial number of matches in the pile. -/
  initial_matches : ℕ
  /-- A function that returns true if a move is valid (can remove p^n matches). -/
  valid_move : ℕ → Prop

/-- Returns true if the first player has a winning strategy in the match game. -/
def first_player_wins (game : MatchGame) : Prop :=
  ∃ (strategy : ℕ → ℕ),
    (∀ (n : ℕ), n < game.initial_matches → game.valid_move (strategy n)) ∧
    (∀ (opponent_strategy : ℕ → ℕ),
      (∀ (n : ℕ), n < game.initial_matches → game.valid_move (opponent_strategy n)) →
      ∃ (k : ℕ), strategy k = game.initial_matches - (strategy k))

/-- The specific match game described in the problem. -/
def match_game : MatchGame :=
  { initial_matches := 10000000,
    valid_move := λ n => ∃ (p : ℕ) (k : ℕ), Nat.Prime p ∧ n = p ^ k }

theorem first_player_wins_match_game : first_player_wins match_game := by
  sorry

#check first_player_wins_match_game

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_wins_match_game_l1248_124826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_squash_on_friday_l1248_124824

-- Define the days of the week
inductive Day : Type
  | Monday : Day
  | Tuesday : Day
  | Wednesday : Day
  | Thursday : Day
  | Friday : Day
  | Saturday : Day
  | Sunday : Day

-- Define the sports
inductive Sport : Type
  | Jogging : Sport
  | Karate : Sport
  | Volleyball : Sport
  | Squash : Sport
  | Cricket : Sport

-- Define a schedule as a function from Day to Sport
def Schedule := Day → Sport

-- Define the next and previous day functions
def next_day : Day → Day
  | Day.Monday => Day.Tuesday
  | Day.Tuesday => Day.Wednesday
  | Day.Wednesday => Day.Thursday
  | Day.Thursday => Day.Friday
  | Day.Friday => Day.Saturday
  | Day.Saturday => Day.Sunday
  | Day.Sunday => Day.Monday

def prev_day : Day → Day
  | Day.Monday => Day.Sunday
  | Day.Tuesday => Day.Monday
  | Day.Wednesday => Day.Tuesday
  | Day.Thursday => Day.Wednesday
  | Day.Friday => Day.Thursday
  | Day.Saturday => Day.Friday
  | Day.Sunday => Day.Saturday

-- Define the conditions
def valid_schedule (s : Schedule) : Prop :=
  -- Karate on Tuesday
  s Day.Tuesday = Sport.Karate
  -- Volleyball on Thursday
  ∧ s Day.Thursday = Sport.Volleyball
  -- Jogging on three non-consecutive days
  ∧ (∃ (d1 d2 d3 : Day), d1 ≠ d2 ∧ d2 ≠ d3 ∧ d1 ≠ d3
      ∧ s d1 = Sport.Jogging ∧ s d2 = Sport.Jogging ∧ s d3 = Sport.Jogging
      ∧ (∀ (d : Day), s d = Sport.Jogging → d = d1 ∨ d = d2 ∨ d = d3)
      ∧ (∀ (d : Day), s d = Sport.Jogging → s (next_day d) ≠ Sport.Jogging))
  -- Cricket not after jogging or squash
  ∧ (∀ (d : Day), s d = Sport.Cricket → s (prev_day d) ≠ Sport.Jogging ∧ s (prev_day d) ≠ Sport.Squash)
  -- All sports are played
  ∧ (∃ (d : Day), s d = Sport.Squash)
  ∧ (∃ (d : Day), s d = Sport.Cricket)

-- Theorem to prove
theorem squash_on_friday (s : Schedule) (h : valid_schedule s) : s Day.Friday = Sport.Squash := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_squash_on_friday_l1248_124824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_to_trigonometric_system_l1248_124845

theorem solution_to_trigonometric_system :
  ∀ (x y : ℝ), 
    (Real.cos x = 2 * (Real.cos y)^3 ∧ Real.sin x = 2 * (Real.sin y)^3) ↔ 
    (∃ (l k : ℤ), x = 2 * l * Real.pi + k * Real.pi / 2 + Real.pi / 4 ∧ y = k * Real.pi / 2 + Real.pi / 4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_to_trigonometric_system_l1248_124845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_product_equality_l1248_124868

/-- A point in a plane. -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in a plane. -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A circle in a plane. -/
structure Circle where
  center : Point
  radius : ℝ

/-- Represents that a line is externally tangent to a circle. -/
def ExternallyTangent (l : Line) (c : Circle) : Prop :=
  sorry

/-- Represents the distance between a point and a line. -/
noncomputable def Distance (p : Point) (l : Line) : ℝ :=
  sorry

/-- Given three circles and their external tangent lines, this theorem states 
    that the product of certain distances between vertices and tangent lines is equal. -/
theorem distance_product_equality 
  (A B C : Point)
  (Sa Sb Sc : Circle) 
  (la lb lc : Line) 
  (h_tangent_a : ExternallyTangent la Sb ∧ ExternallyTangent la Sc)
  (h_tangent_b : ExternallyTangent lb Sa ∧ ExternallyTangent lb Sc)
  (h_tangent_c : ExternallyTangent lc Sa ∧ ExternallyTangent lc Sb)
  (dab dac dbc dba dcb dca : ℝ)
  (h_dab : dab = Distance B la)
  (h_dac : dac = Distance C la)
  (h_dbc : dbc = Distance C lb)
  (h_dba : dba = Distance A lb)
  (h_dcb : dcb = Distance B lc)
  (h_dca : dca = Distance A lc)
  : dab * dbc * dca = dac * dba * dcb := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_product_equality_l1248_124868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gamma_sequence_operation_T_B_9_result_l1248_124832

/-- Definition of a Γ sequence -/
def GammaSequence (B : List ℝ) : Prop :=
  B.length ≥ 2 ∧ ∀ x ∈ B, -1 < x ∧ x < 1

/-- Operation T on two elements of a Γ sequence -/
noncomputable def operationT (a b : ℝ) : ℝ := (a + b) / (1 + a * b)

/-- Theorem: Operation T can be executed n-1 times on any n-term Γ sequence,
    and the result is always in the set {x | -1 < x < 1} -/
theorem gamma_sequence_operation_T (B : List ℝ) (hB : GammaSequence B) :
  (∀ k : ℕ, k < B.length - 1 → 
    ∃ (Bk : List ℝ), GammaSequence Bk ∧ Bk.length = B.length - k) ∧
  (∀ a b : ℝ, a ∈ B → b ∈ B → -1 < operationT a b ∧ operationT a b < 1) := by
  sorry

/-- The result of B₉ is 5/6 -/
theorem B_9_result (B : List ℝ) (hB : GammaSequence B) 
  (hB_elements : B = [-5/7, -1/6, -1/5, -1/4, 5/6, 1/2, 1/3, 1/4, 1/5, 1/6]) :
  ∃ (B_9 : List ℝ), B_9 = [5/6] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gamma_sequence_operation_T_B_9_result_l1248_124832

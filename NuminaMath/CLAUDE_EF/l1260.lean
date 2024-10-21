import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_discounted_price_theorem_price_before_discounts_and_tax_l1260_126066

/-- Calculates the price after applying a series of discounts --/
def apply_discounts (initial_price : ℝ) (discounts : List ℝ) : ℝ :=
  discounts.foldl (fun price discount => price * (1 - discount)) initial_price

/-- Theorem stating the relationship between initial price and discounted price --/
theorem discounted_price_theorem (P : ℝ) (discounts : List ℝ) 
  (h1 : discounts = [0.25, 0.15, 0.10, 0.05])
  (h2 : apply_discounts P discounts = 6800) :
  abs (P - 12476.39) < 0.01 := by
  sorry

/-- Main theorem combining discounts and sales tax --/
theorem price_before_discounts_and_tax (P : ℝ) (discounts : List ℝ) (sales_tax : ℝ)
  (h1 : discounts = [0.25, 0.15, 0.10, 0.05])
  (h2 : apply_discounts P discounts = 6800)
  (h3 : sales_tax = 0.10) :
  abs (P - 12476.39) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_discounted_price_theorem_price_before_discounts_and_tax_l1260_126066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_segments_between_parallel_planes_are_equal_l1260_126093

-- Define a type for points in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a type for planes in 3D space
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

-- Define a function to check if two planes are parallel
def arePlanesParallel (p1 p2 : Plane) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ p1.a = k * p2.a ∧ p1.b = k * p2.b ∧ p1.c = k * p2.c

-- Define a function to calculate the distance between two points
noncomputable def distance (p1 p2 : Point3D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2)

-- Define a function to check if a line segment is parallel to two planes
def isSegmentParallelToPlanes (p1 p2 : Point3D) (plane1 plane2 : Plane) : Prop :=
  ∃ (t : ℝ), 0 < t ∧ t < 1 ∧
    plane1.a * (p1.x + t * (p2.x - p1.x)) +
    plane1.b * (p1.y + t * (p2.y - p1.y)) +
    plane1.c * (p1.z + t * (p2.z - p1.z)) + plane1.d = 0 ∧
    plane2.a * (p1.x + t * (p2.x - p1.x)) +
    plane2.b * (p1.y + t * (p2.y - p1.y)) +
    plane2.c * (p1.z + t * (p2.z - p1.z)) + plane2.d = 0

-- Statement of the theorem
theorem parallel_segments_between_parallel_planes_are_equal
  (plane1 plane2 : Plane)
  (p1 p2 p3 p4 : Point3D)
  (h1 : arePlanesParallel plane1 plane2)
  (h2 : isSegmentParallelToPlanes p1 p2 plane1 plane2)
  (h3 : isSegmentParallelToPlanes p3 p4 plane1 plane2) :
  distance p1 p2 = distance p3 p4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_segments_between_parallel_planes_are_equal_l1260_126093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_minimum_l1260_126015

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := x^3 - x^2 + a*x + b

-- Define the derivative of f(x)
def f_derivative (a x : ℝ) : ℝ := 3*x^2 - 2*x + a

theorem tangent_line_and_minimum :
  ∃ a b : ℝ,
    (f_derivative a 0 = -1) ∧  -- Slope of tangent line at x=0 is -1
    (f a b 0 = 1) ∧           -- f(0) = 1, as the tangent line passes through (0,1)
    (a = -1 ∧ b = 1) ∧        -- Part 1: Values of a and b
    (∀ x ∈ Set.Icc (-2) 2, f (-1) 1 x ≥ -9) ∧  -- Part 2: Minimum value on [-2,2]
    (∃ x ∈ Set.Icc (-2) 2, f (-1) 1 x = -9)    -- The minimum is achieved
:= by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_minimum_l1260_126015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_constants_l1260_126016

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + 3*x + 1

-- Define the inverse function f_inv
noncomputable def f_inv (x a b c : ℝ) : ℝ :=
  ((x - a + Real.sqrt (x^2 - b*x + c))/2)^(1/3) + 
  ((x - a - Real.sqrt (x^2 - b*x + c))/2)^(1/3)

-- State the theorem
theorem inverse_function_constants :
  ∀ (a b c : ℝ), 
    (a > 0 ∧ b > 0 ∧ c > 0) →
    (∀ x : ℝ, f_inv (f x) a b c = x) →
    a + 10*b + 100*c = 521 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_constants_l1260_126016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_unit_sum_2013_l1260_126067

noncomputable def imaginary_unit_sum (n : ℕ) : ℂ :=
  Finset.sum (Finset.range (n + 1)) (fun k => (Complex.I : ℂ) ^ k)

theorem imaginary_unit_sum_2013 :
  imaginary_unit_sum 2013 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_unit_sum_2013_l1260_126067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l1260_126081

/-- The eccentricity of an ellipse with semi-major axis a and semi-minor axis b -/
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - (b^2 / a^2))

/-- The ellipse equation -/
def is_ellipse (x y a b : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1

theorem ellipse_eccentricity :
  let a : ℝ := 4
  let b : ℝ := 2 * Real.sqrt 2
  eccentricity a b = Real.sqrt 2 / 2 ∧
  ∀ x y : ℝ, is_ellipse x y a b ↔ (x^2 / 16) + (y^2 / 8) = 1 :=
by sorry

#check ellipse_eccentricity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l1260_126081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_q_value_at_zeros_distinct_zeros_q_ratio_is_one_l1260_126063

/-- The polynomial f(x) = x^2023 + 19x^2022 + 1 -/
def f (x : ℂ) : ℂ := x^2023 + 19*x^2022 + 1

/-- The zeros of f(x) -/
noncomputable def r : Fin 2023 → ℂ := sorry

/-- The polynomial Q of degree 2023 -/
noncomputable def Q : Polynomial ℂ := sorry

theorem q_value_at_zeros (j : Fin 2023) : Q.eval (r j - (r j)⁻¹) = 0 := by sorry

theorem distinct_zeros (i j : Fin 2023) : i ≠ j → r i ≠ r j := by sorry

theorem q_ratio_is_one : Q.eval 1 / Q.eval (-1) = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_q_value_at_zeros_distinct_zeros_q_ratio_is_one_l1260_126063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exp_of_three_in_24_factorial_l1260_126013

/-- The exponent of 3 in the prime factorization of n! -/
def exp_of_three_in_factorial (n : ℕ) : ℕ :=
  (n / 3) + (n / 9)

/-- 24! is the product of integers from 1 to 24 -/
def factorial_24 : ℕ := Nat.factorial 24

theorem exp_of_three_in_24_factorial : 
  exp_of_three_in_factorial 24 = 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exp_of_three_in_24_factorial_l1260_126013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_count_solution_set_finite_l1260_126002

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := 3 * Real.cos (Real.pi * x)

-- Define the solution set
def solution_set (x : ℝ) : Prop :=
  -3 ≤ x ∧ x ≤ 3 ∧ g (g (g x)) = g x

-- Theorem statement
theorem solution_count : 
  ∃ s : Finset ℝ, s.card = 115 ∧ ∀ x ∈ s, solution_set x := by
  sorry

-- Helper lemma to state that the solution set is finite
theorem solution_set_finite :
  ∃ s : Finset ℝ, ∀ x : ℝ, solution_set x ↔ x ∈ s := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_count_solution_set_finite_l1260_126002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_three_integer_solutions_l1260_126084

theorem range_of_a_for_three_integer_solutions 
  (h1 : ∀ x : ℤ, 3 * (x - 1) ≤ 6 * (x - 2))
  (h2 : ∀ x : ℤ, ∀ a : ℝ, x - a < 2)
  (h3 : ∃! (x₁ x₂ x₃ : ℤ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ 
        (∀ x : ℤ, (3 * (x - 1) ≤ 6 * (x - 2) ∧ x - a < 2) → x = x₁ ∨ x = x₂ ∨ x = x₃)) :
  3 < a ∧ a ≤ 4 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_three_integer_solutions_l1260_126084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_sqrt_2_l1260_126055

-- Define the circle
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 2*x = 0

-- Define the line
def line_equation (x y : ℝ) : Prop := y = x

-- Define the chord length function
noncomputable def chord_length (r d : ℝ) : ℝ := 2 * Real.sqrt (r^2 - d^2)

-- Theorem statement
theorem chord_length_sqrt_2 :
  ∃ (r d : ℝ), r = 1 ∧ d = Real.sqrt 2 / 2 ∧
  chord_length r d = Real.sqrt 2 :=
by
  -- Existential introduction
  use 1, Real.sqrt 2 / 2
  -- Prove the three conjuncts
  constructor
  · rfl  -- r = 1
  constructor
  · rfl  -- d = Real.sqrt 2 / 2
  · -- Prove chord_length 1 (Real.sqrt 2 / 2) = Real.sqrt 2
    unfold chord_length
    -- The rest of the proof would go here
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_sqrt_2_l1260_126055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_is_one_tenth_l1260_126073

-- Define the triangle ABC
variable (A B C : ℝ × ℝ)

-- Define points G, H, I on the sides of the triangle
noncomputable def G : ℝ × ℝ := (2/3 * B.1 + 1/3 * C.1, 2/3 * B.2 + 1/3 * C.2)
noncomputable def H : ℝ × ℝ := (2/3 * C.1 + 1/3 * A.1, 2/3 * C.2 + 1/3 * A.2)
noncomputable def I : ℝ × ℝ := (2/3 * A.1 + 1/3 * B.1, 2/3 * A.2 + 1/3 * B.2)

-- Define the intersection points X, Y, Z
noncomputable def X : ℝ × ℝ := sorry
noncomputable def Y : ℝ × ℝ := sorry
noncomputable def Z : ℝ × ℝ := sorry

-- Define a function to calculate the area of a triangle given three points
noncomputable def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ := 
  (1/2) * abs ((p2.1 - p1.1) * (p3.2 - p1.2) - (p3.1 - p1.1) * (p2.2 - p1.2))

-- State the theorem
theorem area_ratio_is_one_tenth :
  triangleArea X Y Z / triangleArea A B C = 1/10 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_is_one_tenth_l1260_126073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_machines_count_l1260_126049

/-- The number of machines initially working -/
def n : ℕ := sorry

/-- The number of units produced by n machines in 4 days -/
def x : ℚ := sorry

/-- The constant production rate per machine per day -/
noncomputable def rate : ℚ := x / (4 * n)

theorem initial_machines_count :
  (20 * rate * 2 = 2 * x) → n = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_machines_count_l1260_126049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_dimension_change_l1260_126024

theorem rectangle_dimension_change (L W : ℝ) (h : L > 0 ∧ W > 0) :
  let new_L := 1.4 * L
  let new_W := W * (L / new_L)
  let width_decrease_percent := (W - new_W) / W * 100
  ∃ ε > 0, |width_decrease_percent - 28.57| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_dimension_change_l1260_126024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_cosine_inequality_l1260_126031

theorem sine_cosine_inequality (α : ℝ) (h : 0 ≤ α ∧ α ≤ π/2) :
  Real.sin α * Real.cos (α/2) ≤ Real.sin (π/4 + α) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_cosine_inequality_l1260_126031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_i_part_ii_l1260_126012

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
  t.c = Real.sqrt 6 ∧ t.C = 2 * Real.pi / 3

-- Helper function to calculate triangle area
noncomputable def triangle_area (t : Triangle) : ℝ :=
  1 / 2 * t.a * t.b * Real.sin t.C

-- Part I
theorem part_i (t : Triangle) (h : triangle_conditions t) (ha : t.a = Real.sqrt 2) :
  t.b = Real.sqrt 2 := by sorry

-- Part II
theorem part_ii (t : Triangle) (h : triangle_conditions t) (hb : Real.sin t.B = 2 * Real.sin t.A) :
  triangle_area t = 3 * Real.sqrt 3 / 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_i_part_ii_l1260_126012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_largest_angle_l1260_126062

/-- The sum of angles in a hexagon -/
def hexagon_angle_sum : ℝ := 720

/-- Represents the ratio of angles in the hexagon -/
def angle_ratio : List ℝ := [2, 2, 2, 3, 3, 4]

/-- The measure of the largest angle in the hexagon -/
def largest_angle : ℝ := 180

theorem hexagon_largest_angle :
  ∀ (x : ℝ),
  (x > 0) →
  (List.sum (List.map (· * x) angle_ratio) = hexagon_angle_sum) →
  (List.maximum angle_ratio * x = largest_angle) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_largest_angle_l1260_126062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_specific_matrix_l1260_126048

theorem det_specific_matrix :
  let A : Matrix (Fin 3) (Fin 3) ℤ := !![2, -3, 4; 0, 6, -1; 5, -2, 1]
  Matrix.det A = -97 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_specific_matrix_l1260_126048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_5_equals_7_5_l1260_126069

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 5 / (3 - x)

-- Define the inverse function of f
noncomputable def f_inv (x : ℝ) : ℝ := (3 * x - 5) / x

-- Define the function g using f_inv
noncomputable def g (x : ℝ) : ℝ := 1 / (f_inv x) + 7

-- Theorem statement
theorem g_5_equals_7_5 : g 5 = 7.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_5_equals_7_5_l1260_126069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1260_126010

open Real

noncomputable def f (a x : ℝ) : ℝ := (x + a) * log x

noncomputable def g (x : ℝ) : ℝ := x^2 / exp x

noncomputable def m (p q : ℝ) : ℝ := log (min p q)

theorem problem_solution (k : ℝ) (h : k > 0) :
  (∃! x : ℝ, x ∈ Set.Ioo k (k + 1) ∧ f 1 x = g x) ∧
  (∀ a : ℝ, (∀ x : ℝ, x > 0 → f a x = g x) → a = 1) ∧
  (∀ p q : ℝ, p > 0 ∧ q > 0 → m p q ≤ 4 / exp 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1260_126010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_N_satisfies_condition_N_is_smallest_l1260_126052

/-- The number of positive integer divisors of n, including 1 and n itself -/
def d (n : ℕ+) : ℕ := sorry

/-- The function f(n) = d(n) / n^(1/4) -/
noncomputable def f (n : ℕ+) : ℝ := (d n : ℝ) / (n : ℝ)^(1/4 : ℝ)

/-- N is the smallest positive integer satisfying the given condition -/
def N : ℕ+ := 13824

/-- Theorem stating that N satisfies the required condition -/
theorem N_satisfies_condition :
  ∀ (n : ℕ+), n ≠ N → ¬(N ∣ n) → f N ≥ f n := by
  sorry

/-- Theorem stating that N is the smallest positive integer satisfying the condition -/
theorem N_is_smallest :
  ∀ (m : ℕ+), m < N → ∃ (n : ℕ+), n ≠ m ∧ ¬(m ∣ n) ∧ f m < f n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_N_satisfies_condition_N_is_smallest_l1260_126052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inclined_plane_acceleration_l1260_126038

/-- Acceleration of a mass system on an inclined plane -/
theorem inclined_plane_acceleration
  (h : ℝ) -- height of slope
  (l : ℝ) -- length of slope
  (m₁ : ℝ) -- mass on slope
  (m₂ : ℝ) -- hanging mass
  (g : ℝ) -- acceleration due to gravity
  (h_pos : h > 0)
  (l_pos : l > 0)
  (m₁_pos : m₁ > 0)
  (m₂_pos : m₂ > 0)
  (g_pos : g > 0)
  (h_val : h = 1.2)
  (l_val : l = 4.8)
  (m₁_val : m₁ = 14.6)
  (m₂_val : m₂ = 2.2)
  (g_val : g = 980.8) :
  ∃ (a : ℝ), abs (a - 84.7) < 0.1 ∧ a > 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inclined_plane_acceleration_l1260_126038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_corner_distance_is_10_l1260_126027

-- Define the perimeter of the smaller square
def small_square_perimeter : ℝ := 8

-- Define the area of the larger square
def large_square_area : ℝ := 64

-- Define the side length of the smaller square
noncomputable def small_square_side : ℝ := small_square_perimeter / 4

-- Define the side length of the larger square
noncomputable def large_square_side : ℝ := Real.sqrt large_square_area

-- Define the distance between corners
noncomputable def corner_distance : ℝ := Real.sqrt ((large_square_side - small_square_side)^2 + large_square_side^2)

-- Theorem statement
theorem corner_distance_is_10 : corner_distance = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_corner_distance_is_10_l1260_126027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_orthogonality_sum_l1260_126008

open Matrix

def N (x y w : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  !![0, 3*y, w;
     x, 2*y, -w;
     x, -2*y, w]

theorem matrix_orthogonality_sum (x y w : ℝ) :
  (N x y w)ᵀ * (N x y w) = 1 → x^2 + y^2 + w^2 = 91/102 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_orthogonality_sum_l1260_126008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_amount_before_brokerage_l1260_126088

noncomputable def cash_realized : ℝ := 101.25

noncomputable def brokerage_rate : ℝ := 0.0025 -- 1/4% expressed as a decimal

noncomputable def total_amount : ℝ := cash_realized / (1 - brokerage_rate)

theorem total_amount_before_brokerage :
  abs (total_amount - 101.56) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_amount_before_brokerage_l1260_126088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_example_l1260_126044

/-- The area of a trapezium with given parallel sides and height. -/
noncomputable def trapeziumArea (a b h : ℝ) : ℝ := (a + b) * h / 2

/-- Theorem: The area of a trapezium with parallel sides of 10 cm and 18 cm,
    and a height of 15 cm, is 210 square centimeters. -/
theorem trapezium_area_example : trapeziumArea 10 18 15 = 210 := by
  -- Unfold the definition of trapeziumArea
  unfold trapeziumArea
  -- Perform the calculation
  simp [add_mul, mul_div_right_comm]
  -- The rest of the proof is omitted
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_example_l1260_126044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_tangency_points_l1260_126042

/-- Represents an inscribed circle in a triangle with sides a, b, and c. -/
def inscribed_circle (a b c : ℝ) : Set (ℝ × ℝ) := sorry

/-- Checks if a point is a point of tangency for the given circle. -/
def is_point_of_tangency (p : ℝ × ℝ) (circle : Set (ℝ × ℝ)) : Prop := sorry

/-- Calculates the area of a triangle given its three vertices. -/
def area_triangle (A B C : ℝ × ℝ) : ℝ := sorry

/-- Given a triangle with sides a, b, and c, and an inscribed circle,
    this theorem proves the formula for the area of the triangle formed by the points of tangency. -/
theorem area_triangle_tangency_points (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  let s := (a + b + c) / 2
  let T := (2 * (s - a) * (s - b) * (s - c) / (a * b * c)) * Real.sqrt (s * (s - a) * (s - b) * (s - c))
  ∃ (A₁ B₁ C₁ : ℝ × ℝ), 
    T = area_triangle A₁ B₁ C₁ ∧ 
    is_point_of_tangency A₁ (inscribed_circle a b c) ∧
    is_point_of_tangency B₁ (inscribed_circle a b c) ∧
    is_point_of_tangency C₁ (inscribed_circle a b c) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_tangency_points_l1260_126042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_distance_theorem_l1260_126023

noncomputable def line1 (x y : ℝ) : ℝ := -9*x - 12*y + 24
noncomputable def line2 (x y m : ℝ) : ℝ := 3*x + 4*y + m

noncomputable def distance_between_lines (m : ℝ) : ℝ :=
  |m + 8| / (Real.sqrt (3^2 + 4^2))

theorem line_distance_theorem (m : ℝ) :
  (∀ x y, line1 x y = 0 ∧ line2 x y m = 0) →
  distance_between_lines m = 1 →
  m = -3 ∨ m = -13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_distance_theorem_l1260_126023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_clock_actual_time_l1260_126047

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  valid : minutes < 60 := by sorry

/-- Converts Time to minutes since midnight -/
def timeToMinutes (t : Time) : ℕ :=
  t.hours * 60 + t.minutes

/-- Converts minutes since midnight to Time -/
def minutesToTime (m : ℕ) : Time where
  hours := m / 60
  minutes := m % 60
  valid := by sorry

theorem car_clock_actual_time 
  (initialTime : Time)
  (meetingDuration : ℕ)
  (carClockGain : ℕ)
  (laterCarClockTime : Time)
  (h1 : initialTime = ⟨14, 0, by sorry⟩)
  (h2 : meetingDuration = 40)
  (h3 : carClockGain = 10)
  (h4 : laterCarClockTime = ⟨20, 0, by sorry⟩)
  : 
  let actualLaterTime := minutesToTime (timeToMinutes initialTime + 
    (timeToMinutes laterCarClockTime - timeToMinutes initialTime) * meetingDuration / (meetingDuration + carClockGain))
  actualLaterTime = ⟨20, 24, by sorry⟩ := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_clock_actual_time_l1260_126047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_y_l1260_126057

open Real

-- Define the function y
noncomputable def y (θ : ℝ) : ℝ := tan θ + (cos (2 * θ) + 1) / sin (2 * θ)

-- State the theorem
theorem min_value_of_y :
  ∀ θ : ℝ, 0 < θ → θ < π / 2 → y θ ≥ 2 ∧ ∃ θ₀ : ℝ, 0 < θ₀ ∧ θ₀ < π / 2 ∧ y θ₀ = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_y_l1260_126057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_moves_for_train_passing_l1260_126065

/-- Represents a train with a locomotive and wagons -/
structure Train where
  locomotive : ℕ
  wagons : List ℕ

/-- Represents the railway system -/
structure RailwaySystem where
  main_track : List ℕ
  siding : Option ℕ

/-- Represents a move in the train passing problem -/
inductive Move
  | LocoForward
  | LocoBackward
  | PushToSiding
  | PullFromSiding

/-- Counts the number of direction changes in a list of moves -/
def count_direction_changes (moves : List Move) : ℕ := sorry

/-- Checks if a sequence of moves is valid according to the problem rules -/
def is_valid_move_sequence (initial_state : RailwaySystem) (moves : List Move) : Bool := sorry

/-- Checks if the trains have successfully passed each other -/
def trains_passed (initial_state : RailwaySystem) (final_state : RailwaySystem) : Bool := sorry

/-- Applies a list of moves to a railway system -/
def apply_moves (initial_state : RailwaySystem) (moves : List Move) : RailwaySystem := sorry

/-- The main theorem stating that the minimum number of moves is 14 -/
theorem min_moves_for_train_passing :
  ∀ (left_train right_train : Train) (initial_state : RailwaySystem),
    (∃ (moves : List Move),
      is_valid_move_sequence initial_state moves ∧
      trains_passed initial_state (apply_moves initial_state moves) ∧
      count_direction_changes moves = 14) ∧
    (∀ (moves : List Move),
      is_valid_move_sequence initial_state moves →
      trains_passed initial_state (apply_moves initial_state moves) →
      count_direction_changes moves ≥ 14) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_moves_for_train_passing_l1260_126065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_coordinates_l1260_126028

/-- Represents a parabola with equation y = 2px^2 -/
structure Parabola where
  p : ℝ

/-- The focus of a parabola -/
noncomputable def focus (par : Parabola) : ℝ × ℝ := (0, 1 / (16 * par.p))

/-- Theorem: For a parabola y = 2px^2 passing through (1, 4), its focus is (0, 1/16) -/
theorem parabola_focus_coordinates (par : Parabola) 
  (h : 4 = 2 * par.p * 1^2) : -- Condition that parabola passes through (1, 4)
  focus par = (0, 1/16) := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_coordinates_l1260_126028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_excess_weight_is_73_75_l1260_126051

/-- The weight that the bridge can hold --/
noncomputable def bridge_limit : ℝ := 140

/-- Kelly's weight in kilograms --/
noncomputable def kelly_weight : ℝ := 30

/-- Daisy's weight in kilograms --/
noncomputable def daisy_weight : ℝ := 24

/-- Sam's weight in kilograms --/
noncomputable def sam_weight : ℝ := 3 * daisy_weight

/-- Mike's weight in kilograms --/
noncomputable def mike_weight : ℝ := 1.5 * kelly_weight

/-- Megan's weight in kilograms --/
noncomputable def megan_weight : ℝ := (kelly_weight + daisy_weight + sam_weight + mike_weight) / 4

/-- The total weight of all five children --/
noncomputable def total_weight : ℝ := kelly_weight + daisy_weight + sam_weight + mike_weight + megan_weight

/-- Theorem stating that the excess weight is 73.75 kilograms --/
theorem excess_weight_is_73_75 : total_weight - bridge_limit = 73.75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_excess_weight_is_73_75_l1260_126051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_multiple_4_or_6_l1260_126006

def is_multiple_of_4_or_6 (n : Nat) : Bool :=
  n % 4 = 0 || n % 6 = 0

def count_multiples (n : Nat) : Nat :=
  (Finset.range n).filter (fun x => is_multiple_of_4_or_6 (x + 1)) |>.card

theorem probability_multiple_4_or_6 :
  (count_multiples 60 : Rat) / 60 = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_multiple_4_or_6_l1260_126006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vitamin_d_scientific_notation_l1260_126034

noncomputable def scientific_notation (a : ℝ) (n : ℤ) : ℝ := a * (10 : ℝ) ^ n

theorem vitamin_d_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 
    1 ≤ a ∧ a < 10 ∧ 
    scientific_notation a n = (0.0000046 : ℝ) ∧
    a = (4.6 : ℝ) ∧ n = -6 := by
  sorry

#check vitamin_d_scientific_notation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vitamin_d_scientific_notation_l1260_126034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_gums_proof_l1260_126019

/-- The expected number of chewing gums needed to collect all n distinct inserts. -/
def expectedGums (n : ℕ) : ℚ :=
  n * (Finset.range n).sum (fun k => 1 / (k + 1 : ℚ))

/-- 
Theorem: The expected number of chewing gums needed to collect all n distinct inserts, 
where each insert appears with probability 1/n, is equal to n * H_n.
-/
theorem expected_gums_proof (n : ℕ) (h : n > 0) : 
  expectedGums n = n * (Finset.range n).sum (fun k => 1 / (k + 1 : ℚ)) :=
by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_gums_proof_l1260_126019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_l1260_126076

-- Define the cubic function
noncomputable def f (x : ℝ) : ℝ := x^3 - 2*x^2 + x + 1

-- Define the linear function
noncomputable def g (x : ℝ) : ℝ := (5 - x) / 5

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) := {p | f p.1 = g p.1}

-- Theorem statement
theorem intersection_sum : 
  ∃ (p₁ p₂ p₃ : ℝ × ℝ), p₁ ∈ intersection_points ∧ p₂ ∈ intersection_points ∧ p₃ ∈ intersection_points ∧
  p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₂ ≠ p₃ ∧
  p₁.1 + p₂.1 + p₃.1 = 2 ∧ p₁.2 + p₂.2 + p₃.2 = 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_l1260_126076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_difference_l1260_126080

-- Define the function f(x) = x ln x - x
noncomputable def f (x : ℝ) : ℝ := x * Real.log x - x

-- Define the interval [1/2, 2]
def interval : Set ℝ := { x | 1/2 ≤ x ∧ x ≤ 2 }

-- State the theorem
theorem max_min_difference (M N : ℝ) 
  (hM : IsMaxOn f interval M) 
  (hN : IsMinOn f interval N) : 
  M - N = 2 * Real.log 2 - 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_difference_l1260_126080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_non_monotonic_l1260_126026

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a^2 * x^3 + a * x^2 - x

-- Define the derivative of f(x)
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3 * a^2 * x^2 + 2 * a * x - 1

-- Define the interval
def interval : Set ℝ := Set.Icc 1 3

-- Define the property of being not monotonic on the interval
def not_monotonic (a : ℝ) : Prop :=
  ∃ x y, x ∈ interval ∧ y ∈ interval ∧ x < y ∧ f a x > f a y

-- Theorem statement
theorem range_of_a_for_non_monotonic :
  {a : ℝ | not_monotonic a} = {a | -1 < a ∧ a < -1/3} ∪ {a | 1/9 < a ∧ a < 1/3} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_non_monotonic_l1260_126026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orchard_growth_period_l1260_126061

/-- The number of years it takes for trees to grow from initial to final count -/
noncomputable def years_to_grow (initial : ℕ) (final : ℕ) : ℕ :=
  ⌊(Real.log (final / initial : ℝ)) / (Real.log (5 / 4))⌋.toNat

/-- Theorem stating that it takes 4 years for 512 trees to grow to 1250 trees -/
theorem orchard_growth_period : years_to_grow 512 1250 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orchard_growth_period_l1260_126061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_P_l1260_126032

noncomputable def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 4

noncomputable def point_P : ℝ × ℝ := (-1, Real.sqrt 3)

noncomputable def tangent_line_equation (x y : ℝ) : Prop := x - Real.sqrt 3 * y + 4 = 0

theorem tangent_line_at_P :
  circle_equation point_P.1 point_P.2 →
  ∃ (m b : ℝ), ∀ (x y : ℝ),
    (y = m * x + b ↔ tangent_line_equation x y) ∧
    (∀ (x' y' : ℝ), circle_equation x' y' → (y' - point_P.2) = m * (x' - point_P.1) → x' = point_P.1 ∧ y' = point_P.2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_P_l1260_126032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seating_arrangements_l1260_126004

def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem seating_arrangements (n : Nat) (h : n = 8) : 
  (factorial n - factorial (n - 1) * factorial 2) = 30240 :=
by
  sorry

def number_of_seating_arrangements (n : Nat) (m : Nat) : Nat :=
  factorial n - factorial (n - 1) * factorial m

#eval number_of_seating_arrangements 8 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seating_arrangements_l1260_126004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_min_translation_l1260_126083

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x)^2 - 2 * Real.sin x * Real.cos x + 1

-- Theorem 1
theorem sum_of_roots (x₁ x₂ : ℝ) (h₁ : 0 < x₁ ∧ x₁ < π) (h₂ : 0 < x₂ ∧ x₂ < π)
  (h₃ : f x₁ = 1) (h₄ : f x₂ = 1) (h₅ : x₁ ≠ x₂) : x₁ + x₂ = 3 * π / 4 := by
  sorry

-- Theorem 2
theorem min_translation (m : ℝ) (h : m > 0)
  (h_sym : ∀ x, f (x + m) = f (-x + m)) : m ≥ π / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_min_translation_l1260_126083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1260_126033

/-- The function f(x) = 1 - √(x^2 + 1) -/
noncomputable def f (x : ℝ) : ℝ := 1 - Real.sqrt (x^2 + 1)

/-- The function g(x) = ln(ax^2 - 2x + 1) -/
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := Real.log (a * x^2 - 2 * x + 1)

/-- Theorem stating that if for any x₁ ∈ ℝ, there exists a real number x₂ such that
    f(x₁) = g(x₂), then a ≤ 1 -/
theorem range_of_a (a : ℝ) : 
  (∀ x₁ : ℝ, ∃ x₂ : ℝ, f x₁ = g a x₂) → a ≤ 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1260_126033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_inradius_median_inequality_l1260_126030

theorem right_triangle_inradius_median_inequality (a b c r sa sb : ℝ) 
  (h_right : a^2 + b^2 = c^2)
  (h_inradius : r = (a + b - c) / 2)
  (h_median_a : sa^2 = (2*b^2 + 2*c^2 - a^2) / 4)
  (h_median_b : sb^2 = (2*a^2 + 2*c^2 - b^2) / 4)
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) :
  r^2 / (sa^2 + sb^2) ≤ (3 - 2 * Real.sqrt 2) / 5 := by
  sorry

#check right_triangle_inradius_median_inequality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_inradius_median_inequality_l1260_126030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_l_properties_l1260_126003

noncomputable section

/-- The line l with equation (a+1)x + y - 5 - 2a = 0 (a ∈ ℝ) -/
def line_l (a : ℝ) (x y : ℝ) : Prop := (a + 1) * x + y - 5 - 2 * a = 0

/-- The fixed point P that the line l passes through -/
def fixed_point (P : ℝ × ℝ) : Prop :=
  ∀ a : ℝ, line_l a P.1 P.2

/-- The line l intersects the positive x-axis at point A -/
def intersect_x_axis (a : ℝ) (A : ℝ × ℝ) : Prop :=
  line_l a A.1 A.2 ∧ A.2 = 0 ∧ A.1 > 0

/-- The line l intersects the positive y-axis at point B -/
def intersect_y_axis (a : ℝ) (B : ℝ × ℝ) : Prop :=
  line_l a B.1 B.2 ∧ B.1 = 0 ∧ B.2 > 0

/-- The area of triangle AOB -/
def triangle_area (A B : ℝ × ℝ) : ℝ :=
  (A.1 * B.2) / 2

theorem line_l_properties :
  ∃ P : ℝ × ℝ, fixed_point P ∧ P = (2, 3) ∧
  ∃ a : ℝ, ∃ A B : ℝ × ℝ,
    intersect_x_axis a A ∧ intersect_y_axis a B ∧
    (∀ a' : ℝ, ∀ A' B' : ℝ × ℝ,
      intersect_x_axis a' A' ∧ intersect_y_axis a' B' →
      triangle_area A B ≤ triangle_area A' B') ∧
    line_l a 3 4 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_l_properties_l1260_126003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jack_remaining_money_l1260_126001

noncomputable def initial_money : ℝ := 100
def initial_bottles : ℕ := 4
noncomputable def bottle_cost : ℝ := 2
noncomputable def cheese_pound_cost : ℝ := 10
noncomputable def cheese_amount : ℝ := 1/2

def total_bottles : ℕ := initial_bottles + 2 * initial_bottles

noncomputable def water_cost : ℝ := bottle_cost * (total_bottles : ℝ)
noncomputable def cheese_cost : ℝ := cheese_amount * cheese_pound_cost
noncomputable def total_spent : ℝ := water_cost + cheese_cost

theorem jack_remaining_money :
  initial_money - total_spent = 71 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jack_remaining_money_l1260_126001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l1260_126091

-- Define the speed of the train in km/hr
noncomputable def train_speed_kmh : ℝ := 30

-- Define the length of the train in meters
noncomputable def train_length_m : ℝ := 200

-- Define the conversion factor from km/hr to m/s
noncomputable def kmh_to_ms : ℝ := 1000 / 3600

-- Define the time taken to cross the pole
noncomputable def time_to_cross : ℝ := train_length_m / (train_speed_kmh * kmh_to_ms)

-- Theorem statement
theorem train_crossing_time :
  (time_to_cross ≥ 24) ∧ (time_to_cross < 24.02) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l1260_126091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_existence_negation_of_exponential_inequality_l1260_126079

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x, P x) ↔ (∀ x, ¬ P x) :=
by sorry

theorem negation_of_exponential_inequality :
  (¬ ∃ x : ℝ, (2 : ℝ)^x < 1) ↔ (∀ x : ℝ, (2 : ℝ)^x ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_existence_negation_of_exponential_inequality_l1260_126079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_2015_value_l1260_126090

noncomputable def f (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 0 => (x^2 + 2*x + 1) * Real.exp x
  | n+1 => deriv (f n) x

noncomputable def a (n : ℕ) : ℝ := sorry
noncomputable def b (n : ℕ) : ℝ := sorry
noncomputable def c (n : ℕ) : ℝ := sorry

theorem b_2015_value :
  (∀ n : ℕ, n ≥ 1 → f n = fun x => (a n * x^2 + b n * x + c n) * Real.exp x) →
  b 2015 = 4030 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_2015_value_l1260_126090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l1260_126022

/-- The curve f defined piecewise -/
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 5 then x
  else if x ≤ 8 then 2*x - 5
  else 0

/-- The region bounded by x-axis, x=8, and curve f -/
def region : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ 8 ∧ 0 ≤ p.2 ∧ p.2 ≤ f p.1}

/-- The area of the region -/
noncomputable def k : ℝ := sorry

/-- Theorem: The area of the region is 36.5 -/
theorem area_of_region : k = 36.5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l1260_126022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_increasing_interval_l1260_126035

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin x * Real.cos x + Real.sin x ^ 2

noncomputable def g (x : ℝ) : ℝ := f ((x - Real.pi / 6) / 2)

theorem g_increasing_interval :
  ∃ (a b : ℝ), a = -Real.pi / 6 ∧ b = 5 * Real.pi / 6 ∧
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → g x < g y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_increasing_interval_l1260_126035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_altitude_intersection_theorem_l1260_126059

/-- Represents an acute triangle with vertices A, B, and C -/
def AcuteTriangle (A B C : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

/-- Represents that AP is an altitude of triangle ABC from vertex A -/
def Altitude (A P B C : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

/-- Represents that two line segments intersect at a point -/
def IntersectsAt (A P B Q H : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

/-- Represents the length of a line segment between two points -/
noncomputable def SegmentLength (P Q : EuclideanSpace ℝ (Fin 2)) : ℝ := sorry

/-- Given an acute triangle ABC with altitudes AP and BQ intersecting at point H,
    if HP = 3 and HQ = 4, then (BP)(PC) - (AQ)(QC) = -7 -/
theorem altitude_intersection_theorem (A B C P Q H : EuclideanSpace ℝ (Fin 2)) :
  AcuteTriangle A B C →
  Altitude A P B C →
  Altitude B Q A C →
  IntersectsAt A P B Q H →
  SegmentLength H P = 3 →
  SegmentLength H Q = 4 →
  SegmentLength B P * SegmentLength P C - SegmentLength A Q * SegmentLength Q C = -7 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_altitude_intersection_theorem_l1260_126059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_range_g_monotonically_decreasing_l1260_126021

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x + 1

-- Part 1: Range of c
theorem f_inequality_range (c : ℝ) :
  (∀ x > 0, f x ≤ 2 * x + c) ↔ c ≥ -1 := by sorry

-- Define the function g
noncomputable def g (a x : ℝ) : ℝ := (f x - f a) / (x - a)

-- Part 2: Monotonicity of g
theorem g_monotonically_decreasing (a : ℝ) (h : a > 0) :
  ∀ x > 0, x ≠ a → (deriv (g a)) x < 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_range_g_monotonically_decreasing_l1260_126021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_effective_decontamination_duration_minimum_second_release_minimum_a_value_l1260_126050

-- Define the piecewise function f(x)
noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 4 then 16 / (8 - x) - 1
  else if 4 < x ∧ x ≤ 10 then 5 - x / 2
  else 0

-- Define the concentration function for a single release
noncomputable def concentration (a : ℝ) (x : ℝ) : ℝ := a * f x

-- Theorem for part 1
theorem effective_decontamination_duration (h : 1 ≤ 4 ∧ 4 ≤ 4) :
  ∀ x, 0 ≤ x ∧ x ≤ 8 → concentration 4 x ≥ 4 := by sorry

-- Theorem for part 2
theorem minimum_second_release (h : 1 ≤ 24 - 16 * Real.sqrt 2 ∧ 24 - 16 * Real.sqrt 2 ≤ 4) :
  ∀ x, 6 ≤ x ∧ x ≤ 10 →
    concentration 2 x + concentration (24 - 16 * Real.sqrt 2) (x - 6) ≥ 4 := by sorry

-- Theorem to prove the minimum value of a
theorem minimum_a_value :
  ∀ a, a < 24 - 16 * Real.sqrt 2 →
    ∃ x, 6 ≤ x ∧ x ≤ 10 ∧ concentration 2 x + concentration a (x - 6) < 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_effective_decontamination_duration_minimum_second_release_minimum_a_value_l1260_126050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_attendees_receive_all_items_l1260_126098

/-- The number of attendees receiving all three promotional items -/
def attendees_with_all_items (capacity : ℕ) (poster_interval : ℕ) (program_interval : ℕ) (drink_interval : ℕ) : ℕ :=
  capacity / (Nat.lcm (Nat.lcm poster_interval program_interval) drink_interval)

/-- Theorem stating that 5 attendees receive all three items given the specified conditions -/
theorem five_attendees_receive_all_items :
  attendees_with_all_items 5000 100 45 60 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_attendees_receive_all_items_l1260_126098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_log_base_2_l1260_126086

-- Define the function f(x) = log₂x
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 2

-- State the theorem about the domain of f
theorem domain_of_log_base_2 :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x > 0} :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_log_base_2_l1260_126086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_bounded_figure_l1260_126060

-- Define the curve
def curve (y : ℝ) : ℝ := y^(2/3)

-- Define the area function
noncomputable def area : ℝ := ∫ y in Set.Icc 0 1, curve y

-- Theorem statement
theorem area_of_bounded_figure : area = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_bounded_figure_l1260_126060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_ABC_l1260_126092

open Real

-- Define the intersection points of sin x and cos x
def intersection_points : Set (ℝ × ℝ) := {(x, y) | sin x = cos x ∧ y = sin x}

-- Define three consecutive intersection points
noncomputable def A : ℝ × ℝ := sorry
noncomputable def B : ℝ × ℝ := sorry
noncomputable def C : ℝ × ℝ := sorry

-- Define the triangle ABC
def triangle_ABC : Set (ℝ × ℝ) := {A, B, C}

-- Define the area of a triangle (placeholder function)
noncomputable def area_of_triangle (triangle : Set (ℝ × ℝ)) : ℝ := sorry

-- State the theorem
theorem area_of_triangle_ABC : 
  area_of_triangle triangle_ABC = Real.sqrt 2 * π :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_ABC_l1260_126092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_sum_l1260_126039

noncomputable def h (x : ℝ) : ℝ := 3 / (1 + 9 * x^2)

theorem range_sum (a b : ℝ) : 
  (∀ y, y ∈ Set.Ioo a b → ∃ x, h x = y) → 
  (∀ y, y ≤ b → ∃ x, h x = y) → 
  (∀ y, y > b → ∀ x, h x ≠ y) → 
  a + b = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_sum_l1260_126039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_when_a_is_1_g_is_minimum_of_f_smallest_m_for_g_l1260_126071

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x - a - 1

-- Define the domain of x
def X : Set ℝ := { x | 0 ≤ x ∧ x ≤ 2 }

-- Define the minimum value function g
noncomputable def g (a : ℝ) : ℝ :=
  if a ≥ 0 then -a - 1
  else if a ≤ -2 then 3*a + 3
  else -a^2 - a

-- Theorem 1: Range of f when a = 1
theorem range_of_f_when_a_is_1 :
  Set.range (fun x => f 1 x) = { y | -2 ≤ y ∧ y ≤ 6 } := by sorry

-- Theorem 2: g(a) is the minimum value of f(x) for x in [0,2]
theorem g_is_minimum_of_f :
  ∀ a x, x ∈ X → g a ≤ f a x := by sorry

-- Theorem 3: The smallest integer m such that g(a) - m ≤ 0 for all a ∈ ℝ is 0
theorem smallest_m_for_g :
  (∃ m : ℤ, (∀ a : ℝ, g a - ↑m ≤ 0) ∧
    (∀ m' : ℤ, (∀ a : ℝ, g a - ↑m' ≤ 0) → m ≤ m')) ∧
  (∀ a : ℝ, g a ≤ 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_when_a_is_1_g_is_minimum_of_f_smallest_m_for_g_l1260_126071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1260_126078

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  -- Given conditions
  Real.cos C = Real.sqrt 3 / 3 →
  a = 3 →
  (b - a) * (Real.sin B + Real.sin A) = (b - c) * Real.sin C →
  -- Conclusions
  Real.sin B = (3 + Real.sqrt 6) / 6 ∧
  (1 / 2) * a * c * Real.sin B = (3 * Real.sqrt 2 + 2 * Real.sqrt 3) / 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1260_126078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombusAreaApprox_l1260_126014

/-- The area of a rhombus with side length 8 cm and an included angle of 55 degrees -/
noncomputable def rhombusArea : ℝ :=
  let sideLength : ℝ := 8
  let includedAngle : ℝ := 55 * Real.pi / 180  -- Convert degrees to radians
  let diagonal1 : ℝ := 2 * sideLength * Real.cos (includedAngle / 2)
  let diagonal2 : ℝ := 2 * sideLength * Real.sin (includedAngle / 2)
  (diagonal1 * diagonal2) / 2

/-- Theorem stating that the area of the rhombus is approximately 53.288 square centimeters -/
theorem rhombusAreaApprox : ‖rhombusArea - 53.288‖ < 0.001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombusAreaApprox_l1260_126014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_black_cells_count_l1260_126075

/-- Represents a 4x4 square grid where each cell can be either black or white -/
def Grid := Fin 4 → Fin 4 → Bool

/-- Counts the number of black cells in a 2x2 sub-square starting at (i, j) -/
def countBlackInSubSquare (grid : Grid) (i j : Fin 2) : Nat :=
  (Finset.sum (Finset.univ : Finset (Fin 2)) fun x =>
    Finset.sum (Finset.univ : Finset (Fin 2)) fun y =>
      if grid (i + x) (j + y) then 1 else 0)

/-- The list of black cell counts in all nine 2x2 sub-squares -/
def subSquareCounts (grid : Grid) : List Nat :=
  List.map (fun ij => countBlackInSubSquare grid ij.1 ij.2)
    [(0, 0), (0, 1), (1, 0), (1, 1)]

/-- Counts the total number of black cells in the entire 4x4 grid -/
def totalBlackCells (grid : Grid) : Nat :=
  (Finset.sum (Finset.univ : Finset (Fin 4)) fun i =>
    Finset.sum (Finset.univ : Finset (Fin 4)) fun j =>
      if grid i j then 1 else 0)

/-- Theorem stating that if the sub-square counts match the given sequence,
    then the total number of black cells is 11 -/
theorem black_cells_count (grid : Grid) :
  subSquareCounts grid = [0, 2, 2, 3, 3, 4, 4, 4, 4] →
  totalBlackCells grid = 11 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_black_cells_count_l1260_126075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_product_line_equation_l1260_126054

-- Define the line l
def line_l (x y : ℝ) : Prop := x + y - 2 = 0

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + 2*x + y^2 = 0

-- Define a point P on line l
structure Point_P where
  x : ℝ
  y : ℝ
  on_line : line_l x y

-- Define the tangent points A and B
structure Tangent_Points where
  A : ℝ × ℝ
  B : ℝ × ℝ
  on_circle_A : circle_C A.1 A.2
  on_circle_B : circle_C B.1 B.2

-- Define the function to be minimized
noncomputable def distance_product (P : Point_P) (AB : Tangent_Points) : ℝ :=
  let C := (-1, 0)  -- Center of the circle
  let PC := ((P.x - C.1)^2 + (P.y - C.2)^2).sqrt
  let AB_length := ((AB.A.1 - AB.B.1)^2 + (AB.A.2 - AB.B.2)^2).sqrt
  PC * AB_length

-- State the theorem
theorem min_distance_product_line_equation (P : Point_P) (AB : Tangent_Points) :
  (∀ P' : Point_P, ∀ AB' : Tangent_Points, 
    distance_product P AB ≤ distance_product P' AB') →
  ∃ a b c : ℝ, a = 3 ∧ b = 3 ∧ c = 1 ∧ 
    ∀ x y : ℝ, (x = AB.A.1 ∧ y = AB.A.2) ∨ (x = AB.B.1 ∧ y = AB.B.2) → 
      a * x + b * y + c = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_product_line_equation_l1260_126054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_non_prime_sequences_l1260_126017

theorem infinitely_many_non_prime_sequences (k : ℕ) : 
  ∃ f : ℕ → ℕ, Function.Injective f ∧ 
    (∀ m : ℕ, ∀ i ∈ Finset.range k, ¬ Nat.Prime (f m + i + 1)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_non_prime_sequences_l1260_126017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l1260_126041

-- Define the quadratic function
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x^2 - m*x + m - 1

-- Part 1: Prove that if f(0) = f(2), then m = 2
theorem part1 (m : ℝ) : f m 0 = f m 2 → m = 2 := by
  intro h
  -- Expand the definition of f
  simp [f] at h
  -- Simplify the equation
  linarith

-- Part 2: Define the minimum value function
noncomputable def min_value (m : ℝ) : ℝ :=
  if m ≤ -4 then 3*m + 3
  else if m < 4 then -m^2/4 + m - 1
  else 3 - m

-- Prove that min_value gives the minimum of f on [-2, 2]
theorem part2 (m : ℝ) : 
  ∀ x ∈ Set.Icc (-2 : ℝ) 2, f m x ≥ min_value m := by
  sorry -- The proof is omitted for brevity, but can be completed with detailed analysis


end NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l1260_126041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_in_second_quadrant_l1260_126077

theorem complex_in_second_quadrant (A B : Real) (h1 : 0 < A) (h2 : A < π / 2) 
  (h3 : 0 < B) (h4 : B < π / 2) (h5 : A + B < π) : 
  let z : ℂ := Complex.mk (Real.cos B - Real.sin A) (Real.sin B - Real.cos A)
  z.re < 0 ∧ z.im > 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_in_second_quadrant_l1260_126077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_area_ratio_l1260_126045

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y² = 8x -/
def Parabola : Point → Prop :=
  fun p => p.y^2 = 8 * p.x

/-- Represents a line with slope √3 passing through (2, 0) -/
def Line : Point → Prop :=
  fun p => p.y = Real.sqrt 3 * (p.x - 2)

/-- The focus of the parabola -/
noncomputable def F : Point :=
  { x := 2, y := 0 }

/-- Point A is the intersection of the line and parabola in the first quadrant -/
noncomputable def A : Point :=
  { x := 6, y := 4 * Real.sqrt 3 }

/-- Point B is the other intersection of the line and parabola -/
noncomputable def B : Point :=
  { x := 2/3, y := 4 * Real.sqrt 3 / 3 }

/-- Point C is the intersection of the line and the parabola's axis -/
noncomputable def C : Point :=
  { x := -2, y := -4 * Real.sqrt 3 }

/-- The origin point -/
def O : Point :=
  { x := 0, y := 0 }

/-- Calculate the area of a triangle given three points -/
noncomputable def triangleArea (p1 p2 p3 : Point) : ℝ :=
  (1/2) * abs ((p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y))

/-- The main theorem to be proved -/
theorem parabola_line_intersection_area_ratio :
  (triangleArea A O C) / (triangleArea B O F) = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_area_ratio_l1260_126045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_consistency_constant_second_differences_l1260_126058

/-- A sequence of numbers representing y-values of a quadratic function -/
def quadratic_sequence : List ℕ := [64, 121, 196, 289, 400, 529, 676, 841]

/-- Function to calculate first differences of a sequence -/
def first_differences (seq : List ℕ) : List ℕ :=
  List.zipWith (·-·) (seq.tail) seq

/-- Function to calculate second differences of a sequence -/
def second_differences (seq : List ℕ) : List ℕ :=
  first_differences (first_differences seq)

/-- Theorem stating that the given sequence is consistent with a quadratic function -/
theorem quadratic_consistency :
  ∃ (a b c : ℚ), a ≠ 0 ∧
  ∀ (n : ℕ), n < quadratic_sequence.length →
    Option.isSome (quadratic_sequence.get? n) ∧
    (quadratic_sequence.get? n).getD 0 =
      Int.natAbs ((a * (n : ℚ)^2 + b * (n : ℚ) + c).floor) :=
by
  sorry

/-- Theorem stating that all second differences of the sequence are constant -/
theorem constant_second_differences :
  ∀ (n : ℕ), n < (second_differences quadratic_sequence).length - 1 →
    (second_differences quadratic_sequence).get? n =
    (second_differences quadratic_sequence).get? (n + 1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_consistency_constant_second_differences_l1260_126058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zachary_pushups_l1260_126085

/-- Given that David did 78 push-ups and 19 more push-ups than Zachary,
    prove that Zachary did 59 push-ups. -/
theorem zachary_pushups (david_pushups zachary_pushups difference : ℕ) 
  (h1 : david_pushups = 78) 
  (h2 : david_pushups = difference + zachary_pushups) 
  (h3 : difference = 19) : zachary_pushups = 59 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zachary_pushups_l1260_126085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_and_line_properties_l1260_126068

-- Define the curve C
noncomputable def curve_C (α : ℝ) : ℝ × ℝ :=
  (Real.sqrt 2 * (Real.sin α - Real.cos α), Real.sqrt 2 / 2 * (Real.sin α + Real.cos α))

-- Define the line l
def line_l (m : ℝ) (x y : ℝ) : Prop :=
  x + 2 * y + m = 0

-- Maximum distance condition
noncomputable def max_distance : ℝ :=
  4 * Real.sqrt 10 / 5

-- Theorem statement
theorem curve_and_line_properties :
  ∀ (m : ℝ),
    (∀ (x y : ℝ), (∃ α, curve_C α = (x, y)) ↔ x^2 / 4 + y^2 = 1) ∧
    (∀ (x y : ℝ), line_l m x y ↔ x + 2*y + m = 0) ∧
    (∃ (x y : ℝ), curve_C (Real.arctan (y/x)) = (x, y) ∧
      |x + 2*y + m| / Real.sqrt 5 = max_distance →
        m = 2 * Real.sqrt 2 ∨ m = -6 * Real.sqrt 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_and_line_properties_l1260_126068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alternating_seq_is_equal_variance_equal_variance_implies_arithmetic_squares_equal_variance_implies_equal_variance_subsequence_equal_variance_and_arithmetic_implies_constant_l1260_126007

/-- Definition of an equal variance sequence -/
def is_equal_variance_seq (a : ℕ → ℝ) (p : ℝ) :=
  ∀ n, n ≥ 2 → a n ^ 2 - a (n - 1) ^ 2 = p

/-- The sequence (-1)ⁿ -/
def alternating_seq (n : ℕ) : ℝ := (-1) ^ n

/-- Statement 1: The sequence {(-1)ⁿ} is an equal variance sequence -/
theorem alternating_seq_is_equal_variance :
  ∃ p, is_equal_variance_seq alternating_seq p := by
  sorry

/-- Statement 2: If {aₙ} is an equal variance sequence, then {aₙ²} is an arithmetic sequence -/
theorem equal_variance_implies_arithmetic_squares {a : ℕ → ℝ} {p : ℝ} 
  (h : is_equal_variance_seq a p) :
  ∃ d, ∀ n, n ≥ 2 → (a n)^2 - (a (n-1))^2 = d := by
  sorry

/-- Statement 3: If {aₙ} is an equal variance sequence, then {aₖₙ} is also an equal variance sequence -/
theorem equal_variance_implies_equal_variance_subsequence {a : ℕ → ℝ} {p : ℝ} 
  (h : is_equal_variance_seq a p) (k : ℕ) (hk : k ≥ 1) :
  ∃ q, is_equal_variance_seq (λ n ↦ a (k * n)) q := by
  sorry

/-- Statement 4: If {aₙ} is both an equal variance sequence and an arithmetic sequence, then it's constant -/
theorem equal_variance_and_arithmetic_implies_constant {a : ℕ → ℝ} {p d : ℝ} 
  (h1 : is_equal_variance_seq a p)
  (h2 : ∀ n, n ≥ 2 → a n - a (n-1) = d) :
  ∃ c, ∀ n, a n = c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alternating_seq_is_equal_variance_equal_variance_implies_arithmetic_squares_equal_variance_implies_equal_variance_subsequence_equal_variance_and_arithmetic_implies_constant_l1260_126007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_from_polar_equation_l1260_126005

-- Define the polar equation
noncomputable def polar_equation (θ : ℝ) : ℝ := 3 * Real.cos θ + 4 * Real.sin θ

-- State the theorem
theorem circle_area_from_polar_equation :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (∀ θ, polar_equation θ = radius * Real.cos θ + radius * Real.sin θ) ∧
    (π * radius^2 = 25 * π / 4) := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_from_polar_equation_l1260_126005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_symmetry_l1260_126056

/-- Direct proportion function -/
noncomputable def direct_prop (k : ℝ) (x : ℝ) : ℝ := k * x

/-- Inverse proportion function -/
noncomputable def inverse_prop (k : ℝ) (x : ℝ) : ℝ := k / x

/-- Theorem: Intersection points of direct and inverse proportion functions -/
theorem intersection_points_symmetry 
  (k₁ k₂ : ℝ) 
  (h1 : direct_prop k₁ 1 = -2) 
  (h2 : inverse_prop k₂ 1 = -2) :
  direct_prop k₁ (-1) = 2 ∧ inverse_prop k₂ (-1) = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_symmetry_l1260_126056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_theorem_l1260_126074

-- Define the trip distance
noncomputable def trip_distance : ℝ := 660

-- Define a function to calculate travel time given speed
noncomputable def travel_time (speed : ℝ) : ℝ := trip_distance / speed

-- State the theorem
theorem average_speed_theorem (v : ℝ) (h1 : v > 0) (h2 : v + 5 > 0) :
  travel_time v - travel_time (v + 5) = 1 → v = 55 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_theorem_l1260_126074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_centroid_cube_volume_ratio_l1260_126089

/-- Regular tetrahedron with side length s -/
structure RegularTetrahedron where
  s : ℝ
  s_pos : s > 0

/-- Cube whose vertices are the centroids of the faces of a regular tetrahedron -/
structure CentroidCube where
  tetrahedron : RegularTetrahedron

/-- The volume ratio of a regular tetrahedron to its centroid cube -/
noncomputable def volume_ratio (t : RegularTetrahedron) (c : CentroidCube) : ℚ :=
  sorry

theorem tetrahedron_centroid_cube_volume_ratio 
  (t : RegularTetrahedron) (c : CentroidCube) (p q : ℕ) 
  (h_cube : c.tetrahedron = t)
  (h_ratio : volume_ratio t c = (p : ℚ) / (q : ℚ))
  (h_coprime : Nat.Coprime p q) :
  p + q = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_centroid_cube_volume_ratio_l1260_126089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_and_round_to_thousandth_l1260_126000

/-- Rounds a real number to the nearest thousandth -/
noncomputable def roundToThousandth (x : ℝ) : ℝ :=
  (⌊x * 1000 + 0.5⌋ : ℝ) / 1000

/-- The sum of 53.463 and 12.9873 rounded to the nearest thousandth is 66.450 -/
theorem sum_and_round_to_thousandth :
  roundToThousandth (53.463 + 12.9873) = 66.450 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_and_round_to_thousandth_l1260_126000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_term_is_one_l1260_126087

def sequence_a : ℕ → ℕ
  | 0 => 1993^(1094^1995)
  | n + 1 => if sequence_a n % 2 = 0 then sequence_a n / 2 else sequence_a n + 7

def is_smallest (m : ℕ) : Prop :=
  m ∈ Set.range sequence_a ∧ ∀ n, sequence_a n ≥ m

theorem smallest_term_is_one :
  is_smallest 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_term_is_one_l1260_126087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_property_l1260_126095

-- Define the function h
noncomputable def h (x : ℝ) : ℝ := Real.sqrt ((x + 5) / 5)

-- State the theorem
theorem h_property : ∀ x : ℝ, h (3 * x) = 3 * h x ↔ x = -20/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_property_l1260_126095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_l1260_126009

/-- Circle C1 with equation x^2 + (y-1)^2 = 1 -/
def C1 (x y : ℝ) : Prop :=
  x^2 + (y - 1)^2 = 1

/-- Circle C2 with equation x^2 - 6x + y^2 - 8y = 0 -/
def C2 (x y : ℝ) : Prop :=
  x^2 - 6*x + y^2 - 8*y = 0

/-- The distance between two points in 2D space -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

/-- Two circles are intersecting if the distance between their centers is greater than
    the absolute difference of their radii and less than the sum of their radii -/
def are_intersecting (c1_center_x c1_center_y c1_radius c2_center_x c2_center_y c2_radius : ℝ) : Prop :=
  let d := distance c1_center_x c1_center_y c2_center_x c2_center_y
  abs (c1_radius - c2_radius) < d ∧ d < c1_radius + c2_radius

/-- Theorem stating that Circle C1 and Circle C2 are intersecting -/
theorem circles_intersect : are_intersecting 0 1 1 3 4 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_l1260_126009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1260_126094

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.sin x, 1/2)

noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos x + Real.sin x, -1)

noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f (x + T) = f x ∧ 
  ∀ (S : ℝ), S > 0 ∧ (∀ (x : ℝ), f (x + S) = f x) → T ≤ S) ∧
  (∀ (x : ℝ), x ∈ Set.Icc (π/4) (π/2) → f x ≤ 1) ∧
  (∃ (x : ℝ), x ∈ Set.Icc (π/4) (π/2) ∧ f x = 1) ∧
  (∀ (x : ℝ), x ∈ Set.Icc (π/4) (π/2) → f x ≥ 1/2) ∧
  (∃ (x : ℝ), x ∈ Set.Icc (π/4) (π/2) ∧ f x = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1260_126094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_prism_volume_l1260_126029

/-- A right prism with an isosceles trapezoidal base -/
structure IsoscelesTrapezoidPrism where
  a : ℝ
  b : ℝ
  α : ℝ
  β : ℝ
  h_a_gt_b : a > b
  h_α_acute : 0 < α ∧ α < π / 2
  h_β_acute : 0 < β ∧ β < π / 2

/-- The volume of an isosceles trapezoid prism -/
noncomputable def volume (p : IsoscelesTrapezoidPrism) : ℝ :=
  (p.a^2 - p.b^2) * (p.a - p.b) / 8 * (Real.tan p.α)^2 * (Real.tan p.β)

theorem isosceles_trapezoid_prism_volume (p : IsoscelesTrapezoidPrism) :
  volume p = (p.a^2 - p.b^2) * (p.a - p.b) / 8 * (Real.tan p.α)^2 * (Real.tan p.β) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_prism_volume_l1260_126029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l1260_126053

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := x + 1/x - 2

-- State the theorem
theorem f_minimum_value :
  (∀ x > 0, f x ≥ 0) ∧ (∃ x > 0, f x = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l1260_126053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_difference_magnitude_l1260_126070

/-- Given two vectors a and b in a real inner product space, with magnitudes and angle between them specified, prove that the magnitude of their difference is 2. -/
theorem vector_difference_magnitude
  {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]
  (a b : V)
  (ha : ‖a‖ = 2)
  (hb : ‖b‖ = Real.sqrt 3)
  (hab : inner a b = ‖a‖ * ‖b‖ * Real.cos (π / 6)) :
  ‖a - 2 • b‖ = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_difference_magnitude_l1260_126070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_M_N_equals_reals_l1260_126036

open Set Real

def M : Set ℝ := {x | x < 1}
def N : Set ℝ := {x | Real.exp (x * Real.log 2) > 1}

theorem union_M_N_equals_reals : M ∪ N = univ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_M_N_equals_reals_l1260_126036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_attention_analysis_l1260_126020

noncomputable section

def k : ℝ := -8

def f (t : ℝ) : ℝ :=
  if 0 < t ∧ t ≤ 10 then -t^2 + 26*t + 80
  else if 10 < t ∧ t ≤ 20 then 240
  else if 20 < t ∧ t ≤ 40 then k*t + 400
  else 0

theorem attention_analysis :
  (k = -8) ∧
  (∀ t ∈ Set.Icc 0 40, f t ≤ 240) ∧
  (∀ t ∈ Set.Icc 10 20, f t = 240) ∧
  (¬ ∃ t1 t2, t1 ∈ Set.Icc 0 40 ∧ t2 ∈ Set.Icc 0 40 ∧ t2 - t1 ≥ 24 ∧
    ∀ t ∈ Set.Icc t1 t2, f t ≥ 185) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_attention_analysis_l1260_126020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_envelope_extra_charge_l1260_126082

structure Envelope where
  length : ℚ
  height : ℚ

def extra_charge (e : Envelope) : Bool :=
  let ratio := e.length / e.height
  ratio < 1.4 || ratio > 2.6

def envelopes : List Envelope := [
  ⟨8, 6⟩,
  ⟨10, 4⟩,
  ⟨7, 5⟩,
  ⟨13, 5⟩,
  ⟨9, 7⟩
]

theorem one_envelope_extra_charge :
  (envelopes.filter extra_charge).length = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_envelope_extra_charge_l1260_126082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_area_l1260_126025

/-- The area of an octagon formed by two overlapping squares -/
theorem octagon_area (side_length : ℝ) (ab_length : ℝ) : 
  side_length = 2 →
  ab_length = 50 / 121 →
  8 * (1 / 2 * ab_length * Real.sqrt 2) = 400 * Real.sqrt 2 / 121 :=
by
  intros h1 h2
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_area_l1260_126025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_and_distance_l1260_126037

/-- The polar equation of line l -/
def polar_equation (p θ : ℝ) : Prop := p * Real.sin (θ - Real.pi/4) = 2 * Real.sqrt 2

/-- The Cartesian equation of the ellipse C -/
def ellipse_equation (x y : ℝ) : Prop := x^2 / 3 + y^2 / 9 = 1

/-- Theorem stating the Cartesian equation of line l and the minimum distance from the ellipse to the line -/
theorem line_and_distance : 
  (∀ x y : ℝ, polar_equation x y ↔ x - y + 4 = 0) ∧ 
  (∃ d : ℝ, d = 2 * Real.sqrt 2 - Real.sqrt 6 ∧
    ∀ P : ℝ × ℝ, ellipse_equation P.1 P.2 → 
      ∀ Q : ℝ × ℝ, Q.1 - Q.2 + 4 = 0 → 
        Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) ≥ d) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_and_distance_l1260_126037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_max_value_range_condition_l1260_126097

open Real

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := log x - m * x

theorem tangent_line_at_one :
  HasDerivAt (f 1) (-1) 1 := by sorry

theorem max_value (m : ℝ) (h : m > 0) :
  ∃ x > 0, ∀ y > 0, f m x ≥ f m y ∧ f m x = -log m - 1 := by sorry

theorem range_condition (m : ℝ) :
  (0 < m ∧ m < 1) ↔ -log m - 1 > m - 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_max_value_range_condition_l1260_126097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_7times_sum_of_digits_l1260_126072

/-- Sum of digits of a positive integer -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

/-- Predicate for a number being 7 times the sum of its digits -/
def is7TimesSum (n : ℕ) : Prop := n = 7 * sumOfDigits n

/-- Decidable instance for is7TimesSum -/
instance (n : ℕ) : Decidable (is7TimesSum n) :=
  show Decidable (n = 7 * sumOfDigits n) from inferInstance

/-- Count of numbers less than 1000 that are 7 times the sum of their digits -/
def countValid : ℕ := (Finset.range 1000).filter is7TimesSum |>.card

theorem count_7times_sum_of_digits : countValid = 4 := by sorry

#eval countValid

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_7times_sum_of_digits_l1260_126072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_translate_f_equals_g_l1260_126018

-- Define the original function f
noncomputable def f (x : ℝ) : ℝ := 2^(-x) + x

-- Define the translation of a function by h units to the right
def translate (h : ℝ) (f : ℝ → ℝ) (x : ℝ) : ℝ := f (x - h)

-- State the theorem
theorem translate_f_equals_g :
  let g := translate 3 f
  ∀ x, g x = 2^(-x+3) + x - 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_translate_f_equals_g_l1260_126018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_guaranteed_success_l1260_126011

/-- The number of colors in the deck of cards -/
def num_colors : ℕ := 2017

/-- The number of cards of each color -/
def cards_per_color : ℕ := 1000000

/-- The strategy function type for the assistant -/
def Strategy := (n : ℕ) → (Fin n → Fin num_colors) → Fin n

/-- The guessing function type for the magician -/
def Guess := (n : ℕ) → (Fin n → Option (Fin num_colors)) → Fin n → Fin num_colors

/-- A theorem stating that 2018 is the smallest number of cards that guarantees
    the magician can always guess correctly -/
theorem smallest_guaranteed_success :
  ∃ (assistant_strategy : Strategy) (magician_guess : Guess),
    (∀ (n : ℕ) (h : n ≥ 2018),
      ∀ (card_colors : Fin n → Fin num_colors),
        magician_guess n
          (fun i ↦ if i = assistant_strategy n card_colors
                then none
                else some (card_colors i))
          (assistant_strategy n card_colors) = 
        card_colors (assistant_strategy n card_colors)) ∧
    (∀ (n : ℕ) (h : n < 2018),
      ∃ (card_colors : Fin n → Fin num_colors),
        ∀ (assistant_strategy : Strategy) (magician_guess : Guess),
          magician_guess n
            (fun i ↦ if i = assistant_strategy n card_colors
                  then none
                  else some (card_colors i))
            (assistant_strategy n card_colors) ≠ 
          card_colors (assistant_strategy n card_colors)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_guaranteed_success_l1260_126011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1260_126099

/-- Given a hyperbola and a circle, if the maximum ratio of the sum of distances
    from a point on both curves to the foci of the hyperbola and the radius of
    the circle is 4√2, then the eccentricity of the hyperbola is 2√2. -/
theorem hyperbola_eccentricity (a b r : ℝ) (ha : 0 < a) (hb : 0 < b) (hr : 0 < r) :
  let C : ℝ × ℝ → Prop := λ p ↦ p.1^2 / a^2 - p.2^2 / b^2 = 1
  let circle : ℝ × ℝ → Prop := λ p ↦ p.1^2 + p.2^2 = r^2
  let F₁ : ℝ × ℝ := (-Real.sqrt (a^2 + b^2), 0)
  let F₂ : ℝ × ℝ := (Real.sqrt (a^2 + b^2), 0)
  let dist := λ (p q : ℝ × ℝ) ↦ Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  (∃ P : ℝ × ℝ, C P ∧ circle P ∧
    (∀ Q : ℝ × ℝ, C Q ∧ circle Q →
      (dist P F₁ + dist P F₂) / r ≥ (dist Q F₁ + dist Q F₂) / r)) →
  (∃ P : ℝ × ℝ, C P ∧ circle P ∧ (dist P F₁ + dist P F₂) / r = 4 * Real.sqrt 2) →
  Real.sqrt (a^2 + b^2) / a = 2 * Real.sqrt 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1260_126099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pasha_can_win_l1260_126040

-- Define the game state
structure GameState where
  pieces : List ℚ
  deriving Repr

-- Define the players' moves
def cut_piece (state : GameState) (index : Nat) : GameState := sorry

def merge_pieces (state : GameState) (index1 index2 : Nat) : GameState := sorry

-- Define the winning condition
def has_equal_pieces (state : GameState) (n : Nat) : Prop := sorry

-- Theorem statement
theorem pasha_can_win (initial_weight : ℚ) :
  ∃ (strategy : Nat → Nat), 
    ∀ (vova_strategy : GameState → Nat × Nat),
    ∃ (final_state : GameState),
      has_equal_pieces final_state 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pasha_can_win_l1260_126040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l1260_126043

theorem inequality_solution_set (x : ℝ) : 
  (x - 1) * (2 * x + 1) ≤ 0 ↔ x ∈ Set.Icc (-1/2 : ℝ) 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l1260_126043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coupon1_best_at_180_l1260_126046

noncomputable def coupon1_discount (price : ℝ) : ℝ := 
  if price ≥ 50 then 0.15 * price else 0

noncomputable def coupon2_discount (price : ℝ) : ℝ := 
  if price ≥ 100 then 25 else 0

def coupon3_discount (price : ℝ) : ℝ := 
  0.25 * (price - 150)

def is_best_coupon1 (price : ℝ) : Prop :=
  coupon1_discount price > coupon2_discount price ∧
  coupon1_discount price > coupon3_discount price

theorem coupon1_best_at_180 :
  is_best_coupon1 180 ∧
  ∀ p, p ∈ ({180, 205, 230, 250, 280} : Set ℝ) → 
    p < 180 → ¬(is_best_coupon1 p) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coupon1_best_at_180_l1260_126046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_sqrt_6_l1260_126064

/-- The circle C -/
def circle_C (x y : ℝ) : Prop := (x + 2)^2 + (y - 2)^2 = 2

/-- The line m passing through A(0,3) with slope 1 -/
def line_m (x y : ℝ) : Prop := y = x + 3

/-- The chord length formed by the intersection of line m and circle C -/
noncomputable def chord_length : ℝ := Real.sqrt 6

theorem chord_length_is_sqrt_6 :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
    line_m x₁ y₁ ∧ line_m x₂ y₂ ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = chord_length^2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_sqrt_6_l1260_126064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_z_l1260_126096

-- Define the complex number z
noncomputable def z : ℂ := (1 + 2*Complex.I) / (1 - Complex.I)^2

-- Theorem stating that the imaginary part of z is 1/2
theorem imaginary_part_of_z : z.im = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_z_l1260_126096

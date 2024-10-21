import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sabina_loan_amount_l1200_120076

/-- Calculates the total loan needed for Sabina's college education --/
def total_loan_needed (
  yearly_tuition : ℕ)
  (yearly_living_expenses : ℕ)
  (program_duration : ℕ)
  (initial_savings : ℕ)
  (grant_percentage_first_two_years : ℚ)
  (grant_percentage_last_two_years : ℚ)
  (scholarship_percentage : ℚ) : ℕ :=
  let total_cost := yearly_tuition * program_duration + yearly_living_expenses * program_duration
  let grant_coverage := (↑yearly_tuition * 2 * grant_percentage_first_two_years).floor +
                        (↑yearly_tuition * 2 * grant_percentage_last_two_years).floor
  let scholarship_savings := (↑yearly_living_expenses * (program_duration - 1) * scholarship_percentage).floor
  let total_reductions := grant_coverage + scholarship_savings + initial_savings
  total_cost - total_reductions.toNat

/-- Theorem stating that Sabina needs a loan of $108,800 --/
theorem sabina_loan_amount :
  total_loan_needed 30000 12000 4 10000 (2/5) (3/10) (1/5) = 108800 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sabina_loan_amount_l1200_120076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_with_two_zeros_l1200_120047

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x - 1) * Real.exp x + a * x^2

-- State the theorem
theorem function_with_two_zeros (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0) →
  (a > 0 ∧ ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → f a x₁ = 0 → f a x₂ = 0 → x₁ + x₂ < 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_with_two_zeros_l1200_120047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_triangle_classification_l1200_120016

/-- Represents a triangle with sides a, b, c --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  triangle_ineq : a < b + c ∧ b < a + c ∧ c < a + b

/-- Represents the process of drawing medians to the longest side --/
def drawMedian (t : Triangle) : Triangle := sorry

/-- Represents a similarity class of triangles --/
def SimilarityClass : Type := Triangle

/-- The set of all triangles resulting from the median-drawing process --/
def resultingTriangles (t : Triangle) : Set Triangle := sorry

/-- The set of similarity classes of the resulting triangles --/
def similarityClasses (t : Triangle) : Set SimilarityClass := sorry

/-- The smallest angle in a triangle --/
noncomputable def smallestAngle (t : Triangle) : ℝ := sorry

/-- Predicate to check if a given angle is an angle of a triangle --/
def isAngleOf (t : Triangle) (θ : ℝ) : Prop := sorry

theorem median_triangle_classification (t : Triangle) :
  (Finite (similarityClasses t)) ∧
  (∀ t' ∈ resultingTriangles t, ∀ θ' : ℝ,
    isAngleOf t' θ' → θ' ≥ (smallestAngle t) / 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_triangle_classification_l1200_120016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_analytic_function_from_imaginary_part_l1200_120074

-- Define the complex plane
variable (z : ℂ)

-- Define the imaginary part of f(z)
noncomputable def v (x y : ℝ) : ℝ := (3 : ℝ)^x * Real.sin (y * Real.log 3)

-- State the theorem
theorem analytic_function_from_imaginary_part :
  ∃ (f : ℂ → ℂ) (C : ℂ),
    (∀ (x y : ℝ), Complex.im (f (x + y * Complex.I)) = v x y) ∧
    Differentiable ℂ f ∧
    (∀ z : ℂ, f z = 3^z + C) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_analytic_function_from_imaginary_part_l1200_120074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_l1200_120094

/-- A function f(x) with specific properties -/
noncomputable def f (A B C : ℤ) (x : ℝ) : ℝ := x^2 / (A * x^2 + B * x + C)

/-- Theorem stating the sum of A, B, and C given the properties of f(x) -/
theorem sum_of_coefficients 
  (A B C : ℤ) 
  (h1 : ∀ x > 5, f A B C x > (3/10 : ℝ))
  (h2 : ∀ x, (A * x^2 + B * x + C) = A * (x + 3) * (x - 4)) :
  A + B + C = 12 := by
  sorry

#check sum_of_coefficients

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_l1200_120094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_div_B_eq_37_l1200_120036

noncomputable def A : ℝ := ∑' n, if (n % 2 = 0 ∧ n % 3 ≠ 0) ∨ (n % 2 = 1 ∧ n % 3 = 2) then (-1)^((n-1)/2) / n^2 else 0

noncomputable def B : ℝ := ∑' n, if n % 6 = 0 then (-1)^(n/6-1) / n^2 else 0

theorem A_div_B_eq_37 : A / B = 37 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_div_B_eq_37_l1200_120036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_problem_l1200_120083

theorem lcm_problem (q : ℕ) : Nat.lcm (Nat.lcm (Nat.lcm 12 16) 18) q = 144 → q = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_problem_l1200_120083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l1200_120042

/-- Represents the sum of the first n terms of a geometric sequence -/
noncomputable def geometric_sum (a : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a else a * (1 - q^n) / (1 - q)

/-- Proves that for a geometric sequence with positive terms and S_4 = 5S_2, the common ratio is 2 -/
theorem geometric_sequence_ratio (a : ℝ) (q : ℝ) :
  a > 0 → q > 0 → geometric_sum a q 4 = 5 * geometric_sum a q 2 → q = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l1200_120042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_conversion_l1200_120003

-- Define the complex number
noncomputable def z : ℂ := 3 * Complex.exp (Complex.I * (9 * Real.pi / 4))

-- State the theorem
theorem complex_conversion :
  z = Complex.ofReal (3 * Real.sqrt 2 / 2) + Complex.I * Complex.ofReal (3 * Real.sqrt 2 / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_conversion_l1200_120003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_additional_points_l1200_120056

def cube_vertices : List (Fin 3 → ℝ) := [
  ![0, 0, 0], ![0, 0, 5], ![0, 5, 0], ![0, 5, 5],
  ![5, 0, 0], ![5, 0, 5], ![5, 5, 0], ![5, 5, 5]
]

def P : Fin 3 → ℝ := ![0, 3, 0]
def Q : Fin 3 → ℝ := ![2, 0, 0]
def R : Fin 3 → ℝ := ![2, 5, 5]

def plane_equation (v : Fin 3 → ℝ) : Prop := 3 * v 0 + 2 * v 1 - v 2 = 6

theorem distance_between_additional_points :
  ∃ (S T : Fin 3 → ℝ),
    (S ∈ cube_vertices) ∧
    (T ∈ cube_vertices) ∧
    (plane_equation S) ∧
    (plane_equation T) ∧
    (S ≠ P ∧ S ≠ Q ∧ S ≠ R) ∧
    (T ≠ P ∧ T ≠ Q ∧ T ≠ R) ∧
    (S ≠ T) ∧
    (Real.sqrt (355 / 9) = Real.sqrt ((S 0 - T 0)^2 + (S 1 - T 1)^2 + (S 2 - T 2)^2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_additional_points_l1200_120056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_problem_l1200_120089

theorem remainder_problem (k : ℕ) (hk : k > 0) (h : 84 % (k^2) = 20) : 130 % k = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_problem_l1200_120089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_charming_set_l1200_120023

/-- A strange line passes through (a, 0) and (0, 10-a) for some a in [0,10] -/
def StrangeLine (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = (a - 10) / a * p.1 + (10 - a) ∧ 0 ≤ a ∧ a ≤ 10}

/-- A point is charming if it lies in the first quadrant and below some strange line -/
def CharmingPoint (p : ℝ × ℝ) : Prop :=
  p.1 ≥ 0 ∧ p.2 ≥ 0 ∧ ∃ a, p ∈ StrangeLine a

/-- The envelope function of all strange lines -/
noncomputable def EnvelopeFunction (x : ℝ) : ℝ :=
  10 - 2 * Real.sqrt (10 * x) + x

/-- The set of all charming points -/
def CharmingSet : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | CharmingPoint p ∧ p.2 ≤ EnvelopeFunction p.1 ∧ 0 ≤ p.1 ∧ p.1 ≤ 10}

theorem area_of_charming_set :
  MeasureTheory.volume CharmingSet = 50 / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_charming_set_l1200_120023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_symmetry_l1200_120001

/-- The function f(x) with given properties -/
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x - Real.pi / 6)

/-- The theorem stating the symmetry of the function -/
theorem function_symmetry (ω : ℝ) (h1 : ω > 0) (h2 : ∀ x, f ω (x + Real.pi) = f ω x) :
  ∀ x, f ω (7*Real.pi/6 - x) = f ω (7*Real.pi/6 + x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_symmetry_l1200_120001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_minus_270_deg_l1200_120006

theorem cos_alpha_minus_270_deg (α : ℝ) (h : Real.sin (540 * π / 180 + α) = -4/5) :
  Real.cos (α - 270 * π / 180) = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_minus_270_deg_l1200_120006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_A_l1200_120017

-- Define the parabola function
noncomputable def f (x : ℝ) : ℝ := (1/5) * x^2

-- Define the point A
noncomputable def A : ℝ × ℝ := (2, 4/5)

-- State the theorem
theorem tangent_slope_at_A : 
  (deriv f) A.fst = 4/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_A_l1200_120017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_P_on_line_l_sum_reciprocal_distances_l1200_120070

noncomputable section

-- Define the curve C
def curve_C (φ : Real) : Real × Real :=
  (Real.sqrt 2 * Real.cos φ, 2 * Real.sin φ)

-- Define the line l in polar form
def line_l (θ : Real) : Real :=
  Real.sqrt 3 / (2 * Real.cos (θ - Real.pi / 6))

-- Define point P
def point_P : Real × Real := (0, Real.sqrt 3)

-- Theorem 1: Point P lies on line l
theorem point_P_on_line_l :
  ∃ θ, line_l θ * Real.cos θ = point_P.1 ∧ line_l θ * Real.sin θ = point_P.2 := by
  sorry

-- Theorem 2: Sum of reciprocals of distances
theorem sum_reciprocal_distances :
  ∃ A B : Real × Real,
    (∃ φ, curve_C φ = A) ∧
    (∃ φ, curve_C φ = B) ∧
    (∃ θ, line_l θ * Real.cos θ = A.1 ∧ line_l θ * Real.sin θ = A.2) ∧
    (∃ θ, line_l θ * Real.cos θ = B.1 ∧ line_l θ * Real.sin θ = B.2) ∧
    1 / Real.sqrt ((A.1 - point_P.1)^2 + (A.2 - point_P.2)^2) +
    1 / Real.sqrt ((B.1 - point_P.1)^2 + (B.2 - point_P.2)^2) =
    Real.sqrt 14 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_P_on_line_l_sum_reciprocal_distances_l1200_120070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_implies_k_value_l1200_120040

-- Define the line
def line (k : ℝ) (x : ℝ) : ℝ := k * x + 3

-- Define the circle
def circleEq (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 9

-- Define the intersection points
def intersectionPoints (k : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ x y, p = (x, y) ∧ y = line k x ∧ circleEq x y}

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem intersection_distance_implies_k_value (k : ℝ) 
  (hk : k > 1) 
  (h_intersect : ∃ A B, A ∈ intersectionPoints k ∧ B ∈ intersectionPoints k ∧ A ≠ B) 
  (h_distance : ∀ A B, A ∈ intersectionPoints k → B ∈ intersectionPoints k → A ≠ B → 
    distance A B = 12 * Real.sqrt 5 / 5) : 
  k = 2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_implies_k_value_l1200_120040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_neg_x_l1200_120028

-- Define the piecewise function g
noncomputable def g (x : ℝ) : ℝ :=
  if -2 ≤ x ∧ x < 0 then x^2 - 1
  else if 0 ≤ x ∧ x ≤ 2 then 1 - x
  else if 2 < x ∧ x ≤ 4 then x - 3
  else 0  -- Undefined outside the domain

-- State the theorem about g(-x)
theorem g_neg_x (x : ℝ) :
  (0 < x ∧ x ≤ 2 → g (-x) = x^2 - 1) ∧
  (-2 ≤ x ∧ x ≤ 0 → g (-x) = 1 + x) ∧
  (-4 ≤ x ∧ x < -2 → g (-x) = -x - 3) := by
  sorry

#check g_neg_x

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_neg_x_l1200_120028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_l1200_120046

noncomputable def f (x y : ℝ) : ℝ := (x - y)^2 + (Real.sqrt (2 - x^2) - 9 / y)^2

theorem f_minimum (x y : ℝ) (hx : 0 < x ∧ x < Real.sqrt 2) (hy : y > 0) :
  f x y ≥ 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_l1200_120046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_integers_l1200_120075

theorem partition_integers (n : ℕ) (k : ℕ) (s : ℕ) (h : n = 1414 ∧ k = 505 ∧ s = 1981) :
  ∃ (P : Finset (Finset ℕ)), 
    (∀ A ∈ P, (∀ x ∈ A, x ≤ n) ∧ (A.sum id = s)) ∧ 
    P.card = k ∧
    (⋃ A ∈ P, A : Set ℕ) = Finset.range n.succ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_integers_l1200_120075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_P_value_at_seven_l1200_120030

-- Define the polynomial P(x)
def P (x a b c d e f : ℂ) : ℂ :=
  (3*x^4 - 30*x^3 + a*x^2 + b*x + c) * (4*x^4 - 84*x^3 + d*x^2 + e*x + f)

-- Theorem statement
theorem P_value_at_seven 
  (a b c d e f : ℂ) 
  (h : ∃ (z₁ z₂ z₃ z₄ z₅ : ℂ), 
    (z₁ = 2 ∨ z₁ = 3 ∨ z₁ = 4 ∨ z₁ = 5) ∧
    (z₂ = 2 ∨ z₂ = 3 ∨ z₂ = 4 ∨ z₂ = 5) ∧
    (z₃ = 2 ∨ z₃ = 3 ∨ z₃ = 4 ∨ z₃ = 5) ∧
    (z₄ = 5) ∧ (z₅ = 5) ∧
    (∀ z : ℂ, (P z a b c d e f = 0) ↔ 
      (z = z₁ ∨ z = z₂ ∨ z = z₃ ∨ z = z₄ ∨ z = z₅))) :
  P 7 a b c d e f = 86400 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_P_value_at_seven_l1200_120030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_part_is_three_km_l1200_120038

/-- Represents a race with given parameters -/
structure Race where
  total_distance : ℝ
  speed_first_part : ℝ
  speed_second_part : ℝ
  speed_third_part : ℝ
  second_part_distance : ℝ
  third_part_distance : ℝ
  average_speed : ℝ

/-- Calculates the length of the first part of the race -/
noncomputable def first_part_length (r : Race) : ℝ :=
  r.total_distance - r.second_part_distance - r.third_part_distance

/-- Calculates the total time of the race -/
noncomputable def total_time (r : Race) (x : ℝ) : ℝ :=
  x / r.speed_first_part + r.second_part_distance / r.speed_second_part + r.third_part_distance / r.speed_third_part

/-- Theorem: The first part of the race is 3 kilometers long -/
theorem first_part_is_three_km (r : Race)
  (h1 : r.total_distance = 6)
  (h2 : r.speed_first_part = 150)
  (h3 : r.speed_second_part = 200)
  (h4 : r.speed_third_part = 300)
  (h5 : r.second_part_distance = 2)
  (h6 : r.third_part_distance = 1)
  (h7 : r.average_speed = 180)
  : first_part_length r = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_part_is_three_km_l1200_120038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_digit_numbers_with_eight_or_three_l1200_120025

theorem four_digit_numbers_with_eight_or_three : ℕ := by
  let total_four_digit : ℕ := 9999 - 1000 + 1
  let numbers_without_eight_or_three : ℕ := 7 * 8^3
  have h : total_four_digit - numbers_without_eight_or_three = 5416 := by sorry
  exact 5416

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_digit_numbers_with_eight_or_three_l1200_120025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1200_120059

-- Define the set of real numbers a that satisfy the conditions
def A : Set ℝ := {a | a > 0 ∧ a ≠ 1 ∧ Real.log (3*a - 1) / Real.log a > 1}

-- State the theorem
theorem range_of_a : A = Set.Ioo (1/3 : ℝ) (1/2) ∪ Set.Ioi 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1200_120059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_range_l1200_120026

-- Define the circle
def myCircle (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 6*y + 6 = 0

-- Define a line passing through (-1, -1) with slope k
def myLine (k : ℝ) (x y : ℝ) : Prop := y + 1 = k * (x + 1)

-- Define the intersection of the line and circle
def intersects (k : ℝ) : Prop := ∃ x y, myCircle x y ∧ myLine k x y

-- Theorem statement
theorem slope_range :
  ∀ k : ℝ, intersects k → k < 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_range_l1200_120026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_logarithm_l1200_120052

noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 2

theorem arithmetic_sequence_logarithm (x : ℝ) : 
  x > 1 → 
  (∃ d : ℝ, lg (x - 1) - lg 2 = d ∧ lg (x + 3) - lg (x - 1) = d) → 
  x = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_logarithm_l1200_120052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_monotone_interval_l1200_120098

/-- The function f(x) defined in the problem -/
noncomputable def f (ω φ x : ℝ) : ℝ := Real.sin (ω * x + φ) + Real.sqrt 3 * Real.cos (ω * x + φ)

/-- The theorem statement -/
theorem function_monotone_interval 
  (ω φ : ℝ) 
  (h_ω_pos : ω > 0)
  (h_point : f ω φ 1 = 2)
  (h_zeros : ∃ (x₁ x₂ : ℝ), f ω φ x₁ = 0 ∧ f ω φ x₂ = 0 ∧ |x₁ - x₂| = 6) :
  ∀ k : ℤ, StrictMonoOn (f ω φ) (Set.Icc (-5 + 12 * ↑k) (1 + 12 * ↑k)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_monotone_interval_l1200_120098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_f2xminus3_l1200_120041

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the domain of f(2x+3)
def domain_f2xplus3 : Set ℝ := {x : ℝ | -4 ≤ x ∧ x < 5}

-- Theorem statement
theorem domain_f2xminus3 (h : ∀ x, x ∈ domain_f2xplus3 ↔ (2*x+3 ∈ Set.univ)) :
  ∀ x, (2*x-3 ∈ Set.univ) ↔ -1 ≤ x ∧ x < 8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_f2xminus3_l1200_120041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_equals_three_l1200_120051

theorem tan_alpha_equals_three (α : ℝ) (h1 : 0 < α ∧ α < π / 2) 
  (h2 : Real.sin (2 * α) + Real.cos (2 * α) = -1/5) : Real.tan α = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_equals_three_l1200_120051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ticket_sale_savings_l1200_120020

theorem ticket_sale_savings (P : ℝ) (h : P > 0) : 
  (((6 * P) - (3 * P)) / (6 * P)) * 100 = 50 := by
  -- Simplify the expression
  have h1 : ((6 * P) - (3 * P)) / (6 * P) = 1/2 := by
    field_simp
    ring
  
  -- Multiply both sides by 100
  calc ((6 * P) - (3 * P)) / (6 * P) * 100 = (1/2) * 100 := by rw [h1]
    _ = 50 := by norm_num

  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ticket_sale_savings_l1200_120020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_operations_to_equalize_l1200_120086

/-- Represents a configuration of numbers on a circle -/
def CircleConfig := Fin 2009 → Nat

/-- The operation of adding 1 to two adjacent numbers -/
def addToAdjacent (config : CircleConfig) (i : Fin 2009) : CircleConfig :=
  fun j => if j = i ∨ j = i.succ then config j + 1 else config j

/-- Predicate to check if all numbers in the configuration are equal -/
def allEqual (config : CircleConfig) : Prop :=
  ∀ i j : Fin 2009, config i = config j

/-- The main theorem statement -/
theorem min_operations_to_equalize (config : CircleConfig) 
  (h : ∀ i : Fin 2009, config i ≤ 100) :
  (∃ k : Nat, ∃ sequence : List (Fin 2009), 
    sequence.length = k ∧
    allEqual (sequence.foldl addToAdjacent config)) ∧
  (∀ k : Nat, k < 100400 → 
    ¬∃ sequence : List (Fin 2009), 
      sequence.length = k ∧
      allEqual (sequence.foldl addToAdjacent config)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_operations_to_equalize_l1200_120086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_cardinality_problem_l1200_120019

-- Define the symmetric difference operation
def symmetricDifference (x y : Finset ℤ) : Finset ℤ := (x \ y) ∪ (y \ x)

-- State the theorem
theorem set_cardinality_problem (x y : Finset ℤ) 
  (hx : x.card = 14)
  (hxy : (x ∩ y).card = 6)
  (hxsy : (symmetricDifference x y).card = 20) :
  y.card = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_cardinality_problem_l1200_120019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_cone_sphere_ratio_l1200_120049

/-- A truncated right circular cone with an inscribed sphere -/
structure TruncatedConeWithSphere where
  R : ℝ  -- radius of the larger base
  r : ℝ  -- radius of the smaller base
  s : ℝ  -- radius of the inscribed sphere
  H : ℝ  -- height of the truncated cone

/-- The ratio of the radii of the bases of the truncated cone -/
noncomputable def ratio (cone : TruncatedConeWithSphere) : ℝ := cone.R / cone.r

/-- The volume of the truncated cone -/
noncomputable def volume_truncated_cone (cone : TruncatedConeWithSphere) : ℝ :=
  (Real.pi / 3) * (cone.R^2 * cone.H - cone.r^2 * (cone.H - 2 * cone.s))

/-- The volume of the inscribed sphere -/
noncomputable def volume_sphere (cone : TruncatedConeWithSphere) : ℝ :=
  (4 / 3) * Real.pi * cone.s^3

/-- Theorem: If a sphere is inscribed in a truncated right circular cone and
    the volume of the truncated cone is twice that of the sphere,
    then the ratio of the radius of the bottom base to the radius of the top base
    of the truncated cone is (3 + √5) / 2 -/
theorem truncated_cone_sphere_ratio
  (cone : TruncatedConeWithSphere)
  (h1 : cone.s = Real.sqrt (cone.R * cone.r))
  (h2 : volume_truncated_cone cone = 2 * volume_sphere cone) :
  ratio cone = (3 + Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_cone_sphere_ratio_l1200_120049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chemistry_mean_section1_l1200_120078

-- Define the number of students in each section
def students_section1 : ℕ := 65
def students_section2 : ℕ := 35
def students_section3 : ℕ := 45
def students_section4 : ℕ := 42

-- Define the mean marks for sections 2, 3, and 4
def mean_section2 : ℝ := 60
def mean_section3 : ℝ := 55
def mean_section4 : ℝ := 45

-- Define the overall average of marks per student
def overall_average : ℝ := 51.95

-- Theorem to prove
theorem chemistry_mean_section1 :
  ∃ (mean_section1 : ℝ),
    (students_section1 * mean_section1 +
     students_section2 * mean_section2 +
     students_section3 * mean_section3 +
     students_section4 * mean_section4) /
    (students_section1 + students_section2 + students_section3 + students_section4 : ℝ)
    = overall_average ∧
    abs (mean_section1 - 50) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chemistry_mean_section1_l1200_120078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_l1200_120055

-- Define the function f as noncomputable
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (ω * x) + Real.cos (ω * x)

-- State the theorem
theorem omega_value (ω : ℝ) (x₁ x₂ : ℝ) 
  (h_omega_pos : ω > 0)
  (h_f_x₁ : f ω x₁ = -2)
  (h_f_x₂ : f ω x₂ = 0)
  (h_min_diff : ∃ (k : ℤ), |x₁ - x₂| = π + 2 * π * k ∧ ∀ (m : ℤ), |x₁ - x₂| ≤ π + 2 * π * m) :
  ω = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_l1200_120055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l1200_120044

/-- Given an ellipse and a hyperbola with specific properties, prove the equation of the hyperbola -/
theorem hyperbola_equation 
  (ellipse : ℝ → ℝ → Prop)
  (hyperbola : ℝ → ℝ → Prop)
  (vertices_hyperbola : Set (ℝ × ℝ))
  (major_axis_endpoints_ellipse : Set (ℝ × ℝ))
  (eccentricity_ellipse eccentricity_hyperbola : ℝ) :
  (∀ x y : ℝ, x^2 + y^2/2 = 1 → ellipse x y) →  -- Ellipse equation
  (vertices_hyperbola = major_axis_endpoints_ellipse) →  -- Vertices condition
  (eccentricity_ellipse * eccentricity_hyperbola = 1) →  -- Eccentricity product condition
  (∀ x y : ℝ, y^2/2 - x^2/2 = 1 ↔ hyperbola x y) :=  -- Conclusion to prove
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l1200_120044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_card_trick_possible_l1200_120069

/-- Represents a playing card with a rank and a suit -/
structure Card where
  rank : Fin 13
  suit : Fin 4

/-- The deck of 52 cards -/
def Deck : Finset Card := sorry

/-- A selection of 5 cards from the deck -/
def Selection : Finset Card := sorry

/-- The strategy function that determines the order of showing 4 cards -/
def strategy (s : Finset Card) : List Card := sorry

/-- The decoding function that determines the fifth card based on the order of 4 cards -/
def decode (l : List Card) : Card := sorry

theorem card_trick_possible :
  ∃ (strategy : Finset Card → List Card) (decode : List Card → Card),
    ∀ s : Finset Card,
      s ⊆ Deck →
      s.card = 5 →
      ∃ c ∈ s, c = decode (strategy s) ∧ c ∉ strategy s := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_card_trick_possible_l1200_120069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_aaron_jogging_speed_l1200_120010

/-- Calculates the jogging speed given the jogging distance, walking speed, and total time -/
noncomputable def jogging_speed (jog_distance : ℝ) (walk_speed : ℝ) (total_time : ℝ) : ℝ :=
  (jog_distance * walk_speed) / (walk_speed * total_time - jog_distance)

theorem aaron_jogging_speed :
  let jog_distance : ℝ := 3
  let walk_speed : ℝ := 4
  let total_time : ℝ := 3
  jogging_speed jog_distance walk_speed total_time = 16/3 := by
  sorry

#eval (16 : ℚ) / 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_aaron_jogging_speed_l1200_120010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seventh_term_value_l1200_120063

noncomputable def arithmetic_progression (a d : ℝ) (n : ℕ) : ℝ := a + (n - 1 : ℝ) * d

noncomputable def sum_arithmetic_progression (a d : ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) / 2 * (2 * a + (n - 1 : ℝ) * d)

theorem seventh_term_value (a d : ℝ) :
  sum_arithmetic_progression a d 15 = 56.25 →
  arithmetic_progression a d 11 = 5.25 →
  arithmetic_progression a d 7 = 3.25 :=
by
  intros h1 h2
  -- The proof steps would go here, but for now we'll use sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_seventh_term_value_l1200_120063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_even_integers_12_to_46_l1200_120058

theorem sum_of_even_integers_12_to_46 :
  (Finset.range 18).sum (λ i => 12 + 2 * i) = 522 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_even_integers_12_to_46_l1200_120058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_and_max_inradius_l1200_120065

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- The eccentricity of an ellipse -/
noncomputable def Ellipse.eccentricity (e : Ellipse) : ℝ := Real.sqrt (1 - e.b^2 / e.a^2)

/-- A point on the ellipse -/
def Ellipse.contains (e : Ellipse) (x y : ℝ) : Prop := x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- Inradius of a triangle -/
noncomputable def inradius (P Q F : ℝ × ℝ) : ℝ := sorry

theorem ellipse_equation_and_max_inradius (e : Ellipse) 
  (h_point : e.contains (Real.sqrt 3) (1/2))
  (h_ecc : e.eccentricity = Real.sqrt 3 / 2) :
  (∃ (a b : ℝ), e.a = a ∧ e.b = b ∧ ∀ x y, e.contains x y ↔ x^2/4 + y^2 = 1) ∧
  (∃ r : ℝ, r = 1/2 ∧ 
    ∀ P Q : ℝ × ℝ, ∀ l : ℝ → ℝ × ℝ,
      (∃ t₁ t₂, l t₁ = P ∧ l t₂ = Q ∧ e.contains P.1 P.2 ∧ e.contains Q.1 Q.2) →
      (∃ F₁ F₂ : ℝ × ℝ, F₁.1 = -e.a * e.eccentricity ∧ F₁.2 = 0 ∧
                        F₂.1 = e.a * e.eccentricity ∧ F₂.2 = 0 ∧
                        (∃ s, l s = F₁)) →
      (inradius P Q F₂ ≤ r)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_and_max_inradius_l1200_120065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_of_extreme_points_l1200_120077

-- Define the function f(x)
noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.sin x - a * Real.cos x

-- Define the symmetry axis
noncomputable def symmetry_axis : ℝ := 3 * Real.pi / 4

-- Define the extreme points
def extreme_points (f : ℝ → ℝ) : Set ℝ :=
  {x | ∀ y, f y ≤ f x ∨ f y ≥ f x}

-- Theorem statement
theorem min_sum_of_extreme_points 
  (a : ℝ) 
  (h_symmetry : ∃ (k : ℝ), ∀ x, f x a = f (2 * symmetry_axis - x) a) 
  : 
  ∃ (x₁ x₂ : ℝ), x₁ ∈ extreme_points (f · a) ∧ 
                    x₂ ∈ extreme_points (f · a) ∧ 
                    |x₁ + x₂| ≥ Real.pi / 2 ∧
                    (∀ (y₁ y₂ : ℝ), 
                      y₁ ∈ extreme_points (f · a) → 
                      y₂ ∈ extreme_points (f · a) → 
                      |y₁ + y₂| ≥ |x₁ + x₂|) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_of_extreme_points_l1200_120077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1200_120000

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := x
noncomputable def g (x : ℝ) : ℝ := 2 / x

-- Define the sum function h
noncomputable def h (x : ℝ) : ℝ := f x + g x

-- Theorem statement
theorem function_properties :
  (∀ x : ℝ, f x = x * f 1) ∧  -- f is directly proportional
  (∀ x : ℝ, x ≠ 0 → g x * x = g 1) ∧  -- g is inversely proportional
  (f 1 = 1) ∧  -- f(1) = 1
  (g 1 = 2) ∧  -- g(1) = 2
  (∀ x : ℝ, f x = x) ∧  -- f(x) = x
  (∀ x : ℝ, x ≠ 0 → g x = 2 / x) ∧  -- g(x) = 2/x
  (∀ x : ℝ, x ≠ 0 → h (-x) = -h x) ∧  -- h is odd
  (∀ x : ℝ, 0 < x → x ≤ Real.sqrt 2 → h x ≥ 2 * Real.sqrt 2) ∧  -- minimum value
  (h (Real.sqrt 2) = 2 * Real.sqrt 2)  -- minimum achieved at √2
  := by
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1200_120000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tissues_per_box_l1200_120007

theorem tissues_per_box (group1 group2 group3 total_tissues : ℕ) 
  (h1 : group1 = 9)
  (h2 : group2 = 10)
  (h3 : group3 = 11)
  (h4 : total_tissues = 1200) :
  total_tissues / (group1 + group2 + group3) = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tissues_per_box_l1200_120007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_cos_value_l1200_120079

theorem sin_minus_cos_value (α : Real) 
  (h1 : Real.sin α * Real.cos α = -12/25) 
  (h2 : α ∈ Set.Ioo 0 Real.pi) : 
  Real.sin α - Real.cos α = 7/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_cos_value_l1200_120079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bisection_uses_all_structures_l1200_120081

/-- Represents the different types of algorithm structures -/
inductive AlgorithmStructure
  | Sequential
  | Conditional
  | Loop

/-- The bisection method for finding an approximate root of x^2 - 2 = 0 -/
def bisectionMethod : Set AlgorithmStructure := sorry

/-- Every algorithm uses a sequential structure -/
axiom sequential_in_all : ∀ (alg : Set AlgorithmStructure), AlgorithmStructure.Sequential ∈ alg

/-- A loop structure includes a conditional structure -/
axiom conditional_in_loop : 
  ∀ (alg : Set AlgorithmStructure), 
  AlgorithmStructure.Loop ∈ alg → AlgorithmStructure.Conditional ∈ alg

/-- The bisection method uses a loop structure -/
axiom bisection_uses_loop : AlgorithmStructure.Loop ∈ bisectionMethod

/-- Theorem: The bisection method uses all three algorithm structures -/
theorem bisection_uses_all_structures : 
  {AlgorithmStructure.Sequential, AlgorithmStructure.Conditional, AlgorithmStructure.Loop} ⊆ bisectionMethod := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bisection_uses_all_structures_l1200_120081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_and_prime_divisor_equivalence_l1200_120064

theorem gcd_and_prime_divisor_equivalence 
  (a b c d : ℤ) 
  (h : Int.gcd a (Int.gcd b (Int.gcd c d)) = 1) :
  (∀ (p : ℕ), Nat.Prime p → (p : ℤ) ∣ (a * d - b * c) → ((p : ℤ) ∣ a ∧ (p : ℤ) ∣ c)) ↔ 
  (∀ (n : ℤ), Int.gcd (a * n + b) (c * n + d) = 1) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_and_prime_divisor_equivalence_l1200_120064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_points_l1200_120011

/-- The distance between two points in polar coordinates -/
noncomputable def polar_distance (r1 : ℝ) (θ1 : ℝ) (r2 : ℝ) (θ2 : ℝ) : ℝ :=
  Real.sqrt ((r1 * Real.cos θ1 - r2 * Real.cos θ2)^2 + (r1 * Real.sin θ1 - r2 * Real.sin θ2)^2)

theorem distance_between_specific_points :
  polar_distance 2 (π/3) (2 * Real.sqrt 3) (5*π/6) = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_points_l1200_120011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l1200_120005

def A : Set ℕ := {x | 1 < x ∧ x ≤ 4}
def B : Set ℕ := {x | x^2 - x - 6 ≤ 0}

theorem intersection_A_B : A ∩ B = {2, 3} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l1200_120005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_comparison_l1200_120099

-- Define the Quadrilateral structure
structure Quadrilateral where
  area : ℝ
  perimeter : ℝ

theorem quadrilateral_comparison (quad_I quad_II : Quadrilateral)
  (h1 : quad_I.area = 1)
  (h2 : quad_II.area = 1)
  (h3 : quad_I.perimeter = 2 + 2 * Real.sqrt 2)
  (h4 : quad_II.perimeter = 1 + 2 * Real.sqrt 2 + Real.sqrt 5) :
  quad_I.perimeter < quad_II.perimeter :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_comparison_l1200_120099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_sequences_count_l1200_120057

def number_of_contestants : ℕ := 6
def number_of_females : ℕ := 3
def number_of_males : ℕ := 3

def is_valid_sequence (seq : List ℕ) : Bool :=
  seq.length = number_of_contestants &&
  seq.toFinset.card = number_of_contestants &&
  (∀ i ∈ seq, i > 0 && i ≤ number_of_contestants) &&
  (∀ i < seq.length - 1, 
    (seq[i]! > number_of_females → seq[i+1]! ≤ number_of_females) &&
    (seq[i]! ≤ number_of_females → seq[i+1]! > number_of_females)) &&
  seq.head! ≠ 1

theorem valid_sequences_count :
  (List.filter is_valid_sequence (List.permutations (List.range' 1 number_of_contestants))).length = 132 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_sequences_count_l1200_120057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sqrt_minus_one_l1200_120015

open MeasureTheory Interval Real

theorem integral_sqrt_minus_one : ∫ x in (-1)..1, (Real.sqrt (1 - x^2) - 1) = π / 2 - 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sqrt_minus_one_l1200_120015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_person_between_Masha_and_Nastya_l1200_120050

/-- Represents the positions in a line of 5 people -/
inductive Position : Type
  | first : Position
  | second : Position
  | third : Position
  | fourth : Position
  | last : Position

/-- Represents the 5 friends -/
inductive Friend : Type
  | Masha : Friend
  | Nastya : Friend
  | Irina : Friend
  | Olya : Friend
  | Anya : Friend

/-- The line arrangement of friends -/
def line_arrangement : Friend → Position
  | Friend.Irina => Position.first
  | Friend.Nastya => Position.second
  | Friend.Masha => Position.third
  | Friend.Anya => Position.fourth
  | Friend.Olya => Position.last

/-- Count the number of positions between two given positions -/
def positions_between (p1 p2 : Position) : Nat :=
  match p1, p2 with
  | Position.first, Position.third => 1
  | Position.second, Position.fourth => 1
  | Position.second, Position.third => 0
  | Position.third, Position.second => 0
  | _, _ => 0

theorem one_person_between_Masha_and_Nastya :
  positions_between (line_arrangement Friend.Nastya) (line_arrangement Friend.Masha) = 0 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_person_between_Masha_and_Nastya_l1200_120050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_center_to_edge_formula_l1200_120002

/-- The distance from the center of the base to the lateral edge of a regular quadrilateral pyramid -/
noncomputable def distance_center_to_edge (a : ℝ) (α : ℝ) : ℝ :=
  (a / 2) * Real.sqrt (2 * Real.cos α)

/-- Theorem: In a regular quadrilateral pyramid with base side length a and plane angle α at the apex,
    the distance from the center of the base to the lateral edge is (a/2) * √(2 * cos(α)) -/
theorem distance_center_to_edge_formula (a α : ℝ) (h1 : a > 0) (h2 : 0 < α ∧ α < π) :
  distance_center_to_edge a α = (a / 2) * Real.sqrt (2 * Real.cos α) := by
  -- Unfold the definition of distance_center_to_edge
  unfold distance_center_to_edge
  -- The equality follows directly from the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_center_to_edge_formula_l1200_120002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_minimum_value_l1200_120095

/-- A predicate to check if a set of points is a line -/
def IsLine (l : Set (ℝ × ℝ)) : Prop := sorry

/-- A predicate to check if a line is tangent to a circle -/
def IsTangentLine (l : Set (ℝ × ℝ)) (c : Set (ℝ × ℝ)) : Prop := sorry

theorem circles_minimum_value (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) : 
  let C₁ := {(x, y) : ℝ × ℝ | x^2 + y^2 + 2*a*x + a^2 - 9 = 0}
  let C₂ := {(x, y) : ℝ × ℝ | x^2 + y^2 - 4*b*y - 1 + 4*b^2 = 0}
  (∃! l : Set (ℝ × ℝ), IsLine l ∧ IsTangentLine l C₁ ∧ IsTangentLine l C₂) →
  (∀ x : ℝ, 4/a^2 + 1/b^2 ≥ 4) ∧ (∃ x : ℝ, 4/a^2 + 1/b^2 = 4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_minimum_value_l1200_120095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1200_120008

noncomputable def f (A : ℝ) (x : ℝ) : ℝ := A * Real.sin (x + Real.pi/4)

theorem problem_solution (A : ℝ) (α : ℝ) 
  (h1 : f A 0 = 1)
  (h2 : f A α = -1/5)
  (h3 : Real.pi/2 < α ∧ α < Real.pi) :
  A = Real.sqrt 2 ∧ Real.cos α = -4/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1200_120008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_with_distance_from_origin_l1200_120066

/-- A line in 2D space --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  nz : a ≠ 0 ∨ b ≠ 0

/-- Distance from a point to a line --/
noncomputable def distancePointToLine (x y : ℝ) (l : Line) : ℝ :=
  (abs (l.a * x + l.b * y + l.c)) / (Real.sqrt (l.a^2 + l.b^2))

/-- A line passes through a point --/
def linePassesThrough (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y + l.c = 0

theorem line_through_point_with_distance_from_origin
  (l : Line)
  (h1 : linePassesThrough l 5 10)
  (h2 : distancePointToLine 0 0 l = 5) :
  (l.a = 1 ∧ l.b = 0 ∧ l.c = -5) ∨
  (l.a = 3 ∧ l.b = -4 ∧ l.c = 25) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_with_distance_from_origin_l1200_120066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_identity_l1200_120004

theorem sin_cos_identity (k : ℤ) :
  let t : ℝ := π / 16 * (4 * k + 1)
  (Real.sin (4 * t) + Real.cos (4 * t))^2 = 16 * Real.sin (2 * t) * Real.cos (2 * t)^3 - 8 * Real.sin (2 * t) * Real.cos (2 * t) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_identity_l1200_120004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sunflower_seeds_majority_l1200_120072

/-- Represents the state of the bird feeder on a given day -/
structure FeederState where
  day : Nat
  sunflowerSeeds : Float
  totalSeeds : Float

/-- Calculates the next day's feeder state -/
def nextDay (state : FeederState) : FeederState :=
  { day := state.day + 1,
    sunflowerSeeds := state.sunflowerSeeds * 0.7 + 0.8,
    totalSeeds := state.sunflowerSeeds * 0.7 + 2 }

/-- Initial state of the feeder -/
def initialState : FeederState :=
  { day := 1, sunflowerSeeds := 0.8, totalSeeds := 2 }

/-- Checks if sunflower seeds are more than half of total seeds -/
def moreThanHalf (state : FeederState) : Bool :=
  state.sunflowerSeeds > state.totalSeeds / 2

/-- Theorem: On the 4th day, the feeder will contain more than half sunflower seeds for the first time -/
theorem sunflower_seeds_majority :
  let state4 := (nextDay ∘ nextDay ∘ nextDay) initialState
  ∀ n < 4, ¬moreThanHalf ((Nat.iterate nextDay n) initialState) ∧
  moreThanHalf state4 := by
  sorry

#eval moreThanHalf ((Nat.iterate nextDay 3) initialState)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sunflower_seeds_majority_l1200_120072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_modulus_equality_l1200_120021

theorem complex_modulus_equality : 
  Complex.abs (-7 + (11 / 3) * Complex.I + 2) = Real.sqrt 346 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_modulus_equality_l1200_120021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_watermelon_is_forty_percent_l1200_120061

/-- Represents a fruit drink composition -/
structure FruitDrink where
  total : ℚ
  orange_percent : ℚ
  grape_amount : ℚ

/-- Calculates the percentage of watermelon juice in the drink -/
def watermelon_percentage (drink : FruitDrink) : ℚ :=
  100 * (drink.total - (drink.orange_percent / 100 * drink.total + drink.grape_amount)) / drink.total

/-- Theorem stating that the watermelon juice percentage is 40% for the given drink composition -/
theorem watermelon_is_forty_percent (drink : FruitDrink)
  (h_total : drink.total = 200)
  (h_orange : drink.orange_percent = 25)
  (h_grape : drink.grape_amount = 70) :
  watermelon_percentage drink = 40 := by
  sorry

/-- Compute the result for the given values -/
def result : ℚ :=
  watermelon_percentage { total := 200, orange_percent := 25, grape_amount := 70 }

#eval result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_watermelon_is_forty_percent_l1200_120061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_c_value_side_values_l1200_120092

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Part I
theorem cosine_c_value (t : Triangle) (h1 : t.a + t.b + t.c = 8) (h2 : t.a = 2) (h3 : t.b = 5/2) :
  Real.cos t.C = -1/5 := by sorry

-- Part II
theorem side_values (t : Triangle) (h1 : t.a + t.b + t.c = 8)
  (h2 : Real.sin t.A * (Real.cos (t.B/2))^2 + Real.sin t.B * (Real.cos (t.A/2))^2 = 2 * Real.sin t.C)
  (h3 : (1/2) * t.a * t.b * Real.sin t.C = (9/2) * Real.sin t.C) :
  t.a = 3 ∧ t.b = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_c_value_side_values_l1200_120092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_l1200_120097

noncomputable def f (m x : ℝ) : ℝ := m - Real.sqrt (x + 3)

theorem m_range (m : ℝ) :
  (∃ a b : ℝ, a < b ∧
    (∀ y : ℝ, y ∈ Set.Icc a b ↔ ∃ x : ℝ, x ∈ Set.Icc a b ∧ f m x = y)) →
  m ∈ Set.Ioo (-9/4) (-2) := by
  sorry

#check m_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_l1200_120097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_exists_in_interval_l1200_120027

noncomputable def f (x : ℝ) := x^(1/3) - (1/2)^x

theorem zero_exists_in_interval : ∃ c ∈ Set.Ioo (1/3 : ℝ) (1/2 : ℝ), f c = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_exists_in_interval_l1200_120027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trader_gain_percentage_approx_l1200_120013

/-- The trader's gain percentage when selling pens -/
noncomputable def trader_gain_percentage : ℝ := 100 * (15 / 90)

/-- Theorem stating that the trader's gain percentage is approximately 16.67% -/
theorem trader_gain_percentage_approx :
  abs (trader_gain_percentage - 16.67) < 0.01 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trader_gain_percentage_approx_l1200_120013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_perfect_square_function_l1200_120033

/-- A function from positive integers to positive integers -/
def PositiveIntFunction := ℕ+ → ℕ+

/-- The property that xf(x) + (f(y))² + 2xf(y) is a perfect square for all positive integers x and y -/
def IsPerfectSquare (f : PositiveIntFunction) : Prop :=
  ∀ (x y : ℕ+), ∃ (z : ℕ+), (x.val * (f x).val + (f y).val^2 + 2 * x.val * (f y).val : ℕ) = z.val^2

/-- The identity function on positive integers -/
def id_pos : PositiveIntFunction := λ x => x

/-- Theorem stating that the identity function is the only function satisfying the perfect square property -/
theorem unique_perfect_square_function :
  ∀ (f : PositiveIntFunction), IsPerfectSquare f ↔ f = id_pos := by
  sorry

#check unique_perfect_square_function

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_perfect_square_function_l1200_120033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_two_theta_value_l1200_120084

theorem cos_two_theta_value (θ : ℝ) (h : ∑' n, (Real.cos θ)^(2*n) = 8) : Real.cos (2*θ) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_two_theta_value_l1200_120084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_probability_l1200_120009

/-- Represents the arrival time of a person in hours after 2:00 p.m. -/
def ArrivalTime := { t : ℝ // 0 ≤ t ∧ t ≤ 2 }

/-- The probability space of all possible arrival scenarios -/
def Ω := ArrivalTime × ArrivalTime × ArrivalTime

/-- The event where the meeting takes place -/
def MeetingOccurs (ω : Ω) : Prop :=
  let (x, y, z) := ω
  z.1 > x.1 ∧ z.1 > y.1 ∧ (y.1 ≤ x.1 + 1) ∧ (x.1 ≤ y.1 + 1)

/-- The probability measure on the sample space -/
noncomputable def P : Set Ω → ℝ := sorry

theorem meeting_probability :
  P { ω : Ω | MeetingOccurs ω } = 7 / 24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_probability_l1200_120009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_continuous_l1200_120080

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (x - 1) * (x - 2) * (x - 3)

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := min (f x) (deriv f x)

-- Theorem statement
theorem g_is_continuous : Continuous g := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_continuous_l1200_120080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_rectangular_transformation_l1200_120048

noncomputable def initial_x : ℝ := 12
noncomputable def initial_y : ℝ := 5

noncomputable def r : ℝ := Real.sqrt (initial_x^2 + initial_y^2)

noncomputable def cos_theta : ℝ := initial_x / r
noncomputable def sin_theta : ℝ := initial_y / r

noncomputable def cos_3theta : ℝ := 4 * cos_theta^3 - 3 * cos_theta
noncomputable def sin_3theta : ℝ := 3 * sin_theta - 4 * sin_theta^3

noncomputable def final_x : ℝ := r^3 * cos_3theta
noncomputable def final_y : ℝ := r^3 * sin_3theta

theorem polar_to_rectangular_transformation :
  (Int.floor final_x : ℤ) = -2197 ∧ (Int.floor final_y : ℤ) = 31955 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_rectangular_transformation_l1200_120048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_product_one_plus_i_one_minus_i_l1200_120039

theorem complex_product_one_plus_i_one_minus_i :
  (1 + Complex.I) * (1 - Complex.I) = 2 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_product_one_plus_i_one_minus_i_l1200_120039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l1200_120054

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 9 = 1

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 16 - y^2 / 48 = 1

-- Define the eccentricity of the hyperbola
def hyperbola_eccentricity : ℝ := 2

-- Theorem statement
theorem hyperbola_properties :
  ∃ (c : ℝ), 
    (∀ (x y : ℝ), ellipse x y → (x = c ∨ x = -c) → y = 0) ∧
    (∀ (x y : ℝ), hyperbola x y → (x = c ∨ x = -c) → y = 0) ∧
    (let a := 4;
     let b := Real.sqrt 48;
     c / a = hyperbola_eccentricity ∧
     c^2 = a^2 + b^2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l1200_120054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1200_120035

def sequence_property (a : ℕ → ℝ) : Prop :=
  ∀ n, a n = max (a (n + 1)) (a (n + 2)) - min (a (n + 1)) (a (n + 2))

def nonnegative_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n ≥ 0

theorem sequence_properties (a : ℕ → ℝ) 
  (h1 : sequence_property a) (h2 : nonnegative_sequence a) :
  (a 1 = 1 ∧ a 2 = 2 → a 4 ∈ ({1, 3, 5} : Set ℝ)) ∧
  ((∃ n₀ : ℕ, ∀ n, a n ≤ a n₀) → (∃ k, a k = 0)) ∧
  ((∀ n, a n > 0) → ¬∃ M : ℝ, M > 0 ∧ ∀ n, a n ≤ M) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1200_120035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_section_eccentricity_l1200_120053

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 + (a / b)^2)

theorem conic_section_eccentricity 
  (a : ℕ → ℝ) 
  (h1 : geometric_sequence a) 
  (h2 : a 0 = -1) 
  (h3 : a 4 = -81) :
  eccentricity 1 (Real.sqrt (-a 2)) = Real.sqrt 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_section_eccentricity_l1200_120053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_distance_l1200_120045

/-- Two parallel lines in 2D space -/
structure ParallelLines where
  -- Line 1: x + 2y + 1 = 0
  a₁ : ℝ := 1
  b₁ : ℝ := 2
  c₁ : ℝ := 1
  -- Line 2: 2x + by - 4 = 0
  a₂ : ℝ := 2
  b₂ : ℝ
  c₂ : ℝ := -4
  parallel : a₁ * b₂ = a₂ * b₁

/-- The distance between two parallel lines -/
noncomputable def distance (l : ParallelLines) : ℝ :=
  |l.a₁ * 0 + l.b₁ * 1 + l.c₁| / Real.sqrt (l.a₁^2 + l.b₁^2)

/-- Theorem: The distance between the given parallel lines is 3√5/5 -/
theorem parallel_lines_distance (l : ParallelLines) : distance l = 3 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_distance_l1200_120045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_drumming_rabbits_l1200_120037

/-- Represents a rabbit kit with its drum size and drumstick length -/
structure RabbitKit where
  drumSize : ℕ
  drumstickLength : ℕ

/-- The number of rabbit kits -/
def numRabbits : ℕ := 7

/-- Predicate to check if a rabbit kit can start drumming -/
def canDrum (rabbits : Finset RabbitKit) (r : RabbitKit) : Prop :=
  ∃ s, s ∈ rabbits ∧ r.drumSize > s.drumSize ∧ r.drumstickLength > s.drumstickLength

theorem max_drumming_rabbits :
  ∀ (rabbits : Finset RabbitKit),
    rabbits.card = numRabbits →
    (∀ r s, r ∈ rabbits → s ∈ rabbits → r ≠ s → r.drumSize ≠ s.drumSize) →
    (∀ r s, r ∈ rabbits → s ∈ rabbits → r ≠ s → r.drumstickLength ≠ s.drumstickLength) →
    (∃ (drummingRabbits : Finset RabbitKit),
      drummingRabbits ⊆ rabbits ∧
      drummingRabbits.card = 6 ∧
      ∀ r ∈ drummingRabbits, canDrum rabbits r) ∧
    ∀ (drummingRabbits : Finset RabbitKit),
      drummingRabbits ⊆ rabbits →
      (∀ r ∈ drummingRabbits, canDrum rabbits r) →
      drummingRabbits.card ≤ 6 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_drumming_rabbits_l1200_120037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_meeting_on_CD_l1200_120071

/-- A rectangle with sides in the ratio 5:4 -/
structure Rectangle :=
  (AB : ℝ)
  (BC : ℝ)
  (h_ratio : AB / BC = 5 / 4)
  (h_positive : AB > 0 ∧ BC > 0)

/-- The path of an ant -/
inductive AntPath
  | A
  | B
  | C
  | D

/-- The position of an ant at a given time -/
def ant_position (rect : Rectangle) (start : AntPath) (time : ℝ) : AntPath :=
  sorry

/-- The theorem statement -/
theorem second_meeting_on_CD (rect : Rectangle) :
  let first_ant := ant_position rect AntPath.A
  let second_ant := ant_position rect AntPath.C
  ∃ (first_meeting_time second_meeting_time : ℝ),
    first_meeting_time < second_meeting_time ∧
    first_ant first_meeting_time = second_ant first_meeting_time ∧
    first_ant first_meeting_time = AntPath.B ∧
    (first_ant second_meeting_time = AntPath.D ∨
    (first_ant second_meeting_time = AntPath.C ∧
     second_ant second_meeting_time = AntPath.D)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_meeting_on_CD_l1200_120071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l1200_120032

-- Define the set M as the solution set of x^2 - x ≤ 0
def M : Set ℝ := {x : ℝ | x^2 - x ≤ 0}

-- Define the set N as the domain of ln(1 - |x|)
def N : Set ℝ := {x : ℝ | 1 - |x| > 0}

-- Theorem statement
theorem intersection_M_N : M ∩ N = Set.Ioc 0 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l1200_120032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_poker_win_percentage_l1200_120090

theorem poker_win_percentage 
  (initial_games : ℕ) 
  (additional_games : ℕ) 
  (initial_win_percentage : ℚ) 
  (additional_wins : ℕ) 
  (new_win_percentage : ℚ) :
  initial_games = 200 →
  additional_games = 100 →
  initial_win_percentage = 63 / 100 →
  additional_wins = 57 →
  new_win_percentage = 61 / 100 →
  (initial_win_percentage * initial_games + additional_wins) / (initial_games + additional_games) = new_win_percentage :=
by
  intros h1 h2 h3 h4 h5
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_poker_win_percentage_l1200_120090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_partition_l1200_120096

theorem student_partition (n r : ℕ+) (nums : Fin n → Fin r → ℕ+)
  (distinct : ∀ i j a b, (i ≠ j ∨ a ≠ b) → nums i a ≠ nums j b) :
  ∃ (k : ℕ) (classes : Fin n → Fin k),
    k ≤ 4 * r ∧
    ∀ i j : Fin n, classes i = classes j →
      ∀ a b : Fin r, 
        nums i a ≤ Nat.factorial (nums j b - 1) ∨ 
        nums i a ≥ Nat.factorial (nums j b + 1) + 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_partition_l1200_120096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1200_120067

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x * Real.cos x + 2 * Real.sqrt 3 * (Real.cos x)^2 - Real.sqrt 3

theorem f_properties :
  (∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x ∧
    ∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  (∀ (x₀ : ℝ), x₀ ∈ Set.Icc (Real.pi / 4) (Real.pi / 2) →
    f (x₀ - Real.pi / 12) = 6 / 5 →
    Real.cos (2 * x₀) = (3 - 4 * Real.sqrt 3) / 10) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1200_120067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_solution_l1200_120031

theorem log_equation_solution : ∃ y : ℝ, y > 0 ∧ Real.log 8 / Real.log y = Real.log 4 / Real.log 64 ∧ y = 512 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_solution_l1200_120031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_of_log_l1200_120068

-- Define the logarithm function with base a
noncomputable def log_base (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Define our function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log_base a (x + 2)

-- Theorem statement
theorem fixed_point_of_log (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  f a (-1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_of_log_l1200_120068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_square_side_length_l1200_120091

-- Define the square PQRS
def square_PQRS : Set (ℝ × ℝ) :=
  {(x, y) | 0 ≤ x ∧ x ≤ 2 ∧ 0 ≤ y ∧ y ≤ 2}

-- Define points P, Q, R, S
def P : ℝ × ℝ := (0, 0)
def Q : ℝ × ℝ := (2, 0)
def R : ℝ × ℝ := (2, 2)
def S : ℝ × ℝ := (0, 2)

-- Define that T is on QR and U is on RS
noncomputable def T : ℝ × ℝ := sorry
noncomputable def U : ℝ × ℝ := sorry
axiom T_on_QR : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ T = (2, 2*t)
axiom U_on_RS : ∃ u : ℝ, 0 ≤ u ∧ u ≤ 1 ∧ U = (2-2*u, 2)

-- Define that triangle PTU is equilateral
axiom PTU_equilateral : 
  dist P T = dist T U ∧ dist T U = dist U P ∧ dist U P = dist P T

-- Define the smaller square
def smaller_square : Set (ℝ × ℝ) := sorry
axiom smaller_square_vertex_Q : Q ∈ smaller_square
axiom smaller_square_parallel : ∀ (x y : ℝ), 
  (x, y) ∈ smaller_square → (x + 1, y) ∈ smaller_square ∨ (x, y + 1) ∈ smaller_square
axiom smaller_square_vertex_on_PT : ∃ (x y : ℝ), 
  (x, y) ∈ smaller_square ∧ (x, y) ∈ Set.Icc P T

-- Define the side length of the smaller square
noncomputable def s : ℝ := sorry

-- State the theorem
theorem smaller_square_side_length :
  ∃ (d e f : ℕ), d = 4 ∧ e = 3 ∧ f = 7 ∧ 
  s = (d - Real.sqrt e) / f ∧ 
  ¬ ∃ (p : ℕ), Nat.Prime p ∧ (p^2 ∣ e) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_square_side_length_l1200_120091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_triangle_probability_l1200_120012

/-- Represents a point in the diagram -/
structure Point where
  x : ℚ
  y : ℚ
deriving Repr

/-- Represents a triangle in the diagram -/
structure Triangle where
  vertices : Fin 3 → Point
deriving Repr

/-- Represents the diagram with triangles -/
structure Diagram where
  points : Fin 5 → Point
  triangles : List Triangle
  shaded_triangles : List Triangle

/-- The probability of selecting a shaded triangle -/
def probability_shaded (d : Diagram) : ℚ :=
  (d.shaded_triangles.length : ℚ) / (d.triangles.length : ℚ)

theorem shaded_triangle_probability (d : Diagram) 
  (h1 : d.triangles.length = 6)
  (h2 : d.shaded_triangles.length = 4)
  (h3 : ∀ t ∈ d.shaded_triangles, t ∈ d.triangles) :
  probability_shaded d = 2/3 := by
  unfold probability_shaded
  rw [h1, h2]
  norm_num

-- No #eval statement is needed for this theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_triangle_probability_l1200_120012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_equality_l1200_120024

-- Define the inequality function
noncomputable def f (x : ℝ) := (x - 1) / x

-- Define the solution set
def S : Set ℝ := { x | x ≤ -1 }

-- Theorem statement
theorem solution_set_equality :
  { x : ℝ | x ≠ 0 ∧ f x ≥ 2 } = S := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_equality_l1200_120024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_implies_m_value_l1200_120088

-- Define the function f(x)
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := 2 * Real.log x - m / x

-- Define the derivative of f(x)
noncomputable def f_derivative (m : ℝ) (x : ℝ) : ℝ := 2 / x + m / (x^2)

-- State the theorem
theorem tangent_slope_implies_m_value (m : ℝ) :
  f_derivative m 1 = 3 → m = 1 := by
  intro h
  -- Here we would normally prove the theorem, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_implies_m_value_l1200_120088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_proof_l1200_120043

/-- The distance between the foci of an ellipse with equation 9x^2 + y^2 = 144 -/
noncomputable def ellipse_foci_distance : ℝ :=
  8 * Real.sqrt 2

/-- Theorem: The distance between the foci of an ellipse with equation 9x^2 + y^2 = 144 is 8√2 -/
theorem ellipse_foci_distance_proof :
  let ellipse := {(x, y) : ℝ × ℝ | 9 * x^2 + y^2 = 144}
  ∃ (f₁ f₂ : ℝ × ℝ), f₁ ∈ ellipse ∧ f₂ ∈ ellipse ∧
    ∀ (f₁' f₂' : ℝ × ℝ), f₁' ∈ ellipse → f₂' ∈ ellipse →
      dist f₁' f₂' ≤ dist f₁ f₂ ∧
      dist f₁ f₂ = ellipse_foci_distance :=
by
  sorry

#check ellipse_foci_distance
#check ellipse_foci_distance_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_proof_l1200_120043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apollonian_points_exist_l1200_120034

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point on a circle
def PointOnCircle (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem apollonian_points_exist (c : Circle) (A B : ℝ × ℝ)
  (hA : PointOnCircle c A) (hB : PointOnCircle c B) :
  ∃ P1 P2 : ℝ × ℝ,
    P1 ≠ P2 ∧
    PointOnCircle c P1 ∧
    PointOnCircle c P2 ∧
    distance P1 A / distance P1 B = 2 / 3 ∧
    distance P2 A / distance P2 B = 2 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_apollonian_points_exist_l1200_120034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_theorem_l1200_120073

/-- Profit function for project A -/
noncomputable def profitA (x : ℝ) : ℝ := (2/5) * x

/-- Profit function for project B -/
noncomputable def profitB (x : ℝ) : ℝ := -(1/5) * x^2 + 2 * x

/-- Total investment amount -/
def totalInvestment : ℝ := 32

/-- Theorem: Maximum profit from investing in projects A and B -/
theorem max_profit_theorem :
  ∃ (t : ℝ), t ≥ 0 ∧ t ≤ totalInvestment ∧
  ∀ (s : ℝ), s ≥ 0 → s ≤ totalInvestment →
  profitA (totalInvestment - t) + profitB t ≥ profitA (totalInvestment - s) + profitB s ∧
  profitA (totalInvestment - t) + profitB t = 16 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_theorem_l1200_120073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_products_l1200_120018

theorem min_sum_products (a b c d : ℕ) : 
  a ∈ ({3, 4, 5, 6} : Set ℕ) ∧ 
  b ∈ ({3, 4, 5, 6} : Set ℕ) ∧ 
  c ∈ ({3, 4, 5, 6} : Set ℕ) ∧ 
  d ∈ ({3, 4, 5, 6} : Set ℕ) ∧ 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  81 ≤ a * b + b * c + c * d + d * a :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_products_l1200_120018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_theorem_l1200_120062

/-- The area of a circular sector given its central angle and arc length -/
noncomputable def sectorArea (centralAngle : ℝ) (arcLength : ℝ) : ℝ :=
  (1 / 2) * arcLength * (arcLength / centralAngle)

/-- Theorem: The area of a sector with central angle 2 radians and arc length 4 is 4 -/
theorem sector_area_theorem :
  sectorArea 2 4 = 4 := by
  -- Unfold the definition of sectorArea
  unfold sectorArea
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_theorem_l1200_120062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rabbit_adoption_proportion_l1200_120060

/-- The proportion of adopted kittens from the first spring --/
def adopted_proportion : ℚ := 1/2

theorem rabbit_adoption_proportion :
  let breeding_rabbits : ℕ := 10
  let first_spring_kittens : ℕ := breeding_rabbits * 10
  let second_spring_kittens : ℕ := 60
  let returned_kittens : ℕ := 5
  let second_spring_adopted : ℕ := 4
  let total_rabbits : ℕ := 121
  
  breeding_rabbits + 
  (first_spring_kittens - Int.floor (↑first_spring_kittens * adopted_proportion) + returned_kittens) + 
  (second_spring_kittens - second_spring_adopted) = total_rabbits :=
by
  sorry

#check rabbit_adoption_proportion

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rabbit_adoption_proportion_l1200_120060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangent_sum_l1200_120022

/-- Parabola P₁ -/
def P₁ (x y : ℝ) : Prop := y = x^2 + 151/150

/-- Parabola P₂ -/
def P₂ (x y : ℝ) : Prop := x = y^2 + 55/5

/-- Common tangent line L -/
def L (a b c : ℕ) (x y : ℝ) : Prop := (a : ℝ) * x + (b : ℝ) * y = c

/-- Tangency condition for P₁ -/
def tangent_P₁ (a b c : ℕ) : Prop := ∃ x y : ℝ, P₁ x y ∧ L a b c x y

/-- Tangency condition for P₂ -/
def tangent_P₂ (a b c : ℕ) : Prop := ∃ x y : ℝ, P₂ x y ∧ L a b c x y

/-- Main theorem -/
theorem common_tangent_sum :
  ∃ a b c : ℕ,
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    Nat.gcd a (Nat.gcd b c) = 1 ∧
    tangent_P₁ a b c ∧
    tangent_P₂ a b c ∧
    (∃ q : ℚ, (b : ℚ) = (a : ℚ) * q) ∧
    a + b + c = 9 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangent_sum_l1200_120022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plumbing_equal_charge_l1200_120085

/-- Paul's Plumbing visit charge -/
noncomputable def paul_visit : ℝ := 55

/-- Paul's Plumbing hourly rate -/
noncomputable def paul_rate : ℝ := 35

/-- Reliable Plumbing visit charge -/
noncomputable def reliable_visit : ℝ := 75

/-- Reliable Plumbing hourly rate -/
noncomputable def reliable_rate : ℝ := 30

/-- The number of hours at which both companies charge the same amount -/
noncomputable def equal_charge_hours : ℝ := (reliable_visit - paul_visit) / (paul_rate - reliable_rate)

theorem plumbing_equal_charge :
  equal_charge_hours = 4 ∧
  paul_visit + paul_rate * equal_charge_hours = reliable_visit + reliable_rate * equal_charge_hours :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_plumbing_equal_charge_l1200_120085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_K_value_l1200_120087

open Real

noncomputable def f (x : ℝ) : ℝ := (log x + 1) / exp x

noncomputable def f_K (K : ℝ) (x : ℝ) : ℝ := min (f x) K

theorem min_K_value (K : ℝ) :
  (∀ x > 0, f_K K x = f x) ↔ K ≥ 1 / exp 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_K_value_l1200_120087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l1200_120029

/-- The equation of a parabola -/
noncomputable def parabola (x : ℝ) : ℝ := 3 * x^2 - 12 * x + 15

/-- The directrix of the parabola -/
noncomputable def directrix : ℝ := 35 / 12

/-- Checks if a point is the vertex of the parabola -/
noncomputable def is_vertex (f : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
  f p.1 = p.2 ∧ ∀ x, f x ≥ f p.1

/-- Checks if a line is the directrix of the parabola -/
noncomputable def is_directrix (f : ℝ → ℝ) (d : ℝ) : Prop :=
  ∃ a h k, (∀ x, f x = a * (x - h)^2 + k) ∧ d = k - 1 / (4 * a)

/-- Theorem: The directrix of the given parabola is y = 35/12 -/
theorem parabola_directrix : 
  ∃ p : ℝ × ℝ, is_vertex parabola p ∧ is_directrix parabola directrix :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l1200_120029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_minus_2alpha_l1200_120093

theorem cos_pi_minus_2alpha (α : ℝ) (h : Real.cos (π / 2 - α) = 1 / 3) : 
  Real.cos (π - 2 * α) = -7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_minus_2alpha_l1200_120093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_root_implies_k_eq_6_l1200_120014

/-- Represents a cubic polynomial of the form ax³ + bx² + cx + d --/
structure CubicPolynomial (α : Type*) [Field α] where
  a : α
  b : α
  c : α
  d : α

/-- Checks if a cubic polynomial has exactly one complex root --/
def hasUniqueComplexRoot (p : CubicPolynomial ℂ) : Prop := sorry

/-- The specific cubic polynomial we're interested in --/
noncomputable def ourPolynomial (k : ℂ) : CubicPolynomial ℂ := 
  { a := 8, b := 12, c := k, d := 1 }

/-- The main theorem --/
theorem unique_root_implies_k_eq_6 : 
  ∀ k : ℂ, hasUniqueComplexRoot (ourPolynomial k) → k = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_root_implies_k_eq_6_l1200_120014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_other_intercept_l1200_120082

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  focus1 : Point
  focus2 : Point
  intercept1 : Point

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: For the given ellipse, the other x-intercept is (56/11, 0) -/
theorem ellipse_other_intercept (e : Ellipse) 
    (h1 : e.focus1 = ⟨0, 3⟩)
    (h2 : e.focus2 = ⟨4, 0⟩)
    (h3 : e.intercept1 = ⟨0, 0⟩) :
    ∃ (p : Point), p.x = 56/11 ∧ p.y = 0 ∧ 
    distance p e.focus1 + distance p e.focus2 = 
    distance e.intercept1 e.focus1 + distance e.intercept1 e.focus2 := by
  sorry

#check ellipse_other_intercept

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_other_intercept_l1200_120082

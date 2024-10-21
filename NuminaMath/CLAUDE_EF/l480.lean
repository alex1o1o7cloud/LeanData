import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_translation_theorem_l480_48027

/-- Represents a trigonometric function of the form y = A * sin(Bx + C) -/
structure TrigFunction where
  A : ℝ
  B : ℝ
  C : ℝ

/-- The original function y = 3sin(2x) -/
def original : TrigFunction := { A := 3, B := 2, C := 0 }

/-- The amount of right translation -/
noncomputable def translation : ℝ := Real.pi / 6

/-- The translated function -/
noncomputable def translated : TrigFunction := { A := 3, B := 2, C := -Real.pi / 3 }

/-- 
Theorem stating that translating the graph of y = 3sin(2x) to the right by π/6 units 
results in the equation y = 3sin(2x - π/3)
-/
theorem translation_theorem : 
  ∀ (x : ℝ), 
    (original.A * Real.sin (original.B * (x - translation) + original.C)) = 
    (translated.A * Real.sin (translated.B * x + translated.C)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_translation_theorem_l480_48027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_8_divisors_l480_48036

-- Define 8!
def factorial_8 : ℕ := 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1

-- Define the number of positive divisors function
def num_positive_divisors (n : ℕ) : ℕ := (Nat.divisors n).card

-- Theorem statement
theorem factorial_8_divisors :
  num_positive_divisors factorial_8 = 96 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_8_divisors_l480_48036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_curve_to_line_l480_48057

/-- Curve C in the x-y plane -/
def curve_C (x y : ℝ) : Prop :=
  x^2 / 3 + y^2 = 1

/-- Line l in the x-y plane -/
def line_l (x y : ℝ) : Prop :=
  x + y = 4

/-- Distance from a point (x, y) to the line l -/
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  |x + y - 4| / Real.sqrt 2

/-- The maximum distance from any point on curve C to line l is 3√2 -/
theorem max_distance_curve_to_line :
  ∃ (x₀ y₀ : ℝ), curve_C x₀ y₀ ∧
    (∀ (x y : ℝ), curve_C x y → distance_to_line x y ≤ distance_to_line x₀ y₀) ∧
    distance_to_line x₀ y₀ = 3 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_curve_to_line_l480_48057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_quadrant_half_angle_tangent_l480_48054

theorem second_quadrant_half_angle_tangent (α : Real) : 
  (π / 2 < α) ∧ (α < π) → Real.tan (α / 2) > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_quadrant_half_angle_tangent_l480_48054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_congruence_sss_l480_48052

/-- Definition of a triangle -/
structure Triangle where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ

/-- Definition of triangle congruence -/
def Triangle.congruent (T1 T2 : Triangle) : Prop :=
  T1.side1 = T2.side1 ∧ T1.side2 = T2.side2 ∧ T1.side3 = T2.side3

notation:50 T1 " ≅ " T2 => Triangle.congruent T1 T2

/-- Two triangles are congruent if their corresponding sides are equal -/
theorem triangle_congruence_sss (T1 T2 : Triangle) 
  (h1 : T1.side1 = T2.side1)
  (h2 : T1.side2 = T2.side2)
  (h3 : T1.side3 = T2.side3) : 
  T1 ≅ T2 := by
  unfold Triangle.congruent
  exact ⟨h1, h2, h3⟩

#check triangle_congruence_sss

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_congruence_sss_l480_48052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l480_48013

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := f (x + Real.pi / 4) + f (x + 3 * Real.pi / 4)

-- Theorem statement
theorem function_properties :
  (f (Real.pi / 2) = 1) ∧
  (∃ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) ∧
    (∀ (q : ℝ), q > 0 → (∀ (x : ℝ), f (x + q) = f x) → p ≤ q)) ∧
  (∃ (m : ℝ), ∀ (x : ℝ), g x ≥ m ∧ ∃ (x₀ : ℝ), g x₀ = m) ∧
  (∃ (x₀ : ℝ), g x₀ = -2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l480_48013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_perimeter_l480_48029

/-- The perimeter of a right-angled triangle with area S and median to hypotenuse of length m -/
noncomputable def trianglePerimeter (S m : ℝ) : ℝ :=
  Real.sqrt (4 * m^2 + 4 * S) + 2 * m

/-- Theorem: The perimeter of a right-angled triangle with area S and median to hypotenuse of length m is √(4m² + 4S) + 2m -/
theorem right_triangle_perimeter (S m : ℝ) (h1 : S > 0) (h2 : m > 0) :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  a^2 + b^2 = c^2 ∧  -- right-angled triangle condition
  (1/2) * a * b = S ∧  -- area condition
  c = 2 * m ∧  -- median to hypotenuse condition
  a + b + c = trianglePerimeter S m :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_perimeter_l480_48029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l480_48039

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_properties
  (a : ℕ → ℚ)
  (h_arithmetic : arithmetic_sequence a)
  (h_first_term : a 1 = -3)
  (h_condition : 11 * a 5 = 5 * a 8) :
  (∀ n : ℕ, a n = 2 * (n : ℚ) - 5) ∧
  (∀ n : ℕ, ∃ S : ℚ, S = (n : ℚ)^2 - 4*(n : ℚ)) ∧
  (∀ n : ℕ, (n : ℚ)^2 - 4*(n : ℚ) ≥ -4) ∧
  ((2 : ℚ)^2 - 4*2 = -4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l480_48039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_equals_sqrt2_over_2_l480_48021

-- Define the angle α
def α : Real := sorry

-- Define the terminal point
def terminal_point : ℝ × ℝ := (1, -1)

-- Define the distance from origin to the terminal point
noncomputable def r : ℝ := Real.sqrt (terminal_point.1^2 + terminal_point.2^2)

-- Theorem statement
theorem cos_alpha_equals_sqrt2_over_2 :
  Real.cos α = Real.sqrt 2 / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_equals_sqrt2_over_2_l480_48021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_l480_48074

-- Define the set T
def T : Set (ℝ × ℝ × ℝ) :=
  {p | p.1 ≥ 0 ∧ p.2.1 ≥ 0 ∧ p.2.2 ≥ 0 ∧ p.1 + p.2.1 + p.2.2 = 1}

-- Define the support relation
def supports (p : ℝ × ℝ × ℝ) (a b c : ℝ) : Prop :=
  (p.1 ≥ a ∧ p.2.1 ≥ b) ∨ (p.1 ≥ a ∧ p.2.2 ≥ c) ∨ (p.2.1 ≥ b ∧ p.2.2 ≥ c)

-- Define the set S
def S : Set (ℝ × ℝ × ℝ) :=
  {p ∈ T | supports p (1/4) (1/4) (1/4)}

-- Define the area function (noncomputable as it involves integration)
noncomputable def area (s : Set (ℝ × ℝ × ℝ)) : ℝ := sorry

-- State the theorem
theorem area_ratio : (area S) / (area T) = 3/16 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_l480_48074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_functions_properties_l480_48024

-- Define quadratic functions
def f (x : ℝ) : ℝ := -3 * x^2 + 3 * x + 1
def g (x : ℝ) : ℝ := 4 * x^2 - 4 * x + 1

-- Define friendly coaxial quadratic functions
def friendly_coaxial (f g : ℝ → ℝ) : Prop :=
  ∃ (a b c d e : ℝ), (∀ x, f x = a * x^2 + b * x + c) ∧ 
                      (∀ x, g x = d * x^2 + e * x + c) ∧ 
                      a + d = 1 ∧
                      b / (2 * a) = e / (2 * d)

-- Define C₁ and C₂
def C₁ (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 4 * a * x + 4
def C₂ (a : ℝ) (x : ℝ) : ℝ := (1 - a) * x^2 + 4 * (1 - a) * x + 4

theorem quadratic_functions_properties :
  (∀ x, f x = -3 * x^2 + 3 * x + 1) ∧
  (∀ x, g x = 4 * x^2 - 4 * x + 1) ∧
  friendly_coaxial f g ∧
  (∀ a, a ≠ 0 → a ≠ 1 → a ≠ 1/2 → 
    ∃ A B : ℝ × ℝ, 
      C₁ a A.1 = C₂ a A.1 ∧
      C₁ a B.1 = C₂ a B.1 ∧
      A.1 < B.1 ∧
      B.1 - A.1 = 4) ∧
  (∃ a : ℝ, a = -1 ∨ a = 3) ∧
  (∀ a, a = -1 ∨ a = 3 →
    ∃ max min : ℝ,
      (∀ x, -3 ≤ x ∧ x ≤ 0 → C₂ a x ≤ max) ∧
      (∀ x, -3 ≤ x ∧ x ≤ 0 → min ≤ C₂ a x) ∧
      max - min = 8) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_functions_properties_l480_48024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sugar_concentration_problem_l480_48079

theorem sugar_concentration_problem (initial_concentration : ℝ) 
  (final_concentration : ℝ) (replaced_fraction : ℝ) 
  (h1 : initial_concentration = 0.08)
  (h2 : final_concentration = 0.20)
  (h3 : replaced_fraction = 1/3) :
  (final_concentration - (1 - replaced_fraction) * initial_concentration) / replaced_fraction = 0.44 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sugar_concentration_problem_l480_48079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l480_48001

def circle_equation (x y : ℝ) : Prop :=
  x^2 + 16*y + 89 = -y^2 - 12*x

def circle_center : ℝ × ℝ := (-6, -8)

noncomputable def circle_radius : ℝ := Real.sqrt 11

theorem circle_properties :
  let (p, q) := circle_center
  let s := circle_radius
  (∀ x y, circle_equation x y ↔ (x - p)^2 + (y - q)^2 = s^2) ∧
  p + q + s = -14 + Real.sqrt 11 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l480_48001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_after_removal_l480_48007

theorem arithmetic_mean_after_removal (S : Finset ℝ) (x y z : ℝ) :
  S.card = 60 →
  x = 50 ∧ y = 60 ∧ z = 70 →
  x ∈ S ∧ y ∈ S ∧ z ∈ S →
  (S.sum id) / S.card = 42 →
  ((S.sum id) - (x + y + z)) / (S.card - 3) = 41 := by
  sorry

#check arithmetic_mean_after_removal

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_after_removal_l480_48007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_x_cos_x_l480_48019

theorem integral_x_cos_x (x : ℝ) :
  deriv (λ x => x * Real.sin x + Real.cos x) x = x * Real.cos x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_x_cos_x_l480_48019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_escalator_problem_l480_48093

/-- The time taken for a person to cover the entire length of an escalator -/
noncomputable def escalatorTime (escalatorSpeed : ℝ) (escalatorLength : ℝ) (personSpeed : ℝ) : ℝ :=
  escalatorLength / (escalatorSpeed + personSpeed)

/-- Theorem: The time taken is 10 seconds under the given conditions -/
theorem escalator_problem :
  let escalatorSpeed : ℝ := 15
  let escalatorLength : ℝ := 200
  let personSpeed : ℝ := 5
  escalatorTime escalatorSpeed escalatorLength personSpeed = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_escalator_problem_l480_48093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_with_y_intercept_l480_48097

/-- A line in the form ax + by + c = 0 -/
structure Line where
  a : ℚ
  b : ℚ
  c : ℚ

/-- Two lines are parallel if their slopes are equal -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- The y-intercept of a line -/
noncomputable def y_intercept (l : Line) : ℚ :=
  -l.c / l.b

theorem parallel_line_with_y_intercept (m n : ℚ) :
  let l1 : Line := ⟨m, n, 2⟩
  let l2 : Line := ⟨1, -2, -5⟩
  parallel l1 l2 ∧ y_intercept l1 = 1 → m = 1 ∧ n = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_with_y_intercept_l480_48097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l480_48032

/-- Parabola with equation y^2 = 4x -/
structure Parabola where
  focus : ℝ × ℝ

/-- Circle with equation (x-4)^2 + (y-1)^2 = 1 -/
def Circle : Set (ℝ × ℝ) :=
  {p | (p.1 - 4)^2 + (p.2 - 1)^2 = 1}

/-- Distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem min_distance_sum (p : Parabola) :
  ∃ (c : ℝ), ∀ (m : ℝ × ℝ) (a : ℝ × ℝ),
    m.2^2 = 4 * m.1 →
    a ∈ Circle →
    c ≤ distance m a + distance m p.focus ∧
    (∃ (m' : ℝ × ℝ) (a' : ℝ × ℝ),
      m'.2^2 = 4 * m'.1 ∧
      a' ∈ Circle ∧
      c = distance m' a' + distance m' p.focus) ∧
    c = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l480_48032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l480_48004

/-- Hyperbola M: x^2 - y^2/b^2 = 1 (b > 0) -/
def hyperbola (b : ℝ) (x y : ℝ) : Prop := x^2 - y^2/b^2 = 1 ∧ b > 0

/-- Line l with slope 1 -/
def line (x y : ℝ) : Prop := y = x + 1

/-- Point A is the left vertex of the hyperbola -/
def left_vertex : ℝ × ℝ := (-1, 0)

/-- B and C are intersection points of line l and asymptotes of M -/
def intersect_asymptotes (b : ℝ) (B C : ℝ × ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), B = (x₁, y₁) ∧ C = (x₂, y₂) ∧
  x₁^2 - y₁^2/b^2 = 0 ∧ x₂^2 - y₂^2/b^2 = 0 ∧
  line x₁ y₁ ∧ line x₂ y₂

/-- |AB| = |BC| -/
def equal_segments (A B C : ℝ × ℝ) : Prop :=
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2

/-- Eccentricity of a hyperbola -/
noncomputable def eccentricity (b : ℝ) : ℝ := Real.sqrt (1 + 1/b^2)

theorem hyperbola_eccentricity (b : ℝ) (B C : ℝ × ℝ) :
  hyperbola b (left_vertex.1) (left_vertex.2) →
  intersect_asymptotes b B C →
  equal_segments left_vertex B C →
  eccentricity b = Real.sqrt 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l480_48004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_from_surface_area_l480_48061

/-- The radius of a sphere given its surface area -/
theorem sphere_radius_from_surface_area (surface_area : ℝ) : 
  surface_area = 64 * Real.pi → Real.sqrt (surface_area / (4 * Real.pi)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_from_surface_area_l480_48061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_triangle_with_circles_l480_48023

/-- An equilateral triangle with three inscribed circles -/
structure TriangleWithCircles where
  /-- The radius of each inscribed circle -/
  r : ℝ
  /-- Assumption that the radius is positive -/
  r_pos : r > 0

/-- The perimeter of the equilateral triangle -/
noncomputable def perimeter (t : TriangleWithCircles) : ℝ :=
  12 * Real.sqrt 3 + 24

/-- Theorem stating the perimeter of the triangle with inscribed circles of radius 4 -/
theorem perimeter_of_triangle_with_circles :
  ∀ t : TriangleWithCircles, t.r = 4 → perimeter t = 12 * Real.sqrt 3 + 24 := by
  sorry

#check perimeter_of_triangle_with_circles

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_triangle_with_circles_l480_48023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_to_last_digit_of_square_ending_in_five_l480_48083

theorem third_to_last_digit_of_square_ending_in_five (n : ℤ) :
  n % 10 = 5 →
  ∃ k : ℤ, (n^2 % 1000) ∈ ({100 * k + 25 | k ∈ ({0, 2, 6} : Set ℤ)} : Set ℤ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_to_last_digit_of_square_ending_in_five_l480_48083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_earnings_proof_l480_48076

noncomputable def monday_fabric : ℝ := 20
noncomputable def monday_yarn : ℝ := 15
noncomputable def tuesday_fabric : ℝ := 2 * monday_fabric
noncomputable def tuesday_yarn : ℝ := monday_yarn + 10
noncomputable def wednesday_fabric : ℝ := (1 / 4) * tuesday_fabric
noncomputable def wednesday_yarn : ℝ := (1 / 2) * tuesday_yarn

noncomputable def fabric_price : ℝ := 2
noncomputable def yarn_price : ℝ := 3

noncomputable def fabric_discount_threshold : ℝ := 30
noncomputable def yarn_discount_threshold : ℝ := 20

noncomputable def fabric_discount_rate : ℝ := 0.1
noncomputable def yarn_discount_rate : ℝ := 0.05

noncomputable def total_fabric : ℝ := monday_fabric + tuesday_fabric + wednesday_fabric
noncomputable def total_yarn : ℝ := monday_yarn + tuesday_yarn + wednesday_yarn

noncomputable def fabric_earnings : ℝ := total_fabric * fabric_price
noncomputable def yarn_earnings : ℝ := total_yarn * yarn_price

noncomputable def fabric_discount : ℝ := if total_fabric > fabric_discount_threshold then fabric_discount_rate * fabric_earnings else 0
noncomputable def yarn_discount : ℝ := if total_yarn > yarn_discount_threshold then yarn_discount_rate * yarn_earnings else 0

noncomputable def total_earnings : ℝ := (fabric_earnings - fabric_discount) + (yarn_earnings - yarn_discount)

theorem total_earnings_proof : total_earnings = 275.625 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_earnings_proof_l480_48076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_measure_l480_48080

theorem triangle_angle_measure (a b c : ℝ) (h : a^2 - c^2 = b^2 - Real.sqrt 3 * b * c) :
  Real.arccos ((b^2 + c^2 - a^2) / (2 * b * c)) = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_measure_l480_48080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_inequality_l480_48017

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 3  -- Added case for 0
  | 1 => 3
  | (n + 2) => (sequence_a (n + 1))^2 / (n + 2)

theorem sequence_inequality (n : ℕ) (h : n ≥ 2) : sequence_a (n - 1) > 2 * Real.sqrt n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_inequality_l480_48017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_bounded_shape_l480_48067

-- Define the parametric equations
noncomputable def x (t : ℝ) : ℝ := t - Real.sin t
noncomputable def y (t : ℝ) : ℝ := 1 - Real.cos t

-- Define the domain
def domain (t : ℝ) : Prop := 0 < x t ∧ x t < 2 * Real.pi ∧ y t ≥ 1

-- Define the bounded area
noncomputable def bounded_area : ℝ :=
  ∫ t in Set.Icc (Real.pi / 2) (3 * Real.pi / 2), (1 - Real.cos t) * (1 - Real.cos t)

-- State the theorem
theorem area_of_bounded_shape :
  bounded_area = Real.pi / 2 + 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_bounded_shape_l480_48067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_fraction_count_correct_l480_48082

/-- The number of fractions in simplest form -/
def simplestFractionCount : ℕ := 1009

/-- The upper bound of n -/
def upperBound : ℕ := 2017

/-- A fraction n/(n+4) is in simplest form if and only if n is odd -/
def isSimplestFraction (n : ℕ) : Prop := n % 2 = 1

/-- The count of numbers satisfying isSimplestFraction up to upperBound -/
def countSimplestFractions : ℕ :=
  (Finset.range upperBound).filter (fun n => n % 2 = 1) |>.card

theorem simplest_fraction_count_correct :
  countSimplestFractions = simplestFractionCount := by
  sorry

#eval countSimplestFractions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_fraction_count_correct_l480_48082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_equal_angles_with_plane_l480_48002

-- Definitions to match the problem conditions
def Line : Type := sorry
def Plane : Type := sorry

/-- The angle formed by a line with a plane -/
noncomputable def angle_with_plane (l : Line) (p : Plane) : ℝ := sorry

/-- Parallel relation between two lines -/
def parallel (l1 l2 : Line) : Prop := sorry

infix:50 " ∥ " => parallel

/-- Given two lines and a plane, if the lines are parallel, 
    then their angles with the plane are equal, 
    but equal angles do not necessarily imply parallel lines. -/
theorem parallel_lines_equal_angles_with_plane 
  (m n : Line) (α : Plane) : 
  m ∥ n → angle_with_plane m α = angle_with_plane n α ∧ 
  ∃ m' n' : Line, angle_with_plane m' α = angle_with_plane n' α ∧ ¬(m' ∥ n') :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_equal_angles_with_plane_l480_48002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acetic_acid_molecular_weight_l480_48016

/-- The atomic weight of Carbon in g/mol -/
def carbon_weight : ℝ := 12.01

/-- The atomic weight of Hydrogen in g/mol -/
def hydrogen_weight : ℝ := 1.008

/-- The atomic weight of Oxygen in g/mol -/
def oxygen_weight : ℝ := 16.00

/-- The number of Carbon atoms in Acetic acid -/
def carbon_count : ℕ := 2

/-- The number of Hydrogen atoms in Acetic acid -/
def hydrogen_count : ℕ := 4

/-- The number of Oxygen atoms in Acetic acid -/
def oxygen_count : ℕ := 2

/-- The molecular weight of Acetic acid in g/mol -/
def acetic_acid_weight : ℝ :=
  carbon_weight * (carbon_count : ℝ) +
  hydrogen_weight * (hydrogen_count : ℝ) +
  oxygen_weight * (oxygen_count : ℝ)

theorem acetic_acid_molecular_weight :
  abs (acetic_acid_weight - 60.052) < 0.001 := by
  -- Unfold the definition of acetic_acid_weight
  unfold acetic_acid_weight
  -- Simplify the expression
  simp [carbon_weight, hydrogen_weight, oxygen_weight, carbon_count, hydrogen_count, oxygen_count]
  -- Prove the inequality
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_acetic_acid_molecular_weight_l480_48016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_side_approx_pentagon_area_equals_perimeter_l480_48000

/-- The side length of a regular pentagon whose area equals its perimeter -/
noncomputable def pentagon_side : ℝ :=
  20 / Real.sqrt (5 * (5 + 2 * Real.sqrt 5))

/-- The perimeter of a regular pentagon with side length s -/
def perimeter (s : ℝ) : ℝ := 5 * s

/-- The area of a regular pentagon with side length s -/
noncomputable def area (s : ℝ) : ℝ := (1 / 4) * Real.sqrt (5 * (5 + 2 * Real.sqrt 5)) * s^2

theorem pentagon_side_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.001 ∧ abs (pentagon_side - 1.789) < ε :=
by sorry

theorem pentagon_area_equals_perimeter :
  area pentagon_side = perimeter pentagon_side :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_side_approx_pentagon_area_equals_perimeter_l480_48000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_rectangle_area_is_11_1_13_l480_48035

/-- The ellipse equation: 4x^2 + 9y^2 = 36 -/
def ellipse (x y : ℝ) : Prop := 4 * x^2 + 9 * y^2 = 36

/-- A point (x, y) is on the ellipse -/
def on_ellipse (p : ℝ × ℝ) : Prop := ellipse p.1 p.2

/-- The tangent line to the ellipse at point (x₁, y₁) -/
def tangent_line (x₁ y₁ x y : ℝ) : Prop := 4 * x₁ * x + 9 * y₁ * y = 36

/-- A point (x, y) is on the tangent line at (x₁, y₁) -/
def on_tangent (p : ℝ × ℝ) (x₁ y₁ : ℝ) : Prop := tangent_line x₁ y₁ p.1 p.2

/-- The square's diagonals lie on the coordinate axes -/
def square_diagonals_on_axes (s : Set (ℝ × ℝ)) : Prop :=
  ∃ a b : ℝ, (a, 0) ∈ s ∧ (-a, 0) ∈ s ∧ (0, b) ∈ s ∧ (0, -b) ∈ s

/-- The square's sides touch the ellipse -/
def square_touches_ellipse (s : Set (ℝ × ℝ)) : Prop :=
  ∃ p₁ p₂ p₃ p₄ : ℝ × ℝ, 
    p₁ ∈ s ∧ p₂ ∈ s ∧ p₃ ∈ s ∧ p₄ ∈ s ∧
    on_ellipse p₁ ∧ on_ellipse p₂ ∧ on_ellipse p₃ ∧ on_ellipse p₄

/-- The area of the rectangle formed by the points of tangency -/
noncomputable def tangent_rectangle_area (s : Set (ℝ × ℝ)) : ℝ := 
  sorry

/-- Main theorem: The area of the rectangle formed by the points of tangency is 11 1/13 -/
theorem tangent_rectangle_area_is_11_1_13 (s : Set (ℝ × ℝ)) :
  square_diagonals_on_axes s → square_touches_ellipse s → tangent_rectangle_area s = 11 + 1/13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_rectangle_area_is_11_1_13_l480_48035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_bounds_l480_48025

noncomputable def f (x : ℝ) : ℝ := (2 * x + 1) / (x + 1)

theorem f_monotonicity_and_bounds :
  (∀ x₁ x₂ : ℝ, 1 ≤ x₁ ∧ x₁ < x₂ → f x₁ < f x₂) ∧
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 4 → 3/2 ≤ f x ∧ f x ≤ 9/5) ∧
  f 1 = 3/2 ∧ f 4 = 9/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_bounds_l480_48025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_snakes_that_can_add_are_happy_l480_48028

-- Define the universe of snakes
variable (Snake : Type)

-- Define properties of snakes
variable (purple : Snake → Prop)
variable (happy : Snake → Prop)
variable (can_add : Snake → Prop)
variable (can_subtract : Snake → Prop)
variable (magical : Snake → Prop)

-- State the theorem
theorem snakes_that_can_add_are_happy 
  [DecidablePred purple]
  [DecidablePred happy]
  [DecidablePred magical]
  (total_snakes : Finset Snake)
  (h_total : total_snakes.card = 15)
  (h_purple : (total_snakes.filter purple).card = 6)
  (h_happy : (total_snakes.filter happy).card = 7)
  (h_magical : (total_snakes.filter magical).card = 3)
  (h_happy_can_add : ∀ s : Snake, happy s → can_add s)
  (h_purple_cant_subtract : ∀ s : Snake, purple s → ¬can_subtract s)
  (h_cant_subtract_cant_add : ∀ s : Snake, ¬can_subtract s → ¬can_add s)
  (h_magical_def : ∀ s : Snake, magical s ↔ (can_add s ∧ can_subtract s))
  : ∀ s : Snake, can_add s → happy s :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_snakes_that_can_add_are_happy_l480_48028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_result_l480_48012

/-- The result of the expression ( 0.76 × 0.76 × 0.76 − 0.008 ) / ( 0.76 × 0.76 + 0.76 × 0.2 + 0.04 ) is approximately 0.560. -/
theorem expression_result :
  ∃ (expr : ℝ), abs (expr - 0.560) < 0.001 ∧
  expr = (0.76 * 0.76 * 0.76 - 0.008) / (0.76 * 0.76 + 0.76 * 0.2 + 0.04) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_result_l480_48012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_sin_cos_function_l480_48073

theorem max_value_sin_cos_function :
  ∃ M : ℝ, M = 9/2 + 2 * Real.sqrt 2 ∧
  (∀ x : ℝ, (Real.sin x - 2) * (Real.cos x - 2) ≤ M) ∧
  (∃ x : ℝ, (Real.sin x - 2) * (Real.cos x - 2) = M) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_sin_cos_function_l480_48073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_has_property_P_infinite_composites_with_property_P_l480_48063

-- Define property P
def has_property_P (n : ℕ) : Prop :=
  ∀ a : ℕ, (n ∣ a^n - 1) → (n^2 ∣ a^n - 1)

-- Theorem 1: Every prime has property P
theorem prime_has_property_P (p : ℕ) (hp : Nat.Prime p) : has_property_P p :=
sorry

-- Theorem 2: Infinitely many composite numbers have property P
theorem infinite_composites_with_property_P :
  ∃ S : Set ℕ, (Set.Infinite S) ∧ (∀ n ∈ S, ¬Nat.Prime n ∧ has_property_P n) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_has_property_P_infinite_composites_with_property_P_l480_48063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_city_miles_per_tankful_l480_48042

/-- Represents the fuel efficiency and travel distance of a car in different environments --/
structure CarFuelData where
  highway_miles_per_tankful : ℚ
  city_mpg : ℚ
  mpg_difference : ℚ

/-- Calculates the miles per tankful in the city given the car's fuel data --/
def miles_per_tankful_city (data : CarFuelData) : ℚ :=
  let highway_mpg := data.city_mpg + data.mpg_difference
  let tank_size := data.highway_miles_per_tankful / highway_mpg
  tank_size * data.city_mpg

/-- Theorem: Given the specified conditions, the car travels 336 miles per tankful in the city --/
theorem car_city_miles_per_tankful (data : CarFuelData) 
    (h1 : data.highway_miles_per_tankful = 480)
    (h2 : data.mpg_difference = 6)
    (h3 : data.city_mpg = 14) : 
  miles_per_tankful_city data = 336 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_city_miles_per_tankful_l480_48042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_l480_48066

def mySequence : List ℕ := [2, 5, 11, 20, 32, 47]

def differences (s : List ℕ) : List ℕ :=
  List.zipWith (·-·) (s.tail) s

theorem sequence_property :
  differences mySequence = [3, 6, 9, 12, 15] ∧ mySequence[4] = 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_l480_48066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l480_48040

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := 
  Real.sin (2 * x + Real.pi / 3) + Real.cos (2 * x + Real.pi / 6) + m * Real.sin (2 * x)

theorem triangle_perimeter (m : ℝ) (A B C a b c : ℝ) : 
  f m (Real.pi / 12) = 2 →
  b = 2 →
  f m (B / 2) = Real.sqrt 3 →
  (1 / 2) * a * c * Real.sin B = Real.sqrt 3 →
  a + b + c = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l480_48040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_moves_to_determine_l480_48059

/-- The digit sum of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- The result of a query for a given x and a -/
def query_result (x a : ℕ) : ℕ := digit_sum (Int.natAbs (x - a))

/-- The type of a strategy for determining x -/
def Strategy := ℕ → ℕ → ℕ

/-- Whether a strategy determines x within n moves -/
def determines (s : Strategy) (x : ℕ) (n : ℕ) : Prop := sorry

theorem min_moves_to_determine (x : ℕ) (h : digit_sum x = 2012) :
  ∀ (s : Strategy), ∃ (n : ℕ), n ≥ 2012 ∧ determines s x n :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_moves_to_determine_l480_48059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_curve_to_line_l480_48075

noncomputable section

-- Define the curve
noncomputable def curve (x : ℝ) : ℝ := Real.log (2 * x - 1)

-- Define the line
def line (x y : ℝ) : ℝ := 2 * x - y + 8

-- Define the distance function from a point to the line
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  |2 * x - y + 8| / Real.sqrt (2^2 + (-1)^2)

theorem shortest_distance_curve_to_line :
  ∃ (x y : ℝ), y = curve x ∧ distance_to_line x y = 2 * Real.sqrt 5 ∧
  ∀ (x' y' : ℝ), y' = curve x' → distance_to_line x' y' ≥ 2 * Real.sqrt 5 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_curve_to_line_l480_48075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_30_deg_l480_48003

-- Define 30 degrees in radians
noncomputable def angle_30_deg : Real := 30 * (Real.pi / 180)

-- State the theorem
theorem sin_30_deg : 
  Real.sin angle_30_deg = 1/2 :=
by
  -- Assume cos 30° = √3/2
  have cos_30_deg : Real.cos angle_30_deg = Real.sqrt 3 / 2 := by sorry
  
  -- Use the Pythagorean identity: sin²θ + cos²θ = 1
  have sin_squared_plus_cos_squared : Real.sin angle_30_deg ^ 2 + Real.cos angle_30_deg ^ 2 = 1 := by sorry
  
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_30_deg_l480_48003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l480_48049

-- Define the function as noncomputable
noncomputable def f (x : ℝ) := Real.log (x^2 - 6*x + 8)

-- Theorem statement
theorem f_monotone_increasing :
  StrictMonoOn f (Set.Ioi 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l480_48049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_perimeter_constant_l480_48034

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- Defines the equation of the ellipse -/
def on_ellipse (e : Ellipse) (p : Point) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: The perimeter of the triangle formed by any point on the ellipse and its foci is constant -/
theorem ellipse_triangle_perimeter_constant (e : Ellipse) (p : Point) (f1 f2 : Point) :
  on_ellipse e p →
  p ≠ ⟨e.a, 0⟩ →
  p ≠ ⟨-e.a, 0⟩ →
  f1 = ⟨Real.sqrt (e.a^2 - e.b^2), 0⟩ →
  f2 = ⟨-Real.sqrt (e.a^2 - e.b^2), 0⟩ →
  distance p f1 + distance p f2 + distance f1 f2 = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_perimeter_constant_l480_48034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_is_real_magnitude_product_l480_48031

/-- Definition of the inner product-like operation (α, β) for complex numbers -/
noncomputable def inner_product_like (α β : ℂ) : ℝ :=
  1/4 * (Complex.abs (α + β)^2 - Complex.abs (α - β)^2)

/-- Definition of the complex inner product ⟨α, β⟩ -/
noncomputable def complex_inner_product (α β : ℂ) : ℂ :=
  inner_product_like α β + Complex.I * inner_product_like α (Complex.I * β)

/-- Theorem stating that the sum of ⟨α, β⟩ and ⟨β, α⟩ is real -/
theorem sum_is_real (α β : ℂ) :
  ∃ (r : ℝ), complex_inner_product α β + complex_inner_product β α = r := by
  sorry

/-- Theorem expressing the magnitude of ⟨α, β⟩ in terms of |α| and |β| -/
theorem magnitude_product (α β : ℂ) :
  Complex.abs (complex_inner_product α β) = Complex.abs α * Complex.abs β := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_is_real_magnitude_product_l480_48031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_valid_t_values_l480_48096

open Real

noncomputable def isIsosceles (a b c : ℝ × ℝ) : Prop :=
  let d1 := (a.1 - b.1)^2 + (a.2 - b.2)^2
  let d2 := (b.1 - c.1)^2 + (b.2 - c.2)^2
  let d3 := (c.1 - a.1)^2 + (c.2 - a.2)^2
  d1 = d2 ∨ d2 = d3 ∨ d3 = d1

noncomputable def isValidT (t : ℝ) : Prop :=
  0 ≤ t ∧ t ≤ 360 ∧
  isIsosceles (cos (30 * π / 180), sin (30 * π / 180))
              (cos (90 * π / 180), sin (90 * π / 180))
              (cos (t * π / 180), sin (t * π / 180))

theorem sum_of_valid_t_values :
  ∃ (S : Finset ℝ), (∀ t ∈ S, isValidT t) ∧ 
    (∀ t₁ t₂, t₁ ∈ S → t₂ ∈ S → t₁ ≠ t₂ → t₁ ≠ 360 - t₂) ∧ 
    (S.sum id = 1200) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_valid_t_values_l480_48096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_one_type_adjacent_l480_48068

def number_of_items : ℕ := 6
def number_of_types : ℕ := 3
def items_per_type : ℕ := 2

def adjacent_arrangements (n : ℕ) (k : ℕ) (m : ℕ) : ℕ :=
  3 * (2 * n * (n - 1) * (k * m)^2 + 2 * k * m * (k * m - 1) * n^2 + k * m * (k * m - 1) * n^2)

theorem probability_one_type_adjacent :
  (adjacent_arrangements number_of_items number_of_types items_per_type : ℚ) / 
  (number_of_items.factorial : ℚ) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_one_type_adjacent_l480_48068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l480_48030

noncomputable section

variable (a m n : ℝ)

def f (x : ℝ) := x * Real.log (a * x)

def f_derivative (x : ℝ) := Real.log (a * x) + 1

theorem part_one (h : ∀ x > 0, f_derivative a x ≤ x/2 + 1/2) :
  0 < a ∧ a ≤ Real.sqrt (Real.exp 1) / 2 :=
sorry

theorem part_two (h₁ : 0 < Real.exp (-1))
  (h₂ : Real.exp (-1) < m) (h₃ : m < 1)
  (h₄ : Real.exp (-1) < n) (h₅ : n < 1)
  (h₆ : m + n < 1) :
  m * n / (m + n)^2 < (m + n)^(n / m) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l480_48030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangle_longest_side_l480_48055

noncomputable section

-- Define the original triangle
def original_triangle : ℝ × ℝ × ℝ := (8, 10, 12)

-- Define the similar triangle's perimeter
def similar_perimeter : ℝ := 150

-- Function to calculate the area of a triangle using Heron's formula
noncomputable def triangle_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

-- Theorem statement
theorem similar_triangle_longest_side :
  ∃ (x : ℝ), 
    let (a, b, c) := original_triangle
    let (a', b', c') := (x * a, x * b, x * c)
    a' + b' + c' = similar_perimeter ∧
    triangle_area a' b' c' = 3 * triangle_area a b c ∧
    max a' (max b' c') = 60 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangle_longest_side_l480_48055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_cardinality_of_specific_sets_l480_48084

theorem union_cardinality_of_specific_sets :
  let A : Finset ℕ := {1, 2}
  let B : Finset ℕ := {2, 3}
  Finset.card (A ∪ B) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_cardinality_of_specific_sets_l480_48084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_harvard_attendance_l480_48010

theorem harvard_attendance (total_applicants : ℕ) (acceptance_rate : ℚ) (attendance_rate : ℚ) 
  (h1 : total_applicants = 20000)
  (h2 : acceptance_rate = 5 / 100)
  (h3 : attendance_rate = 90 / 100) :
  (total_applicants : ℚ) * acceptance_rate * attendance_rate = 900 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_harvard_attendance_l480_48010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_velocity_at_t_3_l480_48026

-- Define the motion law
noncomputable def motionLaw (t : ℝ) : ℝ := 3 * t^2

-- Define the instantaneous velocity function
noncomputable def instantaneousVelocity (t : ℝ) : ℝ := 
  deriv motionLaw t

-- Theorem statement
theorem velocity_at_t_3 : 
  instantaneousVelocity 3 = 18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_velocity_at_t_3_l480_48026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_even_sum_is_one_third_l480_48062

/-- The set of card numbers -/
def card_numbers : Finset ℕ := {1, 2, 3, 4}

/-- The set of all possible pairs of cards -/
def card_pairs : Finset (ℕ × ℕ) :=
  (card_numbers.product card_numbers).filter (fun p => p.1 < p.2)

/-- Predicate for pairs with even sum -/
def even_sum (p : ℕ × ℕ) : Prop := Even (p.1 + p.2)

/-- The probability of drawing two cards with an even sum -/
noncomputable def prob_even_sum : ℚ :=
  (card_pairs.filter (fun p => Even (p.1 + p.2))).card / card_pairs.card

theorem prob_even_sum_is_one_third :
  prob_even_sum = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_even_sum_is_one_third_l480_48062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l480_48041

-- Define the function f(x) as noncomputable due to the use of Real.sqrt
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (2 * x^2 - x - 3)

-- State the theorem
theorem f_monotone_increasing :
  MonotoneOn f (Set.Ici (3/2)) :=
by
  -- The proof is omitted and replaced with sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l480_48041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_loading_time_correct_worker_problem_solution_l480_48038

/-- The time taken for two workers to load one truck together -/
noncomputable def combined_loading_time (t1 t2 : ℝ) : ℝ :=
  1 / (1 / t1 + 1 / t2)

/-- Theorem stating that the combined loading time is correct -/
theorem combined_loading_time_correct (t1 t2 : ℝ) (h1 : t1 > 0) (h2 : t2 > 0) :
  combined_loading_time t1 t2 = 1 / (1 / t1 + 1 / t2) := by
  sorry

/-- Specific case for the given problem -/
theorem worker_problem_solution :
  ∃ (ε : ℝ), ε > 0 ∧ |combined_loading_time 5 4 - 2.22| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_loading_time_correct_worker_problem_solution_l480_48038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_bridge_l480_48046

/-- The time (in seconds) it takes for a train to cross a bridge -/
noncomputable def train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (bridge_length : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  total_distance / train_speed_ms

/-- Theorem stating that a train of length 110 m, traveling at 72 km/h, 
    takes 11.1 seconds to cross a bridge of length 112 m -/
theorem train_crossing_bridge :
  (train_crossing_time 110 72 112 : ℝ) = 11.1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_bridge_l480_48046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_four_dollar_frisbees_l480_48088

/-- Represents the number of frisbees sold at $3 -/
def x : ℕ := sorry

/-- Represents the number of frisbees sold at $4 -/
def y : ℕ := sorry

/-- The total number of frisbees sold -/
def total_frisbees : ℕ := 60

/-- The total receipts from frisbee sales -/
def total_receipts : ℕ := 200

/-- The statement that the total number of frisbees sold is 60 -/
axiom total_frisbees_eq : x + y = total_frisbees

/-- The statement that the total receipts from frisbee sales is $200 -/
axiom total_receipts_eq : 3 * x + 4 * y = total_receipts

/-- The theorem stating that the minimum number of $4 frisbees sold is 20 -/
theorem min_four_dollar_frisbees : y ≥ 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_four_dollar_frisbees_l480_48088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mia_and_ella_on_olivers_team_l480_48087

-- Define the eye colors and hair colors
inductive EyeColor
| Green
| Gray

inductive HairColor
| Red
| Brown

-- Define a player with their characteristics
structure Player where
  name : String
  eye : EyeColor
  hair : HairColor

-- Define a function to check if two players share exactly one characteristic
def shareOneCharacteristic (p1 p2 : Player) : Prop :=
  (p1.eye = p2.eye ∧ p1.hair ≠ p2.hair) ∨ (p1.eye ≠ p2.eye ∧ p1.hair = p2.hair)

-- Define the players
def daniel : Player := ⟨"Daniel", EyeColor.Green, HairColor.Red⟩
def oliver : Player := ⟨"Oliver", EyeColor.Gray, HairColor.Brown⟩
def mia : Player := ⟨"Mia", EyeColor.Gray, HairColor.Red⟩
def ella : Player := ⟨"Ella", EyeColor.Green, HairColor.Brown⟩
def leo : Player := ⟨"Leo", EyeColor.Green, HairColor.Red⟩
def zoe : Player := ⟨"Zoe", EyeColor.Green, HairColor.Brown⟩

-- Define the theorem
theorem mia_and_ella_on_olivers_team :
  shareOneCharacteristic oliver mia ∧
  shareOneCharacteristic oliver ella ∧
  shareOneCharacteristic mia ella ∧
  ¬(shareOneCharacteristic oliver daniel) ∧
  ¬(shareOneCharacteristic oliver leo) ∧
  ¬(shareOneCharacteristic oliver zoe) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mia_and_ella_on_olivers_team_l480_48087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_janet_return_time_l480_48050

/-- Represents the walking pattern of Janet -/
structure WalkingPattern where
  north : ℕ
  west : ℕ
  south : ℕ
  east : ℕ

/-- Calculates the time needed to return home given a walking pattern and speed -/
def timeToReturnHome (pattern : WalkingPattern) (speed : ℕ) : ℕ :=
  let netSouth := Int.ofNat pattern.south - Int.ofNat pattern.north
  let netWest := Int.ofNat pattern.west - Int.ofNat pattern.east
  let blocksToHome := (netSouth.natAbs + netWest.natAbs : ℕ)
  (blocksToHome + speed - 1) / speed

/-- Janet's specific walking pattern -/
def janetsPattern : WalkingPattern :=
  { north := 3
  , west := 3 * 7
  , south := 8
  , east := 8 * 2 }

/-- Theorem stating that Janet will take 5 minutes to return home -/
theorem janet_return_time :
  timeToReturnHome janetsPattern 2 = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_janet_return_time_l480_48050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_minus_sin_value_l480_48069

noncomputable def f (x : ℝ) : ℝ := Real.sin (3 * x + Real.pi / 4)

theorem cos_minus_sin_value (α : ℝ) :
  (Real.pi / 2 < α ∧ α < Real.pi) →  -- α is in the second quadrant
  f (α / 3) = (4 / 5) * Real.cos (α + Real.pi / 4) * Real.cos (2 * α) →
  Real.cos α - Real.sin α = -Real.sqrt 5 / 2 ∨ Real.cos α - Real.sin α = -Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_minus_sin_value_l480_48069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_parabola_or_line_l480_48033

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in a 2D plane, represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The distance from a point to a line -/
noncomputable def distanceToLine (p : Point) (l : Line) : ℝ :=
  abs (l.a * p.x + l.b * p.y + l.c) / Real.sqrt (l.a^2 + l.b^2)

/-- Possible trajectories of the moving point -/
inductive Trajectory
  | Parabola
  | Line

/-- The theorem stating the trajectory of the moving point -/
theorem trajectory_is_parabola_or_line
  (movingPoint : ℝ → Point)  -- The moving point as a function of time
  (fixedPoint : Point)       -- The fixed point
  (fixedLine : Line)         -- The fixed line
  (h : ∀ t, distance (movingPoint t) fixedPoint = distanceToLine (movingPoint t) fixedLine)
  : ∃ traj : Trajectory, traj = Trajectory.Parabola ∨ traj = Trajectory.Line :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_parabola_or_line_l480_48033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_doll_swapping_doll_swapping_other_l480_48005

/-- Represents a permutation of n elements -/
def Permutation (n : ℕ) := Fin n → Fin n

/-- Represents the identity permutation -/
def identityPerm (n : ℕ) : Permutation n := λ i ↦ i

/-- Represents a swap (transposition) between two elements -/
def swap (n : ℕ) (i j : Fin n) : Permutation n := 
  λ k ↦ if k = i then j else if k = j then i else k

/-- Checks if a permutation is even -/
def isEvenPerm (n : ℕ) (p : Permutation n) : Prop := sorry

theorem doll_swapping (n : ℕ) (h : n > 1) :
  (∃ (perms : List (Permutation n)), 
    perms.length = n * (n - 1) / 2 ∧ 
    perms.foldl (· ∘ ·) (identityPerm n) = identityPerm n) ↔ 
  (n % 4 = 0 ∨ n % 4 = 1) :=
sorry

theorem doll_swapping_other (n : ℕ) (h : n > 1) :
  (∃ (perms : List (Permutation n)), 
    perms.length = n * (n - 1) / 2 ∧ 
    ∀ i : Fin n, (perms.foldl (· ∘ ·) (identityPerm n) i) ≠ i) ↔ 
  n ≠ 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_doll_swapping_doll_swapping_other_l480_48005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partnership_profit_share_l480_48094

/-- Given a partnership where A, B, and C invest different amounts, 
    and B's share of the profit is known, calculate A's share. -/
theorem partnership_profit_share 
  (investment_A investment_B investment_C : ℕ)
  (share_B share_A : ℕ) : 
  investment_A = 7000 →
  investment_B = 11000 →
  investment_C = 18000 →
  share_B = 880 →
  share_A = 560 → True := by
  intro hA hB hC hShareB hShareA
  sorry

#check partnership_profit_share

end NUMINAMATH_CALUDE_ERRORFEEDBACK_partnership_profit_share_l480_48094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_universal_positive_square_l480_48092

theorem negation_of_universal_positive_square :
  (¬ ∀ x : ℝ, x^2 > 0) ↔ (∃ x : ℝ, x^2 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_universal_positive_square_l480_48092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_properties_l480_48085

/-- Given a natural number n, represents the binomial expansion of (1/2 + 2x)^n -/
def binomial_expansion (n : ℕ) (x : ℚ) := (1/2 + 2*x)^n

/-- Returns true if the binomial coefficients of the 5th, 6th, and 7th terms form an arithmetic sequence -/
def arithmetic_sequence (n : ℕ) : Prop :=
  Nat.choose n 4 + Nat.choose n 6 = 2 * Nat.choose n 5

/-- Returns the coefficient of the term with the maximum binomial coefficient -/
noncomputable def max_coefficient (n : ℕ) : ℚ :=
  sorry

/-- Returns the sum of the binomial coefficients of the first three terms -/
def sum_first_three (n : ℕ) : ℕ :=
  Nat.choose n 0 + Nat.choose n 1 + Nat.choose n 2

/-- Returns the term with the maximum coefficient -/
noncomputable def max_term (n : ℕ) : ℚ × ℕ :=
  sorry

theorem binomial_expansion_properties (n : ℕ) :
  (arithmetic_sequence n → max_coefficient n = 3432) ∧
  (sum_first_three n = 79 → max_term n = (16896, 10)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_properties_l480_48085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_angle_sin_l480_48078

/-- A right triangular prism with equal lateral and base edge lengths -/
structure RightTriangularPrism where
  edge_length : ℝ
  edge_length_pos : edge_length > 0

/-- The angle between a diagonal line and a lateral face in the prism -/
noncomputable def diagonal_angle (prism : RightTriangularPrism) : ℝ :=
  Real.arcsin (1 / Real.sqrt 5)

/-- Theorem stating that the sine of the diagonal angle is 1/√5 -/
theorem diagonal_angle_sin (prism : RightTriangularPrism) :
    Real.sin (diagonal_angle prism) = 1 / Real.sqrt 5 := by
  sorry

#check diagonal_angle_sin

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_angle_sin_l480_48078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_x₀_values_l480_48020

/-- The function f(x) = x^3 + 1 -/
def f (x : ℝ) : ℝ := x^3 + 1

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3 * x^2

/-- Point P through which the tangent line passes -/
def P : ℝ × ℝ := (-2, 1)

/-- Point Q where the line is tangent to the graph of f -/
noncomputable def Q (x₀ : ℝ) : ℝ × ℝ := (x₀, f x₀)

/-- The slope of the line through P and Q -/
noncomputable def m (x₀ : ℝ) : ℝ := (f x₀ - P.2) / (x₀ - P.1)

theorem tangent_line_x₀_values :
  ∀ x₀ : ℝ, (m x₀ = f' x₀) → (x₀ = 0 ∨ x₀ = -3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_x₀_values_l480_48020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_iff_congruent_dihedrals_exists_non_regular_with_five_congruent_l480_48064

/-- Represents a tetrahedron -/
structure Tetrahedron where
  vertices : Fin 4 → ℝ × ℝ × ℝ

/-- Calculates the dihedral angle between two faces of a tetrahedron -/
noncomputable def dihedral_angle (t : Tetrahedron) (face1 face2 : Fin 4) : ℝ :=
  sorry

/-- Checks if a tetrahedron is regular -/
def is_regular (t : Tetrahedron) : Prop :=
  sorry

/-- Theorem: A tetrahedron is regular if and only if all six of its dihedral angles are congruent -/
theorem regular_iff_congruent_dihedrals (t : Tetrahedron) :
  is_regular t ↔ ∀ (f1 f2 : Fin 4), f1 ≠ f2 → ∀ (g1 g2 : Fin 4), g1 ≠ g2 → 
    dihedral_angle t f1 f2 = dihedral_angle t g1 g2 :=
  sorry

/-- Theorem: There exists a non-regular tetrahedron with exactly five congruent dihedral angles -/
theorem exists_non_regular_with_five_congruent :
  ∃ (t : Tetrahedron), ¬is_regular t ∧ 
    ∃ (a : ℝ), ∃ (f1 f2 f3 f4 f5 : Fin 4 × Fin 4),
      f1.1 ≠ f1.2 ∧ f2.1 ≠ f2.2 ∧ f3.1 ≠ f3.2 ∧ f4.1 ≠ f4.2 ∧ f5.1 ≠ f5.2 ∧
      f1 ≠ f2 ∧ f1 ≠ f3 ∧ f1 ≠ f4 ∧ f1 ≠ f5 ∧ f2 ≠ f3 ∧ f2 ≠ f4 ∧ f2 ≠ f5 ∧ f3 ≠ f4 ∧ f3 ≠ f5 ∧ f4 ≠ f5 ∧
      dihedral_angle t f1.1 f1.2 = a ∧
      dihedral_angle t f2.1 f2.2 = a ∧
      dihedral_angle t f3.1 f3.2 = a ∧
      dihedral_angle t f4.1 f4.2 = a ∧
      dihedral_angle t f5.1 f5.2 = a :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_iff_congruent_dihedrals_exists_non_regular_with_five_congruent_l480_48064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertices_distance_l480_48065

/-- The equation of the parabolas -/
noncomputable def parabola_equation (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + y^2) + |y - 2| = 5

/-- The vertices of the parabolas -/
def vertices : Set (ℝ × ℝ) :=
  {(0, 3.5), (0, -1.5)}

/-- The distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem parabola_vertices_distance :
  ∀ (p q : ℝ × ℝ), p ∈ vertices → q ∈ vertices → p ≠ q → distance p q = 5 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertices_distance_l480_48065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2012_is_identity_l480_48022

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (1 + x) / (1 - x)

-- Define the sequence of functions fₖ
noncomputable def f_k : ℕ → (ℝ → ℝ)
| 0 => id
| (n+1) => f ∘ (f_k n)

-- Theorem statement
theorem f_2012_is_identity : f_k 2012 = id := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2012_is_identity_l480_48022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_and_reciprocal_product_l480_48071

variable (a b : ℝ)
variable (x₁ x₂ x₃ : ℝ)

-- Define the cubic equation
def cubic_equation (a b : ℝ) (x : ℝ) : Prop :=
  x^3 + a * x^2 + b * x + 1 = 0

-- Define the roots of the equation
def are_roots (a b : ℝ) (x₁ x₂ x₃ : ℝ) : Prop :=
  cubic_equation a b x₁ ∧ cubic_equation a b x₂ ∧ cubic_equation a b x₃

-- Vieta's formulas
def vieta_sum (a : ℝ) (x₁ x₂ x₃ : ℝ) : Prop :=
  x₁ + x₂ + x₃ = -a

def vieta_product (x₁ x₂ x₃ : ℝ) : Prop :=
  x₁ * x₂ * x₃ = -1

-- Theorem to prove
theorem sum_and_reciprocal_product (a b : ℝ) (x₁ x₂ x₃ : ℝ) 
  (h1 : are_roots a b x₁ x₂ x₃) (h2 : vieta_sum a x₁ x₂ x₃) (h3 : vieta_product x₁ x₂ x₃) :
  (x₁ + x₂ + x₃) * (1/x₁ + 1/x₂ + 1/x₃) = a * b :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_and_reciprocal_product_l480_48071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_trips_l480_48045

/-- Represents a trip organized by the school -/
structure Trip where
  students : Finset Nat
  type : Bool  -- True for museum trip, False for nature trip

/-- The set of all students in the school -/
def AllStudents : Finset Nat := Finset.range 2022

/-- The set of all trips organized -/
axiom Trips : Finset Trip

/-- No student participates in the same type of trip twice -/
axiom no_repeat_trips : ∀ s : Nat, s ∈ AllStudents →
  (Trips.filter (λ t => t.type = true)).card ≤ 1 ∧
  (Trips.filter (λ t => t.type = false)).card ≤ 1

/-- The number of students attending each trip is different -/
axiom different_student_count : ∀ t1 t2 : Trip, t1 ∈ Trips → t2 ∈ Trips → t1 ≠ t2 →
  t1.students.card ≠ t2.students.card

/-- No two students participate in the same two trips together -/
axiom no_shared_trips : ∀ s1 s2 : Nat, s1 ∈ AllStudents → s2 ∈ AllStudents → s1 ≠ s2 →
  (Trips.filter (λ t => s1 ∈ t.students ∧ s2 ∈ t.students)).card ≤ 1

/-- The maximum number of trips that can be organized is 77 -/
theorem max_trips : Trips.card = 77 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_trips_l480_48045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vector_scalar_l480_48090

/-- Given vectors a, b, c in ℝ², and λ ∈ ℝ such that (a + λ*b) ⊥ c, prove λ = 1/2 -/
theorem perpendicular_vector_scalar (a b c : ℝ × ℝ) (l : ℝ) 
  (ha : a = (1, 2)) 
  (hb : b = (1, 0)) 
  (hc : c = (4, -3)) 
  (h_perp : (a.1 + l * b.1, a.2 + l * b.2) • c = 0) : 
  l = 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vector_scalar_l480_48090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_minimum_value_l480_48006

/-- A geometric sequence where the first term is equal to the common ratio -/
def GeometricSequence (a : ℕ+ → ℝ) : Prop :=
  ∃ q : ℝ, a 1 = q ∧ ∀ n : ℕ+, a n = q ^ (n : ℝ)

theorem geometric_sequence_minimum_value
  (a : ℕ+ → ℝ) (m n : ℕ+) (h_seq : GeometricSequence a) (h_eq : a m * (a n)^2 = (a 4)^2) :
  2 / (m : ℝ) + 1 / (n : ℝ) ≥ 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_minimum_value_l480_48006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l480_48048

/-- The inclination angle of a line ax + y + 5 = 0, where a > 0 -/
noncomputable def inclination_angle (a : ℝ) (h : a > 0) : ℝ :=
  Real.pi - Real.arctan (-a)

/-- Theorem: The inclination angle of the line ax + y + 5 = 0 (a > 0) is π - arctan(-a) -/
theorem line_inclination_angle (a : ℝ) (h : a > 0) :
  inclination_angle a h = Real.pi - Real.arctan (-a) := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l480_48048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_four_ones_value_probability_four_ones_rounded_value_l480_48058

/-- The probability of rolling exactly 4 ones with 12 six-sided dice -/
def probability_four_ones : ℚ :=
  (495 * 5^8 : ℕ) / 6^12

/-- Theorem stating that the probability of rolling exactly 4 ones
    with 12 six-sided dice is equal to (495 * 5^8) / 6^12 -/
theorem probability_four_ones_value :
  probability_four_ones = (495 * 5^8 : ℕ) / 6^12 := by
  rfl

/-- The probability rounded to three decimal places -/
noncomputable def probability_four_ones_rounded : ℚ :=
  (probability_four_ones * 1000).floor / 1000

/-- Theorem stating that the probability rounded to three decimal places is 0.089 -/
theorem probability_four_ones_rounded_value :
  probability_four_ones_rounded = 89 / 1000 := by
  sorry

#eval (probability_four_ones * 1000).floor / 1000

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_four_ones_value_probability_four_ones_rounded_value_l480_48058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_product_l480_48037

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (-1 + (Real.sqrt 3 / 2) * t, 2 + (1 / 2) * t)

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 = 9

-- Define the point P
def point_P : ℝ × ℝ := (-1, 2)

-- State the theorem
theorem intersection_product :
  ∃ (t₁ t₂ : ℝ),
    circle_C (line_l t₁).1 (line_l t₁).2 ∧
    circle_C (line_l t₂).1 (line_l t₂).2 ∧
    t₁ ≠ t₂ ∧
    |t₁ * t₂| = 4 := by
  sorry

#check intersection_product

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_product_l480_48037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marathon_injury_point_l480_48077

/-- Represents the point of injury during a run --/
noncomputable def injury_point (total_distance : ℝ) (second_half_time : ℝ) (time_difference : ℝ) : ℝ :=
  let first_half_time := second_half_time - time_difference
  let initial_speed := total_distance / (first_half_time + 2 * second_half_time)
  first_half_time * initial_speed

/-- Theorem stating the point of injury for the specific problem --/
theorem marathon_injury_point :
  injury_point 40 10 5 = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marathon_injury_point_l480_48077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modified_mixture_yellow_tint_percentage_l480_48072

/-- Represents a paint mixture -/
structure PaintMixture where
  total_volume : ℚ
  yellow_tint_percentage : ℚ

/-- Calculates the new yellow tint percentage after modifying the mixture -/
def new_yellow_tint_percentage (mixture : PaintMixture) (added_yellow : ℚ) (removed_other : ℚ) : ℚ :=
  let initial_yellow := mixture.total_volume * (mixture.yellow_tint_percentage / 100)
  let new_yellow := initial_yellow + added_yellow
  let new_total := mixture.total_volume + added_yellow - removed_other
  (new_yellow / new_total) * 100

/-- The main theorem stating the result of modifying the paint mixture -/
theorem modified_mixture_yellow_tint_percentage :
  let initial_mixture := PaintMixture.mk 40 35
  let result := new_yellow_tint_percentage initial_mixture 7 3
  ∀ ε > 0, |result - 47723/1000| < ε := by
  sorry

#eval new_yellow_tint_percentage (PaintMixture.mk 40 35) 7 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modified_mixture_yellow_tint_percentage_l480_48072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_equivalence_l480_48009

-- Define a function f
def f : ℝ → ℝ := sorry

-- Define the domain of f(x+1)
def domain_f_plus_one : Set ℝ := Set.Ici 0 ∩ Set.Iio 3

-- Define the domain of f(2^x)
def domain_f_exp : Set ℝ := Set.Ici 0 ∩ Set.Iio 2

-- Theorem statement
theorem domain_equivalence :
  (∀ x, x ∈ domain_f_plus_one ↔ f (x + 1) ∈ Set.univ) →
  (∀ x, x ∈ domain_f_exp ↔ f (2^x) ∈ Set.univ) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_equivalence_l480_48009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_f_f_inequality_when_a_eq_one_l480_48008

-- Define the function f(x) = (ax + 1)ln(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a * x + 1) * Real.log x

-- Define the derivative of f(x)
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + (a * x + 1) / x

-- Theorem for monotonicity of f'(x) when a ≤ 0
theorem f'_decreasing_when_a_nonpositive (a : ℝ) (h : a ≤ 0) :
  ∀ x > 0, StrictMonoOn (fun x => -f' a x) (Set.Ioi 0) :=
sorry

-- Theorem for monotonicity of f'(x) when a > 0 and x < 1/a
theorem f'_decreasing_when_a_positive_x_small (a : ℝ) (h : a > 0) :
  ∀ x > 0, x < 1/a → StrictMonoOn (fun x => -f' a x) (Set.Ioo 0 (1/a)) :=
sorry

-- Theorem for monotonicity of f'(x) when a > 0 and x > 1/a
theorem f'_increasing_when_a_positive_x_large (a : ℝ) (h : a > 0) :
  ∀ x > 1/a, StrictMonoOn (fun x => f' a x) (Set.Ioi (1/a)) :=
sorry

-- Theorem for the inequality when a = 1
theorem inequality_when_a_eq_one (x : ℝ) (h : x > 0) :
  x * (Real.exp x + 1) > f 1 x + 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_f_f_inequality_when_a_eq_one_l480_48008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hundred_days_after_thursday_is_saturday_l480_48015

/-- Represents days of the week -/
inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday
deriving Repr, BEq, Inhabited

/-- Successor function for DayOfWeek -/
def DayOfWeek.succ (d : DayOfWeek) : DayOfWeek :=
  match d with
  | .Sunday => .Monday
  | .Monday => .Tuesday
  | .Tuesday => .Wednesday
  | .Wednesday => .Thursday
  | .Thursday => .Friday
  | .Friday => .Saturday
  | .Saturday => .Sunday

/-- Calculates the day of the week after a given number of days -/
def dayAfter (start : DayOfWeek) (days : ℕ) : DayOfWeek :=
  match (days % 7 : ℕ) with
  | 0 => start
  | 1 => start.succ
  | 2 => start.succ.succ
  | 3 => start.succ.succ.succ
  | 4 => start.succ.succ.succ.succ
  | 5 => start.succ.succ.succ.succ.succ
  | 6 => start.succ.succ.succ.succ.succ.succ
  | _ => start  -- This case should never occur due to the modulo operation

theorem hundred_days_after_thursday_is_saturday :
  dayAfter DayOfWeek.Thursday 100 = DayOfWeek.Saturday := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hundred_days_after_thursday_is_saturday_l480_48015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circles_a_value_l480_48056

/-- Circle in polar coordinates --/
structure PolarCircle where
  equation : ℝ → ℝ → Prop

/-- Circle in parametric form --/
structure ParametricCircle where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- Define circle C₁ passing through O(0,0), A(2, π/2), and B(2√2, π/4) --/
noncomputable def C₁ : PolarCircle :=
  { equation := λ ρ θ => ρ = 2 * Real.sqrt 2 * Real.sin (θ + Real.pi / 4) }

/-- Define circle C₂ with parametric equation x = -1 + a cos θ, y = -1 + a sin θ --/
noncomputable def C₂ (a : ℝ) : ParametricCircle :=
  { x := λ θ => -1 + a * Real.cos θ,
    y := λ θ => -1 + a * Real.sin θ }

/-- Two circles are externally tangent --/
def externally_tangent (c1 : PolarCircle) (c2 : ParametricCircle) : Prop :=
  sorry

theorem tangent_circles_a_value :
  ∀ a : ℝ, externally_tangent C₁ (C₂ a) → a = Real.sqrt 2 ∨ a = -Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circles_a_value_l480_48056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_range_l480_48099

-- Define the circle C
noncomputable def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 2*y + 1 = 0

-- Define the line l
def line_l (x y k : ℝ) : Prop := 3*x - 4*y + k = 0

-- Define the distance function from a point to a line
noncomputable def distance_to_line (x y k : ℝ) : ℝ := 
  abs (3*x - 4*y + k) / Real.sqrt (3^2 + 4^2)

-- Main theorem
theorem circle_line_intersection_range (k : ℝ) : 
  (∃ (x1 y1 x2 y2 : ℝ), 
    circle_C x1 y1 ∧ circle_C x2 y2 ∧ 
    distance_to_line x1 y1 k = 1 ∧ 
    distance_to_line x2 y2 k = 1) →
  k ∈ Set.Ioo (-17) (-7) ∪ Set.Ioo 3 13 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_range_l480_48099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_implies_a_values_l480_48051

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then x^2 - 2*a*x + 2
  else x + 9/x - 3*a

theorem min_value_implies_a_values (a : ℝ) :
  (∀ x : ℝ, f a x ≥ f a 1) → (a = 1 ∨ a = 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_implies_a_values_l480_48051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_equation_l480_48095

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

theorem floor_equation : 
  floor 6.5 * floor (2/3 : ℝ) + floor 2 * (7.2 : ℝ) + floor 8.4 - (6.6 : ℝ) = 15.8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_equation_l480_48095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilibrium_calculations_l480_48011

-- Define the given conditions
def container_volume : ℝ := 2
def initial_N2 : ℝ := 2
def initial_H2 : ℝ := 6
def reaction_time : ℝ := 10
def rate_N2 : ℝ := 0.05

-- Define the reaction stoichiometry
def N2_coefficient : ℝ := 1
def H2_coefficient : ℝ := 3
def NH3_coefficient : ℝ := 2

-- Theorem to prove the required results
theorem equilibrium_calculations :
  let change_N2 : ℝ := rate_N2 * reaction_time
  let change_H2 : ℝ := change_N2 * H2_coefficient / N2_coefficient
  let change_NH3 : ℝ := change_N2 * NH3_coefficient / N2_coefficient
  let final_N2 : ℝ := initial_N2 / container_volume - change_N2
  let final_H2 : ℝ := initial_H2 / container_volume - change_H2
  let final_NH3 : ℝ := change_NH3
  ∃ (rate_NH3 : ℝ) (conc_H2_eq : ℝ) (conv_rate_H2 : ℝ) (total_amount : ℝ),
  rate_NH3 = 0.1 ∧
  conc_H2_eq = 1.5 ∧
  conv_rate_H2 = 50 ∧
  total_amount = 6 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilibrium_calculations_l480_48011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mode_equals_average_l480_48091

noncomputable def data_set (x : ℚ) : Finset ℚ := {1, 2, 3, 4, 5, x}

def is_mode (s : Finset ℚ) (m : ℚ) : Prop :=
  m ∈ s ∧ ∀ y ∈ s, (s.filter (· = m)).card ≥ (s.filter (· = y)).card

def has_unique_mode (s : Finset ℚ) : Prop :=
  ∃! m, is_mode s m

noncomputable def average (s : Finset ℚ) : ℚ :=
  s.sum id / s.card

theorem mode_equals_average (x : ℚ) :
  has_unique_mode (data_set x) →
  (∃ m, is_mode (data_set x) m ∧ average (data_set x) = m) →
  x = 3 := by
  sorry

#check mode_equals_average

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mode_equals_average_l480_48091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_solutions_for_equation_l480_48081

theorem two_solutions_for_equation : 
  ∃! (a b : ℝ), a > 1 ∧ b > 1 ∧ a ≠ b ∧ 
   (4 : ℝ)^(a^2 - 5*a + 6) + 3 = 4 ∧ 
   (4 : ℝ)^(b^2 - 5*b + 6) + 3 = 4 ∧
   (∀ y : ℝ, y > 1 ∧ (4 : ℝ)^(y^2 - 5*y + 6) + 3 = 4 → y = a ∨ y = b) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_solutions_for_equation_l480_48081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_values_l480_48014

def point_on_terminal_side (α : Real) (x y : Real) (m : Real) : Prop :=
  x = -Real.sqrt 3 ∧ y = m

theorem angle_values (α : Real) (m : Real) 
  (h1 : point_on_terminal_side α (-Real.sqrt 3) m m) 
  (h2 : Real.sin α = (Real.sqrt 2 * m) / 4) :
  (Real.cos α = -1 ∨ Real.cos α = -Real.sqrt 6 / 4) ∧
  (Real.sin α = 0 ∨ Real.sin α = Real.sqrt 10 / 4 ∨ Real.sin α = -Real.sqrt 10 / 4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_values_l480_48014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_five_l480_48044

-- Define the function for the closest integer to square root
noncomputable def closestIntToSqrt (n : ℕ+) : ℤ :=
  ⌊(n : ℝ).sqrt + 0.5⌋

-- Define the series term
noncomputable def seriesTerm (n : ℕ+) : ℝ :=
  (3 : ℝ) ^ (closestIntToSqrt n) + (3 : ℝ) ^ (-(closestIntToSqrt n)) / (3 : ℝ) ^ (n : ℕ)

-- State the theorem
theorem series_sum_equals_five :
  (∑' (n : ℕ+), seriesTerm n) = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_five_l480_48044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_in_set_l480_48089

/-- Given a set of 64 consecutive multiples of 4 with the greatest number being 320,
    the smallest number in the set is 68. -/
theorem smallest_number_in_set (s : Finset ℕ) 
  (h1 : ∀ n ∈ s, ∃ k : ℕ, n = 4 * k)
  (h2 : ∃ n ∈ s, ∀ m ∈ s, m ≤ n)
  (h3 : ∃ n ∈ s, ∀ m ∈ s, n ≤ m)
  (h4 : s.card = 64)
  (h5 : 320 ∈ s) :
  ∃ n ∈ s, n = 68 ∧ ∀ m ∈ s, n ≤ m :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_in_set_l480_48089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_deriv_at_one_l480_48053

variable (f g : ℝ → ℝ)

/-- The function g is defined as g(x) = x^2 * f(x) -/
def g_def (f : ℝ → ℝ) : ℝ → ℝ := fun x => x^2 * f x

/-- The value of f at x = 1 is 2 -/
axiom f_at_one : f 1 = 2

/-- The derivative of f at x = 1 is 2 -/
axiom f_deriv_at_one : deriv f 1 = 2

/-- Theorem: Given the definitions above, the derivative of g at x = 1 is 6 -/
theorem g_deriv_at_one (f : ℝ → ℝ) : deriv (g_def f) 1 = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_deriv_at_one_l480_48053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_symmetry_exponential_function_symmetry_l480_48086

-- Definition of local symmetry point
def has_local_symmetry_point (f : ℝ → ℝ) : Prop :=
  ∃ x₀ : ℝ, f (-x₀) = -f x₀

-- Part 1
theorem cubic_function_symmetry (a b c : ℝ) :
  has_local_symmetry_point (λ x ↦ a * x^3 + b * x^2 + c * x - b) :=
sorry

-- Part 2
theorem exponential_function_symmetry :
  ∃ m : ℝ, m ∈ Set.Icc (-325/32) (-27/8) ∧
    has_local_symmetry_point (λ x ↦ 4^x + 2^x + m) ∧
    Set.range (λ x ↦ 4^x + 2^x + m) ⊆ Set.Icc (-1) 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_symmetry_exponential_function_symmetry_l480_48086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_inequality_l480_48018

/-- Predicate stating that a, b, and c form a triangle -/
def Triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

/-- Predicate stating that m is a median to side c in a triangle with sides a, b, and c -/
def IsMedian (m c a b : ℝ) : Prop :=
  Triangle a b c ∧ m^2 = (2*a^2 + 2*b^2 - c^2) / 4

/-- Given a triangle with sides a, b, and c, and m as the median to side c,
    prove that m > (a + b - c) / 2 -/
theorem median_inequality (a b c m : ℝ) (h_triangle : Triangle a b c) (h_median : IsMedian m c a b) :
  m > (a + b - c) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_inequality_l480_48018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_crosses_asymptote_at_seven_fourths_l480_48047

/-- The function f(x) = (3x^2 - 7x - 8) / (x^2 - 5x + 2) -/
noncomputable def f (x : ℝ) : ℝ := (3*x^2 - 7*x - 8) / (x^2 - 5*x + 2)

/-- The horizontal asymptote of f(x) -/
def horizontal_asymptote : ℝ := 3

theorem f_crosses_asymptote_at_seven_fourths :
  ∃ (x : ℝ), x = 7/4 ∧ f x = horizontal_asymptote := by
  use 7/4
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_crosses_asymptote_at_seven_fourths_l480_48047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_coordinates_l480_48070

/-- The focus of a parabola with equation y = 4ax^2 (a ≠ 0) has coordinates (0, 1/(16a)) -/
theorem parabola_focus_coordinates (a : ℝ) (h : a ≠ 0) :
  let parabola := {p : ℝ × ℝ | p.2 = 4 * a * p.1^2}
  ∃ (f : ℝ × ℝ), f = (0, 1 / (16 * a)) ∧ f ∈ parabola :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_coordinates_l480_48070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_of_P_l480_48098

def P : ℕ := 35^5 + 5 * 35^4 + 10 * 35^3 + 10 * 35^2 + 5 * 35 + 1

theorem number_of_factors_of_P : 
  (Finset.filter (fun x : ℕ => x > 0 ∧ P % x = 0) (Finset.range (P + 1))).card = 121 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_of_P_l480_48098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_on_line_l480_48060

noncomputable section

/-- The line equation y = (1/2)x + 3 -/
def line_equation (x : ℝ) : ℝ := (1/2) * x + 3

/-- The reference point (2, -1) -/
def reference_point : ℝ × ℝ := (2, -1)

/-- The proposed closest point (0, 3) -/
def closest_point : ℝ × ℝ := (0, 3)

/-- Theorem stating that the closest_point is on the line and is the closest to the reference_point -/
theorem closest_point_on_line : 
  (closest_point.2 = line_equation closest_point.1) ∧ 
  (∀ x : ℝ, (x - reference_point.1)^2 + (line_equation x - reference_point.2)^2 ≥ 
            (closest_point.1 - reference_point.1)^2 + (closest_point.2 - reference_point.2)^2) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_on_line_l480_48060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fermat_point_sum_l480_48043

-- Define the points
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (8, 0)
def C : ℝ × ℝ := (4, 7)
def P : ℝ × ℝ := (3, 3)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem fermat_point_sum :
  ∃ (x y a b : ℕ), 
    distance P A + distance P B + distance P C = x * Real.sqrt a + y * Real.sqrt b ∧
    x + y = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fermat_point_sum_l480_48043

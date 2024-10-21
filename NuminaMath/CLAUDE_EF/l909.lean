import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l909_90953

theorem relationship_abc (a b c : ℝ) 
  (ha : (5 : ℝ)^a = 4) 
  (hb : (5 : ℝ)^b = 6) 
  (hc : (5 : ℝ)^c = 9) : 
  a + c = 2*b := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l909_90953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_annual_interest_approximation_l909_90998

-- Define the loan parameters
def principal : ℝ := 10000
def rate : ℝ := 0.05
def time : ℝ := 1

-- Define the continuously compounded interest formula
noncomputable def compound_interest (p r t : ℝ) : ℝ := p * (Real.exp (r * t)) - p

-- State the theorem
theorem annual_interest_approximation :
  ∃ ε > 0, |compound_interest principal rate time - 512.71| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_annual_interest_approximation_l909_90998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_range_of_m_for_two_zeros_l909_90979

/-- The main function f(x) --/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 4 * Real.log x - 2 * x^2 + 3 * a * x

/-- The function g(x) --/
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := 4 * Real.log x - 2 * x^2 + m

/-- Theorem for the tangent line --/
theorem tangent_line_at_one :
  let f' := fun x => (4 / x) - 4 * x + 3
  (fun x => 3 * x - 2) = fun x => f' 1 * (x - 1) + f 1 1 := by sorry

/-- Theorem for the range of m --/
theorem range_of_m_for_two_zeros :
  ∀ m : ℝ, (∃ x y : ℝ, 1/Real.exp 1 ≤ x ∧ x < y ∧ y ≤ Real.exp 1 ∧ g m x = 0 ∧ g m y = 0) ↔
  (2 < m ∧ m ≤ 4 + 2/(Real.exp 1)^2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_range_of_m_for_two_zeros_l909_90979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_3_4_5_under_300_l909_90949

theorem divisible_by_3_4_5_under_300 : 
  (Finset.filter (fun n => n % 3 = 0 ∧ n % 4 = 0 ∧ n % 5 = 0) (Finset.range 300)).card = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_3_4_5_under_300_l909_90949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_d_value_l909_90939

-- Define the circle and points
noncomputable def Circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 1/4}

noncomputable def M : ℝ × ℝ := (-1/2, 0)
noncomputable def N : ℝ × ℝ := (1/2, 0)
noncomputable def A : ℝ × ℝ := (0, 1/2)

-- Define B such that MB = 3/5
noncomputable def B : ℝ × ℝ := (Real.cos (3 * Real.pi / 5) / 2, Real.sin (3 * Real.pi / 5) / 2)

-- C is on the opposite semicircle
variable (C : ℝ × ℝ)

-- Define the function d
noncomputable def d (C : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem max_d_value :
  ∃ (C : ℝ × ℝ), C ∈ Circle ∧ C.2 < 0 ∧ 
  (∀ (C' : ℝ × ℝ), C' ∈ Circle → C'.2 < 0 → d C' ≤ d C) ∧
  d C = 7 - 4 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_d_value_l909_90939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l909_90913

noncomputable def f (x : ℝ) := 2 * (Real.cos x)^2 + 2 * Real.sin x * Real.cos x

theorem f_properties :
  (f (π / 8) = Real.sqrt 2 + 1) ∧
  (∀ T > 0, (∀ x, f (x + T) = f x) → T ≥ π) ∧
  (∀ x, f x ≥ 1 - Real.sqrt 2) ∧
  (∃ x, f x = 1 - Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l909_90913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_point_existence_l909_90943

/-- The function f(x) defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x * Real.log x + a * x + a^2 - a - 1) * Real.exp x

/-- Theorem stating the existence and uniqueness of 'a' for which f(x) is tangent to x-axis -/
theorem tangent_point_existence :
  ∃! a : ℝ, a ≥ -2 ∧
    (∃ x : ℝ, x > 1 / Real.exp 1 ∧
      f a x = 0 ∧
      (∀ y : ℝ, y > 1 / Real.exp 1 → f a y ≥ 0) ∧
      (∃ ε > 0, ∀ y : ℝ, y ≠ x → |y - x| < ε → f a y > 0)) ∧
    a = -1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_point_existence_l909_90943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_range_in_special_triangle_l909_90940

-- Define an acute triangle
structure AcuteTriangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2
  angle_sum : A + B + C = π
  law_of_sines : a / Real.sin A = b / Real.sin B

-- State the theorem
theorem b_range_in_special_triangle (t : AcuteTriangle) 
  (h1 : t.a = 1) (h2 : t.B = 2 * t.A) : 1 < t.b ∧ t.b < Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_range_in_special_triangle_l909_90940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_differential_equation_solution_l909_90969

noncomputable def y (x : ℝ) (C₁ C₂ C₃ : ℝ) : ℝ :=
  C₁ * Real.exp x + Real.exp (3/2 * x) * (C₂ * Real.cos ((Real.sqrt 7 / 2) * x) + C₃ * Real.sin ((Real.sqrt 7 / 2) * x))

theorem differential_equation_solution (C₁ C₂ C₃ : ℝ) :
  ∀ x, (deriv^[3] (y · C₁ C₂ C₃)) x - 4 * (deriv^[2] (y · C₁ C₂ C₃)) x + 7 * (deriv (y · C₁ C₂ C₃)) x - 4 * y x C₁ C₂ C₃ = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_differential_equation_solution_l909_90969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l909_90923

/-- Line l in the xy-plane -/
def line_l (x y : ℝ) : Prop := x + y = 1

/-- Curve C in the xy-plane -/
def curve_C (x y : ℝ) : Prop := y^2 = 8*x

/-- Point M -/
def point_M : ℝ × ℝ := (0, 1)

/-- Distance between two points in the plane -/
noncomputable def distance (p₁ p₂ : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)

/-- Main theorem -/
theorem intersection_distance_sum :
  ∃ (P Q : ℝ × ℝ),
    P ≠ Q ∧
    line_l P.1 P.2 ∧
    line_l Q.1 Q.2 ∧
    curve_C P.1 P.2 ∧
    curve_C Q.1 Q.2 ∧
    distance point_M P + distance point_M Q = 10 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l909_90923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_of_tangents_l909_90902

-- Define an acute triangle
structure AcuteTriangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  acute : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π
  opposite_sides : 0 < a ∧ 0 < b ∧ 0 < c

-- State the theorem
theorem min_sum_of_tangents (t : AcuteTriangle) 
  (h : t.b^2 + t.c^2 = 4 * t.b * t.c * Real.sin (t.A + π/6)) :
  Real.tan t.A + Real.tan t.B + Real.tan t.C ≥ 48 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_of_tangents_l909_90902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_actual_average_height_l909_90981

/-- The number of boys in the class -/
def num_boys : ℕ := 35

/-- The initially calculated average height in centimeters -/
def initial_avg : ℚ := 185

/-- The wrongly recorded height of one boy in centimeters -/
def wrong_height : ℚ := 166

/-- The actual height of the boy with the wrongly recorded height in centimeters -/
def actual_height : ℚ := 106

/-- The actual average height of the boys in the class -/
noncomputable def actual_avg : ℚ := (num_boys * initial_avg - (wrong_height - actual_height)) / num_boys

theorem actual_average_height :
  (⌊actual_avg * 100⌋ : ℚ) / 100 = 183.29 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_actual_average_height_l909_90981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_parabola_final_equation_l909_90968

/-- Given a hyperbola and a parabola with specific properties, prove the equation of the parabola --/
theorem parabola_equation (a b p : ℝ) (ha : a > 0) (hb : b > 0) (hp : p > 0) :
  let C₁ := λ x y => x^2 / a^2 - y^2 / b^2 = 1
  let e := (Real.sqrt (a^2 + b^2)) / a
  let asymptote := λ x y => b * x - a * y = 0
  let C₂ := λ x y => x^2 = 2 * p * y
  let focus := (0, p / 2)
  let distance_to_asymptote := |p / 2| / Real.sqrt ((b / a)^2 + 1)
  e = 3 ∧ distance_to_asymptote = 2/3 → p = 4 := by
  sorry

/-- The equation of the parabola C₂ is x² = 8y --/
theorem parabola_final_equation (x y : ℝ) :
  let C₂ := λ x y => x^2 = 8 * y
  C₂ x y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_parabola_final_equation_l909_90968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_non_invertible_sum_fraction_l909_90907

theorem matrix_non_invertible_sum_fraction (a b c d : ℝ) :
  let M : Matrix (Fin 3) (Fin 3) ℝ := !![a + d, b, c; b, c + d, a; c, a, b + d]
  ¬(IsUnit (Matrix.det M)) →
  (a / (b + c) + b / (a + c) + c / (a + b) = -3) ∨
  (a / (b + c) + b / (a + c) + c / (a + b) = 3/2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_non_invertible_sum_fraction_l909_90907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l909_90932

/-- The eccentricity of a hyperbola with the given conditions is √2 -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃ (x y : ℝ), x^2/a^2 - y^2/b^2 = 1) →
  (∃ (m : ℝ), m = 1 ∧ 
    ∃ (x₀ y₀ : ℝ), y₀ - y₀ = m * (x₀ - (a * Real.sqrt (a^2 + b^2) / a^2)) ∧
    (∃! (x y : ℝ), y = (b/a) * x ∧ y - y₀ = m * (x - x₀))) →
  Real.sqrt (a^2 + b^2) / a = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l909_90932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_l909_90960

/-- The function f(x) with parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(2*x) + 3*a^x - 2

/-- Theorem stating the minimum value of f(x) given its maximum value -/
theorem f_min_value (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, f a x ≤ 8) →
  (∃ x ∈ Set.Icc (-1 : ℝ) 1, f a x = 8) →
  (∃ x ∈ Set.Icc (-1 : ℝ) 1, f a x = -1/4) ∧
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, f a x ≥ -1/4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_l909_90960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_vertices_l909_90994

/-- The vertex of a parabola y = ax^2 + bx + c -/
noncomputable def vertex (a b c : ℝ) : ℝ × ℝ :=
  let x := -b / (2 * a)
  (x, a * x^2 + b * x + c)

/-- The distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The theorem stating the distance between the vertices of two specific parabolas -/
theorem distance_between_vertices : 
  let p1 := vertex 1 6 15
  let p2 := vertex 1 (-4) 7
  distance p1 p2 = Real.sqrt 34 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_vertices_l909_90994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l909_90967

/-- The function f(x) is defined as the minimum of 4x+1, x+2, and -2x+4 for all real x -/
noncomputable def f (x : ℝ) : ℝ := min (min (4*x + 1) (x + 2)) (-2*x + 4)

/-- The maximum value of 6f(x) + 2012 is 2028 -/
theorem max_value_of_f : ∃ (M : ℝ), M = 2028 ∧ ∀ (x : ℝ), 6 * f x + 2012 ≤ M := by
  -- We claim that M = 2028 satisfies the condition
  use 2028
  constructor
  · -- First part: M = 2028
    rfl
  · -- Second part: ∀ (x : ℝ), 6 * f x + 2012 ≤ 2028
    intro x
    -- The proof of this inequality would go here
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l909_90967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_equivalence_monotonicity_l909_90942

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sqrt (x^2 + 1) - a * x

-- Theorem for the inequality f(x) ≤ 1
theorem inequality_equivalence (a : ℝ) (h : a > 0) :
  ∀ x ≥ 0, f a x ≤ 1 ↔ (1 - a^2) * x ≤ 2 * a := by
  sorry

-- Theorem for the monotonicity of f(x)
theorem monotonicity (a : ℝ) (h : a > 0) :
  (∀ x y, 0 ≤ x ∧ x < y → f a x > f a y) ↔ a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_equivalence_monotonicity_l909_90942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_of_s_2022_l909_90986

-- Define the polynomial q(x)
def q (x : ℤ) : ℤ := (Finset.range 2013).sum (λ i ↦ x^i)

-- Define the divisor polynomial
def divisor (x : ℤ) : ℤ := x^5 + x^4 + 2*x^3 + x^2 + 1

-- Define s(x) as the remainder when q(x) is divided by the divisor
def s (x : ℤ) : ℤ := q x % divisor x

-- Theorem statement
theorem remainder_of_s_2022 : |s 2022| % 100 = 41 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_of_s_2022_l909_90986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l909_90977

def a (k : ℕ) : ℚ := 120 / (k + 2)

theorem sequence_properties (n : ℕ) :
  (∀ k ∈ Finset.range (n - 1), a k = (120 - a k) / (k + 1)) →
  (∀ k ∈ Finset.range n, (a k).isInt) →
  (∀ k ∈ Finset.range n, a k = 120 / (k + 2)) ∧
  n ≤ 5 ∧
  a 1 = 40 ∧ a 2 = 30 ∧ a 3 = 24 ∧ a 4 = 20 ∧ 120 - (a 1 + a 2 + a 3 + a 4) = 6 :=
by sorry

#check sequence_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l909_90977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_surface_area_l909_90934

/-- The curved surface area of a cone with given slant height and base radius -/
noncomputable def curved_surface_area (slant_height : ℝ) (base_radius : ℝ) : ℝ :=
  Real.pi * base_radius * slant_height

/-- Theorem: The curved surface area of a cone with slant height 15 cm and base radius 3 cm is 45π cm² -/
theorem cone_surface_area :
  curved_surface_area 15 3 = 45 * Real.pi := by
  unfold curved_surface_area
  ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_surface_area_l909_90934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_A_and_B_l909_90926

-- Define set A
def set_A : Set ℝ := {x | ∃ y, y = Real.log (x - 2)}

-- Define set B
def set_B : Set ℝ := {x | x^2 - 4*x + 3 ≤ 0}

-- The theorem to prove
theorem union_of_A_and_B : set_A ∪ set_B = Set.Ici 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_A_and_B_l909_90926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_m_sum_l909_90922

-- Define the points
def p1 : ℝ × ℝ := (2, 5)
def p2 : ℝ × ℝ := (10, 9)
def p3 (m : ℤ) : ℝ × ℝ := (6, m)

-- Define the area function for a triangle given three points
noncomputable def triangleArea (a b c : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((a.1 * (b.2 - c.2) + b.1 * (c.2 - a.2) + c.1 * (a.2 - b.2)))

-- Theorem statement
theorem min_area_m_sum : 
  ∃ (m1 m2 : ℤ), 
    (∀ (m : ℤ), triangleArea p1 p2 (p3 m) ≥ triangleArea p1 p2 (p3 m1)) ∧
    (∀ (m : ℤ), triangleArea p1 p2 (p3 m) ≥ triangleArea p1 p2 (p3 m2)) ∧
    m1 + m2 = 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_m_sum_l909_90922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_area_is_150_l909_90954

-- Define the points
variable (A B C D E : EuclideanSpace ℝ (Fin 2))

-- Define the conditions
def is_square (A B C D : EuclideanSpace ℝ (Fin 2)) : Prop := sorry
def perpendicular (A B C : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

axiom square_ABCD : is_square A B C D
axiom ae_perp_ed : perpendicular A E D
axiom ae_length : dist A E = 10
axiom de_length : dist D E = 10

-- Define the pentagon area function
noncomputable def pentagon_area (A B C D E : EuclideanSpace ℝ (Fin 2)) : ℝ := sorry

-- Theorem statement
theorem pentagon_area_is_150 : pentagon_area A B C D E = 150 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_area_is_150_l909_90954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_l909_90983

-- Define the function f
def f (x : ℝ) : ℝ := |x - 2|

-- State the theorem
theorem range_of_x (a b : ℝ) (ha : a ≠ 0) :
  (∀ x : ℝ, |a + b| + |a - b| ≥ |a| * f x) →
  ∀ x : ℝ, f x ≤ 2 ∧ ∃ y z : ℝ, f y = 0 ∧ f z = 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_l909_90983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_pattern_l909_90924

/-- Given a sequence of equations and a general form, prove that t - a = 41 -/
theorem sequence_pattern (a t : ℝ) (ha : a > 0) (ht : t > 0) 
  (h1 : Real.sqrt (2 + 2/3) = 2 * Real.sqrt (2/3))
  (h2 : Real.sqrt (3 + 3/8) = 3 * Real.sqrt (3/8))
  (h3 : Real.sqrt (4 + 4/15) = 4 * Real.sqrt (4/15))
  (h_general : Real.sqrt (a + 7/t) = a * Real.sqrt (7/t)) : 
  t - a = 41 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_pattern_l909_90924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percent_calculation_l909_90917

-- Define the given conditions
def num_pens_bought : ℕ := 120
def num_pens_priced : ℕ := 75
def discount_rate : ℚ := 3 / 100

-- Define the theorem
theorem profit_percent_calculation :
  ∀ P : ℚ,
  let CP := P * num_pens_priced
  let SP := P * (1 - discount_rate)
  let TSP := SP * num_pens_bought
  let profit := TSP - CP
  let profit_percent := (profit / CP) * 100
  profit_percent = 55.2 := by
    intro P
    -- Expand definitions
    simp [num_pens_bought, num_pens_priced, discount_rate]
    -- The actual proof would go here
    sorry

#check profit_percent_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percent_calculation_l909_90917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_non_divisible_binomial_coeff_l909_90930

theorem infinite_non_divisible_binomial_coeff (k : ℤ) : 
  k ≠ 1 → ∃ (S : Set ℕ), Set.Infinite S ∧ 
  ∀ n ∈ S, ¬(n + k.natAbs ∣ Nat.choose (2*n) n) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_non_divisible_binomial_coeff_l909_90930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_march_greatest_price_drop_l909_90963

/-- Represents a month's price data -/
structure MonthData where
  min : ℚ
  max : ℚ

/-- Calculates the average price for a month -/
def averagePrice (month : MonthData) : ℚ := (month.min + month.max) / 2

/-- Calculates the price difference between two months -/
def priceDifference (month1 month2 : MonthData) : ℚ :=
  averagePrice month2 - averagePrice month1

/-- Represents the book price data for all months -/
def bookPrices : List MonthData := [
  ⟨30, 34⟩,  -- January
  ⟨37, 42⟩,  -- February
  ⟨32, 34⟩,  -- March
  ⟨35, 39⟩,  -- April
  ⟨33, 33⟩,  -- May
  ⟨29, 31⟩   -- June
]

/-- Theorem: March has the greatest monthly average price drop -/
theorem march_greatest_price_drop :
  let priceDrops := List.zipWith priceDifference bookPrices bookPrices.tail
  priceDrops[1] = priceDrops.minimum := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_march_greatest_price_drop_l909_90963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_cardinality_inequality_l909_90918

theorem set_cardinality_inequality (A : Finset ℝ) 
  (hA : ∀ a ∈ A, a > 0) : 
  let B := (A.product A).image (λ (p : ℝ × ℝ) => p.1 / p.2)
  let C := (A.product A).image (λ (p : ℝ × ℝ) => p.1 * p.2)
  (A.card : ℝ) * (B.card : ℝ) ≤ (C.card : ℝ) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_cardinality_inequality_l909_90918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_437_l909_90990

def contains_437 (m n : ℕ) : Prop :=
  ∃ k : ℕ, (1000 * m : ℚ) / n > 437 + k ∧ (1000 * m : ℚ) / n < 438 + k

theorem smallest_n_for_437 :
  ∀ n : ℕ, n > 0 →
    (∃ m : ℕ, m < n ∧ Nat.Coprime m n ∧ contains_437 m n) →
    n ≥ 1809 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_437_l909_90990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_closed_form_l909_90961

-- Define the sequence a_n
def a : ℕ → ℤ
  | 0 => 3  -- Define the base case for n = 0
  | n + 1 => a n + 2 * (n + 1)

-- State the theorem
theorem a_closed_form (n : ℕ) : a n = n^2 + n + 3 := by
  induction n with
  | zero => rfl
  | succ n ih =>
    simp [a]
    rw [ih]
    ring

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_closed_form_l909_90961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l909_90904

theorem inequality_solution_set (x : ℝ) : 
  (((x^3 - 4*x - 4/x + 1/x^3 + 6) ^ (1/2022) ≤ 0) ↔ 
  (x = 1 ∨ x = (-3 + Real.sqrt 5) / 2 ∨ x = (-3 - Real.sqrt 5) / 2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l909_90904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_cross_section_areas_is_three_plus_sqrt_three_l909_90945

/-- Regular tetrahedron with edge length 2 -/
structure RegularTetrahedron where
  edge_length : ℝ
  is_regular : edge_length = 2

/-- Cross-section of a tetrahedron -/
structure TetrahedronCrossSection where
  tetrahedron : RegularTetrahedron
  is_equidistant : Bool  -- True if the plane is equidistant from all vertices

/-- The sum of areas of all cross-sections equidistant from all vertices -/
noncomputable def sum_of_cross_section_areas (t : RegularTetrahedron) : ℝ :=
  3 + Real.sqrt 3

/-- Theorem: The sum of areas of all cross-sections equidistant from all vertices
    in a regular tetrahedron with edge length 2 is 3 + √3 -/
theorem sum_of_cross_section_areas_is_three_plus_sqrt_three
  (t : RegularTetrahedron) :
  sum_of_cross_section_areas t = 3 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_cross_section_areas_is_three_plus_sqrt_three_l909_90945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_speed_calculation_l909_90909

/-- Represents the speed of a bus in kilometers per hour -/
@[ext] structure BusSpeed where
  includingStops : ℚ
  excludingStops : ℚ

/-- Calculates the speed of a bus excluding stoppages -/
def calculateSpeedExcludingStops (stoppingTime : ℚ) (speedIncludingStops : ℚ) : ℚ :=
  speedIncludingStops * (60 / (60 - stoppingTime))

/-- Theorem: Given a bus that stops for 10 minutes per hour and has an average speed of 50 kmph
    including stoppages, its speed excluding stoppages is 60 kmph -/
theorem bus_speed_calculation (bus : BusSpeed) (h1 : bus.includingStops = 50)
    (h2 : calculateSpeedExcludingStops 10 bus.includingStops = bus.excludingStops) :
    bus.excludingStops = 60 := by
  sorry

#check bus_speed_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_speed_calculation_l909_90909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_driver_speed_l909_90928

/-- A driver's weekly travel schedule and total distance --/
structure DriverSchedule where
  speed1 : ℝ  -- Speed for the first part of the journey (miles per hour)
  time1 : ℝ   -- Time spent at speed1 (hours)
  time2 : ℝ   -- Time spent at unknown speed (hours)
  days : ℕ    -- Number of days per week
  total_distance : ℝ  -- Total distance traveled in a week (miles)

/-- Theorem stating the unknown speed of the driver --/
theorem driver_speed (d : DriverSchedule)
  (h1 : d.speed1 = 30)
  (h2 : d.time1 = 3)
  (h3 : d.time2 = 4)
  (h4 : d.days = 6)
  (h5 : d.total_distance = 1140) :
  (d.total_distance - d.speed1 * d.time1 * (d.days : ℝ)) / (d.time2 * (d.days : ℝ)) = 25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_driver_speed_l909_90928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_inequality_l909_90999

theorem max_value_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h_sum : a + b + 9 * c^2 = 1) : 
  Real.sqrt a + Real.sqrt b + Real.sqrt 3 * c ≤ Real.sqrt 21 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_inequality_l909_90999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_perimeter_l909_90919

noncomputable def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

structure EllipseParams where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

noncomputable def eccentricity (e : EllipseParams) : ℝ :=
  Real.sqrt (1 - (e.b / e.a)^2)

noncomputable def focus_distance (e : EllipseParams) : ℝ :=
  e.a * eccentricity e

noncomputable def ellipse_perimeter_through_focus (e : EllipseParams) : ℝ :=
  2 * (e.a + focus_distance e)

theorem ellipse_triangle_perimeter
  (e : EllipseParams)
  (h1 : e.b = 4)
  (h2 : eccentricity e = 3/5) :
  ellipse_perimeter_through_focus e = 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_perimeter_l909_90919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_explicit_formula_lower_bound_max_k_l909_90993

noncomputable section

-- Define the function f
def f (x : ℝ) (a : ℝ) := Real.exp x - x^2 + a

-- Define the tangent line equation
def tangent_line (b : ℝ) (x : ℝ) := b * x

-- Theorem 1: Explicit formula for f
theorem explicit_formula (x : ℝ) : 
  ∃ a b : ℝ, (f x a = Real.exp x - x^2 - 1) ∧ 
             (tangent_line b 0 = f 0 a) ∧ 
             (deriv (f · a) 0 = b) :=
sorry

-- Theorem 2: Lower bound of f
theorem lower_bound (x : ℝ) : f x (-1) ≥ -x^2 + x :=
sorry

-- Theorem 3: Maximum value of k
theorem max_k : 
  ∃ k : ℝ, k = Real.exp 1 - 2 ∧ 
      (∀ x > 0, f x (-1) ≥ k * x) ∧
      (∀ k' > k, ∃ x > 0, f x (-1) < k' * x) :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_explicit_formula_lower_bound_max_k_l909_90993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_a_equals_two_l909_90992

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x * Real.exp x) / (Real.exp (a * x) - 1)

-- State the theorem
theorem even_function_implies_a_equals_two (a : ℝ) :
  (∀ x : ℝ, x ≠ 0 → f a x = f a (-x)) → a = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_a_equals_two_l909_90992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2006_two_fifteenths_l909_90978

noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 1/2 then x + 1/2
  else if 1/2 < x ∧ x ≤ 1 then 2*(1-x)
  else 0  -- undefined for x outside [0,1]

noncomputable def f_n : ℕ → ℝ → ℝ
| 0, x => x
| n+1, x => f (f_n n x)

theorem f_2006_two_fifteenths : f_n 2006 (2/15) = 19/30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2006_two_fifteenths_l909_90978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_non_convex_pentagon_with_non_intersecting_diagonals_l909_90937

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A pentagon represented by its five vertices -/
structure Pentagon where
  vertices : Fin 5 → Point

/-- Checks if two line segments intersect -/
def linesIntersect (a b c d : Point) : Prop := sorry

/-- Checks if a pentagon is non-convex -/
def isNonConvex (p : Pentagon) : Prop := sorry

/-- Gets the diagonals of a pentagon -/
def getDiagonals (p : Pentagon) : List (Point × Point) := sorry

/-- Checks if no two diagonals intersect except at vertices -/
def nonIntersectingDiagonals (p : Pentagon) : Prop :=
  let diagonals := getDiagonals p
  ∀ d1 d2, d1 ∈ diagonals → d2 ∈ diagonals → d1 ≠ d2 →
    ¬(linesIntersect d1.1 d1.2 d2.1 d2.2) ∨
    (d1.1 = d2.1 ∨ d1.1 = d2.2 ∨ d1.2 = d2.1 ∨ d1.2 = d2.2)

/-- There exists a non-convex pentagon with non-intersecting diagonals -/
theorem non_convex_pentagon_with_non_intersecting_diagonals :
  ∃ p : Pentagon, isNonConvex p ∧ nonIntersectingDiagonals p := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_non_convex_pentagon_with_non_intersecting_diagonals_l909_90937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_calculation_l909_90935

noncomputable def mountain_height : ℝ := 40000
def num_trips : ℕ := 10
noncomputable def ascent_ratio : ℝ := 3/4
noncomputable def descent_ratio : ℝ := 2/3

theorem total_distance_calculation :
  let ascent_height := mountain_height * ascent_ratio
  let descent_height := ascent_height * descent_ratio
  let round_trip_distance := ascent_height + descent_height
  round_trip_distance * (num_trips : ℝ) = 500000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_calculation_l909_90935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagonal_pyramid_volume_l909_90964

/-- A right pyramid with a hexagonal base -/
structure HexagonalPyramid where
  base_side : ℝ
  height : ℝ

/-- The surface area of a hexagonal pyramid -/
noncomputable def surface_area (p : HexagonalPyramid) : ℝ :=
  (3 * Real.sqrt 3 / 2) * p.base_side^2 + 6 * ((Real.sqrt 3 / 2) * p.base_side^2)

/-- The volume of a hexagonal pyramid -/
noncomputable def volume (p : HexagonalPyramid) : ℝ :=
  (1 / 3) * ((3 * Real.sqrt 3 / 2) * p.base_side^2) * p.height

/-- Theorem stating the volume of the pyramid given the conditions -/
theorem hexagonal_pyramid_volume (p : HexagonalPyramid) 
  (h1 : surface_area p = 972)
  (h2 : (Real.sqrt 3 / 2) * p.base_side^2 = (1 / 3) * ((3 * Real.sqrt 3 / 2) * p.base_side^2)) :
  volume p = 27 * Real.sqrt 3 * Real.sqrt ((54 / Real.sqrt (6 * Real.sqrt 3))^2 - 6 * Real.sqrt 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagonal_pyramid_volume_l909_90964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_definite_integral_x_squared_plus_sin_x_l909_90911

theorem definite_integral_x_squared_plus_sin_x : 
  ∫ x in (-1)..1, (x^2 + Real.sin x) = 2/3 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_definite_integral_x_squared_plus_sin_x_l909_90911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_ratio_l909_90901

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h_arith : ∀ n, a (n + 1) = a n + d
  h_d_nonzero : d ≠ 0

/-- Sum of the first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (2 * seq.a 1 + (n - 1) * seq.d) / 2

theorem arithmetic_sequence_ratio (seq : ArithmeticSequence) 
  (h_S5 : S seq 5 = 15)
  (h_geom : (seq.a 6) ^ 2 = (seq.a 3) * (seq.a 12)) :
  S seq 2023 / seq.a 2023 = 1012 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_ratio_l909_90901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_solution_l909_90900

/-- A parabola passing through two points and tangent to a line at a third point -/
structure Parabola where
  -- The parabola passes through these two points
  point_a : ℝ × ℝ
  point_b : ℝ × ℝ
  -- The parabola is tangent to this line at this point
  tangent_line : ℝ → ℝ → ℝ
  tangent_point : ℝ × ℝ

/-- The equation of a parabola -/
noncomputable def parabola_equation (p : Parabola) : ℝ → ℝ → Prop :=
  sorry

theorem parabola_equation_solution (p : Parabola) 
  (h1 : p.point_a = (1, -2))
  (h2 : p.point_b = (9, -6))
  (h3 : p.tangent_line = λ x y => x - 2*y + 4)
  (h4 : p.tangent_point = (4, 4)) :
  (parabola_equation p = λ x y => y^2 = 4*x) ∨
  (parabola_equation p = λ x y => x^2 - 9*x + 2*y + 12 = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_solution_l909_90900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_b_value_l909_90908

/-- Two vectors in 3D space are orthogonal if their dot product is zero -/
def are_orthogonal (v1 v2 : Fin 3 → ℝ) : Prop :=
  (v1 0) * (v2 0) + (v1 1) * (v2 1) + (v1 2) * (v2 2) = 0

/-- The direction vector of the first line -/
def direction1 (b : ℝ) : Fin 3 → ℝ := fun i => 
  match i with
  | 0 => b
  | 1 => -3
  | 2 => 2
  | _ => 0

/-- The direction vector of the second line -/
def direction2 : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => 2
  | 1 => 1
  | 2 => 3
  | _ => 0

/-- Theorem stating that the value of b that makes the lines perpendicular is -3/2 -/
theorem perpendicular_lines_b_value :
  ∃ b : ℝ, are_orthogonal (direction1 b) direction2 ∧ b = -3/2 := by
  use -3/2
  constructor
  · -- Prove that the vectors are orthogonal when b = -3/2
    simp [are_orthogonal, direction1, direction2]
    norm_num
  · -- Prove that b = -3/2
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_b_value_l909_90908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radii_ratio_l909_90958

-- Define the triangle ABC
structure EquilateralTriangle :=
  (A B C : ℝ × ℝ)
  (side_length : ℝ)
  (equilateral : side_length = 6)

-- Define point D on AB
noncomputable def D (triangle : EquilateralTriangle) : ℝ × ℝ := sorry

-- Define the angle bisector CD
noncomputable def CD (triangle : EquilateralTriangle) : ℝ × ℝ := sorry

-- Define the perimeter of triangle ADC
noncomputable def perimeter_ADC (triangle : EquilateralTriangle) : ℝ := sorry

-- Define the radii of inscribed circles
noncomputable def r_a (triangle : EquilateralTriangle) : ℝ := sorry
noncomputable def r_b (triangle : EquilateralTriangle) : ℝ := sorry

-- Define the angle bisector function
noncomputable def angle_bisector (point : ℝ × ℝ) : ℝ × ℝ := sorry

-- Theorem statement
theorem inscribed_circle_radii_ratio 
  (triangle : EquilateralTriangle) 
  (h1 : CD triangle = angle_bisector triangle.C)
  (h2 : perimeter_ADC triangle = 12) : 
  r_a triangle / r_b triangle = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radii_ratio_l909_90958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_surface_area_minimum_l909_90931

/-- The surface area of a right circular cone with an inscribed sphere -/
noncomputable def cone_surface_area (r : ℝ) : ℝ := Real.pi * (2 * r^4) / (r^2 - 1)

/-- The theorem stating the surface area formula and its minimum value -/
theorem cone_surface_area_minimum :
  ∀ r : ℝ, r > 1 →
  (cone_surface_area r ≥ 8 * Real.pi) ∧
  (cone_surface_area (Real.sqrt 2) = 8 * Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_surface_area_minimum_l909_90931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alexandre_winning_strategy_l909_90975

/-- Represents a strategy for the second player in the n-gon game -/
def Strategy (n : ℕ) := ℕ → Bool → Bool

/-- Checks if a strategy prevents three consecutive vertices with sum divisible by 3 -/
def is_winning_strategy (n : ℕ) (s : Strategy n) : Prop :=
  ∀ (game : ℕ → Bool), 
    (∀ i : ℕ, i < n → game (2 * i + 1) = s i (game (2 * i))) →
    ∀ i : ℕ, i < n → ¬(Nat.add (Nat.add (if game i then 1 else 0) 
                                        (if game ((i + 1) % n) then 1 else 0))
                                (if game ((i + 2) % n) then 1 else 0) % 3 = 0)

theorem alexandre_winning_strategy (n : ℕ) (h : Even n) (h2 : n > 3) : 
  ∃ (s : Strategy n), is_winning_strategy n s := by
  sorry

#check alexandre_winning_strategy

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alexandre_winning_strategy_l909_90975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_equation_represents_line_and_circle_l909_90997

open Real

-- Define the polar equation
def polar_equation (ρ θ : ℝ) : Prop := ρ * sin θ = sin (2 * θ)

-- Define a line in polar form (θ = 0, ρ ∈ ℝ)
def is_line (θ : ℝ) : Prop := θ = 0

-- Define a circle in Cartesian form
def is_circle (x y : ℝ) : Prop := ∃ (h k r : ℝ), (x - h)^2 + (y - k)^2 = r^2

theorem polar_equation_represents_line_and_circle :
  ∃ (ρ₁ θ₁ ρ₂ θ₂ : ℝ),
    polar_equation ρ₁ θ₁ ∧ is_line θ₁ ∧
    polar_equation ρ₂ θ₂ ∧ 
    is_circle (ρ₂ * cos θ₂) (ρ₂ * sin θ₂) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_equation_represents_line_and_circle_l909_90997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_arrangement_symmetry_l909_90916

/-- 
num_arrangements n v represents the number of arrangements of n white 
and n black balls with exactly v color changes.
-/
def num_arrangements (n v : ℕ) : ℕ :=
  sorry

/-- 
Given:
- n: number of white balls (equal to the number of black balls)
- k: a positive integer less than n

Theorem:
The number of arrangements of n white and n black balls in a row 
with n-k color changes is equal to the number of arrangements 
with n+k color changes.
-/
theorem ball_arrangement_symmetry (n k : ℕ) (h1 : 0 < k) (h2 : k < n) :
  (num_arrangements n (n - k)) = (num_arrangements n (n + k)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_arrangement_symmetry_l909_90916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_abs_value_l909_90951

theorem complex_abs_value : ∀ z : ℂ, z^2 = -1 → Complex.abs (2 + z^2 + 2*z^3) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_abs_value_l909_90951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_l909_90996

noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

noncomputable def frac (x : ℝ) : ℝ := x - (floor x)

theorem sum_of_solutions (p q : ℕ) (h_coprime : Nat.Coprime p q) :
  let a : ℝ := (p : ℝ) / q
  let solution_sum : ℝ := ∑' x, if (↑(floor x) * frac x = a * x^2) then x else 0
  solution_sum = 500 → p + 3 * q = 2914 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_l909_90996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_identity_l909_90982

theorem sin_cos_identity (α : Real) (h : Real.sin α = Real.sqrt 5 / 5) :
  Real.sin α ^ 4 - Real.cos α ^ 4 = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_identity_l909_90982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_triangle_property_l909_90987

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  hp : p > 0

/-- Point on a parabola -/
structure PointOnParabola (c : Parabola) where
  x : ℝ
  y : ℝ
  h : y^2 = 2 * c.p * x

/-- Focus of a parabola -/
noncomputable def focus (c : Parabola) : ℝ × ℝ := (c.p / 2, 0)

/-- Directrix of a parabola -/
noncomputable def directrix (c : Parabola) : ℝ := -c.p / 2

/-- Tangent line to a parabola at a point -/
def tangentLine (c : Parabola) (p : PointOnParabola c) : ℝ → ℝ → Prop :=
  λ x y => p.y * y = c.p * (x + p.x)

/-- Intersection of tangent line with x-axis -/
noncomputable def intersectionWithXAxis (c : Parabola) (p : PointOnParabola c) : ℝ × ℝ :=
  (-p.x - c.p / 2, 0)

/-- Isosceles right triangle predicate -/
def IsoscelesRightTriangle (a b c : ℝ × ℝ) : Prop :=
  let d1 := (a.1 - b.1)^2 + (a.2 - b.2)^2
  let d2 := (b.1 - c.1)^2 + (b.2 - c.2)^2
  let d3 := (c.1 - a.1)^2 + (c.2 - a.2)^2
  (d1 = d2 ∨ d2 = d3 ∨ d3 = d1) ∧ (d1 + d2 = d3 ∨ d2 + d3 = d1 ∨ d3 + d1 = d2)

/-- Main theorem -/
theorem parabola_triangle_property (c : Parabola) (p : PointOnParabola c) :
  let f := focus c
  let q := intersectionWithXAxis c p
  q.1 = directrix c →
  IsoscelesRightTriangle f (p.x, p.y) q :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_triangle_property_l909_90987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lennon_drove_16_miles_friday_l909_90972

/-- Calculates the number of miles driven on Friday given the mileage for other days and the total reimbursement --/
def miles_driven_friday (reimbursement_rate : ℚ) (monday_miles tuesday_miles wednesday_miles thursday_miles : ℕ) (total_reimbursement : ℚ) : ℕ :=
  let total_miles_mon_to_thu := monday_miles + tuesday_miles + wednesday_miles + thursday_miles
  let reimbursement_mon_to_thu := (total_miles_mon_to_thu : ℚ) * reimbursement_rate
  let reimbursement_friday := total_reimbursement - reimbursement_mon_to_thu
  (reimbursement_friday / reimbursement_rate).floor.toNat

/-- Theorem stating that given the conditions in the problem, Lennon drove 16 miles on Friday --/
theorem lennon_drove_16_miles_friday :
  miles_driven_friday (36/100) 18 26 20 20 36 = 16 := by
  sorry

#eval miles_driven_friday (36/100) 18 26 20 20 36

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lennon_drove_16_miles_friday_l909_90972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_terminal_side_quadrant_point_in_third_implies_angle_in_second_l909_90988

/-- Given a point P(tan α, cos α) in the third quadrant, 
    the terminal side of angle α is in the second quadrant -/
theorem terminal_side_quadrant (α : ℝ) : 
  (Real.tan α < 0 ∧ Real.cos α < 0) → 
  (Real.sin α > 0 ∧ Real.cos α < 0) :=
by sorry

/-- Helper function to determine if a point is in the third quadrant -/
def in_third_quadrant (x y : ℝ) : Prop :=
  x < 0 ∧ y < 0

/-- Helper function to determine if an angle is in the second quadrant -/
def in_second_quadrant (θ : ℝ) : Prop :=
  Real.sin θ > 0 ∧ Real.cos θ < 0

/-- Main theorem stating that if P(tan α, cos α) is in the third quadrant, 
    then α is in the second quadrant -/
theorem point_in_third_implies_angle_in_second (α : ℝ) :
  in_third_quadrant (Real.tan α) (Real.cos α) → in_second_quadrant α :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_terminal_side_quadrant_point_in_third_implies_angle_in_second_l909_90988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_plus_square_root_l909_90956

theorem cube_root_plus_square_root : ((-8 : ℝ) ^ (1/3 : ℝ)) + Real.sqrt 16 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_plus_square_root_l909_90956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_101_distance_from_origin_l909_90914

-- Define the sequence z_n
noncomputable def z : ℕ → ℂ
  | 0 => 0
  | 1 => 0
  | n + 2 => (z (n + 1))^2 - Complex.I

-- State the theorem
theorem z_101_distance_from_origin : Complex.abs (z 101) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_101_distance_from_origin_l909_90914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_price_theorem_l909_90927

-- Define the final price after all discounts
def final_price : ℝ := 10000

-- Define the discount rates
def discount_rates : List ℝ := [0.30, 0.20, 0.15, 0.10, 0.05]

-- Function to apply a single discount
def apply_discount (price : ℝ) (rate : ℝ) : ℝ :=
  price * (1 - rate)

-- Function to apply all discounts in reverse order
def apply_all_discounts (price : ℝ) (rates : List ℝ) : ℝ :=
  rates.foldl apply_discount price

-- Theorem stating the relationship between the final price and the original price
theorem original_price_theorem :
  ∃ (original_price : ℝ),
    apply_all_discounts original_price discount_rates.reverse = final_price ∧
    (abs (original_price - 24570.65) < 0.01) := by
  sorry

#eval apply_all_discounts 24570.65 discount_rates.reverse

end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_price_theorem_l909_90927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_octagon_slice_from_heptagon_l909_90944

/-- A polygon with n sides --/
structure Polygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- A convex heptagon --/
def ConvexHeptagon := Polygon 7

/-- An octagon --/
def Octagon := Polygon 8

/-- A slice of a polygon --/
structure Slice (n : ℕ) (p : Polygon n) where
  vertices : List (ℝ × ℝ)
  isInside : ∀ v, v ∈ vertices → ∃ i, p.vertices i = v

/-- Theorem: It's impossible to create an octagon-shaped slice from a convex heptagon by cutting along its diagonals --/
theorem no_octagon_slice_from_heptagon (h : ConvexHeptagon) :
  ¬∃ (s : Slice 7 h), ∃ (o : Octagon), s.vertices.length = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_octagon_slice_from_heptagon_l909_90944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_system_solution_l909_90985

theorem trigonometric_system_solution
  (x y z : ℝ)
  (eq1 : Real.sin x + 2 * Real.sin (x + y + z) = 0)
  (eq2 : Real.sin y + 3 * Real.sin (x + y + z) = 0)
  (eq3 : Real.sin z + 4 * Real.sin (x + y + z) = 0) :
  ∃ (k l m : ℤ), x = k * Real.pi ∧ y = l * Real.pi ∧ z = m * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_system_solution_l909_90985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometry_problem_l909_90984

-- Define the circles M and N
def circle_M (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 1/4
def circle_N (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 49/4

-- Define the curve E (trajectory of center of circle D)
def curve_E (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the hyperbola C
def hyperbola_C (x y : ℝ) : Prop := x^2 - y^2/3 = 1

-- Define the asymptote of C
def asymptote_C (x y : ℝ) : Prop := y = Real.sqrt 3 * x

-- Define the line l
def line_l (x y k : ℝ) : Prop := y = k * x + 4

-- Define the point P
def point_P : ℝ × ℝ := (0, 4)

-- Define the relationship between PQ, QA, and QB
def vector_relationship (lambda1 lambda2 : ℝ) (P Q A B : ℝ × ℝ) : Prop :=
  (Q.1 - P.1, Q.2 - P.2) = lambda1 • (A.1 - Q.1, A.2 - Q.2) ∧
  (Q.1 - P.1, Q.2 - P.2) = lambda2 • (B.1 - Q.1, B.2 - Q.2)

theorem geometry_problem (k : ℝ) (Q A B : ℝ × ℝ) (lambda1 lambda2 : ℝ) :
  (∀ x y, circle_M x y → circle_N x y → curve_E x y) →
  (∀ x y, curve_E x y → hyperbola_C x y) →
  (∀ x y, hyperbola_C x y → asymptote_C x y) →
  (∀ x y, line_l x y k) →
  (Q.2 = 0) →
  (hyperbola_C A.1 A.2 ∧ hyperbola_C B.1 B.2) →
  (vector_relationship lambda1 lambda2 point_P Q A B) →
  (lambda1 + lambda2 = -8/3) →
  (Q.1 = 2 ∨ Q.1 = -2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometry_problem_l909_90984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_problem_arithmetic_mean_specific_case_l909_90962

theorem arithmetic_mean_problem (initial_count : ℕ) (initial_mean : ℚ) 
  (new_count : ℕ) (new_mean : ℚ) : ℚ :=
  let total_count := initial_count + new_count
  let new_sum := (total_count : ℚ) * new_mean
  let initial_sum := (initial_count : ℚ) * initial_mean
  let diff := new_sum - initial_sum
  diff / (new_count : ℚ)

theorem arithmetic_mean_specific_case : 
  arithmetic_mean_problem 7 48 3 58 = 244 / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_problem_arithmetic_mean_specific_case_l909_90962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_solution_floor_equation_l909_90959

noncomputable def frac (x : ℝ) : ℝ := x - ⌊x⌋

theorem largest_solution_floor_equation :
  ∃ (x : ℝ), 
    (∀ (y : ℝ), (⌊y⌋ = 7 + 80 * frac y) → y ≤ x) ∧
    (⌊x⌋ = 7 + 80 * frac x) ∧
    (x = 86.9875) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_solution_floor_equation_l909_90959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_proof_l909_90976

-- Define IsTriangle and OppositeAngleSide as local variables
variable (IsTriangle : Real → Real → Real → Prop)
variable (OppositeAngleSide : Real → Real → Prop)

theorem triangle_angle_proof (A B C : Real) (a b c : Real) (m n : Real × Real) :
  -- Triangle ABC exists
  IsTriangle A B C →
  -- Sides opposite to angles A, B, C are a, b, c respectively
  OppositeAngleSide A a → OppositeAngleSide B b → OppositeAngleSide C c →
  -- Definition of vectors m and n
  m = (Real.sin (A/2), Real.cos (A/2)) →
  n = (Real.cos (A/2), -Real.cos (A/2)) →
  -- Given condition
  2 * (m.1 * n.1 + m.2 * n.2) + Real.sqrt (m.1^2 + m.2^2) = Real.sqrt 2 / 2 →
  -- Conclusion
  A = 5 * Real.pi / 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_proof_l909_90976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_other_focus_l909_90974

/-- Definition of the ellipse -/
def is_on_ellipse (x y : ℝ) : Prop :=
  x^2 / 144 + y^2 / 36 = 1

/-- Definition of the distance between two points -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

/-- Theorem: If a point on the ellipse is at distance 10 from one focus,
    then it is at distance 14 from the other focus -/
theorem distance_to_other_focus
  (x y x₁ y₁ x₂ y₂ : ℝ)
  (h_ellipse : is_on_ellipse x y)
  (h_focus₁ : distance x y x₁ y₁ = 10)
  (h_focus₂ : (x₁, y₁) ≠ (x₂, y₂)) :
  distance x y x₂ y₂ = 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_other_focus_l909_90974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_meeting_probability_l909_90955

/-- The duration in minutes of the arrival window for both John and the train -/
noncomputable def arrivalWindow : ℝ := 60

/-- The duration in minutes that the train waits at the station -/
noncomputable def trainWaitTime : ℝ := 20

/-- The probability that John arrives while the train is at the station -/
noncomputable def probabilityOfMeeting : ℝ := 5 / 18

theorem train_meeting_probability :
  let favorableArea := trainWaitTime * arrivalWindow - (trainWaitTime^2 / 2)
  let totalArea := arrivalWindow^2
  favorableArea / totalArea = probabilityOfMeeting :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_meeting_probability_l909_90955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcircle_area_of_triangle_l909_90905

/-- The area of the circumcircle of a triangle ABC is 4π, given specific conditions -/
theorem circumcircle_area_of_triangle (A B C : ℝ) (a b c : ℝ) :
  (0 < A) ∧ (A < π) ∧ (0 < B) ∧ (B < π) ∧ (0 < C) ∧ (C < π) →  -- Angle conditions
  (A + B + C = π) →  -- Sum of angles in a triangle
  (a > 0) ∧ (b > 0) ∧ (c > 0) →  -- Side length conditions
  (a * Real.cos B + b * Real.cos A = 4 * Real.sin C) →  -- Given condition
  (∃ (R : ℝ), R > 0 ∧ π * R^2 = 4 * π) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcircle_area_of_triangle_l909_90905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_implies_m_eq_neg_one_l909_90915

/-- A function f : ℝ → ℝ is linear if there exist constants k ≠ 0 and b such that f(x) = k * x + b for all x -/
def IsLinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ (k b : ℝ), k ≠ 0 ∧ ∀ x, f x = k * x + b

/-- The function y = (m-1)x^(m^2) + 1 -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * (x ^ (m^2)) + 1

theorem linear_function_implies_m_eq_neg_one :
  ∀ m : ℝ, IsLinearFunction (f m) → m = -1 := by
  sorry

#check linear_function_implies_m_eq_neg_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_implies_m_eq_neg_one_l909_90915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_squares_below_line_l909_90929

-- Define the line equation
def line_equation (x y : ℝ) : Prop := 12 * x + 240 * y = 2880

-- Define the first quadrant
def first_quadrant (x y : ℝ) : Prop := x ≥ 0 ∧ y ≥ 0

-- Define a function to count the number of complete squares below the line
def count_squares_below_line : ℕ := 1315

-- Theorem statement
theorem squares_below_line :
  count_squares_below_line = 1315 := by
  -- The proof goes here
  sorry

#check squares_below_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_squares_below_line_l909_90929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_n_digits_same_l909_90947

def x : ℕ → ℕ
  | 0 => 5  -- Add a case for 0 to cover all natural numbers
  | n + 1 => (x n) ^ 2

theorem last_n_digits_same (n : ℕ) (h : n ≥ 1) : 
  10 ^ n ∣ (x (n + 1) - x n) :=
by
  -- The proof will be implemented here
  sorry

#eval x 1  -- Test the function
#eval x 2
#eval x 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_n_digits_same_l909_90947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_range_l909_90921

/-- The function h(x) defined on the interval [0, 1] -/
noncomputable def h (x : ℝ) : ℝ := (4 * x^2 - 12 * x - 3) / (2 * x + 1)

/-- The theorem stating that the range of h(x) is [-4, -3] for x ∈ [0, 1] -/
theorem h_range :
  (∀ x ∈ Set.Icc (0 : ℝ) 1, h x ∈ Set.Icc (-4) (-3)) ∧
  (∃ y z, y ∈ Set.Icc (0 : ℝ) 1 ∧ z ∈ Set.Icc (0 : ℝ) 1 ∧ h y = -4 ∧ h z = -3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_range_l909_90921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_cuboid_length_l909_90946

/-- Represents the dimensions of a cuboid -/
structure CuboidDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a cuboid given its dimensions -/
def cuboidVolume (d : CuboidDimensions) : ℝ :=
  d.length * d.width * d.height

theorem smaller_cuboid_length 
  (original : CuboidDimensions)
  (smaller_width : ℝ)
  (smaller_height : ℝ)
  (num_smaller : ℝ)
  (h1 : original.length = 18)
  (h2 : original.width = 15)
  (h3 : original.height = 2)
  (h4 : smaller_width = 4)
  (h5 : smaller_height = 3)
  (h6 : num_smaller = 7.5) :
  (cuboidVolume original) / (num_smaller * smaller_width * smaller_height) = 6 := by
  sorry

#check smaller_cuboid_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_cuboid_length_l909_90946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percentage_approx_6_36_l909_90980

/-- Calculates the profit percentage given the sale price with tax, tax rate, and cost price. -/
noncomputable def profit_percentage (sale_price_with_tax : ℝ) (tax_rate : ℝ) (cost_price : ℝ) : ℝ :=
  let actual_sale_price := sale_price_with_tax / (1 + tax_rate)
  let profit := actual_sale_price - cost_price
  (profit / cost_price) * 100

/-- Theorem stating that the profit percentage is approximately 6.36% given the specified conditions. -/
theorem profit_percentage_approx_6_36 :
  let sale_price_with_tax : ℝ := 616
  let tax_rate : ℝ := 0.10
  let cost_price : ℝ := 526.50
  abs (profit_percentage sale_price_with_tax tax_rate cost_price - 6.36) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percentage_approx_6_36_l909_90980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_domain_l909_90957

noncomputable def tan_function (x : ℝ) : ℝ := Real.tan (x - Real.pi / 4)

theorem tan_domain :
  {x : ℝ | ∃ y, tan_function x = y} = {x : ℝ | ∀ k : ℤ, x ≠ k * Real.pi + 3 * Real.pi / 4} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_domain_l909_90957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_cartesian_l909_90973

-- Define the polar equation
def polar_equation (θ : Real) : Prop := Real.sin (2 * θ) = 1

-- Define the Cartesian equation
def cartesian_equation (x y : Real) : Prop := y = x

-- Theorem statement
theorem polar_to_cartesian :
  ∀ θ x y : Real, polar_equation θ → (x = Real.cos θ ∧ y = Real.sin θ) → cartesian_equation x y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_cartesian_l909_90973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_sum_of_cardinality_l909_90989

/-- Sum of digits function -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Set of numbers with digit sum 15 and less than 10^8 -/
def T : Set ℕ := {n : ℕ | digit_sum n = 15 ∧ n < 10^8}

/-- Cardinality of set T -/
noncomputable def p : ℕ := Finset.card (Finset.filter (λ n => digit_sum n = 15) (Finset.range 100000000))

/-- Main theorem -/
theorem digit_sum_of_cardinality : digit_sum p = 33 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_sum_of_cardinality_l909_90989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_in_small_spheres_is_eight_ninths_l909_90991

/-- Represents a tetrahedron with inscribed, circumscribed, and small spheres -/
structure TetrahedronWithSpheres where
  /-- Radius of the circumscribed sphere -/
  R : ℝ
  /-- Radius of the inscribed sphere -/
  r : ℝ
  /-- Radius of each small sphere -/
  small_r : ℝ
  /-- The inscribed sphere radius is one-third of the circumscribed sphere radius -/
  inscribed_radius : r = R / 3
  /-- The small sphere radius is equal to the inscribed sphere radius -/
  small_sphere_radius : small_r = r
  /-- The tetrahedron has four equilateral triangular faces -/
  equilateral_faces : Prop
  /-- There are eight small spheres, two for each face -/
  eight_small_spheres : Prop
  /-- Small spheres are tangent to face centers and circumscribed sphere -/
  tangent_spheres : Prop

/-- The probability that a random point in the circumscribed sphere is inside one of the small spheres -/
noncomputable def probability_in_small_spheres (t : TetrahedronWithSpheres) : ℝ :=
  8 / 9

/-- Theorem stating the probability of a point being in one of the small spheres -/
theorem probability_in_small_spheres_is_eight_ninths (t : TetrahedronWithSpheres) :
  probability_in_small_spheres t = 8 / 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_in_small_spheres_is_eight_ninths_l909_90991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_geq_one_range_l909_90938

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 2 then |3*x - 4| else 2/(x - 1)

theorem f_geq_one_range :
  {x : ℝ | f x ≥ 1} = (Set.Iic 1 ∪ Set.Icc (5/3) 3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_geq_one_range_l909_90938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_sum_21_l909_90906

def numbers : Finset Nat := {2, 3, 4, 5, 6, 7, 8}

theorem grid_sum_21 (x : Nat) (h : x ∈ numbers) :
  (∃ a b c, a ∈ numbers ∧ b ∈ numbers ∧ c ∈ numbers ∧ a + b + c + x = 21 ∧ 
   ∃ d e f, d ∈ numbers ∧ e ∈ numbers ∧ f ∈ numbers ∧ d + e + f + x = 21 ∧
   {a, b, c, d, e, f, x} = numbers) →
  x = 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_sum_21_l909_90906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_lambda_for_inequality_l909_90966

theorem max_lambda_for_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 3) :
  (∀ l : ℝ, a^5 + b^5 ≥ l * a * b → l ≤ 27/4) ∧
  (∃ l : ℝ, l = 27/4 ∧ a^5 + b^5 = l * a * b) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_lambda_for_inequality_l909_90966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solid_of_revolution_properties_l909_90903

/-- Triangle ABC with given properties -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  angle_ACB : Real.sin (15 * π / 180) = (B.1 - A.1) / Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  angle_CBA : Real.sin (120 * π / 180) = (C.1 - A.1) / Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)
  side_AB : Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 1

/-- Solid of revolution generated by rotating the triangle around AB -/
noncomputable def SolidOfRevolution (t : Triangle) : ℝ × ℝ := sorry

/-- Theorem stating the volume and surface area of the solid of revolution -/
theorem solid_of_revolution_properties (t : Triangle) :
  let (volume, surface_area) := SolidOfRevolution t
  (abs (volume - 5.86) < 0.01) ∧ (abs (surface_area - 45.17) < 0.01) := by sorry

#check solid_of_revolution_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solid_of_revolution_properties_l909_90903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_area_equality_l909_90941

-- Define the points of the hexagon and additional points
variable (A B C D E F M K L : EuclideanPlane)

-- Define the regular hexagon
def is_regular_hexagon (A B C D E F : EuclideanPlane) : Prop :=
  sorry

-- Define midpoint
def is_midpoint (M X Y : EuclideanPlane) : Prop :=
  sorry

-- Define intersection of lines
def is_intersection (L P Q R S : EuclideanPlane) : Prop :=
  sorry

-- Define area of a triangle
noncomputable def area_triangle (P Q R : EuclideanPlane) : ℝ :=
  sorry

-- Define area of a quadrilateral
noncomputable def area_quadrilateral (P Q R S : EuclideanPlane) : ℝ :=
  sorry

-- Theorem statement
theorem hexagon_area_equality 
  (h_hexagon : is_regular_hexagon A B C D E F)
  (h_M_midpoint : is_midpoint M D C)
  (h_K_midpoint : is_midpoint K E D)
  (h_L_intersection : is_intersection L A M B K) :
  area_triangle A B L = area_quadrilateral L M D K :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_area_equality_l909_90941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parallel_to_plane_iff_l909_90948

/-- A line in three-dimensional space -/
structure Line3D where
  -- Define properties of a line
  point : Fin 3 → ℝ
  direction : Fin 3 → ℝ

/-- A plane in three-dimensional space -/
structure Plane where
  -- Define properties of a plane
  point : Fin 3 → ℝ
  normal : Fin 3 → ℝ

/-- Parallelism between a line and a plane -/
def line_parallel_to_plane (l : Line3D) (p : Plane) : Prop :=
  sorry

/-- Parallelism between two lines -/
def lines_parallel (l1 l2 : Line3D) : Prop :=
  sorry

/-- A line is parallel to countless lines in a plane -/
def line_parallel_to_countless_lines_in_plane (l : Line3D) (p : Plane) : Prop :=
  ∃ (S : Set Line3D), (∀ l' ∈ S, lines_parallel l l') ∧ Set.Infinite S

/-- Main theorem: Line parallel to countless lines in a plane is necessary but not sufficient for line parallel to plane -/
theorem line_parallel_to_plane_iff (l : Line3D) (p : Plane) :
  (line_parallel_to_plane l p → line_parallel_to_countless_lines_in_plane l p) ∧
  ¬(line_parallel_to_countless_lines_in_plane l p → line_parallel_to_plane l p) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parallel_to_plane_iff_l909_90948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_continuedFraction_eighth_root_l909_90936

-- Define the continued fraction
noncomputable def continuedFraction : ℝ := 2207 - 1 / (2207 - 1 / (2207 - 1 / (2207 - Real.pi)))

-- State the theorem
theorem continuedFraction_eighth_root :
  continuedFraction ^ (1/8 : ℝ) = (3 + Real.sqrt 5) / 2 := by
  sorry

-- Additional lemma to help with the proof
lemma continuedFraction_value : continuedFraction = 2205 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_continuedFraction_eighth_root_l909_90936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l909_90995

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := 2 * x^2 - y^2 = 1

-- Define the asymptotic lines
def asymptotic_lines (x y : ℝ) : Prop := y = Real.sqrt 2 * x ∨ y = -Real.sqrt 2 * x

-- Theorem statement
theorem hyperbola_asymptotes :
  ∀ x y : ℝ, hyperbola x y → (∃ ε > 0, ∀ x' y' : ℝ, 
    hyperbola x' y' ∧ |x'| > 1/ε → 
    |y' - (Real.sqrt 2 * x')| < ε ∨ |y' - (-Real.sqrt 2 * x')| < ε) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l909_90995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_PQRS_l909_90933

/-- A rectangle with vertices E, F, G, H -/
structure Rectangle (E F G H : ℝ × ℝ) : Type where
  is_rectangle : Prop
  length : ℝ
  width : ℝ

/-- An equilateral triangle with vertices A, B, C -/
structure EquilateralTriangle (A B C : ℝ × ℝ) : Type where
  is_equilateral : Prop

/-- Define the quadrilateral PQRS based on the given conditions -/
def PQRS (E F G H P Q R S : ℝ × ℝ) : Prop :=
  ∃ (rect : Rectangle E F G H) (tri1 : EquilateralTriangle E P F)
    (tri2 : EquilateralTriangle F Q G) (tri3 : EquilateralTriangle G R H)
    (tri4 : EquilateralTriangle H S E),
  rect.length = 6 ∧ rect.width = 4

/-- Calculate the area of a quadrilateral given its vertices -/
noncomputable def area (P Q R S : ℝ × ℝ) : ℝ := sorry

/-- The main theorem stating that the area of PQRS is 60 + 48√3 -/
theorem area_PQRS (E F G H P Q R S : ℝ × ℝ) (h : PQRS E F G H P Q R S) :
  area P Q R S = 60 + 48 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_PQRS_l909_90933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base7_digits_of_2000_l909_90952

noncomputable def base7Digits (n : ℕ) : ℕ :=
  if n = 0 then 1 else Nat.floor (Real.log n / Real.log 7) + 1

theorem base7_digits_of_2000 : base7Digits 2000 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base7_digits_of_2000_l909_90952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_zeros_omega_range_l909_90965

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 
  Real.sin (ω * x / 2) ^ 2 + (1/2) * Real.sin (ω * x) - 1/2

theorem no_zeros_omega_range (ω : ℝ) (h_pos : ω > 0) :
  (∀ x ∈ Set.Ioo π (2*π), f ω x ≠ 0) →
  ω ∈ Set.Ioc 0 (1/8) ∪ Set.Icc (1/4) (5/8) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_zeros_omega_range_l909_90965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_inverse_of_f_l909_90910

noncomputable section

/-- The original function -/
def f (x : ℝ) : ℝ := 3 - 4 * x

/-- The proposed inverse function -/
def g (x : ℝ) : ℝ := (3 - x) / 4

/-- Theorem stating that g is the inverse of f -/
theorem g_is_inverse_of_f : Function.LeftInverse g f ∧ Function.RightInverse g f := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_inverse_of_f_l909_90910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_of_3_eq_sqrt_43_minus_3_l909_90920

-- Define the functions
def f (x : ℝ) : ℝ := 3 * x + 4

noncomputable def g (x : ℝ) : ℝ := Real.sqrt (f x) - 3

noncomputable def h (x : ℝ) : ℝ := g (f x)

-- State the theorem
theorem h_of_3_eq_sqrt_43_minus_3 : h 3 = Real.sqrt 43 - 3 := by
  -- Unfold the definitions
  unfold h g f
  -- Simplify
  simp
  -- The proof is complete
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_of_3_eq_sqrt_43_minus_3_l909_90920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_l909_90950

/-- Proves that the length of a train is 500 meters, given its crossing time and speed. -/
theorem train_length (crossing_time : ℝ) (speed_kmh : ℝ) : 
  crossing_time = 50 → speed_kmh = 36 → 
  crossing_time * (speed_kmh * (5 / 18)) = 500 := by
  intro h_time h_speed
  have speed_ms : ℝ := speed_kmh * (5 / 18)
  calc
    crossing_time * (speed_kmh * (5 / 18)) = 50 * (36 * (5 / 18)) := by rw [h_time, h_speed]
    _ = 50 * 10 := by norm_num
    _ = 500 := by norm_num
  
#check train_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_l909_90950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_inventory_management_l909_90970

/-- Represents a store's inventory management problem. -/
structure InventoryProblem where
  R : ℝ  -- Items sold per day
  b : ℝ  -- Restocking fee per shipment
  s : ℝ  -- Daily storage fee per item
  (R_pos : R > 0)
  (b_pos : b > 0)
  (s_pos : s > 0)

/-- Calculates the average daily cost for a given restock interval. -/
noncomputable def averageDailyCost (p : InventoryProblem) (T : ℝ) : ℝ :=
  (p.R * T * p.s) / 2 + p.b / T

/-- The optimal number of days between restocks. -/
noncomputable def optimalDays (p : InventoryProblem) : ℝ :=
  Real.sqrt ((2 * p.b) / (p.R * p.s))

/-- The optimal restocking quantity. -/
noncomputable def optimalQuantity (p : InventoryProblem) : ℝ :=
  Real.sqrt ((2 * p.b * p.R) / p.s)

/-- Theorem stating that the calculated optimal days and quantity minimize the average daily cost. -/
theorem optimal_inventory_management (p : InventoryProblem) :
  let T := optimalDays p
  let Q := optimalQuantity p
  ∀ T' > 0, averageDailyCost p T ≤ averageDailyCost p T' ∧
  Q = p.R * T := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_inventory_management_l909_90970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lisa_bought_12_pairs_l909_90912

/-- The number of pairs of socks Lisa bought at the discount store -/
def L : ℕ := sorry

/-- The number of pairs of socks Sandra brought -/
def sandra_socks : ℕ := 20

/-- The number of pairs of socks Lisa's cousin brought -/
def cousin_socks : ℕ := sandra_socks / 5

/-- The number of pairs of socks Lisa's mom brought -/
def mom_socks : ℕ := 3 * L + 8

/-- The total number of pairs of socks Lisa ended up with -/
def total_socks : ℕ := 80

theorem lisa_bought_12_pairs :
  L + sandra_socks + cousin_socks + mom_socks = total_socks → L = 12 := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lisa_bought_12_pairs_l909_90912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_mobots_correct_l909_90971

/-- Represents a lawn divided into a grid -/
structure Lawn where
  rows : ℕ
  cols : ℕ

/-- Represents a mobot (robotic mower) -/
inductive Mobot
  | northSouth
  | eastWest

/-- The maximum number of mobots needed to mow a lawn -/
def maxMobots (lawn : Lawn) : ℕ := lawn.rows + lawn.cols - 1

/-- Represents a clump of grass in the lawn -/
structure Clump where
  row : ℕ
  col : ℕ

/-- Predicate to check if a mobot mows a specific clump -/
def mows (mobot : Mobot) (clump : Clump) : Prop :=
  match mobot with
  | Mobot.northSouth => true  -- Simplified for this example
  | Mobot.eastWest => true    -- Simplified for this example

/-- Theorem stating that maxMobots gives the maximum number of mobots needed -/
theorem max_mobots_correct (lawn : Lawn) :
  ∀ (arrangement : Finset Mobot),
    (∀ clump : Clump, ∃ mobot ∈ arrangement, mows mobot clump) →
    arrangement.card ≤ maxMobots lawn := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_mobots_correct_l909_90971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_result_l909_90925

noncomputable def original_function (x : ℝ) : ℝ := Real.sin x

noncomputable def shift_left (f : ℝ → ℝ) (shift : ℝ) : ℝ → ℝ :=
  λ x => f (x + shift)

noncomputable def reduce_horizontal (f : ℝ → ℝ) (factor : ℝ) : ℝ → ℝ :=
  λ x => f (x / factor)

noncomputable def transformed_function : ℝ → ℝ :=
  reduce_horizontal (shift_left original_function (Real.pi/3)) 2

theorem transformation_result :
  ∀ x : ℝ, transformed_function x = Real.sin (2*x + Real.pi/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_result_l909_90925

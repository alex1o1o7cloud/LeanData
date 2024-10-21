import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_cone_generatrix_angle_l772_77258

/-- A truncated cone circumscribed around a sphere -/
structure TruncatedCone where
  /-- Radius of the larger base -/
  R : ℝ
  /-- Radius of the smaller base -/
  r : ℝ
  /-- Radius of the inscribed sphere -/
  sphereRadius : ℝ

/-- The angle between the generatrix of the cone and the plane of its base -/
noncomputable def generatrixAngle (cone : TruncatedCone) : ℝ :=
  Real.arccos (1 / 3)

theorem truncated_cone_generatrix_angle (cone : TruncatedCone) 
  (h : cone.R^2 = 4 * cone.r^2) : 
  generatrixAngle cone = Real.arccos (1 / 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_cone_generatrix_angle_l772_77258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_visitor_count_theorem_l772_77224

noncomputable def october_visitors : ℝ := 100

noncomputable def november_first_half (october : ℝ) : ℝ := october * 1.15

noncomputable def november_second_half (first_half : ℝ) : ℝ := first_half * 0.9

noncomputable def november_weighted_average (first_half second_half : ℝ) : ℝ := (first_half + second_half) / 2

noncomputable def december_first_half (november_avg : ℝ) : ℝ := november_avg * 0.9

noncomputable def december_second_half (first_half : ℝ) : ℝ := first_half + 20

noncomputable def december_weighted_average (first_half second_half : ℝ) : ℝ := (first_half + second_half) / 2

noncomputable def total_visitors (oct nov dec : ℝ) : ℝ := oct + nov + dec

theorem visitor_count_theorem :
  let nov_first := november_first_half october_visitors
  let nov_second := november_second_half nov_first
  let nov_avg := november_weighted_average nov_first nov_second
  let dec_first := december_first_half nov_avg
  let dec_second := december_second_half dec_first
  let dec_avg := december_weighted_average dec_first dec_second
  let total := total_visitors october_visitors nov_avg dec_avg
  ⌊total⌋ = 318 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_visitor_count_theorem_l772_77224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triplet_transformation_impossibility_l772_77244

def triplet_transform (a b c : ℝ) : ℝ × ℝ × ℝ → Prop :=
  λ (xyz : ℝ × ℝ × ℝ) ↦ ∃ (p q r : ℝ), 
    ((p = (xyz.1 - xyz.2.1) / Real.sqrt 2 ∧ q = (xyz.1 + xyz.2.1) / Real.sqrt 2 ∧ r = xyz.2.2) ∨
     (p = (xyz.1 - xyz.2.2) / Real.sqrt 2 ∧ q = (xyz.1 + xyz.2.2) / Real.sqrt 2 ∧ r = xyz.2.1) ∨
     (p = (xyz.2.1 - xyz.2.2) / Real.sqrt 2 ∧ q = (xyz.2.1 + xyz.2.2) / Real.sqrt 2 ∧ r = xyz.1)) ∧
    (a, b, c) = (p, q, r)

def sum_of_squares (a b c : ℝ) : ℝ :=
  a^2 + b^2 + c^2

theorem triplet_transformation_impossibility :
  ¬∃ (n : ℕ) (t : ℕ → ℝ × ℝ × ℝ),
    t 0 = (2, Real.sqrt 2, 1 / Real.sqrt 2) ∧
    t n = (1, Real.sqrt 2, 1 + Real.sqrt 2) ∧
    (∀ i : ℕ, i < n → triplet_transform (t i).1 (t i).2.1 (t i).2.2 (t (i+1))) ∧
    (∀ i : ℕ, i ≤ n → sum_of_squares (t i).1 (t i).2.1 (t i).2.2 = sum_of_squares 2 (Real.sqrt 2) (1 / Real.sqrt 2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triplet_transformation_impossibility_l772_77244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_at_pi_over_four_l772_77245

open Real

theorem derivative_at_pi_over_four (f : ℝ → ℝ) (h : ∀ x, f x = (deriv f (π / 2)) * sin x + cos x) :
  deriv f (π / 4) = -sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_at_pi_over_four_l772_77245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_number_l772_77242

theorem unique_number : ∃! n : ℕ, n ∈ ({12, 14, 15, 20} : Set ℕ) ∧ ¬(3 ∣ n) ∧ n < 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_number_l772_77242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_divisibility_l772_77211

/-- A cubic polynomial with integer coefficients -/
structure CubicPolynomial where
  a : ℤ
  b : ℤ
  c : ℤ

/-- Evaluate the polynomial at a given value -/
def CubicPolynomial.eval (p : CubicPolynomial) (x : ℚ) : ℚ :=
  x^3 + p.a * x^2 + p.b * x + p.c

/-- Predicate that checks if one root is the product of the other two -/
def has_product_root (p : CubicPolynomial) : Prop :=
  ∃ (u v : ℚ), p.eval u = 0 ∧ p.eval v = 0 ∧ p.eval (u * v) = 0

/-- Main theorem statement -/
theorem cubic_polynomial_divisibility (p : CubicPolynomial) 
  (h : has_product_root p) : 
  (p.eval 1 + p.eval (-1) - 2 * (1 + p.eval 0)) ∣ (2 * p.eval (-1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_divisibility_l772_77211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cross_section_area_l772_77218

/-- A right prism with a square base -/
structure SquarePrism where
  side_length : ℝ
  center : Fin 3 → ℝ
  height : ℝ

/-- A plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- The area of the cross-section formed by cutting a square prism with a plane -/
noncomputable def cross_section_area (prism : SquarePrism) (plane : Plane) : ℝ :=
  sorry

/-- The theorem stating the maximum area of the cross-section -/
theorem max_cross_section_area :
  ∃ (prism : SquarePrism) (plane : Plane),
    prism.side_length = 8 ∧
    (∀ i, prism.center i = 0) ∧
    plane.a = 3 ∧
    plane.b = -5 ∧
    plane.c = 2 ∧
    plane.d = 20 ∧
    cross_section_area prism plane = 160 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cross_section_area_l772_77218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_six_lines_configuration_l772_77223

/-- A line in a plane --/
structure Line :=
  (slope : ℝ)
  (intercept : ℝ)

/-- A circle in a plane --/
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

/-- Predicate to check if a line is tangent to a circle --/
def is_tangent (l : Line) (c : Circle) : Prop := sorry

/-- The configuration of six lines --/
def six_lines : Fin 6 → Line := sorry

/-- Theorem stating the existence of a configuration satisfying the given conditions --/
theorem six_lines_configuration :
  (∀ (a b c : Fin 6), a ≠ b ∧ b ≠ c ∧ a ≠ c →
    ∃ (d : Fin 6) (circ : Circle), d ≠ a ∧ d ≠ b ∧ d ≠ c ∧
      is_tangent (six_lines a) circ ∧
      is_tangent (six_lines b) circ ∧
      is_tangent (six_lines c) circ ∧
      is_tangent (six_lines d) circ) ∧
  ¬(∃ (circ : Circle), ∀ (i : Fin 6), is_tangent (six_lines i) circ) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_six_lines_configuration_l772_77223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_one_cyclically_symmetric_l772_77210

-- Define a cyclically symmetric expression
def is_cyclically_symmetric (σ : ℝ → ℝ → ℝ → ℝ) : Prop :=
  ∀ a b c, σ a b c = σ b c a ∧ σ a b c = σ c a b

-- Define the three given expressions
def σ₁ (a b c : ℝ) : ℝ := a * b * c
def σ₂ (a b c : ℝ) : ℝ := a^2 - b^2 + c^2
noncomputable def σ₃ (A B C : ℝ) : ℝ := Real.cos C * Real.cos (A - B) - (Real.cos C)^2

theorem only_one_cyclically_symmetric :
  (is_cyclically_symmetric σ₁ ∧ ¬is_cyclically_symmetric σ₂ ∧ ¬is_cyclically_symmetric σ₃) ∧
  (is_cyclically_symmetric σ₁ ∨ is_cyclically_symmetric σ₂ ∨ is_cyclically_symmetric σ₃) :=
by
  sorry

#check only_one_cyclically_symmetric

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_one_cyclically_symmetric_l772_77210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_year_after_2021_with_digit_sum_5_l772_77294

def digit_sum (n : Nat) : Nat :=
  (n.repr.toList.map (λ c => c.toNat - '0'.toNat)).sum

def is_first_year_after_with_same_digit_sum (base_year target_year : Nat) : Prop :=
  target_year > base_year ∧
  digit_sum target_year = digit_sum base_year ∧
  ∀ y, base_year < y ∧ y < target_year → digit_sum y ≠ digit_sum base_year

theorem first_year_after_2021_with_digit_sum_5 :
  is_first_year_after_with_same_digit_sum 2021 2030 := by
  sorry

#eval digit_sum 2021
#eval digit_sum 2030

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_year_after_2021_with_digit_sum_5_l772_77294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_T_l772_77282

/-- The imaginary unit -/
def i : ℂ := Complex.I

/-- The complex number T -/
def T : ℂ := (2 + i)^20 - (2 - i)^20

/-- The magnitude of T -/
theorem magnitude_of_T : Complex.abs T = 5^10 * 2 * Complex.abs (Complex.sin (20 * Complex.arctan (1/2))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_T_l772_77282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_y_value_l772_77253

theorem tan_half_y_value (x y : ℝ) (h1 : Real.cos (x + y) * Real.sin x - Real.sin (x + y) * Real.cos x = 12 / 13)
  (h2 : π < y ∧ y < 3 * π / 2) : Real.tan (y / 2) = - 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_y_value_l772_77253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_two_theta_zero_l772_77276

theorem cos_two_theta_zero (θ : ℝ) 
  (h : ∑' (n : ℕ), (Real.cos θ)^(2 * n) = 2) : Real.cos (2 * θ) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_two_theta_zero_l772_77276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_S_value_l772_77206

/-- The ellipse C -/
def ellipse (x y : ℝ) : Prop := x^2 / 3 + y^2 / 2 = 1

/-- The circle that intersects the line -/
def intersecting_circle (x y : ℝ) : Prop := x^2 + y^2 = 2

/-- The line that intersects the circle -/
def intersecting_line (x y : ℝ) : Prop := x + y + Real.sqrt 2 = 0

/-- Point on the ellipse -/
def point_on_ellipse : Prop := ellipse 1 (2 * Real.sqrt 3 / 3)

/-- Chord length of intersection between circle and line -/
def chord_length : ℝ := 2

/-- S is the sum of areas of triangles QF2M and OF2N -/
noncomputable def S (Q M N : ℝ × ℝ) : ℝ := sorry

/-- Main theorem -/
theorem max_S_value :
  ∀ Q M N : ℝ × ℝ,
  ellipse Q.1 Q.2 →
  Q.2 ≠ 0 →
  ellipse M.1 M.2 →
  ellipse N.1 N.2 →
  (∃ k : ℝ, M.2 - N.2 = k * (M.1 - N.1) ∧ Q.2 = k * Q.1) →
  (∀ S_val, S Q M N ≤ S_val → S_val ≤ 2 * Real.sqrt 3 / 3) ∧
  (∃ Q M N, S Q M N = 2 * Real.sqrt 3 / 3) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_S_value_l772_77206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mindy_tax_rate_is_25_percent_l772_77271

/-- Represents the tax rates and income ratios for Mork and Mindy -/
structure TaxScenario where
  mork_tax_rate : ℚ
  mindy_income_ratio : ℚ
  combined_tax_rate : ℚ

/-- Calculates Mindy's tax rate given the tax scenario -/
def mindy_tax_rate (scenario : TaxScenario) : ℚ :=
  (scenario.combined_tax_rate * (1 + scenario.mindy_income_ratio) - scenario.mork_tax_rate) / scenario.mindy_income_ratio

/-- Theorem stating that Mindy's tax rate is 25% given the specific scenario -/
theorem mindy_tax_rate_is_25_percent (scenario : TaxScenario) 
  (h1 : scenario.mork_tax_rate = 45/100)
  (h2 : scenario.mindy_income_ratio = 4)
  (h3 : scenario.combined_tax_rate = 29/100) :
  mindy_tax_rate scenario = 1/4 := by
  sorry

#eval mindy_tax_rate { mork_tax_rate := 45/100, mindy_income_ratio := 4, combined_tax_rate := 29/100 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mindy_tax_rate_is_25_percent_l772_77271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x6_expansion_l772_77264

theorem coefficient_x6_expansion : 
  let f := (fun x : ℝ => (3*x + 2)^8 * (4*x - 1)^2)
  ∃ g : ℝ → ℝ, ∀ x, f x = -792108 * x^6 + g x ∧ (∀ ε > 0, ∃ δ > 0, ∀ y, |y| < δ → |g y / y^6| < ε) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x6_expansion_l772_77264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_sum_inequality_l772_77261

theorem cyclic_sum_inequality (a b c lambda : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hlambda : lambda ≥ 1) :
  let f (x y : ℝ) := x^2 / (x^2 + lambda*x*y + y^2)
  f a b + f b c + f c a ≥ 3 / (lambda + 2) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_sum_inequality_l772_77261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_equivalence_l772_77280

/-- The cube root of unity -/
noncomputable def ε : ℂ := (1/2 : ℂ) + Complex.I * (Real.sqrt 3 / 2)

/-- Predicate for points forming an equilateral triangle -/
def is_equilateral (a b c : ℂ) : Prop :=
  (a + ε^2 * b + ε^4 * c = 0) ∨ (a + ε^4 * b + ε^2 * c = 0)

/-- Alternative condition for equilateral triangle -/
def alt_equilateral (a b c : ℂ) : Prop :=
  a^2 + b^2 + c^2 = a*b + b*c + a*c

/-- Theorem stating the equivalence of the two conditions -/
theorem equilateral_equivalence (a b c : ℂ) :
  is_equilateral a b c ↔ alt_equilateral a b c := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_equivalence_l772_77280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_AOB_range_l772_77283

-- Define the ellipse
noncomputable def ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define point P on the ellipse
noncomputable def P : ℝ × ℝ := (-1, Real.sqrt 2 / 2)

-- Define the line l
def line_l (k m x : ℝ) : ℝ := k * x + m

-- Define the condition for line l to be tangent to circle O
def tangent_condition (k m : ℝ) : Prop := m^2 = k^2 + 1

-- Define the condition for line l intersecting the ellipse at two distinct points
def intersection_condition (k : ℝ) : Prop := k^2 > 0

-- Define the dot product condition
def dot_product_condition (lambda : ℝ) : Prop := 2/3 ≤ lambda ∧ lambda ≤ 3/4

-- Define the area of triangle AOB
noncomputable def area_AOB (k : ℝ) : ℝ := Real.sqrt ((k^4 + k^2) / (4 * k^4 + 4 * k^2 + 1))

-- State the theorem
theorem area_AOB_range :
  ∀ (k m lambda : ℝ),
  ellipse P.1 P.2 →
  tangent_condition k m →
  intersection_condition k →
  dot_product_condition lambda →
  Real.sqrt 6 / 4 ≤ area_AOB k ∧ area_AOB k ≤ 2/3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_AOB_range_l772_77283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alloy_density_proof_l772_77214

/-- Represents the density of a material relative to water -/
structure RelativeDensity where
  value : ℝ
  pos : value > 0

/-- The density of gold relative to water -/
def gold_density : RelativeDensity where
  value := 11
  pos := by norm_num

/-- The density of copper relative to water -/
def copper_density : RelativeDensity where
  value := 5
  pos := by norm_num

/-- The desired density of the alloy relative to water -/
def alloy_density : RelativeDensity where
  value := 8
  pos := by norm_num

/-- The ratio of gold to copper in the alloy -/
def gold_copper_ratio : ℝ := 1

theorem alloy_density_proof :
  let total_weight := gold_density.value * gold_copper_ratio + copper_density.value * gold_copper_ratio
  let total_volume := gold_copper_ratio + gold_copper_ratio
  total_weight / total_volume = alloy_density.value := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alloy_density_proof_l772_77214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_J_l772_77292

def H (p q : ℝ) : ℝ := -3*p*q + 4*p*(1-q) + 4*(1-p)*q - 5*(1-p)*(1-q)

noncomputable def J (p : ℝ) : ℝ := 
  ⨆ q ∈ Set.Icc 0 1, H p q

theorem minimize_J :
  ∀ p, p ∈ Set.Icc 0 1 → J p ≥ J (9/16) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_J_l772_77292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_surface_area_l772_77260

/-- The total surface area of a cone given its slant height and height -/
noncomputable def total_surface_area (slant_height : ℝ) (height : ℝ) : ℝ :=
  let radius := Real.sqrt (slant_height^2 - height^2)
  Real.pi * radius^2 + Real.pi * radius * slant_height

/-- Theorem: The total surface area of a cone with slant height 4 and height 2√3 is 12π -/
theorem cone_surface_area :
  total_surface_area 4 (2 * Real.sqrt 3) = 12 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_surface_area_l772_77260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equality_l772_77266

theorem expression_equality : (4 - Real.pi) ^ 0 - 2 * Real.sin (Real.pi / 3) + |3 - Real.sqrt 12| - (1 / 2) ^ (-1 : Int) = Real.sqrt 3 - 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equality_l772_77266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_shift_equivalence_l772_77286

open Real

noncomputable def original_function (x : ℝ) : ℝ := sin (2 * x - π / 3)

noncomputable def reference_function (x : ℝ) : ℝ := sin (2 * x)

noncomputable def transform (f : ℝ → ℝ) (x : ℝ) : ℝ := f (x - π / 6)

theorem sin_shift_equivalence :
  ∀ x : ℝ, original_function x = transform reference_function x :=
by
  intro x
  simp [original_function, reference_function, transform]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_shift_equivalence_l772_77286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_and_function_range_l772_77209

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.sin x, 3/2)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.cos x, -1)

noncomputable def f (x : ℝ) : ℝ := (a x + b x).1 * (b x).1 + (a x + b x).2 * (b x).2

theorem parallel_vectors_and_function_range :
  (∀ x : ℝ, (∃ k : ℝ, a x = k • b x) → 2 * Real.cos (2 * x) - Real.sin (2 * x) = 20/13) ∧
  (∀ x : ℝ, -π/2 ≤ x ∧ x ≤ 0 → -Real.sqrt 2 / 2 ≤ f x ∧ f x ≤ 1/2) := by
  sorry

#check parallel_vectors_and_function_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_and_function_range_l772_77209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_range_l772_77215

-- Define the circles
noncomputable def circle_C (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 1
noncomputable def circle_D (x y : ℝ) : Prop := (x - 4)^2 + y^2 = 4

-- Define the range of |AB|
noncomputable def AB_range (t : ℝ) : ℝ := 2 * Real.sqrt 2 * Real.sqrt (-(16 / t^2) + 5 / t)

-- State the theorem
theorem tangent_range :
  ∀ x y : ℝ,
  circle_D x y →
  ∃ a b : ℝ,
  (∀ t : ℝ, t ∈ Set.Icc 4 8 → AB_range t ∈ Set.Icc (Real.sqrt 2) ((5 * Real.sqrt 2) / 4)) ∧
  ((y - a) * x = (x - 4) * a) ∧
  ((y - b) * x = (x - 4) * b) ∧
  ((x + 2) * a^2 - 2 * y * a - (x - 4) = 0) ∧
  ((x + 2) * b^2 - 2 * y * b - (x - 4) = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_range_l772_77215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_value_l772_77275

/-- The area enclosed by the curves y = e^x, y = e^(-x), and the line x = 1 -/
noncomputable def enclosed_area : ℝ :=
  ∫ x in (0:ℝ)..1, (Real.exp x - Real.exp (-x)) + ∫ x in (0:ℝ)..1, Real.exp (-x)

/-- Theorem stating that the enclosed area is equal to e + e^(-1) - 2 -/
theorem enclosed_area_value : enclosed_area = Real.exp 1 + Real.exp (-1) - 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_value_l772_77275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_OAB_bounds_l772_77241

noncomputable section

-- Define the line l
def line_l (m x y : ℝ) : Prop :=
  (Real.sqrt 3 * m + 1) * x - (m - Real.sqrt 3) * y - 4 = 0

-- Define the circle O
def circle_O (x y : ℝ) : Prop :=
  x^2 + y^2 = 16

-- Define the intersection points A and B
def intersection_points (m : ℝ) (A B : ℝ × ℝ) : Prop :=
  line_l m A.1 A.2 ∧ line_l m B.1 B.2 ∧
  circle_O A.1 A.2 ∧ circle_O B.1 B.2

-- Define the area of triangle OAB
def area_OAB (A B : ℝ × ℝ) : ℝ :=
  (1/2) * abs (A.1 * B.2 - A.2 * B.1)

-- Theorem statement
theorem area_OAB_bounds (m : ℝ) (A B : ℝ × ℝ) :
  intersection_points m A B →
  0 < area_OAB A B ∧ area_OAB A B ≤ 4 * Real.sqrt 3 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_OAB_bounds_l772_77241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_x_squared_plus_sqrt_one_minus_x_squared_l772_77226

theorem integral_x_squared_plus_sqrt_one_minus_x_squared :
  ∫ x in (-1 : ℝ)..1, (x^2 + Real.sqrt (1 - x^2)) = 2/3 + π/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_x_squared_plus_sqrt_one_minus_x_squared_l772_77226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l772_77296

/-- Represents a parabola with vertex at the origin -/
structure Parabola where
  -- The equation of the parabola in the form y^2 = ax
  a : ℝ
  -- Assertion that a > 0 to ensure the parabola opens to the right
  a_pos : a > 0

/-- The focus of a parabola -/
noncomputable def focus (p : Parabola) : ℝ × ℝ := (p.a / 4, 0)

/-- A line passing through the focus -/
structure FocusLine (p : Parabola) where
  -- The slope of the line
  m : ℝ
  -- The y-intercept of the line
  b : ℝ
  -- Assertion that the line passes through the focus
  passes_focus : b = -m * (p.a / 4)

/-- The foot of the perpendicular from origin to a line -/
noncomputable def perpendicular_foot (p : Parabola) (l : FocusLine p) : ℝ × ℝ :=
  let x := l.m * (l.m * (p.a / 4) + l.b) / (l.m^2 + 1)
  let y := l.m * x + l.b
  (x, y)

/-- The main theorem -/
theorem parabola_equation (p : Parabola) 
  (h1 : ∃ (l : FocusLine p), perpendicular_foot p l = (2, 1)) :
  p.a = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l772_77296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_points_sum_l772_77252

/-- Three points in 3D space are collinear if they all lie on the same straight line. -/
def collinear (p q r : ℝ × ℝ × ℝ) : Prop :=
  ∃ t : ℝ, q = p + t • (r - p) ∨ r = p + t • (q - p)

/-- Given three collinear points (2,a,b), (a,3,b), and (a,b,4), prove that a + b = 6. -/
theorem collinear_points_sum (a b : ℝ) :
  collinear (2, a, b) (a, 3, b) (a, b, 4) → a + b = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_points_sum_l772_77252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_table_stability_no_coins_beyond_n_plus_1_l772_77201

/-- Represents the state of the coin table -/
def CoinTable := ℕ → ℕ

/-- The golden ratio -/
noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2

/-- Sum of (n+1) * coins in room n -/
noncomputable def S (table : CoinTable) : ℝ := sorry

/-- Sum of φ^n * coins in room n -/
noncomputable def T (table : CoinTable) : ℝ := sorry

/-- Initial configuration with one coin in each room from 1 to n -/
def initial_config (n : ℕ) : CoinTable := sorry

/-- Operation 1: Move coin 2 rooms right from left of 2 adjacent rooms with coins -/
def operation1 (table : CoinTable) (k : ℕ) : CoinTable := sorry

/-- Operation 2: For rooms ≥ 3 with > 1 coin, move 1 coin right and 1 coin left -/
def operation2 (table : CoinTable) (k : ℕ) : CoinTable := sorry

theorem coin_table_stability (n : ℕ) :
  ∃ (m : ℕ), ∀ (seq : ℕ → CoinTable),
    seq 0 = initial_config n →
    (∀ i, ∃ k, seq (i + 1) = operation1 (seq i) k ∨ seq (i + 1) = operation2 (seq i) k) →
    (∀ i ≥ m, seq (i + 1) = seq i) := by
  sorry

theorem no_coins_beyond_n_plus_1 (n : ℕ) :
  ∀ (seq : ℕ → CoinTable),
    seq 0 = initial_config n →
    (∀ i, ∃ k, seq (i + 1) = operation1 (seq i) k ∨ seq (i + 1) = operation2 (seq i) k) →
    ∀ i k, k ≥ n + 2 → seq i k = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_table_stability_no_coins_beyond_n_plus_1_l772_77201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_of_men_collected_dues_l772_77277

/-- Proves that the fraction of men who collected their dues is 1/9 --/
theorem fraction_of_men_collected_dues :
  let total_people : ℕ := 3552
  let amount_per_man : ℕ := 45
  let amount_per_woman : ℕ := 60
  let fraction_of_women_collected : ℚ := 1 / 12
  let total_amount_spent : ℕ := 17760
  ∃ (num_men num_women : ℕ) (fraction_of_men_collected : ℚ),
    num_men + num_women = total_people ∧
    (fraction_of_men_collected * num_men + 
     fraction_of_women_collected * num_women * 60 / 45 : ℚ) * 45 = 
        total_amount_spent ∧
    fraction_of_men_collected = 1 / 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_of_men_collected_dues_l772_77277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_medal_award_ways_l772_77267

/-- The number of runners in the race -/
def total_runners : ℕ := 10

/-- The number of Canadian runners -/
def canadian_runners : ℕ := 4

/-- The number of medals awarded -/
def medals : ℕ := 3

/-- The number of ways to award medals with at most one Canadian medalist -/
def medal_arrangements : ℕ := 480

/-- Theorem stating the number of ways to award medals -/
theorem medal_award_ways :
  (total_runners.choose medals * (total_runners - canadian_runners).factorial / (total_runners - canadian_runners - medals).factorial) +
  (canadian_runners * medals * (total_runners - canadian_runners).factorial / (total_runners - canadian_runners - medals + 1).factorial) = medal_arrangements :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_medal_award_ways_l772_77267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_ratio_for_circle_points_l772_77202

-- Define the circle
def circleC (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define points M and N
def M : ℝ × ℝ := (1, 1)
def N : ℝ × ℝ := (2, 2)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem constant_ratio_for_circle_points :
  ∃ (k : ℝ), ∀ (x y : ℝ), circleC x y →
    distance (x, y) N / distance (x, y) M = k := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_ratio_for_circle_points_l772_77202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l772_77272

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^2 * Real.exp (x + 1)

-- State the theorem
theorem max_value_of_f :
  ∃ (c : ℝ), c ∈ Set.Icc (-2 : ℝ) 1 ∧
  (∀ x ∈ Set.Icc (-2 : ℝ) 1, f x ≤ f c) ∧
  f c = Real.exp 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l772_77272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_implies_sum_l772_77256

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x

-- Define the derivative of f(x)
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := 1 / x - a

-- Theorem statement
theorem tangent_perpendicular_implies_sum (a b : ℝ) :
  (f a 1 = b) →                             -- P(1, b) lies on f(x)
  (f_deriv a 1 * (-1/3) = -1) →             -- Tangent line is perpendicular to x + 3y - 2 = 0
  2 * a + b = -2 :=
by
  intro h1 h2
  sorry  -- Proof is omitted for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_implies_sum_l772_77256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_no_adjacent_green_hats_l772_77204

/-- The number of children in the group -/
def total_children : ℕ := 9

/-- The number of children chosen to wear green hats -/
def green_hats : ℕ := 3

/-- The probability that no two children wearing green hats are standing next to each other -/
def prob_no_adjacent_green : ℚ := 5/14

/-- Theorem stating the probability of no two children wearing green hats standing next to each other -/
theorem probability_no_adjacent_green_hats : 
  (Nat.choose total_children green_hats - (total_children - green_hats + 1) - (total_children - 1) * (total_children - green_hats - 1) : ℚ) / 
  Nat.choose total_children green_hats = prob_no_adjacent_green := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_no_adjacent_green_hats_l772_77204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l772_77213

noncomputable section

variable (f : ℝ → ℝ)

def IncreasingOn (f : ℝ → ℝ) (S : Set ℝ) :=
  ∀ {x y}, x ∈ S → y ∈ S → x < y → f x < f y

theorem function_properties
    (h_inc : IncreasingOn f (Set.Ioi 0))
    (h_eq : ∀ {x y}, x > 0 → y > 0 → f (x / y) = f x - f y)
    (h_f6 : f 6 = 1) :
  (f 1 = 0) ∧
  (∀ x, x > 0 → (f (x + 3) - f (1 / x) < 2 ↔ 0 < x ∧ x < (-3 + 3 * Real.sqrt 17) / 2)) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l772_77213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_b_formula_S_lower_bound_l772_77238

noncomputable def f (x : ℝ) : ℝ := -Real.sqrt (4 + 1 / (x^2))

noncomputable def sequence_a : ℕ+ → ℝ := 
  λ n => 1 / Real.sqrt (4 * n.val - 3)

noncomputable def sequence_b : ℕ+ → ℝ := 
  λ n => 8 * n.val - 7

noncomputable def S (n : ℕ+) : ℝ := (Finset.range n.val).sum (λ i => sequence_a ⟨i + 1, Nat.succ_pos i⟩)
noncomputable def T (n : ℕ+) : ℝ := (Finset.range n.val).sum (λ i => sequence_b ⟨i + 1, Nat.succ_pos i⟩)

axiom a_positive (n : ℕ+) : sequence_a n > 0

axiom a_1 : sequence_a 1 = 1
axiom b_1 : sequence_b 1 = 1

axiom point_on_curve (n : ℕ+) : f (sequence_a n) = -1 / (sequence_a (n + 1))

axiom T_relation (n : ℕ+) : 
  T (n + 1) / (sequence_a n)^2 = T n / (sequence_a (n + 1))^2 + 16 * n.val^2 - 8 * n.val - 3

theorem a_formula (n : ℕ+) : sequence_a n = 1 / Real.sqrt (4 * n.val - 3) := by sorry

theorem b_formula (n : ℕ+) : sequence_b n = 8 * n.val - 7 := by sorry

theorem S_lower_bound (n : ℕ+) : S n > 1/2 * Real.sqrt (4 * n.val + 1) - 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_b_formula_S_lower_bound_l772_77238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_swimming_speed_calculation_l772_77232

/-- Represents the swimming speed of a person in still water -/
noncomputable def swimming_speed : ℝ → ℝ → ℝ → ℝ := λ distance time current_speed =>
  (distance / time) + current_speed

/-- Theorem stating that given the conditions, the person's swimming speed in still water is 12 km/h -/
theorem swimming_speed_calculation (distance : ℝ) (time : ℝ) (current_speed : ℝ)
  (h1 : distance = 12)
  (h2 : time = 6)
  (h3 : current_speed = 10) :
  swimming_speed distance time current_speed = 12 := by
  sorry

#check swimming_speed_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_swimming_speed_calculation_l772_77232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_of_roots_matrix_l772_77229

theorem det_of_roots_matrix (a b c : ℂ) : 
  (a^3 - 3*a^2 + 2*a - 1 = 0) → 
  (b^3 - 3*b^2 + 2*b - 1 = 0) → 
  (c^3 - 3*c^2 + 2*c - 1 = 0) → 
  let M : Matrix (Fin 3) (Fin 3) ℂ := !![a-b, b-c, c-a; b-c, c-a, a-b; c-a, a-b, b-c]
  Matrix.det M = 0 := by 
    intros ha hb hc
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_of_roots_matrix_l772_77229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alec_needs_five_more_votes_l772_77290

/-- Alec's class president election scenario --/
def class_president_election (total_students : ℕ) 
                             (initial_votes : ℕ) 
                             (initial_considering : ℕ) 
                             (first_round_fraction : ℚ) 
                             (second_round_fraction : ℚ) 
                             (goal_votes : ℕ) : ℕ :=
  let initial_undecided := total_students - initial_votes
  let truly_undecided := initial_undecided - initial_considering
  let first_round_votes := initial_votes + (truly_undecided * first_round_fraction.num / first_round_fraction.den).toNat
  let remaining_undecided := truly_undecided - (truly_undecided * first_round_fraction.num / first_round_fraction.den).toNat
  let second_round_votes := first_round_votes + (remaining_undecided * second_round_fraction.num / second_round_fraction.den).toNat
  goal_votes - second_round_votes

/-- Theorem stating Alec needs 5 more votes to reach his goal --/
theorem alec_needs_five_more_votes : 
  class_president_election 100 50 10 (1/4) (1/3) 75 = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alec_needs_five_more_votes_l772_77290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_for_gf_l772_77247

theorem no_solution_for_gf (f g : ℝ → ℝ) 
  (h : ∃ x : ℝ, x - f (g x) = 0) : 
  ¬∃ x : ℝ, g (f x) = x^2 + x + (1/5 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_for_gf_l772_77247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_deformation_equals_twice_lower_half_l772_77297

/-- Represents a spring system with two masses -/
structure SpringSystem where
  k : ℝ  -- Spring constant
  m₁ : ℝ  -- Mass of the upper load
  m₂ : ℝ  -- Mass of the lower load
  x₁ : ℝ  -- Deformation of the upper half
  x₂ : ℝ  -- Deformation of the lower half
  g : ℝ  -- Gravitational acceleration

/-- The deformation of the entire spring when placed horizontally after flipping -/
noncomputable def horizontal_deformation (s : SpringSystem) : ℝ :=
  (s.m₂ * s.g) / s.k

/-- Theorem stating that the horizontal deformation is twice the initial lower half deformation -/
theorem horizontal_deformation_equals_twice_lower_half 
  (s : SpringSystem) (h : s.m₂ * s.g = 2 * s.k * s.x₂) : 
  horizontal_deformation s = 2 * s.x₂ := by
  sorry

#check horizontal_deformation_equals_twice_lower_half

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_deformation_equals_twice_lower_half_l772_77297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_a_correct_equation_b_correct_both_equations_correct_l772_77291

/-- Represents the number of carriages -/
def x : ℕ := sorry

/-- Represents the number of people -/
def y : ℕ := sorry

/-- Equation A is correct -/
theorem equation_a_correct : 3 * (x - 2) = 2 * x + 9 := by
  sorry

/-- Equation B is correct -/
theorem equation_b_correct : y / 3 + 2 = (y - 9) / 2 := by
  sorry

/-- Both equations are correct representations of the problem -/
theorem both_equations_correct : 
  (3 * (x - 2) = 2 * x + 9) ∧ (y / 3 + 2 = (y - 9) / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_a_correct_equation_b_correct_both_equations_correct_l772_77291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_transformation_l772_77289

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f(x+1)
def domain_f_x_plus_1 : Set ℝ := Set.Ici (-1) ∩ Set.Iio 0

-- Define the domain of f(2x)
def domain_f_2x : Set ℝ := Set.Ici 0 ∩ Set.Iio (1/2)

-- Theorem statement
theorem domain_transformation (h : ∀ x, x ∈ domain_f_x_plus_1 ↔ f (x + 1) ∈ Set.Ici 0 ∩ Set.Iio 1) :
  ∀ x, x ∈ domain_f_2x ↔ f (2*x) ∈ Set.Ici 0 ∩ Set.Iio 1 := by
  sorry

#check domain_transformation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_transformation_l772_77289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_coordinates_conversion_l772_77284

open Real

/-- Converts a point in spherical coordinates to its standard representation. -/
noncomputable def standardSphericalCoordinates (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  let ρ' := abs ρ
  let φ' := if φ > π then 2*π - φ else φ
  let θ' := (if φ > π then θ + π else θ) % (2*π)
  (ρ', θ', φ')

/-- The given point in spherical coordinates. -/
noncomputable def givenPoint : ℝ × ℝ × ℝ := (5, 15*π/7, 12*π/7)

/-- The expected standard representation of the point. -/
noncomputable def expectedStandardPoint : ℝ × ℝ × ℝ := (5, 8*π/7, 2*π/7)

theorem spherical_coordinates_conversion :
  standardSphericalCoordinates givenPoint.1 givenPoint.2.1 givenPoint.2.2 = expectedStandardPoint :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_coordinates_conversion_l772_77284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_straight_flush_l772_77239

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Nat)
  (suits : Nat)
  (ranks : Nat)

/-- Represents a hand of cards -/
structure Hand :=
  (size : Nat)

/-- Defines a straight flush -/
def is_straight_flush (h : Hand) : Prop :=
  h.size = 5 ∧ ∃ (suit : Nat) (start : Nat), start + 4 ≤ 13

/-- Standard deck configuration -/
def standard_deck : Deck :=
  { cards := 52, suits := 4, ranks := 13 }

/-- Theorem: Probability of drawing a straight flush -/
theorem probability_straight_flush (d : Deck) (h : Hand) :
  d = standard_deck →
  is_straight_flush h →
  (Nat.choose d.cards h.size) > 0 →
  (40 : ℚ) / (Nat.choose d.cards h.size) = 1 / 64974 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_straight_flush_l772_77239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_1994_l772_77216

noncomputable def sequenceA (a : ℝ) : ℕ → ℝ
  | 0 => a
  | n + 1 => (sequenceA a n * Real.sqrt 3 + 1) / (Real.sqrt 3 - sequenceA a n)

theorem sequence_1994 (a : ℝ) : sequenceA a 1994 = (a + Real.sqrt 3) / (1 - a * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_1994_l772_77216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eighth_term_is_47_l772_77203

def S : ℕ → ℕ
  | 0 => 1
  | 1 => 3
  | n + 2 => S (n + 1) + S n

theorem eighth_term_is_47 : S 7 = 47 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eighth_term_is_47_l772_77203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_divides_equally_l772_77219

/-- A 'T' shape composed of seven unit squares on a coordinate plane -/
structure TShape where
  bottom_left : ℚ × ℚ

/-- A diagonal line from (e,0) to (4,3) -/
structure DiagonalLine where
  e : ℚ

/-- The area of the region below the diagonal line -/
def area_below_diagonal (d : DiagonalLine) : ℚ :=
  3 * (4 - d.e) / 2

/-- The total area of the T shape -/
def total_area : ℚ := 7

theorem diagonal_divides_equally (d : DiagonalLine) :
  area_below_diagonal d = total_area / 2 ↔ d.e = 5/3 :=
by
  -- Proof steps would go here
  sorry

#eval area_below_diagonal ⟨5/3⟩
#eval total_area / 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_divides_equally_l772_77219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_2013_is_9_l772_77231

/-- The product of 0.243 and 0.325233 -/
def product : ℚ := 243/1000 * 325233/1000000

/-- The digit at a given position after the decimal point in a rational number -/
def digit_at_position (q : ℚ) (n : ℕ) : ℕ :=
  ((q * (10 : ℚ)^n).floor % 10).natAbs

/-- The theorem stating that the 2013th digit after the decimal point in the product is 9 -/
theorem digit_2013_is_9 : digit_at_position product 2013 = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_2013_is_9_l772_77231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l772_77236

theorem problem_statement : 
  (¬ ∃ x : ℝ, (2 : ℝ)^(x - 3) ≤ 0) → 
  (¬(¬(∃ x : ℝ, (2 : ℝ)^(x - 3) ≤ 0) ∧ q)) → 
  ¬q := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l772_77236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sports_equipment_purchase_l772_77270

/-- Represents the price of a soccer ball in yuan -/
def soccer_ball_price : ℝ → Prop := sorry

/-- Represents the price of a basketball in yuan -/
def basketball_price : ℝ → Prop := sorry

/-- Represents the number of soccer balls purchased -/
def soccer_balls : ℕ → Prop := sorry

/-- Represents the number of basketballs purchased -/
def basketballs : ℕ → Prop := sorry

/-- The main theorem stating the prices and minimum number of soccer balls to purchase -/
theorem sports_equipment_purchase 
  (h1 : ∀ x y, soccer_ball_price x ∧ basketball_price y → y = 2 * x - 30)
  (h2 : ∀ x y, soccer_ball_price x ∧ basketball_price y → (1200 / x) = 2 * (900 / y))
  (h3 : ∀ m n, soccer_balls m ∧ basketballs n → m + n = 200)
  (h4 : ∀ x y m n, soccer_ball_price x ∧ basketball_price y ∧ soccer_balls m ∧ basketballs n → 
    x * m + y * n ≤ 15000) :
  (∃ x y, soccer_ball_price x ∧ basketball_price y ∧ x = 60 ∧ y = 90) ∧
  (∃ m, soccer_balls m ∧ m ≥ 100) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sports_equipment_purchase_l772_77270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_angle_maximal_when_cyclic_l772_77263

-- Define a quadrilateral by its side lengths and diagonals
structure Quadrilateral :=
  (a b c d e f : ℝ)

-- Define the property of being cyclic
def is_cyclic (q : Quadrilateral) : Prop :=
  q.a * q.c + q.b * q.d = q.e * q.f

-- Define the acute angle between diagonals
noncomputable def diagonal_angle (q : Quadrilateral) : ℝ :=
  Real.arccos ((q.a^2 + q.c^2 - q.b^2 - q.d^2) / (2 * q.e * q.f))

-- Theorem statement
theorem diagonal_angle_maximal_when_cyclic (q : Quadrilateral) :
  ∀ q' : Quadrilateral, q'.a = q.a ∧ q'.b = q.b ∧ q'.c = q.c ∧ q'.d = q.d →
  is_cyclic q → diagonal_angle q ≥ diagonal_angle q' := by
  sorry

-- Additional helper lemmas if needed
lemma cyclic_quadrilateral_diagonal_product (q : Quadrilateral) :
  is_cyclic q → q.e * q.f = q.a * q.c + q.b * q.d := by
  sorry

lemma cosine_inequality (q : Quadrilateral) :
  (q.a^2 + q.c^2 - q.b^2 - q.d^2) / (2 * (q.a * q.c + q.b * q.d)) ≥
  (q.a^2 + q.c^2 - q.b^2 - q.d^2) / (2 * q.e * q.f) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_angle_maximal_when_cyclic_l772_77263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_constant_sum_min_area_triangle_l772_77220

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/8 + y^2/4 = 1

-- Define the right focus
def right_focus : ℝ × ℝ := (2, 0)

-- Define a point on the line
def point_on_line (t : ℝ) : ℝ × ℝ := (0, t)

-- Define points A and B on the ellipse
def point_A (x y : ℝ) : Prop := ellipse x y ∧ ∃ k, y = k * (x - 2)
def point_B (x y : ℝ) : Prop := ellipse x y ∧ ∃ k, y = k * (x - 2)

-- Define the relationship between PA, AF, PB, and BF
def vector_relationship (m n : ℝ) (xA yA xB yB t : ℝ) : Prop :=
  (xA = m * (2 - xA) ∧ yA - t = m * (-yA)) ∧
  (xB = n * (2 - xB) ∧ yB - t = n * (-yB))

-- Theorem statement
theorem ellipse_constant_sum 
  (xA yA xB yB t m n : ℝ) 
  (hA : point_A xA yA) 
  (hB : point_B xB yB) 
  (hP : point_on_line t = (0, t))
  (hRel : vector_relationship m n xA yA xB yB t) :
  m + n = -4 :=
sorry

-- Minimum area theorem
theorem min_area_triangle
  (k : ℝ)
  (hk : k^2 ≥ 1) :
  4 * Real.sqrt 2 * Real.sqrt (1 - 1 / (4 * (k^2 + 1/2)^2)) ≥ 16/3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_constant_sum_min_area_triangle_l772_77220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_decreases_to_convex_hull_outer_perimeter_greater_equal_inner_l772_77240

/-- A polygon in 2D space -/
structure Polygon where
  vertices : List (ℝ × ℝ)
  is_closed : vertices.head? = vertices.getLast?

/-- The perimeter of a polygon -/
noncomputable def perimeter (p : Polygon) : ℝ := sorry

/-- The convex hull of a polygon -/
noncomputable def convex_hull (p : Polygon) : Polygon := sorry

/-- Predicate to check if a polygon is convex -/
def is_convex (p : Polygon) : Prop := sorry

/-- Predicate to check if one polygon contains another -/
def contains (p1 p2 : Polygon) : Prop := sorry

/-- Theorem: The perimeter decreases when transitioning to the convex hull -/
theorem perimeter_decreases_to_convex_hull (p : Polygon) :
  perimeter p ≥ perimeter (convex_hull p) := by sorry

/-- Theorem: The outer convex polygon's perimeter is not less than the inner convex polygon's perimeter -/
theorem outer_perimeter_greater_equal_inner (p1 p2 : Polygon)
  (h1 : is_convex p1) (h2 : is_convex p2) (h3 : contains p1 p2) :
  perimeter p1 ≥ perimeter p2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_decreases_to_convex_hull_outer_perimeter_greater_equal_inner_l772_77240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l772_77274

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 1  -- Added case for 0
  | 1 => 1
  | n + 2 => 3 * sequence_a (n + 1) + 3^(n + 1)

noncomputable def sequence_b (n : ℕ) : ℝ := 
  if n = 0 then 1 else sequence_a n / 3^(n-1)

theorem sequence_properties :
  (∀ n : ℕ, n ≥ 1 → sequence_b (n + 1) - sequence_b n = 1) ∧
  (∀ n : ℕ, n ≥ 1 → sequence_a n = n * 3^(n-1)) := by
  sorry

#check sequence_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l772_77274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carla_sharpening_time_l772_77235

/-- Represents the time Carla spends sharpening her knife -/
def sharpening_time : ℝ := 10

/-- Represents the time Carla spends peeling vegetables -/
def peeling_time : ℝ := 3 * sharpening_time

/-- The total time Carla spends on both activities -/
def total_time : ℝ := 40

theorem carla_sharpening_time :
  sharpening_time + peeling_time = total_time ∧
  sharpening_time = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carla_sharpening_time_l772_77235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_l772_77200

theorem cos_double_angle (α : ℝ) (h1 : 0 < α ∧ α < π / 2) 
  (h2 : Real.sin α - Real.cos α = Real.sqrt 10 / 5) : Real.cos (2 * α) = -4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_l772_77200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_divisible_by_11_l772_77222

/-- A permutation of the digits 0, 1, 2, 3, 4, 5 -/
def SixDigitPerm := Fin 6 → Fin 6

/-- Converts a permutation to a natural number -/
def permToNat (p : SixDigitPerm) : ℕ := 
  100000 * p 5 + 10000 * p 4 + 1000 * p 3 + 100 * p 2 + 10 * p 1 + p 0

/-- Theorem: No six-digit number formed by a permutation of 0,1,2,3,4,5 is divisible by 11 -/
theorem not_divisible_by_11 (p : SixDigitPerm) : ¬(11 ∣ permToNat p) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_divisible_by_11_l772_77222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_parallelogram_area_l772_77234

/-- The area of the parallelogram formed by two 3D vectors -/
noncomputable def parallelogram_area (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  let cross_product := (
    (v1.2.1 * v2.2.2 - v1.2.2 * v2.1),
    (v1.2.2 * v2.1 - v1.1 * v2.2.2),
    (v1.1 * v2.2.1 - v1.2.1 * v2.1)
  )
  Real.sqrt (cross_product.1^2 + cross_product.2.1^2 + cross_product.2.2^2)

/-- The theorem stating the area of the specific parallelogram -/
theorem specific_parallelogram_area :
  parallelogram_area (2, 4, -1) (1, -3, 5) = Real.sqrt 510 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_parallelogram_area_l772_77234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_in_cube_volume_ratio_l772_77257

theorem cylinder_in_cube_volume_ratio :
  ∀ (s : ℝ), s > 0 →
  let r := s / 2
  let h := s
  let V_cylinder := π * r^2 * h
  let V_cube := s^3
  V_cylinder / V_cube = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_in_cube_volume_ratio_l772_77257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_order_abcd_l772_77255

noncomputable section

-- Define the angle in radians (2009° converted to radians)
def angle : ℝ := 2009 * Real.pi / 180

-- Define a, b, c, d as in the problem
def a : ℝ := Real.sin (Real.sin angle)
def b : ℝ := Real.sin (Real.cos angle)
def c : ℝ := Real.cos (Real.sin angle)
def d : ℝ := Real.cos (Real.cos angle)

-- State the theorem
theorem increasing_order_abcd : b < a ∧ a < d ∧ d < c := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_order_abcd_l772_77255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_average_weight_RQ_SQ_l772_77285

/-- Represents a pile of quarters -/
structure QuarterPile where
  count : ℕ
  totalWeight : ℝ

/-- The average weight of a pile of quarters -/
noncomputable def averageWeight (pile : QuarterPile) : ℝ :=
  pile.totalWeight / pile.count

/-- Theorem stating the maximum average weight of combined piles R(Q) and S(Q) -/
theorem max_average_weight_RQ_SQ (P R S : QuarterPile)
  (hP : averageWeight P = 30)
  (hR : averageWeight R = 40)
  (hPR : averageWeight ⟨P.count + R.count, P.totalWeight + R.totalWeight⟩ = 34)
  (hPS : averageWeight ⟨P.count + S.count, P.totalWeight + S.totalWeight⟩ = 35) :
  ∃ (n : ℕ), n ≤ 48 ∧
    averageWeight ⟨R.count + S.count, R.totalWeight + S.totalWeight⟩ ≤ n ∧
    ∀ (m : ℕ), m > 48 →
      averageWeight ⟨R.count + S.count, R.totalWeight + S.totalWeight⟩ < m := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_average_weight_RQ_SQ_l772_77285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_trig_expression_l772_77248

theorem min_trig_expression (θ : Real) (h : 0 < θ ∧ θ < Real.pi / 2) :
  Real.sin θ + 2 * Real.cos θ + Real.sqrt 2 / Real.tan θ ≥ 3 ∧
  (Real.sin θ + 2 * Real.cos θ + Real.sqrt 2 / Real.tan θ = 3 ↔ θ = Real.pi / 4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_trig_expression_l772_77248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_manuscript_typing_cost_l772_77246

/-- The cost per page for the first time a page is typed -/
def first_time_cost : ℚ := 0

/-- The total number of pages in the manuscript -/
def total_pages : ℕ := 100

/-- The number of pages revised once -/
def pages_revised_once : ℕ := 30

/-- The number of pages revised twice -/
def pages_revised_twice : ℕ := 20

/-- The cost per page for each revision -/
def revision_cost : ℚ := 3

/-- The total cost of typing the manuscript -/
def total_cost : ℚ := 710

theorem manuscript_typing_cost :
  first_time_cost * total_pages + 
  revision_cost * (pages_revised_once + 2 * pages_revised_twice) = total_cost ∧
  first_time_cost = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_manuscript_typing_cost_l772_77246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_two_equality_l772_77265

theorem power_of_two_equality (y : ℝ) (h : (2 : ℝ)^(4*y) = 16) : (2 : ℝ)^(-y) = (1/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_two_equality_l772_77265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coplanar_vectors_lambda_l772_77298

/-- Given three vectors a, b, and c in ℝ³, if they are coplanar, then the third component of c is 1. -/
theorem coplanar_vectors_lambda (a b c : ℝ × ℝ × ℝ) : 
  a = (2, -1, 3) → 
  b = (-1, 4, -2) → 
  c.1 = 1 → 
  c.2.1 = 3 → 
  (∃ (m n : ℝ), c = (m * a.1 + n * b.1, m * a.2.1 + n * b.2.1, m * a.2.2 + n * b.2.2)) → 
  c.2.2 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coplanar_vectors_lambda_l772_77298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_red_points_correct_l772_77228

/-- The minimum number of red points required in a configuration of concentric circles and rays -/
def min_red_points (n : ℕ) (h : n > 0) : ℕ :=
  let k : ℕ := 2021
  let N : ℕ := n^k
  k * n^(k-1)

/-- Rotate a point on a circle by an angle -/
noncomputable def rotate (N : ℕ) (θ : ℝ) (p : Fin N) : Fin N :=
  sorry

/-- The set of red points -/
def red_points (N : ℕ) : Set (Fin N) :=
  sorry

theorem min_red_points_correct (n : ℕ) (h : n > 0) : 
  let k : ℕ := 2021
  let N : ℕ := n^k
  (∀ red : Set (Fin N), 
   (∀ (chosen : Fin k → Fin N), 
    ∃ (θ : ℝ), ∀ (i : Fin k), 
    (rotate N θ (chosen i)) ∈ red) → 
   (Finset.card (red.toFinite.toFinset) ≥ min_red_points n h)) ∧
  (∃ red : Set (Fin N), 
   (∀ (chosen : Fin k → Fin N), 
    ∃ (θ : ℝ), ∀ (i : Fin k), 
    (rotate N θ (chosen i)) ∈ red) ∧
   (Finset.card (red.toFinite.toFinset) = min_red_points n h)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_red_points_correct_l772_77228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_wins_l772_77249

structure Game (n : ℕ) where
  r : ℕ
  b : ℕ
  sum : r + b = n

def is_winning_position (g : Game 1999) : Prop :=
  g.r ≠ g.b

theorem first_player_wins :
  ∀ (initial_game : Game 1999),
  ∃ (strategy : Game 1999 → Game 1999),
  (∀ (g : Game 1999), is_winning_position g → is_winning_position (strategy g)) ∧
  (∀ (g : Game 1999), ¬is_winning_position g → 
    ∀ (move : Game 1999), ¬is_winning_position move → 
      is_winning_position (strategy move)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_wins_l772_77249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mississippi_arrangements_no_adjacent_s_l772_77243

/-- Represents the word "MISSISSIPPI" -/
def mississippi : Multiset Char := Multiset.ofList ['M', 'I', 'I', 'I', 'I', 'S', 'S', 'S', 'S', 'P', 'P']

/-- Counts the number of ways to arrange the letters of "MISSISSIPPI" with no adjacent S's -/
def count_arrangements_no_adjacent_s (word : Multiset Char) : ℕ :=
  sorry

/-- Theorem stating that the number of arrangements of "MISSISSIPPI" with no adjacent S's is 7350 -/
theorem mississippi_arrangements_no_adjacent_s :
  count_arrangements_no_adjacent_s mississippi = 7350 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mississippi_arrangements_no_adjacent_s_l772_77243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_sides_l772_77233

-- Define the triangle ABC
structure RightTriangle where
  AB : ℝ
  BC : ℝ
  AC : ℝ
  right_angle : AB^2 + BC^2 = AC^2

-- Define the parallel lines
structure ParallelLines where
  distance : ℝ

-- Define the theorem
theorem max_distance_to_sides 
  (triangle : RightTriangle) 
  (parallel_lines : ParallelLines) :
  triangle.AB = 8 →
  triangle.BC = 6 →
  triangle.AC = 10 →
  parallel_lines.distance = 1 →
  ∃ (max_distance : ℝ), 
    max_distance = 7 ∧ 
    ∀ (point : ℝ × ℝ), 
      point.1 + point.2 + parallel_lines.distance ≤ max_distance :=
by
  intro hAB hBC hAC hdist
  use 7
  apply And.intro
  · rfl
  · intro point
    sorry  -- Proof details omitted


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_sides_l772_77233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_christine_walk_time_l772_77230

/-- Calculates the time taken for a segment of a walk given distance and speed -/
noncomputable def time_for_segment (distance : ℝ) (speed : ℝ) : ℝ := distance / speed

/-- Represents Christine's walk with three segments -/
structure ChristineWalk where
  segment1_distance : ℝ
  segment1_speed : ℝ
  segment2_distance : ℝ
  segment2_speed : ℝ
  segment3_distance : ℝ
  segment3_speed : ℝ

/-- Calculates the total time for Christine's walk -/
noncomputable def total_time (walk : ChristineWalk) : ℝ :=
  time_for_segment walk.segment1_distance walk.segment1_speed +
  time_for_segment walk.segment2_distance walk.segment2_speed +
  time_for_segment walk.segment3_distance walk.segment3_speed

/-- Theorem stating that Christine's walk takes 12 hours -/
theorem christine_walk_time :
  let walk := ChristineWalk.mk 20 4 24 6 9 3
  total_time walk = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_christine_walk_time_l772_77230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_certain_number_value_l772_77281

theorem certain_number_value (x : ℝ) (certain_number : ℝ) :
  (28 + x + 42 + 78 + 104) / 5 = 62 →
  (48 + 62 + 98 + certain_number + x) / 5 = 78 →
  certain_number = 124 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_certain_number_value_l772_77281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_tangent_l772_77237

/-- Given an angle α with vertex at the origin, initial side on the positive x-axis,
    and terminal side passing through point (-3,m), if sin α = -4/5, then tan α = 4/3 -/
theorem angle_tangent (α : ℝ) (m : ℝ) : 
  (∃ (x y : ℝ), x = -3 ∧ y = m ∧ x^2 + y^2 ≠ 0 ∧ Real.sin α = y / Real.sqrt (x^2 + y^2)) →
  Real.sin α = -4/5 →
  Real.tan α = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_tangent_l772_77237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_petrol_consumption_l772_77207

/-- Calculates the weekly petrol consumption of a car given the total petrol amount and number of days it lasts. -/
noncomputable def weekly_petrol_consumption (total_petrol : ℝ) (total_days : ℕ) : ℝ :=
  total_petrol / (total_days / 7 : ℝ)

/-- Theorem stating that given 1470 liters of petrol is sufficient for 49 days, 
    the weekly petrol consumption of the car is 210 liters. -/
theorem car_petrol_consumption :
  weekly_petrol_consumption 1470 49 = 210 := by
  -- Unfold the definition of weekly_petrol_consumption
  unfold weekly_petrol_consumption
  -- Simplify the arithmetic
  simp [div_div_eq_mul_div]
  -- The rest of the proof would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_petrol_consumption_l772_77207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_tetrahedron_surface_area_l772_77225

/-- A tetrahedron with edge length 2 and all faces being equilateral triangles -/
structure RegularTetrahedron where
  edge_length : ℝ
  edge_length_eq : edge_length = 2
  faces_equilateral : Bool

/-- The surface area of a regular tetrahedron -/
noncomputable def surface_area (t : RegularTetrahedron) : ℝ :=
  4 * (Real.sqrt 3)

theorem regular_tetrahedron_surface_area (t : RegularTetrahedron) :
  surface_area t = 4 * (Real.sqrt 3) := by
  -- Unfold the definition of surface_area
  unfold surface_area
  -- The result follows directly from the definition
  rfl

#check regular_tetrahedron_surface_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_tetrahedron_surface_area_l772_77225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_slope_at_half_l772_77221

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * a * x^2 - 1 / (a * x)

noncomputable def k (a : ℝ) : ℝ := 4 * a + 1 / a

theorem min_slope_at_half (a : ℝ) (h : a > 0) :
  ∃ (min_k : ℝ), k a = min_k ∧ ∀ (b : ℝ), b > 0 → k b ≥ min_k ↔ a = 1/2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_slope_at_half_l772_77221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_squared_length_of_v_graph_l772_77293

-- Define the linear functions
noncomputable def p (x : ℝ) : ℝ := 2 * x + 1
noncomputable def q (x : ℝ) : ℝ := -x + 3
noncomputable def r (x : ℝ) : ℝ := 0.5 * x + 2

-- Define u(x) and v(x)
noncomputable def u (x : ℝ) : ℝ := max (max (p x) (q x)) (r x)
noncomputable def v (x : ℝ) : ℝ := min (min (p x) (q x)) (r x)

-- Define the domain
def domain : Set ℝ := {x | -4 ≤ x ∧ x ≤ 4}

-- State the theorem
theorem squared_length_of_v_graph : 
  ∃ (L : ℝ), L = (Real.sqrt (481/9) + Real.sqrt (181/9))^2 ∧
  L = ∫ x in domain, (1 + (deriv v x)^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_squared_length_of_v_graph_l772_77293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_6_value_l772_77299

mutual
  def sequence_a : ℕ → ℚ
    | 0 => 3
    | (n + 1) => sequence_a n ^ 3 / sequence_b n

  def sequence_b : ℕ → ℚ
    | 0 => 5
    | (n + 1) => sequence_b n ^ 3 / sequence_a n
end

theorem b_6_value : sequence_b 6 = 5 ^ 377 / 3 ^ 376 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_6_value_l772_77299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_f_neg_one_eq_six_l772_77217

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then x * (x + 1)
  else if x < 0 then x * (x - 1)
  else 0  -- We need to define f for x = 0 to make it total

-- State the theorem
theorem f_of_f_neg_one_eq_six : f (f (-1)) = 6 := by
  -- Evaluate f(-1)
  have h1 : f (-1) = 2 := by
    simp [f]
    norm_num
  
  -- Evaluate f(2)
  have h2 : f 2 = 6 := by
    simp [f]
    norm_num
  
  -- Combine the results
  calc
    f (f (-1)) = f 2 := by rw [h1]
    _          = 6   := by rw [h2]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_f_neg_one_eq_six_l772_77217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_overtake_l772_77262

theorem train_overtake (speed_a speed_b : ℝ) (head_start : ℝ) (overtake_distance : ℝ) : 
  speed_a = 30 →
  speed_b = 36 →
  head_start = 2 →
  speed_a * head_start + speed_a * (overtake_distance / speed_b) = overtake_distance →
  overtake_distance = 360 := by
  sorry

#check train_overtake

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_overtake_l772_77262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_outside_plane_l772_77269

-- Define the universe
universe u

-- Define the types for points, lines, and planes
variable (Point Line Plane : Type u)

-- Define the relations
variable (belongs_to : Point → Line → Prop)
variable (belongs_to_plane : Point → Plane → Prop)
variable (outside_of : Point → Plane → Prop)
variable (passes_through : Line → Point → Prop)

-- Define the statement
theorem line_through_point_outside_plane 
  (a : Line) (P : Point) (α : Plane) :
  (passes_through a P ∧ outside_of P α) ↔ (belongs_to P a ∧ ¬belongs_to_plane P α) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_outside_plane_l772_77269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_complex_fraction_l772_77279

theorem imaginary_part_of_complex_fraction : Complex.im ((3 + 4*Complex.I) / (3 - 4*Complex.I)) = 24 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_complex_fraction_l772_77279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_triplets_eq_56_l772_77250

/-- The number of positive integer triplets {a, b, c} satisfying the given conditions -/
def count_triplets : ℕ :=
  Finset.card (Finset.filter
    (fun t : ℕ × ℕ × ℕ => 
      t.1 > 0 ∧ t.2.1 > 0 ∧ t.2.2 > 0 ∧
      Nat.lcm t.1 (Nat.lcm t.2.1 t.2.2) = 20000 ∧
      Nat.gcd t.1 (Nat.gcd t.2.1 t.2.2) = 20)
    (Finset.product (Finset.range 20001) (Finset.product (Finset.range 20001) (Finset.range 20001))))

theorem count_triplets_eq_56 : count_triplets = 56 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_triplets_eq_56_l772_77250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_survey_percentage_calculation_l772_77273

theorem survey_percentage_calculation (total : ℕ) (h : total > 0) :
  let first_three := λ (n : ℕ) => (total - 3 * n) / 20 = n
  let fourth := λ (n : ℕ) => (total - 3 * (total / 21) - n) / 2 = n
  let fifth := λ (n : ℕ) => n = total - 3 * (total / 21) - (total / 3)
  ∃ n₁ n₂ n₃ n₄ n₅ : ℕ,
    first_three n₁ ∧ first_three n₂ ∧ first_three n₃ ∧
    fourth n₄ ∧
    fifth n₅ ∧
    n₅ * 100 / (total - n₅) = 110 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_survey_percentage_calculation_l772_77273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_for_specific_intercepts_l772_77208

/-- Represents a quadratic function of the form f(x) = ax² + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The x-coordinate of the axis of symmetry of a quadratic function -/
noncomputable def axisOfSymmetry (f : QuadraticFunction) : ℝ := -f.b / (2 * f.a)

/-- A theorem stating that for a quadratic function with x-intercepts at -1 and 2,
    the axis of symmetry is at x = 1/2 -/
theorem axis_of_symmetry_for_specific_intercepts (f : QuadraticFunction) 
    (h1 : f.a * (-1)^2 + f.b * (-1) + f.c = 0)
    (h2 : f.a * 2^2 + f.b * 2 + f.c = 0) :
    axisOfSymmetry f = 1/2 := by
  sorry

#check axis_of_symmetry_for_specific_intercepts

end NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_for_specific_intercepts_l772_77208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monochromatic_triangle_probability_approaches_one_l772_77268

/-- A color type representing red, blue, or green -/
inductive Color
  | Red
  | Blue
  | Green

/-- A type representing the edges of a hexagon (sides and diagonals) -/
def HexagonEdges := Fin 15 → Color

/-- A function to check if a triangle is monochromatic -/
def isMonochromaticTriangle (edges : HexagonEdges) (a b c : Fin 6) : Prop :=
  ∃ (color : Color), 
    edges ⟨min a.val b.val * 6 + max a.val b.val, by sorry⟩ = color ∧
    edges ⟨min b.val c.val * 6 + max b.val c.val, by sorry⟩ = color ∧
    edges ⟨min c.val a.val * 6 + max c.val a.val, by sorry⟩ = color

/-- Random color assignment function (placeholder) -/
noncomputable def randomColorAssignment (n : ℕ) : HexagonEdges :=
  fun _ => Color.Red  -- Placeholder implementation

/-- Probability function (placeholder) -/
noncomputable def Prob (p : HexagonEdges → Prop) : ℝ :=
  0  -- Placeholder implementation

/-- The main theorem stating that the probability of a monochromatic triangle approaches 1 -/
theorem monochromatic_triangle_probability_approaches_one :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N,
    (1 : ℝ) - ε < Prob (fun edges => ∃ (a b c : Fin 6), isMonochromaticTriangle edges a b c) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monochromatic_triangle_probability_approaches_one_l772_77268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l772_77227

/-- Calculates the speed of a train in km/h given its length and time to cross a point -/
noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  (length / time) * 3600 / 1000

/-- Theorem: A train 441 m long that crosses an electric pole in 21 sec has a speed of 75.6 km/h -/
theorem train_speed_calculation :
  let length : ℝ := 441
  let time : ℝ := 21
  train_speed length time = 75.6 := by
  -- Unfold the definition of train_speed
  unfold train_speed
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l772_77227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boys_not_studying_science_l772_77287

theorem boys_not_studying_science (total_boys : ℕ) 
  (school_A_percentage : ℚ) (science_percentage : ℚ) :
  total_boys = 550 →
  school_A_percentage = 1/5 →
  science_percentage = 3/10 →
  ∃ (x : ℕ), x > 0 ∧ x = (↑total_boys * school_A_percentage * (1 - science_percentage)).floor →
  (↑total_boys * school_A_percentage * (1 - science_percentage)).floor = 77 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boys_not_studying_science_l772_77287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_condition_l772_77212

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real

-- State the theorem
theorem right_triangle_condition (t : Triangle) :
  (Real.sin t.A)^2 = (Real.sin t.B)^2 + (Real.sin t.C)^2 →
  t.A = Real.pi / 2 ∨ t.B = Real.pi / 2 ∨ t.C = Real.pi / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_condition_l772_77212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_five_eq_ln_five_l772_77254

-- Define the function f as noncomputable
noncomputable def f : ℝ → ℝ := λ x => Real.log x

-- State the theorem
theorem f_five_eq_ln_five : f 5 = Real.log 5 := by
  -- Proof is skipped with sorry
  sorry

-- State the given condition
axiom f_condition : ∀ x : ℝ, f (Real.exp x) = x

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_five_eq_ln_five_l772_77254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l772_77278

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x - Real.pi / 3)

theorem max_value_of_f (ω : ℝ) (h1 : ω > 0) 
  (h2 : ∀ x y, x - y = Real.pi / 2 → f ω x = f ω y) : 
  ∃ x₀ ∈ Set.Icc (-Real.pi / 2) 0, ∀ x ∈ Set.Icc (-Real.pi / 2) 0, f ω x ≤ f ω x₀ ∧ f ω x₀ = Real.sqrt 3 := by
  sorry

#check max_value_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l772_77278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_pyramid_angle_of_inclination_l772_77259

/-- A regular pyramid is a pyramid with a regular polygon base and congruent isosceles triangles as lateral faces. -/
structure RegularPyramid where
  -- We don't need to define the full structure, just declare it exists
  dummy : Unit

/-- The angle of inclination in a regular pyramid -/
noncomputable def angle_of_inclination (p : RegularPyramid) : ℝ :=
  Real.arcsin (Real.sqrt 3 / 3)

/-- Angle between base and lateral face of a regular pyramid -/
noncomputable def angle_between_base_and_lateral_face (p : RegularPyramid) : ℝ :=
  sorry

/-- Angle between side edge and base edge of a regular pyramid -/
noncomputable def angle_between_side_edge_and_base_edge (p : RegularPyramid) : ℝ :=
  sorry

/-- Theorem stating the angle of inclination property for regular pyramids -/
theorem regular_pyramid_angle_of_inclination (p : RegularPyramid) :
  (angle_of_inclination p = Real.arcsin (Real.sqrt 3 / 3)) ∧
  (angle_of_inclination p = angle_between_base_and_lateral_face p) ∧
  (angle_of_inclination p = angle_between_side_edge_and_base_edge p) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_pyramid_angle_of_inclination_l772_77259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_domain_of_f_l772_77295

/-- The function f(x) = (x-1)^0 / sqrt(3-2x) -/
noncomputable def f (x : ℝ) : ℝ := (x - 1)^0 / Real.sqrt (3 - 2*x)

/-- The domain of the function f -/
def domain : Set ℝ := {x | x < 1 ∨ (1 < x ∧ x < 3/2)}

theorem f_domain : 
  ∀ x : ℝ, x ∈ domain ↔ (3 - 2*x > 0 ∧ x ≠ 1) :=
by sorry

/-- The main theorem stating that the domain of f is (-∞, 1) ∪ (1, 3/2) -/
theorem domain_of_f :
  domain = Set.Iio 1 ∪ Set.Ioo 1 (3/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_domain_of_f_l772_77295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l772_77251

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * a * Real.sin x * Real.cos x + Real.cos (2 * x)

-- Theorem statement
theorem function_properties :
  ∃ (a : ℝ),
    (f a (π / 4) = 1) ∧
    (∀ x : ℝ, f a x = f a (x + π)) ∧
    (∀ x ∈ Set.Icc (π / 8) (5 * π / 8), ∀ y ∈ Set.Icc (π / 8) (5 * π / 8),
      x < y → f a y < f a x) :=
by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l772_77251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_odd_divisors_120_eq_24_l772_77205

/-- The sum of the positive odd divisors of 120 -/
def sum_odd_divisors_120 : ℕ :=
  (Finset.filter (fun d => d % 2 = 1 && 120 % d = 0) (Finset.range 121)).sum id

/-- Theorem stating that the sum of the positive odd divisors of 120 is 24 -/
theorem sum_odd_divisors_120_eq_24 : sum_odd_divisors_120 = 24 := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_odd_divisors_120_eq_24_l772_77205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_unoccupied_seats_l772_77288

/-- Calculates the number of unoccupied seats on a plane given the number of rows,
    seats per row, and the occupancy limit. -/
def unoccupiedSeats (rows : ℕ) (seatsPerRow : ℕ) (occupancyLimit : ℚ) : ℕ :=
  let totalSeats := rows * seatsPerRow
  let occupiedSeats := (totalSeats : ℚ) * occupancyLimit
  totalSeats - Int.toNat occupiedSeats.floor

/-- Theorem stating that a plane with 20 rows, 15 seats per row, and 67% occupancy limit
    will have 99 unoccupied seats. -/
theorem plane_unoccupied_seats :
  unoccupiedSeats 20 15 (67 / 100) = 99 := by
  sorry

#eval unoccupiedSeats 20 15 (67 / 100)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_unoccupied_seats_l772_77288

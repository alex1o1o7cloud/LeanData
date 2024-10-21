import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l888_88882

/-- Line l defined by parametric equations -/
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (-1 + Real.sqrt 2 / 2 * t, Real.sqrt 2 / 2 * t)

/-- Curve C defined by polar equation -/
noncomputable def curve_C (θ : ℝ) : ℝ := Real.sin θ / (1 - Real.sin θ ^ 2)

/-- Point M -/
def point_M : ℝ × ℝ := (-1, 0)

/-- Theorem stating that the product of distances from M to intersection points is 2 -/
theorem intersection_distance_product :
  ∃ (A B : ℝ × ℝ),
    (∃ (t : ℝ), line_l t = A) ∧
    (∃ (θ : ℝ), (curve_C θ * Real.cos θ, curve_C θ * Real.sin θ) = A) ∧
    (∃ (t : ℝ), line_l t = B) ∧
    (∃ (θ : ℝ), (curve_C θ * Real.cos θ, curve_C θ * Real.sin θ) = B) ∧
    A ≠ B ∧
    (Real.sqrt ((A.1 - point_M.1)^2 + (A.2 - point_M.2)^2)) *
    (Real.sqrt ((B.1 - point_M.1)^2 + (B.2 - point_M.2)^2)) = 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l888_88882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_iff_a_in_range_l888_88880

-- Define the function f as noncomputable
noncomputable def f (a : ℝ) : ℝ → ℝ := λ x => Real.log (a^2 - 2*a + 1) / Real.log (2*a - 1)

-- State the theorem
theorem f_positive_iff_a_in_range :
  ∀ a : ℝ, (∀ x : ℝ, f a x > 0) ↔ (a ∈ Set.Ioo (1/2 : ℝ) 1 ∪ Set.Ioi 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_iff_a_in_range_l888_88880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_k_times_value_range_l888_88881

/-- The function f(x) = ln(x) + x --/
noncomputable def f (x : ℝ) : ℝ := Real.log x + x

/-- k-times value function property --/
def is_k_times_value (f : ℝ → ℝ) (k : ℝ) : Prop :=
  ∃ a b, a < b ∧ f a = k * a ∧ f b = k * b

/-- The theorem stating the range of k for which f is a k-times value function --/
theorem f_k_times_value_range :
  ∀ k : ℝ, (is_k_times_value f k) ↔ (1 < k ∧ k < 1 + Real.exp (-1)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_k_times_value_range_l888_88881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_center_and_perpendicular_l888_88871

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + 2*x + y^2 = 0

-- Define the perpendicular line
def perp_line (x y : ℝ) : Prop := x + y = 0

-- Define the result line
def result_line (x y : ℝ) : Prop := x - y + 1 = 0

-- Theorem statement
theorem line_through_center_and_perpendicular :
  ∃ (cx cy : ℝ), 
    (∀ x y, my_circle x y ↔ (x - cx)^2 + (y - cy)^2 = (1 : ℝ)) ∧ 
    result_line cx cy ∧
    (∀ x₁ y₁ x₂ y₂, perp_line x₁ y₁ ∧ result_line x₂ y₂ → (x₂ - cx) * (x₁ - x₂) + (y₂ - cy) * (y₁ - y₂) = 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_center_and_perpendicular_l888_88871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_pyramid_properties_l888_88847

/-- Truncated triangular pyramid -/
structure TruncatedPyramid where
  base_edge1 : ℝ
  base_edge2 : ℝ
  base_edge3 : ℝ
  side_angle : ℝ
  height : ℝ

/-- Calculate the surface area of a truncated triangular pyramid -/
noncomputable def surface_area (p : TruncatedPyramid) : ℝ := sorry

/-- Calculate the volume of a truncated triangular pyramid -/
noncomputable def volume (p : TruncatedPyramid) : ℝ := sorry

theorem truncated_pyramid_properties :
  let p : TruncatedPyramid := {
    base_edge1 := 148,
    base_edge2 := 156,
    base_edge3 := 208,
    side_angle := 112.62 * Real.pi / 180,  -- Convert degrees to radians
    height := 27
  }
  (abs (surface_area p - 74352) < 1) ∧ (abs (volume p - 395280) < 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_pyramid_properties_l888_88847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_geq_q_l888_88886

theorem p_geq_q (x : ℝ) : (Real.exp x + Real.exp (-x)) ≥ (Real.sin x + Real.cos x)^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_geq_q_l888_88886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_triples_satisfying_equations_l888_88858

theorem two_triples_satisfying_equations : 
  ∃! n : ℕ, n = (Finset.filter 
    (fun t : ℕ × ℕ × ℕ ↦ 
      let (a, b, c) := t
      a > 0 ∧ b > 0 ∧ c > 0 ∧ 
      a * b + a * c = 35 ∧ 
      b * c + a * c = 12)
    (Finset.product (Finset.range 36) (Finset.product (Finset.range 36) (Finset.range 36)))).card ∧
  n = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_triples_satisfying_equations_l888_88858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_median_angle_relation_l888_88890

/-- 
Given a triangle ABC with:
- k₃: median to side AB
- α: angle at vertex A
- β: angle at vertex B
- γ₁: angle between median k₃ and side AC
- γ₂: angle between median k₃ and side BC

This theorem states a relationship between these angles.
-/
theorem triangle_median_angle_relation 
  (k₃ : ℝ) (α β γ₁ γ₂ : ℝ) 
  (h_k₃_pos : k₃ > 0)
  (h_α_pos : α > 0) (h_β_pos : β > 0)
  (h_γ₁_pos : γ₁ > 0) (h_γ₂_pos : γ₂ > 0)
  (h_α_β_sum : α + β < π)
  (h_γ₁_γ₂_sum : γ₁ + γ₂ = π) :
  Real.tan ((γ₁ - γ₂) / 2) = 
    (Real.tan ((α - β) / 2) / Real.tan ((α + β) / 2)) * 
    (1 / Real.tan ((α + β) / 2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_median_angle_relation_l888_88890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_hash_ratio_l888_88895

-- Define the @ operation
def atOp (a b : ℝ) : ℝ := a^2 * b - a * b^2

-- Define the # operation
def hashOp (a b : ℝ) : ℝ := a^2 + b^2 - a * b

-- Theorem statement
theorem at_hash_ratio : (atOp 10 3) / (hashOp 10 3) = 210 / 79 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_hash_ratio_l888_88895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l888_88825

-- Define the given conditions
def solution_set (x : ℝ) : Prop := 2 < x ∧ x < 4

-- Define the inequality
def inequality (x a b : ℝ) : Prop := |x + a| < b

-- Theorem statement
theorem problem_solution :
  ∃ (a b : ℝ),
    (∀ x, solution_set x ↔ inequality x a b) ∧
    a = -3 ∧
    b = 1 ∧
    (∀ t, Real.sqrt (a * t + 12) + Real.sqrt (3 * b * t) ≤ 2 * Real.sqrt 6) ∧
    (∃ t, Real.sqrt (a * t + 12) + Real.sqrt (3 * b * t) = 2 * Real.sqrt 6) :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l888_88825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_movie_ticket_price_l888_88888

/-- The price of a regular adult movie ticket -/
def regular_adult_ticket_price : ℚ :=
  15.2

theorem movie_ticket_price :
  let num_adults : ℕ := 5
  let num_children : ℕ := 2
  let children_concessions : ℚ := 6
  let adults_concessions : ℚ := 20
  let total_cost : ℚ := 112
  let child_ticket_price : ℚ := 7
  let adult_discount : ℚ := 2
  let num_discounted_adults : ℕ := 2
  regular_adult_ticket_price * num_adults +
    child_ticket_price * num_children +
    children_concessions + adults_concessions -
    (adult_discount * num_discounted_adults) = total_cost :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_movie_ticket_price_l888_88888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_minimum_side_l888_88868

/-- Given a triangle ABC with area S and angle C, prove that side c is minimized when a = b = √(2S / sin C) -/
theorem triangle_minimum_side (S : ℝ) (C : ℝ) (hS : S > 0) (hC : 0 < C ∧ C < π) :
  ∃ (a b : ℝ),
    (a > 0 ∧ b > 0) ∧
    (1/2 * a * b * Real.sin C = S) ∧
    (∀ (x y : ℝ), x > 0 → y > 0 → 1/2 * x * y * Real.sin C = S →
      x^2 + y^2 - 2*x*y*Real.cos C ≥ 2*S / Real.sin C) ∧
    (a = Real.sqrt (2*S / Real.sin C) ∧ b = Real.sqrt (2*S / Real.sin C)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_minimum_side_l888_88868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compare_exponents_l888_88863

theorem compare_exponents : 
  let y₁ : ℝ := (4 : ℝ)^(0.9 : ℝ)
  let y₂ : ℝ := (8 : ℝ)^(0.48 : ℝ)
  let y₃ : ℝ := ((1/2) : ℝ)^((-1.5) : ℝ)
  y₁ > y₃ ∧ y₃ > y₂ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compare_exponents_l888_88863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_sum_ratio_l888_88840

noncomputable section

-- Define the angles in radians
def angle_10 : ℝ := 10 * Real.pi / 180
def angle_50 : ℝ := 50 * Real.pi / 180
def angle_60 : ℝ := 60 * Real.pi / 180
def angle_120 : ℝ := 120 * Real.pi / 180

theorem tangent_sum_ratio :
  (Real.tan angle_10 + Real.tan angle_50 + Real.tan angle_120) / (Real.tan angle_10 * Real.tan angle_50) = -Real.sqrt 3 :=
by
  have h1 : Real.tan angle_60 = (Real.tan angle_10 + Real.tan angle_50) / (1 - Real.tan angle_10 * Real.tan angle_50) := by sorry
  have h2 : Real.tan angle_60 = Real.sqrt 3 := by sorry
  have h3 : Real.tan angle_120 = -Real.sqrt 3 := by sorry
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_sum_ratio_l888_88840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tomato_price_calculation_l888_88829

/-- The original price per pound of tomatoes -/
noncomputable def original_price : ℝ := sorry

/-- The percentage of tomatoes that were not ruined -/
def remaining_percentage : ℝ := 0.85

/-- The desired profit percentage -/
def profit_percentage : ℝ := 0.08

/-- The selling price per pound of the remaining tomatoes -/
def selling_price : ℝ := 1.0165

theorem tomato_price_calculation : 
  ∃ (ε : ℝ), abs (original_price - 0.9294) < ε ∧ ε > 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tomato_price_calculation_l888_88829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_z_equals_one_l888_88830

noncomputable def z : ℂ := (Real.sqrt 2 + Complex.I ^ 2019) / (Real.sqrt 2 + Complex.I)

theorem abs_z_equals_one : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_z_equals_one_l888_88830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_multiples_of_13_and_7_l888_88812

theorem three_digit_multiples_of_13_and_7 : 
  (Finset.filter (fun n => n % 13 = 0 ∧ n % 7 = 0) (Finset.range 900)).card = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_multiples_of_13_and_7_l888_88812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_ABC_l888_88849

-- Define the triangle ABC
noncomputable def triangle_ABC (a b c : ℝ) (A B C : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  A > 0 ∧ B > 0 ∧ C > 0 ∧
  A + B + C = Real.pi

-- Define the area of a triangle using the sine formula
noncomputable def area (a b C : ℝ) : ℝ := (1/2) * a * b * Real.sin C

-- State the theorem
theorem area_of_triangle_ABC :
  ∀ (a b c A B C : ℝ),
  triangle_ABC a b c A B C →
  a = 1 →
  b = 2 →
  C = 5/6 * Real.pi →
  area a b C = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_ABC_l888_88849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_exists_in_interval_l888_88853

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := 2^x + (1/4) * x - 5

-- Theorem statement
theorem root_exists_in_interval :
  ∃ x ∈ Set.Ioo 2 3, f x = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_exists_in_interval_l888_88853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_t_l888_88856

-- We don't need to redefine Vector as it's already defined in Mathlib
-- Define vector addition
def add_vectors (v w : Fin 2 → ℝ) : Fin 2 → ℝ :=
  λ i => v i + w i

-- Define scalar multiplication
def scalar_mult (k : ℝ) (v : Fin 2 → ℝ) : Fin 2 → ℝ :=
  λ i => k * v i

-- Define parallel vectors
def parallel (v w : Fin 2 → ℝ) : Prop :=
  ∃ k : ℝ, v = scalar_mult k w

theorem parallel_vectors_t (t : ℝ) : 
  let a : Fin 2 → ℝ := λ i => if i = 0 then -1 else 1
  let b : Fin 2 → ℝ := λ i => if i = 0 then 3 else t
  parallel b (add_vectors a b) → t = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_t_l888_88856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_second_quadrant_l888_88892

theorem angle_in_second_quadrant (x : ℝ) :
  Real.tan x < 0 → Real.sin x - Real.cos x > 0 → x ∈ Set.Ioo (π / 2) π := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_second_quadrant_l888_88892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_with_half_perimeter_l888_88898

noncomputable section

variable {a b c s : ℝ}

/-- Triangle inequality: any side is less than the sum of the other two sides -/
axiom triangle_inequality (a b c : ℝ) : 0 < a ∧ 0 < b ∧ 0 < c → a < b + c ∧ b < a + c ∧ c < a + b

/-- Definition of s as half the perimeter -/
def half_perimeter (a b c : ℝ) : ℝ := (a + b + c) / 2

/-- Main theorem: inequality for triangle sides and half-perimeter -/
theorem triangle_inequality_with_half_perimeter 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_s : s = half_perimeter a b c) :
  (1 / (s - a)) + (1 / (s - b)) + (1 / (s - c)) ≥ 2 * (1/a + 1/b + 1/c) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_with_half_perimeter_l888_88898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_l888_88800

theorem sum_of_coefficients (p : ℕ) (hp : p ≥ 3) : 
  let f : Polynomial ℚ := (Finset.range p).sum (λ i => X^i) ^ (p + 1)
  let coeff := λ i => f.coeff i
  (Finset.range (p + 1)).sum (λ i => coeff (i * p)) = p^(p + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_l888_88800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_cuts_formula_expected_cuts_for_2016_l888_88832

/-- The expected number of cuts for a string of length x -/
noncomputable def expectedCuts (x : ℝ) : ℝ :=
  1 + Real.log x

/-- Theorem: The expected number of cuts for a string of length x is 1 + log(x) -/
theorem expected_cuts_formula (x : ℝ) (hx : x ≥ 1) :
  expectedCuts x = 1 + Real.log x :=
by sorry

/-- The expected number of cuts for a string of length 2016 millimeters -/
noncomputable def expectedCutsFor2016 : ℝ :=
  expectedCuts 2016

/-- Theorem: The expected number of cuts for a string of length 2016 millimeters
    is equal to 1 + log(2016) -/
theorem expected_cuts_for_2016 :
  expectedCutsFor2016 = 1 + Real.log 2016 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_cuts_formula_expected_cuts_for_2016_l888_88832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mixture_carbonated_water_percentage_l888_88883

/-- Represents a solution with lemonade and carbonated water -/
structure Solution where
  lemonade : ℝ
  carbonated_water : ℝ
  volume : ℝ

/-- Represents the mixture of two solutions -/
structure Mixture where
  solution1 : Solution
  solution2 : Solution
  total_volume : ℝ
  carbonated_water_percent : ℝ

/-- Approximate equality for real numbers -/
def approx_eq (x y : ℝ) (ε : ℝ) : Prop := abs (x - y) < ε

theorem mixture_carbonated_water_percentage
  (s1 : Solution)
  (s2 : Solution)
  (m : Mixture)
  (h1 : s1.lemonade = 0.2 * s1.volume)
  (h2 : s1.carbonated_water = 0.8 * s1.volume)
  (h3 : s2.lemonade = 0.45 * s2.volume)
  (h4 : m.solution1 = s1)
  (h5 : m.solution2 = s2)
  (h6 : m.carbonated_water_percent = 0.6)
  (h7 : s1.volume = 0.1999999999999997 * m.total_volume)
  : approx_eq (s2.carbonated_water / s2.volume) 0.55 0.0000001 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mixture_carbonated_water_percentage_l888_88883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_and_m_range_l888_88860

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (2 - x) + 1 / Real.sqrt (x^2 - 1)

-- Define the set A
def A (m : ℝ) : Set ℝ := {x | m - 2 < x ∧ x < 2*m}

-- Define the domain D
def D : Set ℝ := {x | (1 < x ∧ x ≤ 2) ∨ x ≤ -1}

-- Theorem statement
theorem domain_and_m_range :
  (∀ m : ℝ, A m ⊆ D) →
  (∀ x : ℝ, x ∈ D ↔ (1 < x ∧ x ≤ 2) ∨ x ≤ -1) ∧
  (∀ m : ℝ, (∀ x : ℝ, x ∈ A m → x ∈ D) ↔ m ≤ -1/2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_and_m_range_l888_88860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_l888_88843

noncomputable section

-- Define the vertices of the triangle
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (40, 0)
def C : ℝ × ℝ := (20, 30)

-- Define the midpoints
def D : ℝ × ℝ := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
def E : ℝ × ℝ := ((C.1 + A.1) / 2, (C.2 + A.2) / 2)
def F : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Define the orthocenter
def O : ℝ × ℝ := (20, 15)

-- Define the area of the base triangle
def baseArea : ℝ := (1 / 2) * abs (B.1 * C.2 - B.2 * C.1)

-- Define the height of the pyramid
def pyramidHeight : ℝ := O.2 - B.2

-- Theorem: The volume of the pyramid is 3000 cubic units
theorem pyramid_volume : (1 / 3) * baseArea * pyramidHeight = 3000 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_l888_88843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_n_separation_sequence_l888_88842

/-- Represents a sequence of 3972 terms where each number from 1 to 1986 appears twice -/
def Sequence := Fin 3972 → Fin 1986

/-- Predicate to check if a sequence satisfies the condition that each number appears exactly twice -/
def IsValidSequence (s : Sequence) : Prop :=
  ∀ n : Fin 1986, (∃ i j : Fin 3972, i ≠ j ∧ s i = n ∧ s j = n) ∧
    (∀ k : Fin 3972, s k = n → (∃ i j : Fin 3972, (k = i ∨ k = j) ∧ i ≠ j ∧ s i = n ∧ s j = n))

/-- Predicate to check if a sequence satisfies the n-separation condition -/
def SatisfiesNSeparation (s : Sequence) : Prop :=
  ∀ n : Fin 1986, ∃ i j : Fin 3972, i < j ∧ s i = n ∧ s j = n ∧ j - i = n.val + 1

/-- Theorem stating that no valid sequence can satisfy the n-separation condition -/
theorem no_valid_n_separation_sequence :
  ¬∃ s : Sequence, IsValidSequence s ∧ SatisfiesNSeparation s := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_n_separation_sequence_l888_88842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_l888_88836

/-- Represents a geometric sequence -/
structure GeometricSequence where
  firstTerm : ℝ
  commonRatio : ℝ

/-- Computes the sum of the first n terms of a geometric sequence -/
noncomputable def sumOfTerms (seq : GeometricSequence) (n : ℕ) : ℝ :=
  seq.firstTerm * (1 - seq.commonRatio^n) / (1 - seq.commonRatio)

theorem geometric_sequence_sum (seq : GeometricSequence) :
  sumOfTerms seq 2011 = 200 ∧ 
  sumOfTerms seq 4022 = 380 → 
  sumOfTerms seq 6033 = 542 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_l888_88836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_squares_area_ratio_l888_88884

/-- The ratio of areas of squares inscribed in an ellipse and a circle -/
theorem inscribed_squares_area_ratio 
  (r : ℝ) 
  (h : r > 0) :
  (min r (r / 2))^2 / (2 * r^2) = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_squares_area_ratio_l888_88884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_platform_passing_time_l888_88804

/-- The time (in seconds) it takes for a train to pass a platform -/
noncomputable def train_passing_time (train_length platform_length : ℝ) (train_speed_kmph : ℝ) : ℝ :=
  let total_distance := train_length + platform_length
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  total_distance / train_speed_mps

/-- Theorem stating the time it takes for a specific train to pass a specific platform -/
theorem train_platform_passing_time :
  let train_length := (120 : ℝ)
  let platform_length := (240 : ℝ)
  let train_speed_kmph := (60 : ℝ)
  let result := train_passing_time train_length platform_length train_speed_kmph
  ∃ ε > 0, |result - 21.6| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_platform_passing_time_l888_88804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_three_and_nine_l888_88808

theorem power_of_three_and_nine : (3^4 * 9^8 : ℚ)^(1/4) = 243 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_three_and_nine_l888_88808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l888_88896

/-- Given an ellipse with semi-major axis a and semi-minor axis b, 
    if the product of slopes from the left vertex to two symmetric points is 1/3, 
    then the eccentricity is √6/3 -/
theorem ellipse_eccentricity (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  (∃ (x y : ℝ), 
    x^2/a^2 + y^2/b^2 = 1 ∧ 
    (y / (x + a)) * (y / (a - x)) = 1/3) →
  Real.sqrt (1 - b^2/a^2) = Real.sqrt 6 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l888_88896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_even_heads_fair_coin_prob_even_heads_biased_coin_l888_88894

noncomputable section

/-- Probability of getting an even number of heads in n coin tosses -/
def prob_even_heads (p : ℝ) (n : ℕ) : ℝ :=
  (1 + (1 - 2*p)^n) / 2

/-- A fair coin has probability 0.5 of landing heads -/
def is_fair_coin (p : ℝ) : Prop := p = 0.5

/-- A biased coin has probability p of landing heads, where 0 < p < 1 -/
def is_biased_coin (p : ℝ) : Prop := 0 < p ∧ p < 1

theorem prob_even_heads_fair_coin (p : ℝ) (n : ℕ) (h : is_fair_coin p) :
  prob_even_heads p n = 0.5 := by sorry

theorem prob_even_heads_biased_coin (p : ℝ) (n : ℕ) (h : is_biased_coin p) :
  prob_even_heads p n = (1 + (1 - 2*p)^n) / 2 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_even_heads_fair_coin_prob_even_heads_biased_coin_l888_88894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_m_values_l888_88866

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {a b c d : ℝ} (h : a ≠ 0 ∧ c ≠ 0) :
  (∀ x y : ℝ, a * x + b * y = 0 ↔ c * x + d * y = 0) ↔ a / b = c / d

/-- Definition of line l₁ -/
def l₁ (m : ℝ) : ℝ → ℝ → Prop :=
  λ x y ↦ x + m^2 * y + 6 = 0

/-- Definition of line l₂ -/
def l₂ (m : ℝ) : ℝ → ℝ → Prop :=
  λ x y ↦ (m - 2) * x + 3 * m * y + 2 * m = 0

theorem parallel_lines_m_values :
  ∀ m : ℝ, (∀ x y : ℝ, l₁ m x y ↔ l₂ m x y) → m = 0 ∨ m = -1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_m_values_l888_88866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_real_roots_of_sum_of_powers_l888_88855

theorem max_real_roots_of_sum_of_powers (n : ℕ+) :
  ∃ (max : ℕ), max = 1 ∧ 
  (∀ (k : ℕ), (∃ (roots : Finset ℝ), 
    (∀ x ∈ roots, (Finset.range (n + 1)).sum (λ i => x^(n - i)) = 0) ∧
    k = roots.card) → k ≤ max) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_real_roots_of_sum_of_powers_l888_88855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l888_88899

def A : Set ℝ := {x | 1/2 ≤ (2 : ℝ)^x ∧ (2 : ℝ)^x ≤ 4}
def B : Set ℝ := {0, 1, 2, 3}

theorem intersection_of_A_and_B : A ∩ B = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l888_88899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_solution_f_inequality_range_l888_88803

def f (a x : ℝ) : ℝ := |x - a| - |x - 3|

theorem f_inequality_solution (x : ℝ) : 
  {x | f (-1) x ≥ 2} = {x | x ≥ 2} := by sorry

theorem f_inequality_range (a : ℝ) : 
  (∃ x, f a x ≤ -a/2) ↔ a ∈ Set.Iic 2 ∪ Set.Ici 6 := by sorry

#check f_inequality_solution
#check f_inequality_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_solution_f_inequality_range_l888_88803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_divisibility_l888_88872

noncomputable def polyComp (P : Polynomial ℝ) (n : ℕ) : Polynomial ℝ :=
  match n with
  | 0 => Polynomial.X
  | n + 1 => P.comp (polyComp P n)

noncomputable def Q (P : Polynomial ℝ) (n : ℕ) : Polynomial ℝ :=
  (polyComp P n) - Polynomial.X

theorem polynomial_divisibility (P : Polynomial ℝ) (h : P ≠ Polynomial.X) :
  ∀ n : ℕ, ∃ R : Polynomial ℝ, Q P n = R * (Q P 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_divisibility_l888_88872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ten_numbers_theorem_l888_88854

theorem ten_numbers_theorem :
  ∃ (S : Finset ℕ),
    Finset.card S = 10 ∧
    (∀ x y, x ∈ S → y ∈ S → x ≠ y → x ≠ y) ∧
    (∃ A : Finset ℕ, A ⊆ S ∧ Finset.card A = 3 ∧ ∀ x ∈ A, x % 5 = 0) ∧
    (∃ B : Finset ℕ, B ⊆ S ∧ Finset.card B = 4 ∧ ∀ x ∈ B, x % 4 = 0) ∧
    Finset.sum S id < 75 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ten_numbers_theorem_l888_88854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_trees_formula_total_trees_when_A_plants_12_l888_88852

/-- The number of trees planted by student B -/
def x : ℝ := sorry

/-- The number of trees planted by student A -/
def trees_A (x : ℝ) : ℝ := 1.2 * x

/-- The number of trees planted by student C -/
def trees_C (x : ℝ) : ℝ := trees_A x - 2

/-- The total number of trees planted by A, B, and C -/
def total_trees (x : ℝ) : ℝ := x + trees_A x + trees_C x

theorem total_trees_formula (x : ℝ) : 
  total_trees x = 3.4 * x - 2 := by
  -- Proof steps would go here
  sorry

theorem total_trees_when_A_plants_12 : 
  total_trees (12 / 1.2) = 32 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_trees_formula_total_trees_when_A_plants_12_l888_88852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_photos_per_album_equal_l888_88841

theorem photos_per_album_equal (total_photos : ℕ) (first_batch_photos : ℕ) (first_batch_albums : ℕ) (second_batch_albums : ℕ) 
  (h1 : total_photos = 4500)
  (h2 : first_batch_photos = 1500)
  (h3 : first_batch_albums = 30)
  (h4 : second_batch_albums = 60)
  (h5 : first_batch_photos ≤ total_photos) :
  first_batch_photos / first_batch_albums = (total_photos - first_batch_photos) / second_batch_albums ∧ 
  first_batch_photos / first_batch_albums = 50 := by
  sorry

#check photos_per_album_equal

end NUMINAMATH_CALUDE_ERRORFEEDBACK_photos_per_album_equal_l888_88841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pastrami_price_l888_88821

/-- The cost of a Reuben sandwich -/
def reuben_cost : ℝ := sorry

/-- The cost of a pastrami sandwich -/
def pastrami_cost : ℝ := sorry

/-- Pastrami costs $2 more than Reuben -/
axiom pastrami_premium : pastrami_cost = reuben_cost + 2

/-- Total sales equation -/
axiom total_sales : 10 * reuben_cost + 5 * pastrami_cost = 55

/-- Theorem: The cost of a pastrami sandwich is $5 -/
theorem pastrami_price : pastrami_cost = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pastrami_price_l888_88821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_divisor_of_four_consecutive_odd_integers_l888_88815

theorem largest_divisor_of_four_consecutive_odd_integers : ∃ (n : ℕ), 
  (∀ (x : ℤ), (n : ℤ) ∣ (x * (x + 2) * (x + 4) * (x + 6))) ∧ 
  (∀ (m : ℕ), m > n → ∃ (y : ℤ), ¬((m : ℤ) ∣ (y * (y + 2) * (y + 4) * (y + 6)))) :=
by
  -- We claim that 48 is the largest such divisor
  use 48
  constructor
  · -- Prove that 48 divides the product for all x
    intro x
    sorry -- Proof omitted
  · -- Prove that for any m > 48, there exists a y for which m doesn't divide the product
    intro m hm
    sorry -- Proof omitted

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_divisor_of_four_consecutive_odd_integers_l888_88815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_roller_coaster_duration_l888_88828

def roller_coaster_durations : List Nat := [
  28, 28, 50, 55, 60, 62, 70, 140, 145, 155, 163, 165, 170, 180, 180, 180, 185, 210, 216, 240, 250
]

theorem median_roller_coaster_duration : 
  List.get! roller_coaster_durations 10 = 163 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_roller_coaster_duration_l888_88828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_g_l888_88861

open Real

-- Define the function g(x) on the open interval (0, e^2]
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := log x + a / x - 2

-- State the theorem
theorem min_value_of_g (a : ℝ) :
  (∀ x, 0 < x ∧ x ≤ exp 2 → g a x ≥ 2) ∧
  (∃ x, 0 < x ∧ x ≤ exp 2 ∧ g a x = 2) ↔
  a = 2 * exp 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_g_l888_88861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oplus_range_l888_88810

-- Define the ⊕ operation
noncomputable def oplus (a b : ℝ) : ℝ :=
  if a > b then a else b

-- State the theorem
theorem oplus_range (x : ℝ) :
  oplus (2 * x + 1) (x + 3) = x + 3 → x < 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oplus_range_l888_88810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interval_of_decrease_l888_88893

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.log (-x^2 - 2*x + 8)

-- Define the domain
def domain : Set ℝ := {x : ℝ | -4 < x ∧ x < 2}

-- Theorem statement
theorem interval_of_decrease :
  ∀ (x₁ x₂ : ℝ), x₁ ∈ domain → x₂ ∈ domain → -1 < x₁ → x₁ < x₂ → x₂ < 2 → f x₁ > f x₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interval_of_decrease_l888_88893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_and_trajectory_l888_88833

-- Define the parabola G
def parabola (p : ℝ) (x y : ℝ) : Prop := x^2 = 2*p*y

-- Define the line m
def line_m (x y : ℝ) : Prop := y = (1/2) * (x + 4)

-- Define point A
def point_A : ℝ × ℝ := (-4, 0)

-- Define the relationship between vectors AC and AB
def vector_relation (xb yb xc yc : ℝ) : Prop :=
  (xc + 4, yc) = 4 • (xb + 4, yb)

-- Define the midpoint M
def midpoint_M (xb yb xc yc x y : ℝ) : Prop :=
  x = (xb + xc) / 2 ∧ y = (yb + yc) / 2

-- State the theorem
theorem parabola_and_trajectory :
  ∃ (p xb yb xc yc : ℝ),
    p > 0 ∧
    parabola p xb yb ∧
    parabola p xc yc ∧
    line_m xb yb ∧
    line_m xc yc ∧
    vector_relation xb yb xc yc →
  (∀ x y : ℝ, parabola 2 x y) ∧
  (∃ (x y : ℝ), (x > 0 ∨ x < -8) ∧
    midpoint_M xb yb xc yc x y ∧
    y = (1/2) * x^2 + 2*x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_and_trajectory_l888_88833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_example_l888_88806

/-- The distance from a point to a line in 2D space -/
noncomputable def distance_point_to_line (x y a b c : ℝ) : ℝ :=
  abs (a * x + b * y + c) / Real.sqrt (a^2 + b^2)

/-- Theorem: The distance from the point (1, 1) to the line x + y - 1 = 0 is √2/2 -/
theorem distance_point_to_line_example : distance_point_to_line 1 1 1 1 (-1) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_example_l888_88806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_center_division_equality_l888_88837

-- Define a geometric figure
structure GeometricFigure where
  -- Add necessary properties to define the figure
  is_symmetric : Bool

-- Define a 2D line (simplified representation)
structure Line2D where
  -- Simplified line representation
  dummy : Unit

-- Define a division of the figure
structure Division where
  figure : GeometricFigure
  line : Line2D

-- Define equality of parts
def equal_parts (d : Division) : Prop :=
  -- Placeholder for the condition of equal parts
  true

-- Define a vertical line through the center
def vertical_center_line (f : GeometricFigure) : Line2D :=
  -- Placeholder for the vertical line through the center
  { dummy := () }

-- Theorem statement
theorem vertical_center_division_equality (f : GeometricFigure) 
  (h : f.is_symmetric = true) : 
  equal_parts ⟨f, vertical_center_line f⟩ := by
  sorry

#check vertical_center_division_equality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_center_division_equality_l888_88837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_through_two_points_l888_88864

/-- The set of points (x, y) that form a line through two given points (x₁, y₁) and (x₂, y₂). -/
def line_through_points (p₁ p₂ : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ t : ℝ, p = (1 - t) • p₁ + t • p₂}

/-- Given two points (x₁, y₁) and (x₂, y₂) in a 2D plane, 
    the equation (y - y₁)(x₂ - x₁) = (x - x₁)(y₂ - y₁) 
    represents the line passing through these points. -/
theorem line_equation_through_two_points 
  (x₁ y₁ x₂ y₂ : ℝ) : 
  ∀ (x y : ℝ), (y - y₁) * (x₂ - x₁) = (x - x₁) * (y₂ - y₁) ↔ 
  (x, y) ∈ line_through_points (x₁, y₁) (x₂, y₂) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_through_two_points_l888_88864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l888_88889

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (3 - x^2) * Real.exp x

-- State the theorem
theorem f_increasing_on_interval :
  ∀ x ∈ Set.Ioo (-3 : ℝ) 1, 
    ∀ y ∈ Set.Ioo (-3 : ℝ) 1, 
      x < y → f x < f y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l888_88889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_sum_theorem_l888_88807

theorem base_sum_theorem (R₁ R₂ : ℕ) (h₁ : R₁ > 1) (h₂ : R₂ > 1) : 
  (4 * R₁ + 8) * (R₂^2 - 1) = (5 * R₂ + 9) * (R₁^2 - 1) →
  (8 * R₁ + 4) * (R₂^2 - 1) = (9 * R₂ + 5) * (R₁^2 - 1) →
  R₁ + R₂ = 24 := by
  intro h1 h2
  sorry

#check base_sum_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_sum_theorem_l888_88807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_ratio_range_l888_88822

-- Define the function f(x) = |e^x - 1|
noncomputable def f (x : ℝ) : ℝ := |Real.exp x - 1|

-- Define the theorem
theorem tangent_ratio_range (x₂ : ℝ) (h₁ : x₂ > 0) :
  let x₁ := -x₂
  let A := (x₁, f x₁)
  let B := (x₂, f x₂)
  let M := (0, (1 - Real.exp x₁) + x₁ * Real.exp x₁)
  let N := (0, (Real.exp x₂ - 1) - x₂ * Real.exp x₂)
  let AM := Real.sqrt (x₁^2 + (M.2 - A.2)^2)
  let BN := Real.sqrt (x₂^2 + (N.2 - B.2)^2)
  0 < AM / BN ∧ AM / BN < 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_ratio_range_l888_88822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diver_descent_time_l888_88857

/-- Calculates the time taken to reach a target depth given a constant descent rate. -/
noncomputable def time_to_reach_depth (descent_rate : ℝ) (target_depth : ℝ) : ℝ :=
  target_depth / descent_rate

/-- Theorem: A diver descending at 30 feet per minute will reach a depth of 2400 feet in 80 minutes. -/
theorem diver_descent_time :
  let descent_rate : ℝ := 30
  let target_depth : ℝ := 2400
  time_to_reach_depth descent_rate target_depth = 80 := by
  -- Unfold the definition of time_to_reach_depth
  unfold time_to_reach_depth
  -- Perform the division
  norm_num
  -- QED
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diver_descent_time_l888_88857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_sum_l888_88839

theorem square_root_sum (a : ℝ) (h1 : a > 0) (h2 : a + a⁻¹ = 3) :
  a^(1/2 : ℝ) + a^(-(1/2) : ℝ) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_sum_l888_88839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_count_l888_88805

/-- A point on the perimeter of the square -/
structure PerimeterPoint where
  x : ℤ
  y : ℤ
  on_perimeter : (x = 0 ∧ 0 ≤ y ∧ y ≤ 20) ∨ 
                 (x = 20 ∧ 0 ≤ y ∧ y ≤ 20) ∨ 
                 (y = 0 ∧ 0 ≤ x ∧ x ≤ 20) ∨ 
                 (y = 20 ∧ 0 ≤ x ∧ x ≤ 20)

/-- The set of 80 equally spaced points on the perimeter -/
def perimeter_points : Finset PerimeterPoint :=
  sorry

/-- Three non-collinear points selected from the perimeter points -/
structure TrianglePoints where
  P : PerimeterPoint
  Q : PerimeterPoint
  R : PerimeterPoint
  in_perimeter_points : P ∈ perimeter_points ∧ Q ∈ perimeter_points ∧ R ∈ perimeter_points
  non_collinear : sorry

/-- The centroid of a triangle -/
def centroid (t : TrianglePoints) : ℚ × ℚ :=
  ((t.P.x + t.Q.x + t.R.x) / 3, (t.P.y + t.Q.y + t.R.y) / 3)

/-- The set of all possible centroids -/
noncomputable def possible_centroids : Finset (ℚ × ℚ) :=
  sorry

theorem centroid_count : Finset.card possible_centroids = 3481 :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_count_l888_88805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_segment_theorem_l888_88869

/-- Point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Vector in 3D space -/
def Vector3D := Point3D

/-- Scalar multiplication for Vector3D -/
noncomputable def scalarMul (s : ℝ) (v : Vector3D) : Vector3D :=
  { x := s * v.x, y := s * v.y, z := s * v.z }

/-- Vector addition for Vector3D -/
noncomputable def vectorAdd (v1 v2 : Vector3D) : Vector3D :=
  { x := v1.x + v2.x, y := v1.y + v2.y, z := v1.z + v2.z }

/-- Definition of point P on line segment AB with given ratio -/
noncomputable def pointOnSegment (A B : Point3D) (ratio : ℝ) : Point3D :=
  { x := (ratio * B.x + A.x) / (ratio + 1),
    y := (ratio * B.y + A.y) / (ratio + 1),
    z := (ratio * B.z + A.z) / (ratio + 1) }

theorem point_on_segment_theorem (A B : Point3D) :
  let P := pointOnSegment A B (4 : ℝ)
  vectorAdd (scalarMul 0.2 A) (scalarMul 0.8 B) = P := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_segment_theorem_l888_88869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vending_machine_theorem_l888_88844

def vending_machine_problem (total_candies : ℕ) (candy_cost : ℚ) : Prop :=
  ∃ (orange apple grape strawberry : ℕ),
    orange + apple + grape + strawberry = total_candies ∧
    apple = 2 * orange ∧
    strawberry = 2 * grape ∧
    apple = 2 * strawberry ∧
    let min_candies := min (min orange apple) (min grape strawberry)
    (min_candies + 3 * 3) * candy_cost = 1.4

theorem vending_machine_theorem :
  vending_machine_problem 90 (1/10) := by
  sorry

#check vending_machine_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vending_machine_theorem_l888_88844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_line_l888_88809

-- Define the curve y = x^2 - ln(x)
noncomputable def curve (x : ℝ) : ℝ := x^2 - Real.log x

-- Define the line x - y - 4 = 0
def line (x y : ℝ) : Prop := x - y - 4 = 0

-- Define the distance function from a point to the line
noncomputable def distance_to_line (x y : ℝ) : ℝ := |x - y - 4| / Real.sqrt 2

-- Statement of the theorem
theorem min_distance_to_line :
  ∃ (x y : ℝ), x > 0 ∧ y = curve x ∧
  ∀ (x' y' : ℝ), x' > 0 → y' = curve x' →
  distance_to_line x y ≤ distance_to_line x' y' ∧
  distance_to_line x y = 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_line_l888_88809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_car_distance_l888_88876

/-- Prove that the distance the first car ran before taking the right turn is 25 km -/
theorem first_car_distance (total_distance : ℝ) (second_car_distance : ℝ) (final_gap : ℝ) 
  (h1 : total_distance = 150)
  (h2 : second_car_distance = 62)
  (h3 : final_gap = 38) :
  (total_distance - second_car_distance - final_gap) / 2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_car_distance_l888_88876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_coins_arbitrary_distribution_max_coins_equal_distribution_l888_88862

/-- Represents a distribution of warriors into squads -/
structure WarriorDistribution where
  squads : List Nat
  squad_sum : List.sum squads = 33

/-- Calculates coins Chernomor gets from a given distribution and coin allocation -/
def chernomor_coins (d : WarriorDistribution) (coin_allocation : List Nat) : Nat :=
  sorry

/-- Proves the maximum coins Chernomor can get with arbitrary distribution -/
theorem max_coins_arbitrary_distribution :
  ∃ (d : WarriorDistribution) (coin_allocation : List Nat),
    List.sum coin_allocation = 240 ∧
    chernomor_coins d coin_allocation = 31 ∧
    ∀ (d' : WarriorDistribution) (coin_allocation' : List Nat),
      List.sum coin_allocation' = 240 →
      chernomor_coins d' coin_allocation' ≤ 31 :=
by
  sorry

/-- Proves the maximum coins Chernomor can get with equal distribution -/
theorem max_coins_equal_distribution :
  ∃ (d : WarriorDistribution) (coins_per_squad : Nat),
    coins_per_squad * d.squads.length = 240 ∧
    chernomor_coins d (List.replicate d.squads.length coins_per_squad) = 30 ∧
    ∀ (d' : WarriorDistribution) (coins_per_squad' : Nat),
      coins_per_squad' * d'.squads.length = 240 →
      chernomor_coins d' (List.replicate d'.squads.length coins_per_squad') ≤ 30 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_coins_arbitrary_distribution_max_coins_equal_distribution_l888_88862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_interval_l888_88879

-- Define the function f(x) as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (-x^2 + 2*x + 3)

-- State the theorem
theorem f_decreasing_on_interval :
  ∀ x y, 1 < x ∧ x < y ∧ y < 3 → f x > f y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_interval_l888_88879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nick_needs_160_candy_bars_l888_88877

/-- Represents the fundraising problem --/
structure FundraisingProblem where
  candy_price : ℕ
  orange_price : ℕ
  goal : ℕ
  oranges_sold : ℕ

/-- The specific instance of the fundraising problem --/
def nicks_problem : FundraisingProblem :=
  { candy_price := 5,
    orange_price := 10,
    goal := 1000,
    oranges_sold := 20 }

/-- Calculates the number of candy bars needed to reach the fundraising goal --/
def candy_bars_needed (p : FundraisingProblem) : ℕ :=
  (p.goal - p.orange_price * p.oranges_sold) / p.candy_price

/-- Theorem stating that Nick needs to sell 160 candy bars to reach his goal --/
theorem nick_needs_160_candy_bars :
  candy_bars_needed nicks_problem = 160 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nick_needs_160_candy_bars_l888_88877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l888_88850

/-- Line l with parametric equation x = t + 1, y = √3 * t + 1 -/
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (t + 1, Real.sqrt 3 * t + 1)

/-- Curve C with Cartesian equation y² = 2x -/
def curve_C (x y : ℝ) : Prop := y^2 = 2 * x

/-- Line l' parallel to l passing through (2,0) -/
noncomputable def line_l' (s : ℝ) : ℝ × ℝ := (2 + s / 2, Real.sqrt 3 * s / 2)

/-- Theorem stating the distance between intersection points -/
theorem intersection_distance :
  ∃ (s₁ s₂ : ℝ),
    let (x₁, y₁) := line_l' s₁
    let (x₂, y₂) := line_l' s₂
    curve_C x₁ y₁ ∧ curve_C x₂ y₂ ∧
    Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) = 4 * Real.sqrt 13 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l888_88850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_area_max_triangle_area_l888_88811

-- Define the curve C
noncomputable def C (x : ℝ) : ℝ := 4 - x^2

-- Define points A and B
def A : ℝ := -2
def B : ℝ := 2

-- Define point P
noncomputable def P (t : ℝ) : ℝ × ℝ := (t, 0)

-- Define point Q
noncomputable def Q (t : ℝ) : ℝ × ℝ := (t, C t)

-- Area of triangle APQ
noncomputable def S (t : ℝ) : ℝ := (1/2) * (t + 2) * (C t)

-- Theorem statements
theorem isosceles_right_triangle_area :
  ∃ t : ℝ, A < t ∧ t < B ∧ 
  (P t).1 - A = (Q t).2 - (P t).2 ∧
  (P t).1 - A = (Q t).1 - (P t).1 ∧
  S t = 9/2 := by sorry

theorem max_triangle_area :
  ∃ t : ℝ, A < t ∧ t < B ∧
  (∀ u : ℝ, A < u ∧ u < B → S u ≤ S t) ∧
  S t = 128/27 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_area_max_triangle_area_l888_88811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_school_is_31_25_l888_88873

/-- Represents Johnny's journey to and from school -/
structure JourneyData where
  total_time : ℝ
  jog_speed : ℝ
  jog_time : ℝ
  bike_speed : ℝ
  bike_time : ℝ
  bus_speed : ℝ
  walk_speed : ℝ
  skateboard_speed : ℝ
  bus_skateboard_time : ℝ

/-- Calculates the distance to school based on the given journey data -/
noncomputable def distance_to_school (data : JourneyData) : ℝ :=
  let jog_distance := data.jog_speed * data.jog_time
  let bike_distance := data.bike_speed * data.bike_time
  let bus_time := data.total_time / 2 - data.jog_time - data.bike_time
  let bus_distance := data.bus_speed * bus_time
  jog_distance + bike_distance + bus_distance

/-- Theorem stating that the distance to school is 31.25 miles -/
theorem distance_to_school_is_31_25 (data : JourneyData) : 
  data.total_time = 2 ∧ 
  data.jog_speed = 5 ∧ 
  data.jog_time = 0.25 ∧ 
  data.bike_speed = 10 ∧ 
  data.bike_time = 0.5 ∧ 
  data.bus_speed = 20 ∧ 
  data.walk_speed = 3 ∧ 
  data.skateboard_speed = 8 ∧ 
  data.bus_skateboard_time = 1.25 →
  distance_to_school data = 31.25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_school_is_31_25_l888_88873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l888_88802

open Real

/-- The function f as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt 3 * sin x * cos x - cos (π + 2 * x)

/-- Triangle ABC with given properties -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  C : ℝ
  h1 : f C = 1
  h2 : c = Real.sqrt 3
  h3 : a + b = 2 * Real.sqrt 3

/-- The theorem stating the area of the triangle -/
theorem triangle_area (t : Triangle) : 
  (1/2) * t.a * t.b * sin t.C = (3 * Real.sqrt 3) / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l888_88802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lunch_class_size_l888_88867

/-- Represents a class of students with the given lunch hosting properties. -/
structure LunchClass (n : ℕ) where
  /-- For any two students, at least one has had lunch at the other's home. -/
  lunch_relation : ∀ (i j : Fin n), i ≠ j → 
    (∃ (has_lunched_at : Fin n → Fin n → Prop), has_lunched_at i j ∨ has_lunched_at j i)
  /-- Each student has hosted exactly one quarter of the students at whose home they have had lunch. -/
  host_ratio : ∀ (i : Fin n), ∃ (num_hosted num_lunched_at : Fin n → ℕ),
    4 * (num_hosted i) = num_lunched_at i

/-- The theorem stating the possible values of n for a LunchClass. -/
theorem lunch_class_size (n : ℕ) (c : LunchClass n) : n % 7 = 0 ∨ n % 7 = 1 := by
  sorry

#check lunch_class_size

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lunch_class_size_l888_88867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixteen_twos_in_sequence_l888_88834

/-- Count the number of occurrences of a digit in a natural number -/
def countDigit (n : ℕ) (d : ℕ) : ℕ := sorry

/-- Check if a sequence of natural numbers is consecutive -/
def isConsecutive (seq : List ℕ) : Prop := sorry

/-- The sequence of 7 consecutive natural numbers starting from 2215 -/
def sequenceNumbers : List ℕ := [2215, 2216, 2217, 2218, 2219, 2220, 2221]

theorem sixteen_twos_in_sequence :
  isConsecutive sequenceNumbers ∧ 
  sequenceNumbers.length = 7 ∧
  (sequenceNumbers.map (fun n => countDigit n 2)).sum = 16 := by
  sorry

#eval sequenceNumbers

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixteen_twos_in_sequence_l888_88834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_sequence_properties_l888_88875

-- Define the original function
def y (x : ℝ) : ℝ := x^2 - 2*x - 11

-- Define the tangent line function f
def f (x : ℝ) : ℝ := 2*x - 15

-- Define the sequence a_n
def a (n : ℕ) : ℝ := f n

-- Define the sum of the first n terms
def S (n : ℕ) : ℝ := n^2 - 14*n

theorem tangent_line_and_sequence_properties :
  (∀ x, (deriv y) 2 = (deriv f) x) ∧ 
  (∀ n, a n = 2*n - 15) ∧
  (∀ n, S n = n^2 - 14*n) ∧
  (∃ n : ℕ, ∀ m : ℕ, n * S n ≤ m * S m) ∧
  (9 * S 9 = -405) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_sequence_properties_l888_88875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reporters_covering_local_politics_l888_88848

theorem reporters_covering_local_politics (total_reporters : ℕ) 
  (politics_not_local : ℝ) (not_politics : ℝ) : ℝ :=
  by
  -- Condition 1: 25% of reporters who cover politics do not cover local politics
  have h1 : politics_not_local = 0.25 := by sorry
  -- Condition 2: 60% of reporters do not cover politics
  have h2 : not_politics = 0.60 := by sorry
  -- Theorem: Percentage of reporters covering local politics in country x is 30%
  have percentage_local_politics : ℝ := 0.30
  
  -- Proof
  sorry

#check reporters_covering_local_politics

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reporters_covering_local_politics_l888_88848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_for_m_l888_88831

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := 3 * x^2 - 1/x + 4
noncomputable def g (x m : ℝ) : ℝ := x^2 - m

-- State the theorem
theorem solve_for_m : 
  ∃ m : ℝ, (f 3 - g 3 m = 5) ∧ (m = -50/3) := by
  -- Introduce m
  use (-50/3)
  
  -- Split the goal into two parts
  constructor
  
  -- Prove f 3 - g 3 m = 5
  · simp [f, g]
    -- The rest of the proof would go here
    sorry
  
  -- Prove m = -50/3
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_for_m_l888_88831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_property_l888_88851

theorem quadratic_root_property (x₁ x₂ : ℝ) : 
  x₁^2 + x₁ - 3 = 0 → x₂^2 + x₂ - 3 = 0 → x₁^2 - x₂ + 2023 = 2027 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_property_l888_88851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mango_tree_distance_l888_88820

/-- Represents a mango tree garden -/
structure MangoGarden where
  rows : ℕ
  columns : ℕ
  boundaryDistance : ℚ
  length : ℚ

/-- Calculates the distance between two mango trees in the garden -/
def treeDistance (garden : MangoGarden) : ℚ :=
  (garden.length - 2 * garden.boundaryDistance) / garden.rows.pred

/-- Theorem stating the distance between two mango trees in the given garden -/
theorem mango_tree_distance (garden : MangoGarden) 
  (h_rows : garden.rows = 10)
  (h_columns : garden.columns = 12)
  (h_boundary : garden.boundaryDistance = 5)
  (h_length : garden.length = 32) :
  treeDistance garden = 22 / 9 := by
  sorry

#eval treeDistance { rows := 10, columns := 12, boundaryDistance := 5, length := 32 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mango_tree_distance_l888_88820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_circular_arcs_l888_88885

-- Define a square in a 2D plane
structure Square where
  center : ℝ × ℝ
  side_length : ℝ
  side_length_pos : side_length > 0

-- Define a point in the 2D plane
def Point := ℝ × ℝ

-- Define the angle at which the square is seen from a point
noncomputable def viewing_angle (s : Square) (p : Point) : ℝ := sorry

-- Define the locus of points
def locus (s : Square) : Set Point :=
  {p : Point | viewing_angle s p = 30 * Real.pi / 180}

-- Theorem statement
theorem locus_is_circular_arcs (s : Square) :
  ∃ (centers : List Point) (radii : List ℝ),
    locus s = ⋃ (i : Fin (centers.length)), 
      {p : Point | ∃ (θ : ℝ), 
        p = (centers[i].1 + radii[i]! * Real.cos θ, centers[i].2 + radii[i]! * Real.sin θ) ∧ 
        θ ∈ Set.Icc 0 (2 * Real.pi)} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_circular_arcs_l888_88885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_current_calculation_l888_88813

-- Define complex numbers
def V : ℂ := 3 + 2 * Complex.I
def Z : ℂ := 2 - 5 * Complex.I

-- Define the current I
noncomputable def I : ℂ := V / Z

-- Theorem statement
theorem current_calculation :
  I = -4/29 + 19/29 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_current_calculation_l888_88813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tile_drawing_probability_l888_88897

/-- Represents a box of numbered tiles -/
structure TileBox where
  tiles : Finset ℕ

/-- The probability of an event occurring when drawing from a TileBox -/
noncomputable def probability (box : TileBox) (event : ℕ → Prop) [DecidablePred event] : ℚ :=
  (box.tiles.filter event).card / box.tiles.card

theorem tile_drawing_probability : 
  let box_a : TileBox := ⟨Finset.range 25⟩
  let box_b : TileBox := ⟨Finset.range 32 \ Finset.range 12⟩
  let event_a (n : ℕ) := n ≤ 18
  let event_b (n : ℕ) := n % 2 = 1 ∨ n > 26
  probability box_a event_a * probability box_b event_b = 117 / 250 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tile_drawing_probability_l888_88897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regression_analysis_for_height_weight_l888_88823

/-- Represents a person's physical measurements -/
structure PersonMeasurements where
  height : ℝ
  weight : ℝ

/-- Represents different analysis methods -/
inductive AnalysisMethod
  | ResidualAnalysis
  | RegressionAnalysis
  | IsoplethBarChart
  | IndependenceTest

/-- Determines if two variables have a linear relationship -/
def hasLinearRelationship (x y : PersonMeasurements → ℝ) : Prop := sorry

/-- Determines the appropriate analysis method for given measurements -/
def appropriateAnalysisMethod (measurements : PersonMeasurements → ℝ × ℝ) : AnalysisMethod := sorry

/-- Theorem stating that regression analysis is appropriate for analyzing height and weight -/
theorem regression_analysis_for_height_weight :
  let measurements := fun (p : PersonMeasurements) => (p.height, p.weight)
  hasLinearRelationship (fun p => p.height) (fun p => p.weight) →
  appropriateAnalysisMethod measurements = AnalysisMethod.RegressionAnalysis := by
  sorry

#check regression_analysis_for_height_weight

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regression_analysis_for_height_weight_l888_88823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_increase_l888_88845

theorem rectangle_area_increase (L B : ℝ) (hL : L > 0) (hB : B > 0) : 
  (1.3 * 1.45 - 1) * 100 = 88.5 := by
  sorry

#eval (1.3 * 1.45 - 1) * 100

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_increase_l888_88845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_of_three_primes_l888_88814

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- A function that checks if a set of three natural numbers uses each digit from 1 to 9 exactly once -/
def usesAllDigitsOnce (a b c : ℕ) : Prop :=
  ∃ (d1 d2 d3 d4 d5 d6 d7 d8 d9 : ℕ),
    d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d1 ≠ d5 ∧ d1 ≠ d6 ∧ d1 ≠ d7 ∧ d1 ≠ d8 ∧ d1 ≠ d9 ∧
    d2 ≠ d3 ∧ d2 ≠ d4 ∧ d2 ≠ d5 ∧ d2 ≠ d6 ∧ d2 ≠ d7 ∧ d2 ≠ d8 ∧ d2 ≠ d9 ∧
    d3 ≠ d4 ∧ d3 ≠ d5 ∧ d3 ≠ d6 ∧ d3 ≠ d7 ∧ d3 ≠ d8 ∧ d3 ≠ d9 ∧
    d4 ≠ d5 ∧ d4 ≠ d6 ∧ d4 ≠ d7 ∧ d4 ≠ d8 ∧ d4 ≠ d9 ∧
    d5 ≠ d6 ∧ d5 ≠ d7 ∧ d5 ≠ d8 ∧ d5 ≠ d9 ∧
    d6 ≠ d7 ∧ d6 ≠ d8 ∧ d6 ≠ d9 ∧
    d7 ≠ d8 ∧ d7 ≠ d9 ∧
    d8 ≠ d9 ∧
    d1 ∈ Finset.range 10 \ {0} ∧ d2 ∈ Finset.range 10 \ {0} ∧ d3 ∈ Finset.range 10 \ {0} ∧
    d4 ∈ Finset.range 10 \ {0} ∧ d5 ∈ Finset.range 10 \ {0} ∧ d6 ∈ Finset.range 10 \ {0} ∧
    d7 ∈ Finset.range 10 \ {0} ∧ d8 ∈ Finset.range 10 \ {0} ∧ d9 ∈ Finset.range 10 \ {0} ∧
    a = 100 * d1 + 10 * d2 + d3 ∧
    b = 100 * d4 + 10 * d5 + d6 ∧
    c = 100 * d7 + 10 * d8 + d9

theorem smallest_sum_of_three_primes :
  ∃ (a b c : ℕ),
    isPrime a ∧ isPrime b ∧ isPrime c ∧
    a ≠ b ∧ a ≠ c ∧ b ≠ c ∧
    usesAllDigitsOnce a b c ∧
    a + b + c = 999 ∧
    (∀ (x y z : ℕ),
      isPrime x ∧ isPrime y ∧ isPrime z ∧
      x ≠ y ∧ x ≠ z ∧ y ≠ z ∧
      usesAllDigitsOnce x y z →
      x + y + z ≥ 999) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_of_three_primes_l888_88814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_division_ratio_l888_88826

/-- Represents an irregular hexagon composed of unit squares and a triangle -/
structure IrregularHexagon where
  unit_squares : ℕ
  triangle_base : ℚ
  triangle_height : ℚ

/-- Calculates the total area of the irregular hexagon -/
def total_area (h : IrregularHexagon) : ℚ :=
  h.unit_squares + (h.triangle_base * h.triangle_height) / 2

/-- Represents a line segment dividing the hexagon -/
structure DividingLine where
  hexagon : IrregularHexagon
  area_ratio : ℚ  -- ratio of area above the line to area below

theorem hexagon_division_ratio 
  (h : IrregularHexagon) 
  (l : DividingLine) 
  (h_squares : h.unit_squares = 12)
  (h_base : h.triangle_base = 6)
  (h_height : h.triangle_height = 3)
  (h_line : l.hexagon = h)
  (h_ratio : l.area_ratio = 2)
  : ∃ (rs st : ℚ), rs / st = 1 / 2 ∧ rs + st = h.triangle_base := by
  sorry

#check hexagon_division_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_division_ratio_l888_88826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_books_read_june_correct_l888_88874

def summer_reading_goal : ℕ := 100
def books_read_july : ℕ := 28
def books_read_august : ℕ := 30

def books_read_june : ℕ := 42

theorem books_read_june_correct : books_read_june = summer_reading_goal - (books_read_july + books_read_august) := by
  rfl

#check books_read_june
#check books_read_june_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_books_read_june_correct_l888_88874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mice_sales_target_l888_88865

/-- Calculates the sales target given previous sales and increase percentage -/
def salesTarget (prevSales : ℕ) (increasePercentage : ℚ) : ℕ :=
  prevSales + (((prevSales : ℚ) * increasePercentage / 100).floor.toNat)

theorem mice_sales_target :
  let wirelessPrev : ℕ := 48
  let opticalPrev : ℕ := 24
  let trackballPrev : ℕ := 8
  let wirelessIncrease : ℚ := 12.5
  let opticalIncrease : ℚ := 37.5
  let trackballIncrease : ℚ := 20
  (salesTarget wirelessPrev wirelessIncrease = 54) ∧
  (salesTarget opticalPrev opticalIncrease = 33) ∧
  (salesTarget trackballPrev trackballIncrease = 10) := by
  sorry

#check mice_sales_target

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mice_sales_target_l888_88865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_values_sum_and_range_l888_88824

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log ((x / a) - 1) + 1 / x + a / 2

noncomputable def g (a : ℝ) : ℝ := 1 / a + a

theorem extreme_values_sum_and_range (a : ℝ) (h1 : 0 < a) (h2 : a < 1 / 4) :
  (∀ x > a, ∃ y > a, f a x = f a y → x = y ∨ g a = f a x + f a y) ∧
  (∀ z : ℝ, g a < z ↔ 17 / 4 < z) := by
  sorry

#check extreme_values_sum_and_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_values_sum_and_range_l888_88824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_sequence_ratio_sequence_satisfies_conditions_l888_88870

def a (n : ℕ) : ℚ :=
  if n = 0 then 0
  else (n^2 : ℚ) - n + 33

theorem min_value_of_sequence_ratio :
  ∀ n : ℕ, n > 0 → a n / n ≥ 21/2 ∧
  ∃ m : ℕ, m > 0 ∧ a m / m = 21/2 := by
  sorry

theorem sequence_satisfies_conditions :
  a 1 = 33 ∧
  ∀ n : ℕ, n > 0 → a (n + 1) - a n = 2 * n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_sequence_ratio_sequence_satisfies_conditions_l888_88870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_C_is_proportional_l888_88817

-- Define the sets of line segments
noncomputable def set_A : List ℝ := [2, 2.5, 3, 3.5]
noncomputable def set_B : List ℝ := [Real.sqrt 3, 3, 3, 4 * Real.sqrt 3]
noncomputable def set_C : List ℝ := [2, 4, 9, 18]
noncomputable def set_D : List ℝ := [4, 5, 6, 7]

-- Define the proportionality condition
def is_proportional (s : List ℝ) : Prop :=
  s.length = 4 ∧ s[0]! * s[3]! = s[1]! * s[2]!

-- Theorem statement
theorem only_C_is_proportional :
  ¬(is_proportional set_A) ∧
  ¬(is_proportional set_B) ∧
  (is_proportional set_C) ∧
  ¬(is_proportional set_D) := by
  sorry

#check only_C_is_proportional

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_C_is_proportional_l888_88817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_of_f_is_F_l888_88859

open Real

-- Define the rational function
noncomputable def f (x : ℝ) : ℝ := (9 - 5*x) / (x^3 - 6*x^2 + 11*x - 6)

-- Define the antiderivative
noncomputable def F (x : ℝ) : ℝ := Real.log (abs ((x-1)^2 * (x-2) / (x-3)^3))

-- State the theorem
theorem integral_of_f_is_F (x : ℝ) (hx1 : x ≠ 1) (hx2 : x ≠ 2) (hx3 : x ≠ 3) :
  deriv F x = f x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_of_f_is_F_l888_88859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_1988_11_equals_169_l888_88835

/-- Sum of digits of a positive integer -/
def sumOfDigits (k : ℕ) : ℕ :=
  if k < 10 then k else k % 10 + sumOfDigits (k / 10)

/-- f₁(k) is the square of the sum of digits of k -/
def f₁ (k : ℕ) : ℕ := (sumOfDigits k) ^ 2

/-- fₙ(k) for n ≥ 1 -/
def fₙ : ℕ → ℕ → ℕ
  | 0, k => k  -- Base case for n = 0
  | 1, k => f₁ k
  | n + 1, k => fₙ n (f₁ k)

/-- The main theorem to prove -/
theorem f_1988_11_equals_169 : fₙ 1988 11 = 169 := by sorry

#eval fₙ 1988 11  -- This will compute the actual result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_1988_11_equals_169_l888_88835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_fraction_in_X_after_pouring_l888_88891

/- Define the capacities of jugs X and Y -/
variable (Cx Cy : ℝ)

/- Define the initial water amounts in jugs X and Y -/
noncomputable def initial_water_X (Cx : ℝ) : ℝ := (1/3 : ℝ) * Cx
noncomputable def initial_water_Y (Cy : ℝ) : ℝ := (2/3 : ℝ) * Cy

/- Define the total amount of water -/
noncomputable def total_water (Cx Cy : ℝ) : ℝ := initial_water_X Cx + initial_water_Y Cy

/- Define the amount of water needed to fill jug Y -/
noncomputable def water_to_fill_Y (Cy : ℝ) : ℝ := Cy - initial_water_Y Cy

/- Define the remaining water in jug X after pouring -/
noncomputable def remaining_water_X (Cx Cy : ℝ) : ℝ := initial_water_X Cx - water_to_fill_Y Cy

/- Theorem: The fraction of water remaining in jug X is 1/6 of its capacity -/
theorem water_fraction_in_X_after_pouring (Cx Cy : ℝ) (h : Cx > 0) : 
  remaining_water_X Cx Cy / Cx = (1/6 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_fraction_in_X_after_pouring_l888_88891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersection_implies_k_equals_one_l888_88818

/-- Two lines in 3D space -/
structure Line3D where
  point : Fin 3 → ℝ
  direction : Fin 3 → ℝ

/-- Check if two lines intersect -/
def intersect (l1 l2 : Line3D) : Prop :=
  ∃ s t : ℝ, ∀ i : Fin 3, 
    l1.point i + s * l1.direction i = l2.point i + t * l2.direction i

/-- The main theorem -/
theorem line_intersection_implies_k_equals_one :
  ∀ k : ℝ,
  let l1 : Line3D := ⟨λ i => [1, 2, 3].get i, λ i => [2, -1, k].get i⟩
  let l2 : Line3D := ⟨λ i => [3, 5, 7].get i, λ i => [-k, 3, 2].get i⟩
  intersect l1 l2 → k = 1 :=
by
  intro k l1 l2 h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersection_implies_k_equals_one_l888_88818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_plus_pi_sixth_l888_88827

theorem cos_alpha_plus_pi_sixth (α : ℝ) 
  (h1 : α ∈ Set.Ioo 0 (π/2)) 
  (h2 : Real.cos α = Real.sqrt 3 / 3) : 
  Real.cos (α + π/6) = (3 - Real.sqrt 6) / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_plus_pi_sixth_l888_88827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_function_properties_l888_88887

/-- Given a function f(x) = a*sin(2x) + b*cos(2x) where a and b are real numbers,
    ab ≠ 0, and f(x) ≤ |f(π/6)| for all real x, prove that:
    1. f(11π/12) = 0
    2. |f(7π/12)| < |f(π/5)|
    3. f is neither an odd function nor an even function -/
theorem trigonometric_function_properties (a b : ℝ) (hab : a * b ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * Real.sin (2 * x) + b * Real.cos (2 * x)
  (∀ x, f x ≤ |f (π / 6)|) →
  (f (11 * π / 12) = 0) ∧
  (|f (7 * π / 12)| < |f (π / 5)|) ∧
  (¬ Odd f ∧ ¬ Even f) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_function_properties_l888_88887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l888_88819

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (x^2 - 2*x + 1) / x

-- State the theorem
theorem min_value_of_f :
  ∃ (x : ℝ), x ∈ Set.Icc (1/2 : ℝ) 3 ∧
  (∀ (y : ℝ), y ∈ Set.Icc (1/2 : ℝ) 3 → f x ≤ f y) ∧
  f x = 0 := by
  sorry

#check min_value_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l888_88819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_medication_duration_approx_4_5_months_l888_88801

/-- Calculates the duration of a medication supply in months -/
def medication_duration (daily_dose : ℚ) (supply : ℕ) : ℚ :=
  let days_per_pill := 1 / daily_dose
  let total_days := supply * days_per_pill
  let days_per_month := 30
  total_days / days_per_month

/-- Proves that the medication supply lasts approximately 4.5 months -/
theorem medication_duration_approx_4_5_months :
  let daily_dose : ℚ := 2/3
  let supply : ℕ := 90
  abs (medication_duration daily_dose supply - 4.5) < 0.01 := by
  sorry

#eval medication_duration (2/3) 90

end NUMINAMATH_CALUDE_ERRORFEEDBACK_medication_duration_approx_4_5_months_l888_88801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equation_solution_l888_88816

theorem vector_equation_solution (a b : ℝ × ℝ) (lambda mu : ℝ) :
  a = (1, 3) →
  b = (1, -2) →
  lambda • a + mu • b = (0, 0) →
  lambda = 0 ∧ mu = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equation_solution_l888_88816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_probability_l888_88878

/-- The probability that the product of two independently and uniformly
    selected real numbers from the interval [-25, 15] is non-negative -/
noncomputable def prob_non_negative_product : ℝ := 17/32

/-- The lower bound of the interval -/
noncomputable def lower_bound : ℝ := -25

/-- The upper bound of the interval -/
noncomputable def upper_bound : ℝ := 15

/-- The length of the interval -/
noncomputable def interval_length : ℝ := upper_bound - lower_bound

/-- The probability of selecting a negative number from the interval -/
noncomputable def prob_negative : ℝ := (0 - lower_bound) / interval_length

/-- The probability of selecting a positive number from the interval -/
noncomputable def prob_positive : ℝ := (upper_bound - 0) / interval_length

theorem product_probability :
  prob_non_negative_product = prob_negative * prob_negative + prob_positive * prob_positive :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_probability_l888_88878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_db_length_l888_88838

-- Define the triangle and point D
def Triangle (A B C : ℝ × ℝ) : Prop := True
def RightAngled (A B C : ℝ × ℝ) : Prop := True
def PointOn (D : ℝ × ℝ) (B C : ℝ × ℝ) : Prop := True
def Perpendicular (AD BC : (ℝ × ℝ) → (ℝ × ℝ)) : Prop := True

-- Define the length function
noncomputable def Length (P Q : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem triangle_db_length 
  (A B C D : ℝ × ℝ) 
  (h1 : Triangle A B C)
  (h2 : RightAngled A B C)
  (h3 : Length A B = 50)
  (h4 : Length A C = 120)
  (h5 : PointOn D B C)
  (h6 : Perpendicular (λ x => x - A) (λ x => x - B))
  : abs (Length D B - 31.95) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_db_length_l888_88838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regression_analysis_properties_l888_88846

/-- Correlation coefficient between two variables -/
noncomputable def correlation_coefficient (X Y : Type) [NormedAddCommGroup X] [NormedAddCommGroup Y] : ℝ := sorry

/-- Linear regression line -/
noncomputable def linear_regression_line (X Y : Type) [NormedAddCommGroup X] [NormedAddCommGroup Y] : X → Y := sorry

/-- Sum of squared residuals -/
noncomputable def sum_squared_residuals (X Y : Type) [NormedAddCommGroup X] [NormedAddCommGroup Y] : ℝ := sorry

/-- Sample center -/
noncomputable def sample_center (X Y : Type) [NormedAddCommGroup X] [NormedAddCommGroup Y] : X × Y := sorry

/-- Stronger linear correlation -/
def stronger_linear_correlation (r₁ r₂ : ℝ) : Prop := sorry

/-- Better model fit -/
def better_model_fit (ssr₁ ssr₂ : ℝ) : Prop := sorry

theorem regression_analysis_properties 
  (X Y : Type) [NormedAddCommGroup X] [NormedAddCommGroup Y] :
  (∀ r₁ r₂ : ℝ, |r₁| > |r₂| → 
    stronger_linear_correlation r₁ r₂) ∧ 
  (linear_regression_line X Y (sample_center X Y).1 = (sample_center X Y).2) ∧
  (∀ ssr₁ ssr₂ : ℝ, ssr₁ < ssr₂ → 
    better_model_fit ssr₁ ssr₂) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regression_analysis_properties_l888_88846

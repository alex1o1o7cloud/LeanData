import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_435_degrees_l895_89595

theorem tan_435_degrees :
  Real.tan (435 * π / 180) = 2 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_435_degrees_l895_89595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_triangle_area_l895_89526

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.cos x * (Real.cos x + Real.sqrt 3 * Real.sin x)

-- Theorem for the minimum value of f
theorem f_minimum : ∃ x : ℝ, ∀ y : ℝ, f y ≥ f x ∧ f x = -1/2 := by sorry

-- Define a triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)
  (positive_sides : a > 0 ∧ b > 0 ∧ c > 0)
  (angle_sum : A + B + C = Real.pi)
  (cosine_law : c^2 = a^2 + b^2 - 2*a*b*Real.cos C)

-- Theorem for the area of triangle ABC
theorem triangle_area 
  (t : Triangle) 
  (h1 : f t.C = 1) 
  (h2 : t.c = Real.sqrt 7) 
  (h3 : t.a + t.b = 4) : 
  (1/2) * t.a * t.b * Real.sin t.C = (3 * Real.sqrt 3) / 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_triangle_area_l895_89526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_exactly_10_defective_prob_10_to_20_defective_l895_89525

/-- The number of products manufactured -/
def n : ℕ := 500

/-- The probability of a product being defective -/
def p : ℝ := 0.02

/-- The probability of a product not being defective -/
def q : ℝ := 1 - p

/-- The expected number of defective products -/
def np : ℝ := n * p

/-- The variance of the number of defective products -/
def npq : ℝ := n * p * q

/-- The normal approximation to the binomial distribution -/
noncomputable def normal_approx (k : ℕ) : ℝ :=
  1 / Real.sqrt (2 * Real.pi * npq) * Real.exp (-(k - np)^2 / (2 * npq))

/-- The standard normal cumulative distribution function -/
noncomputable def Φ (x : ℝ) : ℝ := sorry

/-- Theorem: The probability of exactly 10 defective items is approximately 0.127 -/
theorem prob_exactly_10_defective : 
  ∃ ε > 0, |normal_approx 10 - 0.127| < ε := by sorry

/-- Theorem: The probability of between 10 and 20 defective items is approximately 0.499 -/
theorem prob_10_to_20_defective : 
  ∃ ε > 0, |Φ ((20 - np) / Real.sqrt npq) - Φ ((10 - np) / Real.sqrt npq) - 0.499| < ε := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_exactly_10_defective_prob_10_to_20_defective_l895_89525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l895_89508

-- Define the function as noncomputable due to the use of Real.sqrt
noncomputable def f (x : ℝ) : ℝ := x + 2 * Real.cos x - Real.sqrt 3

-- State the theorem
theorem max_value_of_f :
  ∃ (max : ℝ), max = f (Real.pi / 6) ∧
  max = Real.pi / 6 + Real.sqrt 3 / 2 ∧
  ∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≤ max :=
by
  -- Proof placeholder
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l895_89508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_distance_sum_l895_89556

-- Define the curve C
def curve_C (x y : ℝ) : Prop := x^2 / 9 + y^2 = 1

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y + 2 = 0

-- Define point P
def point_P : ℝ × ℝ := (0, 2)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem intersection_points_distance_sum :
  ∃ A B : ℝ × ℝ,
    curve_C A.1 A.2 ∧ curve_C B.1 B.2 ∧
    line_l A.1 A.2 ∧ line_l B.1 B.2 ∧
    distance point_P A + distance point_P B = 18 * Real.sqrt 2 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_distance_sum_l895_89556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_determination_and_maximum_l895_89542

noncomputable def f (ω : ℝ) (m : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x) * Real.cos (ω * x) + m

noncomputable def g (ω : ℝ) (m : ℝ) (x : ℝ) : ℝ := f ω m (x - Real.pi / 6)

theorem function_determination_and_maximum (ω : ℝ) (m : ℝ) 
  (h_ω_pos : ω > 0)
  (h_f_zero : f ω m 0 = 1)
  (h_symmetry : ∀ (x : ℝ), f ω m (x + Real.pi / (2 * ω)) = f ω m x) :
  ω = 1 ∧ m = 1 ∧ ∀ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) → g ω m x ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_determination_and_maximum_l895_89542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_l895_89570

-- Define the line l: y = mx + 1
def line (m : ℝ) (x y : ℝ) : Prop := y = m * x + 1

-- Define the circle C: x² + y² = 1
def circle' (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the intersection of the line and circle
def intersects (m : ℝ) : Prop := ∃ (x₁ y₁ x₂ y₂ : ℝ), 
  x₁ ≠ x₂ ∧ line m x₁ y₁ ∧ line m x₂ y₂ ∧ circle' x₁ y₁ ∧ circle' x₂ y₂

theorem sufficient_not_necessary (m : ℝ) : 
  (m > 0 → intersects m) ∧ ¬(intersects m → m > 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_l895_89570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l895_89580

theorem inequality_proof (φ : Real) (h : φ ∈ Set.Ioo 0 (Real.pi / 2)) :
  Real.sin (Real.cos φ) < Real.cos φ ∧ Real.cos φ < Real.cos (Real.sin φ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l895_89580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_redistribution_theorem_l895_89502

/-- Calculates the amount of solution in each beaker after redistribution -/
noncomputable def solution_per_beaker (mL_per_tube : ℚ) (num_tubes : ℕ) (num_beakers : ℕ) : ℚ :=
  (mL_per_tube * num_tubes) / num_beakers

/-- Theorem: Given 7 mL of solution in each of 6 test tubes, when evenly distributed into 3 beakers, 
    each beaker contains 14 mL of solution -/
theorem redistribution_theorem : solution_per_beaker 7 6 3 = 14 := by
  -- Unfold the definition of solution_per_beaker
  unfold solution_per_beaker
  -- Simplify the arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_redistribution_theorem_l895_89502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_tan_l895_89558

theorem cos_double_angle_tan (α : ℝ) (h : Real.tan α = 3) : Real.cos (2 * α) = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_tan_l895_89558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_intervals_F_extreme_values_l895_89505

noncomputable section

variable (a : ℝ)

def f (x : ℝ) : ℝ := Real.log x - a * x
def F (x : ℝ) : ℝ := f a x + a * x^2 + a * x

theorem f_increasing_intervals :
  (a ≤ 0 → ∀ x > 0, HasDerivAt (f a) ((1/x) - a) x) ∧
  (a > 0 → ∀ x ∈ Set.Ioo 0 (1/a), HasDerivAt (f a) ((1/x) - a) x) := by sorry

theorem F_extreme_values :
  (a ≥ 0 → ∀ x > 0, HasDerivAt (F a) ((1/x) + 2*a*x) x) ∧
  (a < 0 → ∃ x_max, x_max = Real.sqrt (-1/(2*a)) ∧
    IsLocalMax (F a) x_max ∧
    F a x_max = Real.log (Real.sqrt (-1/(2*a))) - 1/2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_intervals_F_extreme_values_l895_89505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l895_89590

open Real

noncomputable def g (A : ℝ) : ℝ := (sin A)^2 * (5 + 4*(cos A)^2) + 2*(cos A)^4

theorem g_range (A : ℝ) (h : ∀ k : ℤ, A ≠ k*π + π/3) : 
  2 ≤ g A ∧ g A ≤ 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l895_89590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_term_is_seven_general_term_is_correct_l895_89560

/-- An arithmetic sequence satisfying given conditions -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_condition : a 1 + a 2 + a 3 = 21
  product_condition : a 1 * a 2 * a 3 = 231

/-- The second term of the sequence is 7 -/
theorem second_term_is_seven (seq : ArithmeticSequence) : seq.a 2 = 7 := by
  sorry

/-- The general term of the sequence -/
noncomputable def general_term (seq : ArithmeticSequence) : ℕ → ℚ
  | n => if seq.a 1 = 11 then -4 * n + 15 else 4 * n - 1

/-- The general term satisfies the arithmetic sequence properties -/
theorem general_term_is_correct (seq : ArithmeticSequence) :
  ∀ n, seq.a n = general_term seq n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_term_is_seven_general_term_is_correct_l895_89560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_critical_points_sin_plus_cos_l895_89554

open Real

theorem critical_points_sin_plus_cos :
  let f : ℝ → ℝ := λ x ↦ sin x + cos x
  let criticalPoints : Set ℝ := {x | x ∈ Set.Ioo 0 (2 * π) ∧ deriv f x = 0}
  criticalPoints = {π / 4, 5 * π / 4} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_critical_points_sin_plus_cos_l895_89554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f1_derivative_f2_derivative_f3_l895_89523

-- Function definitions
noncomputable def f1 (x : ℝ) : ℝ := (x + 1)^2 * (x - 1)
noncomputable def f2 (x : ℝ) : ℝ := x^2 * Real.sin x
noncomputable def f3 (x : ℝ) : ℝ := (Real.exp x + 1) / (Real.exp x - 1)

-- Theorems to prove
theorem derivative_f1 : 
  ∀ x : ℝ, deriv f1 x = 3*x^2 + 2*x - 1 := by sorry

theorem derivative_f2 : 
  ∀ x : ℝ, deriv f2 x = 2*x * Real.sin x + x^2 * Real.cos x := by sorry

theorem derivative_f3 : 
  ∀ x : ℝ, x ≠ 0 → deriv f3 x = -2 * Real.exp x / (Real.exp x - 1)^2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f1_derivative_f2_derivative_f3_l895_89523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angela_delivers_three_meals_l895_89540

/-- Represents the number of meals Angela delivers -/
def meals : ℕ := sorry

/-- Represents the number of packages Angela delivers -/
def packages : ℕ := sorry

/-- The number of packages is 8 times the number of meals -/
axiom package_meal_ratio : packages = 8 * meals

/-- The total number of deliveries is 27 -/
axiom total_deliveries : meals + packages = 27

/-- Theorem: Given the conditions, Angela delivers 3 meals -/
theorem angela_delivers_three_meals : meals = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angela_delivers_three_meals_l895_89540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vector_sum_l895_89515

-- Define the parabola
def Parabola (p : ℝ × ℝ) : Prop := p.2^2 = 4 * p.1

-- Define the focus of the parabola
def Focus : ℝ × ℝ := (1, 0)

-- Define the vector from focus to a point
def VectorFromFocus (p : ℝ × ℝ) : ℝ × ℝ := (p.1 - Focus.1, p.2 - Focus.2)

-- Define the magnitude of a vector
noncomputable def Magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

theorem parabola_vector_sum (A B C : ℝ × ℝ) 
  (hA : Parabola A) (hB : Parabola B) (hC : Parabola C)
  (h : VectorFromFocus A + 2 • VectorFromFocus B + 3 • VectorFromFocus C = (0, 0)) :
  Magnitude (VectorFromFocus A) + 2 * Magnitude (VectorFromFocus B) + 3 * Magnitude (VectorFromFocus C) = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vector_sum_l895_89515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_equals_six_l895_89578

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (1 + 4 * x^2) - 2 * x) + 3

-- State the theorem
theorem f_sum_equals_six : 
  f (Real.log 2 / Real.log 10) + f (Real.log (1/2) / Real.log 10) = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_equals_six_l895_89578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_always_five_l895_89597

-- Define the ∆ (triangle) operation as minimum
noncomputable def triangle (a b : ℝ) : ℝ := min a b

-- Define the ∇ (nabla) operation as maximum
noncomputable def nabla (a b : ℝ) : ℝ := max a b

-- Theorem statement
theorem expression_always_five (x : ℝ) :
  nabla 5 (nabla 4 (triangle x 4)) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_always_five_l895_89597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digits_599_600_l895_89533

def digit_list : List ℕ := sorry

-- The list contains positive integers with first digit 2 in increasing order
axiom list_property (n : ℕ) : n ∈ digit_list → (n ≥ 2 ∧ n < 3000)

-- The list has at least 600 digits when written out
axiom list_length : (digit_list.map (λ n => (toString n).length)).sum ≥ 600

-- Function to get the nth digit in the list
def nth_digit (n : ℕ) : ℕ := sorry

theorem digits_599_600 : 
  nth_digit 599 = 6 ∧ nth_digit 600 = 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digits_599_600_l895_89533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_complement_f_sum_fraction_l895_89541

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (4^x) / (4^x + 2)

-- Theorem 1: f(x) + f(1-x) = 1 for all x
theorem f_sum_complement (x : ℝ) : f x + f (1 - x) = 1 := by
  sorry

-- Theorem 2: Sum of f(k/2014) for k = 1 to 2013 equals 1006.5
theorem f_sum_fraction : 
  (Finset.range 2013).sum (λ k => f ((k + 1 : ℝ) / 2014)) = 1006.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_complement_f_sum_fraction_l895_89541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_fixed_points_x₆_l895_89566

noncomputable def x (n : ℕ) (x₀ : ℝ) : ℝ :=
  match n with
  | 0 => x₀
  | n + 1 => 
    let xₙ := x n x₀
    if 2 * xₙ < 1 then 2 * xₙ else 2 * xₙ - 1

theorem count_fixed_points_x₆ :
  ∃ (S : Finset ℝ), (∀ x₀ ∈ S, 0 ≤ x₀ ∧ x₀ < 1) ∧ 
    (∀ x₀ ∈ S, x 6 x₀ = x₀) ∧
    (∀ x₀ ∉ S, x₀ = 0 ∨ x₀ ≥ 1 ∨ x 6 x₀ ≠ x₀) ∧
    S.card = 63 :=
by sorry

#check count_fixed_points_x₆

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_fixed_points_x₆_l895_89566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_condition_extreme_values_condition_l895_89534

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a / x

-- Theorem 1
theorem zeros_condition (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ > 0 ∧ x₂ > 0 ∧ f a x₁ = 0 ∧ f a x₂ = 0) →
  (0 < a ∧ a < Real.exp (-1)) :=
by
  sorry

-- Theorem 2
theorem extreme_values_condition (a : ℝ) (m : ℝ) (x₁ x₂ : ℝ) :
  (∃ x₀ : ℝ, x₀ > 0 ∧ (∀ x > 0, f a x ≥ f a x₀)) →
  (x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧ f a x₁ = m ∧ f a x₂ = m) →
  x₁ + x₂ > 2 * a :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_condition_extreme_values_condition_l895_89534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_analysis_l895_89567

-- Define the function f
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x + b / x

-- Define the conditions
def condition1 (a b : ℝ) : Prop := f a b 2 = 5/2
def condition2 (a b : ℝ) : Prop := f a b (-1) = -2

-- Define the function g
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := x^2 + 1/x^2 + 2 * m * (x + 1/x)

-- State the theorem
theorem function_analysis (a b : ℝ) (h1 : condition1 a b) (h2 : condition2 a b) :
  (∀ x : ℝ, f a b x = x + 1/x) ∧
  (∀ m : ℝ, 
    (∀ x ∈ Set.Icc 1 2, g m x ≥ 
      (if m ≥ -2 then 4*m + 2
       else if m > -5/2 then -m^2 - 2
       else 5*m + 17/4))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_analysis_l895_89567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_union_equals_three_l895_89551

open Set

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4}

-- Define sets A and B
def A : Set Nat := {2, 4}
def B : Set Nat := {1, 4}

-- Theorem statement
theorem complement_of_union_equals_three :
  (U \ (A ∪ B) : Set Nat) = {3} := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_union_equals_three_l895_89551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_is_even_l895_89569

-- Define the set of integers
def U : Set Int := Set.univ

-- Define set A
def A : Set Int := {x | ∃ k : Int, x = 2 * k + 1}

-- Define the complement of A in U
def complement_A : Set Int := U \ A

-- Theorem statement
theorem complement_A_is_even : complement_A = {x | ∃ k : Int, x = 2 * k} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_is_even_l895_89569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_ratio_curve_intersections_l895_89592

open Real

-- Define the polar equations of C₁ and C₂
noncomputable def C₁ (ρ θ : ℝ) : Prop := ρ * (cos θ + sin θ) = 4
noncomputable def C₂ (ρ θ : ℝ) : Prop := ρ = 2 * cos θ

-- Define the ratio function
noncomputable def ratio (α : ℝ) : ℝ :=
  (2 * cos α) / (4 / (cos α + sin α))

-- Theorem statement
theorem max_ratio_curve_intersections :
  ∃ (max_ratio : ℝ),
    (∀ α, -π/4 < α ∧ α < π/2 → ratio α ≤ max_ratio) ∧
    (∃ α, -π/4 < α ∧ α < π/2 ∧ ratio α = max_ratio) ∧
    max_ratio = (sqrt 2 + 1) / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_ratio_curve_intersections_l895_89592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_value_l895_89516

theorem max_a_value (a : ℝ) : 
  (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 1 (Real.exp 1) → x₂ ∈ Set.Icc 1 (Real.exp 1) → x₁ < x₂ → 
    Real.log (x₁ / x₂) < a * (x₁ - x₂)) ↔ 
  a ≤ 1 / Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_value_l895_89516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_area_l895_89528

/-- Given a grid of congruent squares, calculate its total area based on the diagonal of a larger square within the grid. -/
theorem grid_area (n : ℕ) (d : ℝ) (h1 : n = 20) (h2 : d = 10) : ∃ (total_area : ℝ), total_area = 62.5 := by
  let num_squares : ℕ := n
  let diagonal : ℝ := d
  let large_square_size : ℕ := 4
  let large_square_area : ℝ := diagonal ^ 2 / 2
  let small_square_area : ℝ := large_square_area / (large_square_size ^ 2 : ℝ)
  let total_area : ℝ := num_squares * small_square_area
  
  have h3 : total_area = 62.5 := by sorry
  
  exact ⟨total_area, h3⟩

#check grid_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_area_l895_89528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_values_zeros_range_l895_89512

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a * x

-- Part 1
theorem tangent_line_values (a m : ℝ) :
  (∀ x, (deriv (f a)) 1 * (x - 1) + f a 1 = 2 * x + m) →
  a = 1 ∧ m = -1 :=
sorry

-- Part 2
theorem zeros_range (a : ℝ) :
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ x₁ > 0 ∧ x₂ > 0 ∧ f a x₁ = 0 ∧ f a x₂ = 0) →
  a > -1 / Real.exp 1 ∧ a < 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_values_zeros_range_l895_89512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_of_quartic_equation_l895_89501

theorem solutions_of_quartic_equation :
  let S : Set ℂ := {z : ℂ | z^4 - 4*z^2 + 3 = 0}
  S = {-Complex.I * Real.sqrt 3, -1, 1, Complex.I * Real.sqrt 3} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_of_quartic_equation_l895_89501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_meetings_same_point_meeting_2015_l895_89568

/-- Represents a point on a line segment --/
structure Point where
  position : ℝ

/-- Represents a moving object --/
structure MovingObject where
  speed : ℝ
  startPosition : Point
  startTime : ℝ

/-- Represents a system with two moving objects --/
structure TwoObjectSystem where
  objectA : MovingObject
  objectB : MovingObject
  segmentLength : ℝ

/-- Represents a meeting between two objects --/
structure Meeting where
  position : Point
  time : ℝ

/-- Function to calculate the nth meeting point --/
noncomputable def nthMeetingPoint (system : TwoObjectSystem) (n : ℕ) : Point :=
  sorry

/-- Theorem stating that odd-numbered meetings occur at the same point --/
theorem odd_meetings_same_point (system : TwoObjectSystem) (n : ℕ) :
  nthMeetingPoint system (2 * n + 1) = nthMeetingPoint system 1 := by
  sorry

/-- Corollary for the 2015th meeting --/
theorem meeting_2015 (system : TwoObjectSystem) :
  nthMeetingPoint system 2015 = nthMeetingPoint system 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_meetings_same_point_meeting_2015_l895_89568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_non_zero_digits_after_decimal_l895_89521

def fraction : ℚ := 120 / (2^4 * 5^10)

theorem non_zero_digits_after_decimal (f : ℚ := fraction) : 
  (f.num.natAbs.digits 10).length = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_non_zero_digits_after_decimal_l895_89521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_plus_two_beta_l895_89598

theorem sin_alpha_plus_two_beta 
  (α β : ℝ) 
  (h1 : 0 < α ∧ α < π/2) 
  (h2 : 0 < β ∧ β < π/2) 
  (h3 : Real.cos (α + β) = -5/13) 
  (h4 : Real.sin β = 3/5) : 
  Real.sin (α + 2*β) = 33/65 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_plus_two_beta_l895_89598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_lines_l895_89522

/-- The distance between two parallel lines in 2D space -/
noncomputable def distance_between_parallel_lines (a b c₁ c₂ : ℝ) : ℝ :=
  abs (c₁ - c₂) / Real.sqrt (a^2 + b^2)

/-- Theorem: The distance between the lines 2x - y = 0 and 2x - y + 5 = 0 is √5 -/
theorem distance_between_specific_lines : 
  distance_between_parallel_lines 2 (-1) 0 (-5) = Real.sqrt 5 := by
  sorry

#check distance_between_specific_lines

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_lines_l895_89522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_composite_difference_91_l895_89503

/-- A number is composite if it has at least two distinct prime factors -/
def IsComposite (n : ℕ) : Prop := ∃ p q : ℕ, Prime p ∧ Prime q ∧ p ≠ q ∧ p * q ∣ n

/-- The minimum positive difference between two composite numbers that sum to 91 -/
def MinCompositeDifference : ℕ := 7

/-- Theorem stating that the minimum positive difference between two composite numbers
    that sum to 91 is 7 -/
theorem min_composite_difference_91 :
  ∀ a b : ℕ, IsComposite a → IsComposite b → a + b = 91 →
  ∀ c d : ℕ, IsComposite c → IsComposite d → c + d = 91 →
  (a : ℤ) - (b : ℤ) ≥ MinCompositeDifference ∧ (c : ℤ) - (d : ℤ) ≥ MinCompositeDifference :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_composite_difference_91_l895_89503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_length_l895_89593

-- Define the right triangle DEF
def Triangle (DE DF : ℝ) : Prop :=
  DE > 0 ∧ DF > 0 ∧ DE^2 + DF^2 = (DE^2 + DF^2)

-- Define the median DN
noncomputable def Median (DE DF : ℝ) : ℝ :=
  Real.sqrt (DE^2 + DF^2) / 2

theorem median_length :
  ∀ (DE DF : ℝ),
  Triangle DE DF →
  DE = 5 →
  DF = 12 →
  Median DE DF = 6.5 := by
  sorry

#check median_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_length_l895_89593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conditional_prob_B_given_A_l895_89550

def coin_toss_space : Type := Bool × Bool

def event_A : Set coin_toss_space := {x | x.1 = true}
def event_B : Set coin_toss_space := {x | x.2 = false}

noncomputable def prob_A : ℝ := 1/2
noncomputable def prob_B : ℝ := 1/2
noncomputable def prob_A_and_B : ℝ := 1/4

theorem conditional_prob_B_given_A :
  (prob_A_and_B / prob_A : ℝ) = 1/2 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_conditional_prob_B_given_A_l895_89550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rosy_completion_time_l895_89579

/-- The number of days Mary takes to complete the work -/
noncomputable def mary_days : ℚ := 28

/-- The efficiency ratio of Rosy compared to Mary -/
noncomputable def rosy_efficiency_ratio : ℚ := 14/10

/-- The number of days Rosy takes to complete the work -/
noncomputable def rosy_days : ℚ := mary_days / rosy_efficiency_ratio

theorem rosy_completion_time :
  rosy_days = 20 := by
  -- Unfold the definitions
  unfold rosy_days mary_days rosy_efficiency_ratio
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rosy_completion_time_l895_89579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_point_existence_l895_89584

noncomputable section

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x + 2)^2 + (y - 2)^2 = 8

-- Define the point Q
def point_Q : ℝ × ℝ := (4/5, 12/5)

-- Define the point F
def point_F : ℝ × ℝ := (4, 0)

-- Define the origin O
def origin : ℝ × ℝ := (0, 0)

theorem circle_and_point_existence :
  -- Circle C is in the second quadrant
  (∃ x y, x < 0 ∧ y > 0 ∧ circle_C x y) ∧
  -- Circle C has radius 2√2
  (∀ x y, circle_C x y → (x + 2)^2 + (y - 2)^2 = 8) ∧
  -- Circle C is tangent to y = x at origin
  circle_C 0 0 ∧
  -- Point Q exists on circle C
  circle_C (point_Q.1) (point_Q.2) ∧
  -- Q is different from origin
  point_Q ≠ origin ∧
  -- Distance from Q to F equals length of OF
  (point_Q.1 - point_F.1)^2 + (point_Q.2 - point_F.2)^2 = 
  point_F.1^2 + point_F.2^2 :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_point_existence_l895_89584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangles_shortest_side_l895_89582

theorem similar_triangles_shortest_side 
  (a b c : ℝ) 
  (triangle1_sides : Set.toFinset {a, b, c} = Set.toFinset {8, 10, 12}) 
  (triangle2_perimeter : a + b + c = 150) 
  (similarity_ratio : ℝ) 
  (similar : a = 8 * similarity_ratio ∧ 
             b = 10 * similarity_ratio ∧ 
             c = 12 * similarity_ratio) : 
  min a (min b c) = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangles_shortest_side_l895_89582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_k_value_l895_89529

/-- The function f(x) as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ := (x - 2) * Real.exp x + (1/2) * x^2 - x

/-- The theorem stating the maximum value of k -/
theorem max_k_value (k : ℝ) :
  (∀ x : ℝ, x ≥ 0 → f x ≥ k * x - 2) → k ≤ -2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_k_value_l895_89529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_range_l895_89557

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the lengths of sides
noncomputable def side_length (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Define the angle C
noncomputable def angle_C (t : Triangle) : ℝ :=
  Real.arccos ((side_length t.A t.B^2 + side_length t.B t.C^2 - side_length t.C t.A^2) / 
               (2 * side_length t.A t.B * side_length t.B t.C))

-- Theorem statement
theorem angle_C_range (t : Triangle) :
  side_length t.B t.C = 2 → side_length t.A t.B = Real.sqrt 3 →
  0 < angle_C t ∧ angle_C t ≤ π / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_range_l895_89557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_2019_equals_one_l895_89507

-- Define the integer function
noncomputable def intFunc (x : ℝ) : ℤ := ⌊x⌋

-- Define the sequence a_n
noncomputable def a (n : ℕ) : ℤ := intFunc ((1 / 7) * (2 ^ n))

-- Define the sequence b_n
noncomputable def b : ℕ → ℤ
  | 0 => 0  -- Add a case for 0 to avoid "missing cases" error
  | 1 => a 1
  | n + 1 => a (n + 1) - 2 * a n

-- Theorem to prove
theorem b_2019_equals_one : b 2019 = 1 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_2019_equals_one_l895_89507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_and_theta_l895_89599

open Real

-- Define the function f
noncomputable def f (θ : ℝ) (x : ℝ) : ℝ := cos x * cos (x - θ) - (1/2) * cos θ

-- Define the function g
noncomputable def g (θ : ℝ) (x : ℝ) : ℝ := 2 * f θ ((3/2) * x)

theorem max_value_and_theta :
  ∃ (θ : ℝ), 0 < θ ∧ θ < π ∧
  (∀ x, f θ x ≤ f θ (π/3)) ∧
  θ = 2*π/3 ∧
  (∀ x, x ∈ Set.Icc 0 (π/3) → g θ x ≤ 1) ∧
  (∃ x, x ∈ Set.Icc 0 (π/3) ∧ g θ x = 1) := by
  sorry

#check max_value_and_theta

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_and_theta_l895_89599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l895_89589

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/25 + y^2/9 = 1

-- Define the foci
def F₁ : ℝ × ℝ := (-4, 0)
def F₂ : ℝ × ℝ := (4, 0)

-- Define a point on the ellipse
def P : ℝ × ℝ → Prop := λ p => ellipse p.1 p.2

-- Define perpendicularity condition
def perpendicular (p : ℝ × ℝ) : Prop :=
  (p.1 - F₁.1) * (p.1 - F₂.1) + (p.2 - F₁.2) * (p.2 - F₂.2) = 0

-- Helper function to calculate triangle area
noncomputable def area_triangle (p q r : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((q.1 - p.1) * (r.2 - p.2) - (r.1 - p.1) * (q.2 - p.2))

-- Theorem statement
theorem ellipse_properties :
  (∀ x y, x^2/25 + y^2/9 = 1 ↔ ellipse x y) ∧
  (∀ p, P p → perpendicular p → area_triangle p F₁ F₂ = 9) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l895_89589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l895_89583

noncomputable def f (x : ℝ) := 2 * Real.sin (2 * x - Real.pi / 3)

theorem f_properties :
  (∀ x, f (π/6 + x) = -f (π/6 - x)) ∧
  (∀ k : ℤ, ∀ x ∈ Set.Icc (k * π + 5*π/12) (k * π + 11*π/12), 
    ∀ y ∈ Set.Icc (k * π + 5*π/12) (k * π + 11*π/12), 
    x < y → f x > f y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l895_89583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_is_one_percent_l895_89500

/-- The discount percentage given by a retailer on pens -/
noncomputable def discount_percentage (num_pens : ℕ) (price_in_pens : ℕ) (profit_percentage : ℝ) : ℝ :=
  let cost_price : ℝ := price_in_pens
  let selling_price := cost_price * (1 + profit_percentage)
  let selling_price_per_pen := selling_price / num_pens
  let discount_amount := 1 - selling_price_per_pen
  discount_amount * 100

/-- Theorem stating that the discount percentage is 1% given the specific conditions -/
theorem discount_is_one_percent :
  discount_percentage 80 36 1.2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_is_one_percent_l895_89500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_with_same_terminal_side_as_negative_sixty_l895_89517

/-- Predicate to represent that two angles have the same terminal side -/
def has_same_terminal_side (α β : ℝ) : Prop :=
  ∃ k : ℤ, α = β + k * 360

/-- Given an angle α in degrees such that 0° < α < 360° and α has the same terminal side as -60°,
    prove that α = 300°. -/
theorem angle_with_same_terminal_side_as_negative_sixty (α : ℝ) : 
  0 < α ∧ α < 360 ∧ has_same_terminal_side α (-60) → α = 300 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_with_same_terminal_side_as_negative_sixty_l895_89517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l895_89513

/-- The function f(x) as defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -a/2 * x^2 + (a-1)*x + Real.log x

/-- Theorem stating the inequality to be proved -/
theorem inequality_proof (a : ℝ) (x : ℝ) (h1 : a > 1) (h2 : x > 0) :
  (2*a - 1) * f a x < 3 * Real.exp (a - 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l895_89513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_other_person_time_l895_89510

/-- The time it takes for Dan to complete the job alone -/
noncomputable def dan_time : ℝ := 15

/-- The time Dan works before stopping -/
noncomputable def dan_work_time : ℝ := 3

/-- The time it takes the other person to complete the remaining work after Dan stops -/
noncomputable def other_completion_time : ℝ := 8

/-- The portion of the job Dan completes in his work time -/
noncomputable def dan_portion : ℝ := dan_work_time / dan_time

/-- The time it takes for the other person to complete the job alone -/
noncomputable def other_time : ℝ := 10

theorem other_person_time : 
  (1 - dan_portion) * other_time = other_completion_time := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_other_person_time_l895_89510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_relation_max_l895_89543

/-- Given a triangle ABC with area S_△ABC = (√3/12)a², 
    the maximum value of m that satisfies sin²B + sin²C = m*sinB*sinC is 4 -/
theorem triangle_angle_relation_max (A B C : ℝ) (a b c : ℝ) (m : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  (Real.sqrt 3 / 12) * a^2 = (1 / 2) * b * c * Real.sin A ∧
  Real.sin A^2 + Real.sin B^2 = m * Real.sin A * Real.sin B →
  ∃ (m_max : ℝ), m ≤ m_max ∧ m_max = 4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_relation_max_l895_89543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_l895_89587

-- Define the parametric equations of the ellipse
noncomputable def x (θ : ℝ) : ℝ := 3 + 3 * Real.cos θ
noncomputable def y (θ : ℝ) : ℝ := -1 + 5 * Real.sin θ

-- Define the center of the ellipse
def center : ℝ × ℝ := (3, -1)

-- Define the semi-major and semi-minor axes
def a : ℝ := 5
def b : ℝ := 3

-- Define the distance from the center to a focus
noncomputable def c : ℝ := Real.sqrt (a^2 - b^2)

-- Define the foci of the ellipse
noncomputable def focus1 : ℝ × ℝ := (center.1, center.2 + c)
noncomputable def focus2 : ℝ × ℝ := (center.1, center.2 - c)

-- Theorem statement
theorem ellipse_foci :
  focus1 = (3, 3) ∧ focus2 = (3, -5) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_l895_89587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_l895_89572

theorem geometric_sequence_sum (u v : ℝ) : 
  (∃ (a : ℝ), a > 0 ∧ 
    (a * (1/4)^3 = u) ∧ 
    (a * (1/4)^4 = v) ∧ 
    (a * (1/4)^5 = 4)) →
  u + v = 80 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_l895_89572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l895_89524

-- Define the function f
noncomputable def f (x : ℝ) := Real.log (x^2 - x - 6)

-- Theorem statement
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x < -2 ∨ x > 3} :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l895_89524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_midpoint_to_line_l895_89514

-- Define the curve C
noncomputable def curve_C (α : ℝ) : ℝ × ℝ := (2 * Real.sqrt 3 * Real.cos α, 2 * Real.sin α)

-- Define the point P
def point_P : ℝ × ℝ := (4, 4)

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x - y - 10 = 0

-- Define the distance function from a point to the line
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  |x - y - 10| / Real.sqrt 2

-- Statement of the theorem
theorem max_distance_midpoint_to_line :
  ∃ (max_distance : ℝ),
    max_distance = 6 * Real.sqrt 2 ∧
    ∀ (α : ℝ), 0 < α ∧ α < Real.pi →
      let (qx, qy) := curve_C α
      let (mx, my) := ((qx + point_P.1) / 2, (qy + point_P.2) / 2)
      distance_to_line mx my ≤ max_distance :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_midpoint_to_line_l895_89514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_k_value_l895_89594

/-- An arithmetic sequence starting with 1 -/
def arithmetic_seq (d : ℕ) : ℕ → ℕ
  | 0 => 1
  | n + 1 => arithmetic_seq d n + d

/-- A geometric sequence starting with 1 -/
def geometric_seq (r : ℕ) : ℕ → ℕ
  | 0 => 1
  | n + 1 => geometric_seq r n * r

/-- The sum of the nth terms of the arithmetic and geometric sequences -/
def c_seq (d r : ℕ) (n : ℕ) : ℕ := arithmetic_seq d n + geometric_seq r n

theorem c_k_value (d r k : ℕ) (h1 : k > 0) :
  (c_seq d r (k - 1) = 200) →
  (c_seq d r (k + 1) = 2000) →
  (c_seq d r k = 423) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_k_value_l895_89594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equivalences_l895_89574

-- Define the base of the logarithm
variable (a : ℝ)

-- Define the logarithm function
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log a

-- Define what it means for f to be decreasing
def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

-- Original proposition
def P (a : ℝ) : Prop := is_decreasing (f a) → f a 2 < 0

-- Converse of P
def converse_P (a : ℝ) : Prop := f a 2 < 0 → is_decreasing (f a)

-- Contrapositive of P
def contrapositive_P (a : ℝ) : Prop := f a 2 ≥ 0 → ¬(is_decreasing (f a))

-- Theorem stating the logical equivalences
theorem log_equivalences (h1 : a > 0) (h2 : a ≠ 1) :
  (P a ↔ converse_P a) ∧ (P a ↔ contrapositive_P a) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equivalences_l895_89574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_segment_length_from_volume_l895_89581

/-- A line segment in 3D space -/
structure LineSegment where
  start : ℝ × ℝ × ℝ
  end_ : ℝ × ℝ × ℝ

/-- The volume of a region around a line segment -/
noncomputable def volume_around_segment (s : LineSegment) (radius : ℝ) : ℝ := sorry

/-- The length of a line segment -/
noncomputable def segment_length (s : LineSegment) : ℝ := sorry

theorem line_segment_length_from_volume 
  (CD : LineSegment)
  (h1 : volume_around_segment CD 4 = 352 * Real.pi) :
  segment_length CD = 50 / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_segment_length_from_volume_l895_89581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_AC_less_than_10_l895_89565

noncomputable section

/-- The distance from A to B -/
def AB : ℝ := 12

/-- The distance from B to C -/
def BC : ℝ := 8

/-- The maximum distance for AC -/
def max_AC : ℝ := 10

/-- The angle α between AB and BC, chosen randomly from (0, π) -/
noncomputable def α : ℝ := Real.pi / 2  -- We use π/2 as a placeholder, as the actual value is random

/-- The probability that AC is less than max_AC -/
def prob_AC_less_than_max : ℝ := 1 / 3

/-- Theorem stating the existence of a probability density function -/
theorem probability_AC_less_than_10 :
  ∃ (f : ℝ → ℝ), 
    (∀ x ∈ Set.Ioo 0 Real.pi, 0 ≤ f x ∧ f x ≤ 1) ∧
    (∫ x in Set.Ioo 0 Real.pi, f x) / Real.pi = prob_AC_less_than_max :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_AC_less_than_10_l895_89565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_bounded_sequence_with_inequality_l895_89520

/-- The fractional part of a real number -/
noncomputable def frac (x : ℝ) : ℝ := x - ⌊x⌋

/-- The sequence defined as x_n = 4 * frac(n * √2) -/
noncomputable def x (n : ℕ) : ℝ := 4 * frac (n * Real.sqrt 2)

/-- Theorem stating the existence of an infinite bounded sequence satisfying the given inequality -/
theorem exists_bounded_sequence_with_inequality :
  ∃ (x : ℕ → ℝ), (∀ n : ℕ, x n ∈ Set.Icc 0 4) ∧
    ∀ (m k : ℕ), m ≠ k → |x m - x k| ≥ 1 / (|Int.ofNat m - Int.ofNat k|) :=
by
  -- Use our defined sequence x
  use x
  constructor
  
  -- Prove that x is bounded between 0 and 4
  · intro n
    simp [x, frac]
    -- The fractional part is always in [0, 1)
    sorry
  
  -- Prove the inequality
  · intros m k hmk
    -- Use the properties of irrational numbers and fractional parts
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_bounded_sequence_with_inequality_l895_89520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_plus_d_equals_three_l895_89532

-- Define the function f
noncomputable def f (c d : ℝ) (x : ℝ) : ℝ :=
  if x < 3 then c * x + d else 7 - 2 * x

-- State the theorem
theorem c_plus_d_equals_three (c d : ℝ) :
  (∀ x, f c d (f c d x) = x) → c + d = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_plus_d_equals_three_l895_89532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_difference_given_sum_l895_89571

theorem tan_difference_given_sum (α : ℝ) : 
  Real.tan (α + π/6) = 1 → Real.tan (α - π/6) = -2 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_difference_given_sum_l895_89571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l895_89552

theorem equation_solution (t : ℝ) (k : ℤ) (h1 : Real.sin t ≠ 0) (h2 : Real.cos t ≠ 0) :
  5.54 * (Real.sin t ^ 2 - Real.tan t ^ 2) / (Real.cos t ^ 2 - (1 / Real.tan t) ^ 2) + 2 * Real.tan t ^ 3 + 1 = 0 ↔
  t = Real.pi / 4 * (4 * ↑k - 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l895_89552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_sector_volume_ratio_l895_89518

/-- Represents a right circular cone -/
structure RightCircularCone where
  height : ℝ
  baseRadius : ℝ

/-- Represents a sector of a right circular cone -/
structure ConeSector where
  cone : RightCircularCone
  sectorAngle : ℝ

/-- Volume of a right circular cone -/
noncomputable def coneVolume (cone : RightCircularCone) : ℝ :=
  (1/3) * Real.pi * cone.baseRadius^2 * cone.height

/-- Volume of a cone sector -/
noncomputable def sectorVolume (sector : ConeSector) : ℝ :=
  (sector.sectorAngle / (2 * Real.pi)) * coneVolume sector.cone

/-- Theorem: The ratio of volumes of any two sectors in a cone divided into four equal parts is 1 -/
theorem equal_sector_volume_ratio 
  (cone : RightCircularCone) 
  (sector1 sector2 : ConeSector) 
  (h1 : sector1.cone = cone) 
  (h2 : sector2.cone = cone) 
  (h3 : sector1.sectorAngle = Real.pi/2) 
  (h4 : sector2.sectorAngle = Real.pi/2) : 
  sectorVolume sector1 / sectorVolume sector2 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_sector_volume_ratio_l895_89518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_g_range_l895_89509

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin x + Real.cos x

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := f x * Real.cos x

-- Theorem for the maximum value of f
theorem f_max_value : 
  ∃ (M : ℝ), M = 2 ∧ ∀ x, f x ≤ M := by
  sorry

-- Theorem for the range of g
theorem g_range : 
  ∀ x ∈ Set.Icc 0 (Real.pi / 2), 
  1 ≤ g x ∧ g x ≤ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_g_range_l895_89509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_S_l895_89547

/-- A complex number is five-presentable if it can be expressed as w - 1/w for some w with |w| = 5 -/
def FivePresentable (z : ℂ) : Prop :=
  ∃ w : ℂ, Complex.abs w = 5 ∧ z = w - 1 / w

/-- The set of all five-presentable complex numbers -/
def S : Set ℂ :=
  {z : ℂ | FivePresentable z}

/-- The area of the set S -/
noncomputable def area_S : ℝ :=
  (624 / 25) * Real.pi

theorem area_of_S : MeasureTheory.volume S = ENNReal.ofReal area_S := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_S_l895_89547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l895_89563

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x - x^2

-- State the theorem
theorem f_monotone_increasing :
  ∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → x₂ < 1 → f x₁ < f x₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l895_89563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_existence_l895_89573

/-- Definition of an angle bisector -/
def angle_bisector (P Q R : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {X | dist P X / dist X Q = dist P X / dist X R}

/-- Distance function between two points -/
noncomputable def dist (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

/-- Given a triangle PQR, prove the existence of a and c for the angle bisector equation of ∠P -/
theorem angle_bisector_existence (P Q R : ℝ × ℝ) : ∃ (a c : ℝ),
  P = (-5, 3) ∧ Q = (-10, -15) ∧ R = (4, -5) →
  ∀ (x y : ℝ), (x, y) ∈ angle_bisector P Q R ↔ a * x + 3 * y + c = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_existence_l895_89573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_modification_range_l895_89577

/-- Represents the fuel efficiency and capacity of a car before and after modification -/
structure Car where
  initial_efficiency : ℚ  -- miles per gallon
  tank_capacity : ℚ       -- gallons
  fuel_reduction : ℚ      -- percentage of original fuel consumption after modification

/-- Calculates the additional miles a car can travel after modification -/
def additional_miles (c : Car) : ℚ :=
  let new_efficiency := c.initial_efficiency / c.fuel_reduction
  let initial_range := c.initial_efficiency * c.tank_capacity
  let new_range := new_efficiency * c.tank_capacity
  new_range - initial_range

/-- Theorem stating the additional miles for the given car specifications -/
theorem car_modification_range (c : Car) 
    (h1 : c.initial_efficiency = 33)
    (h2 : c.tank_capacity = 16)
    (h3 : c.fuel_reduction = 3/4) : 
  additional_miles c = 132 := by
  sorry

#eval additional_miles { initial_efficiency := 33, tank_capacity := 16, fuel_reduction := 3/4 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_modification_range_l895_89577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_fraction_is_correct_l895_89538

/-- The fraction of the volume of a sphere that circumscribes four mutually touching unit spheres,
    which also touch the circumscribing sphere. -/
noncomputable def volume_fraction : ℝ :=
  32 / (Real.sqrt 6 + 2)^3

/-- The radius of the circumscribing sphere. -/
noncomputable def outer_sphere_radius : ℝ := (Real.sqrt 6 + 2) / 2

/-- Theorem stating that the volume fraction is correct. -/
theorem volume_fraction_is_correct : 
  let outer_volume := (4 / 3) * Real.pi * outer_sphere_radius^3
  let inner_volumes := 4 * ((4 / 3) * Real.pi)
  volume_fraction = inner_volumes / outer_volume := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_fraction_is_correct_l895_89538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l895_89531

/-- The function f(x) as defined in the problem -/
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ :=
  Real.sqrt 3 * (Real.cos (ω * x))^2 + Real.sin (ω * x) * Real.cos (ω * x) - Real.sqrt 3 / 2

/-- The property that ω is positive -/
def is_positive (ω : ℝ) : Prop := ω > 0

/-- The property that the smallest positive period of f is π -/
def has_period_pi (ω : ℝ) : Prop :=
  ∀ x : ℝ, f ω (x + Real.pi) = f ω x ∧ ∀ t : ℝ, 0 < t → t < Real.pi → ∃ y : ℝ, f ω (y + t) ≠ f ω y

/-- The interval where f is monotonically decreasing -/
def monotonic_decreasing_interval (ω : ℝ) (k : ℤ) : Set ℝ :=
  Set.Icc (Real.pi / 12 + k * Real.pi) (7 * Real.pi / 12 + k * Real.pi)

/-- The main theorem -/
theorem f_properties (ω : ℝ) (h1 : is_positive ω) (h2 : has_period_pi ω) :
  ω = 1 ∧ ∀ x : ℝ, (∃ k : ℤ, x ∈ monotonic_decreasing_interval ω k) ↔ 
    ∀ y : ℝ, y ∈ Set.Ioo x (x + Real.pi / 6) → f ω y ≤ f ω x :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l895_89531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_chord_theorem_l895_89559

-- Define the points
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (2, 0)
def B : ℝ × ℝ := (2, 2)

-- Define the line l
def l (x : ℝ) : ℝ := x - 1

-- Define the circle M
def M (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 2

-- Define the chord PQ
noncomputable def chord_PQ : ℝ := Real.sqrt 6

theorem circle_and_chord_theorem :
  -- The circle M passes through O, A, and B
  M O.1 O.2 ∧ M A.1 A.2 ∧ M B.1 B.2 ∧
  -- The line l intersects the circle M at two points
  ∃ P Q : ℝ × ℝ, P ≠ Q ∧ M P.1 P.2 ∧ M Q.1 Q.2 ∧ P.2 = l P.1 ∧ Q.2 = l Q.1 ∧
  -- The length of chord PQ is √6
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = chord_PQ :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_chord_theorem_l895_89559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_segment_length_range_l895_89576

-- Define the ellipse
def ellipse (a : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 = 1

-- Define the circle intersecting the ellipse
def circle_intersects_ellipse (a : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), x₁ ≠ x₂ ∧ ellipse a x₁ y₁ ∧ ellipse a x₂ y₂ ∧
  (x₁ + 1)^2 + y₁^2 = 4 ∧ (x₂ + 1)^2 + y₂^2 = 4

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop :=
  y = k * (x + 1)

-- Define the perpendicular bisector of AB
def perp_bisector (k : ℝ) (x y : ℝ) : Prop :=
  y - k / (1 + 2 * k^2) = -1/k * (x + (2 * k^2 - 2) / (1 + 2 * k^2))

-- Define the x-coordinate of point P
noncomputable def x_coord_P (k : ℝ) : ℝ :=
  -k^2 / (1 + 2 * k^2)

-- Define the length of AB
noncomputable def length_AB (k : ℝ) : ℝ :=
  (2 * Real.sqrt 2 * (1 + k^2)) / (1 + 2 * k^2)

theorem ellipse_segment_length_range (a : ℝ) (k : ℝ) :
  a > 0 →
  circle_intersects_ellipse a →
  k ≠ 0 →
  (∃ (x₁ y₁ x₂ y₂ : ℝ), x₁ ≠ x₂ ∧
    ellipse a x₁ y₁ ∧ ellipse a x₂ y₂ ∧
    line_l k x₁ y₁ ∧ line_l k x₂ y₂) →
  -1/4 < x_coord_P k ∧ x_coord_P k < 0 →
  3 * Real.sqrt 2 / 2 < length_AB k ∧ length_AB k < 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_segment_length_range_l895_89576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_department_selection_l895_89506

/-- Calculates the number of selected students within a given range for a systematic sample. -/
def selectedInRange (totalStudents : ℕ) (sampleSize : ℕ) (firstSelected : ℕ) (rangeStart : ℕ) (rangeEnd : ℕ) : ℕ :=
  let interval := totalStudents / sampleSize
  let inRange (n : ℕ) := rangeStart ≤ firstSelected + interval * n ∧ firstSelected + interval * n ≤ rangeEnd
  (Finset.range sampleSize).filter inRange |>.card

/-- Theorem stating that 17 students from the second department (301-495) are selected in the systematic sample. -/
theorem second_department_selection :
  selectedInRange 600 50 3 301 495 = 17 := by
  sorry

#eval selectedInRange 600 50 3 301 495

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_department_selection_l895_89506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_outfit_count_l895_89546

/-- Represents the colors available for clothing items -/
inductive Color
| Tan | Black | Blue | Gray | Red | White | Purple

/-- Represents a clothing item -/
structure ClothingItem where
  color : Color

/-- Represents an outfit -/
structure Outfit where
  shirt : ClothingItem
  pants : ClothingItem
  hat : ClothingItem

def is_valid_outfit (o : Outfit) : Prop :=
  ¬(o.shirt.color = o.pants.color ∧ o.pants.color = o.hat.color)

def num_shirts : Nat := 9
def num_pants : Nat := 5
def num_hats : Nat := 7

def pants_colors : List Color := [Color.Tan, Color.Black, Color.Blue, Color.Gray, Color.Red]
def shirt_hat_colors : List Color := [Color.Tan, Color.Black, Color.Blue, Color.Gray, Color.Red, Color.White, Color.Purple]

theorem valid_outfit_count :
  (num_shirts * num_pants * num_hats) - (pants_colors.length) = 310 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_outfit_count_l895_89546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_calculation_l895_89527

/-- Calculate simple interest rate given principal, time, and interest earned -/
noncomputable def simple_interest_rate (principal : ℝ) (time : ℝ) (interest : ℝ) : ℝ :=
  (interest * 100) / (principal * time)

/-- Calculate compound interest given principal, rate, and time -/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate / 100) ^ time - principal

/-- Theorem: Given the conditions, the compound interest earned is 512.5 -/
theorem compound_interest_calculation 
  (principal : ℝ) 
  (time : ℝ) 
  (simple_interest : ℝ) 
  (h1 : principal = 5000) 
  (h2 : time = 2) 
  (h3 : simple_interest = 500) :
  compound_interest principal (simple_interest_rate principal time simple_interest) time = 512.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_calculation_l895_89527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_in_pi_range_l895_89544

theorem count_integers_in_pi_range : 
  (Finset.range (Int.toNat (Int.floor (15 * Real.pi) - Int.ceil (-5 * Real.pi) + 1))).card = 63 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_in_pi_range_l895_89544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_quadrant_trig_identity_l895_89591

theorem fourth_quadrant_trig_identity (θ m : ℝ) :
  Real.sin θ = (m - 3) / (m + 5) →
  Real.cos θ = (4 - 2*m) / (m + 5) →
  θ ∈ Set.Icc (3*Real.pi/2) (2*Real.pi) →
  m = 0 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_quadrant_trig_identity_l895_89591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_failing_percentage_proof_l895_89562

/-- Represents a class of students with a certain failing percentage -/
structure StudentClass where
  total : ℕ
  failing : ℕ

/-- The percentage of failing students in a class -/
def failingPercentage (c : StudentClass) : ℚ :=
  (c.failing : ℚ) / (c.total : ℚ) * 100

theorem failing_percentage_proof (c : StudentClass) :
  (failingPercentage { total := c.total, failing := c.failing - 1 } = 24) →
  (failingPercentage { total := c.total - 1, failing := c.failing - 1 } = 25) →
  failingPercentage c = 28 := by
  sorry

#eval Int.floor (failingPercentage { total := 25, failing := 7 })

end NUMINAMATH_CALUDE_ERRORFEEDBACK_failing_percentage_proof_l895_89562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_6_eq_eval_3_5a_plus_3_5c_log_30_div_log_15_log_15_squared_minus_log_15_l895_89555

open Real

noncomputable section

variable (a b c : ℝ)

-- Define the given conditions
axiom log_2 : log 2 = a
axiom log_3 : log 3 = b
axiom log_5 : log 5 = c

-- Theorem statements
theorem log_6_eq : log 6 = a + b := by
  sorry

theorem eval_3_5a_plus_3_5c : 3.5 * a + 3.5 * c = 3.5 := by
  sorry

theorem log_30_div_log_15 : (log 30) / (log 15) = (a + b + c) / (b + c) := by
  sorry

theorem log_15_squared_minus_log_15 : (log 15)^2 - log 15 = (b + c) * (b + c - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_6_eq_eval_3_5a_plus_3_5c_log_30_div_log_15_log_15_squared_minus_log_15_l895_89555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_value_implies_index_l895_89539

def t : ℕ → ℚ
  | 0 => 3  -- Adding case for 0 to cover all natural numbers
  | 1 => 3
  | n + 2 => if (n + 2) % 2 = 0 then 2 + t ((n + 2) / 2) else 2 / t (n + 1)

theorem sequence_value_implies_index : t 6 = 7 / 29 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_value_implies_index_l895_89539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_chord_equation_for_given_point_and_circle_l895_89588

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A point in a 2D plane -/
def Point := ℝ × ℝ

/-- Check if a point is outside a circle -/
def is_outside (p : Point) (c : Circle) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 > c.radius^2

/-- The equation of the line containing tangency points -/
def tangent_chord_equation (p : Point) (c : Circle) : ℝ → ℝ → Prop :=
  λ x y ↦ let (x₀, y₀) := p; x₀ * x + y₀ * y = c.radius^2

theorem tangent_chord_equation_for_given_point_and_circle :
  let p : Point := (4, -5)
  let c : Circle := ⟨(0, 0), 2⟩
  is_outside p c →
  ∀ x y, tangent_chord_equation p c x y ↔ 4 * x - 5 * y = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_chord_equation_for_given_point_and_circle_l895_89588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_triangle_areas_l895_89596

/-- ABCD is a unit square -/
def is_unit_square (A B C D : ℝ × ℝ) : Prop :=
  A = (0, 0) ∧ B = (1, 0) ∧ C = (1, 1) ∧ D = (0, 1)

/-- R₁ divides CD in the ratio 1:3 (from C to D) -/
noncomputable def R₁ (C D : ℝ × ℝ) : ℝ × ℝ :=
  (3/4 * C.1 + 1/4 * D.1, 3/4 * C.2 + 1/4 * D.2)

/-- Sᵢ is the intersection of AR_i and BD -/
noncomputable def S (i : ℕ) (A B D : ℝ × ℝ) (R : ℕ → ℝ × ℝ) : ℝ × ℝ :=
  sorry  -- Definition of S_i based on A, B, D, and R_i

/-- R_{i+1} is on CD such that R_iR_{i+1} is 1/3 the length of R_iD -/
noncomputable def R : ℕ → ℝ × ℝ → ℝ × ℝ → ℝ × ℝ
  | 0, C, D => R₁ C D
  | n + 1, C, D => (2/3 * (R n C D).1 + 1/3 * D.1, 2/3 * (R n C D).2 + 1/3 * D.2)

/-- Area of triangle DR_iS_i -/
noncomputable def triangle_area (i : ℕ) (D : ℝ × ℝ) (R : ℕ → ℝ × ℝ) (S : ℕ → ℝ × ℝ) : ℝ :=
  sorry  -- Definition of triangle area based on D, R_i, and S_i

/-- The main theorem -/
theorem sum_of_triangle_areas (A B C D : ℝ × ℝ) :
  is_unit_square A B C D →
  (∑' i, triangle_area i D (R · C D) (S · A B D (R · C D))) = 1/128 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_triangle_areas_l895_89596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_passes_through_points_l895_89548

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := x^2 + 2*y^2 = 2

/-- The line equation -/
def line (x y : ℝ) : Prop := x + 2*y - 1 = 0

/-- Point A -/
def A : ℝ × ℝ := (2, 2)

/-- The circle equation -/
def circle_eq (x y : ℝ) : Prop := 6*x^2 + 6*y^2 - 8*x - 12*y - 3 = 0

/-- Theorem stating that the circle passes through A and the intersection points of the ellipse and line -/
theorem circle_passes_through_points :
  ∃ (B C : ℝ × ℝ),
    (ellipse B.1 B.2 ∧ line B.1 B.2) ∧
    (ellipse C.1 C.2 ∧ line C.1 C.2) ∧
    B ≠ C ∧
    circle_eq A.1 A.2 ∧
    circle_eq B.1 B.2 ∧
    circle_eq C.1 C.2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_passes_through_points_l895_89548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kens_change_is_correct_l895_89564

/-- Represents the price of items and sale conditions --/
structure PriceInfo where
  steak_price : ℝ
  steak_discount : ℝ
  eggs_price : ℝ
  milk_price : ℝ
  bagels_price : ℝ
  tax_rate : ℝ

/-- Represents the amount paid by Ken --/
structure Payment where
  bill_20 : ℕ
  bill_10 : ℕ
  bill_5 : ℕ
  coin_1 : ℕ
  coin_075 : ℕ

/-- Calculates the change Ken receives --/
def calculate_change (prices : PriceInfo) (payment : Payment) : ℝ :=
  let steak_cost := prices.steak_price + prices.steak_price * (1 - prices.steak_discount)
  let subtotal := steak_cost + prices.eggs_price + prices.milk_price + prices.bagels_price
  let total_cost := subtotal * (1 + prices.tax_rate)
  let amount_paid := (payment.bill_20 : ℝ) * 20 + (payment.bill_10 : ℝ) * 10 + (payment.bill_5 : ℝ) * 5 +
                     (payment.coin_1 : ℝ) * 1 + (payment.coin_075 : ℝ) * 0.75
  amount_paid - total_cost

/-- Theorem stating that Ken's change is $18.60 --/
theorem kens_change_is_correct (prices : PriceInfo) (payment : Payment) :
  prices.steak_price = 7 →
  prices.steak_discount = 0.5 →
  prices.eggs_price = 3 →
  prices.milk_price = 4 →
  prices.bagels_price = 6 →
  prices.tax_rate = 0.07 →
  payment.bill_20 = 1 →
  payment.bill_10 = 1 →
  payment.bill_5 = 2 →
  payment.coin_1 = 3 →
  payment.coin_075 = 1 →
  calculate_change prices payment = 18.60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kens_change_is_correct_l895_89564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_number_positive_l895_89536

-- Define the discriminant function
def discriminant (a b : ℝ) : ℝ := b^2 - 4*a

-- Define the transformation rules
inductive BoardPair : ℝ × ℝ → Prop where
  | initial : BoardPair (1, 1)
  | trans1 (x y : ℝ) : BoardPair (x, y-1) → BoardPair (x+y, y+1)
  | trans2 (x y : ℝ) : BoardPair (x+y, y+1) → BoardPair (x, y-1)
  | trans3 (x y : ℝ) : BoardPair (x, x*y) → BoardPair (1/x, y)
  | trans4 (x y : ℝ) : BoardPair (1/x, y) → BoardPair (x, x*y)

-- State the theorem
theorem first_number_positive (a b : ℝ) :
  BoardPair (a, b) → a > 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_number_positive_l895_89536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_at_one_l895_89553

/-- The function f(x) = (log x) / x -/
noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

/-- The statement that the derivative of f at x = 1 is equal to 1 / (ln 10) -/
theorem derivative_f_at_one : 
  deriv f 1 = 1 / Real.log 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_at_one_l895_89553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_line_ellipse_intersection_l895_89586

/-- The length of the chord formed by the intersection of a line and an ellipse -/
theorem chord_length_line_ellipse_intersection :
  let line := fun x : ℝ => x + 1
  let ellipse := fun (x y : ℝ) => x^2/4 + y^2 = 1
  let intersection_points := {p : ℝ × ℝ | ellipse p.1 p.2 ∧ p.2 = line p.1}
  let chord_length := Real.sqrt (2 * (8/5)^2)
  ∃ A B : ℝ × ℝ, A ∈ intersection_points ∧ B ∈ intersection_points ∧ dist A B = chord_length :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_line_ellipse_intersection_l895_89586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l895_89549

noncomputable def f (x : ℝ) := 1 / (x + 9) + 1 / (x^2 - 9) + 1 / (x^4 + 9)

theorem f_domain :
  {x : ℝ | ∃ y, f x = y} = {x | x < -9 ∨ (-9 < x ∧ x < -3) ∨ (-3 < x ∧ x < 3) ∨ 3 < x} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l895_89549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l895_89504

noncomputable section

/-- A quadratic function f(x) with specific properties -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 - (4/3) * a * x + b

/-- The derivative of f(x) -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 2 * a * x - (4/3) * a

theorem quadratic_function_properties (a b : ℝ) :
  f a b 1 = 2 ∧ f_derivative a 1 = 1 →
  (∀ x, f a b x = (3/2) * x^2 - 2 * x + 5/2) ∧
  (∀ x y, x - y + 1 = 0 ↔ y = f a b 1 + f_derivative a 1 * (x - 1)) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l895_89504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_sin_cos_greater_than_sqrt6_over_2_l895_89585

open Real MeasureTheory Measure

theorem probability_sin_cos_greater_than_sqrt6_over_2 :
  let X := Set.Icc (0 : ℝ) π
  let E := {x ∈ X | sin x + cos x > Real.sqrt 6 / 2}
  (volume.restrict X) E / volume X = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_sin_cos_greater_than_sqrt6_over_2_l895_89585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_b_c_relationship_l895_89519

-- Define constants for degree-to-radian conversion
noncomputable def deg_to_rad : ℝ := Real.pi / 180

-- Define a, b, and c as in the problem
noncomputable def a : ℝ := (Real.sqrt 2 / 2) * (Real.sin (17 * deg_to_rad) + Real.cos (17 * deg_to_rad))
noncomputable def b : ℝ := 2 * (Real.cos (13 * deg_to_rad))^2 - 1
noncomputable def c : ℝ := Real.sin (37 * deg_to_rad) * Real.sin (67 * deg_to_rad) + 
                 Real.sin (53 * deg_to_rad) * Real.sin (23 * deg_to_rad)

-- Theorem to prove
theorem a_b_c_relationship : c < a ∧ a < b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_b_c_relationship_l895_89519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_price_is_60_cents_l895_89535

/-- Represents the price and quantity of fruits --/
structure FruitData where
  apple_price : ℚ
  orange_price : ℚ
  total_fruits : ℕ
  initial_avg_price : ℚ
  final_avg_price : ℚ
  oranges_removed : ℕ

/-- Theorem stating that given the problem conditions, the price of each orange is 60 cents --/
theorem orange_price_is_60_cents (data : FruitData)
  (h1 : data.apple_price = 40/100)
  (h2 : data.total_fruits = 20)
  (h3 : data.initial_avg_price = 56/100)
  (h4 : data.final_avg_price = 52/100)
  (h5 : data.oranges_removed = 10) :
  data.orange_price = 60/100 := by
  sorry

/-- Example calculation using the theorem --/
def example_calculation : Bool :=
  let data : FruitData :=
    { apple_price := 40/100
    , orange_price := 60/100
    , total_fruits := 20
    , initial_avg_price := 56/100
    , final_avg_price := 52/100
    , oranges_removed := 10 }
  orange_price_is_60_cents data rfl rfl rfl rfl rfl = rfl

#eval example_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_price_is_60_cents_l895_89535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_base_digit_sum_not_25_base_7_digit_sum_not_25_twelve_pow_four_is_20736_sum_of_digits_20736_base_10_largest_base_theorem_l895_89575

/-- The sum of digits of a natural number n in base b -/
def sumOfDigits (n : ℕ) (b : ℕ) : ℕ := sorry

/-- Convert a number from base 10 to another base -/
def toBase (n : ℕ) (b : ℕ) : ℕ := sorry

theorem largest_base_digit_sum_not_25 : 
  ∀ b : ℕ, b > 7 → sumOfDigits (toBase 20736 b) b = 25 :=
by sorry

theorem base_7_digit_sum_not_25 : 
  sumOfDigits (toBase 20736 7) 7 ≠ 25 :=
by sorry

theorem twelve_pow_four_is_20736 : 12^4 = 20736 := by sorry

theorem sum_of_digits_20736_base_10 : sumOfDigits 20736 10 = 18 := by sorry

/-- 7 is the largest base b such that the sum of digits of 12^4 in base b is not equal to 25 -/
theorem largest_base_theorem : 
  ∀ b : ℕ, b > 7 → sumOfDigits (toBase (12^4) b) b = 25 ∧
  sumOfDigits (toBase (12^4) 7) 7 ≠ 25 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_base_digit_sum_not_25_base_7_digit_sum_not_25_twelve_pow_four_is_20736_sum_of_digits_20736_base_10_largest_base_theorem_l895_89575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spider_can_eat_all_flies_l895_89561

/-- Represents a position on the grid -/
structure Position :=
  (x : Nat)
  (y : Nat)

/-- Represents a move on the grid -/
inductive Move
  | up
  | down
  | left
  | right

/-- The web grid -/
def WebGrid :=
  {p : Position // p.x < 100 ∧ p.y < 100}

/-- A path on the grid is a list of moves -/
def SpiderPath := List Move

/-- The set of positions where flies are located -/
def FlyPositions := Finset WebGrid

/-- Function to check if a path visits all fly positions -/
def visitsAllFlies (path : SpiderPath) (flies : FlyPositions) : Prop :=
  sorry

/-- Function to count the number of moves in a path -/
def countMoves (path : SpiderPath) : Nat :=
  path.length

theorem spider_can_eat_all_flies :
  ∀ (flies : FlyPositions),
    flies.card = 100 →
    ∃ (path : SpiderPath),
      visitsAllFlies path flies ∧
      countMoves path ≤ 1980 :=
by
  sorry

#check spider_can_eat_all_flies

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spider_can_eat_all_flies_l895_89561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l895_89530

-- Define the arithmetic sequence and its sum
noncomputable def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d
noncomputable def sequence_sum (a₁ d : ℝ) (n : ℕ) : ℝ := n * a₁ + (n * (n - 1) / 2) * d

-- State the theorem
theorem arithmetic_sequence_properties
  (a₁ d : ℝ)
  (h₁ : a₁ > 0)
  (h₂ : sequence_sum a₁ d 8 = sequence_sum a₁ d 16) :
  d < 0 ∧
  arithmetic_sequence a₁ d 13 < 0 ∧
  (∀ n : ℕ, sequence_sum a₁ d n ≤ sequence_sum a₁ d 12) ∧
  (∀ n : ℕ, sequence_sum a₁ d n < 0 → n ≥ 25) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l895_89530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cafe_meeting_probability_l895_89545

/-- The probability that Alice and Charlie see each other at the café -/
noncomputable def probability_meet : ℝ := 5 / 9

/-- The duration of Alice and Charlie's stay at the café in hours -/
noncomputable def stay_duration : ℝ := 1 / 3

/-- The time range (in hours) during which Alice and Charlie can arrive -/
noncomputable def arrival_time_range : ℝ := 1

/-- Theorem stating the probability of Alice and Charlie meeting at the café -/
theorem cafe_meeting_probability :
  probability_meet = 1 - 2 * (1 / 2 * (arrival_time_range - stay_duration)^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cafe_meeting_probability_l895_89545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_pass_platform_approx_32_seconds_l895_89511

-- Define the given parameters
def train_speed_kmh : ℝ := 54
def time_to_pass_man : ℝ := 20
def platform_length : ℝ := 180.0144

-- Define the function to calculate the time to pass the platform
noncomputable def time_to_pass_platform : ℝ :=
  let train_speed_ms : ℝ := train_speed_kmh * 1000 / 3600
  let train_length : ℝ := train_speed_ms * time_to_pass_man
  let total_distance : ℝ := train_length + platform_length
  total_distance / train_speed_ms

-- Theorem statement
theorem time_to_pass_platform_approx_32_seconds :
  ⌊time_to_pass_platform⌋₊ = 32 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_pass_platform_approx_32_seconds_l895_89511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_circles_l895_89537

-- Define the circles
def circle_C1 (x y : ℝ) : Prop := (x - 4)^2 + (y - 2)^2 = 9
def circle_C2 (x y : ℝ) : Prop := (x + 2)^2 + (y + 1)^2 = 4

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- Theorem statement
theorem min_distance_between_circles :
  ∃ (min_dist : ℝ),
    (∀ (x1 y1 x2 y2 : ℝ),
      circle_C1 x1 y1 → circle_C2 x2 y2 →
      distance x1 y1 x2 y2 ≥ min_dist) ∧
    (∃ (x1 y1 x2 y2 : ℝ),
      circle_C1 x1 y1 ∧ circle_C2 x2 y2 ∧
      distance x1 y1 x2 y2 = min_dist) ∧
    min_dist = 3 * Real.sqrt 5 - 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_circles_l895_89537

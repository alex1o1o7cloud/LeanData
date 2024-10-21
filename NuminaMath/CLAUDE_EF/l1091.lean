import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_100_terms_l1091_109110

noncomputable def f (n : ℕ) : ℝ := (n^2 : ℝ) * Real.cos (n * Real.pi)

noncomputable def a (n : ℕ) : ℝ := f n + f (n + 1)

theorem sum_of_first_100_terms :
  (Finset.range 100).sum (fun i => a (i + 1)) = -100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_100_terms_l1091_109110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correlation_strength_absolute_value_l1091_109119

/-- Correlation coefficient between two variables -/
def correlation_coefficient (X Y : List ℝ) : ℝ := sorry

/-- Strength of correlation between two variables -/
def correlation_strength (r : ℝ) : ℝ := sorry

/-- Theorem stating that the correlation strength is not directly related to the raw value of r -/
theorem correlation_strength_absolute_value :
  ∃ X Y : List ℝ, ∃ r₁ r₂ : ℝ, 
    abs r₁ < abs r₂ ∧ 
    correlation_strength r₁ > correlation_strength r₂ ∧
    r₁ = correlation_coefficient X Y ∧
    r₂ = correlation_coefficient X Y :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correlation_strength_absolute_value_l1091_109119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_m_values_l1091_109181

theorem ellipse_eccentricity_m_values (m : ℝ) :
  (∀ x y : ℝ, x^2 / m + y^2 / 4 = 1) →
  (∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c^2 = a^2 - b^2 ∧ c / a = Real.sqrt 2 / 2) →
  m = 2 ∨ m = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_m_values_l1091_109181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_value_given_tan_l1091_109158

theorem cos_value_given_tan (α : ℝ) (h1 : Real.tan α = -Real.sqrt 15 / 4) 
  (h2 : α ∈ Set.Ioo (3 * Real.pi / 4) (2 * Real.pi)) : Real.cos α = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_value_given_tan_l1091_109158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_max_min_on_open_interval_l1091_109114

/-- The function f(x) = x^3 - 3x has neither a maximum value nor a minimum value on the open interval (-1, 1). -/
theorem no_max_min_on_open_interval :
  let f : ℝ → ℝ := fun x ↦ x^3 - 3*x
  ¬ (∃ (c : ℝ), c ∈ Set.Ioo (-1) 1 ∧ (∀ (x : ℝ), x ∈ Set.Ioo (-1) 1 → f x ≤ f c)) ∧
  ¬ (∃ (c : ℝ), c ∈ Set.Ioo (-1) 1 ∧ (∀ (x : ℝ), x ∈ Set.Ioo (-1) 1 → f x ≥ f c)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_max_min_on_open_interval_l1091_109114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_win_and_second_chance_area_l1091_109165

-- Define the spinner
noncomputable def spinner_radius : ℝ := 10

-- Define the probabilities
noncomputable def win_probability : ℝ := 1 / 4
noncomputable def second_chance_probability : ℝ := 1 / 2

-- Define the theorem
theorem win_and_second_chance_area :
  let total_area := π * spinner_radius^2
  let win_area := win_probability * total_area
  let second_chance_area := second_chance_probability * total_area
  win_area + second_chance_area = 75 * π := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_win_and_second_chance_area_l1091_109165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_abs_x_properties_l1091_109136

-- Define the function g
noncomputable def g (x : ℝ) : ℝ :=
  if x ≥ -2 ∧ x ≤ 1 then 2*x + 1
  else if x > 1 ∧ x ≤ 2 then -Real.sqrt (1 - (x-1)^2) + 1
  else if x > 2 ∧ x ≤ 3 then x - 2
  else 0  -- Define a default value for x outside the specified ranges

-- State the theorem
theorem g_abs_x_properties (x : ℝ) :
  (0 ≤ x ∧ x ≤ 3 → g (|x|) = g x) ∧
  (-3 ≤ x ∧ x < 0 → g (|x|) = g (-x)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_abs_x_properties_l1091_109136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_domain_g_l1091_109121

/-- The function g satisfying the given properties -/
noncomputable def g : ℝ → ℝ := sorry

/-- The domain of g -/
def domain_g : Set ℝ := {x : ℝ | x ≠ 0}

/-- Theorem stating that the domain of g is the largest possible -/
theorem largest_domain_g :
  (∀ x ∈ domain_g, (1 / x) ∈ domain_g) ∧
  (∀ x ∈ domain_g, g x + g (1 / x) = x^2) →
  ∀ S : Set ℝ, (∀ x ∈ S, (1 / x) ∈ S) ∧ (∀ x ∈ S, g x + g (1 / x) = x^2) →
  S ⊆ domain_g :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_domain_g_l1091_109121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_g_50_52_l1091_109178

def g (x : ℤ) : ℤ := x^2 - 2*x + 2022

theorem gcd_g_50_52 : Int.gcd (g 50) (g 52) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_g_50_52_l1091_109178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_l1091_109113

/-- A cubic function f(x) = ax³ + bx + 1 with a tangent line at (1, f(1)) with equation 4x - y - 1 = 0 -/
structure CubicWithTangent where
  a : ℝ
  b : ℝ
  f : ℝ → ℝ
  h1 : ∀ x, f x = a * x^3 + b * x + 1
  h2 : (λ x ↦ 4*x - f 1 - 1) = (λ x ↦ deriv f 1 * (x - 1) + f 1)

/-- The sum of coefficients a and b is 2 -/
theorem sum_of_coefficients (c : CubicWithTangent) : c.a + c.b = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_l1091_109113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_trajectory_l1091_109154

noncomputable def trajectory_of_midpoint (k : ℝ) : Prop :=
  k > 0 →
  ∃ (xA yA xB yB xE yE xF yF : ℝ),
    -- Circle equation
    ∀ (x y : ℝ), (x - 2)^2 + (y - 4)^2 = k ↔ (x, y) ∈ (frontier {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 - 4)^2 ≤ k}) →
    -- Intersection with y-axis
    xA = 0 ∧ xB = 0 ∧ yA > yB ∧
    (0 - 2)^2 + (yA - 4)^2 = k ∧ (0 - 2)^2 + (yB - 4)^2 = k →
    -- Intersection with x-axis
    yE = 0 ∧ yF = 0 ∧ xE > xF ∧
    (xE - 2)^2 + (0 - 4)^2 = k ∧ (xF - 2)^2 + (0 - 4)^2 = k →
    -- Midpoint trajectory
    ∀ (x y : ℝ), 
      x = (xA + xE) / 2 ∧ y = (yA + yE) / 2 →
      (y - 2)^2 - (x - 1)^2 = 3 ∧ x > 1 ∧ y > 2 + Real.sqrt 3

theorem midpoint_trajectory (k : ℝ) : trajectory_of_midpoint k := by
  sorry

#check midpoint_trajectory

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_trajectory_l1091_109154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1091_109143

-- Define the hyperbola and its parameters
variable (a b : ℝ)

-- Define the eccentricity function
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 + (b/a)^2)

-- Define the condition for the length of AB
noncomputable def AB_length_condition (a : ℝ) : ℝ := Real.sqrt 3 * a

-- State the theorem
theorem hyperbola_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : ∃ (A B : ℝ × ℝ), AB_length_condition a = (A.1 - B.1)^2 + (A.2 - B.2)^2) :
  eccentricity a b = 2 ∨ eccentricity a b = (2 * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1091_109143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_term_of_arithmetic_sequence_l1091_109125

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) : ℕ → ℝ
  | 0 => a₁
  | n + 1 => arithmetic_sequence a₁ d n + d

theorem third_term_of_arithmetic_sequence :
  let a₁ := 4
  let d := 2
  arithmetic_sequence a₁ d 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_term_of_arithmetic_sequence_l1091_109125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_9_equals_3_l1091_109170

-- Define the function g(x) = log(x-1) + 9
noncomputable def g (x : ℝ) : ℝ := Real.log (x - 1) + 9

-- Define the property of f being symmetric to g about y = x
def symmetric_about_y_eq_x (f : ℝ → ℝ) : Prop :=
  ∀ x y, f x = y ↔ g y = x

-- State the theorem
theorem f_9_equals_3 (f : ℝ → ℝ) (h : symmetric_about_y_eq_x f) : f 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_9_equals_3_l1091_109170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_minus_x_l1091_109162

-- Define the function g(x)
noncomputable def g (x : ℝ) : ℝ :=
  if -3 ≤ x ∧ x < -2 then x + 1
  else if -2 ≤ x ∧ x < 0 then x
  else if 0 ≤ x ∧ x < 2 then x - 1
  else if 2 ≤ x ∧ x ≤ 3 then x
  else 0  -- This case should never occur given the domain

-- State the theorem
theorem range_of_g_minus_x :
  Set.range (fun x => g x - x) = Set.Icc (-1 : ℝ) 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_minus_x_l1091_109162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_1234_3269_reappearance_1975_appearance_8197_l1091_109175

/-- Definition of the sequence -/
def seq : ℕ → ℕ
| 0 => 1
| 1 => 9
| 2 => 7
| 3 => 5
| n + 4 => (seq n + seq (n + 1) + seq (n + 2) + seq (n + 3)) % 10

/-- The digit groups 1234 and 3269 do not appear in the sequence -/
theorem no_1234_3269 : ∀ n : ℕ, 
  ¬(seq n = 1 ∧ seq (n + 1) = 2 ∧ seq (n + 2) = 3 ∧ seq (n + 3) = 4) ∧
  ¬(seq n = 3 ∧ seq (n + 1) = 2 ∧ seq (n + 2) = 6 ∧ seq (n + 3) = 9) := by
  sorry

/-- The digit group 1975 appears more than once in the sequence -/
theorem reappearance_1975 : ∃ m n : ℕ, m ≠ n ∧
  seq m = 1 ∧ seq (m + 1) = 9 ∧ seq (m + 2) = 7 ∧ seq (m + 3) = 5 ∧
  seq n = 1 ∧ seq (n + 1) = 9 ∧ seq (n + 2) = 7 ∧ seq (n + 3) = 5 := by
  sorry

/-- The digit group 8197 appears in the sequence -/
theorem appearance_8197 : ∃ n : ℕ, 
  seq n = 8 ∧ seq (n + 1) = 1 ∧ seq (n + 2) = 9 ∧ seq (n + 3) = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_1234_3269_reappearance_1975_appearance_8197_l1091_109175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_arithmetic_expression_evaluation_l1091_109100

theorem complex_arithmetic_expression_evaluation :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.00000001 ∧
  ((64 + 5 * 12) / (180 / 3) + Real.sqrt 49 - 2^3 * (4 * 3 * 2 * 1)) ∈ 
  Set.Icc (-182.93333334) (-182.93333332) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_arithmetic_expression_evaluation_l1091_109100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_liquid_transfer_height_l1091_109120

noncomputable def cone_volume (r h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

noncomputable def cylinder_volume (r h : ℝ) : ℝ := Real.pi * r^2 * h

theorem liquid_transfer_height : 
  let cone_radius : ℝ := 10
  let cone_height : ℝ := 20
  let cylinder_radius : ℝ := 20
  let cone_vol := cone_volume cone_radius cone_height
  let cylinder_height := cone_vol / (Real.pi * cylinder_radius^2)
  cylinder_height = 5/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_liquid_transfer_height_l1091_109120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_local_extrema_properties_l1091_109128

/-- The function f(x) = x³ + ax² + x + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + x + 1

theorem local_extrema_properties (a : ℝ) (x₁ x₂ : ℝ) :
  (∀ x : ℝ, deriv (f a) x = 3*x^2 + 2*a*x + 1) →
  IsLocalMax (f a) x₁ →
  IsLocalMin (f a) x₂ →
  x₁ ≠ x₂ →
  (a < -Real.sqrt 3 ∨ a > Real.sqrt 3) ∧
  (∀ y : ℝ, y > 0 → ∃ b : ℝ, (b < -Real.sqrt 3 ∨ b > Real.sqrt 3) ∧ f b x₁ - f b x₂ = y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_local_extrema_properties_l1091_109128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_is_984_l1091_109129

/-- Represents a trapezoid ABCD with perpendicular diagonals -/
structure Trapezoid where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  AB_length : ℝ
  CD_length : ℝ
  perpendicular_diagonals : (A.1 - C.1) * (B.1 - D.1) + (A.2 - C.2) * (B.2 - D.2) = 0

/-- The dot product of vectors AD and BC in the trapezoid -/
def dot_product_AD_BC (t : Trapezoid) : ℝ :=
  let AD := (t.D.1 - t.A.1, t.D.2 - t.A.2)
  let BC := (t.C.1 - t.B.1, t.C.2 - t.B.2)
  AD.1 * BC.1 + AD.2 * BC.2

/-- Theorem stating that the dot product of AD and BC is 984 for the given trapezoid -/
theorem dot_product_is_984 (t : Trapezoid) 
  (h1 : t.AB_length = 41)
  (h2 : t.CD_length = 24) :
  dot_product_AD_BC t = 984 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_is_984_l1091_109129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_ratio_l1091_109177

theorem inverse_proportion_ratio (x₁ x₂ y₁ y₂ k : ℝ) : 
  x₁ ≠ 0 → x₂ ≠ 0 → y₁ ≠ 0 → y₂ ≠ 0 →
  (∀ x y, x * y = k) →
  x₁ * y₁ = k → x₂ * y₂ = k →
  x₁ / x₂ = 3 / 4 →
  y₁ / y₂ = 4 / 3 := by
  intros hx₁ hx₂ hy₁ hy₂ h_prop hk₁ hk₂ h_ratio
  sorry

#check inverse_proportion_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_ratio_l1091_109177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_range_l1091_109148

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b
  k : b > 0

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (e.a^2 - e.b^2) / e.a

/-- The focal parameter c of an ellipse -/
noncomputable def focal_parameter (e : Ellipse) : ℝ :=
  Real.sqrt (e.a^2 - e.b^2)

/-- Theorem: If the maximum value of |PF₁| * |PF₂| is in [2c², 3c²],
    then the eccentricity is in [√3/3, √2/2] -/
theorem ellipse_eccentricity_range (e : Ellipse) :
  (2 * (focal_parameter e)^2 ≤ e.a^2 ∧ e.a^2 ≤ 3 * (focal_parameter e)^2) →
  (Real.sqrt 3 / 3 ≤ eccentricity e ∧ eccentricity e ≤ Real.sqrt 2 / 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_range_l1091_109148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_earnings_l1091_109107

theorem investment_earnings
  (investment_ratio_a investment_ratio_b investment_ratio_c : ℕ)
  (return_ratio_a return_ratio_b return_ratio_c : ℕ)
  (earnings_difference : ℕ)
  (h1 : investment_ratio_a = 3 ∧ investment_ratio_b = 4 ∧ investment_ratio_c = 5)
  (h2 : return_ratio_a = 6 ∧ return_ratio_b = 5 ∧ return_ratio_c = 4)
  (h3 : earnings_difference = 100)
  (h4 : investment_ratio_b * return_ratio_b - investment_ratio_a * return_ratio_a = 2 * earnings_difference) :
  (investment_ratio_a * return_ratio_a +
   investment_ratio_b * return_ratio_b +
   investment_ratio_c * return_ratio_c) * earnings_difference / 2 = 60000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_earnings_l1091_109107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_over_cos2alpha_l1091_109196

theorem sin_2alpha_over_cos2alpha (α : ℝ) 
  (h1 : Real.sin α = 3/5) 
  (h2 : α ∈ Set.Ioo (π/2) π) : 
  (Real.sin (2*α)) / ((Real.cos α)^2) = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_over_cos2alpha_l1091_109196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_expression_l1091_109160

noncomputable def a : ℕ → ℝ
  | 0 => 2  -- Add a case for 0 to avoid missing cases error
  | 1 => 2
  | n + 1 => a n + Real.log (1 + 1 / n)

theorem a_expression (n : ℕ) (hn : n ≥ 1) : a n = 2 + Real.log n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_expression_l1091_109160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_18_l1091_109182

-- Define the triangle XYZ
structure Triangle :=
  (X Y Z : ℝ × ℝ)

-- Define properties of the triangle
def isRightTriangle (t : Triangle) : Prop :=
  let (x1, y1) := t.X
  let (x2, y2) := t.Y
  let (x3, y3) := t.Z
  (x2 - x1) * (x3 - x1) + (y2 - y1) * (y3 - y1) = 0

def sideRatio (t : Triangle) : Prop :=
  let xy := ((t.X.1 - t.Y.1)^2 + (t.X.2 - t.Y.2)^2).sqrt
  let yz := ((t.Y.1 - t.Z.1)^2 + (t.Y.2 - t.Z.2)^2).sqrt
  let xz := ((t.X.1 - t.Z.1)^2 + (t.X.2 - t.Z.2)^2).sqrt
  xy = yz ∧ xz = xy * (2 : ℝ).sqrt

noncomputable def sumOfTwoSides (t : Triangle) : ℝ :=
  let xy := ((t.X.1 - t.Y.1)^2 + (t.X.2 - t.Y.2)^2).sqrt
  let yz := ((t.Y.1 - t.Z.1)^2 + (t.Y.2 - t.Z.2)^2).sqrt
  xy + yz

noncomputable def triangleArea (t : Triangle) : ℝ :=
  let xy := ((t.X.1 - t.Y.1)^2 + (t.X.2 - t.Y.2)^2).sqrt
  let yz := ((t.Y.1 - t.Z.1)^2 + (t.Y.2 - t.Z.2)^2).sqrt
  0.5 * xy * yz

-- Theorem statement
theorem triangle_area_is_18 (t : Triangle) 
  (h1 : isRightTriangle t) 
  (h2 : sideRatio t) 
  (h3 : sumOfTwoSides t = 12) : 
  triangleArea t = 18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_18_l1091_109182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_sets_l1091_109176

theorem union_of_sets : ∃ (P Q : Set ℕ), P ∪ Q = {1, 2, 3, 4, 6} := by
  -- Define sets P and Q
  let P : Set ℕ := {1, 3, 6}
  let Q : Set ℕ := {1, 2, 4, 6}

  -- Prove the existence of P and Q
  use P, Q

  -- State and prove the equality
  calc P ∪ Q
    = {1, 3, 6} ∪ {1, 2, 4, 6} := rfl
    _ = {1, 2, 3, 4, 6} := by
      ext x
      simp [Set.mem_union, Set.mem_singleton]
      tauto


end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_sets_l1091_109176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_failing_both_tests_l1091_109123

theorem probability_failing_both_tests 
  (total_students : ℕ) 
  (passed_first : ℕ) 
  (passed_second : ℕ) 
  (passed_both : ℕ) 
  (h1 : total_students = 100)
  (h2 : passed_first = 60)
  (h3 : passed_second = 40)
  (h4 : passed_both = 20) :
  (total_students - (passed_first + passed_second - passed_both)) / total_students = 1 / 5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_failing_both_tests_l1091_109123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_composition_equals_result_l1091_109163

-- Define the function g
def g (x : ℝ) : ℝ := 3 * x^2 + 3 * x + 1

-- Define the inverse function of g
noncomputable def g_inv (y : ℝ) : ℝ := (-3 + Real.sqrt (12 * y - 3)) / 6

-- Theorem statement
theorem inverse_composition_equals_result : g_inv (g_inv 37) = (-3 + Real.sqrt 129) / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_composition_equals_result_l1091_109163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_mixture_price_l1091_109198

/-- Calculates the desired selling price per pound for a candy mixture -/
theorem candy_mixture_price
  (cheap_price : ℚ)
  (expensive_price : ℚ)
  (total_weight : ℚ)
  (cheap_weight : ℚ)
  (h1 : cheap_price = 2)
  (h2 : expensive_price = 3)
  (h3 : total_weight = 80)
  (h4 : cheap_weight = 64) :
  (cheap_weight * cheap_price + (total_weight - cheap_weight) * expensive_price) / total_weight = 11/5 := by
  sorry

#eval (11 : ℚ) / 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_mixture_price_l1091_109198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_arithmetic_sequence_common_difference_l1091_109199

/-- An arithmetic sequence with given properties --/
structure ArithmeticSequence where
  firstTerm : ℚ
  lastTerm : ℚ
  sum : ℚ
  numTerms : ℕ

/-- The common difference of an arithmetic sequence --/
def commonDifference (seq : ArithmeticSequence) : ℚ :=
  (seq.lastTerm - seq.firstTerm) / (seq.numTerms - 1 : ℚ)

/-- Theorem stating the common difference of the specific arithmetic sequence --/
theorem specific_arithmetic_sequence_common_difference :
  ∃ (seq : ArithmeticSequence),
    seq.firstTerm = 3 ∧
    seq.lastTerm = 48 ∧
    seq.sum = 255 ∧
    seq.numTerms = 10 ∧
    commonDifference seq = 5 := by
  sorry

#eval commonDifference { firstTerm := 3, lastTerm := 48, sum := 255, numTerms := 10 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_arithmetic_sequence_common_difference_l1091_109199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_workers_time_l1091_109126

/-- The time needed for Alpha to complete the job alone -/
noncomputable def A : ℝ := sorry

/-- The time needed for Beta to complete the job alone -/
noncomputable def B : ℝ := sorry

/-- The time needed for Delta to complete the job alone -/
noncomputable def D : ℝ := sorry

/-- The time needed for all three workers to complete the job together -/
noncomputable def T : ℝ := sorry

/-- The conditions given in the problem -/
axiom cond1 : 1/A + 1/B + 1/D = 1/(A - 8)
axiom cond2 : 1/A + 1/B + 1/D = 1/(B + 3)
axiom cond3 : 1/A + 1/B + 1/D = 4/(3*D)

/-- The time needed for Alpha and Beta to complete the job together -/
noncomputable def AB_time : ℝ := 1/(1/A + 1/B)

/-- The main theorem to prove -/
theorem workers_time : AB_time = 2871/72 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_workers_time_l1091_109126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_opposite_side_l1091_109180

structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

def median (t : Triangle) (M : ℝ × ℝ) : Prop :=
  M.1 = (t.B.1 + t.C.1) / 2 ∧ M.2 = (t.B.2 + t.C.2) / 2

noncomputable def angle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

noncomputable def distance_point_to_line (p : ℝ × ℝ) (l : (ℝ × ℝ) × (ℝ × ℝ)) : ℝ := sorry

theorem distance_to_opposite_side (t : Triangle) (M : ℝ × ℝ) :
  median t M →
  angle t.C t.A M = π / 6 →
  Real.sqrt ((t.A.1 - t.C.1)^2 + (t.A.2 - t.C.2)^2) = 2 →
  distance_point_to_line t.B (t.A, t.C) = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_opposite_side_l1091_109180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_athlete_difference_l1091_109142

/-- Calculates the difference in the number of athletes over two nights at a fitness camp -/
theorem athlete_difference (initial_athletes leaving_rate leaving_hours arriving_rate arriving_hours : ℕ)
  (h1 : initial_athletes = 300)
  (h2 : leaving_rate = 28)
  (h3 : leaving_hours = 4)
  (h4 : arriving_rate = 15)
  (h5 : arriving_hours = 7) :
  initial_athletes - 
  (initial_athletes - leaving_rate * leaving_hours + arriving_rate * arriving_hours) = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_athlete_difference_l1091_109142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_increasing_m_value_l1091_109146

/-- A power function with coefficient and exponent depending on m -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m^2 - m - 1) * x^(-5*m - 3)

/-- The theorem stating that if f is a power function and increasing on (0, +∞), then m = -1 -/
theorem power_function_increasing_m_value (m : ℝ) :
  (∃ (a b : ℝ), ∀ x > 0, f m x = a * x^b) →  -- f is a power function
  (∀ x y, 0 < x ∧ x < y → f m x < f m y) →  -- f is increasing on (0, +∞)
  m = -1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_increasing_m_value_l1091_109146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axis_of_shifted_sine_l1091_109156

noncomputable def original_function (x : ℝ) : ℝ := Real.sin (Real.pi / 6 - 2 * x)

noncomputable def shifted_function (x : ℝ) : ℝ := original_function (x - Real.pi / 12)

theorem symmetry_axis_of_shifted_sine :
  ∀ x : ℝ, shifted_function (5 * Real.pi / 12 + x) = shifted_function (5 * Real.pi / 12 - x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axis_of_shifted_sine_l1091_109156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_750_degrees_l1091_109185

theorem sin_750_degrees : Real.sin (750 * Real.pi / 180) = 1/2 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_750_degrees_l1091_109185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_A_and_B_l1091_109132

def A : Set ℝ := {x | (x - 2) / (x + 1) ≤ 0}

def B : Set ℝ := {x | -2 * x^2 + 7 * x + 4 > 0}

theorem union_of_A_and_B : A ∪ B = Set.Ioo (-1 : ℝ) 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_A_and_B_l1091_109132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_k_power_difference_not_sum_l1091_109188

theorem infinite_k_power_difference_not_sum (k : ℕ) (hk : k > 1) :
  ∃ (S : Set ℕ), (¬Set.Finite S) ∧
  (∀ x ∈ S, ∃ (a b : ℕ), x = a^k - b^k) ∧
  (∀ x ∈ S, ¬∃ (c d : ℕ), x = c^k + d^k) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_k_power_difference_not_sum_l1091_109188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stair_climbing_theorem_l1091_109192

/-- Represents the stair climbing problem -/
structure StairClimbingProblem where
  numFlights : ℕ
  firstFlightTime : ℕ
  timeIncrement : ℕ
  stepsPerFlight : ℕ
  stepHeight : ℝ

/-- Calculate the total time taken to climb the stairs -/
def totalClimbingTime (p : StairClimbingProblem) : ℕ :=
  let lastFlightTime := p.firstFlightTime + (p.numFlights - 1) * p.timeIncrement
  p.numFlights * (p.firstFlightTime + lastFlightTime) / 2

/-- Calculate the total vertical distance climbed -/
noncomputable def totalClimbingDistance (p : StairClimbingProblem) : ℝ :=
  (p.numFlights * p.stepsPerFlight * p.stepHeight) / 100

/-- Theorem stating the correctness of the calculation -/
theorem stair_climbing_theorem (p : StairClimbingProblem) 
  (h1 : p.numFlights = 6)
  (h2 : p.firstFlightTime = 30)
  (h3 : p.timeIncrement = 10)
  (h4 : p.stepsPerFlight = 12)
  (h5 : p.stepHeight = 20) :
  totalClimbingTime p = 330 ∧ totalClimbingDistance p = 14.4 := by
  sorry

#eval totalClimbingTime {numFlights := 6, firstFlightTime := 30, timeIncrement := 10, stepsPerFlight := 12, stepHeight := 20}
-- Note: We can't use #eval for totalClimbingDistance because it's noncomputable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stair_climbing_theorem_l1091_109192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_employee_Y_base_pay_l1091_109109

/-- Represents the base weekly pay of Employee Y in Rs. -/
def P : ℝ := sorry

/-- The total weekly payment for all employees in Rs. -/
def total_payment : ℝ := 1800

/-- Employee Y's performance-based bonus in Rs. -/
def Y_bonus : ℝ := 100

/-- Employee X's overtime hours -/
def X_overtime_hours : ℝ := 2

/-- Overtime rate multiplier -/
def overtime_rate : ℝ := 1.5

/-- Theorem stating the relationship between Employee Y's base pay and the total payment -/
theorem employee_Y_base_pay :
  P * 7.98 + Y_bonus = total_payment := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_employee_Y_base_pay_l1091_109109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_distances_to_other_focus_l1091_109115

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- Checks if a point lies on the ellipse -/
def onEllipse (p : Point) (e : Ellipse) : Prop :=
  p.y^2 / e.b^2 + p.x^2 / e.a^2 = 1

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: Sum of distances from ellipse points to other focus -/
theorem sum_distances_to_other_focus
  (e : Ellipse)
  (f1 f2 a b : Point)
  (h1 : e.a = 2 ∧ e.b = 3)  -- Ellipse equation coefficients
  (h2 : onEllipse a e ∧ onEllipse b e)  -- A and B are on the ellipse
  (h3 : distance a b = 4)  -- |AB| = 4
  (h4 : ∃ (t : ℝ), a.x = f2.x + t * (b.x - f2.x) ∧ a.y = f2.y + t * (b.y - f2.y))  -- A, B, and F2 are collinear
  : distance a f1 + distance b f1 = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_distances_to_other_focus_l1091_109115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1091_109134

-- Define the function f(x) = 2x + 2/(x-1)
noncomputable def f (x : ℝ) : ℝ := 2*x + 2/(x-1)

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x > 1, f x ≥ a) ↔ a ≤ 6 := by
  sorry

#check range_of_a

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1091_109134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_constraints_l1091_109190

-- Define the linear function
def linear_function (a x : ℝ) : ℝ := (2*a + 1)*x + a - 2

-- Define the slope of the function
def slope_function (a : ℝ) : ℝ := 2*a + 1

-- Define the y-intercept of the function
def y_intercept (a : ℝ) : ℝ := a - 2

theorem linear_function_constraints (a : ℝ) :
  (slope_function a > 0 ∧ y_intercept a < 0) → (-1/2 < a ∧ a < 2) :=
by
  intro h
  constructor
  · -- Prove -1/2 < a
    have h1 : slope_function a > 0 := h.left
    simp [slope_function] at h1
    linarith
  · -- Prove a < 2
    have h2 : y_intercept a < 0 := h.right
    simp [y_intercept] at h2
    linarith

#check linear_function_constraints

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_constraints_l1091_109190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_angle_solution_l1091_109141

theorem smallest_positive_angle_solution (y : ℝ) :
  (∀ z : ℝ, z > 0 → z < y → Real.sin (4 * z) * Real.sin (5 * z) ≠ Real.cos (4 * z) * Real.cos (5 * z)) →
  Real.sin (4 * y) * Real.sin (5 * y) = Real.cos (4 * y) * Real.cos (5 * y) →
  y = Real.pi / 18 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_angle_solution_l1091_109141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_range_l1091_109117

noncomputable def a (n : ℕ) (p : ℝ) : ℝ := -n + p

noncomputable def b (n : ℕ) : ℝ := 3^(n - 4)

noncomputable def c (n : ℕ) (p : ℝ) : ℝ := max (a n p) (b n)

theorem p_range (p : ℝ) : 
  (∀ n : ℕ, n ≥ 1 → c n p > c 4 p) ↔ 4 < p ∧ p < 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_range_l1091_109117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_symmetry_x_axis_l1091_109194

-- Define the parametric equations
noncomputable def x (t : ℝ) : ℝ := (1/2) * Real.cos (2*t) + Real.sin t ^ 2
noncomputable def y (t : ℝ) : ℝ := Real.cos t + Real.sin t

-- Theorem statement
theorem curve_symmetry_x_axis : 
  ∀ t : ℝ, ∃ t' : ℝ, x t = x t' ∧ y t = -y t' :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_symmetry_x_axis_l1091_109194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_l1091_109138

-- Define propositions p and q as functions of m
def p (m : ℝ) : Prop := ∃ (x y : ℝ), 5 * x^2 + m * y^2 = 1 ∧ m > 5

def q (m : ℝ) : Prop := ∃ (x y : ℝ), (m + 1) * x^2 - m * y^2 = 1 ∧ m > 0

-- Define the theorem
theorem m_range : 
  (∀ m : ℝ, (p m ∧ q m → False) ∧ (p m ∨ q m)) → 
  ∃ m : ℝ, 0 < m ∧ m ≤ 5 :=
by
  sorry

-- Additional lemmas that might be useful for the proof
lemma p_condition (m : ℝ) : p m → m > 5 :=
by
  sorry

lemma q_condition (m : ℝ) : q m → m > 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_l1091_109138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_radii_l1091_109191

/-- A tetrahedron with three mutually perpendicular edges meeting at one vertex -/
structure Tetrahedron where
  edge1 : ℝ
  edge2 : ℝ
  edge3 : ℝ
  perpendicular : edge1 > 0 ∧ edge2 > 0 ∧ edge3 > 0

/-- The radius of the circumscribed sphere of the tetrahedron -/
noncomputable def circumradius (t : Tetrahedron) : ℝ :=
  Real.sqrt (t.edge1^2 + t.edge2^2 + t.edge3^2) / 2

/-- The radius of the inscribed sphere of the tetrahedron -/
noncomputable def inradius (t : Tetrahedron) : ℝ :=
  let s_total := t.edge1 * t.edge2 / 2 + t.edge1 * t.edge3 / 2 + t.edge2 * t.edge3 / 2 +
                 Real.sqrt ((t.edge1^2 + t.edge2^2) * (t.edge1^2 + t.edge3^2) * (t.edge2^2 + t.edge3^2)) / 2
  24 / (s_total * 3)

theorem tetrahedron_radii (t : Tetrahedron) (h1 : t.edge1 = 3) (h2 : t.edge2 = 4) (h3 : t.edge3 = 4) :
  circumradius t = Real.sqrt 41 / 2 ∧
  inradius t = 12 * (10 - Real.sqrt 34) / 33 := by
  sorry

-- Remove #eval statements as they are not computable
-- #eval circumradius ⟨3, 4, 4, by norm_num⟩
-- #eval inradius ⟨3, 4, 4, by norm_num⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_radii_l1091_109191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functions_increasing_on_negative_interval_l1091_109166

-- Define the interval (-∞, 0)
noncomputable def openNegativeInterval : Set ℝ := {x : ℝ | x < 0}

-- Define the functions
def f (x : ℝ) : ℝ := x
def g (x : ℝ) : ℝ := -x^2
noncomputable def h (x : ℝ) : ℝ := -1/x

-- Theorem statement
theorem functions_increasing_on_negative_interval :
  (StrictMonoOn f openNegativeInterval) ∧
  (StrictMonoOn g openNegativeInterval) ∧
  (StrictMonoOn h openNegativeInterval) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_functions_increasing_on_negative_interval_l1091_109166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_touching_circles_l1091_109130

theorem sphere_touching_circles (R : ℝ) (r : ℝ) (s : ℝ) :
  R = 2 →
  r = 1 →
  s > 0 →
  (∃ (O : ℝ × ℝ × ℝ) (O₁ O₂ O₃ O₄ : ℝ × ℝ × ℝ),
    -- O is the center of the sphere
    ‖O‖ = 0 ∧
    -- O₁, O₂, O₃ are centers of the three circles
    ‖O₁ - O‖ = R ∧
    ‖O₂ - O‖ = R ∧
    ‖O₃ - O‖ = R ∧
    -- The three circles touch each other
    ‖O₁ - O₂‖ = 2 * r ∧
    ‖O₂ - O₃‖ = 2 * r ∧
    ‖O₃ - O₁‖ = 2 * r ∧
    -- O₄ is the center of the smaller circle
    ‖O₄ - O‖ = R - s ∧
    -- The smaller circle touches all three circles
    ‖O₄ - O₁‖ = r + s ∧
    ‖O₄ - O₂‖ = r + s ∧
    ‖O₄ - O₃‖ = r + s) →
  s = 1 - Real.sqrt (2/3) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_touching_circles_l1091_109130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_satisfies_conditions_l1091_109124

/-- A vector (a, b) that points from the origin to the line parameterized by
    x = 3s + 5 and y = 2s + 4, and is parallel to (3, 2) -/
noncomputable def vector_to_line : ℝ × ℝ := (6, 14/3)

/-- The line parameterized by x = 3s + 5 and y = 2s + 4 -/
def line (s : ℝ) : ℝ × ℝ := (3*s + 5, 2*s + 4)

/-- The direction vector -/
def direction_vector : ℝ × ℝ := (3, 2)

theorem vector_satisfies_conditions :
  ∃ (t : ℝ), vector_to_line = line t ∧
  ∃ (k : ℝ), vector_to_line = k • direction_vector := by
  sorry

#check vector_satisfies_conditions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_satisfies_conditions_l1091_109124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorization_example_l1091_109139

/-- Represents a factorization of a polynomial from left to right -/
def is_factorization (f g : ℝ → ℝ) : Prop :=
  ∀ x, f x = g x ∧ (∃ p q : ℝ → ℝ, g = λ y => (p y) * (q y))

/-- The specific polynomial and its factorization -/
def left_poly : ℝ → ℝ := λ x => x^2 - 2*x + 1
def right_poly : ℝ → ℝ := λ x => (x - 1)^2

/-- Theorem stating that the given equation represents a factorization -/
theorem factorization_example :
  is_factorization left_poly right_poly := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorization_example_l1091_109139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_reflection_in_polygon_l1091_109133

-- Define the polygon M
def Polygon (M : Set (ℝ × ℝ)) : Prop :=
  -- Add properties of a polygon here
  sorry

-- Define point symmetry for the polygon M
def PointSymmetrical (M : Set (ℝ × ℝ)) (O : ℝ × ℝ) : Prop :=
  ∀ p ∈ M, (2 * O.1 - p.1, 2 * O.2 - p.2) ∈ M

-- Define a triangle
def Triangle (T : Set (ℝ × ℝ)) : Prop :=
  ∃ A B C : ℝ × ℝ, T = {A, B, C}

-- Define that a point is inside a triangle
def InsideTriangle (p : ℝ × ℝ) (T : Set (ℝ × ℝ)) : Prop :=
  -- Add definition for a point being inside a triangle
  sorry

-- Define reflection of a point about another point
def Reflect (p : ℝ × ℝ) (center : ℝ × ℝ) : ℝ × ℝ :=
  (2 * center.1 - p.1, 2 * center.2 - p.2)

-- Define reflection of a set of points about a center
def ReflectSet (S : Set (ℝ × ℝ)) (center : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ q ∈ S, p = Reflect q center}

-- Theorem statement
theorem triangle_reflection_in_polygon 
  (M : Set (ℝ × ℝ)) (T T' : Set (ℝ × ℝ)) (O P : ℝ × ℝ) :
  Polygon M →
  PointSymmetrical M O →
  Triangle T →
  T ⊆ M →
  InsideTriangle P T →
  T' = ReflectSet T P →
  ∃ v ∈ T', v ∈ M := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_reflection_in_polygon_l1091_109133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_at_60_yuan_selling_price_for_1350_profit_l1091_109131

/-- Represents the craft sales model -/
structure CraftSales where
  cost : ℚ                -- Cost per piece
  basePrice : ℚ           -- Base selling price
  baseVolume : ℚ          -- Base daily sales volume
  priceIncrement : ℚ      -- Price increment
  volumeDecrement : ℚ     -- Volume decrement per price increment
  maxPrice : ℚ            -- Maximum allowed selling price

/-- Calculates the daily sales profit for a given selling price -/
def dailySalesProfit (model : CraftSales) (sellingPrice : ℚ) : ℚ :=
  let volumeChange := (sellingPrice - model.basePrice) / model.priceIncrement * model.volumeDecrement
  let currentVolume := model.baseVolume - volumeChange
  let profitPerPiece := sellingPrice - model.cost
  profitPerPiece * currentVolume

/-- Finds the selling price that yields a target daily profit -/
noncomputable def findSellingPrice (model : CraftSales) (targetProfit : ℚ) : ℚ :=
  sorry  -- The actual implementation would involve solving a quadratic equation

/-- Theorem stating the daily sales profit at 60 yuan selling price -/
theorem profit_at_60_yuan (model : CraftSales) 
  (h1 : model.cost = 40) 
  (h2 : model.basePrice = 50) 
  (h3 : model.baseVolume = 100) 
  (h4 : model.priceIncrement = 1) 
  (h5 : model.volumeDecrement = 2) 
  (h6 : model.maxPrice = 65) :
  dailySalesProfit model 60 = 1600 := by sorry

/-- Theorem stating the selling price for a daily profit of 1350 yuan -/
theorem selling_price_for_1350_profit (model : CraftSales) 
  (h1 : model.cost = 40) 
  (h2 : model.basePrice = 50) 
  (h3 : model.baseVolume = 100) 
  (h4 : model.priceIncrement = 1) 
  (h5 : model.volumeDecrement = 2) 
  (h6 : model.maxPrice = 65) :
  findSellingPrice model 1350 = 55 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_at_60_yuan_selling_price_for_1350_profit_l1091_109131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_integers_lcm_divisibility_no_solution_for_three_l1091_109173

theorem consecutive_integers_lcm_divisibility (n : ℕ) : n ≥ 4 → ∃ m : ℕ,
  (m > n) ∧
  (m ∣ (List.range' (m - (n - 1)) (n - 1)).prod) ∧
  (∀ k < n - 1, m - k > 0) :=
by sorry

theorem no_solution_for_three :
  ¬∃ m : ℕ, (m > 3) ∧
  (m ∣ Nat.lcm (m - 1) (m - 2)) ∧
  (∀ k < 2, m - k > 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_integers_lcm_divisibility_no_solution_for_three_l1091_109173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_twenty_new_trailers_l1091_109106

/-- Represents the trailer home situation on Maple Drive -/
structure TrailerSituation where
  initial_count : ℕ
  initial_average_age : ℕ
  years_passed : ℕ
  current_average_age : ℕ
  new_trailer_age : ℕ

/-- Calculates the number of new trailers added -/
def calculate_new_trailers (s : TrailerSituation) : ℚ :=
  let total_age_increase := s.initial_count * s.years_passed
  let n := (s.initial_count * (s.initial_average_age + s.years_passed) - s.current_average_age * s.initial_count) / 
           (s.current_average_age - s.new_trailer_age)
  n

/-- The main theorem stating that 20 new trailers were added -/
theorem twenty_new_trailers (s : TrailerSituation) 
  (h1 : s.initial_count = 30)
  (h2 : s.initial_average_age = 15)
  (h3 : s.years_passed = 3)
  (h4 : s.current_average_age = 12)
  (h5 : s.new_trailer_age = 3) :
  calculate_new_trailers s = 20 := by
  sorry

#eval calculate_new_trailers { 
  initial_count := 30, 
  initial_average_age := 15, 
  years_passed := 3, 
  current_average_age := 12, 
  new_trailer_age := 3 
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_twenty_new_trailers_l1091_109106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l1091_109149

/-- Sum of arithmetic sequence -/
noncomputable def S (a₁ d : ℝ) (n : ℕ) : ℝ := n * a₁ + (n * (n - 1) / 2) * d

/-- Theorem: For an arithmetic sequence with first term a₁ and common difference d,
    if S₆ = 36, Sₙ = 324, Sₙ₋₆ = 144, and n > 6, then n = 18 -/
theorem arithmetic_sequence_sum (a₁ d : ℝ) (n : ℕ) 
  (h₁ : S a₁ d 6 = 36)
  (h₂ : S a₁ d n = 324)
  (h₃ : S a₁ d (n - 6) = 144)
  (h₄ : n > 6) :
  n = 18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l1091_109149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_qin_jiushao_operations_l1091_109197

/-- Represents a polynomial function -/
def MyPolynomial (α : Type) := List α

/-- Qin Jiushao algorithm for polynomial evaluation -/
def qin_jiushao_eval (p : MyPolynomial ℝ) (x : ℝ) : ℝ × ℕ × ℕ :=
  sorry

/-- The specific polynomial f(x) = 5x^5 + 4x^4 + 3x^3 + 2x^2 + x + 1 -/
def f : MyPolynomial ℝ := [5, 4, 3, 2, 1, 1]

theorem qin_jiushao_operations :
  let (result, muls, adds) := qin_jiushao_eval f 5
  muls = 5 ∧ adds = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_qin_jiushao_operations_l1091_109197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sets_and_range_l1091_109103

-- Define sets A, B, and C
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 10}
def B : Set ℝ := {x | 2 * x - 6 ≥ 0}
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 1}

-- Define the theorem
theorem sets_and_range :
  (∃ (a : ℝ), C a ⊆ A) →
  (Set.compl (A ∪ B) = {x : ℝ | x < 3 ∨ x > 10}) ∧
  (∀ a : ℝ, C a ⊆ A ↔ -1 ≤ a ∧ a ≤ 9) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sets_and_range_l1091_109103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_polyhedron_volume_l1091_109183

-- Define the edge length of the hexahedron
def edge_length : ℚ := 1

-- Define the volume of a unit cube
noncomputable def unit_cube_volume : ℚ := edge_length ^ 3

-- Define the volume of one triangular pyramid
noncomputable def pyramid_volume : ℚ := (1/3) * (1/2) * (edge_length/2) ^ 3

-- Define the number of pyramids removed
def num_pyramids : ℕ := 8

-- Define the total volume of removed pyramids
noncomputable def total_removed_volume : ℚ := num_pyramids * pyramid_volume

-- Theorem statement
theorem remaining_polyhedron_volume :
  unit_cube_volume - total_removed_volume = 5/6 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_polyhedron_volume_l1091_109183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_l1091_109195

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (log x) / x

-- State the theorem
theorem inequality_holds (a b : ℝ) (h1 : 3 < a) (h2 : a < b) :
  f b < f ((a + b) / 2) ∧ f ((a + b) / 2) < f (Real.sqrt (a * b)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_l1091_109195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bc_length_is_two_l1091_109127

/-- A triangle with specific properties -/
structure SpecialTriangle where
  /-- Point A of the triangle -/
  A : EuclideanSpace ℝ (Fin 2)
  /-- Point B of the triangle -/
  B : EuclideanSpace ℝ (Fin 2)
  /-- Point C of the triangle -/
  C : EuclideanSpace ℝ (Fin 2)
  /-- The length of side AB is 1 -/
  ab_length : dist A B = 1
  /-- The length of side AC is 3 -/
  ac_length : dist A C = 3
  /-- The median from A to BC has the same length as BC -/
  median_eq_bc : let M := midpoint ℝ B C; dist A M = dist B C

/-- The length of BC in a SpecialTriangle is 2 -/
theorem bc_length_is_two (t : SpecialTriangle) : dist t.B t.C = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bc_length_is_two_l1091_109127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_half_angle_l1091_109111

theorem cos_half_angle (α : ℝ) (h1 : Real.sin α = 4/5) (h2 : 0 < α ∧ α < π/2) :
  Real.cos (α/2) = 2*Real.sqrt 5/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_half_angle_l1091_109111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_five_terms_l1091_109118

def geometricSequence (a₁ : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  a₁ * r^(n - 1)

def sumOfGeometricSequence (a₁ : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  if r = 1 then n * a₁
  else a₁ * (r^n - 1) / (r - 1)

theorem sum_of_first_five_terms :
  sumOfGeometricSequence 1 2 5 = 31 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_five_terms_l1091_109118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_projection_l1091_109145

theorem vector_projection (a b : Fin 2 → ℝ) 
  (h1 : Real.sqrt ((a 0)^2 + (a 1)^2) = 2) 
  (h2 : (a 0 * (b 0 - a 0) + a 1 * (b 1 - a 1)) = -3) : 
  (a 0 * b 0 + a 1 * b 1) / Real.sqrt ((a 0)^2 + (a 1)^2) = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_projection_l1091_109145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_increasing_l1091_109168

-- Define the function f(x) = log₁/₄(-x² + 2x + 3)
noncomputable def f (x : ℝ) : ℝ := Real.log (-x^2 + 2*x + 3) / Real.log (1/4)

-- Define the domain of f
def domain : Set ℝ := {x : ℝ | -1 < x ∧ x < 3}

-- State the theorem
theorem f_strictly_increasing :
  ∀ x ∈ domain, ∀ y ∈ domain, 
    1 ≤ x ∧ x < y ∧ y < 3 → f x < f y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_increasing_l1091_109168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_f_2x_minus_1_l1091_109140

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f(x+1)
def domain_f_x_plus_1 : Set ℝ := Set.Icc 1 2

-- State the theorem
theorem domain_f_2x_minus_1 :
  {x : ℝ | f (2*x - 1) = f (2*x - 1)} = Set.Icc (3/2) 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_f_2x_minus_1_l1091_109140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_75_degrees_l1091_109152

theorem sin_75_degrees :
  Real.sin (75 * π / 180) = (Real.sqrt 6 + Real.sqrt 2) / 4 := by
  sorry

/- Proof omitted -/

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_75_degrees_l1091_109152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_value_l1091_109112

theorem x_value (x y : ℝ) (h1 : (3 : ℝ)^x * (4 : ℝ)^y = 19683) (h2 : x - y = 9) : x = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_value_l1091_109112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equals_four_minus_sqrt_two_l1091_109187

-- Define the expression
noncomputable def expression : ℝ := ((-27/8) ^ (1/3 : ℝ)) + Real.sqrt 16 - |Real.sqrt 2 - 1| + Real.sqrt ((-1/2)^2)

-- State the theorem
theorem expression_equals_four_minus_sqrt_two : expression = 4 - Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equals_four_minus_sqrt_two_l1091_109187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equality_l1091_109104

theorem complex_equality (a : ℝ) (h1 : a > 0) : 
  (Complex.abs (2 + a * Complex.I) = Complex.abs (2 - Complex.I)) → a = 1 := by
  intro h
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equality_l1091_109104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_for_symmetry_l1091_109155

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 3)

noncomputable def shifted_f (ω : ℝ) (x : ℝ) : ℝ := f ω (x + Real.pi / 2)

def is_symmetric_about_y_axis (g : ℝ → ℝ) : Prop :=
  ∀ x, g x = g (-x)

theorem min_omega_for_symmetry :
  ∃ ω_min : ℝ, ω_min > 0 ∧
  (∀ ω : ℝ, ω > 0 →
    (is_symmetric_about_y_axis (shifted_f ω) → ω ≥ ω_min)) ∧
  ω_min = 1 / 3 := by
  sorry

#check min_omega_for_symmetry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_for_symmetry_l1091_109155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_problem_l1091_109150

noncomputable def f (x : ℝ) : ℝ := (4 / x) + (9 / (5 - x))

theorem inequality_problem :
  (∃ (m : ℝ), ∀ (x : ℝ), x ∈ Set.Ioo 0 5 → f x ≥ m) ∧
  (∀ (m : ℝ), (∀ (x : ℝ), x ∈ Set.Ioo 0 5 → f x ≥ m) → m ≤ 5) ∧
  (∀ (x : ℝ), |x - 5| + |x + 2| ≤ 9 ↔ -3 ≤ x ∧ x ≤ 6) := by
  sorry

#check inequality_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_problem_l1091_109150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1091_109167

open Real

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given condition
def given_condition (t : Triangle) : Prop :=
  (3 * t.a - t.b) / cos t.B = t.c / cos t.C

-- Define the law of sines
def law_of_sines (t : Triangle) : Prop :=
  t.a / sin t.A = t.b / sin t.B ∧ t.b / sin t.B = t.c / sin t.C

-- Define the cosine law
def cosine_law (t : Triangle) : Prop :=
  t.c^2 = t.a^2 + t.b^2 - 2 * t.a * t.b * cos t.C

-- Theorem statement
theorem triangle_properties (t : Triangle) 
  (h1 : given_condition t) 
  (h2 : law_of_sines t) 
  (h3 : cosine_law t) : 
  sin t.C = 2 * sqrt 2 / 3 ∧ 
  (t.c = sqrt 3 → ∃ (S : ℝ), S ≤ 3 * sqrt 2 / 4 ∧ 
    ∀ (S' : ℝ), S' = 1/2 * t.a * t.b * sin t.C → S' ≤ S) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1091_109167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_inequality_l1091_109179

theorem lcm_inequality (n : ℕ) : 
  Nat.lcm (Nat.lcm n (n + 1)) (n + 2) > Nat.lcm (Nat.lcm (n + 1) (n + 2)) (n + 3) ↔ 
  n ≥ 5 ∧ n % 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_inequality_l1091_109179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_matrix_A_even_l1091_109151

variable {G : Type*} [Group G] [Fintype G] [DecidableEq G]

def matrix_A (enum : Fin (Fintype.card G) → G) : Matrix (Fin (Fintype.card G)) (Fin (Fintype.card G)) ℤ :=
  λ i j => if (enum i * (enum j)⁻¹ = enum j * (enum i)⁻¹) then 0 else 1

theorem det_matrix_A_even (enum : Fin (Fintype.card G) → G) (h_enum : Function.Injective enum) :
  Even (Matrix.det (matrix_A enum)) := by
  sorry

#check det_matrix_A_even

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_matrix_A_even_l1091_109151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_inequality_l1091_109108

theorem exponential_inequality (m n : ℝ) : (2 : ℝ)^m > (2 : ℝ)^n → m > n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_inequality_l1091_109108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_l1091_109122

/-- The distance between the foci of an ellipse -/
noncomputable def distance_between_foci (a b : ℝ) : ℝ := 2 * Real.sqrt (a^2 - b^2)

/-- Theorem: For an ellipse with semi-major axis 8 and semi-minor axis 3, 
    the distance between the foci is 2√55 -/
theorem ellipse_foci_distance :
  distance_between_foci 8 3 = 2 * Real.sqrt 55 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_l1091_109122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1091_109147

noncomputable def f (x : ℝ) : ℝ := (1/2) ^ abs (x - 2)

theorem f_properties :
  (∀ x y, x < y ∧ y < 2 → f x < f y) ∧  -- f is increasing on (-∞, 2)
  (∀ x y, 2 < x ∧ x < y → f x > f y) ∧  -- f is decreasing on (2, +∞)
  (f 2 = 1) ∧                           -- maximum value is 1 at x = 2
  (∀ ε > 0, ∃ M, ∀ x, abs x > M → f x < ε)  -- f approaches 0 as x → ±∞
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1091_109147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tsunami_sand_amount_l1091_109137

/-- Represents the digging scenario of Pirate Rick --/
structure DiggingScenario where
  initial_time : ℚ
  initial_depth : ℚ
  final_time : ℚ

/-- Calculates the amount of new sand added by the tsunami --/
def tsunami_sand (scenario : DiggingScenario) : ℚ :=
  let digging_rate := scenario.initial_depth / scenario.initial_time
  let sand_after_storm := scenario.initial_depth / 2
  let final_depth := digging_rate * scenario.final_time
  final_depth - sand_after_storm

/-- Theorem stating the amount of new sand added by the tsunami --/
theorem tsunami_sand_amount (scenario : DiggingScenario) 
  (h1 : scenario.initial_time = 4)
  (h2 : scenario.initial_depth = 8)
  (h3 : scenario.final_time = 3) :
  tsunami_sand scenario = 2 := by
  sorry

def example_scenario : DiggingScenario := {
  initial_time := 4,
  initial_depth := 8,
  final_time := 3
}

#eval tsunami_sand example_scenario

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tsunami_sand_amount_l1091_109137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_theorem_l1091_109105

noncomputable section

def f : ℝ → ℝ := sorry

axiom f_derivative (x : ℝ) : deriv f x = Real.exp x * (2 * x + 3) + f x * Real.exp 1

axiom f_initial : f 0 = 1

def has_exactly_two_integer_solutions (k : ℝ) : Prop :=
  ∃ (a b : ℤ), (∀ x : ℤ, f x - k < 0 ↔ x = a ∨ x = b) ∧ a ≠ b

theorem f_range_theorem :
  {k : ℝ | has_exactly_two_integer_solutions k} = Set.Ioo (-1 / Real.exp 2) 0 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_theorem_l1091_109105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_simplification_l1091_109144

theorem sqrt_simplification : Real.sqrt (85 - 42 * Real.sqrt 5) = 7 - 3 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_simplification_l1091_109144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_ab_slope_constant_l1091_109101

theorem ellipse_ab_slope_constant (C : Set (ℝ × ℝ)) (P Q A B : ℝ × ℝ) :
  (∀ (x y : ℝ), ((x, y) ∈ C ↔ x^2/16 + y^2/12 = 1)) →
  P = (2, 3) →
  Q = (2, -3) →
  P ∈ C →
  Q ∈ C →
  A ∈ C →
  B ∈ C →
  (∃ (k : ℝ), ∀ (x y : ℝ), ((x, y) = A ∨ (x, y) = B) → (y - 3 = k * (x - 2) ∨ y - 3 = -k * (x - 2))) →
  (A.1 - 2) * (B.2 - 3) = (B.1 - 2) * (A.2 - 3) →
  (B.2 - A.2) / (B.1 - A.1) = 1/2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_ab_slope_constant_l1091_109101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bill_present_worth_l1091_109171

/-- Represents the present worth of a bill -/
def present_worth : ℝ → ℝ := sorry

/-- Represents the true discount of a bill -/
def true_discount : ℝ → ℝ := sorry

/-- Represents the banker's discount of a bill -/
def bankers_discount : ℝ → ℝ := sorry

/-- The relationship between banker's discount, true discount, and present worth -/
axiom discount_relation (bill : ℝ) :
  bankers_discount bill = (true_discount bill * (present_worth bill + true_discount bill)) / present_worth bill

/-- Theorem stating the present worth of the bill given the true and banker's discounts -/
theorem bill_present_worth (bill : ℝ) 
  (h1 : true_discount bill = 20)
  (h2 : bankers_discount bill = 21) :
  present_worth bill = 400 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bill_present_worth_l1091_109171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_set_connected_l1091_109169

/-- A non-convex and non-self-intersecting polygon on a plane -/
structure NonConvexPolygon where
  vertices : List (ℝ × ℝ)
  -- Add more conditions as needed

/-- The set of points belonging to the diagonals of the polygon, which do not go outside its boundaries -/
def DiagonalSet (p : NonConvexPolygon) : Set (ℝ × ℝ) :=
  sorry

/-- A broken line in the plane -/
structure BrokenLine where
  points : List (ℝ × ℝ)

/-- Theorem stating that any two points from the diagonal set can be connected by a broken line within the set -/
theorem diagonal_set_connected (p : NonConvexPolygon) :
  ∀ (x y : ℝ × ℝ), x ∈ DiagonalSet p → y ∈ DiagonalSet p →
  ∃ (l : BrokenLine), (∀ z, z ∈ l.points → z ∈ DiagonalSet p) ∧ 
    (l.points.head? = some x) ∧ (l.points.getLast? = some y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_set_connected_l1091_109169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_and_triangle_area_l1091_109157

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 8*y = 0

-- Define point P
def P : ℝ × ℝ := (2, 0)

-- Define the line l passing through P
def line_l (x y : ℝ) : Prop := ∃ (t : ℝ), x = 2 + t ∧ y = t

-- Define points A and B as intersections of line l and circle C
noncomputable def A : ℝ × ℝ := sorry
noncomputable def B : ℝ × ℝ := sorry

-- Define point M as the midpoint of AB
noncomputable def M : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Define area_triangle function
noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ := sorry

theorem line_equation_and_triangle_area :
  (∀ (x y : ℝ), line_l x y ↔ x + 2*y - 2 = 0) ∧
  (area_triangle A B (0, 4) = (12 * Real.sqrt 11) / 5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_and_triangle_area_l1091_109157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_inequality_range_l1091_109161

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := 2^x + 2^(-x) + |x|

-- State the theorem
theorem g_inequality_range (x : ℝ) :
  g (2 * x - 1) < g 3 ↔ -1 < x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_inequality_range_l1091_109161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_binomial_expansion_l1091_109189

theorem constant_term_binomial_expansion :
  ∃ (constant_term : ℝ),
  let binomial := (x : ℝ) + 2 / Real.sqrt x
  let expansion := binomial ^ 6
  constant_term = 240 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_binomial_expansion_l1091_109189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_250_of_18_div_7_l1091_109184

def decimal_representation (n d : ℕ) : List ℕ :=
  sorry

def repeating_sequence (l : List ℕ) : List ℕ :=
  sorry

def period (l : List ℕ) : ℕ :=
  sorry

theorem digit_250_of_18_div_7 :
  let rep := repeating_sequence (decimal_representation 18 7)
  let p := period rep
  (250 % p) = 4 →
  rep.get? ((250 % p) - 1) = some 4 :=
by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_250_of_18_div_7_l1091_109184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_result_l1091_109116

def initial_point : ℝ × ℝ × ℝ := (2, 3, 2)

def rotate_y_90 (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (p.2.2, p.2.1, -p.1)

def reflect_xz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (p.1, -p.2.1, p.2.2)

def rotate_z_90 (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (-p.2.1, p.1, p.2.2)

def reflect_yz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (-p.1, p.2.1, p.2.2)

theorem transformation_result :
  (reflect_yz ∘ rotate_z_90 ∘ reflect_xz ∘ rotate_y_90) initial_point = (-3, 2, -2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_result_l1091_109116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_fixed_points_l1091_109186

-- Define the line and ellipse
def line (x y m : ℝ) : Prop := Real.sqrt 2 * x - y + m = 0
def ellipse (x y : ℝ) : Prop := y^2 / 4 + x^2 / 2 = 1

-- Define the set M
def M : Set ℝ := {m : ℝ | m ∈ (Set.Ioo (-2 * Real.sqrt 2) 0) ∪ (Set.Ioo 0 (2 * Real.sqrt 2))}

-- Define the property of intersection at two distinct points
def intersects_at_two_points (m : ℝ) : Prop :=
  ∃ (x1 y1 x2 y2 : ℝ), x1 ≠ x2 ∧ 
    line x1 y1 m ∧ ellipse x1 y1 ∧
    line x2 y2 m ∧ ellipse x2 y2

-- Define the property of not passing through the origin
def not_through_origin (m : ℝ) : Prop := m ≠ 0

-- Define the property of complementary angles
def complementary_angles (x0 y0 m : ℝ) : Prop :=
  ∀ (x1 y1 x2 y2 : ℝ), 
    line x1 y1 m ∧ ellipse x1 y1 ∧
    line x2 y2 m ∧ ellipse x2 y2 →
    (y1 - y0) / (x1 - x0) + (y2 - y0) / (x2 - x0) = 0

-- State the theorem
theorem intersection_and_fixed_points :
  (∀ m : ℝ, m ∈ M ↔ (intersects_at_two_points m ∧ not_through_origin m)) ∧
  (∀ m : ℝ, m ∈ M → (complementary_angles 1 (Real.sqrt 2) m ∧ complementary_angles (-1) (-Real.sqrt 2) m)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_fixed_points_l1091_109186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_interval_l1091_109135

open Real

noncomputable def f (x : ℝ) : ℝ := 3 * sin (π / 3 - 2 * x)

theorem monotone_increasing_interval (k : ℤ) :
  StrictMonoOn f (Set.Icc (k * π + 5 * π / 12) (k * π + 11 * π / 12)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_interval_l1091_109135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_chord_length_l1091_109193

/-- The circle C with equation x^2 + y^2 - (6 - 2m)x - 4my + 5m^2 - 6m = 0 -/
def Circle (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - (6 - 2*m)*p.1 - 4*m*p.2 + 5*m^2 - 6*m = 0}

/-- The line l passing through point (1, 0) with slope -2 -/
def Line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = -2 * (p.1 - 1)}

/-- The length of the chord intercepted by circle C on line l -/
noncomputable def ChordLength (m : ℝ) : ℝ :=
  2 * Real.sqrt (9 - (4 / Real.sqrt 5)^2)

theorem constant_chord_length :
  ∀ m : ℝ, ChordLength m = (2 * Real.sqrt 145) / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_chord_length_l1091_109193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_count_l1091_109159

/-- The set S of 11 distinct integers -/
def S : Finset ℤ := sorry

/-- S has 11 distinct integers -/
axiom S_card : S.card = 11

/-- S contains 1, 4, 8, 10, 13, and 17 -/
axiom S_contains : {1, 4, 8, 10, 13, 17} ⊆ S

/-- Define the set of possible median values -/
def possible_medians : Finset ℤ := S.filter (fun x => 
  (S.filter (· < x)).card ≥ 5 ∧ (S.filter (· > x)).card ≥ 5)

/-- Theorem: The number of possible median values is 6 -/
theorem median_count : possible_medians.card = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_count_l1091_109159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_divisor_for_natural_fractions_l1091_109153

theorem largest_divisor_for_natural_fractions :
  let fractions : List ℚ := [154/195, 385/156, 231/130]
  let divisor : ℚ := 77/780
  (∀ f ∈ fractions, (f / divisor).num % (f / divisor).den = 0) ∧
  (∀ d : ℚ, d > divisor → ∃ f ∈ fractions, (f / d).num % (f / d).den ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_divisor_for_natural_fractions_l1091_109153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jasmine_coffee_beans_l1091_109164

/-- The number of pounds of coffee beans Jasmine bought -/
def coffee_beans : ℝ := sorry

/-- The cost of one pound of coffee beans in dollars -/
def coffee_cost : ℝ := 2.50

/-- The number of gallons of milk Jasmine bought -/
def milk_gallons : ℝ := 2

/-- The cost of one gallon of milk in dollars -/
def milk_cost : ℝ := 3.50

/-- The total amount Jasmine paid in dollars -/
def total_cost : ℝ := 17

/-- Theorem stating that Jasmine bought 4 pounds of coffee beans -/
theorem jasmine_coffee_beans :
  coffee_beans * coffee_cost + milk_gallons * milk_cost = total_cost →
  coffee_beans = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jasmine_coffee_beans_l1091_109164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zoo_revenue_calculation_l1091_109174

/-- Represents the revenue calculation for a zoo over two days with specific visitor numbers and pricing rules. -/
theorem zoo_revenue_calculation (
  -- Monday visitors
  monday_children : ℕ)
  (monday_adults : ℕ)
  (monday_seniors : ℕ)
  (monday_student_groups : ℕ)
  
  -- Tuesday visitors
  (tuesday_children : ℕ)
  (tuesday_adults : ℕ)
  (tuesday_seniors : ℕ)
  (tuesday_student_groups : ℕ)
  
  -- Ticket prices
  (monday_child_price : ℚ)
  (monday_adult_price : ℚ)
  (monday_senior_price : ℚ)
  (monday_student_group_price : ℚ)
  
  (tuesday_child_price : ℚ)
  (tuesday_adult_price : ℚ)
  (tuesday_senior_price : ℚ)
  (tuesday_student_group_price : ℚ)
  
  -- Promotional rules
  (monday_promo_child_price : ℚ)
  (monday_promo_limit : ℕ)
  (tuesday_discount_rate : ℚ)
  (student_group_size : ℕ)
  
  -- Assumptions
  (h1 : monday_children = 7)
  (h2 : monday_adults = 5)
  (h3 : monday_seniors = 3)
  (h4 : monday_student_groups = 2)
  (h5 : tuesday_children = 9)
  (h6 : tuesday_adults = 6)
  (h7 : tuesday_seniors = 2)
  (h8 : tuesday_student_groups = 1)
  (h9 : monday_child_price = 3)
  (h10 : monday_adult_price = 4)
  (h11 : monday_senior_price = 3)
  (h12 : monday_student_group_price = 25)
  (h13 : tuesday_child_price = 4)
  (h14 : tuesday_adult_price = 5)
  (h15 : tuesday_senior_price = 3)
  (h16 : tuesday_student_group_price = 30)
  (h17 : monday_promo_child_price = 2)
  (h18 : monday_promo_limit = 3)
  (h19 : tuesday_discount_rate = 0.1)
  (h20 : student_group_size = 10) :
  -- The total revenue for both days
  let monday_revenue := 
    (min monday_promo_limit monday_children * monday_promo_child_price + 
     (monday_children - min monday_promo_limit monday_children) * monday_child_price) +
    (monday_adults * monday_adult_price) +
    (monday_seniors * monday_senior_price) +
    (monday_student_groups * monday_student_group_price)
  
  let tuesday_subtotal := 
    (tuesday_children * tuesday_child_price) +
    (tuesday_adults * tuesday_adult_price) +
    (tuesday_seniors * tuesday_senior_price)
  
  let tuesday_revenue :=
    (tuesday_subtotal * (1 - tuesday_discount_rate)) +
    (tuesday_student_groups * tuesday_student_group_price)
  
  let total_revenue := monday_revenue + tuesday_revenue
  
  total_revenue = 191.8 := by
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zoo_revenue_calculation_l1091_109174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_residue_diversity_l1091_109102

theorem residue_diversity (p : ℕ) (hp : Nat.Prime p) (a : Fin p → ℤ) :
  ∃ k : ℤ, Finset.card (Finset.image (λ i : Fin p ↦ (a i + i.val * k) % p) Finset.univ) ≥ p / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_residue_diversity_l1091_109102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1091_109172

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.sin x)^2 - Real.cos (2*x + Real.pi/2)

theorem f_properties :
  (f (Real.pi/8) = 1) ∧
  (∀ y : ℝ, f (y + Real.pi) = f y) ∧
  (∀ k : ℤ, ∀ x : ℝ, 
    (x ≥ k * Real.pi - Real.pi/8 ∧ x ≤ k * Real.pi + 3*Real.pi/8) →
    (∀ y : ℝ, y ∈ Set.Icc (k * Real.pi - Real.pi/8) (k * Real.pi + 3*Real.pi/8) →
      x ≤ y → f x ≤ f y)) :=
by sorry

#check f_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1091_109172

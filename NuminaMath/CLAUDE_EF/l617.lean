import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_not_in_first_second_fourth_quadrant_l617_61783

noncomputable section

open Real

theorem angle_not_in_first_second_fourth_quadrant (θ : ℝ) 
  (h : 1 + (sin θ) * (sqrt ((sin θ) ^ 2)) + (cos θ) * (sqrt ((cos θ) ^ 2)) = 0) :
  ¬ (0 < θ ∧ θ < π / 2) ∧ 
  ¬ (π / 2 < θ ∧ θ < π) ∧ 
  ¬ (3 * π / 2 < θ ∧ θ < 2 * π) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_not_in_first_second_fourth_quadrant_l617_61783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_f_l617_61717

noncomputable def f (x : ℝ) : ℝ := (Real.cos (Real.arctan (1 / Real.tan 3)) * (Real.cos (14 * x))^2) / (28 * Real.sin (28 * x))

theorem derivative_of_f (x : ℝ) :
  deriv f x = -Real.cos (Real.arctan (1 / Real.tan 3)) / (4 * (Real.sin (14 * x))^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_f_l617_61717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_double_box_13_l617_61747

def box (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id

theorem double_box_13 : box (box 13) = 24 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_double_box_13_l617_61747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_symmetric_l617_61738

/-- The function f(x) = ln(√(1+x²) - x) + 2 -/
noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (1 + x^2) - x) + 2

/-- Theorem: For any real number a, f(a) + f(-a) = 4 -/
theorem f_sum_symmetric (a : ℝ) : f a + f (-a) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_symmetric_l617_61738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mutually_exclusive_events_l617_61714

-- Define the types of pens
inductive PenGrade where
  | FirstGrade
  | SecondGrade
deriving DecidableEq

-- Define the box of pens
def box : Finset PenGrade := sorry

-- Define the event of selecting 1 first-grade and 2 second-grade pens
def event_1first_2second (selection : Finset PenGrade) : Prop :=
  selection.card = 3 ∧
  (selection.filter (· = PenGrade.FirstGrade)).card = 1 ∧
  (selection.filter (· = PenGrade.SecondGrade)).card = 2

-- Define event A: Selecting at least 2 first-grade pens
def event_A (selection : Finset PenGrade) : Prop :=
  selection.card = 3 ∧
  (selection.filter (· = PenGrade.FirstGrade)).card ≥ 2

-- Define event B: Selecting at most 1 second-grade pen
def event_B (selection : Finset PenGrade) : Prop :=
  selection.card = 3 ∧
  (selection.filter (· = PenGrade.SecondGrade)).card ≤ 1

-- Define event D: Selecting no second-grade pen
def event_D (selection : Finset PenGrade) : Prop :=
  selection.card = 3 ∧
  (selection.filter (· = PenGrade.SecondGrade)).card = 0

-- Theorem stating that events A, B, and D are mutually exclusive with the given event
theorem mutually_exclusive_events
  (selection : Finset PenGrade)
  (h_box : box.card = 5)
  (h_first : (box.filter (· = PenGrade.FirstGrade)).card = 3)
  (h_second : (box.filter (· = PenGrade.SecondGrade)).card = 2)
  (h_selection : selection ⊆ box) :
  (event_1first_2second selection → ¬event_A selection) ∧
  (event_1first_2second selection → ¬event_B selection) ∧
  (event_1first_2second selection → ¬event_D selection) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mutually_exclusive_events_l617_61714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_is_20_over_T_l617_61780

/-- Represents the ratio of increase in average marks to original average marks when one pupil's marks are increased by 20 in a class of 40 pupils -/
noncomputable def ratio_increase_to_original (T : ℝ) : ℝ :=
  let original_average := T / 40
  let new_average := (T + 20) / 40
  let increase := new_average - original_average
  increase / original_average

/-- Theorem stating that the ratio of increase in average marks to original average marks is 20/T -/
theorem ratio_is_20_over_T (T : ℝ) (h : T > 0) : ratio_increase_to_original T = 20 / T := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_is_20_over_T_l617_61780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flea_strategies_modulo_l617_61720

def flea_strategies (p : ℕ) : ℤ :=
  (List.range (p - 1)).foldl
    (fun acc _ => acc + (if acc % 3 = 0 then 1 else if acc % 3 = 1 then 1 else -1))
    0

theorem flea_strategies_modulo (p : ℕ) (h : Nat.Prime p) :
  (flea_strategies p).toNat % p =
    if p = 2 ∨ p = 3 then
      0
    else if p % 3 = 1 then
      1
    else
      p - 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flea_strategies_modulo_l617_61720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_sum_inequality_l617_61757

open Real

theorem cyclic_sum_inequality (a b c d e : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0) 
  (h_prod : a * b * c * d * e = 1) : 
  (a + a*b*c) / (1 + a*b + a*b*c*d) +
  (b + b*c*d) / (1 + b*c + b*c*d*e) +
  (c + c*d*e) / (1 + c*d + c*d*e*a) +
  (d + d*e*a) / (1 + d*e + d*e*a*b) +
  (e + e*a*b) / (1 + e*a + e*a*b*c) ≥ 10/3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_sum_inequality_l617_61757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_trapezoid_area_l617_61701

/-- An isosceles trapezoid circumscribed around a circle -/
structure CircumscribedIsoscelesTrapezoid where
  /-- The length of the longer base of the trapezoid -/
  long_base : ℝ
  /-- One of the base angles of the trapezoid in radians -/
  base_angle : ℝ
  /-- Assumption that the trapezoid is isosceles -/
  isosceles : True
  /-- Assumption that the trapezoid is circumscribed around a circle -/
  circumscribed : True
  /-- The longer base is positive -/
  long_base_pos : long_base > 0
  /-- The base angle is between 0 and π/2 -/
  base_angle_range : 0 < base_angle ∧ base_angle < Real.pi / 2

/-- The area of the circumscribed isosceles trapezoid -/
noncomputable def trapezoid_area (t : CircumscribedIsoscelesTrapezoid) : ℝ :=
  144 * Real.sqrt 2 - 72

/-- Theorem stating the area of the specific trapezoid -/
theorem specific_trapezoid_area :
  ∀ t : CircumscribedIsoscelesTrapezoid,
    t.long_base = 24 ∧ t.base_angle = Real.pi / 4 →
    trapezoid_area t = 144 * Real.sqrt 2 - 72 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_trapezoid_area_l617_61701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subtracted_value_l617_61710

theorem subtracted_value (numbers : Fin 5 → ℝ) (X : ℝ)
  (avg_original : (Finset.sum Finset.univ (λ i => numbers i)) / 5 = 5)
  (avg_new : (Finset.sum Finset.univ (λ i => numbers i) - 4 * X) / 5 = 3.4) : X = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subtracted_value_l617_61710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equilateral_foci_eccentricity_l617_61796

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - (e.b / e.a)^2)

/-- The focal distance of an ellipse -/
noncomputable def focal_distance (e : Ellipse) : ℝ :=
  Real.sqrt (e.a^2 - e.b^2)

/-- Theorem: If the triangle formed by the upper vertex and foci of an ellipse
    is equilateral, then the eccentricity of the ellipse is 1/2 -/
theorem ellipse_equilateral_foci_eccentricity (e : Ellipse) :
  focal_distance e = e.b / Real.sqrt 3 →
  eccentricity e = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equilateral_foci_eccentricity_l617_61796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_geometric_sequences_l617_61743

-- Define the parabola
def Γ (x y : ℝ) : Prop := y^2 = 8*x

-- Define the lines
def l (k m_i x y : ℝ) : Prop := y = k*x + m_i

-- Define the distance from point T to line l_i
noncomputable def d (t m_i : ℝ) : ℝ := |t + m_i| / Real.sqrt 2

-- Define the area of triangle TA_i B_i
noncomputable def S (m_i : ℝ) : ℝ := 4 * Real.sqrt (2 - m_i)

-- Define the theorem
theorem existence_of_geometric_sequences (t : ℝ) (m_1 m_2 m_3 : ℝ) : 
  m_1 < m_2 → m_2 < m_3 → m_3 < -t →
  (∃ (r : ℝ), S m_2 = r * S m_1 ∧ S m_3 = r * S m_2) →
  (∃ (q : ℝ), d t m_2 = q * d t m_1 ∧ d t m_3 = q * d t m_2) →
  t = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_geometric_sequences_l617_61743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l617_61754

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the theorem
theorem triangle_theorem (t : Triangle) :
  -- Condition: sin B(tan A + tan C) = tan A tan C
  (Real.sin t.B) * (Real.tan t.A + Real.tan t.C) = Real.tan t.A * Real.tan t.C →
  -- Part I: a, b, c form a geometric sequence
  t.b^2 = t.a * t.c ∧
  -- Part II: If a = 1 and c = 2, then area = √7/4
  (t.a = 1 ∧ t.c = 2 →
    1/2 * t.a * t.c * Real.sin t.B = Real.sqrt 7 / 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l617_61754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_table_height_is_30_l617_61789

/-- Represents the length of a wood block in inches -/
def block_length : ℝ := sorry

/-- Represents the width of a wood block in inches -/
def block_width : ℝ := sorry

/-- Represents the height of the table in inches -/
def table_height : ℝ := sorry

/-- The first configuration spans 45 inches -/
axiom first_config : block_length + table_height = 45

/-- The second configuration spans 40 inches -/
axiom second_config : table_height + block_width = 40

/-- The length of a block is 5 inches more than its width -/
axiom block_dimension : block_length = block_width + 5

theorem table_height_is_30 : table_height = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_table_height_is_30_l617_61789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_translation_for_symmetry_l617_61744

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x) ^ 2 - 1 / 2

theorem min_translation_for_symmetry 
  (ω : ℝ) (h_ω_pos : ω > 0) 
  (h_period : ∀ x, f ω (x + π / 2) = f ω x) 
  (a : ℝ) (h_a_pos : a > 0)
  (h_symmetry : ∀ x, f ω (x + a) = -f ω (-x - a)) :
  a ≥ π / 8 ∧ ∃ a₀, a₀ = π / 8 ∧ 
    (∀ x, f ω (x + a₀) = -f ω (-x - a₀)) := by
  sorry

#check min_translation_for_symmetry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_translation_for_symmetry_l617_61744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_solutions_l617_61778

theorem average_of_solutions :
  (∃ x : ℝ, Real.sqrt (3 * x^2 + 2) = Real.sqrt 50) →
  (∃ a b : ℝ, {x : ℝ | Real.sqrt (3 * x^2 + 2) = Real.sqrt 50} = {a, b} ∧ (a + b) / 2 = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_solutions_l617_61778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_fraction_from_cubic_sum_l617_61770

theorem rational_fraction_from_cubic_sum (S : Set ℝ) (h_nonempty : S.Nonempty) 
  (h_positive : ∀ x, x ∈ S → x > 0) 
  (h_rational_sum : ∀ a b c, a ∈ S → b ∈ S → c ∈ S → ∃ q : ℚ, (a^3 + b^3 + c^3 - 3*a*b*c : ℝ) = ↑q) :
  ∀ a b, a ∈ S → b ∈ S → ∃ q : ℚ, ((a - b) / (a + b) : ℝ) = ↑q :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_fraction_from_cubic_sum_l617_61770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_b_value_l617_61758

/-- A cubic polynomial P(x) = x^3 + ax^2 + bx + c -/
structure CubicPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The mean of the zeros of a cubic polynomial -/
noncomputable def meanOfZeros (p : CubicPolynomial) : ℝ := -p.a / 3

/-- The product of the zeros of a cubic polynomial -/
def productOfZeros (p : CubicPolynomial) : ℝ := -p.c

/-- The sum of the coefficients of a cubic polynomial -/
def sumOfCoefficients (p : CubicPolynomial) : ℝ := 1 + p.a + p.b + p.c

/-- The y-intercept of a cubic polynomial -/
def yIntercept (p : CubicPolynomial) : ℝ := p.c

theorem cubic_polynomial_b_value (p : CubicPolynomial) :
  meanOfZeros p = productOfZeros p ∧
  meanOfZeros p = sumOfCoefficients p ∧
  yIntercept p = 5 →
  p.b = -26 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_b_value_l617_61758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_k_for_minimum_area_l617_61764

/-- Given three points in a 2D plane, calculate the area of the triangle formed by these points. -/
noncomputable def triangleArea (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  (1/2) * abs ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))

/-- Check if a given value k results in a minimum non-zero area triangle. -/
def isMinimumNonZeroArea (k : ℤ) : Prop :=
  let area := triangleArea 2 9 14 18 6 (k : ℝ)
  area > 0 ∧ ∀ (m : ℤ), triangleArea 2 9 14 18 6 (m : ℝ) > 0 → area ≤ triangleArea 2 9 14 18 6 (m : ℝ)

/-- The main theorem stating that the sum of k values resulting in minimum non-zero area is 31. -/
theorem sum_of_k_for_minimum_area : 
  ∃ (k1 k2 : ℤ), k1 ≠ k2 ∧ isMinimumNonZeroArea k1 ∧ isMinimumNonZeroArea k2 ∧ k1 + k2 = 31 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_k_for_minimum_area_l617_61764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l617_61722

noncomputable def g (x : ℝ) : ℝ := (Real.sin x ^ 3 + 7 * Real.sin x ^ 2 + 2 * Real.sin x + 3 * Real.cos x ^ 2 - 10) / (Real.sin x - 1)

theorem g_range :
  ∀ x : ℝ, Real.sin x ≠ 1 → 3 ≤ g x ∧ g x < 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l617_61722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersection_y_axis_l617_61708

/-- A line passing through two points (x₁, y₁) and (x₂, y₂) -/
structure Line where
  x₁ : ℚ
  y₁ : ℚ
  x₂ : ℚ
  y₂ : ℚ

/-- The y-intercept of a line -/
def y_intercept (l : Line) : ℚ × ℚ :=
  (0, l.y₁ + (l.y₂ - l.y₁) / (l.x₂ - l.x₁) * (-l.x₁))

/-- The theorem stating that the line passing through (2, 8) and (4, 14) 
    intersects the y-axis at (0, 2) -/
theorem line_intersection_y_axis :
  y_intercept (Line.mk 2 8 4 14) = (0, 2) := by
  sorry

#eval y_intercept (Line.mk 2 8 4 14)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersection_y_axis_l617_61708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_H_in_C4H8O2_approx_l617_61768

/-- The mass percentage of hydrogen in C4H8O2 -/
noncomputable def mass_percentage_H_in_C4H8O2 : ℝ :=
  let molar_mass_C : ℝ := 12.01
  let molar_mass_H : ℝ := 1.008
  let molar_mass_O : ℝ := 16.00
  let molar_mass_C4H8O2 : ℝ := 4 * molar_mass_C + 8 * molar_mass_H + 2 * molar_mass_O
  let mass_H_in_C4H8O2 : ℝ := 8 * molar_mass_H
  (mass_H_in_C4H8O2 / molar_mass_C4H8O2) * 100

/-- Theorem stating that the mass percentage of H in C4H8O2 is approximately 9.15% -/
theorem mass_percentage_H_in_C4H8O2_approx :
  ‖mass_percentage_H_in_C4H8O2 - 9.15‖ < 0.01 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_H_in_C4H8O2_approx_l617_61768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_zero_points_and_lower_bound_l617_61752

/-- The function f(x) = ln x + a/x -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a / x

theorem f_zero_points_and_lower_bound (a : ℝ) :
  (a > 0) →
  (∃ x > 0, f a x = 0) ↔ (0 < a ∧ a ≤ Real.exp (-1)) ∧
  (a ≥ 2 / Real.exp 1 → ∀ x > 0, f a x > Real.exp (-x)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_zero_points_and_lower_bound_l617_61752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_definition_correct_l617_61713

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 - 3*x + 4
  else x^2 + 3*x + 4

def f_is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

theorem f_definition_correct (h : f_is_even_function f) :
  ∀ x, f x = if x ≥ 0 then x^2 - 3*x + 4 else x^2 + 3*x + 4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_definition_correct_l617_61713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_non_monotonic_cubic_l617_61781

theorem non_monotonic_cubic (k : ℝ) : 
  (∃ x y, x ∈ Set.Ioo (k - 1) (k + 1) ∧ y ∈ Set.Ioo (k - 1) (k + 1) ∧
    ((x < y ∧ (x^3 - 12*x > y^3 - 12*y)) ∨ 
    (x < y ∧ (x^3 - 12*x < y^3 - 12*y)))) ↔ 
  (-3 < k ∧ k < -1) ∨ (1 < k ∧ k < 3) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_non_monotonic_cubic_l617_61781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l617_61705

noncomputable section

variable (x x₁ x₂ a : ℝ)

def f (a : ℝ) (x : ℝ) : ℝ := a * x - Real.log x
def g (x : ℝ) : ℝ := (Real.log x) / x
def h (a : ℝ) (x : ℝ) : ℝ := f a x - x^2 + 1

theorem problem_solution :
  (∀ x > 0, Antitone (h a) ↔ a ≤ 2 * Real.sqrt 2) ∧
  (∀ x₁ ∈ Set.Icc (0 : ℝ) (Real.exp 1), ∀ x₂ > 0, f 1 x₁ > g x₂ + 1/2) ∧
  (∀ x ∈ Set.Ioc (0 : ℝ) (Real.exp 1), f a x ≥ 3 ↔ a ≥ (Real.exp 1)^2) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l617_61705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equation_solution_l617_61777

theorem complex_equation_solution (z : ℂ) : (1 + 3*Complex.I)*z = 10*Complex.I → z = 3 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equation_solution_l617_61777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_range_l617_61749

theorem dot_product_range (a b : ℝ × ℝ) : 
  ‖a‖ = 5 → ‖b‖ = 12 → -60 ≤ a.1 * b.1 + a.2 * b.2 ∧ a.1 * b.1 + a.2 * b.2 ≤ 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_range_l617_61749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_3_power_6_l617_61771

-- Define the functions f and g
noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

-- State the conditions
axiom f_g_composition (x : ℝ) (h : x ≥ 1) : f (g x) = x^4
axiom g_f_composition (x : ℝ) (h : x ≥ 1) : g (f x) = x^6
axiom g_81 : g 81 = 81

-- State the theorem to be proved
theorem g_3_power_6 : (g 3)^6 = 81 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_3_power_6_l617_61771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_intersection_points_l617_61786

/-- Represents a line segment on a plane -/
structure Segment where
  start : ℝ × ℝ
  end_ : ℝ × ℝ

/-- Represents an intersection point between two segments -/
structure IntersectionPoint where
  point : ℝ × ℝ
  ratio : ℚ

/-- The configuration of segments and their intersections -/
structure SegmentConfiguration where
  segments : List Segment
  intersections : List IntersectionPoint

/-- Checks if a configuration is valid according to the problem constraints -/
def is_valid_configuration (config : SegmentConfiguration) : Prop :=
  config.segments.length = 10 ∧
  ∀ i ∈ config.intersections, i.ratio = 3/4

/-- The theorem stating the maximum number of intersection points -/
theorem max_intersection_points (config : SegmentConfiguration) 
  (h : is_valid_configuration config) : 
  config.intersections.length ≤ 10 := by
  sorry

#check max_intersection_points

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_intersection_points_l617_61786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_circle_area_final_result_correct_l617_61709

-- Define the unit square
def unit_square : Set (ℝ × ℝ) := {p | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1}

-- Define the circle Γ
def circle_set (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p | (p.1 - center.1)^2 + (p.2 - center.2)^2 ≤ radius^2}

-- Define the angle between three points
noncomputable def angle (A B C : ℝ × ℝ) : ℝ := sorry

-- Define the condition for points outside Γ
def outside_condition (Γ : Set (ℝ × ℝ)) : Prop :=
  ∀ P, P ∉ Γ → 
    min (angle P (0,0) (1,0)) (min (angle P (1,0) (1,1)) (min (angle P (1,1) (0,1)) (angle P (0,1) (0,0)))) ≤ 60

-- State the theorem
theorem min_circle_area (Γ : Set (ℝ × ℝ)) (h : outside_condition Γ) :
  ∃ center radius, Γ = circle_set center radius ∧ π * radius^2 ≥ π / 6 := by
  sorry

-- Define the final result
def final_result : ℕ := 106

-- Prove that the final result is correct
theorem final_result_correct : final_result = 106 := by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_circle_area_final_result_correct_l617_61709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_difference_zero_l617_61737

/-- The line equation y - 2x + 5 = 0 -/
def line (x y : ℝ) : Prop := y - 2*x + 5 = 0

/-- The parabola equation y² = x + 1 -/
def parabola (x y : ℝ) : Prop := y^2 = x + 1

/-- Point Q -/
def Q : ℝ × ℝ := (2, 1)

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem intersection_distance_difference_zero :
  ∃ (C D : ℝ × ℝ),
    line C.1 C.2 ∧ line D.1 D.2 ∧
    parabola C.1 C.2 ∧ parabola D.1 D.2 ∧
    C ≠ D ∧
    |distance C Q - distance D Q| = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_difference_zero_l617_61737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_certain_number_approximation_l617_61736

/-- If 102 times x equals a certain number y, and 10 minus x approximately equals 0.9992159937279498,
    then y is approximately 918.0800495985541. -/
theorem certain_number_approximation (x y : ℝ) 
  (h1 : 102 * x = y) 
  (h2 : |10 - x - 0.9992159937279498| < 1e-10) : 
  |y - 918.0800495985541| < 1e-10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_certain_number_approximation_l617_61736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_intersection_distance_l617_61790

/-- The distance between adjacent intersection points of y = a and y = tan x is π -/
theorem tan_intersection_distance (a : ℝ) : ∃ (x₁ x₂ : ℝ), 
  x₁ < x₂ ∧ 
  Real.tan x₁ = a ∧ 
  Real.tan x₂ = a ∧ 
  x₂ - x₁ = π ∧
  ∀ x, x₁ < x ∧ x < x₂ → Real.tan x ≠ a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_intersection_distance_l617_61790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_M_in_range_l617_61766

/-- Definition of M(a) -/
def M (a : ℕ+) : ℕ :=
  (Finset.filter (fun b => (a + b : ℕ) ∣ (a * b : ℕ)) (Finset.range 2014)).card

/-- Theorem statement -/
theorem max_M_in_range :
  ∃! a : ℕ+, 1 ≤ a ∧ a ≤ 2013 ∧
  ∀ k : ℕ+, 1 ≤ k ∧ k ≤ 2013 → M k ≤ M a ∧
  a = 1680 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_M_in_range_l617_61766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_propositions_false_l617_61751

theorem all_propositions_false :
  -- Proposition 1
  (∃ (a b : ℝ), a < b) ∧
  -- Proposition 2
  (let z : ℂ := Complex.I - 1; z.re < 0 ∧ z.im > 0) ∧
  -- Proposition 3
  (∃ (x : ℝ), x ≠ 1 ∧ x ≠ -1 ∧ (x^2 - 1 : ℂ) + (x^2 + 3*x + 2 : ℂ) * Complex.I = Complex.I * (x^2 + 3*x + 2)) ∧
  -- Proposition 4
  (∃ (z₁ z₂ z₃ : ℂ), (z₁ - z₂)^2 + (z₂ - z₃)^2 = 0 ∧ (z₁ ≠ z₂ ∨ z₂ ≠ z₃)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_propositions_false_l617_61751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_point_theorem_l617_61797

/-- Two men walking towards each other -/
structure WalkingMen where
  total_distance : ℚ
  r_speed : ℚ
  s_initial_speed : ℚ
  s_speed_increase : ℚ
  meeting_time : ℕ

/-- Calculate the distance walked by the man from R -/
def distance_from_r (w : WalkingMen) : ℚ :=
  w.r_speed * (w.meeting_time : ℚ)

/-- Calculate the distance walked by the man from S -/
def distance_from_s (w : WalkingMen) : ℚ :=
  (w.meeting_time : ℚ) / 2 * (2 * w.s_initial_speed + (w.meeting_time - 1 : ℚ) * w.s_speed_increase)

/-- The difference in distance from R and S to the meeting point -/
def meeting_point_difference (w : WalkingMen) : ℚ :=
  distance_from_s w - distance_from_r w

/-- Theorem stating that the men meet 4 miles closer to R than S -/
theorem meeting_point_theorem (w : WalkingMen) 
  (h1 : w.total_distance = 76)
  (h2 : w.r_speed = 9/2)
  (h3 : w.s_initial_speed = 13/4)
  (h4 : w.s_speed_increase = 1/2)
  (h5 : w.meeting_time = 8)
  (h6 : distance_from_r w + distance_from_s w = w.total_distance) :
  meeting_point_difference w = 4 := by
  sorry

#eval meeting_point_difference {
  total_distance := 76,
  r_speed := 9/2,
  s_initial_speed := 13/4,
  s_speed_increase := 1/2,
  meeting_time := 8
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_point_theorem_l617_61797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_circles_l617_61741

noncomputable section

-- Define the centers and radii of the given circles
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry
def C : ℝ × ℝ := sorry

noncomputable def a : ℝ := 1/2
noncomputable def b : ℝ := 3/2
noncomputable def c : ℝ := 3/2

-- Define the centers and radii of the tangent circles
def S : ℝ × ℝ := sorry
def L : ℝ × ℝ := sorry

noncomputable def r : ℝ := (2 * Real.sqrt 7 - 5) / 2
noncomputable def R : ℝ := (2 * Real.sqrt 7 + 5) / 2

-- Distances between centers
axiom AB_eq : dist A B = 2
axiom AC_eq : dist A C = 2
axiom BC_eq : dist B C = 3

-- Tangency conditions
axiom S_tangent : (dist S A = a + r) ∧ (dist S B = b + r) ∧ (dist S C = c + r)
axiom L_tangent : (dist L A = a + R) ∧ (dist L B = b + R) ∧ (dist L C = c + R)

-- Theorem to prove
theorem area_between_circles : 
  π * R^2 - π * r^2 = 10 * Real.sqrt 7 * π := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_circles_l617_61741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_points_relationship_l617_61799

theorem linear_function_points_relationship (k a b : ℝ) :
  (∃ x : ℝ, (-4, a) = ((k^2 + 1) * x + 1, x)) →
  (∃ x : ℝ, (2, b) = ((k^2 + 1) * x + 1, x)) →
  a < b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_points_relationship_l617_61799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_actual_plot_area_l617_61750

/-- Represents the scale factor of the plot -/
def scale : ℝ := 3

/-- Represents the bottom length of the trapezoid in cm -/
def bottom : ℝ := 12

/-- Represents the top length of the trapezoid in cm -/
def top : ℝ := 18

/-- Represents the height of the trapezoid in cm -/
def trapezoid_height : ℝ := 10

/-- Represents the number of acres in a square mile -/
def acres_per_square_mile : ℝ := 640

/-- Calculates the area of the actual plot in acres -/
theorem actual_plot_area :
  ((bottom + top) * trapezoid_height / 2) * scale^2 * acres_per_square_mile = 864000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_actual_plot_area_l617_61750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nuts_per_meter_approx_l617_61729

/-- Represents a tree with its height and nut counts -/
structure TreeData where
  height : ℝ
  acorns : ℕ
  walnuts : ℕ

/-- Represents the fraction of nuts collected by squirrels -/
structure Collection where
  acornFraction : ℚ
  walnutFraction : ℚ

/-- Represents the fraction of nuts stolen by the bird -/
structure Theft where
  acornFraction : ℚ
  walnutFraction : ℚ

def oakTree : TreeData := { height := 7, acorns := 12, walnuts := 20 }
def pineTree : TreeData := { height := 12, acorns := 40, walnuts := 50 }
def mapleTree : TreeData := { height := 18, acorns := 48, walnuts := 30 }

def oakCollection : Collection := { acornFraction := 3/4, walnutFraction := 1/2 }
def pineCollection : Collection := { acornFraction := 1/4, walnutFraction := 2/5 }
def mapleCollection : Collection := { acornFraction := 5/8, walnutFraction := 5/6 }

def birdTheft : Theft := { acornFraction := 1/10, walnutFraction := 3/20 }

/-- Calculates the number of nuts per meter of total tree height -/
noncomputable def nutsPerMeter (trees : List TreeData) (collections : List Collection) (theft : Theft) : ℝ :=
  sorry

theorem nuts_per_meter_approx :
  let trees := [oakTree, pineTree, mapleTree]
  let collections := [oakCollection, pineCollection, mapleCollection]
  abs (nutsPerMeter trees collections birdTheft - 2.43) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nuts_per_meter_approx_l617_61729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_line_slope_undefined_l617_61727

/-- A line in the 2D plane defined by a constant x-coordinate. -/
structure VerticalLine where
  x : ℝ

/-- The slope of a line, which can be either a real number or undefined. -/
inductive Slope
  | Real (m : ℝ)
  | Undefined

/-- Function to get the slope of a line. For vertical lines, it returns Undefined. -/
def getSlope (l : VerticalLine) : Slope := Slope.Undefined

/-- The slope of a vertical line is undefined. -/
theorem vertical_line_slope_undefined (l : VerticalLine) : getSlope l = Slope.Undefined := by
  -- The proof is trivial since getSlope is defined to always return Slope.Undefined for VerticalLine
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_line_slope_undefined_l617_61727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_line_equation_l617_61704

-- Define the centers and radii of the circles
def center1 : ℝ × ℝ := (-12, -2)
def center2 : ℝ × ℝ := (5, 11)
def radius1 : ℝ := 15
noncomputable def radius2 : ℝ := Real.sqrt 90

-- Define the equation of the line passing through the intersection points
def line_equation (x y : ℝ) : Prop := x + y = -1/2

-- Theorem statement
theorem intersection_line_equation :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    -- Points (x₁, y₁) and (x₂, y₂) lie on both circles
    ((x₁ - center1.1)^2 + (y₁ - center1.2)^2 = radius1^2) ∧
    ((x₁ - center2.1)^2 + (y₁ - center2.2)^2 = radius2^2) ∧
    ((x₂ - center1.1)^2 + (y₂ - center1.2)^2 = radius1^2) ∧
    ((x₂ - center2.1)^2 + (y₂ - center2.2)^2 = radius2^2) ∧
    -- The line passing through these points satisfies the equation
    line_equation x₁ y₁ ∧
    line_equation x₂ y₂ :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_line_equation_l617_61704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_ratio_is_nine_sixteenths_l617_61779

/-- A cone inscribed in a sphere with slant height equal to base diameter -/
structure InscribedCone where
  R : ℝ  -- Radius of the sphere
  slant_height : ℝ  -- Slant height of the cone
  base_diameter : ℝ  -- Diameter of the cone's base
  h_slant_eq_base : slant_height = base_diameter  -- Slant height equals base diameter
  h_inscribed : slant_height = R * Real.sqrt 3  -- Property of inscribed cone

/-- The ratio of the total surface area of the cone to the surface area of the sphere -/
noncomputable def surface_area_ratio (cone : InscribedCone) : ℝ :=
  let cone_surface_area := (3 * Real.pi * cone.R^2) / 2 + (3 * Real.pi * cone.R^2) / 4
  let sphere_surface_area := 4 * Real.pi * cone.R^2
  cone_surface_area / sphere_surface_area

/-- Theorem: The ratio of surface areas is 9/16 -/
theorem surface_area_ratio_is_nine_sixteenths (cone : InscribedCone) :
  surface_area_ratio cone = 9/16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_ratio_is_nine_sixteenths_l617_61779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_krakozyabr_count_l617_61716

/-- Represents the set of all krakozyabrs -/
def Krakozyabr : Type := Unit

/-- Predicate for krakozyabrs with horns -/
def has_horns : Krakozyabr → Prop := sorry

/-- Predicate for krakozyabrs with wings -/
def has_wings : Krakozyabr → Prop := sorry

/-- The total number of krakozyabrs -/
def total_krakozyabrs : ℕ := sorry

/-- Count function for finite sets -/
noncomputable def Count {α : Type} (s : Set α) : ℚ := sorry

theorem krakozyabr_count : 
  (∀ k : Krakozyabr, has_horns k ∨ has_wings k) →
  (∃ n : ℕ, n > 0 ∧ 
    Count {k : Krakozyabr | has_horns k ∧ has_wings k} = 
    (1/5 : ℚ) * Count {k : Krakozyabr | has_horns k}) →
  (∃ n : ℕ, n > 0 ∧ 
    Count {k : Krakozyabr | has_horns k ∧ has_wings k} = 
    (1/4 : ℚ) * Count {k : Krakozyabr | has_wings k}) →
  25 < total_krakozyabrs ∧ total_krakozyabrs < 35 →
  total_krakozyabrs = 32 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_krakozyabr_count_l617_61716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l617_61702

/-- The domain of the function f(x) = √(cos x - 1/2) -/
def domain_f : Set ℝ :=
  {x : ℝ | ∃ k : ℤ, 2 * k * Real.pi - Real.pi / 3 ≤ x ∧ x ≤ 2 * k * Real.pi + Real.pi / 3}

/-- The function f(x) = √(cos x - 1/2) -/
noncomputable def f (x : ℝ) : ℝ :=
  Real.sqrt (Real.cos x - 1/2)

theorem domain_of_f : {x : ℝ | ∃ y : ℝ, f x = y} = domain_f := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l617_61702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_integer_property_l617_61769

theorem least_integer_property (n m : ℕ) (hn : n > 1) :
  ∃ k : ℕ+, k.val = Nat.gcd m n * (n / (2 * Nat.gcd m n)) ∧
  (∀ (a : Fin k → ℤ), (∀ i j : Fin k, i.val < j.val → ¬ (n : ℤ) ∣ (a i - a j)) →
    ∃ p s : Fin k, p ≠ s ∧ (n : ℤ) ∣ (m : ℤ) + a p - a s) ∧
  (∀ k' : ℕ+, k'.val < k.val →
    ∃ (a : Fin k' → ℤ), (∀ i j : Fin k', i.val < j.val → ¬ (n : ℤ) ∣ (a i - a j)) ∧
      ∀ p s : Fin k', p ≠ s → ¬ (n : ℤ) ∣ (m : ℤ) + a p - a s) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_integer_property_l617_61769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_O_in_CuCO3_l617_61798

/-- The atomic mass of copper in g/mol -/
noncomputable def atomic_mass_Cu : ℝ := 63.55

/-- The atomic mass of carbon in g/mol -/
noncomputable def atomic_mass_C : ℝ := 12.01

/-- The atomic mass of oxygen in g/mol -/
noncomputable def atomic_mass_O : ℝ := 16.00

/-- The number of copper atoms in CuCO3 -/
def num_Cu : ℕ := 1

/-- The number of carbon atoms in CuCO3 -/
def num_C : ℕ := 1

/-- The number of oxygen atoms in CuCO3 -/
def num_O : ℕ := 3

/-- The molar mass of CuCO3 in g/mol -/
noncomputable def molar_mass_CuCO3 : ℝ :=
  num_Cu * atomic_mass_Cu + num_C * atomic_mass_C + num_O * atomic_mass_O

/-- The mass of oxygen in one mole of CuCO3 in g -/
noncomputable def mass_O_in_CuCO3 : ℝ := num_O * atomic_mass_O

/-- The mass percentage of oxygen in CuCO3 -/
noncomputable def mass_percentage_O : ℝ := (mass_O_in_CuCO3 / molar_mass_CuCO3) * 100

theorem mass_percentage_O_in_CuCO3 :
  abs (mass_percentage_O - 38.83) < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_O_in_CuCO3_l617_61798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passes_jogger_in_24_seconds_l617_61784

/-- The time (in seconds) it takes for a train to pass a jogger -/
noncomputable def trainPassingTime (joggerSpeed trainSpeed : ℝ) (trainLength joggerLead : ℝ) : ℝ :=
  let joggerSpeedMPS := joggerSpeed * 1000 / 3600
  let trainSpeedMPS := trainSpeed * 1000 / 3600
  let relativeSpeed := trainSpeedMPS - joggerSpeedMPS
  let totalDistance := joggerLead + trainLength
  totalDistance / relativeSpeed

/-- Theorem stating that the train will pass the jogger in 24 seconds -/
theorem train_passes_jogger_in_24_seconds :
  trainPassingTime 9 45 120 120 = 24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passes_jogger_in_24_seconds_l617_61784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stone_distribution_l617_61732

/-- A move operation on circular containers -/
def move (n : ℕ) (containers : Vector ℕ (n + 1)) (A B : Fin (n + 1)) : Vector ℕ (n + 1) := sorry

/-- Predicate to check if the distribution is valid -/
def valid_distribution (n : ℕ) (containers : Vector ℕ (n + 1)) (full : Fin (n + 1)) : Prop :=
  ∀ i : Fin (n + 1), i ≠ full → containers.get i = 1

/-- The main theorem -/
theorem stone_distribution (n : ℕ) :
  (∃ (initial : Vector ℕ (n + 1)) (full : Fin (n + 1)),
    initial.get full = n ∧
    (∀ i : Fin (n + 1), i ≠ full → initial.get i = 0) ∧
    (∃ (final : Vector ℕ (n + 1)),
      (∃ (moves : List (Fin (n + 1) × Fin (n + 1))),
        final = moves.foldl (λ acc m ↦ move n acc m.1 m.2) initial) ∧
      valid_distribution n final full)) ↔
  Even n :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stone_distribution_l617_61732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_m_l617_61755

-- Define the sets A and B
def A (m : ℝ) : Set ℝ := {1, m - 2}
def B : Set ℝ := sorry -- We don't know the exact contents of B, so we use 'sorry'

-- State the theorem
theorem find_m (m : ℝ) : A m ∩ B = {2} → m = 4 := by
  intro h
  -- The proof steps would go here
  sorry

#check find_m

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_m_l617_61755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_is_13_l617_61740

def S : Set ℂ := {z | ∃ x y : ℝ, z = x + y * Complex.I ∧ 1/2 ≤ x ∧ x ≤ Real.sqrt 2 / 2}

theorem smallest_m_is_13 :
  ∃ m : ℕ+, (∀ n : ℕ+, n ≥ m → ∃ z ∈ S, z^(n : ℕ) = 1) ∧
  (∀ m' : ℕ+, m' < m → ∃ n : ℕ+, n ≥ m' ∧ ∀ z ∈ S, z^(n : ℕ) ≠ 1) ∧
  m = 13 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_is_13_l617_61740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_traditionalist_progressive_ratio_is_one_fourth_l617_61728

/-- Represents a country with provinces, traditionalists, and progressives -/
structure Country where
  num_provinces : ℕ
  total_population : ℝ
  traditionalists_per_province : ℝ
  fraction_traditionalist : ℝ

/-- The ratio of traditionalists in one province to total progressives -/
noncomputable def traditionalist_progressive_ratio (c : Country) : ℝ :=
  c.traditionalists_per_province / (c.total_population * (1 - c.fraction_traditionalist))

/-- Theorem stating the ratio of traditionalists in one province to total progressives -/
theorem traditionalist_progressive_ratio_is_one_fourth
  (c : Country)
  (h1 : c.num_provinces = 6)
  (h2 : c.fraction_traditionalist = 0.6)
  (h3 : c.traditionalists_per_province * c.num_provinces = c.total_population * c.fraction_traditionalist) :
  traditionalist_progressive_ratio c = 1/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_traditionalist_progressive_ratio_is_one_fourth_l617_61728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quartic_equation_solutions_l617_61759

theorem quartic_equation_solutions :
  {z : ℂ | z^4 - 6*z^2 + 8 = 0} = {-2, -Complex.I * Real.sqrt 2, Complex.I * Real.sqrt 2, 2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quartic_equation_solutions_l617_61759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monic_quadratic_sum_of_minima_l617_61735

noncomputable section

/-- Monic quadratic polynomial -/
def MonicQuadratic (a b : ℝ) : ℝ → ℝ := fun x ↦ x^2 + a*x + b

/-- The zeros of a polynomial -/
def HasZeros (p : ℝ → ℝ) (zeros : List ℝ) : Prop :=
  ∀ z ∈ zeros, p z = 0

/-- The minimum value of a quadratic function -/
def QuadraticMinimum (a b c : ℝ) : ℝ := c - b^2 / (4*a)

theorem monic_quadratic_sum_of_minima :
  ∀ b c p q : ℝ,
  let P := MonicQuadratic b c
  let Q := MonicQuadratic p q
  HasZeros (fun x ↦ P (Q x)) [-7, -5, -3, -1] →
  HasZeros (fun x ↦ Q (P x)) [-14, -12, -10, -8] →
  QuadraticMinimum 1 b c + QuadraticMinimum 1 p q = -180 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monic_quadratic_sum_of_minima_l617_61735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_asymptote_of_f_l617_61715

/-- The function f(x) = (5x^2 - 9) / (3x^2 + 5x + 2) -/
noncomputable def f (x : ℝ) : ℝ := (5 * x^2 - 9) / (3 * x^2 + 5 * x + 2)

/-- The horizontal asymptote of f(x) is y = 5/3 -/
theorem horizontal_asymptote_of_f :
  ∀ ε > 0, ∃ N : ℝ, ∀ x : ℝ, x > N → |f x - 5/3| < ε :=
by
  sorry

#check horizontal_asymptote_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_asymptote_of_f_l617_61715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wheat_in_granary_l617_61724

/-- Calculates the approximate amount of wheat in a batch of rice given the total amount,
    sample size, and number of wheat grains in the sample. -/
noncomputable def wheat_amount (total_stones : ℝ) (sample_size : ℕ) (wheat_in_sample : ℕ) : ℝ :=
  total_stones * (wheat_in_sample : ℝ) / (sample_size : ℝ)

/-- Rounds a real number to one decimal place -/
noncomputable def round_to_one_decimal (x : ℝ) : ℝ :=
  ⌊x * 10 + 0.5⌋ / 10

theorem wheat_in_granary (total_stones : ℝ) (sample_size : ℕ) (wheat_in_sample : ℕ)
    (h1 : total_stones = 1534)
    (h2 : sample_size = 254)
    (h3 : wheat_in_sample = 28) :
    round_to_one_decimal (wheat_amount total_stones sample_size wheat_in_sample) = 169.1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wheat_in_granary_l617_61724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_equals_three_l617_61739

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x

theorem a_equals_three 
  (a : ℝ) 
  (h1 : a > 0) 
  (h2 : a ≠ 1) 
  (h3 : ∀ x y, x < y → f a x < f a y)
  (h4 : (f a 1 - f a (-1)) / (f a 2 - f a (-2)) = 3/10) : 
  a = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_equals_three_l617_61739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l617_61753

noncomputable def f (t : ℝ) : ℝ := (t^2 + 2*t) / (t^2 + 2)

theorem range_of_f :
  ∀ y : ℝ, (∃ t : ℝ, f t = y) ↔ -1 ≤ y ∧ y ≤ 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l617_61753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_midpoint_and_length_l617_61775

noncomputable section

-- Define the line l
def line_l (t : ℝ) : ℝ × ℝ := (2 + (3/5) * t, (4/5) * t)

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 2*x

-- Define the intersection points A and B
def intersections (A B : ℝ × ℝ) : Prop :=
  ∃ (t1 t2 : ℝ), 
    line_l t1 = A ∧ 
    line_l t2 = B ∧ 
    parabola A.1 A.2 ∧ 
    parabola B.1 B.2 ∧
    t1 ≠ t2

-- Define the midpoint M
def midpoint_of (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

theorem intersection_midpoint_and_length 
  (A B : ℝ × ℝ) 
  (h_intersect : intersections A B) : 
  midpoint_of A B = (41/16, 3/4) ∧ 
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = (5/8) * Real.sqrt 73 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_midpoint_and_length_l617_61775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_sum_is_negative_two_l617_61746

/-- The numeric value assigned to a letter based on its position in the alphabet -/
def letter_value (n : ℕ) : ℤ :=
  match n % 6 with
  | 0 => 2
  | 1 => 1
  | 2 => 0
  | 3 => -1
  | 4 => -2
  | 5 => -1
  | _ => 0  -- This case should never occur due to the modulo operation

/-- The position of a letter in the alphabet (1-based index) -/
def letter_position (c : Char) : ℕ :=
  match c with
  | 'A' => 1
  | 'E' => 5
  | 'L' => 12
  | 'P' => 16
  | 'R' => 18
  | _ => 0  -- For any other character, return 0

/-- The sum of numeric values for the letters in "PARALLEL" -/
def parallel_sum : ℤ :=
  (List.map (λ c => letter_value (letter_position c)) ['P', 'A', 'R', 'A', 'L', 'L', 'E']).sum

theorem parallel_sum_is_negative_two : parallel_sum = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_sum_is_negative_two_l617_61746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_valid_configurations_l617_61723

-- Define the L-shaped figure
structure LShape :=
  (squares : Fin 3 → Unit)  -- Using Unit as a placeholder for Square

-- Define the additional lettered squares
inductive LetteredSquare
  | A | B | C | D | E | F | G
  deriving Fintype, DecidableEq

-- Define a function to check if a configuration can form a topless cubical box
def canFormToplessCubicalBox (l : LShape) (s1 s2 : LetteredSquare) : Bool :=
  match s1, s2 with
  | LetteredSquare.A, LetteredSquare.B => true
  | LetteredSquare.B, LetteredSquare.A => true
  | LetteredSquare.E, LetteredSquare.F => true
  | LetteredSquare.F, LetteredSquare.E => true
  | _, _ => false

-- Theorem statement
theorem two_valid_configurations (l : LShape) :
  ∃! (n : Nat), n = (Finset.filter 
    (fun p : LetteredSquare × LetteredSquare => 
      canFormToplessCubicalBox l p.1 p.2)
    (Finset.product Finset.univ Finset.univ)).card ∧
  n = 2 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_valid_configurations_l617_61723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_time_l617_61712

noncomputable def boat_speed : ℝ := 16
noncomputable def stream_speed : ℝ := 2
noncomputable def distance : ℝ := 7020

noncomputable def upstream_speed : ℝ := boat_speed - stream_speed
noncomputable def downstream_speed : ℝ := boat_speed + stream_speed

noncomputable def time_upstream : ℝ := distance / upstream_speed
noncomputable def time_downstream : ℝ := distance / downstream_speed

noncomputable def total_time : ℝ := time_upstream + time_downstream

theorem round_trip_time : ∃ ε > 0, |total_time - 891.4286| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_time_l617_61712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ram_completes_in_six_days_l617_61791

/-- The time (in days) it takes for Ram to complete the work alone, given that:
    1. Raja and Ram together can complete the work in 4 days
    2. Raja alone can complete the work in 12 days -/
noncomputable def ram_work_time (total_work : ℝ) : ℝ :=
  let raja_ram_rate := total_work / 4
  let raja_rate := total_work / 12
  let ram_rate := raja_ram_rate - raja_rate
  total_work / ram_rate

/-- Theorem stating that Ram can complete the work alone in 6 days -/
theorem ram_completes_in_six_days (total_work : ℝ) (total_work_pos : total_work > 0) :
  ram_work_time total_work = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ram_completes_in_six_days_l617_61791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_vector_satisfies_condition_l617_61792

-- Define the parametric equations for lines l and m
def line_l (t : ℝ) : ℝ × ℝ := (3 + 5*t, 2 + 4*t)
def line_m (s : ℝ) : ℝ × ℝ := (-7 + 5*s, 3 + 4*s)

-- Define points A and B on lines l and m respectively
def A (t : ℝ) : ℝ × ℝ := line_l t
def B (s : ℝ) : ℝ × ℝ := line_m s

-- Define the direction vector of line m
def direction_m : ℝ × ℝ := (5, 4)

-- Define the theorem
theorem projection_vector_satisfies_condition :
  ∃ (v₁ v₂ : ℝ), v₁ * v₂ = 12 ∧
  ∀ (t s : ℝ),
  let BA := ((A t).1 - (B s).1, (A t).2 - (B s).2)
  let projection := (v₁, v₂)
  (∃ (k : ℝ), BA = (k * projection.1, k * projection.2)) ∧
  (projection.1 * direction_m.1 + projection.2 * direction_m.2 = 0) ∧
  v₁ = 6 ∧ v₂ = 2 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_vector_satisfies_condition_l617_61792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l617_61745

theorem tan_alpha_value (α : Real) (h1 : α ∈ Set.Ioo 0 (π / 2)) 
  (h2 : Real.tan (2 * α) = Real.cos α / (2 - Real.sin α)) : 
  Real.tan α = Real.sqrt 15 / 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l617_61745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_modulus_example_l617_61730

theorem complex_modulus_example : Complex.abs (-3 + (11/3) * Complex.I) = Real.sqrt 202 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_modulus_example_l617_61730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin6_plus_cos6_l617_61731

theorem sin6_plus_cos6 (θ : ℝ) (h : Real.cos (2 * θ) = 1/4) :
  Real.sin θ ^ 6 + Real.cos θ ^ 6 = 19/64 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin6_plus_cos6_l617_61731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_arccos_cos_l617_61772

open Real

-- Define the function
noncomputable def f (x : ℝ) : ℝ := arccos (cos x)

-- Define the interval
noncomputable def a : ℝ := π / 4
noncomputable def b : ℝ := 9 * π / 4

-- State the theorem
theorem area_enclosed_by_arccos_cos : ∫ x in a..b, f x = 3 * π^2 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_arccos_cos_l617_61772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_expensive_product_price_l617_61776

/-- Represents a product with its price -/
structure Product where
  price : ℚ
  deriving Repr

/-- Represents a company's product line -/
structure Company where
  products : List Product
  deriving Repr

def Company.averagePrice (c : Company) : ℚ :=
  (c.products.map Product.price).sum / c.products.length

noncomputable def Company.minPrice (c : Company) : ℚ :=
  (c.products.map Product.price).minimum?.getD 0

def Company.countLessThan (c : Company) (threshold : ℚ) : Nat :=
  c.products.filter (fun p => p.price < threshold) |>.length

theorem max_expensive_product_price (c : Company) 
    (h1 : c.products.length = 20)
    (h2 : c.averagePrice = 1200)
    (h3 : c.minPrice ≥ 400)
    (h4 : c.countLessThan 1000 = 10) :
    (c.products.map Product.price).maximum?.getD 0 ≤ 11000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_expensive_product_price_l617_61776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_product_theorem_l617_61793

theorem log_product_theorem (a b c : ℝ) (ha : a = 2) (hb : b = 4) (hc : c = 16) :
  Real.log b / Real.log a * (Real.log c / Real.log b) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_product_theorem_l617_61793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_six_people_round_table_l617_61700

/-- The number of unique seating arrangements for n people around a round table -/
def roundTableArrangements (n : ℕ) : ℕ := Nat.factorial (n - 1)

theorem six_people_round_table : roundTableArrangements 6 = 120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_six_people_round_table_l617_61700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_game_probability_theorem_l617_61725

/-- The probability of a specific outcome in a game with 6 rounds -/
def game_probability (p_alex p_mel p_chelsea : ℚ) : Prop :=
  -- Total number of rounds
  let total_rounds : ℕ := 6
  -- Probabilities sum to 1
  (p_alex + p_mel + p_chelsea = 1)
  -- Alex's winning probability
  ∧ (p_alex = 1/3)
  -- Chelsea's winning probability is three times Mel's
  ∧ (p_chelsea = 3 * p_mel)
  -- The probability of Alex winning 4, Mel 1, and Chelsea 1 round
  ∧ ((Nat.choose total_rounds 4 * Nat.choose 2 1 : ℚ) * p_alex^4 * p_mel * p_chelsea = 5/162)

/-- The main theorem to prove -/
theorem game_probability_theorem :
  ∃ (p_alex p_mel p_chelsea : ℚ), game_probability p_alex p_mel p_chelsea := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_game_probability_theorem_l617_61725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_direction_form_equation_l617_61756

/-- Given a line l passing through point P(1, 2) with direction vector (3, -4),
    prove that the point-direction form equation of line l is (x - 1)/3 = (y - 2)/(-4). -/
theorem point_direction_form_equation 
  (l : Set (ℝ × ℝ))
  (h1 : (1, 2) ∈ l)
  (h2 : ∃ t : ℝ, ∀ (x y : ℝ), (x, y) ∈ l ↔ ∃ s : ℝ, x = 1 + 3*s ∧ y = 2 - 4*s) :
  ∀ x y : ℝ, (x, y) ∈ l ↔ (x - 1) / 3 = (y - 2) / (-4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_direction_form_equation_l617_61756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_convergence_l617_61707

noncomputable def series_sum (k : ℕ) : ℝ := (2 * k : ℝ) / (4 ^ k)

theorem series_convergence : ∑' k, series_sum k = 8 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_convergence_l617_61707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l617_61706

/-- The time it takes for train A to pass the observer, in seconds -/
noncomputable def passing_time : ℝ := 9

/-- The length of train A, in meters -/
noncomputable def train_length : ℝ := 280

/-- The speed of train A, in kilometers per hour -/
noncomputable def train_a_speed : ℝ := 62

/-- Conversion factor from km/h to m/s -/
noncomputable def kmh_to_ms : ℝ := 5 / 18

/-- The speed of train B, in kilometers per hour -/
noncomputable def train_b_speed : ℝ := 50

theorem train_speed_calculation :
  train_b_speed = 
    (train_length / (passing_time * kmh_to_ms)) - train_a_speed := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l617_61706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_P_and_Q_l617_61795

-- Define the sets P and Q
def P : Set ℝ := { y | ∃ x : ℝ, y = x^2 + 3*x + 1 }
def Q : Set ℝ := { y | ∃ x : ℝ, y = -x^2 - 3*x + 1 }

-- State the theorem
theorem intersection_of_P_and_Q :
  P ∩ Q = Set.Icc (-5/4 : ℝ) (13/4 : ℝ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_P_and_Q_l617_61795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l617_61787

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (-x^2 + x)

-- Theorem statement
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = Set.Ioo 0 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l617_61787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_equals_sqrt_three_l617_61760

theorem tan_alpha_equals_sqrt_three (α : ℝ) 
  (h1 : Real.cos (α - π) = 1/2) 
  (h2 : -π < α) 
  (h3 : α < 0) : 
  Real.tan α = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_equals_sqrt_three_l617_61760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_surface_area_is_7pi_l617_61761

/-- The surface area of a cone with base radius 1 and central angle of unfolded lateral surface 60° -/
noncomputable def coneSurfaceArea : ℝ := 7 * Real.pi

/-- Theorem: The surface area of a cone with base radius 1 and central angle of unfolded lateral surface 60° is 7π -/
theorem cone_surface_area_is_7pi :
  let baseRadius : ℝ := 1
  let centralAngle : ℝ := 60 * (Real.pi / 180)  -- Convert 60° to radians
  coneSurfaceArea = 7 * Real.pi := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_surface_area_is_7pi_l617_61761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_expression_l617_61742

theorem max_value_expression (a b c d : ℝ) 
  (ha : a ∈ Set.Icc (-4 : ℝ) 4)
  (hb : b ∈ Set.Icc (-4 : ℝ) 4)
  (hc : c ∈ Set.Icc (-4 : ℝ) 4)
  (hd : d ∈ Set.Icc (-4 : ℝ) 4) :
  (∀ x y z w, x ∈ Set.Icc (-4 : ℝ) 4 → y ∈ Set.Icc (-4 : ℝ) 4 → z ∈ Set.Icc (-4 : ℝ) 4 → w ∈ Set.Icc (-4 : ℝ) 4 →
    x + 2*y + z + 2*w - x*y - y*z - z*w - w*x ≤ 72) ∧
  (∃ x y z w, x ∈ Set.Icc (-4 : ℝ) 4 ∧ y ∈ Set.Icc (-4 : ℝ) 4 ∧ z ∈ Set.Icc (-4 : ℝ) 4 ∧ w ∈ Set.Icc (-4 : ℝ) 4 ∧
    x + 2*y + z + 2*w - x*y - y*z - z*w - w*x = 72) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_expression_l617_61742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l617_61711

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 2 then -x + 6 else 3 + (Real.log x) / (Real.log a)

theorem range_of_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ y : ℝ, y ≥ 4 → ∃ x : ℝ, f a x = y) ∧ 
  (∀ x : ℝ, f a x ≥ 4) →
  1 < a ∧ a ≤ 2 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l617_61711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_jar_difference_l617_61767

theorem marble_jar_difference : ∀ (a b : ℤ),
  (10 * a = 9 * b) →  -- Same number of marbles in each jar
  (3 * a + b = 110) →  -- Total green marbles
  (7 * a - 8 * b = -51) :=  -- Difference in blue marbles
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_jar_difference_l617_61767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_150_of_47_over_777_l617_61762

def decimal_expansion (n d : ℕ) : List ℕ := sorry

def list_cycle (l : List α) : List α := sorry

theorem digit_150_of_47_over_777 : 
  let expansion := decimal_expansion 47 777
  let repeating_sequence := [0, 6, 0, 6, 0, 6]
  let repeat_cycle_length := 6
  expansion.drop 1 = list_cycle repeating_sequence →
  expansion.get? 150 = some 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_150_of_47_over_777_l617_61762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_between_spheres_l617_61782

theorem max_distance_between_spheres :
  let sphere1_center : ℝ × ℝ × ℝ := (0, 0, 0)
  let sphere1_radius : ℝ := 25
  let sphere2_center : ℝ × ℝ × ℝ := (20, 15, -25)
  let sphere2_radius : ℝ := 60
  ∃ (p1 p2 : ℝ × ℝ × ℝ),
    (‖p1 - sphere1_center‖ = sphere1_radius) ∧
    (‖p2 - sphere2_center‖ = sphere2_radius) ∧
    (∀ (q1 q2 : ℝ × ℝ × ℝ),
      ‖q1 - sphere1_center‖ = sphere1_radius →
      ‖q2 - sphere2_center‖ = sphere2_radius →
      ‖q1 - q2‖ ≤ ‖p1 - p2‖) ∧
    ‖p1 - p2‖ = 85 + 25 * Real.sqrt 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_between_spheres_l617_61782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_condition_l617_61726

theorem perfect_square_condition (x y : ℕ+) : 
  ∃ (a b d k m n : ℕ+), 
    (↑x * ↑y + 1) * (↑x * ↑y + ↑x + 2) = (↑k * ↑d)^2 → 
    x = a * d - 1 ∧
    y * a = b * d - 2 ∧
    a = n^2 - m^2 ∧
    a * b * d + a - b = m^2 ∧
    a * b * d + 2 * a - b = n^2 ∧
    Nat.gcd a b = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_condition_l617_61726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l617_61721

noncomputable def f (x : ℝ) : ℝ := x / (4 - x^2)

theorem f_properties :
  let S := Set.Ioo (-2 : ℝ) 2
  (∀ x₁ x₂, x₁ ∈ S → x₂ ∈ S → 0 < x₁ → x₁ < x₂ → x₂ < 2 → f x₁ < f x₂) ∧
  (∀ x ∈ S, f (-x) = -f x) ∧
  {t : ℝ | f t + f (1 - 2*t) > 0} = Set.Ioo (-1/2 : ℝ) 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l617_61721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l617_61773

noncomputable section

open Real

-- Define the piecewise function f
def f (x : ℝ) : ℝ :=
  if x ≤ 1/2 then -1/2 * x + 1/4
  else 2 * x^2 / (x + 2)

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ :=
  a * sin (π/3 * x + 3/2 * π) - 2 * a + 2

-- Theorem statement
theorem function_properties :
  (∀ x ∈ Set.Icc (0 : ℝ) 1, f x ∈ Set.Icc (0 : ℝ) (2/3)) ∧
  (∀ a > 0, StrictMono (g a)) ∧
  (∀ a ∈ Set.Icc (4/9 : ℝ) (4/5),
    ∃ x₁ x₂, x₁ ∈ Set.Icc (0 : ℝ) 1 ∧ x₂ ∈ Set.Icc (0 : ℝ) 1 ∧ f x₁ = g a x₂) ∧
  (∀ a > 0, (∃ x₁ x₂, x₁ ∈ Set.Icc (0 : ℝ) 1 ∧ x₂ ∈ Set.Icc (0 : ℝ) 1 ∧ f x₁ = g a x₂) → 
    a ∈ Set.Icc (4/9 : ℝ) (4/5)) :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l617_61773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_power_six_l617_61734

noncomputable def A : Matrix (Fin 2) (Fin 2) ℝ := !![Real.sqrt 3, -1; 1, Real.sqrt 3]

theorem matrix_power_six :
  A^6 = !![(-64 : ℝ), 0; 0, -64] := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_power_six_l617_61734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_sum_l617_61719

theorem trigonometric_sum (x : ℝ) (a b c : ℕ+) :
  0 < x ∧ x < π / 2 →
  Real.sin x - Real.cos x = π / 4 →
  Real.tan x + 1 / Real.tan x = (a : ℝ) / (b - π^(c : ℝ)) →
  Real.tan x + 1 / Real.tan x = 32 / (16 - π^2) →
  a + b + c = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_sum_l617_61719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_constant_l617_61733

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * (Real.sin (Real.log x))^2 + Real.cos (Real.log (x^2)) - 5

-- State the theorem
theorem f_constant (x : ℝ) (h : x > 0) : f x = -4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_constant_l617_61733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eddy_travel_time_is_three_hours_l617_61748

/-- Represents the travel scenario of Eddy and Freddy -/
structure TravelScenario where
  distance_AB : ℝ
  distance_AC : ℝ
  freddy_time : ℝ
  speed_ratio : ℝ

/-- Calculates Eddy's travel time given a TravelScenario -/
noncomputable def eddy_travel_time (scenario : TravelScenario) : ℝ :=
  scenario.distance_AB / (scenario.speed_ratio * (scenario.distance_AC / scenario.freddy_time))

/-- Theorem stating that Eddy's travel time is 3 hours given the specific scenario -/
theorem eddy_travel_time_is_three_hours :
  let scenario : TravelScenario := {
    distance_AB := 600,
    distance_AC := 360,
    freddy_time := 4,
    speed_ratio := 2.2222222222222223
  }
  eddy_travel_time scenario = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eddy_travel_time_is_three_hours_l617_61748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_fraction_parts_l617_61703

def repeating_decimal : ℚ := 45 / 99

theorem sum_of_fraction_parts : 
  let fraction := repeating_decimal
  (fraction.num.natAbs + fraction.den) = 16 := by
  rw [repeating_decimal]
  norm_num
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_fraction_parts_l617_61703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplified_function_properties_l617_61788

/-- The original function f(x) -/
noncomputable def f (x : ℝ) : ℝ := (x^3 + 9*x^2 + 28*x + 30) / (x + 3)

/-- The theorem stating the properties of the simplified function -/
theorem simplified_function_properties :
  ∃ (A B C D : ℝ),
    (∀ x : ℝ, x ≠ D → f x = A*x^2 + B*x + C) ∧
    (¬ ∃ y : ℝ, f D = y) ∧
    A + B + C + D = 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplified_function_properties_l617_61788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inradius_inequality_l617_61785

/-- A triangle in 2D Euclidean space -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Predicate to check if R is the circumradius of triangle ABC -/
def is_circumradius (ABC : Triangle) (R : ℝ) : Prop := sorry

/-- Predicate to check if r is the inradius of triangle ABC -/
def is_inradius (ABC : Triangle) (r : ℝ) : Prop := sorry

/-- Predicate to check if p is the inradius of the orthic triangle of ABC -/
def is_orthic_inradius (ABC : Triangle) (p : ℝ) : Prop := sorry

/-- Given a triangle ABC with circumradius R, inradius r, and orthic triangle inradius p,
    prove that p/R ≤ 1 - (1/3) * (1 + r/R)^2 -/
theorem triangle_inradius_inequality (ABC : Triangle) (R r p : ℝ) 
    (h_R : R > 0)
    (h_r : r > 0)
    (h_p : p > 0)
    (h_circum : is_circumradius ABC R)
    (h_in : is_inradius ABC r)
    (h_orthic : is_orthic_inradius ABC p) : 
  p / R ≤ 1 - (1 / 3) * (1 + r / R)^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inradius_inequality_l617_61785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inaccessible_angle_bisector_l617_61718

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an angle -/
structure Angle where
  vertex : Point
  ray1 : Point → Prop
  ray2 : Point → Prop

/-- Represents a line -/
structure Line where
  point1 : Point
  point2 : Point

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ := sorry

/-- Check if a point is on a ray -/
def on_ray (p : Point) (ray : Point → Prop) : Prop := sorry

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop := sorry

/-- The angle bisector of an angle -/
noncomputable def angle_bisector (a : Angle) : Line := sorry

theorem inaccessible_angle_bisector 
  (A : Point) 
  (angle : Angle)
  (h_vertex : angle.vertex = A)
  (h_inaccessible : ¬ ∃ (p : Point), p = A)
  (A₁ B₁ : Point)
  (h_A₁_on_ray1 : on_ray A₁ angle.ray1)
  (h_B₁_on_ray2 : on_ray B₁ angle.ray2)
  (h_equal_distance : distance A A₁ = distance A B₁)
  (l₁ l₂ : Line)
  (h_l₁_parallel : parallel l₁ (Line.mk A B₁))
  (h_l₂_parallel : parallel l₂ (Line.mk A A₁))
  (h_A₁_on_l₁ : A₁ = l₁.point1 ∨ A₁ = l₁.point2)
  (h_A₁_on_l₂ : A₁ = l₂.point1 ∨ A₁ = l₂.point2)
  (angle_A₁ : Angle)
  (h_angle_A₁_vertex : angle_A₁.vertex = A₁)
  (h_angle_A₁_rays : ∃ (p1 p2 : Point), angle_A₁.ray1 p1 ∧ angle_A₁.ray2 p2) :
  angle_bisector angle_A₁ = angle_bisector angle := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inaccessible_angle_bisector_l617_61718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_calculations_l617_61763

theorem arithmetic_calculations :
  (7 - 5 + (-4) - (-10) = 8) ∧
  (-2^3 + (-5)^2 * (2/5) - |(-3)| = -1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_calculations_l617_61763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_box_diagonal_sum_l617_61794

/-- Represents a rectangular box with side lengths a, b, and c -/
structure RectangularBox where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The sum of all edge lengths of a rectangular box -/
def sumOfEdges (box : RectangularBox) : ℝ :=
  4 * (box.a + box.b + box.c)

/-- The total surface area of a rectangular box -/
def surfaceArea (box : RectangularBox) : ℝ :=
  2 * (box.a * box.b + box.b * box.c + box.c * box.a)

/-- The sum of lengths of all interior diagonals of a rectangular box -/
noncomputable def sumOfDiagonals (box : RectangularBox) : ℝ :=
  4 * Real.sqrt (box.a^2 + box.b^2 + box.c^2)

/-- Theorem: For a rectangular box with total surface area 130 square inches
    and sum of edge lengths 60 inches, the sum of lengths of all interior
    diagonals is 4√95 inches -/
theorem box_diagonal_sum :
  ∀ box : RectangularBox,
    surfaceArea box = 130 →
    sumOfEdges box = 60 →
    sumOfDiagonals box = 4 * Real.sqrt 95 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_box_diagonal_sum_l617_61794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_two_largest_l617_61765

def numbers : List ℕ := [10, 11, 12]

theorem product_of_two_largest (nums := numbers) : 
  (nums.maximum?.getD 0) * (nums.filter (· ≠ nums.maximum?.getD 0)).maximum?.getD 0 = 132 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_two_largest_l617_61765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_odd_with_five_prime_factors_l617_61774

theorem smallest_odd_with_five_prime_factors :
  ∀ n : ℕ, Odd n → (∃ p₁ p₂ p₃ p₄ p₅ : ℕ, Nat.Prime p₁ ∧ Nat.Prime p₂ ∧ Nat.Prime p₃ ∧ Nat.Prime p₄ ∧ Nat.Prime p₅ ∧
    p₁ < p₂ ∧ p₂ < p₃ ∧ p₃ < p₄ ∧ p₄ < p₅ ∧
    n = p₁ * p₂ * p₃ * p₄ * p₅) →
  n ≥ 15015 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_odd_with_five_prime_factors_l617_61774

import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l138_13829

noncomputable def A : Set ℝ := {x | ∃ k : ℤ, x = k * Real.pi + Real.pi / 2}

noncomputable def vec_a (α β : ℝ) : ℝ × ℝ := (2 * Real.cos ((α + β) / 2), Real.sin ((α - β) / 2))

noncomputable def vec_b (α β : ℝ) : ℝ × ℝ := (Real.cos ((α + β) / 2), 3 * Real.sin ((α - β) / 2))

theorem part_one (α β : ℝ) (h1 : α + β = 2 * Real.pi / 3) (h2 : vec_a α β = 2 • vec_b α β) :
  ∃ k : ℤ, α = k * Real.pi + Real.pi / 3 ∧ β = -k * Real.pi + Real.pi / 3 := by
  sorry

theorem part_two (α β : ℝ) (h1 : α ∈ A) (h2 : β ∈ A) 
  (h3 : vec_a α β • vec_b α β = 5 / 2) :
  Real.tan α * Real.tan β = -1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l138_13829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_third_quadrant_sin_cos_tan_point_l138_13816

/-- Given an angle α in the third quadrant with tan α = 1/3, 
    prove that sin α = -√10/10 and cos α = -3√10/10 -/
theorem sin_cos_third_quadrant (α : Real) 
  (h1 : α ∈ Set.Icc π (3*π/2)) 
  (h2 : Real.tan α = 1/3) : 
  Real.sin α = -Real.sqrt 10 / 10 ∧ Real.cos α = -3 * Real.sqrt 10 / 10 := by
  sorry

/-- Given a point P(3a, 4a) on the terminal side of angle α, where a ≠ 0,
    prove that:
    if a > 0, then sin α = 4/5, cos α = 3/5, and tan α = 4/3
    if a < 0, then sin α = -4/5, cos α = -3/5, and tan α = 4/3 -/
theorem sin_cos_tan_point (α : Real) (a : Real) 
  (h1 : a ≠ 0) 
  (h2 : ∃ (r : Real), r * (Real.cos α) = 3*a ∧ r * (Real.sin α) = 4*a) :
  (a > 0 → Real.sin α = 4/5 ∧ Real.cos α = 3/5 ∧ Real.tan α = 4/3) ∧
  (a < 0 → Real.sin α = -4/5 ∧ Real.cos α = -3/5 ∧ Real.tan α = 4/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_third_quadrant_sin_cos_tan_point_l138_13816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_condition_upper_bound_no_solution_l138_13825

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x + Real.log x

-- Theorem I
theorem increasing_condition (a : ℝ) :
  (∀ x ∈ Set.Icc 1 2, MonotoneOn (f a) (Set.Icc 1 2)) → a ≥ -1/2 := by
  sorry

-- Theorem II
theorem upper_bound :
  ∀ x > 0, f (-Real.exp 1) x + 2 ≤ 0 := by
  sorry

-- Theorem III
theorem no_solution :
  ∀ x > 0, |f (-Real.exp 1) x| > Real.log x / x + 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_condition_upper_bound_no_solution_l138_13825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_composite_functions_l138_13802

noncomputable def f (x : ℝ) : ℝ := (3 * x - 1) / (x + 1)

noncomputable def f₁ (x : ℝ) : ℝ := f x

noncomputable def f₂ (x : ℝ) : ℝ := f (f x)

noncomputable def f₃ (x : ℝ) : ℝ := f (f₂ x)

theorem composite_functions (x : ℝ) (h : x ≠ -1 ∧ x ≠ 0 ∧ x ≠ 1/3) : 
  f₁ x = (3 * x - 1) / (x + 1) ∧ 
  f₂ x = 2 - 1/x ∧ 
  f₃ x = (5 * x - 3) / (3 * x - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_composite_functions_l138_13802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_l138_13862

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines the parametric curve -/
noncomputable def parametric_curve (θ : ℝ) : Point where
  x := Real.sin θ
  y := (Real.sin θ) ^ 2

/-- Defines the line y = x + 2 -/
def line (x : ℝ) : ℝ :=
  x + 2

/-- Theorem stating that (-1, 1) is the only intersection point -/
theorem intersection_point :
  ∃! p : Point, 
    (∃ θ : ℝ, p = parametric_curve θ) ∧ 
    p.y = line p.x ∧ 
    -1 ≤ p.x ∧ p.x ≤ 1 ∧
    p = ⟨-1, 1⟩ := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_l138_13862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_G_101_equals_103_l138_13895

def G : ℕ → ℚ
  | 0 => 3  -- Adding the base case for 0
  | 1 => 3
  | (n + 1) => (3 * G n + 3) / 3

theorem G_101_equals_103 : G 101 = 103 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_G_101_equals_103_l138_13895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l138_13840

/-- Calculates the speed of a train given its length and time to pass a signal post -/
noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  (length / 1000) / (time / 3600)

/-- Theorem stating that a train of length 400 meters passing a signal post in 24 seconds 
    has a speed of approximately 59.97 km/hour -/
theorem train_speed_calculation :
  let length : ℝ := 400
  let time : ℝ := 24
  abs (train_speed length time - 59.97) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l138_13840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_at_x0_l138_13832

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 13 - 8*x + Real.sqrt 2 * x^2

-- State the theorem
theorem derivative_at_x0 (x₀ : ℝ) : 
  deriv f x₀ = 4 → x₀ = 3 * Real.sqrt 2 := by
  sorry

#check derivative_at_x0

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_at_x0_l138_13832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slopes_product_constant_l138_13827

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2/16 + y^2/12 = 1

-- Define points M, N, and T
variable (M N T : ℝ × ℝ)

-- Define that M, N, and T are on the ellipse
axiom hM : is_on_ellipse M.1 M.2
axiom hN : is_on_ellipse N.1 N.2
axiom hT : is_on_ellipse T.1 T.2

-- Define that MN passes through the origin
axiom hMN_origin : ∃ (t : ℝ), t • M = N

-- Define slopes of MT and NT
noncomputable def slope_MT (M T : ℝ × ℝ) : ℝ := (T.2 - M.2) / (T.1 - M.1)
noncomputable def slope_NT (N T : ℝ × ℝ) : ℝ := (T.2 - N.2) / (T.1 - N.1)

-- Theorem statement
theorem slopes_product_constant (M N T : ℝ × ℝ) :
  slope_MT M T * slope_NT N T = -3/4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slopes_product_constant_l138_13827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_l138_13885

/-- The function f(x) defined in the problem -/
noncomputable def f (x : ℝ) : ℝ := x^2 + 1/x^2 + 1/(x^2 + 1/x^2)

/-- Theorem stating the minimum value of f(x) for x > 0 -/
theorem f_min_value :
  (∀ x > 0, f x ≥ 2.5) ∧ (∃ x > 0, f x = 2.5) := by
  sorry

#check f_min_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_l138_13885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_8_625_l138_13842

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

def intersection_point : Point := ⟨3, 3⟩

noncomputable def line1 : Line := ⟨1/3, 2⟩
noncomputable def line2 : Line := ⟨3, -6⟩
noncomputable def line3 : Line := ⟨-1, 12⟩

/-- Calculates the area of a triangle given three points -/
noncomputable def triangle_area (A B C : Point) : ℝ :=
  (1/2) * abs (A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y))

/-- Finds the intersection of a line with the line x + y = 12 -/
noncomputable def intersection_with_line3 (l : Line) : Point :=
  let x := (12 - l.intercept) / (l.slope + 1)
  ⟨x, l.slope * x + l.intercept⟩

theorem triangle_area_is_8_625 :
  let A := intersection_point
  let B := intersection_with_line3 line1
  let C := intersection_with_line3 line2
  triangle_area A B C = 8.625 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_8_625_l138_13842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_bounded_by_curves_l138_13812

-- Define the bounding functions
noncomputable def f (y : ℝ) : ℝ := Real.sqrt (4 - y^2)
def g (_ : ℝ) : ℝ := 0

-- Define the bounds for y
def lower_bound : ℝ := 0
def upper_bound : ℝ := 1

-- State the theorem
theorem area_bounded_by_curves :
  (∫ y in lower_bound..upper_bound, f y - g y) = Real.sqrt 3 / 2 + Real.pi / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_bounded_by_curves_l138_13812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_exponent_simplification_l138_13884

theorem fraction_exponent_simplification :
  (5 / 9 : ℚ) ^ 4 * (5 / 9 : ℚ) ^ (-(2 : ℤ)) = 25 / 81 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_exponent_simplification_l138_13884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_greater_than_one_iff_l138_13891

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then (1/2)^x else x^(1/2)

-- Theorem statement
theorem f_greater_than_one_iff (a : ℝ) :
  f a > 1 ↔ a ∈ Set.Iio 0 ∪ Set.Ioi 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_greater_than_one_iff_l138_13891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_is_correct_l138_13839

/-- A lattice point in an xy-coordinate system is any point (x, y) where both x and y are integers. -/
def is_lattice_point (x y : ℤ) : Prop := True

/-- The graph of y = mx + 5 passes through no lattice point with 0 < x ≤ 150 for all m such that 1/3 < m < a -/
def no_lattice_points (m a : ℚ) : Prop :=
  ∀ x y : ℤ, 0 < x → x ≤ 150 → is_lattice_point x y → 1/3 < m → m < a → ↑y ≠ m * ↑x + 5

/-- The maximum possible value of a -/
def max_a : ℚ := 52/151

theorem max_a_is_correct : 
  (∀ a : ℚ, (∀ m : ℚ, no_lattice_points m a) → a ≤ max_a) ∧ 
  (∀ m : ℚ, no_lattice_points m max_a) :=
sorry

#check max_a_is_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_is_correct_l138_13839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_is_maximum_l138_13819

/-- The function f(x) defined in the problem -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (1/2) * Real.cos (2*x) + m * Real.sin x + (1/2)

/-- The maximum value of f(x) for a given m -/
noncomputable def f_max (m : ℝ) : ℝ :=
  if m ≤ -2 then -m
  else if m < 2 then m^2/4 + 1
  else m

/-- Theorem stating that f_max(m) is the maximum value of f(x) for any x -/
theorem f_max_is_maximum (m : ℝ) : ∀ x, f m x ≤ f_max m := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_is_maximum_l138_13819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_symmetry_l138_13833

-- Define the two functions
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 3
noncomputable def g (x : ℝ) : ℝ := Real.log (9 * x) / Real.log (1/3)

-- Define symmetry about y = -1
def symmetric_about_y_neg_1 (f g : ℝ → ℝ) : Prop :=
  ∀ x > 0, ∃ y > 0, f x + g y = -2 ∧ f y + g x = -2

-- Theorem statement
theorem log_symmetry : symmetric_about_y_neg_1 f g := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_symmetry_l138_13833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_is_72_l138_13823

/-- The speed of a train in km/hr, given its length and time to cross a pole -/
noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  (length / time) * 3.6

/-- Theorem: The speed of a train is 72 km/hr -/
theorem train_speed_is_72 (length : ℝ) (time : ℝ) 
  (h1 : length = 180) 
  (h2 : time = 9) : 
  train_speed length time = 72 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_is_72_l138_13823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_simplification_l138_13892

theorem complex_expression_simplification :
  -(1^2022) + (3 - Real.pi)^0 - 1/8 * (-1/2)^(-2 : ℤ) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_simplification_l138_13892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_atomic_clock_accuracy_notation_l138_13803

/-- Scientific notation representation -/
noncomputable def scientific_notation (a : ℝ) (n : ℤ) : ℝ := a * (10 : ℝ) ^ n

/-- Definition of valid scientific notation -/
def is_valid_scientific_notation (a : ℝ) (n : ℤ) : Prop :=
  1 ≤ a ∧ a < 10

theorem atomic_clock_accuracy_notation :
  ∃ (a : ℝ) (n : ℤ), 
    scientific_notation a n = 1700000 ∧ 
    is_valid_scientific_notation a n ∧
    a = 1.7 ∧ n = 6 := by
  use 1.7, 6
  apply And.intro
  · -- Prove scientific_notation 1.7 6 = 1700000
    sorry
  apply And.intro
  · -- Prove is_valid_scientific_notation 1.7 6
    sorry
  apply And.intro
  · -- Prove 1.7 = 1.7
    rfl
  · -- Prove 6 = 6
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_atomic_clock_accuracy_notation_l138_13803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_at_two_thirty_l138_13806

/-- Represents a 12-hour analog clock --/
structure AnalogClock where
  hours : Nat
  minutes : Nat

/-- Calculates the angle of the hour hand from 12 o'clock position --/
noncomputable def hour_hand_angle (c : AnalogClock) : Real :=
  (c.hours % 12 + c.minutes / 60 : Real) * 30

/-- Calculates the angle of the minute hand from 12 o'clock position --/
def minute_hand_angle (c : AnalogClock) : Real :=
  (c.minutes : Real) * 6

/-- Calculates the smaller angle between hour and minute hands --/
noncomputable def smaller_angle (c : AnalogClock) : Real :=
  min (abs (hour_hand_angle c - minute_hand_angle c))
      (360 - abs (hour_hand_angle c - minute_hand_angle c))

/-- Theorem: At 2:30, the smaller angle between hour and minute hands is 105° --/
theorem angle_at_two_thirty :
  smaller_angle { hours := 2, minutes := 30 } = 105 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_at_two_thirty_l138_13806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_theorem_l138_13889

/-- A point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance squared between two points -/
def distanceSquared (p q : Point) : ℝ :=
  (p.x - q.x)^2 + (p.y - q.y)^2

/-- The parabola y^2 = 16x -/
def isOnParabola (p : Point) : Prop :=
  p.y^2 = 16 * p.x

/-- A line passing through a point with a given slope -/
def lineThrough (m : Point) (slope : ℝ) (p : Point) : Prop :=
  p.x = m.x + slope * (p.y - m.y)

/-- The theorem to be proved -/
theorem parabola_intersection_theorem (m : ℝ) : 
  (∃ C : ℝ, ∀ slope : ℝ, 
    let M : Point := ⟨m, 0⟩
    ∃ P Q : Point, 
      isOnParabola P ∧ isOnParabola Q ∧
      lineThrough M slope P ∧ lineThrough M slope Q ∧
      1 / distanceSquared M P + 1 / distanceSquared M Q = C) →
  m = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_theorem_l138_13889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_power_of_18_dividing_30_factorial_l138_13843

def factorial (n : ℕ) : ℕ := Finset.prod (Finset.range n) (λ i ↦ i + 1)

theorem largest_power_of_18_dividing_30_factorial :
  (∃ m : ℕ, 18^m ∣ factorial 30) ∧
  (∀ k : ℕ, k > 7 → ¬(18^k ∣ factorial 30)) ∧
  (18^7 ∣ factorial 30) := by
  sorry

#eval factorial 30

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_power_of_18_dividing_30_factorial_l138_13843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_sum_l138_13846

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- Checks if a point is on the ellipse -/
def isOnEllipse (p : Point) (e : Ellipse) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: Sum of distances from foci to points on ellipse intersected by a line through one focus -/
theorem ellipse_foci_distance_sum
  (e : Ellipse)
  (f1 f2 a b : Point)
  (h1 : e.a^2 = 16 ∧ e.b^2 = 9)
  (h2 : isOnEllipse a e ∧ isOnEllipse b e)
  (h3 : ∃ (t : ℝ), a = Point.mk (f2.x + t * (b.x - f2.x)) (f2.y + t * (b.y - f2.y)))
  (h4 : distance a b = 6) :
  distance a f1 + distance b f1 = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_sum_l138_13846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_monotonicity_l138_13898

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^2 - 8 * Real.log x + 3

-- State the theorem
theorem tangent_line_and_monotonicity (x : ℝ) (h : x > 0) :
  -- 1. Equation of tangent line at x = 1
  (fun y => 6 * x + y - 10 = 0) (f 1) ∧
  -- 2. Monotonically increasing for x > 2
  (∀ x₁ x₂, 2 < x₁ ∧ x₁ < x₂ → f x₁ < f x₂) ∧
  -- 3. Monotonically decreasing for 0 < x < 2
  (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 2 → f x₁ > f x₂) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_monotonicity_l138_13898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_increasing_l138_13893

noncomputable def f (ω φ x : Real) : Real := Real.sin (ω * x + φ)

theorem function_increasing (ω φ : Real) 
  (h1 : ω > 0) 
  (h2 : 0 < φ ∧ φ < Real.pi) 
  (h3 : ∀ x, f ω φ (x + Real.pi / ω) = f ω φ x) 
  (h4 : f ω φ (-Real.pi/6 + Real.pi/3) = 1) : 
  ∀ x y, -Real.pi/3 ≤ x ∧ x < y ∧ y ≤ Real.pi/6 → f ω φ x < f ω φ y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_increasing_l138_13893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pear_distribution_equation_l138_13815

/-- Represents the number of children -/
def x : ℕ → ℕ := λ n => n

/-- The total number of pears when each child gets 4 and 12 are left over -/
def total_pears_scenario1 (n : ℕ) : ℕ := 4 * x n + 12

/-- The total number of pears when each child gets 6 and none are left over -/
def total_pears_scenario2 (n : ℕ) : ℕ := 6 * x n

/-- Theorem stating that the equation 4x + 12 = 6x correctly represents the pear distribution scenario -/
theorem pear_distribution_equation (n : ℕ) : total_pears_scenario1 n = total_pears_scenario2 n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pear_distribution_equation_l138_13815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_l138_13856

/-- A geometric sequence with sum of first n terms S_n = 3^(n-1) + t has t = -1/3 --/
theorem geometric_sequence_sum (a : ℕ → ℝ) (t : ℝ) : 
  (∀ n : ℕ, ∃ r : ℝ, a (n + 1) = r * a n) →  -- geometric sequence condition
  (∀ n : ℕ, (Finset.range n).sum (λ i => a (i + 1)) = 3^(n - 1) + t) →  -- sum condition
  t = -1/3 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_l138_13856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dividend_ratio_is_46_55_l138_13800

/-- Represents the dividend calculation for a company --/
structure DividendCalculation where
  expectedEarnings : ℚ
  additionalDividendRate : ℚ
  additionalEarningsThreshold : ℚ
  actualEarnings : ℚ

/-- Calculates the ratio of dividend paid to total earnings --/
def dividendRatio (c : DividendCalculation) : ℚ × ℚ :=
  let additionalEarnings := c.actualEarnings - c.expectedEarnings
  let additionalDividend := (additionalEarnings / c.additionalEarningsThreshold).floor * c.additionalDividendRate
  let totalDividend := c.expectedEarnings + additionalDividend
  let ratio := (totalDividend * 100) / (c.actualEarnings * 100)
  (ratio.num, ratio.den)

/-- Theorem stating that the dividend ratio for the given conditions is 46:55 --/
theorem dividend_ratio_is_46_55 :
  let c : DividendCalculation := {
    expectedEarnings := 4/5,
    additionalDividendRate := 1/25,
    additionalEarningsThreshold := 1/10,
    actualEarnings := 11/10
  }
  dividendRatio c = (46, 55) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dividend_ratio_is_46_55_l138_13800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parallel_to_intersection_of_planes_l138_13876

-- Define the necessary structures
structure Line where

structure Plane where

-- Define the parallel relation between a line and a plane
def Line.parallelTo (l : Line) (p : Plane) : Prop := sorry

-- Define the parallel relation between two lines
def Line.parallelTo' (l1 l2 : Line) : Prop := sorry

-- Define the intersection of two planes
def Plane.intersection (p1 p2 : Plane) : Line := sorry

-- Theorem statement
theorem line_parallel_to_intersection_of_planes 
  (a : Line) (α β : Plane) (b : Line)
  (h1 : a.parallelTo α)
  (h2 : a.parallelTo β)
  (h3 : Plane.intersection α β = b) :
  Line.parallelTo' a b :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parallel_to_intersection_of_planes_l138_13876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l138_13870

/-- An arithmetic sequence with first term 2 and second sum equal to third term -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  first_term : a 1 = 2
  second_sum_eq_third : a 1 + a 2 = a 3

/-- Sum of first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  seq.a 2 = 4 ∧ sum_n seq 10 = 110 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l138_13870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_l138_13882

theorem sum_remainder (a b c : ℕ) : 
  a % 53 = 31 → 
  b % 53 = 17 → 
  c % 53 = 8 → 
  5 ∣ a → 
  (a + b + c) % 53 = 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_l138_13882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_iff_nonzero_imag_part_l138_13851

-- Define what it means for a complex number to be imaginary
def is_imaginary (z : ℂ) : Prop :=
  ∃ b : ℝ, z = Complex.I * b

-- State the theorem
theorem imaginary_iff_nonzero_imag_part (a b : ℝ) :
  is_imaginary (a + b * Complex.I) ↔ b ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_iff_nonzero_imag_part_l138_13851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_peters_soda_purchase_l138_13858

/-- The problem of calculating the number of ounces of soda Peter bought --/
def soda_purchase (cost_per_ounce : ℚ) (discount_rate : ℚ) (tax_rate : ℚ) 
  (money_brought : ℚ) (money_left : ℚ) : ℕ :=
  let money_spent := money_brought - money_left
  let discounted_price := cost_per_ounce * (1 - discount_rate)
  let final_price := discounted_price * (1 + tax_rate)
  let ounces := money_spent / final_price
  (Int.floor ounces).toNat

/-- The specific instance of the soda purchase problem --/
theorem peters_soda_purchase : 
  soda_purchase (25/100) (1/10) (2/25) 2 (1/2) = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_peters_soda_purchase_l138_13858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_sum_of_endpoints_l138_13859

noncomputable def f (x : ℝ) : ℝ := 3 / (3 + 9 * x^2)

theorem range_of_f :
  ∀ y, y ∈ Set.range f → 0 < y ∧ y ≤ 1 :=
by sorry

theorem sum_of_endpoints :
  ∃ a b, (∀ y, y ∈ Set.range f ↔ a < y ∧ y ≤ b) ∧ a + b = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_sum_of_endpoints_l138_13859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_length_l138_13881

/-- A rectangular floor with a length 450% more than its breadth -/
structure RectangularFloor where
  breadth : ℝ
  length : ℝ
  length_eq : length = 5.5 * breadth

/-- The cost of painting the floor -/
def paintCost : ℝ := 1360

/-- The rate of painting per square meter -/
def paintRate : ℝ := 5

/-- The theorem stating the length of the floor -/
theorem floor_length (floor : RectangularFloor) : 
  ∃ ε > 0, |floor.length - 38.665| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_length_l138_13881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_ratios_l138_13897

theorem square_area_ratios (x : ℝ) (hx : x > 0) : 
  (x^2 / (5*x)^2 = 1 / 25) ∧ ((2*x)^2 / (5*x)^2 = 4 / 25) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_ratios_l138_13897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_and_range_l138_13822

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ -1 then
    (1/2)^x - 2
  else
    (x - 2) * (|x| - 1)

theorem f_composition_and_range :
  (f (f (-2)) = 0) ∧
  (∀ x : ℝ, f x ≥ 2 ↔ x ≥ 3 ∨ x = 0) := by
  sorry

#check f_composition_and_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_and_range_l138_13822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_l138_13878

-- Define the parabola C
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

-- Define points M and N
def M : ℝ × ℝ := (1, 2)
def N : ℝ × ℝ := (5, -2)

-- Define line l
def line_l (x y : ℝ) : Prop := x - y - 7 = 0

-- Define the perpendicularity of MN and AB
def perpendicular (A B : ℝ × ℝ) : Prop :=
  (N.1 - M.1) * (B.1 - A.1) + (N.2 - M.2) * (B.2 - A.2) = 0

-- Define a point being on a circle with diameter AB
def on_circle_with_diameter (P A B : ℝ × ℝ) : Prop :=
  (P.1 - A.1) * (B.1 - P.1) + (P.2 - A.2) * (B.2 - P.2) = 0

theorem parabola_line_intersection (p : ℝ) (A B : ℝ × ℝ) :
  parabola p M.1 M.2 →
  line_l N.1 N.2 →
  parabola p A.1 A.2 →
  parabola p B.1 B.2 →
  line_l A.1 A.2 →
  line_l B.1 B.2 →
  perpendicular A B →
  (∀ x y, line_l x y ↔ x - y - 7 = 0) ∧
  on_circle_with_diameter M A B :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_l138_13878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_coordinates_l138_13818

/-- The parabola y² = 4x with focus F(1, 0) -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 4 * p.1}

/-- The focus of the parabola -/
def F : ℝ × ℝ := (1, 0)

/-- Distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem parabola_point_coordinates :
  ∀ P : ℝ × ℝ, P ∈ Parabola → distance P F = 5 →
  (P.1 = 4 ∧ (P.2 = 4 ∨ P.2 = -4)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_coordinates_l138_13818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_three_similar_parts_l138_13899

theorem impossible_three_similar_parts :
  ∀ (x : ℝ), x > 0 →
  ¬∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
  a + b + c = x ∧
  (a ≤ Real.sqrt 2 * b ∧ b ≤ Real.sqrt 2 * a) ∧
  (a ≤ Real.sqrt 2 * c ∧ c ≤ Real.sqrt 2 * a) ∧
  (b ≤ Real.sqrt 2 * c ∧ c ≤ Real.sqrt 2 * b) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_three_similar_parts_l138_13899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_is_nine_l138_13854

-- Define the triangle ABC
structure EquilateralTriangle where
  side : ℝ
  side_positive : side > 0

-- Define the extended triangle A'B'C'
structure ExtendedTriangle (ABC : EquilateralTriangle) where
  AB' : ℝ
  BC' : ℝ
  CA' : ℝ
  AB'_def : AB' = 3 * ABC.side
  BC'_def : BC' = 5 * ABC.side
  CA'_def : CA' = 3 * ABC.side

-- Define the area ratio function
noncomputable def areaRatio (ABC : EquilateralTriangle) (A'B'C' : ExtendedTriangle ABC) : ℝ :=
  (A'B'C'.AB' * A'B'C'.BC' * A'B'C'.CA') / (ABC.side * ABC.side * ABC.side)

-- Theorem statement
theorem area_ratio_is_nine (ABC : EquilateralTriangle) (A'B'C' : ExtendedTriangle ABC) :
  areaRatio ABC A'B'C' = 9 := by
  -- Unfold the definition of areaRatio
  unfold areaRatio
  -- Substitute the values of AB', BC', and CA'
  rw [A'B'C'.AB'_def, A'B'C'.BC'_def, A'B'C'.CA'_def]
  -- Simplify the expression
  simp [mul_assoc, mul_comm, mul_left_comm]
  -- The rest of the proof
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_is_nine_l138_13854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l138_13850

noncomputable def f (x : ℝ) := Real.sin (x/2)^2 + Real.sqrt 3 * Real.sin (x/2) * Real.cos (x/2)

theorem f_properties :
  (∃ T > 0, ∀ x, f (x + T) = f x ∧ ∀ S ∈ Set.Ioo 0 T, ∃ y, f (y + S) ≠ f y) ∧
  (∀ x ∈ Set.Icc (π/2) π, f x ≤ 3/2) ∧
  (∀ x ∈ Set.Icc (π/2) π, f x ≥ 1) ∧
  (∃ x ∈ Set.Icc (π/2) π, f x = 3/2) ∧
  (∃ x ∈ Set.Icc (π/2) π, f x = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l138_13850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_alone_finish_l138_13867

-- Define the total work as a positive real number
variable (W : ℝ) (hW : W > 0)

-- Define the daily work rates of A and B
variable (A B : ℝ) (hA : A > 0) (hB : B > 0)

-- A and B together can finish the work in 40 days
def together_finish (A B W : ℝ) : Prop := (A + B) * 40 = W

-- A and B work together for 10 days, then A works alone for 6 days to finish
def actual_work (A B W : ℝ) : Prop := (A + B) * 10 + A * 6 = W

-- Theorem: If the above conditions are met, A alone can finish the job in 8 days
theorem a_alone_finish (h1 : together_finish A B W) (h2 : actual_work A B W) : A * 8 = W := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_alone_finish_l138_13867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_untouched_area_equals_72_sqrt_3_l138_13875

/-- The edge length of the regular tetrahedron -/
noncomputable def tetrahedron_edge : ℝ := 4 * Real.sqrt 6

/-- The radius of the small ball -/
def ball_radius : ℝ := 1

/-- The area of one face of the regular tetrahedron -/
noncomputable def face_area : ℝ := (tetrahedron_edge^2 * Real.sqrt 3) / 4

/-- The side length of the smaller equilateral triangle on each face that the ball can touch -/
noncomputable def small_triangle_side : ℝ := tetrahedron_edge - 2 * ball_radius

/-- The area of the smaller equilateral triangle on each face that the ball can touch -/
noncomputable def small_triangle_area : ℝ := (small_triangle_side^2 * Real.sqrt 3) / 4

/-- The untouched area on one face of the tetrahedron -/
noncomputable def untouched_area_per_face : ℝ := face_area - small_triangle_area

/-- The total untouched area for all four faces of the tetrahedron -/
noncomputable def total_untouched_area : ℝ := 4 * untouched_area_per_face

/-- Theorem: The total area of the container's inner wall that the small ball will never touch is 72√3 -/
theorem untouched_area_equals_72_sqrt_3 : total_untouched_area = 72 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_untouched_area_equals_72_sqrt_3_l138_13875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_4theta_value_l138_13869

open Real

theorem sin_4theta_value (θ : ℝ) (h : ∑' (n : ℕ), (sin θ) ^ (2 * n) = 3) : 
  sin (4 * θ) = -4/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_4theta_value_l138_13869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_circle_area_approximation_l138_13860

noncomputable section

/-- The area of a square constructed using a specific method on a circle --/
def square_area (r : ℝ) : ℝ := (16 / 5) * r^2

/-- The area of a circle --/
def circle_area (r : ℝ) : ℝ := Real.pi * r^2

/-- The theorem stating the approximation of the square's area to the circle's area --/
theorem square_circle_area_approximation (r : ℝ) (h : r > 0) :
  ∃ ε > 0, ε < 1/50 ∧ |square_area r - circle_area r| ≤ ε * circle_area r := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_circle_area_approximation_l138_13860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roque_walks_three_times_l138_13845

/-- Represents the number of times Roque walks to and from work per week -/
def W : ℕ := sorry

/-- Time it takes Roque to walk to work (one way) in hours -/
def walk_time : ℕ := 2

/-- Time it takes Roque to ride his bike to work (one way) in hours -/
def bike_time : ℕ := 1

/-- Number of times Roque rides his bike to and from work per week -/
def bike_trips : ℕ := 2

/-- Total time Roque spends commuting per week in hours -/
def total_time : ℕ := 16

theorem roque_walks_three_times :
  W * (2 * walk_time) + bike_trips * (2 * bike_time) = total_time → W = 3 := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roque_walks_three_times_l138_13845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_growth_l138_13886

/-- Calculates the future value of an investment with annual contributions -/
noncomputable def futureValue (principal : ℝ) (rate : ℝ) (years : ℕ) (contribution : ℝ) : ℝ :=
  principal * (1 + rate) ^ years + contribution * ((1 + rate) ^ years - 1) / rate

/-- Proves that the investment scenario results in the expected amount -/
theorem investment_growth :
  let principal : ℝ := 12000
  let rate : ℝ := 0.08
  let years : ℕ := 7
  let contribution : ℝ := 500
  let result := futureValue principal rate years contribution
  ⌊result⌋ = 25018 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_growth_l138_13886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_area_on_parabola_l138_13834

/-- Given two equilateral triangles with vertices at (0, 1), (a, a+1), and (b, b^2+1),
    the area of the larger triangle is 26√3 + 45. -/
theorem equilateral_triangle_area_on_parabola :
  ∀ a b : ℝ,
  let v1 : ℝ × ℝ := (0, 1)
  let v2 : ℝ × ℝ := (a, a + 1)
  let v3 : ℝ × ℝ := (b, b^2 + 1)
  let d12 := Real.sqrt ((v2.1 - v1.1)^2 + (v2.2 - v1.2)^2)
  let d23 := Real.sqrt ((v3.1 - v2.1)^2 + (v3.2 - v2.2)^2)
  let d31 := Real.sqrt ((v1.1 - v3.1)^2 + (v1.2 - v3.2)^2)
  let is_equilateral := d12 = d23 ∧ d23 = d31
  let area := (Real.sqrt 3 / 4) * d12^2
  is_equilateral → (∃ area_larger : ℝ, area ≤ area_larger ∧ area_larger = 26 * Real.sqrt 3 + 45) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_area_on_parabola_l138_13834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_range_l138_13873

theorem triangle_ratio_range (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  0 < B → B < Real.pi / 2 →
  Real.sin B * Real.sin B = 8 * Real.sin A * Real.sin C →
  ∃ (x : ℝ), Real.sqrt 6 / 3 < x ∧ x < 2 * Real.sqrt 5 / 5 ∧ b / (a + c) = x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_range_l138_13873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_solution_sets_l138_13814

-- Define the solution set type
def SolutionSet := Set ℝ

-- Define the quadratic function type
def QuadraticFunction := ℝ → ℝ → ℝ → ℝ → ℝ

-- Define the first quadratic function
def f : QuadraticFunction := fun a b c x ↦ a * x^2 + b * x + c

-- Define the second quadratic function
def g : QuadraticFunction := fun a b c x ↦ a * x^2 + (a + 2*b) * x - c

-- State the theorem
theorem quadratic_inequality_solution_sets 
  (a b c : ℝ) 
  (h1 : {x | f a b c x > 0} = Set.Ioo (-2) 1) :
  {x | g a b c x < 0} = {x | x < -2 ∨ x > -1} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_solution_sets_l138_13814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_section_length_l138_13883

noncomputable def golden_ratio : ℝ := (1 + Real.sqrt 5) / 2

def is_golden_section_point (a b p : ℝ) : Prop :=
  (b - a) / (p - a) = golden_ratio ∧ (p - a) / (b - p) = golden_ratio

theorem golden_section_length (a b p : ℝ) :
  is_golden_section_point a b p → (p - a) > (b - p) → b - a = 2 →
  b - p = 3 - Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_section_length_l138_13883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_n_formula_l138_13874

/-- A positive geometric sequence with specific properties -/
structure PositiveGeometricSequence where
  a : ℕ → ℝ
  is_positive : ∀ n, a n > 0
  is_geometric : ∃ q : ℝ, q > 0 ∧ ∀ n, a (n + 1) = q * a n
  a2_eq_2 : a 2 = 2
  s3_condition : (a 1) + (a 2) + (a 3) = 2 * (a 3) - 1

/-- Sum of the first n terms of a geometric sequence -/
def sum_n (seq : PositiveGeometricSequence) (n : ℕ) : ℝ :=
  (Finset.range n).sum (fun i => seq.a (i + 1))

/-- The main theorem -/
theorem sum_n_formula (seq : PositiveGeometricSequence) :
  ∀ n : ℕ, sum_n seq n = 2^n - 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_n_formula_l138_13874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_inscribable_quadrilateral_properties_l138_13866

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : Point)

-- Define the properties of the quadrilateral
def is_cyclic (Q : Quadrilateral) : Prop := sorry
def is_inscribable (Q : Quadrilateral) : Prop := sorry

-- Define the given measurements
def side_AB (Q : Quadrilateral) : ℝ := sorry
def diagonal_AC (Q : Quadrilateral) : ℝ := sorry
def angle_ADC (Q : Quadrilateral) : ℝ := sorry
def side_CD (Q : Quadrilateral) : ℝ := sorry

-- Define the other sides and angles
def side_BC (Q : Quadrilateral) : ℝ := sorry
def side_DA (Q : Quadrilateral) : ℝ := sorry
def angle_ABC (Q : Quadrilateral) : ℝ := sorry

-- Theorem statement
theorem cyclic_inscribable_quadrilateral_properties
  (Q : Quadrilateral)
  (h_cyclic : is_cyclic Q)
  (h_inscribable : is_inscribable Q) :
  (angle_ABC Q = 180 - angle_ADC Q) ∧
  (side_AB Q + side_CD Q = side_BC Q + side_DA Q) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_inscribable_quadrilateral_properties_l138_13866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_general_solution_l138_13824

/-- The system of functions x₁(t) and x₂(t) -/
noncomputable def x₁ (C₁ C₂ t : ℝ) : ℝ := C₁ * Real.exp (-t) + C₂ * Real.exp (3 * t)
noncomputable def x₂ (C₁ C₂ t : ℝ) : ℝ := 2 * C₁ * Real.exp (-t) - 2 * C₂ * Real.exp (3 * t)

/-- The system of differential equations -/
def diff_eq₁ (x₁ x₂ : ℝ → ℝ) : Prop :=
  ∀ t, deriv x₁ t = x₁ t - x₂ t
def diff_eq₂ (x₁ x₂ : ℝ → ℝ) : Prop :=
  ∀ t, deriv x₂ t = x₂ t - 4 * x₁ t

/-- The theorem stating that x₁ and x₂ form a general solution -/
theorem general_solution (C₁ C₂ : ℝ) :
  diff_eq₁ (x₁ C₁ C₂) (x₂ C₁ C₂) ∧ diff_eq₂ (x₁ C₁ C₂) (x₂ C₁ C₂) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_general_solution_l138_13824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_calculation_l138_13879

-- Define the force function
def F (x : ℝ) : ℝ := 3 * x^2

-- Define the work done by integrating the force function
noncomputable def work_done (a b : ℝ) : ℝ := ∫ x in a..b, F x

-- Theorem statement
theorem work_calculation :
  work_done 0 4 = 64 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_calculation_l138_13879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_product_l138_13847

-- Define the polynomials
noncomputable def p (x : ℝ) : ℝ := 3 * x^3 - 9 * x^2 - 6 * x + 54
noncomputable def q (x : ℝ) : ℝ := 4 * x^3 + 8 * x^2 - 16 * x + 32

-- Define the sum of roots for a cubic polynomial
noncomputable def sum_of_roots (a b c d : ℝ) : ℝ := -b / a

-- Theorem statement
theorem sum_of_roots_product :
  sum_of_roots 3 (-9) (-6) 54 + sum_of_roots 4 8 (-16) 32 = 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_product_l138_13847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_ratio_theorem_l138_13828

noncomputable def f (x : ℝ) : ℝ := (x^2 + 1) / (2 * x)

noncomputable def f_iter : ℕ → (ℝ → ℝ)
  | 0 => id
  | n + 1 => f ∘ f_iter n

theorem f_ratio_theorem (n : ℕ) (x : ℝ) (hx : x ≠ -1 ∧ x ≠ 0 ∧ x ≠ 1) :
  (f_iter n x) / (f_iter (n + 1) x) = 1 + 1 / f (((x + 1) / (x - 1))^(2 * n)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_ratio_theorem_l138_13828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_meet_time_l138_13817

/-- Calculates the time for two trains to meet given their lengths, initial distance, and speeds. -/
noncomputable def time_to_meet (length1 length2 initial_distance : ℝ) (speed1 speed2 : ℝ) : ℝ :=
  let total_distance := initial_distance + length1 + length2
  let relative_speed := speed1 + speed2
  total_distance / relative_speed

/-- Theorem stating that the time for two trains to meet is approximately 18.75 seconds. -/
theorem trains_meet_time :
  let length1 := (150 : ℝ)
  let length2 := (250 : ℝ)
  let initial_distance := (850 : ℝ)
  let speed1 := (110 * 1000 / 3600 : ℝ)
  let speed2 := (130 * 1000 / 3600 : ℝ)
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
    |time_to_meet length1 length2 initial_distance speed1 speed2 - 18.75| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_meet_time_l138_13817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_transitivity_l138_13820

-- Define the universe
variable {U : Type*} [LinearOrderedField U]

-- Define the plane α
variable (α : Set (U × U × U))

-- Define lines m and n
variable (m n : Set (U × U × U))

-- Define parallelism for lines
def parallel_lines (l1 l2 : Set (U × U × U)) : Prop := sorry

-- Define parallelism between a line and a plane
def parallel_line_plane (l : Set (U × U × U)) (p : Set (U × U × U)) : Prop := sorry

-- Define a line being outside of a plane
def line_outside_plane (l : Set (U × U × U)) (p : Set (U × U × U)) : Prop := sorry

-- Theorem statement
theorem parallel_transitivity 
  (h1 : line_outside_plane m α)
  (h2 : line_outside_plane n α)
  (h3 : parallel_lines m n)
  (h4 : parallel_line_plane m α) :
  parallel_line_plane n α := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_transitivity_l138_13820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sinusoidal_function_unique_form_l138_13811

/-- Given a sinusoidal function with specific properties, prove its unique form --/
theorem sinusoidal_function_unique_form (f : ℝ → ℝ) 
  (h_form : ∃ A ω φ k, f = λ x ↦ A * Real.sin (ω * x + φ) + k)
  (h_A_pos : ∃ A, A > 0 ∧ ∃ ω φ k, f = λ x ↦ A * Real.sin (ω * x + φ) + k)
  (h_ω_pos : ∃ ω, ω > 0 ∧ ∃ A φ k, f = λ x ↦ A * Real.sin (ω * x + φ) + k)
  (h_φ_bound : ∃ φ, |φ| < π/2 ∧ ∃ A ω k, f = λ x ↦ A * Real.sin (ω * x + φ) + k)
  (h_max : f 2 = 2)
  (h_min : f 8 = -4)
  (h_period : ∃ T, T > 0 ∧ ∀ x, f (x + T) = f x ∧ 8 - 2 = T/2) :
  f = λ x ↦ 3 * Real.sin ((π/6) * x + π/6) - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sinusoidal_function_unique_form_l138_13811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sine_equation_l138_13836

theorem cosine_sine_equation (x : ℝ) : 
  Real.cos x - 5 * Real.sin x = 2 → Real.sin x + 5 * Real.cos x = -28/13 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sine_equation_l138_13836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_nondivisibility_l138_13890

theorem polynomial_nondivisibility 
  (n : ℕ+) 
  (f : Polynomial ℤ) 
  (hf_deg : f.degree = n)
  (hf_no_int_roots : ∀ (k : ℤ), f.eval k ≠ 0) :
  ∃ (m : ℕ), m ≤ 3 * n ∧ ¬(f.eval (↑m) ∣ f.eval (↑m + 1)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_nondivisibility_l138_13890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l138_13804

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1/2 * x^2 - a*x - 3*Real.log x

theorem min_value_of_f (a : ℝ) :
  (∃ (x : ℝ), x > 0 ∧ x ≠ 3 ∧ (∀ (y : ℝ), y > 0 → f a y ≥ f a x)) →
  (∀ (y : ℝ), y > 0 → deriv (f a) y = 0 → y = 3) →
  (∃ (x : ℝ), x > 0 ∧ f a x = -3/2 - 3*Real.log 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l138_13804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_people_arrangement_problem_l138_13849

/-- The number of ways to arrange 7 people in a row. -/
def total_arrangements : ℕ := 7 * 6 * 5 * 4 * 3 * 2 * 1

/-- The number of ways to arrange 6 distinct elements in a row. -/
def arrangements_of_six : ℕ := 6 * 5 * 4 * 3 * 2 * 1

/-- The number of ways to arrange 5 distinct elements in a row. -/
def arrangements_of_five : ℕ := 5 * 4 * 3 * 2 * 1

/-- The number of ways to arrange 4 distinct elements in a row. -/
def arrangements_of_four : ℕ := 4 * 3 * 2 * 1

/-- The number of ways to arrange 3 distinct elements in a row. -/
def arrangements_of_three : ℕ := 3 * 2 * 1

/-- The number of ways to choose 2 positions from 6 available positions. -/
def choose_two_from_six : ℕ := 6 * 5

/-- The number of ways to choose 3 positions from 5 available positions. -/
def choose_three_from_five : ℕ := 5 * 4 * 3

theorem people_arrangement_problem :
  /- (1) Number of ways A and B can stand next to each other -/
  (arrangements_of_six * 2 = 1440) ∧
  /- (2) Number of ways A and B do not stand next to each other -/
  (total_arrangements - arrangements_of_six * 2 = 3600) ∧
  /- (3) Number of ways A, B, and C stand so that no two of them are next to each other -/
  (arrangements_of_four * choose_three_from_five = 1440) ∧
  /- (4) Number of ways A, B, and C stand so that at most two of them are not next to each other -/
  (total_arrangements - arrangements_of_five * arrangements_of_three = 4320) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_people_arrangement_problem_l138_13849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_patch_area_difference_l138_13880

theorem patch_area_difference : ∃ (alan_length alan_width betty_length betty_width : ℝ),
  alan_length = 30 ∧
  alan_width = 50 ∧
  betty_length = 35 ∧
  betty_width = 40 ∧
  (alan_length * alan_width) - (betty_length * betty_width) = 100 := by
  use 30, 50, 35, 40
  simp [mul_comm]
  norm_num

#check patch_area_difference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_patch_area_difference_l138_13880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l138_13888

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x-1)

theorem problem_solution (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (f a 3 = 4 → a = 2) ∧
  (f a (Real.log a / Real.log 10) = 100 → a = 100 ∨ a = 1/10) ∧
  ((a > 1 → f a (Real.log (1/100) / Real.log 10) > f a (-2.1)) ∧
   (0 < a ∧ a < 1 → f a (Real.log (1/100) / Real.log 10) < f a (-2.1))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l138_13888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_minus_sin_value_l138_13835

theorem cos_minus_sin_value (θ : Real) 
  (h1 : Real.sin θ * Real.cos θ = 1/8) 
  (h2 : π/4 < θ) 
  (h3 : θ < π/2) : 
  Real.cos θ - Real.sin θ = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_minus_sin_value_l138_13835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_propositions_false_l138_13868

-- Define complex numbers
variable (z z₁ z₂ z₃ : ℂ)

-- Define real number x
variable (x : ℝ)

theorem all_propositions_false :
  -- Proposition 1
  (∃ a b : ℝ, a < b) ∧
  -- Proposition 2
  (z = Complex.I - 1 → z.re > 0 ∨ z.im > 0) ∧
  -- Proposition 3
  ((x^2 - 1 : ℝ) + (x^2 + 3*x + 2)*Complex.I = (0 : ℂ) → x ≠ 1 ∧ x ≠ -1) ∧
  -- Proposition 4
  (∃ z₁ z₂ z₃ : ℂ, (z₁ - z₂)^2 + (z₂ - z₃)^2 = 0 ∧ (z₁ ≠ z₂ ∨ z₂ ≠ z₃)) :=
by
  sorry

#check all_propositions_false

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_propositions_false_l138_13868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_adult_ticket_cost_is_19_l138_13853

/-- The cost of an adult ticket for a play, given the following conditions:
  * Total receipts: $7200
  * Total attendance: 400
  * Number of adults: 280
  * Number of children: 120
  * Cost of children's ticket: $15
-/
def adult_ticket_cost : ℤ :=
  let total_receipts : ℕ := 7200
  let total_attendance : ℕ := 400
  let num_adults : ℕ := 280
  let num_children : ℕ := 120
  let child_ticket_cost : ℕ := 15
  
  let adult_total : ℕ := total_receipts - (num_children * child_ticket_cost)
  let adult_cost : ℚ := adult_total / num_adults
  
  Int.floor (adult_cost + 1/2)

theorem adult_ticket_cost_is_19 : adult_ticket_cost = 19 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_adult_ticket_cost_is_19_l138_13853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_l138_13855

/-- The distance from a point to a line in 2D space -/
noncomputable def distancePointToLine (x y a b c : ℝ) : ℝ :=
  |a * x + b * y + c| / Real.sqrt (a^2 + b^2)

/-- Theorem: The distance from the point (-1, 1) to the line x + y - 2 = 0 is √2 -/
theorem distance_point_to_line :
  distancePointToLine (-1) 1 1 1 (-2) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_l138_13855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_l138_13821

/-- The function f(x) defined as 1/(2^x - 1) + 1/2 --/
noncomputable def f (x : ℝ) : ℝ := 1 / (2^x - 1) + 1/2

/-- Theorem stating that f is an odd function --/
theorem f_is_odd : ∀ x : ℝ, f (-x) = -f x := by
  intro x
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_l138_13821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_divisible_l138_13863

/-- The smallest integer greater than (3 + √5)^(2n) -/
noncomputable def T (n : ℕ) : ℤ :=
  ⌈(3 + Real.sqrt 5) ^ (2 * n)⌉

/-- Theorem: For any positive integer n, T(n) is divisible by 2^(n+1) -/
theorem smallest_integer_divisible (n : ℕ) (hn : n > 0) :
  ∃ k : ℤ, T n = k * (2^(n+1)) := by
  sorry

#check smallest_integer_divisible

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_divisible_l138_13863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_of_equilateral_hyperbola_l138_13810

/-- An equilateral hyperbola is a hyperbola where the distances from the center to a vertex on each axis are equal. -/
structure EquilateralHyperbola where
  a : ℝ
  b : ℝ
  h : a = b

/-- The eccentricity of a hyperbola is the ratio of the distance from the center to a focus to the distance from the center to a vertex on the principal axis. -/
noncomputable def eccentricity (h : EquilateralHyperbola) : ℝ :=
  let c := Real.sqrt (2 * h.a ^ 2)
  c / h.a

/-- The eccentricity of an equilateral hyperbola is √2. -/
theorem eccentricity_of_equilateral_hyperbola (h : EquilateralHyperbola) :
  eccentricity h = Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_of_equilateral_hyperbola_l138_13810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_l138_13826

noncomputable section

-- Define the expression
noncomputable def expression : ℝ :=
  (Real.sqrt 3 - 1)^(2 - Real.sqrt 5) / (Real.sqrt 3 + 1)^(2 + Real.sqrt 5)

-- Define the simplified form
noncomputable def simplified_form : ℝ :=
  (28 - 16 * Real.sqrt 3) * 2^(-2 - Real.sqrt 5)

-- Theorem statement
theorem expression_simplification :
  expression = simplified_form :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_l138_13826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x0_value_l138_13861

-- Define the function f
def f (x : ℝ) : ℝ := x^3

-- State the theorem
theorem x0_value (x0 : ℝ) (h : (deriv f) x0 = 6) : x0 = Real.sqrt 2 ∨ x0 = -Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x0_value_l138_13861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_angle_of_inclination_l138_13813

noncomputable section

/-- The curve function -/
def f (x : ℝ) : ℝ := (1/2) * x^2 - 2

/-- The point P on the curve -/
def P : ℝ × ℝ := (1, -3/2)

/-- The angle of inclination in radians -/
def θ : ℝ := Real.pi/4

theorem tangent_line_angle_of_inclination :
  P.1 = 1 ∧ 
  P.2 = f P.1 ∧ 
  (∀ x, HasDerivAt f (x) x) →
  Real.tan θ = (deriv f) P.1 :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_angle_of_inclination_l138_13813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_general_term_l138_13872

noncomputable def mySequence (n : ℕ) : ℚ := (-1 : ℚ)^(n+1) / (n+1 : ℚ)

theorem sequence_general_term :
  (mySequence 1 = 1/2) ∧
  (mySequence 2 = -1/3) ∧
  (mySequence 3 = 1/4) ∧
  (mySequence 4 = -1/5) ∧
  (∀ n : ℕ, mySequence n = (-1 : ℚ)^(n+1) / (n+1 : ℚ)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_general_term_l138_13872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_C2_equation_slope_product_constant_l138_13894

-- Define the ellipses
def C1 (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1
def C2 (a b x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the conditions
def C2_conditions (a b : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ ∃ (x y : ℝ), C2 a b x y ∧ x^2 = 5 ∧ y = 0

-- Define the line intersecting C2
def line_intersects_C2 (a b : ℝ) : Prop :=
  ∃ (x1 y1 x2 y2 : ℝ),
    C2 a b x1 y1 ∧ C2 a b x2 y2 ∧
    y2 - y1 = x2 - x1 ∧
    (x1 + x2) / 2 = 2 ∧ (y1 + y2) / 2 = -1

-- Theorem 1: Equation of C2
theorem C2_equation (a b : ℝ) 
  (h1 : C2_conditions a b) 
  (h2 : line_intersects_C2 a b) : 
  a^2 = 10 ∧ b^2 = 5 := by
  sorry

-- Define points on C1 and C2
def point_on_C1 (x y : ℝ) : Prop := C1 x y
def point_on_C2 (a b x y : ℝ) : Prop := C2 a b x y

-- Define the vector relation
def vector_relation (x0 y0 x1 y1 x2 y2 : ℝ) : Prop :=
  x0 = x1 + 2*x2 ∧ y0 = y1 + 2*y2

-- Theorem 2: Product of slopes
theorem slope_product_constant (a b : ℝ) 
  (h : a^2 = 10 ∧ b^2 = 5) :
  ∀ (x0 y0 x1 y1 x2 y2 : ℝ),
    point_on_C2 a b x0 y0 →
    point_on_C1 x1 y1 →
    point_on_C1 x2 y2 →
    vector_relation x0 y0 x1 y1 x2 y2 →
    (y1 / x1) * (y2 / x2) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_C2_equation_slope_product_constant_l138_13894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ordered_triples_count_l138_13857

theorem ordered_triples_count : 
  let S := {(a, b, c) : ℕ × ℕ × ℕ | 
    a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    a * b * c + 9 = a * b + b * c + c * a ∧ 
    a + b + c = 10}
  Finset.card (Finset.filter (fun (a, b, c) => 
    a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    a * b * c + 9 = a * b + b * c + c * a ∧ 
    a + b + c = 10) (Finset.range 11 ×ˢ Finset.range 11 ×ˢ Finset.range 11)) = 4 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ordered_triples_count_l138_13857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_with_divisible_pair_l138_13877

/-- The set of odd positive integers less than 30m that are not multiples of 5 -/
def S (m : ℕ) : Set ℕ :=
  {n : ℕ | n < 30 * m ∧ Odd n ∧ ¬(5 ∣ n)}

/-- A function that checks if for any subset of k elements from S,
    there exist two different elements where one divides the other -/
def has_divisible_pair (m k : ℕ) : Prop :=
  ∀ T : Finset ℕ, ↑T ⊆ S m → T.card = k →
    ∃ a b : ℕ, a ∈ T ∧ b ∈ T ∧ a ≠ b ∧ a ∣ b

/-- The smallest k such that any subset of k elements from S
    contains two different elements where one divides the other -/
theorem smallest_k_with_divisible_pair (m : ℕ) :
  ∃ k : ℕ, has_divisible_pair m k ∧ 
    (∀ j : ℕ, j < k → ¬(has_divisible_pair m j)) ∧ 
    k = 8 * m + 1 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_with_divisible_pair_l138_13877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_intervals_l138_13864

noncomputable def f (x : ℝ) : ℝ := Real.sin (Real.pi / 3 - x / 2)

def is_monotonic_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x ≤ f y

theorem monotonic_increasing_intervals :
  (is_monotonic_increasing f (-2*Real.pi) (-Real.pi/3)) ∧
  (is_monotonic_increasing f (5*Real.pi/3) (2*Real.pi)) ∧
  (∀ a b, -2*Real.pi ≤ a ∧ a < b ∧ b ≤ 2*Real.pi →
    is_monotonic_increasing f a b →
    (a = -2*Real.pi ∧ b ≤ -Real.pi/3) ∨ (a ≥ 5*Real.pi/3 ∧ b ≤ 2*Real.pi)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_intervals_l138_13864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_division_l138_13865

/-- Given a segment AB of length 12 units and a point K on AB that is 5 units closer to A than to B,
    prove that AK = 3.5 units and BK = 8.5 units. -/
theorem segment_division (A B K : ℝ) : 
  B - A = 12 →
  K ∈ Set.Icc A B →
  (B - K) - (K - A) = 5 →
  K - A = 3.5 ∧ B - K = 8.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_division_l138_13865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_SRQ_measure_l138_13837

-- Define the triangle RSQ
structure Triangle (R S Q : EuclideanSpace ℝ (Fin 2)) where
  r_ne_s : R ≠ S
  r_ne_q : R ≠ Q
  s_ne_q : S ≠ Q

-- Define parallel lines
def ParallelLines (l k : Set (EuclideanSpace ℝ (Fin 2))) : Prop := sorry

-- Define perpendicular line to side
def PerpendicularToSide (RQ k : Set (EuclideanSpace ℝ (Fin 2))) : Prop := sorry

-- Define angle measure
def AngleMeasure (A B C : EuclideanSpace ℝ (Fin 2)) : ℝ := sorry

-- Theorem statement
theorem angle_SRQ_measure 
  (R S Q : EuclideanSpace ℝ (Fin 2)) 
  (RSQ : Triangle R S Q)
  (l k : Set (EuclideanSpace ℝ (Fin 2)))
  (RQ : Set (EuclideanSpace ℝ (Fin 2)))
  (h1 : ParallelLines l k)
  (h2 : PerpendicularToSide RQ k)
  (h3 : AngleMeasure R S Q = 130) :
  AngleMeasure S R Q = 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_SRQ_measure_l138_13837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_8_256_l138_13844

theorem log_8_256 : Real.log 256 / Real.log 8 = 8 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_8_256_l138_13844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_distance_l138_13801

/-- A hyperbola with given asymptotes and a point it passes through -/
structure Hyperbola where
  /-- First asymptote: y = 2x + 3 -/
  asymptote1 : ℝ → ℝ := λ x => 2 * x + 3
  /-- Second asymptote: y = -2x + 1 -/
  asymptote2 : ℝ → ℝ := λ x => -2 * x + 1
  /-- Point the hyperbola passes through -/
  point : ℝ × ℝ := (4, 5)

/-- The distance between the foci of the hyperbola -/
noncomputable def focalDistance (h : Hyperbola) : ℝ := 2 * Real.sqrt 113 / 9

/-- Theorem stating that the distance between the foci of the given hyperbola is 2√113/9 -/
theorem hyperbola_focal_distance (h : Hyperbola) : 
  focalDistance h = 2 * Real.sqrt 113 / 9 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_distance_l138_13801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_P_to_AD_l138_13805

noncomputable section

-- Define the square ABCD
def A : ℝ × ℝ := (0, 4)
def B : ℝ × ℝ := (4, 4)
def C : ℝ × ℝ := (4, 0)
def D : ℝ × ℝ := (0, 0)

-- Define M as the midpoint of CD
def M : ℝ × ℝ := (2, 0)

-- Define the circles
def circle_M (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4
def circle_A (x y : ℝ) : Prop := x^2 + (y - 4)^2 = 16

-- Define point P as the intersection of the circles (other than D)
def P : ℝ × ℝ := (16/5, 8/5)

-- Theorem statement
theorem distance_P_to_AD :
  circle_M P.1 P.2 ∧ circle_A P.1 P.2 → P.2 = 8/5 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_P_to_AD_l138_13805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l138_13887

theorem equation_solution :
  ∃ y : ℝ, (1/8 : ℝ)^(3*y + 6) = (32 : ℝ)^(y + 4) ↔ y = -19/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l138_13887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_coplanar_implies_lambda_equals_one_l138_13807

def a : ℝ × ℝ × ℝ := (2, -1, 3)
def b : ℝ × ℝ × ℝ := (-1, 4, -2)
def c (l : ℝ) : ℝ × ℝ × ℝ := (1, 3, l)

theorem vectors_coplanar_implies_lambda_equals_one :
  ∀ (l : ℝ), (∃ (m n : ℝ), c l = m • a + n • b) → l = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_coplanar_implies_lambda_equals_one_l138_13807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_repeating_block_length_l138_13896

/-- The length of the smallest repeating block in the decimal expansion of 7/13 -/
def repeating_block_length : ℕ := 6

/-- The fraction we're considering -/
def fraction : ℚ := 7 / 13

/-- Function to calculate the period of a repeating decimal -/
noncomputable def repeating_decimal_period (q : ℚ) : ℕ := sorry

/-- Theorem stating that the repeating_block_length is equal to the period of the decimal expansion of fraction -/
theorem fraction_repeating_block_length :
  repeating_block_length = repeating_decimal_period fraction := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_repeating_block_length_l138_13896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_operations_correct_l138_13848

/-- Represents the position of a letter in the circle -/
inductive Position
| Center
| Outer (index : Fin 6)

/-- Represents a letter from A to G -/
inductive Letter
| A | B | C | D | E | F | G

/-- Represents an arrangement of letters in the circle -/
def Arrangement := Position → Letter

/-- Represents a valid rotation operation -/
structure Rotation :=
(pos1 pos2 pos3 : Position)
(is_valid : pos1 ≠ pos2 ∧ pos2 ≠ pos3 ∧ pos3 ≠ pos1 ∧ 
            (pos1 = Position.Center ∨ pos2 = Position.Center ∨ pos3 = Position.Center))

/-- The initial arrangement of letters -/
def initial_arrangement : Arrangement := sorry

/-- The final arrangement of letters -/
def final_arrangement : Arrangement := sorry

/-- Applies a rotation to an arrangement -/
def apply_rotation (a : Arrangement) (r : Rotation) : Arrangement := sorry

/-- The minimum number of operations needed -/
def min_operations : ℕ := 3

theorem min_operations_correct :
  ∀ (ops : List Rotation),
    (ops.foldl apply_rotation initial_arrangement = final_arrangement) →
    ops.length ≥ min_operations := by
  sorry

#check min_operations_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_operations_correct_l138_13848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_implies_a_equals_one_range_of_f_on_interval_line_intersection_implies_m_range_l138_13871

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 3*x - 1

-- Theorem 1: Given f(x) = x³ - 3ax - 1 has an extremum at x = -1, then a = 1
theorem extremum_implies_a_equals_one (a : ℝ) :
  (∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), x ≠ -1 ∧ |x + 1| < ε → 
    (x^3 - 3*a*x - 1) ≤ (-1)^3 - 3*a*(-1) - 1) →
  a = 1 :=
sorry

-- Theorem 2: The range of f(x) on the interval [-2, 3] is [-3, 17]
theorem range_of_f_on_interval :
  Set.range (fun x ↦ f x) ∩ Set.Icc (-2) 3 = Set.Icc (-3) 17 :=
sorry

-- Theorem 3: If y = 9x + m has three distinct common points with y = f(x), then m ∈ (-17, 15)
theorem line_intersection_implies_m_range :
  ∀ m : ℝ, (∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    f x₁ = 9*x₁ + m ∧ f x₂ = 9*x₂ + m ∧ f x₃ = 9*x₃ + m) →
  m ∈ Set.Ioo (-17) 15 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_implies_a_equals_one_range_of_f_on_interval_line_intersection_implies_m_range_l138_13871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_circle_arrangement_exists_l138_13830

/-- Represents a circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if two circles are touching -/
def areTouching (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c1.radius + c2.radius)^2

/-- Represents an arrangement of four circles -/
structure FourCircleArrangement where
  c1 : Circle
  c2 : Circle
  c3 : Circle
  c4 : Circle
  sameRadius : c1.radius = c2.radius ∧ c2.radius = c3.radius ∧ c3.radius = c4.radius

/-- The main theorem: There exists an arrangement of four circles such that
    a fifth circle of the same radius can touch all four simultaneously -/
theorem four_circle_arrangement_exists :
  ∃ (arr : FourCircleArrangement) (c5 : Circle),
    c5.radius = arr.c1.radius ∧
    areTouching c5 arr.c1 ∧
    areTouching c5 arr.c2 ∧
    areTouching c5 arr.c3 ∧
    areTouching c5 arr.c4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_circle_arrangement_exists_l138_13830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_unique_max_l138_13841

/-- The function f(x) = sin(x + π/3) -/
noncomputable def f (x : ℝ) : ℝ := Real.sin (x + Real.pi/3)

/-- Theorem: If f(x) = sin(x + π/3) has exactly one point with vertical coordinate 1 
    on the interval [0, m], then m = π/6 -/
theorem sin_unique_max (m : ℝ) : 
  (∃! x, x ∈ Set.Icc 0 m ∧ f x = 1) → m = Real.pi/6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_unique_max_l138_13841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equation_solution_l138_13838

theorem vector_equation_solution (m n : ℝ) : 
  let a : Fin 2 → ℝ := ![2, 1]
  let b : Fin 2 → ℝ := ![1, -2]
  (∀ i : Fin 2, m * a i + n * b i = ![9, -8] i) → m - n = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equation_solution_l138_13838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projectile_reaches_20m_at_0_8810s_l138_13809

/-- The height of a projectile as a function of time -/
noncomputable def projectile_height (t : ℝ) : ℝ := -4.2 * t^2 + 18.9 * t

/-- The time at which the projectile reaches a specific height -/
noncomputable def time_at_height (h : ℝ) : ℝ :=
  let a := -4.2
  let b := 18.9
  let c := -h
  (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)

/-- Theorem stating that the projectile reaches 20 meters at approximately 0.8810 seconds -/
theorem projectile_reaches_20m_at_0_8810s :
  |time_at_height 20 - 0.8810| < 0.0001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_projectile_reaches_20m_at_0_8810s_l138_13809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_l_proof_trajectory_Q_proof_l138_13808

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the line l
def line_l (x y : ℝ) : Prop := (3*x - 4*y + 5 = 0) ∨ (x = 1)

-- Define the point P
def point_P : ℝ × ℝ := (1, 2)

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Define the trajectory of point Q
def trajectory_Q (x y : ℝ) : Prop := x^2/4 + y^2/16 = 1 ∧ y ≠ 0

theorem line_l_proof :
  ∀ A B : ℝ × ℝ,
  circle_C A.1 A.2 → circle_C B.1 B.2 →
  line_l A.1 A.2 → line_l B.1 B.2 →
  line_l point_P.1 point_P.2 →
  distance A.1 A.2 B.1 B.2 = 2 * Real.sqrt 3 →
  line_l A.1 A.2 ∧ line_l B.1 B.2 := by sorry

theorem trajectory_Q_proof :
  ∀ M N Q : ℝ × ℝ,
  circle_C M.1 M.2 →
  N.1 = 0 →
  N.2 = M.2 →
  Q.1 = M.1 + N.1 →
  Q.2 = M.2 + N.2 →
  trajectory_Q Q.1 Q.2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_l_proof_trajectory_Q_proof_l138_13808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_crossing_time_l138_13852

/-- The time it takes for two trains traveling in opposite directions to cross each other -/
noncomputable def time_to_cross (speed1 speed2 : ℝ) (crossing_time : ℝ) : ℝ :=
  let distance := speed1 * crossing_time / 3600
  let relative_speed := speed1 + speed2
  distance / (relative_speed / 3600)

/-- The theorem stating the time it takes for the trains to cross each other -/
theorem trains_crossing_time :
  let speed1 := (90 : ℝ) -- km/h
  let speed2 := (120 : ℝ) -- km/h
  let crossing_time := (5 : ℝ) -- seconds
  abs (time_to_cross speed1 speed2 crossing_time - 2.14) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_crossing_time_l138_13852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertex_on_line_l138_13831

/-- The vertex of the parabola y = x^2 + 2mx + m^2 + m - 1 lies on the line y = -x - 1 for all real m -/
theorem parabola_vertex_on_line :
  ∀ (m : ℝ), 
  let f : ℝ → ℝ := λ x => x^2 + 2*m*x + m^2 + m - 1
  let vertex : ℝ × ℝ := (- m, f (- m))
  vertex.1 + vertex.2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertex_on_line_l138_13831

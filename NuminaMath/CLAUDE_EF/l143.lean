import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l143_14364

def sequenceA (n : ℕ) : ℕ := 2^n + 49

theorem unique_solution :
  ∃! n : ℕ,
    ∃ p q r s : ℕ,
      sequenceA n = p * q ∧
      sequenceA (n + 1) = r * s ∧
      Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ Nat.Prime s ∧
      p < q ∧ r < s ∧
      q - p = s - r :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l143_14364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_Q_l143_14354

/-- The original ellipse -/
def original_ellipse (x y : ℝ) : Prop := x^2/16 + y^2/9 = 1

/-- The locus ellipse -/
def locus_ellipse (x y : ℝ) : Prop := x^2/16 + y^2/(256/9) = 1

/-- Point A on the x-axis -/
def point_A : ℝ × ℝ := (4, 0)

/-- Point A' on the x-axis -/
def point_A' : ℝ × ℝ := (-4, 0)

/-- A point P on the original ellipse -/
noncomputable def point_P (θ : ℝ) : ℝ × ℝ := (4 * Real.cos θ, 3 * Real.sin θ)

/-- Line AP -/
def line_AP (θ : ℝ) (x y : ℝ) : Prop :=
  y = (3 * Real.sin θ) / (4 * Real.cos θ - 4) * (x - 4)

/-- Line A'P -/
def line_A'P (θ : ℝ) (x y : ℝ) : Prop :=
  y = (3 * Real.sin θ) / (4 * Real.cos θ + 4) * (x + 4)

/-- Line AQ (perpendicular to AP) -/
def line_AQ (θ : ℝ) (x y : ℝ) : Prop :=
  y = -(4 * Real.cos θ - 4) / (3 * Real.sin θ) * (x - 4)

/-- Line A'Q (perpendicular to A'P) -/
def line_A'Q (θ : ℝ) (x y : ℝ) : Prop :=
  y = -(4 * Real.cos θ + 4) / (3 * Real.sin θ) * (x + 4)

/-- The main theorem -/
theorem locus_of_Q :
  ∀ θ x y : ℝ,
  original_ellipse (point_P θ).1 (point_P θ).2 →
  line_AQ θ x y →
  line_A'Q θ x y →
  locus_ellipse x y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_Q_l143_14354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_M_equals_closed_interval_one_to_infinity_l143_14358

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := 1 / Real.sqrt (1 - x)
noncomputable def g (x : ℝ) : ℝ := Real.log (1 + x)

-- Define the domains M and N
def M : Set ℝ := {x | x < 1}
def N : Set ℝ := {x | x > -1}

-- State the theorem
theorem complement_M_equals_closed_interval_one_to_infinity :
  Set.compl M = Set.Ici 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_M_equals_closed_interval_one_to_infinity_l143_14358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_sum_proof_l143_14349

/-- The sum placed on simple interest -/
noncomputable def simple_interest_sum : ℝ := 1750

/-- The principal amount for compound interest calculation -/
noncomputable def compound_interest_principal : ℝ := 4000

/-- Number of years for simple interest -/
noncomputable def simple_interest_years : ℝ := 3

/-- Number of years for compound interest -/
noncomputable def compound_interest_years : ℝ := 2

/-- Annual interest rate for simple interest (in decimal form) -/
noncomputable def simple_interest_rate : ℝ := 0.08

/-- Annual interest rate for compound interest (in decimal form) -/
noncomputable def compound_interest_rate : ℝ := 0.10

/-- Number of times interest is compounded per year -/
noncomputable def compounding_frequency : ℝ := 1

/-- Calculate compound interest -/
noncomputable def compound_interest (p r n t : ℝ) : ℝ :=
  p * ((1 + r/n)^(n*t) - 1)

/-- Calculate simple interest -/
noncomputable def simple_interest (p r t : ℝ) : ℝ :=
  p * r * t

theorem simple_interest_sum_proof :
  simple_interest simple_interest_sum simple_interest_rate simple_interest_years =
  (1/2) * compound_interest compound_interest_principal compound_interest_rate compounding_frequency compound_interest_years :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_sum_proof_l143_14349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_better_fit_smaller_residuals_appropriate_model_from_residual_plot_narrower_band_higher_precision_accuracy_l143_14320

-- Define the linear regression model
structure LinearRegressionModel where
  x : ℝ → ℝ  -- Independent variable
  y : ℝ → ℝ  -- Dependent variable
  a : ℝ       -- Intercept
  b : ℝ       -- Slope
  e : ℝ → ℝ  -- Error term

-- Define the sum of squared residuals
noncomputable def sumSquaredResiduals (model : LinearRegressionModel) : ℝ :=
  ∫ (x : ℝ), (model.y x - (model.a + model.b * model.x x))^2

-- Define the fitting effect
def fittingEffect (model : LinearRegressionModel) : ℝ :=
  sorry -- Abstract measure of fitting effect

-- Define the residual plot properties
structure ResidualPlot where
  evenDistribution : Prop
  bandWidth : ℝ

-- Define the model appropriateness
def modelAppropriateness (plot : ResidualPlot) : Prop :=
  plot.evenDistribution

-- Define the fitting precision and prediction accuracy
def fittingPrecisionAndAccuracy (plot : ResidualPlot) : ℝ :=
  sorry -- Abstract measure of precision and accuracy

-- Theorem statements
theorem better_fit_smaller_residuals (model1 model2 : LinearRegressionModel) :
  sumSquaredResiduals model1 < sumSquaredResiduals model2 →
  fittingEffect model1 > fittingEffect model2 := by
  sorry

theorem appropriate_model_from_residual_plot (plot : ResidualPlot) :
  plot.evenDistribution → modelAppropriateness plot := by
  sorry

theorem narrower_band_higher_precision_accuracy (plot1 plot2 : ResidualPlot) :
  plot1.bandWidth < plot2.bandWidth →
  fittingPrecisionAndAccuracy plot1 > fittingPrecisionAndAccuracy plot2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_better_fit_smaller_residuals_appropriate_model_from_residual_plot_narrower_band_higher_precision_accuracy_l143_14320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_half_f_general_f_inverse_l143_14378

-- Define the function f
noncomputable def f (t : ℝ) : ℝ :=
  let x := (1 - t) / 2
  if x = 0 then 0 else (1 - x^2) / x^2

-- State the theorems
theorem f_at_half : f (1/2) = 15 := by sorry

theorem f_general (x : ℝ) (h : x ≠ 1) : f x = 4 / (x - 1)^2 - 1 := by sorry

theorem f_inverse (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 1) :
  f (1/x) = (4 * x^2) / (x - 1)^2 - 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_half_f_general_f_inverse_l143_14378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_card_value_decrease_l143_14311

theorem card_value_decrease (initial_value : ℝ) (h : initial_value > 0) : 
  (initial_value * (1 - 0.3) * (1 - 0.1) - initial_value) / initial_value = -0.37 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_card_value_decrease_l143_14311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tiles_along_length_l143_14366

-- Define the room dimensions
noncomputable def room_area : ℝ := 360
noncomputable def tile_size : ℝ := 8 / 12  -- 8 inches in feet

-- Define the relationship between length and width
noncomputable def length_width_relation (w : ℝ) : ℝ := 2 * w

-- Theorem statement
theorem tiles_along_length :
  ∃ (w : ℝ), w > 0 ∧ 
  room_area = length_width_relation w * w ∧
  (length_width_relation w / tile_size : ℝ) = 18 * Real.sqrt 5 := by
  sorry

#check tiles_along_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tiles_along_length_l143_14366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_Z_measure_l143_14351

-- Define the angle measure type
def AngleMeasure := ℝ

-- Define the concept of parallel lines
def Parallel (l1 l2 : Line) : Prop := sorry

-- Define the angle Z
def angleZ (p q : Line) (angleX angleY : AngleMeasure) : AngleMeasure := sorry

-- Theorem statement
theorem angle_Z_measure (p q : Line) (angleX angleY : AngleMeasure) :
  Parallel p q → angleX = (100 : ℝ) → angleY = (140 : ℝ) → angleZ p q angleX angleY = (120 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_Z_measure_l143_14351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_and_minimization_l143_14386

noncomputable def inequality_holds (m : ℝ) : Prop :=
  ∀ x : ℝ, |x - 3| + |x - m| ≥ 2 * m

noncomputable def max_m : ℝ := 1

def abc_constraint (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 1

noncomputable def quadratic_expr (a b c : ℝ) : ℝ :=
  4 * a^2 + 9 * b^2 + c^2

noncomputable def min_value : ℝ := 36 / 49

noncomputable def min_point : ℝ × ℝ × ℝ := (9/49, 4/49, 36/49)

theorem inequality_and_minimization :
  (∀ m : ℝ, inequality_holds m → m ≤ max_m) ∧
  (∀ a b c : ℝ, abc_constraint a b c →
    quadratic_expr a b c ≥ min_value) ∧
  (let (a, b, c) := min_point
   abc_constraint a b c ∧
   quadratic_expr a b c = min_value) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_and_minimization_l143_14386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_break_even_price_is_10_7_l143_14332

/-- Represents the lychee purchase and damage scenario -/
structure LycheePurchase where
  total_boxes : ℕ
  cost_per_kg : ℝ
  weight_per_box : ℝ
  sample_size : ℕ
  sample_mean : ℝ
  sample_mode : ℝ
  sample_median : ℝ

/-- Calculates the break-even selling price per kg -/
noncomputable def break_even_price (purchase : LycheePurchase) : ℝ :=
  let total_cost := purchase.cost_per_kg * (purchase.total_boxes : ℝ) * purchase.weight_per_box
  let damaged_weight := (purchase.total_boxes : ℝ) * (purchase.weight_per_box - purchase.sample_mode)
  let remaining_weight := (purchase.total_boxes : ℝ) * purchase.weight_per_box - damaged_weight
  total_cost / remaining_weight

/-- Theorem stating that the break-even price is approximately 10.7 units -/
theorem break_even_price_is_10_7 (purchase : LycheePurchase) 
  (h1 : purchase.total_boxes = 2000)
  (h2 : purchase.cost_per_kg = 10)
  (h3 : purchase.weight_per_box = 5)
  (h4 : purchase.sample_size = 20)
  (h5 : purchase.sample_mean = 4.75)
  (h6 : purchase.sample_mode = 4.7)
  (h7 : purchase.sample_median = 4.75) :
  ∃ ε > 0, |break_even_price purchase - 10.7| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_break_even_price_is_10_7_l143_14332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_foci_distance_product_l143_14301

/-- Given an ellipse and a hyperbola with the same foci, the product of the distances
    from any intersection point to the foci is constant. -/
theorem intersection_foci_distance_product (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  ∀ (x y : ℝ), 
    (x^2 / m^2 + y^2 / 16 = 1) → 
    (x^2 / n^2 - y^2 / 9 = 1) → 
    ∃ (f₁ f₂ : ℝ × ℝ), 
      (∀ (x' y' : ℝ), x'^2 / m^2 + y'^2 / 16 = 1 → 
        (Real.sqrt ((x' - f₁.1)^2 + (y' - f₁.2)^2) + Real.sqrt ((x' - f₂.1)^2 + (y' - f₂.2)^2) = 2 * m)) ∧
      (∀ (x' y' : ℝ), x'^2 / n^2 - y'^2 / 9 = 1 → 
        |Real.sqrt ((x' - f₁.1)^2 + (y' - f₁.2)^2) - Real.sqrt ((x' - f₂.1)^2 + (y' - f₂.2)^2)| = 2 * n) →
      Real.sqrt ((x - f₁.1)^2 + (y - f₁.2)^2) * Real.sqrt ((x - f₂.1)^2 + (y - f₂.2)^2) = 25 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_foci_distance_product_l143_14301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l143_14338

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (Real.cos x) ^ 2 - 4 * Real.sin x

-- State the theorem
theorem f_range : Set.range f = Set.Icc (-4 : ℝ) 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l143_14338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_variance_bound_min_std_dev_value_min_std_dev_legs_l143_14395

/-- A right triangle with hypotenuse 3 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  hypotenuse_eq : a^2 + b^2 = 9

/-- The variance of the side lengths of a right triangle -/
noncomputable def variance (t : RightTriangle) : ℝ :=
  6 - ((t.a + t.b + 3) / 3)^2

/-- The standard deviation of the side lengths of a right triangle -/
noncomputable def std_dev (t : RightTriangle) : ℝ :=
  Real.sqrt (variance t)

theorem right_triangle_variance_bound (t : RightTriangle) :
  variance t < 5 := by
  sorry

theorem min_std_dev_value :
  ∃ (t : RightTriangle), ∀ (s : RightTriangle), std_dev t ≤ std_dev s ∧ std_dev t = Real.sqrt (2 - 2*Real.sqrt 2 + 1) := by
  sorry

theorem min_std_dev_legs (t : RightTriangle) :
  (∀ (s : RightTriangle), std_dev t ≤ std_dev s) → t.a = 3 * Real.sqrt 2 / 2 ∧ t.b = 3 * Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_variance_bound_min_std_dev_value_min_std_dev_legs_l143_14395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_is_real_z_is_complex_z_not_purely_imaginary_l143_14350

-- Define the complex number z as a function of a
noncomputable def z (a : ℝ) : ℂ := (a^2 - 7*a + 6) / (a^2 - 1) + (a^2 - 5*a - 6) * Complex.I

-- Theorem for real number case
theorem z_is_real : ∀ a : ℝ, (z a).im = 0 ↔ a = 6 := by sorry

-- Theorem for complex number case
theorem z_is_complex : ∀ a : ℝ, (z a).re ≠ 0 ∧ (z a).im ≠ 0 ↔ 
  a < -1 ∨ (-1 < a ∧ a < 1) ∨ (1 < a ∧ a < 6) ∨ a > 6 := by sorry

-- Theorem for purely imaginary case
theorem z_not_purely_imaginary : ¬ ∃ a : ℝ, (z a).re = 0 ∧ (z a).im ≠ 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_is_real_z_is_complex_z_not_purely_imaginary_l143_14350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_ABD_area_is_four_l143_14382

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a quadrilateral pyramid -/
structure QuadPyramid where
  S : Point3D  -- apex
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- Represents a plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Checks if a point is the axis of symmetry for two other points -/
def isAxisOfSymmetry (a b c : Point3D) : Prop := sorry

/-- Checks if a plane intersects a pyramid forming a regular hexagon -/
def intersectsRegularHexagon (p : QuadPyramid) (α : Plane) (sideLength : ℝ) : Prop := sorry

/-- Checks if a plane intersects the midpoints of pyramid edges -/
def intersectsMidpoints (p : QuadPyramid) (α : Plane) : Prop := sorry

/-- Checks if a plane intersects specific segments of a pyramid -/
def intersectsSegments (α : Plane) (b b₁ d d₁ : Point3D) : Prop := sorry

/-- Theorem stating the area of triangle ABD in the given quadrilateral pyramid -/
theorem area_of_triangle_ABD (SABCD : QuadPyramid) (α : Plane) : ℝ := by
  let AC := 9  -- Length of diagonal AC
  let E : Point3D := sorry  -- Intersection of diagonals
  let AE_length : ℝ := sorry  -- Length of AE
  let EC_length : ℝ := AC - AE_length
  have h1 : AE_length < EC_length := sorry
  have h2 : isAxisOfSymmetry SABCD.A SABCD.C SABCD.B := sorry
  have h3 : intersectsRegularHexagon SABCD α 2 := sorry
  have h4 : intersectsMidpoints SABCD α := sorry
  have h5 : intersectsSegments α SABCD.B (sorry : Point3D) SABCD.D (sorry : Point3D) := sorry
  exact 4

/-- The area of triangle ABD is 4 -/
theorem area_is_four (SABCD : QuadPyramid) (α : Plane) :
    area_of_triangle_ABD SABCD α = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_ABD_area_is_four_l143_14382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_n_l143_14383

theorem no_valid_n : ¬∃ n : ℕ, 
  n > 0 ∧
  (100 ≤ n / 4 ∧ n / 4 ≤ 999) ∧ 
  (100 ≤ 4 * n ∧ 4 * n ≤ 999) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_n_l143_14383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_euler_totient_equals_divisor_count_l143_14314

/-- Euler's totient function -/
def φ : ℕ+ → ℕ := sorry

/-- Number of positive divisors function -/
def d : ℕ+ → ℕ := sorry

/-- Theorem stating that φ(n) = d(n) if and only if n is in the set {1, 3, 8, 10, 18, 24, 30} -/
theorem euler_totient_equals_divisor_count (n : ℕ+) :
  φ n = d n ↔ n ∈ ({1, 3, 8, 10, 18, 24, 30} : Set ℕ+) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_euler_totient_equals_divisor_count_l143_14314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l143_14310

-- Define the sets M and N
def M : Set ℝ := {x | (2 - x) / (x + 1) ≥ 0}
def N : Set ℝ := {y | ∃ x > 0, y = Real.log x}

-- State the theorem
theorem intersection_M_N : M ∩ N = Set.Ioo (-1 : ℝ) 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l143_14310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_divisor_problem_l143_14304

theorem second_divisor_problem (n d : ℕ) (h1 : n % 2 = 1) (h2 : n % d = 5) (h3 : (n + 5) % 10 = 0) : d = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_divisor_problem_l143_14304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_faulty_clock_ahead_l143_14326

/-- Represents a day of the week -/
inductive Day
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a time of day in hours (0-23) -/
def Time := Fin 24

structure ClockState where
  day : Day
  time : Time

/-- Advances the clock state by the given number of hours -/
def advanceTime (start : ClockState) (hours : Nat) : ClockState :=
  sorry

/-- Calculates the time lost by the faulty clock after a given number of actual hours -/
def timeLost (actualHours : Nat) : Nat :=
  actualHours / 4

theorem faulty_clock_ahead (start : ClockState) :
  start.day = Day.Monday ∧ start.time = ⟨12, by norm_num⟩ →
  ∃ (hours : Nat), 
    let end_state := advanceTime start hours
    timeLost hours = 23 ∧
    end_state.day = Day.Friday ∧ end_state.time = ⟨8, by norm_num⟩ :=
  by
    sorry

#eval timeLost 92

end NUMINAMATH_CALUDE_ERRORFEEDBACK_faulty_clock_ahead_l143_14326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_question_1_question_2_question_3_l143_14384

open Set

-- Define the universal set U as ℝ
def U : Set ℝ := univ

-- Define set A
def A : Set ℝ := {x | abs x ≤ 2}

-- Define set B
def B (a : ℝ) : Set ℝ := {x | x^2 - 2*a*x - 3*a^2 < 0}

-- Question 1
theorem question_1 : B 1 ∩ (U \ A) = {x | 2 < x ∧ x < 3} := by sorry

-- Question 2
theorem question_2 : ∀ a : ℝ, -6 ∈ B a → a < -2 ∨ a > 6 := by sorry

-- Question 3
theorem question_3 : ∀ a : ℝ, A ∪ B a = Ioc (-3) 2 → a = -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_question_1_question_2_question_3_l143_14384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_range_l143_14388

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (a * x^2 - x + 1/2)

-- State the theorem
theorem f_positive_range (a : ℝ) :
  (∀ x ∈ Set.Icc 1 2, f a x > 0) ↔ (1/2 < a ∧ a < 5/8) ∨ (a > 3/2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_range_l143_14388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_equality_root_properties_l143_14345

-- Part 1
def M (m : ℝ) : Set ℂ := {2, Complex.mk (m^2 - 2*m) (m^2 + m - 2)}
def P : Set ℂ := {-1, 2, Complex.I * 4}

theorem union_equality (m : ℝ) : M m ∪ P = P → m = 1 ∨ m = 2 := by sorry

-- Part 2
def quadratic_equation (a : ℝ) (x : ℂ) := x^2 + 4*x + (a : ℂ)

theorem root_properties (a : ℝ) : 
  quadratic_equation a (Complex.mk (-2) 1) = 0 → 
  a = 5 ∧ quadratic_equation a (Complex.mk (-2) (-1)) = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_equality_root_properties_l143_14345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_line_equation_l143_14399

-- Define the circle
def circle_eq (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 9

-- Define point P
def P : ℝ × ℝ := (1, 1)

-- Define the line equation
def line_equation (x y : ℝ) : Prop := 2*x - y - 1 = 0

-- Theorem statement
theorem chord_line_equation :
  ∀ M N : ℝ × ℝ,
  (∃ t : ℝ, M = (P.1 - t, P.2 - 2*t) ∧ N = (P.1 + t, P.2 + 2*t)) →  -- P is midpoint of MN
  circle_eq M.1 M.2 →  -- M is on the circle
  circle_eq N.1 N.2 →  -- N is on the circle
  ∀ x y : ℝ, (x, y) ∈ Set.Icc M N → line_equation x y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_line_equation_l143_14399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_one_coprime_l143_14343

def sequenceA (n : ℕ) : ℕ := 2^n + 3^n + 6^n - 1

theorem only_one_coprime : 
  ∀ k : ℕ, k > 0 → (∀ n : ℕ, n > 0 → Nat.Coprime k (sequenceA n)) → k = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_one_coprime_l143_14343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salt_solution_dilution_l143_14318

/-- Represents the amount of water added to dilute a salt solution -/
noncomputable def water_added (m : ℝ) : ℝ := 10 * m / (m + 20)

/-- Theorem stating the relationship between initial and final salt solutions -/
theorem salt_solution_dilution (m : ℝ) (h : m > 30) :
  let initial_salt := m * m / 100
  let final_volume := m + 2 * water_added m
  let final_concentration := (m - 20) / 100
  initial_salt = final_concentration * final_volume :=
by
  -- Introduce the local definitions
  have initial_salt := m * m / 100
  have final_volume := m + 2 * water_added m
  have final_concentration := (m - 20) / 100
  
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_salt_solution_dilution_l143_14318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l143_14333

/-- The eccentricity of a hyperbola with specific properties -/
theorem hyperbola_eccentricity (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) : 
  let e := c / a
  let hyperbola := fun (x y : ℝ) ↦ x^2 / a^2 - y^2 / b^2 = 1
  let F1 : ℝ × ℝ := (-c, 0)
  let F2 : ℝ × ℝ := (c, 0)
  let M : ℝ × ℝ := (0, Real.sqrt 3 * c)
  let midpoint : ℝ × ℝ := (-(c/2), (Real.sqrt 3 / 2) * c)
  hyperbola midpoint.1 midpoint.2 ∧ b^2 = c^2 - a^2 → e = 1 + Real.sqrt 3 :=
by
  sorry

#check hyperbola_eccentricity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l143_14333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l143_14390

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that under certain conditions, angle B and the area of the triangle
    have specific values. -/
theorem triangle_properties (a b c A B C : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
    (h_angles : 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π)
    (h_sin_cos : Real.sin A / (Real.cos B * Real.cos C) = 2 * Real.sqrt 3 * a^2 / (a^2 + b^2 - c^2))
    (h_sides : a + c = 2 * Real.sqrt 6 * Real.sin C)
    (h_b : b = 3) : B = π / 3 ∧ (1/2 * a * c * Real.sin B = 3 * Real.sqrt 3 / 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l143_14390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_lateral_surface_area_l143_14316

/-- The lateral surface area of a frustum of a right circular cone. -/
noncomputable def lateralSurfaceArea (r1 r2 h : Real) : Real :=
  let s := Real.sqrt ((r1 - r2)^2 + h^2)
  Real.pi * (r1 + r2) * s

/-- Theorem: The lateral surface area of a frustum of a right circular cone
    with lower base radius 8 cm, upper base radius 4 cm, and height 7 cm
    is equal to 12π√65 cm². -/
theorem frustum_lateral_surface_area :
  lateralSurfaceArea 8 4 7 = 12 * Real.pi * Real.sqrt 65 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_lateral_surface_area_l143_14316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l143_14329

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive : 0 < b ∧ b < a

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt ((h.a ^ 2 + h.b ^ 2) / h.a ^ 2)

/-- The left focus of a hyperbola -/
noncomputable def left_focus (h : Hyperbola) : ℝ × ℝ :=
  (-(Real.sqrt (h.a ^ 2 - h.b ^ 2)), 0)

/-- A line with slope 1 passing through a point -/
def line_slope_one (p : ℝ × ℝ) (x : ℝ) : ℝ :=
  x - p.1 + p.2

/-- The asymptotes of a hyperbola -/
def asymptotes (h : Hyperbola) (x : ℝ) : Set ℝ :=
  {y | y = h.b / h.a * x ∨ y = -h.b / h.a * x}

/-- Theorem: If a line with slope 1 passing through the left focus of a hyperbola
    intersects its asymptotes at points A and B such that |AF|/|BF| = 1/2,
    then the eccentricity of the hyperbola is √10 -/
theorem hyperbola_eccentricity (h : Hyperbola)
  (A B : ℝ × ℝ)
  (h_line : ∀ x, line_slope_one (left_focus h) x ∈ asymptotes h x)
  (h_AF_BF : (Real.sqrt ((A.1 - (left_focus h).1)^2 + (A.2 - (left_focus h).2)^2)) /
             (Real.sqrt ((B.1 - (left_focus h).1)^2 + (B.2 - (left_focus h).2)^2)) = 1/2) :
  eccentricity h = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l143_14329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_travel_fraction_is_one_fourth_l143_14324

-- Define the initial amount
noncomputable def initial_amount : ℝ := 1249.9999999999998

-- Define the fraction spent on clothes
def clothes_fraction : ℚ := 1/3

-- Define the fraction spent on food
def food_fraction : ℚ := 1/5

-- Define the final amount left
noncomputable def final_amount : ℝ := 500

-- Theorem to prove
theorem travel_fraction_is_one_fourth :
  let remaining_after_clothes := initial_amount * (1 - clothes_fraction)
  let remaining_after_food := remaining_after_clothes * (1 - food_fraction)
  let spent_on_travel := remaining_after_food - final_amount
  abs ((spent_on_travel / remaining_after_food) - (1/4 : ℝ)) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_travel_fraction_is_one_fourth_l143_14324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_parabola_equation_l143_14331

-- Ellipse problem
theorem ellipse_equation (e : Set (ℝ × ℝ)) :
  ((0 : ℝ), 5/3) ∈ e ∧ (1, 1) ∈ e →
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a ≠ b ∧
  e = {p : ℝ × ℝ | (p.1^2/a^2) + (p.2^2/b^2) = 1} →
  e = {p : ℝ × ℝ | (p.1^2/(25/16)) + (p.2^2/(25/9)) = 1} :=
sorry

-- Parabola problem
theorem parabola_equation (p : ℝ) (M : ℝ × ℝ) :
  p > 0 →
  M.2^2 = 2*p*M.1 →
  (M.1 + p/2)^2 + M.2^2 = 100 →
  M.2 = 6 ∨ M.2 = -6 →
  p = 2 ∨ p = 18 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_parabola_equation_l143_14331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bird_on_time_speed_l143_14313

/-- The speed (in mph) at which Mr. Bird arrives 5 minutes late -/
noncomputable def late_speed : ℝ := 50

/-- The speed (in mph) at which Mr. Bird arrives 5 minutes early -/
noncomputable def early_speed : ℝ := 70

/-- The time (in hours) that Mr. Bird is late when driving at late_speed -/
noncomputable def late_time : ℝ := 5 / 60

/-- The time (in hours) that Mr. Bird is early when driving at early_speed -/
noncomputable def early_time : ℝ := 5 / 60

/-- The speed (in mph) at which Mr. Bird arrives exactly on time -/
noncomputable def on_time_speed : ℝ := 55

theorem bird_on_time_speed :
  ∃ (d t : ℝ),
    d > 0 ∧ t > 0 ∧
    d = late_speed * (t + late_time) ∧
    d = early_speed * (t - early_time) ∧
    on_time_speed = d / t :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bird_on_time_speed_l143_14313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_equals_radius_l143_14307

/-- The maximum distance a satellite reaches from Earth's surface when launched vertically with the first cosmic velocity -/
def max_distance (g : ℝ) (R : ℝ) : ℝ :=
  R

/-- The first cosmic velocity -/
noncomputable def first_cosmic_velocity (g : ℝ) (R : ℝ) : ℝ :=
  Real.sqrt (g * R)

theorem max_distance_equals_radius (g R : ℝ) (hg : g > 0) (hR : R > 0) :
  max_distance g R = R :=
by
  -- The proof goes here
  sorry

#check max_distance_equals_radius

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_equals_radius_l143_14307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exp_function_convex_log2_concave_l143_14315

-- Part 1: Convexity of exponential function
noncomputable def exp_function (a : ℝ) (x : ℝ) : ℝ := a^x

theorem exp_function_convex (a : ℝ) (h : a > 0) :
  ConvexOn ℝ Set.univ (exp_function a) :=
by sorry

-- Part 2: Concavity of logarithm function
noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

theorem log2_concave :
  ConcaveOn ℝ (Set.Ioi 0) log2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exp_function_convex_log2_concave_l143_14315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_cosine_ratio_theorem_l143_14328

theorem sine_cosine_ratio_theorem (x y : ℝ) 
  (h1 : Real.sin x / Real.sin y = 4)
  (h2 : Real.cos x / Real.cos y = 1/2) :
  Real.sin (2*x) / Real.sin (2*y) + Real.cos (2*x) / Real.cos (2*y) = 27/19 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_cosine_ratio_theorem_l143_14328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_M_and_N_l143_14363

-- Define set M
def M : Set ℝ := {x | x^2 + x - 6 < 0}

-- Define set N
def N : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

-- Theorem statement
theorem intersection_of_M_and_N : M ∩ N = Set.Icc 1 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_M_and_N_l143_14363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_trapezoid_midsegment_l143_14391

/-- A trapezoid inscribed in a circle -/
structure InscribedTrapezoid where
  /-- The perimeter of the trapezoid -/
  perimeter : ℝ
  /-- The sum of the lengths of opposite sides is equal -/
  opposite_sides_sum_equal : ℝ

/-- The midsegment of a trapezoid -/
noncomputable def midsegment (t : InscribedTrapezoid) : ℝ := t.opposite_sides_sum_equal / 2

/-- Theorem: For a trapezoid inscribed in a circle with perimeter 40, its midsegment is 10 -/
theorem inscribed_trapezoid_midsegment (t : InscribedTrapezoid) 
  (h : t.perimeter = 40) : midsegment t = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_trapezoid_midsegment_l143_14391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_high_scorer_gender_relationship_l143_14396

/-- Represents the sample data and population information -/
structure SampleData where
  total_students : Nat
  total_boys : Nat
  sample_size : Nat
  high_scorers : Nat
  high_scorer_girls : Nat

/-- Calculates the chi-square statistic -/
noncomputable def chi_square (data : SampleData) : Real :=
  let a := data.high_scorers - data.high_scorer_girls
  let b := (data.sample_size * data.total_boys / data.total_students) - a
  let c := data.high_scorer_girls
  let d := data.sample_size - a - b - c
  let n := data.sample_size
  (n : Real) * (a * d - b * c)^2 / ((a + b) * (c + d) * (a + c) * (b + d))

/-- The critical value for 97.5% confidence level -/
def critical_value : Real := 5.024

/-- Theorem stating the relationship between high-scoring players and gender -/
theorem high_scorer_gender_relationship (data : SampleData) 
  (h1 : data.total_students = 1000)
  (h2 : data.total_boys = 400)
  (h3 : data.sample_size = 100)
  (h4 : data.high_scorers = 25)
  (h5 : data.high_scorer_girls = 10) : 
  chi_square data > critical_value := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_high_scorer_gender_relationship_l143_14396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_outstanding_nonmembers_probability_l143_14334

/-- The probability of outstanding students in the school -/
noncomputable def p_outstanding : ℝ := 0.1

/-- The proportion of students who are group members -/
noncomputable def p_members : ℝ := 0.2

/-- The probability of outstanding students among group members -/
noncomputable def p_outstanding_members : ℝ := 0.4

/-- The probability of outstanding students among non-members -/
noncomputable def p_outstanding_nonmembers : ℝ := (p_outstanding - p_members * p_outstanding_members) / (1 - p_members)

theorem outstanding_nonmembers_probability :
  p_outstanding_nonmembers = 0.025 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_outstanding_nonmembers_probability_l143_14334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_B_equal_set_C_equal_set_D_equal_set_A_not_equal_l143_14365

-- Define the functions for each set
noncomputable def f_A1 (x : ℝ) : ℝ := x
noncomputable def f_A2 (x : ℝ) : ℝ := Real.sqrt (x^2)

noncomputable def f_B1 (x : ℝ) : ℝ := 2 * x
noncomputable def f_B2 (x : ℝ) : ℝ := 2 * (x^3)^(1/3)

def f_C1 (x : ℝ) : ℝ := x^2 + x + 3
def f_C2 (t : ℝ) : ℝ := t^2 + t + 3

def f_D1 (x : ℝ) : ℝ := x^2
noncomputable def f_D2 (x : ℝ) : ℝ := Real.sqrt (x^4)

-- Theorem statements
theorem set_B_equal : ∀ x : ℝ, f_B1 x = f_B2 x := by sorry

theorem set_C_equal : ∀ x : ℝ, f_C1 x = f_C2 x := by sorry

theorem set_D_equal : ∀ x : ℝ, f_D1 x = f_D2 x := by sorry

-- Theorem to show that set A is not equal
theorem set_A_not_equal : ∃ x : ℝ, f_A1 x ≠ f_A2 x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_B_equal_set_C_equal_set_D_equal_set_A_not_equal_l143_14365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_half_mile_revolutions_l143_14380

/-- The number of revolutions a wheel makes when traveling a certain distance -/
noncomputable def revolutions (diameter : ℝ) (distance : ℝ) : ℝ :=
  distance / (Real.pi * diameter)

/-- Conversion factor from miles to feet -/
def milestoFeet : ℝ := 5280

theorem half_mile_revolutions (d : ℝ) (h : d = 8) :
  revolutions d (0.5 * milestoFeet) = 330 / Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_half_mile_revolutions_l143_14380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l143_14374

-- Define a triangle ABC
structure Triangle :=
  (A B C : Real)  -- Angles
  (a b c : Real)  -- Sides opposite to A, B, C respectively

-- Define the statements to prove
theorem triangle_properties (abc : Triangle) :
  (∀ (A B : Real), Real.sin A > Real.sin B → A > B) ∧ 
  (∀ (AC CB : Real × Real), AC.1 * CB.1 + AC.2 * CB.2 > 0 → abc.C > Real.pi / 2) ∧
  (abc.a^3 + abc.b^3 = abc.c^3 → abc.A < Real.pi / 2 ∧ abc.B < Real.pi / 2 ∧ abc.C < Real.pi / 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l143_14374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_solution_is_19_l143_14394

noncomputable def is_solution (x : ℕ) : Prop :=
  x > 1 ∧ Real.cos (x * Real.pi / 180) = Real.cos ((x^2) * Real.pi / 180)

theorem smallest_solution_is_19 :
  is_solution 19 ∧ ∀ x : ℕ, 1 < x → x < 19 → ¬ is_solution x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_solution_is_19_l143_14394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l143_14377

noncomputable def f (x : ℝ) : ℝ := Real.cos x ^ 4 - 2 * Real.sin x * Real.cos x - Real.sin x ^ 4

theorem f_properties :
  -- Smallest positive period is π
  (∃ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) ∧
    (∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q)) ∧
  -- Monotonically increasing intervals
  (∀ (k : ℤ), ∀ (x y : ℝ),
    -5*Real.pi/8 + k*Real.pi ≤ x ∧ x < y ∧ y ≤ -Real.pi/8 + k*Real.pi →
    f x < f y) ∧
  -- Set of x values where f(x) ≥ 1
  (∀ (x : ℝ), f x ≥ 1 ↔ ∃ (k : ℤ), -Real.pi/4 + k*Real.pi ≤ x ∧ x ≤ k*Real.pi) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l143_14377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_increasing_interval_subset_domain_l143_14346

-- Define the function f(x) = ln(x) / x
noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

-- State the theorem
theorem f_monotone_increasing :
  ∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → x₂ < Real.exp 1 → f x₁ < f x₂ := by
  sorry

-- Define the domain of f
def f_domain : Set ℝ := {x : ℝ | x > 0}

-- Define the interval where f is monotonically increasing
def f_increasing_interval : Set ℝ := {x : ℝ | 0 < x ∧ x < Real.exp 1}

-- State that f_increasing_interval is a subset of f_domain
theorem increasing_interval_subset_domain :
  f_increasing_interval ⊆ f_domain := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_increasing_interval_subset_domain_l143_14346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l143_14306

open Set Real

-- Define the sets M and N
def M : Set ℝ := {x | log x > 0}
def N : Set ℝ := {x | x^2 - 3*x - 4 > 0}

-- State the theorem
theorem intersection_M_N : M ∩ N = Ioi 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l143_14306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_ratio_bound_l143_14339

/-- Represents an ellipse with the given conditions -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b ∧ b > 0

/-- Represents a line with slope k passing through (1, 0) -/
structure Line where
  k : ℝ
  h_k : k ≠ 0

/-- Represents the intersection points and related constructions -/
structure Intersection (e : Ellipse) (l : Line) where
  M : ℝ × ℝ
  N : ℝ × ℝ
  D : ℝ
  P : ℝ × ℝ
  h_M : M.1^2 / e.a^2 + M.2^2 / e.b^2 = 1
  h_N : N.1^2 / e.a^2 + N.2^2 / e.b^2 = 1
  h_MN_on_l : M.2 = l.k * (M.1 - 1) ∧ N.2 = l.k * (N.1 - 1)
  h_P_midpoint : P = ((M.1 + N.1) / 2, (M.2 + N.2) / 2)

/-- The main theorem -/
theorem ellipse_intersection_ratio_bound (e : Ellipse) (l : Line) (i : Intersection e l) :
  0 < |i.D - i.P.1| / Real.sqrt ((i.M.1 - i.N.1)^2 + (i.M.2 - i.N.2)^2) ∧
  |i.D - i.P.1| / Real.sqrt ((i.M.1 - i.N.1)^2 + (i.M.2 - i.N.2)^2) < 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_ratio_bound_l143_14339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_anniversary_day_theorem_l143_14308

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Calculates the day of the week n days before a given day -/
def daysBefore (d : DayOfWeek) : ℕ → DayOfWeek
  | 0 => d
  | n + 1 => match d with
    | .Sunday => daysBefore .Saturday n
    | .Monday => daysBefore .Sunday n
    | .Tuesday => daysBefore .Monday n
    | .Wednesday => daysBefore .Tuesday n
    | .Thursday => daysBefore .Wednesday n
    | .Friday => daysBefore .Thursday n
    | .Saturday => daysBefore .Friday n

/-- The number of days to go back for a 150-year period -/
def daysToGoBack : ℕ := 186

/-- Theorem: If a person's 150th anniversary falls on a Monday, they were born on a Wednesday -/
theorem anniversary_day_theorem : 
  daysBefore DayOfWeek.Monday daysToGoBack = DayOfWeek.Wednesday := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_anniversary_day_theorem_l143_14308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_decreasing_l143_14327

-- Define the function f(x) = 3x^2 - 3ln(x)
noncomputable def f (x : ℝ) : ℝ := 3 * x^2 - 3 * Real.log x

-- Define the domain of f(x)
def domain : Set ℝ := {x | x > 0}

-- Define the monotonic decreasing interval
def monotonic_decreasing_interval : Set ℝ := Set.Ioo 0 (Real.sqrt 2 / 2)

-- Theorem statement
theorem f_monotonic_decreasing :
  ∀ x ∈ monotonic_decreasing_interval ∩ domain,
    ∀ y ∈ monotonic_decreasing_interval ∩ domain,
      x < y → f x > f y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_decreasing_l143_14327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_max_value_l143_14335

noncomputable section

open Real

-- Define the function f
def f (θ : ℝ) (x : ℝ) : ℝ := Real.cos x * Real.cos (x - θ) - (1/2) * Real.cos θ

-- Define the function g
def g (θ : ℝ) (x : ℝ) : ℝ := 2 * f θ ((3/2) * x)

-- State the theorem
theorem function_max_value (θ : ℝ) :
  θ ∈ Set.Ioo 0 π →
  (∀ x, f θ x ≤ f θ (π/3)) →
  θ = 2*π/3 ∧ 
  (∀ x ∈ Set.Icc 0 (π/3), g θ x ≤ 1) ∧
  (∃ x ∈ Set.Icc 0 (π/3), g θ x = 1) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_max_value_l143_14335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neg_one_eq_neg_three_l143_14342

noncomputable def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then x^2 + 2/x else -(-x^2 - 2/(-x))  -- Define f for all x, extending it to negative x

theorem f_neg_one_eq_neg_three
  (h_odd : is_odd_function f)
  (h_pos : ∀ x > 0, f x = x^2 + 2/x) :
  f (-1) = -3 := by
  sorry

#check f_neg_one_eq_neg_three

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neg_one_eq_neg_three_l143_14342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_product_l143_14369

/-- A geometric sequence with common ratio q and first term a₁ -/
def geometricSequence (q : ℝ) (a₁ : ℝ) : ℕ → ℝ :=
  fun n ↦ a₁ * q^(n - 1)

theorem geometric_sequence_product (q : ℝ) (hq : |q| ≠ 1) :
  ∃ m : ℕ,
    geometricSequence q (-1) m =
      (geometricSequence q (-1) 1) *
      (geometricSequence q (-1) 2) *
      (geometricSequence q (-1) 3) *
      (geometricSequence q (-1) 4) *
      (geometricSequence q (-1) 5) ∧
    m = 11 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_product_l143_14369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_one_value_l143_14355

noncomputable def f (x : ℝ) : ℝ := 3 * x^2 + 2

noncomputable def g (x : ℝ) : ℝ := Real.sqrt (f x) - 3

noncomputable def h (x : ℝ) : ℝ := f (g x)

theorem h_one_value : h 1 = 44 - 18 * Real.sqrt 5 := by
  -- Expand the definitions of h, g, and f
  unfold h g f
  -- Simplify the expression
  simp
  -- The proof is omitted for brevity
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_one_value_l143_14355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_line_equation_l143_14323

/-- Given a line l₁ with equation y = x + 1, prove that the line l₂ symmetric to l₁ 
    with respect to the point (1, 1) has the equation x - y - 1 = 0. -/
theorem symmetric_line_equation : 
  let l₁ : ℝ → ℝ := fun x ↦ x + 1
  let center : ℝ × ℝ := (1, 1)
  let l₂ : ℝ → ℝ := fun x ↦ -x + 3
  (∀ x y : ℝ, y = l₂ x ↔ (2 - x) + 1 = 2 - y) → 
  (∀ x : ℝ, l₂ x = -x + 3) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_line_equation_l143_14323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_of_angle_with_given_tan_and_point_l143_14373

/-- If the terminal side of angle α passes through point P(1, m), 
    and tan(α) = -2, then sin(α) = -2√5/5 -/
theorem sin_of_angle_with_given_tan_and_point :
  ∀ (α : ℝ) (m : ℝ),
  (∃ (P : ℝ × ℝ), P.1 = 1 ∧ P.2 = m ∧ P ∈ Set.range (fun t => (t * Real.cos α, t * Real.sin α))) →
  Real.tan α = -2 →
  Real.sin α = -2 * Real.sqrt 5 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_of_angle_with_given_tan_and_point_l143_14373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_denominator_for_repeating_decimal_l143_14381

theorem smallest_denominator_for_repeating_decimal (m n : ℕ+) : 
  Nat.Coprime m.val n.val → 
  m < n → 
  (∃ k : ℕ, m.val * 1000 = n.val * (251 + 1000 * k)) →
  127 ≤ n.val :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_denominator_for_repeating_decimal_l143_14381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l143_14361

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x

theorem f_properties :
  (∀ x, f x = f (-x)) ∧ 
  (∀ x, f (x + π/2) = -f (-x - π/2)) ∧ 
  (∀ x, f x ≤ 2) ∧ 
  (∃ x, f x = 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l143_14361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_equivalence_circle_special_cases_l143_14389

/-- Theorem: Equivalence of vector and coordinate forms of a circle equation -/
theorem circle_equation_equivalence 
  (R : ℝ) -- Radius of the circle
  (a b : ℝ) -- Coordinates of the center
  (r : ℝ × ℝ) -- Point on the circle (as a pair of reals)
  (c : ℝ × ℝ := (a, b)) -- Center of the circle (as a pair of reals)
  : (r.1 - c.1)^2 + (r.2 - c.2)^2 = R^2 ↔ ‖r - c‖^2 = R^2 :=
by sorry

/-- Corollary: Special cases of circle equations -/
theorem circle_special_cases 
  (R : ℝ) -- Radius of the circle
  (a b : ℝ) -- Coordinates of the center
  (x y : ℝ) -- Variables representing a point on the circle
  : (x - a)^2 + (y - b)^2 = R^2 →
    (b = 0 → (x - a)^2 + y^2 = R^2) ∧
    (a = 0 → x^2 + (y - b)^2 = R^2) ∧
    (a = 0 ∧ b = 0 → x^2 + y^2 = R^2) ∧
    (R^2 = a^2 + b^2 → x^2 - 2*a*x + y^2 - 2*b*y = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_equivalence_circle_special_cases_l143_14389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_phi_condition_l143_14356

variable (A ω φ : ℝ)
variable (k : ℤ)

noncomputable def f (x : ℝ) := A * Real.cos (ω * x + φ)

theorem odd_function_phi_condition
  (h_A : A > 0)
  (h_ω : ω > 0)
  (h_odd : ∀ x, f A ω φ (-x) = -(f A ω φ x)) :
  ∃ k, φ = π / 2 + k * π := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_phi_condition_l143_14356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribing_square_area_l143_14367

-- Define the circle
def my_circle (x y : ℝ) : Prop := (x - 2)^2 + (y - 2)^2 = 25

-- Define the square that inscribes the circle
def inscribing_square (s : ℝ) : Prop :=
  ∃ (x₀ y₀ : ℝ), 
    (∀ x y, my_circle x y → x₀ ≤ x ∧ x ≤ x₀ + s ∧ y₀ ≤ y ∧ y ≤ y₀ + s) ∧
    (∃ x y, my_circle x y ∧ x = x₀) ∧
    (∃ x y, my_circle x y ∧ x = x₀ + s) ∧
    (∃ x y, my_circle x y ∧ y = y₀) ∧
    (∃ x y, my_circle x y ∧ y = y₀ + s)

-- Theorem statement
theorem inscribing_square_area : 
  ∃ (s : ℝ), inscribing_square s ∧ s^2 = 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribing_square_area_l143_14367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_revolutions_half_mile_l143_14398

/-- The number of revolutions for a wheel with given diameter to travel a certain distance -/
noncomputable def revolutions (diameter : ℝ) (distance : ℝ) : ℝ :=
  distance / (Real.pi * diameter)

/-- Half a mile in feet -/
def half_mile_in_feet : ℝ := 0.5 * 5280

theorem wheel_revolutions_half_mile :
  revolutions 10 half_mile_in_feet = 264 / Real.pi :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_revolutions_half_mile_l143_14398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_tiles_required_l143_14322

/-- Represents a tile type -/
inductive TileType
  | Gamma  -- Γ-shaped tile covering 3 squares
  | Zeta   -- Z-shaped tile covering 4 squares
deriving BEq, Repr

/-- Represents a tiling configuration -/
def Tiling (m n : ℕ) := List TileType

/-- Checks if a tiling configuration is valid for the given dimensions -/
def isValidTiling (m n : ℕ) (t : Tiling m n) : Prop :=
  (t.length ≥ m * n) ∧
  (t.count TileType.Gamma * 3 + t.count TileType.Zeta * 4 = (2 * m - 1) * (2 * n - 1))

/-- The main theorem: minimum number of tiles required -/
theorem min_tiles_required (m n : ℕ) (hm : m ≥ 4) (hn : n ≥ 4) :
  ∃ (t : Tiling m n), isValidTiling m n t ∧
    ∀ (t' : Tiling m n), isValidTiling m n t' → t.length ≤ t'.length := by
  sorry

#check min_tiles_required

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_tiles_required_l143_14322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_sum_of_squares_l143_14302

theorem roots_sum_of_squares (a b : ℝ) : 
  (a^2 - 9*a + 9 = 0) → (b^2 - 9*b + 9 = 0) → (a^2 + b^2 = 63) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_sum_of_squares_l143_14302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_equidistant_l143_14319

/-- An angle in a plane --/
structure Angle where
  vertex : EuclideanSpace ℝ (Fin 2)
  side1 : EuclideanSpace ℝ (Fin 2)
  side2 : EuclideanSpace ℝ (Fin 2)

/-- The bisector of an angle --/
noncomputable def angleBisector (a : Angle) : Set (EuclideanSpace ℝ (Fin 2)) := sorry

/-- The distance from a point to a line --/
noncomputable def distanceToLine (p : EuclideanSpace ℝ (Fin 2)) (l : Set (EuclideanSpace ℝ (Fin 2))) : ℝ := sorry

/-- A line through two points --/
noncomputable def lineThrough (p q : EuclideanSpace ℝ (Fin 2)) : Set (EuclideanSpace ℝ (Fin 2)) := sorry

/-- Theorem: Any point on the angle bisector is equidistant from the two sides of the angle --/
theorem angle_bisector_equidistant (a : Angle) (p : EuclideanSpace ℝ (Fin 2)) :
  p ∈ angleBisector a →
  distanceToLine p (lineThrough a.vertex a.side1) = distanceToLine p (lineThrough a.vertex a.side2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_equidistant_l143_14319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_condition_l143_14372

/-- The condition for two functions to have exactly two intersection points -/
def has_two_intersections (f g : ℝ → ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f x₁ = g x₁ ∧ f x₂ = g x₂ ∧
  ∀ x, f x = g x → x = x₁ ∨ x = x₂

/-- The main theorem -/
theorem intersection_condition (a b : ℝ) :
  let f := λ x : ℝ ↦ 2 * x^4 - a^2 * x^2 + b - 1
  let g := λ x : ℝ ↦ 2 * a * x^3 - 1
  has_two_intersections f g ↔ b < 0 ∨ (3 * a^4 / 128 < b ∧ b < a^4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_condition_l143_14372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_five_terms_half_l143_14375

/-- Sum of a geometric series with n terms, first term a, and common ratio r -/
noncomputable def geometricSum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

/-- The problem statement -/
theorem geometric_sum_five_terms_half :
  geometricSum (1/2) (1/2) 5 = 31/32 := by
  -- Expand the definition of geometricSum
  unfold geometricSum
  -- Simplify the expression
  simp [pow_succ]
  -- Perform arithmetic operations
  ring
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_five_terms_half_l143_14375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_real_part_of_z1_minus_z2_times_i_l143_14330

theorem real_part_of_z1_minus_z2_times_i : 
  let z₁ : ℂ := 4 + 29 * Complex.I
  let z₂ : ℂ := 6 + 9 * Complex.I
  Complex.re ((z₁ - z₂) * Complex.I) = -20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_real_part_of_z1_minus_z2_times_i_l143_14330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l143_14392

theorem relationship_abc : 
  let a := 2 * Real.log 0.3 / Real.log 2
  let b := (2 : Real) ^ (0.1 : Real)
  let c := (0.2 : Real) ^ (1.3 : Real)
  b > c ∧ c > a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l143_14392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_properties_l143_14348

noncomputable def f (x : ℝ) : ℝ := -8/3 * x^2 + 16 * x - 40/3

theorem quadratic_properties : 
  f 1 = 0 ∧ f 5 = 0 ∧ f 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_properties_l143_14348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_product_solution_l143_14360

/-- The vector a --/
def a : ℝ × ℝ × ℝ := (3, 1, -2)

/-- The vector b --/
def b : ℝ × ℝ × ℝ := (1, 2, -2)

/-- The cross product of two 3D vectors --/
def cross (v w : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (v.2.1 * w.2.2 - v.2.2 * w.2.1,
   v.2.2 * w.1 - v.1 * w.2.2,
   v.1 * w.2.1 - v.2.1 * w.1)

/-- The dot product of two 3D vectors --/
def dot (v w : ℝ × ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2.1 * w.2.1 + v.2.2 * w.2.2

/-- Scalar multiplication for 3D vectors --/
def smul (r : ℝ) (v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (r * v.1, r * v.2.1, r * v.2.2)

/-- Addition of 3D vectors --/
def add (v w : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (v.1 + w.1, v.2.1 + w.2.1, v.2.2 + w.2.2)

theorem cross_product_solution (p q r : ℝ) :
  add (add (smul 5 a) (smul 3 b)) (smul (-7) (cross a b)) =
  add (add (smul p a) (smul q b)) (smul r (cross a b)) →
  r = -13/45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_product_solution_l143_14360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_profit_share_l143_14303

/-- Calculates the share of profit for a partner in a business based on investments and durations --/
theorem calculate_profit_share 
  (x_investment y_investment z_investment : ℕ)
  (x_duration y_duration z_duration : ℕ)
  (total_profit : ℕ) :
  let total_investment_months := x_investment * x_duration + y_investment * y_duration + z_investment * z_duration
  let z_share := (z_investment * z_duration * total_profit) / total_investment_months
  x_investment = 36000 →
  y_investment = 42000 →
  z_investment = 48000 →
  x_duration = 12 →
  y_duration = 12 →
  z_duration = 8 →
  total_profit = 13860 →
  z_share = 2520 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_profit_share_l143_14303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_sum_l143_14344

theorem triangle_side_sum (a b c : ℝ) (A : ℝ) :
  a = Real.sqrt 3 →
  Real.cos A = 1 / 3 →
  b * c = 3 / 2 →
  b + c = Real.sqrt 7 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_sum_l143_14344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_job_completion_time_l143_14385

/-- Represents the number of days it takes for person A to complete the job alone -/
def days_for_A : ℝ → Prop := sorry

/-- Represents the number of days it takes for person B to complete the job alone -/
def days_for_B : ℝ → Prop := sorry

/-- Represents the number of days A and B work together -/
def days_together : ℝ → Prop := sorry

/-- Represents the fraction of work left after A and B work together -/
def fraction_left : ℝ → Prop := sorry

theorem job_completion_time 
  (hB : days_for_B 20)
  (h_together : days_together 4)
  (h_left : fraction_left 0.4) :
  days_for_A 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_job_completion_time_l143_14385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_blue_marbles_l143_14300

def total_marbles : ℚ := 10
def blue_marbles : ℚ := 6
def red_marbles : ℚ := 4
def num_draws : ℕ := 5
def num_blue_draws : ℕ := 2

def prob_blue : ℚ := blue_marbles / total_marbles
def prob_red : ℚ := red_marbles / total_marbles

theorem exactly_two_blue_marbles :
  (Nat.choose num_draws num_blue_draws : ℚ) * prob_blue ^ num_blue_draws * prob_red ^ (num_draws - num_blue_draws) = 720 / 3125 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_blue_marbles_l143_14300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_five_eq_half_l143_14336

def f : ℕ → ℚ
| 0 => 1
| 1 => 1/2
| 2 => 1/2
| (n + 3) => f n

theorem f_five_eq_half : f 5 = 1/2 := by
  rfl

#eval f 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_five_eq_half_l143_14336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l143_14347

noncomputable def f (x : ℝ) := Real.sin (x / 2) + Real.sqrt 3 * Real.cos (x / 2)

theorem f_properties :
  ∃ (max_set : Set ℝ) (period : ℝ) (mono_intervals : Set (Set ℝ)),
    (∀ k : ℤ, (4 * k * Real.pi - 2 * Real.pi / 3) ∈ max_set) ∧
    (∀ x ∈ max_set, ∀ y : ℝ, f y ≤ f x) ∧
    period = 4 * Real.pi ∧
    (∀ x : ℝ, f (x + period) = f x) ∧
    (∀ k : ℤ, Set.Icc (-5 * Real.pi / 3 + 4 * k * Real.pi) (Real.pi / 3 + 4 * k * Real.pi) ∈ mono_intervals) ∧
    (∀ s ∈ mono_intervals, ∀ x y : ℝ, x ∈ s → y ∈ s → x ≤ y → f x ≤ f y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l143_14347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_trig_expression_l143_14362

theorem simplify_trig_expression (α : ℝ) (h : Real.sin α = 3 * Real.cos α) :
  (1 - 3 * Real.sin α * Real.cos α) / (2 * Real.cos (2 * α) + 4) = 1 / 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_trig_expression_l143_14362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_purely_imaginary_tan_theta_l143_14341

theorem purely_imaginary_tan_theta : 
  ∀ θ : ℝ, 
  let z : ℂ := Complex.mk (Real.sin θ - 3/5) (Real.cos θ - 4/5)
  (Complex.re z = 0 ∧ Complex.im z ≠ 0) → 
  Real.tan θ = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_purely_imaginary_tan_theta_l143_14341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_range_l143_14371

/-- The function f(x) = ((a-x)/(x+1)) * e^(2x) where x > -1 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := ((a - x) / (x + 1)) * Real.exp (2 * x)

/-- The function g(x) = (2mx + m - e^(2x)) / x^2 where x > 0 -/
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := (2 * m * x + m - Real.exp (2 * x)) / (x^2)

/-- h(m) is the maximum value of g(x) for x > 0 -/
noncomputable def h (m : ℝ) : ℝ := sorry

theorem h_range (m : ℝ) (hm : m ∈ Set.Icc 0 1) :
  h m ∈ Set.Icc (-(Real.exp 2)) (-2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_range_l143_14371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_tangents_ratio_l143_14305

-- Define the function f(x) = xe^x
noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

-- Define the point P
noncomputable def P : ℝ × ℝ := (1, Real.exp 1)

-- Theorem statement
theorem perpendicular_tangents_ratio (a b : ℝ) :
  -- Line equation condition
  (∃ (x y : ℝ), a * x - b * y - 3 = 0) →
  -- Tangent lines are perpendicular at point P
  (∃ (m₁ m₂ : ℝ), m₁ * m₂ = -1 ∧
    m₁ = (deriv f) P.fst ∧
    m₂ = a / b) →
  -- Conclusion
  a / b = -1 / (2 * Real.exp 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_tangents_ratio_l143_14305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_through_point_four_zero_and_same_asymptotes_l143_14312

/-- A hyperbola with the same shape as x²/4 - y² = 1, passing through (4,0) -/
def Hyperbola (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / 4 - p.2^2 = m}

/-- The asymptotes of a hyperbola with the same shape as x²/4 - y² = 1 -/
def Asymptotes : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = p.1 / 2 ∨ p.2 = -p.1 / 2}

theorem hyperbola_through_point_four_zero_and_same_asymptotes :
  ∃ m : ℝ, (4, 0) ∈ Hyperbola m ∧ 
  (∀ p : ℝ × ℝ, p ∈ Hyperbola m ↔ p.1^2 / 16 - p.2^2 / 4 = 1) ∧
  (∀ p : ℝ × ℝ, p ∈ Asymptotes ↔ p ∈ closure (Hyperbola m) \ Hyperbola m) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_through_point_four_zero_and_same_asymptotes_l143_14312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_age_sum_difference_is_eleven_l143_14353

-- Define the ages of A, B, and C
variable (A B C : ℕ)

-- Define the condition that C is 11 years younger than A
def age_difference (A C : ℕ) : Prop := C = A - 11

-- Define the difference between (A + B) and (B + C)
def age_sum_difference (A B C : ℕ) : ℕ := (A + B) - (B + C)

-- Theorem statement
theorem age_sum_difference_is_eleven (h : age_difference A C) : 
  age_sum_difference A B C = 11 := by
  unfold age_difference at h
  unfold age_sum_difference
  rw [h]
  ring
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_age_sum_difference_is_eleven_l143_14353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_work_days_l143_14357

/-- The number of days A takes to finish the work alone -/
noncomputable def a_days : ℚ := 10

/-- The ratio of A's wages to total wages when working together with B -/
noncomputable def a_wage_ratio : ℚ := 1860 / 3100

/-- The number of days B takes to finish the work alone -/
noncomputable def b_days : ℚ := 15

/-- Theorem stating that given A's work rate and wage ratio, B takes 15 days to finish the work alone -/
theorem b_work_days (a_rate : ℚ) (b_rate : ℚ) :
  a_rate = 1 / a_days →
  b_rate = 1 / b_days →
  a_rate / (a_rate + b_rate) = a_wage_ratio →
  b_days = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_work_days_l143_14357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l143_14317

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := 
  if x > 0 ∧ x ≤ 2 then 2^x - 1
  else if x < 0 ∧ x ≥ -2 then -(2^(-x) - 1)
  else 0

def g (x m : ℝ) : ℝ := x^2 - 2*x + m

-- State the theorem
theorem range_of_m (m : ℝ) :
  (∀ x, x ∈ Set.Icc (-2) 2 → f x ∈ Set.Icc (-3) 3) →
  (∀ x₁, x₁ ∈ Set.Icc (-2) 2 → ∃ x₂ ∈ Set.Icc (-2) 2, g x₂ m = f x₁) →
  m ∈ Set.Icc (-5) (-2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l143_14317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_asha_borrowed_from_mother_l143_14321

/-- Amount borrowed from brother -/
noncomputable def brother_loan : ℚ := 20

/-- Amount borrowed from father -/
noncomputable def father_loan : ℚ := 40

/-- Amount gifted by granny -/
noncomputable def granny_gift : ℚ := 70

/-- Amount in savings -/
noncomputable def savings : ℚ := 100

/-- Fraction of money spent -/
noncomputable def spent_fraction : ℚ := 3/4

/-- Amount remaining after spending -/
noncomputable def remaining : ℚ := 65

/-- Amount borrowed from mother -/
noncomputable def mother_loan : ℚ := 30

theorem asha_borrowed_from_mother :
  brother_loan + father_loan + granny_gift + savings + mother_loan =
  remaining / (1 - spent_fraction) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_asha_borrowed_from_mother_l143_14321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_625_l143_14368

/-- A function g : ℕ → ℕ satisfying the given property -/
def g : ℕ → ℕ := sorry

/-- The property of function g -/
axiom g_property : ∀ (x y m : ℕ), x > 0 → y > 0 → m > 0 → x + y = 5^m → g x + g y = 3 * m^2

/-- The theorem to prove -/
theorem g_625 : g 625 = 48 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_625_l143_14368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_problem_l143_14340

/-- Calculates compound interest amount -/
noncomputable def compoundInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Calculates simple interest -/
def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem simple_interest_problem (principal : ℝ) :
  let compound_amount := compoundInterest 4000 0.1 2
  let compound_interest := compound_amount - 4000
  let simple_interest := simpleInterest principal 0.08 2
  simple_interest = compound_interest / 2 → principal = 2625 := by
  sorry

#check simple_interest_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_problem_l143_14340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diver_descent_time_l143_14359

/-- The time taken for a diver to reach a certain depth -/
noncomputable def time_to_reach (depth : ℝ) (descent_rate : ℝ) : ℝ :=
  depth / descent_rate

/-- Theorem: The time taken for a diver to reach a depth of 2400 feet
    while descending at a rate of 30 feet per minute is 80 minutes -/
theorem diver_descent_time :
  time_to_reach 2400 30 = 80 := by
  -- Unfold the definition of time_to_reach
  unfold time_to_reach
  -- Simplify the division
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_diver_descent_time_l143_14359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_two_nonzero_digits_of_72_factorial_l143_14325

theorem last_two_nonzero_digits_of_72_factorial (n : ℕ) : 
  n = 64 → ∃ k : ℕ, Nat.factorial 72 = 100 * k + n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_two_nonzero_digits_of_72_factorial_l143_14325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_nine_pi_fourteenth_minus_x_l143_14309

theorem cos_nine_pi_fourteenth_minus_x (x : ℝ) :
  Real.sin (π / 7 - x) = -2 / 3 → Real.cos (9 * π / 14 - x) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_nine_pi_fourteenth_minus_x_l143_14309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_n_for_constant_term_l143_14393

theorem min_n_for_constant_term : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∃ (k : ℕ), (Finset.range (n + 1)).sum (λ i ↦ 
    (n.choose i) * (2^(n - i)) * ((-1/3)^i) * ((4*(n - i) - 3*i) : ℝ)) ≠ 0) ∧
  (∀ (m : ℕ), 0 < m → m < n → 
    (∀ (k : ℕ), (Finset.range (m + 1)).sum (λ i ↦ 
      (m.choose i) * (2^(m - i)) * ((-1/3)^i) * ((4*(m - i) - 3*i) : ℝ)) = 0)) ∧
  n = 7 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_n_for_constant_term_l143_14393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_increasing_l143_14397

noncomputable def f (x : ℝ) : ℝ := Real.log (-x^2 + x + 6) / Real.log 0.8

theorem f_strictly_increasing :
  ∀ x₁ x₂, 1/2 < x₁ ∧ x₁ < x₂ ∧ x₂ < 3 →
  -2 < x₁ ∧ x₁ < 3 ∧ -2 < x₂ ∧ x₂ < 3 →
  0 < 0.8 ∧ 0.8 < 1 →
  f x₁ < f x₂ :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_increasing_l143_14397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l143_14379

def M : Set ℤ := {x | -4 < x ∧ x < 2}
def N : Set ℝ := {x | x^2 < 4}

def M_real : Set ℝ := {x | ∃ n : ℤ, n ∈ M ∧ x = n}

theorem intersection_M_N : M_real ∩ N = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l143_14379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_video_loading_time_l143_14352

/-- The time it takes to load a video on a cellphone, in seconds -/
noncomputable def cellphone_time : ℝ := 9 * 60

/-- The time it takes to load a video on both devices simultaneously, in seconds -/
noncomputable def combined_time : ℝ := 14.59

/-- The time it takes to load a video on a laptop, in seconds -/
noncomputable def laptop_time : ℝ := (cellphone_time * combined_time) / (cellphone_time - combined_time)

theorem video_loading_time :
  |laptop_time - 14.99| < 0.01 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_video_loading_time_l143_14352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_square_area_l143_14387

-- Define the line and parabola
def line (x : ℝ) : ℝ := 2 * x + 5
def parabola (x : ℝ) : ℝ := x^2 - 4 * x

-- Define a square with vertices on the line and parabola
structure Square where
  x1 : ℝ
  x2 : ℝ
  y1 : ℝ
  y2 : ℝ
  on_line : y1 = line x1 ∧ y2 = line x2
  on_parabola : y1 = parabola x1 ∧ y2 = parabola x2

-- Function to calculate the area of the square
def square_area (s : Square) : ℝ :=
  (s.x1 - s.x2)^2 + (s.y1 - s.y2)^2

-- Theorem stating the smallest possible area
theorem smallest_square_area :
  ∃ (min_area : ℝ), (min_area ≥ 3.39 ∧ min_area ≤ 3.41) ∧
  ∀ (s : Square), square_area s ≥ min_area :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_square_area_l143_14387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blanket_rate_solution_l143_14337

/-- Calculates the unknown rate of two blankets given the following conditions:
    - 3 blankets @ Rs. 100 each
    - 5 blankets @ Rs. 150 each
    - 2 blankets at an unknown rate
    - Average price of all blankets is Rs. 150 -/
def unknown_blanket_rate (blanket_count1 : ℕ) (price1 : ℕ) (blanket_count2 : ℕ) (price2 : ℕ) 
  (unknown_blanket_count : ℕ) (avg_price : ℕ) : ℕ :=
  let total_blankets := blanket_count1 + blanket_count2 + unknown_blanket_count
  let known_cost := blanket_count1 * price1 + blanket_count2 * price2
  let total_cost := avg_price * total_blankets
  let unknown_cost := total_cost - known_cost
  unknown_cost / unknown_blanket_count

/-- The unknown rate of the two blankets is 225 Rs. each -/
theorem blanket_rate_solution : unknown_blanket_rate 3 100 5 150 2 150 = 225 := by
  sorry

#eval unknown_blanket_rate 3 100 5 150 2 150

end NUMINAMATH_CALUDE_ERRORFEEDBACK_blanket_rate_solution_l143_14337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_10_l143_14376

/-- An arithmetic sequence with positive terms -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ ∃ d, ∀ n, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
noncomputable def SumFirstN (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * (a 1 + a n) / 2

theorem arithmetic_sequence_sum_10 (a : ℕ → ℝ) :
  ArithmeticSequence a →
  a 3^2 + a 8^2 + 2*(a 3)*(a 8) = 9 →
  SumFirstN a 10 = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_10_l143_14376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_semicircle_area_60_degrees_l143_14370

/-- The area of a figure formed by rotating a semicircle around one of its ends -/
noncomputable def rotated_semicircle_area (R : ℝ) (α : ℝ) : ℝ :=
  (α / (2 * Real.pi)) * 2 * Real.pi * R^2

theorem rotated_semicircle_area_60_degrees (R : ℝ) (h : R > 0) :
  rotated_semicircle_area R (Real.pi / 3) = (2 * Real.pi * R^2) / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_semicircle_area_60_degrees_l143_14370

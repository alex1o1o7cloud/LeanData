import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_of_curve_l798_79818

noncomputable def curve (x : ℝ) : ℝ := Real.arcsin (Real.sqrt x) - Real.sqrt (x - x^2)

theorem arc_length_of_curve : ∫ x in Set.Icc 0 1, Real.sqrt (1 + (deriv curve x)^2) = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_of_curve_l798_79818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_always_has_two_solutions_probability_is_one_l798_79899

theorem equation_always_has_two_solutions (b : ℝ) (h : b ∈ Set.Icc (-9) 27) :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  x₁^4 + 9*b^2 = (6*b^2 - 18*b)*x₁^2 ∧
  x₂^4 + 9*b^2 = (6*b^2 - 18*b)*x₂^2 :=
by sorry

theorem probability_is_one :
  ∃ (m n : ℕ+), Nat.Coprime m n ∧ (m : ℝ) / (n : ℝ) = 1 ∧ m + n = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_always_has_two_solutions_probability_is_one_l798_79899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_l798_79865

-- Problem 1
theorem problem_1 : Real.pi^(0 : ℝ) - (1/2)^(-2 : ℝ) + 3^(2 : ℝ) = 6 := by sorry

-- Problem 2
theorem problem_2 (x : ℝ) : (2*x^2)^2 - x * x^3 - x^5 / x = 2*x^4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_l798_79865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_vectors_x_value_l798_79846

theorem opposite_vectors_x_value (x : ℝ) : 
  let a : ℝ × ℝ := (2*x, 1)
  let b : ℝ × ℝ := (4, x)
  (∃ (k : ℝ), k < 0 ∧ a = k • b) → x = -Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_vectors_x_value_l798_79846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_negative_integers_l798_79834

theorem max_negative_integers (a b c d e f : ℤ) (h : a * b + c * d * e * f < 0) :
  ∃ (s : ℕ), s = 4 ∧ ∀ (k : ℕ), (∃ (neg : Finset ℤ), neg.card = k ∧ 
    neg ⊆ {a, b, c, d, e, f} ∧ ∀ x ∈ neg, x < 0) → k ≤ s := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_negative_integers_l798_79834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_number_last_four_digits_l798_79801

def is_valid_number (n : ℕ) : Prop :=
  ∃ (digits : List ℕ),
    n = digits.foldl (λ acc d => 10 * acc + d) 0 ∧
    digits.all (λ d => d = 4 ∨ d = 9) ∧
    (digits.filter (λ d => d = 4)).length ≥ 2 ∧
    (digits.filter (λ d => d = 9)).length ≥ 2

def last_four_digits (n : ℕ) : ℕ :=
  n % 10000

theorem smallest_valid_number_last_four_digits :
  ∃ (m : ℕ),
    is_valid_number m ∧
    m % 4 = 0 ∧
    m % 9 = 0 ∧
    (∀ (k : ℕ), k < m → ¬(is_valid_number k ∧ k % 4 = 0 ∧ k % 9 = 0)) ∧
    last_four_digits m = 4944 := by
  sorry

#check smallest_valid_number_last_four_digits

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_number_last_four_digits_l798_79801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_width_is_four_l798_79877

/-- Represents a rectangular cistern with given dimensions and wet surface area. -/
structure Cistern where
  length : ℝ
  depth : ℝ
  wet_surface_area : ℝ

/-- Calculates the width of a cistern given its dimensions and wet surface area. -/
noncomputable def calculate_width (c : Cistern) : ℝ :=
  (c.wet_surface_area - 2 * c.length * c.depth) / (2 * c.depth + c.length)

/-- Theorem stating that a cistern with given dimensions has a width of 4 meters. -/
theorem cistern_width_is_four (c : Cistern)
    (h1 : c.length = 7)
    (h2 : c.depth = 1.25)
    (h3 : c.wet_surface_area = 55.5) :
    calculate_width c = 4 := by
  sorry

-- Remove the #eval statement as it's not necessary for the theorem and may cause issues

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_width_is_four_l798_79877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_choir_arrangement_l798_79892

theorem choir_arrangement (n : ℕ) : 
  (∀ m ∈ ({5, 9, 12} : Set ℕ), n % m = 0) ∧ 
  (∃ k : ℕ, n = k * k) ∧
  (∀ p : ℕ, p < n → ¬((∀ m ∈ ({5, 9, 12} : Set ℕ), p % m = 0) ∧ (∃ k : ℕ, p = k * k))) →
  n = 360 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_choir_arrangement_l798_79892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_max_value_change_l798_79825

/-- A quadratic polynomial f(x) = ax^2 + bx + c -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The maximum value of a quadratic polynomial -/
noncomputable def maxValue (f : QuadraticPolynomial) : ℝ := -f.b^2 / (4 * f.a) + f.c

/-- The change in maximum value when adding kx^2 to f(x) -/
noncomputable def maxValueChange (f : QuadraticPolynomial) (k : ℝ) : ℝ :=
  maxValue { a := f.a + k, b := f.b, c := f.c } - maxValue f

theorem quadratic_max_value_change 
  (f : QuadraticPolynomial) 
  (h1 : maxValueChange f 1 = 27/2) 
  (h2 : maxValueChange f (-4) = -9) : 
  maxValueChange f (-2) = -27/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_max_value_change_l798_79825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_corrected_scores_theorem_l798_79868

/-- Represents the test scores of a class -/
structure ClassScores where
  scores : List Float
  incorrect_index1 : Nat
  incorrect_index2 : Nat
  correct_score1 : Float
  correct_score2 : Float

/-- Calculates the average of a list of scores -/
def average (scores : List Float) : Float :=
  scores.sum / scores.length.toFloat

/-- Calculates the variance of a list of scores -/
def variance (scores : List Float) (mean : Float) : Float :=
  (scores.map (fun x => (x - mean) ^ 2)).sum / scores.length.toFloat

theorem corrected_scores_theorem (cs : ClassScores) :
  cs.scores.length = 50 →
  average cs.scores = 70 →
  variance cs.scores 70 = 102 →
  cs.scores[cs.incorrect_index1]! = 50 →
  cs.scores[cs.incorrect_index2]! = 90 →
  cs.correct_score1 = 80 →
  cs.correct_score2 = 60 →
  let corrected_scores := cs.scores.set cs.incorrect_index1 cs.correct_score1
                                       |>.set cs.incorrect_index2 cs.correct_score2
  average corrected_scores = 70 ∧
  variance corrected_scores 70 = 90 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_corrected_scores_theorem_l798_79868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_five_is_square_root_one_is_solution_to_inequality_irrationals_are_non_terminating_decimals_l798_79803

-- Statement 1: -5 is a square root of (-5)²
theorem negative_five_is_square_root : (-5 : ℝ)^2 = 25 ∧ ((-5 : ℝ) * (-5 : ℝ) = 25) := by
  sorry

-- Statement 2: x = 1 is a solution to the inequality 3x - 5 ≤ -2
theorem one_is_solution_to_inequality : (3 * 1 - 5 : ℝ) ≤ -2 := by
  sorry

-- Define NonTerminatingDecimal as a property of real numbers
def NonTerminatingDecimal (x : ℝ) : Prop :=
  ∀ n : ℕ, ∃ m : ℕ, x ≠ (m : ℝ) / (10^n : ℝ)

-- Statement 3: All irrational numbers are non-terminating decimals
theorem irrationals_are_non_terminating_decimals :
  ∀ x : ℝ, Irrational x → NonTerminatingDecimal x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_five_is_square_root_one_is_solution_to_inequality_irrationals_are_non_terminating_decimals_l798_79803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_theorem_l798_79858

-- Define the circles and their properties
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def internally_tangent (c1 c2 : Circle) (p : ℝ × ℝ) : Prop :=
  (c1.radius < c2.radius) ∧
  (distance c1.center p = c1.radius) ∧
  (distance c2.center p = c2.radius)

def intersect (c1 c2 : Circle) : Prop :=
  ∃ p1 p2 : ℝ × ℝ, p1 ≠ p2 ∧
    (distance c1.center p1 = c1.radius) ∧
    (distance c1.center p2 = c1.radius) ∧
    (distance c2.center p1 = c2.radius) ∧
    (distance c2.center p2 = c2.radius)

def on_segment (p a b : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ p = (1 - t) • a + t • b

-- State the theorem
theorem circle_intersection_theorem
  (Γ Γ1 Γ2 : Circle)
  (A B : ℝ × ℝ)
  (h_radii : Γ1.radius < Γ2.radius ∧ Γ2.radius < Γ.radius)
  (h_tangent1 : internally_tangent Γ1 Γ A)
  (h_tangent2 : internally_tangent Γ2 Γ B)
  (h_distinct : A ≠ B)
  (h_intersect : intersect Γ1 Γ2) :
  (∃ C : ℝ × ℝ, (distance Γ1.center C = Γ1.radius) ∧
                (distance Γ2.center C = Γ2.radius) ∧
                (on_segment C A B)) ↔
  (Γ1.radius + Γ2.radius = Γ.radius) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_theorem_l798_79858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_l798_79824

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 5 + y^2 / 4 = 1

-- Define the right focus
def right_focus : ℝ × ℝ := (1, 0)

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k * (x - 1)

-- Define the dot product of OM and ON
def dot_product_condition (M N : ℝ × ℝ) : Prop :=
  M.1 * N.1 + M.2 * N.2 = -3

-- Define the condition for P
def P_condition (P M N : ℝ × ℝ) : Prop :=
  (M.1 - P.1)^2 + (M.2 - P.2)^2 = (N.1 - P.1)^2 + (N.2 - P.2)^2

-- Main theorem
theorem ellipse_line_intersection :
  ∀ (M N : ℝ × ℝ),
  (∀ x y, ellipse x y ↔ (x, y) = M ∨ (x, y) = N) →
  (∃ k, line_l k M.1 M.2 ∧ line_l k N.1 N.2) →
  dot_product_condition M N →
  (∃ k, line_l k M.1 M.2 ∧ line_l k N.1 N.2 ∧ (k = Real.sqrt 2 ∨ k = -Real.sqrt 2)) ∧
  (∃ a : ℝ, 0 ≤ a ∧ a < 1/5 ∧ P_condition (a, 0) M N) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_l798_79824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oh_squared_equals_526_l798_79814

/-- Triangle ABC with circumcenter O and orthocenter H -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  O : ℝ × ℝ
  H : ℝ × ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  R : ℝ

/-- The circumcenter is the center of the circumscribed circle -/
def is_circumcenter (t : Triangle) : Prop :=
  ((t.O.1 - t.A.1)^2 + (t.O.2 - t.A.2)^2 = t.R^2) ∧
  ((t.O.1 - t.B.1)^2 + (t.O.2 - t.B.2)^2 = t.R^2) ∧
  ((t.O.1 - t.C.1)^2 + (t.O.2 - t.C.2)^2 = t.R^2)

/-- The orthocenter is the point where the altitudes of the triangle intersect -/
def is_orthocenter (t : Triangle) : Prop :=
  ((t.H.1 - t.A.1) * (t.B.1 - t.C.1) + (t.H.2 - t.A.2) * (t.B.2 - t.C.2) = 0) ∧
  ((t.H.1 - t.B.1) * (t.A.1 - t.C.1) + (t.H.2 - t.B.2) * (t.A.2 - t.C.2) = 0)

/-- The side lengths of the triangle -/
def side_lengths (t : Triangle) : Prop :=
  (t.a^2 = (t.B.1 - t.C.1)^2 + (t.B.2 - t.C.2)^2) ∧
  (t.b^2 = (t.A.1 - t.C.1)^2 + (t.A.2 - t.C.2)^2) ∧
  (t.c^2 = (t.A.1 - t.B.1)^2 + (t.A.2 - t.B.2)^2)

theorem oh_squared_equals_526 (t : Triangle) 
  (h_circ : is_circumcenter t)
  (h_orth : is_orthocenter t)
  (h_sides : side_lengths t)
  (h_R : t.R = 8)
  (h_sum : t.a^2 + t.b^2 + t.c^2 = 50) :
  (t.O.1 - t.H.1)^2 + (t.O.2 - t.H.2)^2 = 526 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oh_squared_equals_526_l798_79814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_equation_properties_l798_79873

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (3 * x - Real.pi / 3)

theorem function_and_equation_properties :
  -- Conditions
  ∃ (ω φ : ℝ),
    ω > 0 ∧
    -Real.pi/2 < φ ∧ φ < 0 ∧
    Real.tan φ = -Real.sqrt 3 ∧
    (∀ x₁ x₂, |f x₁ - f x₂| = 4 → |x₁ - x₂| ≥ Real.pi/3) ∧
    (∃ x₁ x₂, |f x₁ - f x₂| = 4 ∧ |x₁ - x₂| = Real.pi/3) →
  -- Conclusions
  (∀ x, f x = 2 * Real.sin (3 * x - Real.pi / 3)) ∧
  (∀ m, m = 1/12 ∨ (-10 < m ∧ m ≤ 0) ↔
    ∃ x₁ x₂, Real.pi/9 < x₁ ∧ x₁ < x₂ ∧ x₂ < 4*Real.pi/9 ∧
    x₁ ≠ x₂ ∧
    3 * (f x₁)^2 - f x₁ + m = 0 ∧
    3 * (f x₂)^2 - f x₂ + m = 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_equation_properties_l798_79873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_acre_in_sq_feet_l798_79870

/-- Represents the dimensions and rental details of a rectangular plot of farmland. -/
structure FarmPlot where
  length : ℝ  -- Length in feet
  width : ℝ   -- Width in feet
  monthlyRent : ℝ  -- Total monthly rent in dollars
  ratePerAcre : ℝ  -- Monthly rental rate per acre in dollars

/-- Calculates the area of the farm plot in square feet. -/
noncomputable def areaInSqFeet (plot : FarmPlot) : ℝ :=
  plot.length * plot.width

/-- Calculates the number of acres in the farm plot. -/
noncomputable def acresInPlot (plot : FarmPlot) : ℝ :=
  plot.monthlyRent / plot.ratePerAcre

/-- Theorem stating that given the specified plot dimensions and rental details,
    one acre is equal to 43,560 square feet. -/
theorem one_acre_in_sq_feet (plot : FarmPlot) 
  (h1 : plot.length = 360) 
  (h2 : plot.width = 1210) 
  (h3 : plot.monthlyRent = 600) 
  (h4 : plot.ratePerAcre = 60) : 
  (areaInSqFeet plot) / (acresInPlot plot) = 43560 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_acre_in_sq_feet_l798_79870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_product_of_matrices_l798_79836

open Matrix

variable {n : Type*} [Fintype n] [DecidableEq n]
variable (A B : Matrix n n ℝ)

theorem det_product_of_matrices
  (h1 : Matrix.det A = 3)
  (h2 : Matrix.det B = 5) :
  Matrix.det (3 • A * (B ^ 2)) = 3^(Fintype.card n + 1) * 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_product_of_matrices_l798_79836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_slice_volume_ratio_l798_79894

/-- Represents a right circular cone -/
structure RightCircularCone where
  height : ℝ
  baseRadius : ℝ

/-- Represents a slice of a cone -/
structure ConeSlice where
  bottomRadius : ℝ
  topRadius : ℝ
  height : ℝ

/-- Calculate the volume of a cone slice -/
noncomputable def coneSliceVolume (slice : ConeSlice) : ℝ :=
  (1/3) * Real.pi * slice.height * (slice.bottomRadius^2 + slice.topRadius^2 + slice.bottomRadius * slice.topRadius)

/-- Calculate the volume ratio of two cone slices -/
noncomputable def volumeRatio (slice1 slice2 : ConeSlice) : ℝ :=
  coneSliceVolume slice1 / coneSliceVolume slice2

theorem cone_slice_volume_ratio 
  (cone : RightCircularCone) 
  (h : cone.height > 0) 
  (r : cone.baseRadius > 0) : 
  let sliceHeight := cone.height / 5
  let secondSmallestSlice : ConeSlice := {
    bottomRadius := 3 * cone.baseRadius / 5,
    topRadius := 2 * cone.baseRadius / 5,
    height := sliceHeight
  }
  let middleSlice : ConeSlice := {
    bottomRadius := 3 * cone.baseRadius / 5,
    topRadius := 2 * cone.baseRadius / 5,
    height := sliceHeight
  }
  volumeRatio secondSmallestSlice middleSlice = 56 / 27 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_slice_volume_ratio_l798_79894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trader_loss_percentage_l798_79802

noncomputable def car_price : ℝ := 325475
noncomputable def gain_percent : ℝ := 15
noncomputable def loss_percent : ℝ := 15

noncomputable def cost_price_1 : ℝ := car_price / (1 + gain_percent / 100)
noncomputable def cost_price_2 : ℝ := car_price / (1 - loss_percent / 100)

noncomputable def total_cost_price : ℝ := cost_price_1 + cost_price_2
noncomputable def total_selling_price : ℝ := 2 * car_price

noncomputable def profit_loss : ℝ := total_selling_price - total_cost_price
noncomputable def profit_loss_percent : ℝ := (profit_loss / total_cost_price) * 100

theorem trader_loss_percentage :
  ∃ ε > 0, abs (profit_loss_percent + 2.33) < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trader_loss_percentage_l798_79802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_plane_coloring_l798_79811

-- Define the type for points with rational coordinates
def RationalPoint := ℚ × ℚ

-- Define the distance function between two rational points
noncomputable def distance (p q : RationalPoint) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define a two-coloring of rational points
def TwoColoring := RationalPoint → Bool

theorem rational_plane_coloring
  (d : ℝ) (r s : ℚ) (h : d^2 = r^2 + s^2) :
  ∃ (coloring : TwoColoring),
    ∀ (p q : RationalPoint),
      distance p q = d → coloring p ≠ coloring q := by
  sorry

#check rational_plane_coloring

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_plane_coloring_l798_79811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_angle_special_case_l798_79827

/-- The cosine of the angle between two 2D vectors -/
noncomputable def cosine_angle (a b : ℝ × ℝ) : ℝ :=
  (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))

/-- Proof that the cosine of the angle between (3,4) and (5,12) is 63/65 -/
theorem cosine_angle_special_case : cosine_angle (3, 4) (5, 12) = 63 / 65 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_angle_special_case_l798_79827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_six_two_in_terms_of_a_l798_79828

theorem log_six_two_in_terms_of_a (a : ℝ) (h : Real.log 3 / Real.log 6 = a) : 
  Real.log 2 / Real.log 6 = 1 - a := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_six_two_in_terms_of_a_l798_79828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l798_79879

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def is_arithmetic_sequence (t : Triangle) : Prop :=
  t.b = t.a + 1 ∧ t.c = t.b + 1

def satisfies_sine_condition (t : Triangle) : Prop :=
  3 * Real.sin t.C = 5 * Real.sin t.A

def is_obtuse (t : Triangle) : Prop :=
  t.c^2 > t.a^2 + t.b^2

-- Helper function to calculate area
noncomputable def area (t : Triangle) : ℝ :=
  1/2 * t.a * t.b * Real.sin t.C

-- Main theorem
theorem triangle_properties (t : Triangle) 
  (h1 : is_arithmetic_sequence t)
  (h2 : satisfies_sine_condition t) :
  (area t = 6) ∧ (∃ a : ℕ, a > 0 ∧ is_obtuse { a := ↑a, b := ↑a + 1, c := ↑a + 2, A := t.A, B := t.B, C := t.C }) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l798_79879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_z_l798_79830

/-- The maximum value of z = 2x + y is 12, given that x and y satisfy certain conditions. -/
theorem max_value_of_z (x y : ℝ) (h : Set (ℝ × ℝ)) : ∃ (m : ℝ), ∀ z, z = 2 * x + y → z ≤ m ∧ m = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_z_l798_79830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_ethanol_percentage_l798_79874

/-- Calculates the optimal ethanol percentage in a fuel mixture --/
theorem optimal_ethanol_percentage
  (initial_volume : ℝ)
  (initial_ethanol_percentage : ℝ)
  (added_ethanol : ℝ)
  (h1 : initial_volume = 36)
  (h2 : initial_ethanol_percentage = 0.05)
  (h3 : added_ethanol = 2) :
  let initial_ethanol := initial_volume * initial_ethanol_percentage
  let total_ethanol := initial_ethanol + added_ethanol
  let final_volume := initial_volume + added_ethanol
  let final_ethanol_percentage := total_ethanol / final_volume
  final_ethanol_percentage = 0.1 := by
  sorry

#check optimal_ethanol_percentage

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_ethanol_percentage_l798_79874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_girls_left_of_boys_l798_79851

/-- The number of boys in the lineup -/
def num_boys : ℕ := 10

/-- The number of girls in the lineup -/
def num_girls : ℕ := 7

/-- The total number of people in the lineup -/
def total_people : ℕ := num_boys + num_girls

/-- The expected number of girls standing to the left of all boys -/
def expected_girls_left : ℚ := 7 / 11

/-- Theorem stating that the expected number of girls standing to the left of all boys
    in a random lineup of 10 boys and 7 girls is 7/11 -/
theorem expected_girls_left_of_boys :
  let lineup := Finset.range total_people
  ∀ (boys_positions : Finset (Finset ℕ)),
    boys_positions.card = num_boys →
    (∀ pos ∈ boys_positions, pos ⊆ lineup) →
    (Finset.sum boys_positions (λ s ↦ s.card) / boys_positions.card : ℚ) = expected_girls_left :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_girls_left_of_boys_l798_79851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_side_b_equals_five_l798_79866

-- Define the triangle ABC
structure Triangle where
  A : ℝ  -- Angle A
  B : ℝ  -- Angle B
  C : ℝ  -- Angle C
  a : ℝ  -- Side opposite to A
  b : ℝ  -- Side opposite to B
  c : ℝ  -- Side opposite to C
  area : ℝ -- Area of the triangle

-- Define the specific triangle from the problem
noncomputable def specificTriangle : Triangle where
  A := Real.pi / 2 - Real.pi / 4  -- Complement of B (45°)
  B := Real.pi / 4  -- 45° in radians
  C := Real.pi / 2  -- Remaining angle (90°)
  a := 1
  b := 5  -- The value we want to prove
  c := 4 * Real.sqrt 2  -- Not directly given, but can be derived
  area := 2

-- Theorem statement
theorem side_b_equals_five (t : Triangle) 
  (h1 : t.a = 1) 
  (h2 : t.B = Real.pi / 4) 
  (h3 : t.area = 2) : 
  t.b = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_side_b_equals_five_l798_79866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_in_square_grid_l798_79880

/-- The area of a shaded figure in a square grid --/
theorem shaded_area_in_square_grid (a : ℝ) (h : a > 0) :
  (7 * Real.pi / 6 + Real.sqrt 3 - 1) * a^2 =
    (4 * Real.pi * a^2) -
    ((4 * Real.pi / 3 - Real.sqrt 3) * a^2) -
    (2 * Real.pi * a^2) +
    ((Real.pi / 2 - 1) * a^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_in_square_grid_l798_79880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_ratio_l798_79838

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Three terms form a geometric sequence -/
def geometric_sequence (x y z : ℝ) : Prop :=
  y^2 = x * z

/-- The common ratio of a geometric sequence -/
noncomputable def common_ratio (x y : ℝ) : ℝ := y / x

theorem arithmetic_geometric_ratio (a : ℕ → ℝ) :
  arithmetic_sequence a →
  geometric_sequence (a 2) (a 3) (a 6) →
  (common_ratio (a 2) (a 3) = 3 ∨ common_ratio (a 2) (a 3) = 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_ratio_l798_79838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l798_79843

theorem triangle_problem (φ A B C a b c : Real) :
  (0 < φ) ∧ (φ < π) ∧
  (Real.sin (π/6 + φ) = 1) ∧
  (a^2 + b^2 - c^2 = a*b) ∧
  (Real.sin (A + π/2) = Real.sqrt 2/2) →
  (φ = π/3) ∧ (Real.sin (5*π/12) = (Real.sqrt 2 + Real.sqrt 6)/4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l798_79843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mess_expense_increase_l798_79898

theorem mess_expense_increase 
  (initial_students : ℕ) 
  (new_students : ℕ) 
  (original_expense : ℚ) 
  (per_head_decrease : ℚ) 
  (h1 : initial_students = 35)
  (h2 : new_students = 7)
  (h3 : original_expense = 400)
  (h4 : per_head_decrease = 1)
  : (initial_students + new_students) * 
    (original_expense / initial_students - per_head_decrease) - 
    original_expense = 38 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mess_expense_increase_l798_79898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nth_nonsquare_formula_l798_79872

/-- The integer closest to a real number x -/
noncomputable def closest_integer (x : ℝ) : ℤ := 
  ⌊x + 1/2⌋

/-- The nth positive nonsquare integer -/
def nth_nonsquare (n : ℕ) : ℕ :=
  sorry

theorem nth_nonsquare_formula (n : ℕ) :
  nth_nonsquare n = n + closest_integer (Real.sqrt (n : ℝ)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nth_nonsquare_formula_l798_79872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_equivalence_l798_79812

theorem divisibility_equivalence (n : ℕ) (h : n ≥ 2) :
  (n ∣ (Finset.sum (Finset.range (n-1)) (λ i ↦ (i+1)^(n-1))) + 1) ↔
  (∀ p : ℕ, Nat.Prime p → p ∣ n → (p ∣ (n/p - 1) ∧ (p-1) ∣ (n/p - 1))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_equivalence_l798_79812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l798_79826

/-- The function g(x) = (3^x - 1) / (3^x + 1) -/
noncomputable def g (x : ℝ) : ℝ := (3^x - 1) / (3^x + 1)

/-- g is an odd function -/
theorem g_is_odd : ∀ x : ℝ, g (-x) = -g x := by
  intro x
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l798_79826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_perimeter_is_56_l798_79837

/-- Represents the side lengths of nine squares in a rectangle dissection -/
structure SquareSides where
  a₁ : ℕ+
  a₂ : ℕ+
  a₄ : ℕ+
  a₅ : ℕ+
  a₆ : ℕ+
  a₇ : ℕ+
  a₈ : ℕ+
  a₉ : ℕ+
  rel₁ : a₁ + a₂ = a₄
  rel₂ : a₁ + a₄ = a₅
  rel₃ : a₂ + a₅ = a₆
  rel₄ : a₁ + a₆ = a₇
  rel₅ : a₄ + a₇ = a₈
  rel₆ : a₅ + a₈ = a₉

/-- The perimeter of a rectangle dissected into nine squares -/
def rectanglePerimeter (sides : SquareSides) : ℕ :=
  2 * (sides.a₉.val + sides.a₅.val)

/-- Theorem stating the perimeter of the rectangle is 56 -/
theorem rectangle_perimeter_is_56 (sides : SquareSides) 
  (width height : ℕ+) 
  (coprime : Nat.Coprime width height) 
  (width_def : width = sides.a₅ + sides.a₈) 
  (height_def : height = sides.a₉) : 
  rectanglePerimeter sides = 56 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_perimeter_is_56_l798_79837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_saved_approx_34_l798_79857

/-- Represents the hat pricing structure at the 2023 Maple County Fair --/
structure HatDeal where
  regularPrice : ℚ
  secondHatDiscount : ℚ
  thirdHatDiscount : ℚ
  additionalDiscount : ℚ
  additionalDiscountThreshold : ℚ

/-- Calculates the total cost for buying three hats under the deal --/
def totalCost (deal : HatDeal) : ℚ :=
  let firstHat := deal.regularPrice
  let secondHat := deal.regularPrice * (1 - deal.secondHatDiscount)
  let thirdHat := deal.regularPrice * (1 - deal.thirdHatDiscount)
  let subtotal := firstHat + secondHat + thirdHat
  if subtotal > deal.additionalDiscountThreshold then
    subtotal - deal.additionalDiscount
  else
    subtotal

/-- Calculates the percentage saved compared to buying three hats at regular price --/
def percentageSaved (deal : HatDeal) : ℚ :=
  let regularTotal := 3 * deal.regularPrice
  let actualTotal := totalCost deal
  (regularTotal - actualTotal) / regularTotal * 100

/-- The main theorem stating that the percentage saved is approximately 34% --/
theorem percentage_saved_approx_34 (ε : ℚ) (hε : ε > 0) :
  ∃ (deal : HatDeal),
    deal.regularPrice = 60 ∧
    deal.secondHatDiscount = 3/10 ∧
    deal.thirdHatDiscount = 11/20 ∧
    deal.additionalDiscount = 10 ∧
    deal.additionalDiscountThreshold = 100 ∧
    |percentageSaved deal - 34| < ε :=
by
  sorry

#eval totalCost { regularPrice := 60, secondHatDiscount := 3/10, thirdHatDiscount := 11/20, additionalDiscount := 10, additionalDiscountThreshold := 100 }
#eval percentageSaved { regularPrice := 60, secondHatDiscount := 3/10, thirdHatDiscount := 11/20, additionalDiscount := 10, additionalDiscountThreshold := 100 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_saved_approx_34_l798_79857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_are_coaxial_l798_79853

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A circle in the plane -/
structure Circle where
  center : Point
  radius : ℝ

/-- The intersection of a line with a line segment -/
noncomputable def lineIntersection (l : Line) (p1 p2 : Point) : Point :=
  sorry

/-- Check if three circles are coaxial -/
def areCoaxial (c1 c2 c3 : Circle) : Prop :=
  sorry

/-- The main theorem -/
theorem circles_are_coaxial 
  (l : Line) 
  (A B C D : Point) 
  (X : Point) (X' : Point) 
  (Y : Point) (Y' : Point) 
  (Z : Point) (Z' : Point)
  (h1 : X = lineIntersection l A B)
  (h2 : X' = lineIntersection l C D)
  (h3 : Y = lineIntersection l A D)
  (h4 : Y' = lineIntersection l B C)
  (h5 : Z = lineIntersection l A C)
  (h6 : Z' = lineIntersection l B D)
  : areCoaxial 
      { center := { x := (X.x + X'.x)/2, y := (X.y + X'.y)/2 },
        radius := ((X.x - X'.x)^2 + (X.y - X'.y)^2)/4 }
      { center := { x := (Y.x + Y'.x)/2, y := (Y.y + Y'.y)/2 },
        radius := ((Y.x - Y'.x)^2 + (Y.y - Y'.y)^2)/4 }
      { center := { x := (Z.x + Z'.x)/2, y := (Z.y + Z'.y)/2 },
        radius := ((Z.x - Z'.x)^2 + (Z.y - Z'.y)^2)/4 } :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_are_coaxial_l798_79853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_statue_selling_price_l798_79860

/-- The selling price of a statue given its original cost and profit percentage -/
noncomputable def selling_price (original_cost : ℝ) (profit_percentage : ℝ) : ℝ :=
  original_cost * (1 + profit_percentage / 100)

/-- Theorem: The selling price of a statue with original cost $407.41 and 35% profit is $550.00 -/
theorem statue_selling_price :
  (Int.floor (selling_price 407.41 35 * 100)) / 100 = 550 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_statue_selling_price_l798_79860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l798_79845

noncomputable section

-- Define the vectors a and b
def a (x : ℝ) : ℝ × ℝ := (Real.cos (3*x/2), Real.sin (3*x/2))
def b (x : ℝ) : ℝ × ℝ := (Real.cos (x/2), -Real.sin (x/2))

-- Define the dot product of two 2D vectors
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define the magnitude of a 2D vector
def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

-- Define the function f
def f (x m : ℝ) : ℝ := dot_product (a x) (b x) - m * magnitude (a x + b x) + 1

-- Define the function g
def g (x m : ℝ) : ℝ := f x m + 24/49 * m^2

-- Define the interval for x
def I : Set ℝ := Set.Icc (-Real.pi/3) (Real.pi/4)

theorem problem_solution :
  (∀ x ∈ I, f (Real.pi/6) 0 = 3/2) ∧
  (∃ m : ℝ, m = Real.sqrt 2 ∧ ∀ x ∈ I, f x m ≥ -1 ∧ ∃ x₀ ∈ I, f x₀ m = -1) ∧
  (∃ m₁ m₂ : ℝ, m₁ = 7 * Real.sqrt 2 / 6 ∧ m₂ = 7/4 ∧
    ∀ m, m₁ ≤ m ∧ m < m₂ → ∃ x₁ x₂ x₃ x₄ : ℝ, 
      x₁ ∈ I ∧ x₂ ∈ I ∧ x₃ ∈ I ∧ x₄ ∈ I ∧
      x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
      g x₁ m = 0 ∧ g x₂ m = 0 ∧ g x₃ m = 0 ∧ g x₄ m = 0) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l798_79845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_problem_l798_79896

-- Define the ellipse C
noncomputable def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  y^2 / a^2 + x^2 / b^2 = 1 ∧ a > b ∧ b > 0

-- Define the eccentricity
noncomputable def eccentricity (a b : ℝ) : ℝ := 
  Real.sqrt (1 - b^2 / a^2)

-- Define the area of triangle MNF₂
noncomputable def triangle_area (a b : ℝ) : ℝ := 
  Real.sqrt 3

-- Define the line l
def line (k m : ℝ) (x y : ℝ) : Prop :=
  y = k * x + m

-- Define the vector equation
def vector_equation (lambda : ℝ) (xa ya xb yb xp yp : ℝ) : Prop :=
  (xa, ya) + lambda • (xb, yb) = 4 • (xp, yp)

-- Main theorem
theorem ellipse_and_line_problem 
  (a b : ℝ) 
  (h_ellipse : ∀ x y, ellipse a b x y → x^2 + y^2/4 = 1) 
  (h_eccentricity : eccentricity a b = Real.sqrt 3 / 2)
  (h_area : triangle_area a b = Real.sqrt 3)
  (k m : ℝ) 
  (xa ya xb yb xp yp : ℝ)
  (h_line : line k m xa ya ∧ line k m xb yb ∧ line k m xp yp)
  (h_vector : ∃ lambda, vector_equation lambda xa ya xb yb xp yp) :
  m ∈ Set.union (Set.union (Set.Ioo (-2) (-1)) (Set.Ioo 1 2)) (Set.singleton 0) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_problem_l798_79896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_count_indeterminate_l798_79863

theorem candy_count_indeterminate (num_bags : ℕ) (cookies_per_bag : ℕ) (total_cookies : ℕ) :
  num_bags = 26 →
  cookies_per_bag = 2 →
  total_cookies = 52 →
  num_bags * cookies_per_bag = total_cookies →
  ∀ (candy_count : ℕ), True :=
by
  intros h1 h2 h3 h4 candy_count
  exact True.intro


end NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_count_indeterminate_l798_79863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_trig_expression_equality_condition_l798_79864

theorem min_value_trig_expression (x : ℝ) (h : 0 < x ∧ x < Real.pi / 2) :
  (Real.sin x + 1 / Real.cos x)^2 + (Real.cos x + 1 / Real.sin x)^2 ≥ 3 :=
by sorry

theorem equality_condition (x : ℝ) :
  (Real.sin x + 1 / Real.cos x)^2 + (Real.cos x + 1 / Real.sin x)^2 = 3 ↔ x = Real.pi / 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_trig_expression_equality_condition_l798_79864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_f_monotonic_increase_l798_79895

noncomputable def f (x : ℝ) := Real.sqrt 3 * Real.sin (2 * x) - 2 * (Real.sin x) ^ 2

theorem f_properties :
  ∃ (T : ℝ), T > 0 ∧ 
  (∀ x, f (x + T) = f x) ∧
  (∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T') ∧
  (∀ x ∈ Set.Icc (-Real.pi/3) (Real.pi/3), f x ≥ -3) ∧
  f (-Real.pi/3) = -3 := by
  sorry

-- Theorem for the interval of monotonic increase
theorem f_monotonic_increase :
  ∀ k : ℤ, ∀ x ∈ Set.Icc (-Real.pi/3 + k * Real.pi) (Real.pi/6 + k * Real.pi),
  ∀ y ∈ Set.Icc (-Real.pi/3 + k * Real.pi) (Real.pi/6 + k * Real.pi),
  x < y → f x < f y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_f_monotonic_increase_l798_79895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factor_sum_of_four_digit_permutations_l798_79861

def is_distinct (a b c d : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

def is_valid_digit (x : ℕ) : Prop :=
  1 ≤ x ∧ x ≤ 9

def sum_of_permutations (a b c d : ℕ) : ℕ :=
  6666 * (a + b + c + d)

noncomputable def largest_prime_factor_sum (n : ℕ) : ℕ :=
  (Nat.factors n).toFinset.sum id

theorem largest_prime_factor_sum_of_four_digit_permutations :
  ∀ a b c d : ℕ,
    is_distinct a b c d →
    is_valid_digit a →
    is_valid_digit b →
    is_valid_digit c →
    is_valid_digit d →
    largest_prime_factor_sum (sum_of_permutations a b c d) ≤ 146 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factor_sum_of_four_digit_permutations_l798_79861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_small_triangle_l798_79832

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Helper function to calculate the area of a triangle given three points -/
noncomputable def triangle_area (p1 p2 p3 : Point) : ℝ :=
  (1/2) * abs ((p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y))

/-- The theorem statement -/
theorem existence_of_small_triangle (points : Finset Point) 
  (h1 : points.card = 500)
  (h2 : ∀ p ∈ points, 0 ≤ p.x ∧ p.x ≤ 2 ∧ 0 ≤ p.y ∧ p.y ≤ 1) :
  ∃ p1 p2 p3, p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points ∧
  p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧
  triangle_area p1 p2 p3 < 50 / 10000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_small_triangle_l798_79832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l798_79891

-- Define the function f
def f (a x : ℝ) : ℝ := |a - x| * x

-- Part 1
theorem part_one : 
  let a := 1
  {x : ℝ | f a x > 2 * x} = Set.union (Set.Ioo (-1) 0) (Set.Ioi 3) := by sorry

-- Part 2
theorem part_two (a : ℝ) :
  (∀ x ∈ Set.Icc 1 4, f a x - x < 4) → 2 < a ∧ a < 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l798_79891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_number_l798_79876

def is_six_digit (n : ℕ) : Prop := 100000 ≤ n ∧ n < 1000000

def has_digit (n : ℕ) (d : ℕ) : Prop :=
  ∃ k, n / (10^k) % 10 = d

def swap_digits (n : ℕ) (i j : ℕ) : ℕ :=
  let d₁ := (n / (10^i)) % 10
  let d₂ := (n / (10^j)) % 10
  n + (d₂ - d₁) * (10^i - 10^j)

def satisfies_conditions (n : ℕ) : Prop :=
  is_six_digit n ∧
  has_digit n 7 ∧
  n % 9 = 0 ∧
  ∃ i j k l m o p q : ℕ,
    i ≠ j ∧ k ≠ l ∧ m ≠ o ∧ p ≠ q ∧
    (swap_digits n i j - n) % 2525 = 0 ∧
    (swap_digits n k l - n) % 2168 = 0 ∧
    (swap_digits n m o - n) % 4375 = 0 ∧
    (swap_digits n p q - n) % 6875 = 0 ∧
    swap_digits n i j ≠ n ∧
    swap_digits n k l ≠ n ∧
    swap_digits n m o ≠ n ∧
    swap_digits n p q ≠ n

theorem unique_number : ∃! n, satisfies_conditions n ∧ n = 924741 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_number_l798_79876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x₄_x₁_diff_l798_79833

-- Define the quadratic functions f and g
noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

-- Define the relationship between f and g
axiom g_def : ∀ x, g x = -f (120 - x)

-- Define the x-intercepts
noncomputable def x₁ : ℝ := sorry
noncomputable def x₂ : ℝ := sorry
noncomputable def x₃ : ℝ := sorry
noncomputable def x₄ : ℝ := sorry

-- The x-intercepts are in increasing order
axiom x_order : x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < x₄

-- The difference between x₃ and x₂
axiom x₃_x₂_diff : x₃ - x₂ = 180

-- The graph of g contains the vertex of f
axiom vertex_condition : ∃ v, f v = g v ∧ ∀ x, f x ≤ f v

-- Theorem to prove
theorem x₄_x₁_diff : x₄ - x₁ = 540 + 360 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x₄_x₁_diff_l798_79833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_badminton_probability_l798_79815

theorem badminton_probability (m n : ℕ) 
  (hm : m > 0) (hn : n > 0) (hcoprime : Nat.Coprime m n)
  (hwin : (9 : ℚ) / 10 = m / (4 * n - 3 * m)) : m + n = 73 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_badminton_probability_l798_79815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_M_to_C₁_l798_79816

-- Define the circle C₁
def C₁ (x y : ℝ) : Prop := x^2 + y^2 - 2*y = 0

-- Define the point M
def M : ℝ × ℝ := (2, 0)

-- Define the distance between two points
noncomputable def distance (p₁ p₂ : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)

-- Theorem statement
theorem max_distance_M_to_C₁ :
  ∃ (max_dist : ℝ), max_dist = Real.sqrt 5 + 1 ∧
    ∀ (x y : ℝ), C₁ x y → distance M (x, y) ≤ max_dist := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_M_to_C₁_l798_79816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_book_width_l798_79844

noncomputable def book_widths : List ℝ := [3, 4, 3/4, 1.5, 7, 2, 5.25, 8]

theorem average_book_width :
  (book_widths.sum / book_widths.length : ℝ) = 3.9375 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_book_width_l798_79844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bombardment_death_percentage_approx_l798_79875

/-- Represents the percentage of people who died by bombardment -/
noncomputable def bombardment_death_percentage (initial_population final_population : ℕ) : ℝ :=
  let x := (1 - (final_population : ℝ) / initial_population) / (0.8 : ℝ) * 100
  ⌊x * 100⌋ / 100

/-- Theorem stating the percentage of people who died by bombardment -/
theorem bombardment_death_percentage_approx :
  abs (bombardment_death_percentage 4599 3312 - 9.98) < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bombardment_death_percentage_approx_l798_79875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l798_79890

theorem solve_exponential_equation (y : ℝ) (x : ℝ) : (3 : ℝ)^(2*y) = (3 : ℝ)^x ∧ y = 6 → x = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l798_79890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_a_2_range_of_a_l798_79807

-- Define the function f(x, a)
def f (x a : ℝ) : ℝ := |x - a^2| + |x - 2*a + 1|

-- Part 1: Solution set when a = 2
theorem solution_set_a_2 :
  {x : ℝ | f x 2 ≥ 4} = {x : ℝ | x ≤ 3/2 ∨ x ≥ 11/2} := by sorry

-- Part 2: Range of values for a
theorem range_of_a :
  {a : ℝ | ∀ x, f x a ≥ 4} = Set.Iic (-1) ∪ Set.Ici 3 := by sorry

-- Helper definitions for the sets in Part 2
def A : Set ℝ := {a : ℝ | ∀ x, f x a ≥ 4}
def B : Set ℝ := Set.Iic (-1) ∪ Set.Ici 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_a_2_range_of_a_l798_79807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_minimal_positive_solutions_l798_79847

noncomputable def floor (x : ℝ) := ⌊x⌋

def is_solution (x : ℝ) : Prop :=
  x - floor x = 1 / (2 * floor x)

def is_minimal_positive_solution (x : ℝ) : Prop :=
  is_solution x ∧ x > 0 ∧ ∀ y, 0 < y ∧ y < x → ¬is_solution y

theorem sum_of_minimal_positive_solutions :
  ∃ (x y z : ℝ),
    is_minimal_positive_solution x ∧
    is_minimal_positive_solution y ∧
    is_minimal_positive_solution z ∧
    x < y ∧ y < z ∧
    x + y + z = 83 / 12 := by
  sorry

#check sum_of_minimal_positive_solutions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_minimal_positive_solutions_l798_79847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_vertex_to_asymptote_l798_79862

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 2 - y^2 / 4 = 1

-- Define the asymptote
noncomputable def asymptote (x y : ℝ) : Prop := y = Real.sqrt 2 * x

-- Define the vertex
noncomputable def vertex : ℝ × ℝ := (Real.sqrt 2, 0)

-- Theorem statement
theorem distance_vertex_to_asymptote :
  let (x₀, y₀) := vertex
  ∃ (d : ℝ), d = (2 * Real.sqrt 3) / 3 ∧
    d = abs (Real.sqrt 2 * x₀ - y₀) / Real.sqrt ((Real.sqrt 2)^2 + 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_vertex_to_asymptote_l798_79862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overlap_area_is_ten_l798_79800

/-- Represents a square with a given side length -/
structure Square where
  side : ℝ
  side_positive : side > 0

/-- Calculates the area of a square -/
noncomputable def Square.area (s : Square) : ℝ := s.side * s.side

/-- Represents the configuration of two adjoining squares -/
structure SquareConfiguration where
  large_square : Square
  small_square : Square
  overlap_height : ℝ
  overlap_height_valid : overlap_height > 0 ∧ overlap_height ≤ small_square.side

/-- Calculates the area of overlap between the two squares -/
noncomputable def overlap_area (config : SquareConfiguration) : ℝ :=
  config.small_square.area - (1/2 * config.overlap_height * config.small_square.side)

/-- Theorem statement for the area of overlap -/
theorem overlap_area_is_ten (config : SquareConfiguration) 
  (h1 : config.large_square.side = 12)
  (h2 : config.small_square.side = 4)
  (h3 : config.overlap_height = 3) : 
  overlap_area config = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_overlap_area_is_ten_l798_79800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_even_numbers_between_500_and_800_l798_79884

theorem count_even_numbers_between_500_and_800 : 
  (Finset.filter (fun x => x % 2 = 0 ∧ x > 500 ∧ x < 800) (Finset.range 800)).card = 149 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_even_numbers_between_500_and_800_l798_79884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tray_height_l798_79804

/-- The height of a tray formed from a rectangular paper with specific cuts -/
theorem tray_height (paper_length paper_width cut_distance : ℝ) :
  paper_length = 120 →
  paper_width = 80 →
  cut_distance = 5 →
  (cut_distance * Real.sqrt 3) / 2 = (5 * Real.sqrt 3) / 2 :=
by
  intros h_length h_width h_cut
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tray_height_l798_79804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_in_range_l798_79849

theorem count_integers_in_range : 
  (Finset.filter (fun n : ℕ => 300 < n^2 ∧ n^2 < 1200) (Finset.range 35)).card = 17 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_in_range_l798_79849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_square_equal_area_l798_79840

/-- Given an equilateral triangle and a square with equal areas,
    where s is the side length of the square,
    prove that the height of the triangle is s * sqrt(2 * sqrt(3)) -/
theorem equilateral_triangle_square_equal_area
  (s : ℝ) -- side length of the square
  (a : ℝ) -- side length of the equilateral triangle
  (h_positive : s > 0) -- assume s is positive
  (h_area_eq : (Real.sqrt 3 / 4) * a^2 = s^2) -- areas are equal
  : (Real.sqrt 3 / 2) * a = s * Real.sqrt (2 * Real.sqrt 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_square_equal_area_l798_79840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sinusoidal_amplitude_l798_79854

/-- Given a sinusoidal function y = a * Real.sin(b * x + c) + d where a, b, c, and d are positive constants,
    if the function oscillates between 5 and -3, then a = 4. -/
theorem sinusoidal_amplitude (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h_max : ∀ x, a * Real.sin (b * x + c) + d ≤ 5)
  (h_min : ∀ x, -3 ≤ a * Real.sin (b * x + c) + d)
  (h_reaches_max : ∃ x, a * Real.sin (b * x + c) + d = 5)
  (h_reaches_min : ∃ x, a * Real.sin (b * x + c) + d = -3) :
  a = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sinusoidal_amplitude_l798_79854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_m_for_sum_triple_l798_79869

def S (m : ℕ) : Set ℕ := {n : ℕ | 5 ≤ n ∧ n ≤ m}

def has_sum_triple (A : Set ℕ) : Prop :=
  ∃ a b c, a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ a + b = c

theorem minimal_m_for_sum_triple :
  ∀ m : ℕ, m ≥ 5 →
    (∀ A B : Set ℕ, A ∪ B = S m → A ∩ B = ∅ → 
      has_sum_triple A ∨ has_sum_triple B) ↔ m ≥ 19 := by
  sorry

#check minimal_m_for_sum_triple

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_m_for_sum_triple_l798_79869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l798_79883

/-- Given a hyperbola Γ with the equation (y^2 / a^2) - (x^2 / b^2) = 1,
    where a > 0, b > 0, and a^2, b^2, c^2 form an arithmetic sequence,
    the equation of the asymptotes is y = ± (√2/2)x -/
theorem hyperbola_asymptotes (a b c : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_arith : 2 * b^2 = a^2 + c^2) (h_hyp : c^2 = a^2 + b^2) :
  ∃ (k : ℝ), k = Real.sqrt 2 / 2 ∧ 
  (∀ (x y : ℝ), (y^2 / a^2 - x^2 / b^2 = 1) → 
    (y = k * x ∨ y = -k * x)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l798_79883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fence_painting_problem_l798_79813

/-- Time taken by Jamshid and Taimour to paint a fence together -/
noncomputable def combined_time : ℝ := 5

/-- Represents the time taken by Taimour to paint the fence alone -/
noncomputable def taimour_time : ℝ := 15

/-- Represents the time taken by Jamshid to paint the fence alone -/
noncomputable def jamshid_time (t : ℝ) : ℝ := t / 2

/-- The rate at which Taimour paints the fence -/
noncomputable def taimour_rate (t : ℝ) : ℝ := 1 / t

/-- The rate at which Jamshid paints the fence -/
noncomputable def jamshid_rate (t : ℝ) : ℝ := 1 / (jamshid_time t)

/-- The combined rate at which Jamshid and Taimour paint the fence -/
noncomputable def combined_rate (t : ℝ) : ℝ := taimour_rate t + jamshid_rate t

theorem fence_painting_problem :
  combined_rate taimour_time = 1 / combined_time :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fence_painting_problem_l798_79813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_f_l798_79878

-- Define the function f(x) = (1/a)^x
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/a)^x

-- State the theorem
theorem derivative_of_f (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  deriv (f a) = λ x ↦ -a^(-x) * Real.log a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_f_l798_79878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_restaurant_bill_proof_l798_79839

/-- Represents the total bill at a restaurant -/
noncomputable def total_bill (appetizers dessert mary_drinks nancy_drinks fred_drinks steve_drinks : ℚ) : ℚ :=
  appetizers + dessert + mary_drinks + nancy_drinks + fred_drinks + steve_drinks

/-- The amount each person should pay without considering dessert -/
noncomputable def equal_share (total : ℚ) (num_people : ℕ) : ℚ :=
  total / num_people

theorem restaurant_bill_proof :
  let appetizers : ℚ := 24
  let mary_drinks : ℚ := 12
  let nancy_drinks : ℚ := 12
  let fred_drinks : ℚ := 10
  let steve_drinks : ℚ := 5
  let fred_paid : ℚ := 35
  let steve_paid : ℚ := 35
  let mary_paid : ℚ := 40
  let total_paid : ℚ := fred_paid + steve_paid + mary_paid
  let remaining : ℚ := total_paid - appetizers - mary_drinks - nancy_drinks - fred_drinks - steve_drinks
  let share_without_dessert : ℚ := equal_share remaining 4
  let dessert : ℚ := (fred_paid - share_without_dessert) + (steve_paid - share_without_dessert) + (mary_paid - share_without_dessert)
  total_bill appetizers dessert mary_drinks nancy_drinks fred_drinks steve_drinks = 184.75
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_restaurant_bill_proof_l798_79839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_waiting_time_difference_l798_79850

/-- Proves that the difference in total waiting time between swings and slide is 180 seconds -/
theorem waiting_time_difference : 
  let kids_swings : ℕ := 6
  let kids_slide : ℕ := 4 * kids_swings
  let wait_time_swings : ℚ := 3.5 * 60 -- in seconds
  let wait_time_slide : ℕ := 45 -- in seconds
  let total_time_swings : ℚ := kids_swings * wait_time_swings
  let total_time_slide : ℕ := kids_slide * wait_time_slide
  (total_time_swings : ℝ) - (total_time_slide : ℝ) = 180 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_waiting_time_difference_l798_79850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_equivalence_l798_79823

open Real

/-- The function f(x) = (x^2 + 1) / x -/
noncomputable def f (x : ℝ) : ℝ := (x^2 + 1) / x

/-- The function g(x) = x / e^x -/
noncomputable def g (x : ℝ) : ℝ := x / (exp x)

/-- Theorem stating the equivalence of the inequality and the range of k -/
theorem inequality_equivalence (k : ℝ) :
  (k > 0) →
  (∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → g x₁ / k ≤ f x₂ / (k + 1)) ↔
  k ≥ 1 / (2 * exp 1 - 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_equivalence_l798_79823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_not_exist_at_zero_l798_79819

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then Real.exp (x * Real.sin (5 / x)) - 1 else 0

-- State the theorem
theorem derivative_not_exist_at_zero :
  ¬ ∃ (L : ℝ), ∀ ε > 0, ∃ δ > 0, ∀ x ≠ 0, |x| < δ → |(f x - f 0) / x - L| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_not_exist_at_zero_l798_79819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cake_eaters_l798_79881

theorem cake_eaters (n : ℕ) (h1 : n > 0) : 
  (∃ (portions : Fin n → ℚ), 
    (∀ i, 0 < portions i) ∧ 
    (∀ i, portions i ≤ 1/11) ∧
    (∃ j, portions j = 1/11) ∧
    (∀ i, 1/14 ≤ portions i) ∧
    (∃ k, portions k = 1/14) ∧
    (Finset.sum (Finset.univ : Finset (Fin n)) portions = 1)) ↔ 
  (n = 12 ∨ n = 13) :=
by
  sorry

#check cake_eaters

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cake_eaters_l798_79881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_sum_l798_79820

/-- A function representing f(x) = e^x + x - a --/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x + x - a

/-- The derivative of f(x) --/
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := Real.exp x + 1

/-- Theorem stating that if y = 2x + b is tangent to f(x) = e^x + x - a, then a + b = 1 --/
theorem tangent_line_sum (a b : ℝ) :
  (∃ x₀ : ℝ, f_deriv a x₀ = 2 ∧ f a x₀ = 2 * x₀ + b) →
  a + b = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_sum_l798_79820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_power_identity_l798_79806

/-- For any positive natural number n, (1/2 + (√3/2)i)^n = cos(nπ/3) + i*sin(nπ/3) -/
theorem complex_power_identity (n : ℕ) :
  (1/2 : ℂ) + Complex.I * ((Real.sqrt 3)/2) ^ n = Complex.cos (n * Real.pi/3) + Complex.I * Complex.sin (n * Real.pi/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_power_identity_l798_79806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_pole_in_stadium_l798_79871

/-- The length of the longest pole that can be placed in a rectangular stadium -/
noncomputable def longest_pole (length width height : ℝ) : ℝ :=
  Real.sqrt (length^2 + width^2 + height^2)

/-- Theorem: The length of the longest pole in a 24m x 18m x 16m stadium is 34m -/
theorem longest_pole_in_stadium : 
  longest_pole 24 18 16 = 34 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval longest_pole 24 18 16

end NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_pole_in_stadium_l798_79871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_property_l798_79821

-- Define the curve C
def curve_C (x y : ℝ) : Prop := y^2 = 2*x

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y - 2 = 0

-- Define point P
def point_P : ℝ × ℝ := (-2, -4)

-- Define the theorem
theorem intersection_property :
  ∀ A B : ℝ × ℝ,
  curve_C A.1 A.2 ∧ curve_C B.1 B.2 ∧
  line_l A.1 A.2 ∧ line_l B.1 B.2 →
  let PA := Real.sqrt ((A.1 - point_P.1)^2 + (A.2 - point_P.2)^2)
  let PB := Real.sqrt ((B.1 - point_P.1)^2 + (B.2 - point_P.2)^2)
  let AB := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  PA * PB = AB^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_property_l798_79821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_min_dot_product_l798_79893

structure IsoscelesTrapezoid (A B C D : ℝ × ℝ) : Prop where
  parallel : (B.2 - A.2) / (B.1 - A.1) = (C.2 - D.2) / (C.1 - D.1)
  isosceles : (B.1 - C.1)^2 + (B.2 - C.2)^2 = (A.1 - D.1)^2 + (A.2 - D.2)^2

def distance (P Q : ℝ × ℝ) : ℝ :=
  ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)^(1/2)

def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

theorem isosceles_trapezoid_min_dot_product 
  (A B C D : ℝ × ℝ) 
  (h_trapezoid : IsoscelesTrapezoid A B C D)
  (h_AB : distance A B = 2)
  (h_BC : distance B C = 1)
  (h_angle : Real.cos (Real.arccos ((B.1 - A.1) * (C.1 - B.1) + (B.2 - A.2) * (C.2 - B.2)) / (distance A B * distance B C)) = 1/2)
  : ∀ (lambda : ℝ), 
    let E : ℝ × ℝ := (B.1 + lambda * (C.1 - B.1), B.2 + lambda * (C.2 - B.2))
    let F : ℝ × ℝ := (D.1 + (1/(9*lambda)) * (C.1 - D.1), D.2 + (1/(9*lambda)) * (C.2 - D.2))
    dot_product (E.1 - A.1, E.2 - A.2) (F.1 - A.1, F.2 - A.2) ≥ 29/18 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_min_dot_product_l798_79893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_price_approximation_l798_79885

def initial_price : ℝ := 9795.3216374269
def discount1 : ℝ := 0.20
def discount2 : ℝ := 0.10
def discount3 : ℝ := 0.05

def final_price : ℝ :=
  initial_price * (1 - discount1) * (1 - discount2) * (1 - discount3)

theorem final_price_approximation :
  ∃ ε > 0, |final_price - 6700| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_price_approximation_l798_79885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_solution_set_l798_79822

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

-- Define the interval (3, 4.5]
def interval : Set ℝ := {x | 3 < x ∧ x ≤ 4.5}

-- Define the solution set
def solution_set : Set ℤ := {y | ∃ x ∈ interval, floor x = y}

-- Theorem statement
theorem floor_solution_set : solution_set = {3, 4} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_solution_set_l798_79822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l798_79810

noncomputable def f (x y : ℝ) : ℝ :=
  Real.sqrt (x^2 + y^2 - 6*y + 9) + Real.sqrt (x^2 + y^2 + 2*Real.sqrt 3*x + 3) + Real.sqrt (x^2 + y^2 - 2*Real.sqrt 3*x + 3)

theorem f_minimum_value :
  ∀ x y : ℝ, f x y ≥ 6 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l798_79810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_de_moivre_de_moivre_explicit_l798_79882

theorem de_moivre (θ : ℝ) (n : ℕ) :
  let z : ℂ := Complex.ofReal (Real.cos θ) + Complex.I * Complex.ofReal (Real.sin θ)
  (z^n : ℂ) = Complex.ofReal (Real.cos (n * θ)) + Complex.I * Complex.ofReal (Real.sin (n * θ)) := by sorry

theorem de_moivre_explicit (θ : ℝ) (n : ℕ) :
  let z : ℂ := Complex.exp (θ * Complex.I)
  (z^n : ℂ) = Complex.exp (n * θ * Complex.I) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_de_moivre_de_moivre_explicit_l798_79882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_l798_79835

/-- The sum of the geometric series for -1 < r < 1 -/
noncomputable def T (r : ℝ) : ℝ := 18 / (1 - r)

/-- Theorem: For -1 < b < 1, if T(b)T(-b) = 2916, then T(b) + T(-b) = 324 -/
theorem geometric_series_sum (b : ℝ) (h1 : -1 < b) (h2 : b < 1) 
  (h3 : T b * T (-b) = 2916) : T b + T (-b) = 324 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_l798_79835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_section_area_theorem_l798_79856

/-- Represents a cube with edge length a -/
structure Cube (a : ℝ) where
  edge_length : a > 0

/-- Represents a plane passing through the midpoint of a diagonal of the cube and perpendicular to it -/
structure DiagonalPlane (a : ℝ) (cube : Cube a) where
  passes_through_midpoint : True
  perpendicular_to_diagonal : True

/-- The area of the cross-section formed by the diagonal plane -/
noncomputable def cross_section_area (a : ℝ) (cube : Cube a) (plane : DiagonalPlane a cube) : ℝ :=
  (3 * a^2 * Real.sqrt 3) / 4

/-- Theorem stating that the area of the cross-section is (3*a^2*√3)/4 -/
theorem cross_section_area_theorem (a : ℝ) (cube : Cube a) (plane : DiagonalPlane a cube) :
  cross_section_area a cube plane = (3 * a^2 * Real.sqrt 3) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_section_area_theorem_l798_79856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_theorem_l798_79897

-- Define the circle C
def circle_C (x y m : ℝ) : Prop := x^2 + y^2 - 4*x + 2*y + m = 0

-- Define the line passing through P(0,2) with slope k
def line_P (x y k : ℝ) : Prop := y = k*x + 2

-- Define the intersection points of the circle and y-axis
def intersects_y_axis (m : ℝ) : Prop := ∃ (y₁ y₂ : ℝ), circle_C 0 y₁ m ∧ circle_C 0 y₂ m ∧ y₁ ≠ y₂

-- Define right angle at center
def right_angle_at_center (m : ℝ) : Prop := 
  ∃ (y₁ y₂ : ℝ), circle_C 0 y₁ m ∧ circle_C 0 y₂ m ∧ 
  let c_x := 2
  let c_y := -1
  (c_x - 0)*(c_x - 0) + (c_y - y₁)*(c_y - y₂) = 0

-- Define the theorem
theorem circle_intersection_theorem (m k : ℝ) : 
  intersects_y_axis m →
  right_angle_at_center m →
  (
    -- Part I
    m = -3 ∧
    -- Part II
    (∀ x₁ y₁ x₂ y₂ : ℝ, 
      circle_C x₁ y₁ m → circle_C x₂ y₂ m → 
      line_P x₁ y₁ k → line_P x₂ y₂ k →
      (x₁ - x₂)^2 + (y₁ - y₂)^2 ≥ 16 → 
      k ≤ -5/12) ∧
    -- Part III
    (∀ x₁ y₁ x₂ y₂ : ℝ,
      circle_C x₁ y₁ m → circle_C x₂ y₂ m →
      line_P x₁ y₁ k → line_P x₂ y₂ k →
      (∃ (t : ℝ), x₁ + x₂ = 2*t ∧ y₁ + y₂ = -t) →
      k = -3/2)
  ) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_theorem_l798_79897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trickster_identification_theorem_l798_79887

/-- Represents an inhabitant of the village -/
inductive Inhabitant
| Knight
| Trickster
deriving BEq, Repr

/-- Represents a question about a group of inhabitants -/
structure Question where
  group : List Inhabitant
deriving Repr

/-- A strategy is a function that takes the current state and returns the next question to ask -/
def Strategy := List Question → Question

/-- The result of asking a question -/
inductive Answer
| Yes
| No

/-- A function that simulates asking a question to an inhabitant -/
def ask (q : Question) (i : Inhabitant) : Answer :=
  match i with
  | Inhabitant.Knight => if q.group.all (· == Inhabitant.Knight) then Answer.Yes else Answer.No
  | Inhabitant.Trickster => sorry -- Can return either Yes or No

/-- A function that determines if a strategy correctly identifies the tricksters -/
def identifiesTricksters (s : Strategy) (village : List Inhabitant) : Prop :=
  sorry -- Implementation details omitted

/-- The main theorem stating that it's possible to identify tricksters with at most 16 questions -/
theorem trickster_identification_theorem :
  ∃ (s : Strategy),
    ∀ (village : List Inhabitant),
      village.length = 65 →
      (village.filter (· == Inhabitant.Trickster)).length = 2 →
      identifiesTricksters s village ∧
      (List.length (sorry : List Question)) ≤ 16 := by
  sorry

#check trickster_identification_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trickster_identification_theorem_l798_79887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shifted_sine_function_l798_79867

/-- Given a function f(x) = √3 * sin(ωx + φ) with specified properties,
    prove that shifting it right by 1/3 unit results in g(x) = √3 * sin(πx/2 - π/6) -/
theorem shifted_sine_function 
  (ω φ : ℝ) 
  (h_ω_pos : ω > 0) 
  (h_φ_bound : |φ| < π/2) 
  (f : ℝ → ℝ) 
  (hf : f = λ x ↦ Real.sqrt 3 * Real.sin (ω * x + φ))
  (h_symmetry : f 2 = 0) 
  (h_distance : ∃ x : ℝ, f (x + π/ω) = -f x ∧ |f x - f (x + π/ω)| = 4) 
  (g : ℝ → ℝ) 
  (hg : g = λ x ↦ f (x - 1/3)) :
  g = λ x ↦ Real.sqrt 3 * Real.sin (π/2 * x - π/6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shifted_sine_function_l798_79867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_three_zeros_a_range_l798_79842

/-- The function f(x) defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x + Real.log x - x^2 / (x - Real.log x)

/-- The theorem stating the range of a when f has three different zeros -/
theorem f_three_zeros_a_range (a : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f a x = 0 ∧ f a y = 0 ∧ f a z = 0) →
  1 < a ∧ a < Real.exp 1 / (Real.exp 1 - 1) - 1 / Real.exp 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_three_zeros_a_range_l798_79842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l798_79855

/-- The time taken for a train to cross a man walking in the opposite direction -/
theorem train_crossing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) :
  train_length = 500 →
  train_speed = 174.98560115190784 →
  man_speed = 5 →
  abs ((train_length / ((train_speed + man_speed) * 1000 / 3600)) - 10.0008) < 0.0001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l798_79855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monoid_properties_l798_79809

variable {M : Type} [Monoid M]

def invertible (a : M) : Prop := ∃ b : M, a * b = 1 ∧ b * a = 1

def M_a (a : M) : Set M := {x : M | a * x * x * a = 1}

theorem monoid_properties (a : M) (h : invertible a) :
  (∃ x, x ∈ M_a a) ∧
  (∀ b, b ∈ M_a a → invertible b → (∃ b_inv, b * b_inv = 1 ∧ b_inv * b = 1 ∧ (b_inv ∈ M_a a ↔ a * a * a * a = 1))) ∧
  (∀ x, x ∈ M_a a → (∀ y z, y ∈ M_a a → z ∈ M_a a → y * z ∈ M_a a) → (∃ e_a, e_a ∈ M_a a ∧ ∀ y, y ∈ M_a a → e_a * y = y ∧ y * e_a = y) → x * x = 1) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monoid_properties_l798_79809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_population_difference_l798_79889

/-- Proves the difference in population between Lake View and Seattle --/
theorem population_difference 
  (seattle : ℕ)  -- Population of Seattle
  (boise : ℕ)    -- Population of Boise
  (lakeview : ℕ) -- Population of Lake View
  (h1 : boise = (3 * seattle) / 5)  -- Boise has 3/5 times Seattle's population
  (h2 : lakeview > seattle)         -- Lake View has more people than Seattle
  (h3 : lakeview = 24000)           -- Lake View's population
  (h4 : seattle + boise + lakeview = 56000)  -- Total population
  : lakeview - seattle = 4000 := by
  sorry

#check population_difference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_population_difference_l798_79889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vessel_base_length_is_10_l798_79805

/-- The length of a rectangular vessel's base, given specific conditions -/
noncomputable def vessel_base_length (cube_edge : ℝ) (vessel_width : ℝ) (water_rise : ℝ) : ℝ :=
  (cube_edge ^ 3) / (vessel_width * water_rise)

/-- Theorem stating that the vessel base length is 10 cm under given conditions -/
theorem vessel_base_length_is_10 :
  vessel_base_length 5 5 2.5 = 10 := by
  -- Unfold the definition of vessel_base_length
  unfold vessel_base_length
  -- Simplify the expression
  simp
  -- Prove the equality
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vessel_base_length_is_10_l798_79805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_between_sqrt3_and_3pi_l798_79817

theorem count_integers_between_sqrt3_and_3pi :
  (Finset.filter (fun n : ℕ => (↑n : ℝ) > Real.sqrt 3 ∧ (↑n : ℝ) < 3 * Real.pi) (Finset.range 10)).card = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_between_sqrt3_and_3pi_l798_79817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_three_digit_numbers_product_four_zeros_l798_79886

/-- A three-digit number -/
def ThreeDigitNumber := { n : ℕ // 100 ≤ n ∧ n < 1000 }

/-- Check if a list of natural numbers uses only distinct digits -/
def distinctDigits (numbers : List ℕ) : Prop :=
  let digits := numbers.bind (λ n => n.digits 10)
  digits.Nodup ∧ digits.length = 9

/-- The main theorem -/
theorem exist_three_digit_numbers_product_four_zeros :
  ∃ (a b c : ThreeDigitNumber),
    distinctDigits [a.val, b.val, c.val] ∧
    (a.val * b.val * c.val) % 10000 = 0 := by
  sorry

#check exist_three_digit_numbers_product_four_zeros

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_three_digit_numbers_product_four_zeros_l798_79886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_triangle_area_l798_79888

/-- The area of the shaded triangle in a figure composed of rectangles --/
theorem shaded_triangle_area (h₁ h₂ h₃ w₁ w₂ w₃ : ℝ) 
  (h₁_eq : h₁ = 4) (w₁_eq : w₁ = 3) 
  (h₂_eq : h₂ = 12) (w₂_eq : w₂ = 15) 
  (h₃_eq : h₃ = 16) (w₃_eq : w₃ = 3) : 
  (1 / 2) * (w₁ + w₂ + w₃) * ((h₂ * w₂) / (w₁ + w₂)) = 1536 / 13 := by
  sorry

#eval (1 / 2 : ℚ) * 21 * (12 * 15 / 18) -- This line is optional, for checking the calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_triangle_area_l798_79888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l798_79831

-- Define the function f(x) = e^(ax) * sin(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (a * x) * Real.sin x

-- Define the theorem
theorem function_properties :
  -- Condition: a ≥ 1
  ∀ (a : ℝ) (b : ℝ), a ≥ 1 →
  -- Part 1: Monotonically increasing intervals when a = 1
  (∀ (x : ℝ), x ∈ Set.Icc 0 (3 * π / 4) ∪ Set.Ioc (7 * π / 4) (2 * π) →
    (deriv (f 1)) x > 0) ∧
  -- Part 2: Minimum value of b - e^2a
  (∀ (x : ℝ), x ∈ Set.Icc 0 (π / 2) → f a x ≤ b * x) →
  b - Real.exp (2 * a) ≥ -2 * Real.exp 2 / π := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l798_79831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_function_satisfies_conditions_l798_79841

-- Define the set X as the set of all non-zero reals
def X : Set ℝ := {x : ℝ | x ≠ 0}

-- State the theorem
theorem no_function_satisfies_conditions :
  ¬ ∃ (f : ℝ → ℝ),
    (∀ x, x ∈ X → f (-x) = -f x) ∧
    (∀ x y, x ∈ X → y ∈ X → x + y ∈ X → 
      f (1 / (x + y)) = f (1 / x) + f (1 / y) + 2 * (x * y - 1000)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_function_satisfies_conditions_l798_79841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l798_79829

theorem relationship_abc (a b c : ℝ) 
  (ha : a = Real.log (1/3))
  (hb : b = 2^(3/10))  -- Changed 0.3 to 3/10
  (hc : c = (1/3)^2) : 
  a < c ∧ c < b := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l798_79829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_factorization_l798_79808

theorem polynomial_factorization (x : Polynomial ℤ) :
  3 * (x + 3) * (x + 7) * (x + 11) * (x + 13) - 5 * x^2 = 
  x * (3 * x^3 + 117 * x^2 + 1430 * x + 14895) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_factorization_l798_79808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_sum_of_roots_l798_79852

noncomputable def cbrt (x : ℝ) : ℝ := Real.rpow x (1/3)

theorem cube_sum_of_roots (u v w : ℝ) : 
  (u - cbrt 17) * (u - cbrt 67) * (u - cbrt 97) = 1/2 →
  (v - cbrt 17) * (v - cbrt 67) * (v - cbrt 97) = 1/2 →
  (w - cbrt 17) * (w - cbrt 67) * (w - cbrt 97) = 1/2 →
  u ≠ v → u ≠ w → v ≠ w →
  u^3 + v^3 + w^3 = 181.5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_sum_of_roots_l798_79852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l798_79848

/-- A sequence a : ℕ → ℝ is an arithmetic sequence if the difference between
    any two consecutive terms is constant -/
def IsArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- Given an arithmetic sequence a, prove that if a₅ = 1, then a₄ + a₅ + a₆ = 3 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h : IsArithmeticSequence a) (h_a5 : a 5 = 1) :
  a 4 + a 5 + a 6 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l798_79848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_pq_inclination_l798_79859

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define the angle of inclination function
noncomputable def angle_of_inclination (p1 p2 : Point) : ℝ :=
  Real.arctan ((p2.2 - p1.2) / (p2.1 - p1.1))

theorem line_pq_inclination :
  let p : Point := (-2, 3)
  let q : Point := (-6, -1)
  angle_of_inclination p q = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_pq_inclination_l798_79859
